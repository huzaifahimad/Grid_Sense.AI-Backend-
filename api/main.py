"""
GridSense AI — API Serving Layer

*** VERCEL-COMPATIBLE VERSION ***
Deliberately does NOT import shap or torch. Those pull in numba/llvmlite
(~95MB combined). Feature-contribution explanations are read from a
precomputed JSON lookup (data/models/risk_factors_by_bucket.json, generated
once by models/risk_classifier.py at training time) instead of computed live
via SHAP per request. See that script's docstring for the full trade-off.

Endpoints (System Architecture doc Section 1):
  GET /health                -> readiness: model/artifact load state + data window
  GET /forecast/{zone}       -> recursive 1..48h-ahead load forecast, calibrated P10/P50/P90
  GET /risk/{zone}           -> overload risk score + top contributing factors
  GET /shed-schedule/{zone}  -> recommended shed schedule (rules layer, NOT a model)
  GET /assets                -> ranked list of all monitored assets by risk
  GET /validation            -> closed-loop replay of the held-out window, with
                                PASS/WARN/FAIL release gates

Local run:
    uvicorn api.main:app --reload --port 8000
"""

import sys
from pathlib import Path
_ROOT = Path(__file__).parent.parent
sys.path.append(str(_ROOT))

import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("gridsense.api")

# Preload vendored OpenMP runtime on Linux before lightgbm imports.
# lightgbm's compiled lib_lightgbm.so links against libgomp.so.1; some PaaS
# base images (FastAPI Cloud's included) don't ship it.
if sys.platform.startswith("linux"):
    _vendored_libgomp = _ROOT / "vendor" / "libgomp.so.1"
    if _vendored_libgomp.exists():
        try:
            import ctypes
            ctypes.CDLL(str(_vendored_libgomp), mode=ctypes.RTLD_GLOBAL)
            log.info("preloaded vendored libgomp from %s", _vendored_libgomp)
        except OSError as _e:
            log.warning("could not preload vendored libgomp: %s", _e)

import json
import os
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Optional

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from config import MODELS_DIR, DATA_FEATURES_DIR, ENTSOE_ZONES
from models.decision_rules import (
    MAX_HORIZON_HOURS,
    bucket_for_score,
    dashboard_status,
    shed_priority,
)
from models.forecasting import MIN_HISTORY_HOURS, apply_intervals, recursive_forecast
from models.risk_classifier import FEATURE_COLS as RISK_FEATURE_COLS
from models.train_baselines import FEATURE_COLS as FORECAST_FEATURE_COLS

# Recursive forecast needs 168h of history; the extra rows are headroom.
ROWS_PER_ZONE_BUFFER = 240

_state: dict = {
    "forecast_model": None,
    "risk_model": None,
    "risk_factors": None,
    "intervals": None,
    "features": None,
    "closed_loop": None,
}


def _load_artifacts() -> None:
    lgb_path = MODELS_DIR / "lightgbm_load_forecast.joblib"
    risk_path = MODELS_DIR / "risk_classifier.joblib"
    factors_path = MODELS_DIR / "risk_factors_by_bucket.json"
    intervals_path = MODELS_DIR / "forecast_intervals.json"
    closed_loop_path = MODELS_DIR / "closed_loop_report.json"
    feat_path = DATA_FEATURES_DIR / "training_features.parquet"

    if lgb_path.exists():
        _state["forecast_model"] = joblib.load(lgb_path)
        log.info("loaded LightGBM forecast model")
    else:
        log.warning("no forecast model found — run models/train_baselines.py first")

    if risk_path.exists():
        _state["risk_model"] = joblib.load(risk_path)
        log.info("loaded risk classifier")
    else:
        log.warning("no risk classifier found — run models/risk_classifier.py first")

    if factors_path.exists():
        _state["risk_factors"] = json.loads(factors_path.read_text())
        log.info("loaded precomputed risk factors (no live SHAP)")
    else:
        log.warning("no precomputed risk factors found — run models/risk_classifier.py first")

    if intervals_path.exists():
        _state["intervals"] = json.loads(intervals_path.read_text())
        log.info("loaded calibrated forecast intervals (%s)", _state["intervals"].get("method"))
    else:
        log.warning("no forecast_intervals.json — falling back to fixed ±8%% band; "
                    "run models/calibrate_forecast_intervals.py")

    if closed_loop_path.exists():
        _state["closed_loop"] = json.loads(closed_loop_path.read_text())
        log.info("loaded closed-loop report — overall %s", _state["closed_loop"].get("overall"))
    else:
        log.warning("no closed_loop_report.json — /validation will 404; "
                    "run python -m models.closed_loop_validation")

    if feat_path.exists():
        full_df = pd.read_parquet(feat_path)
        # Endpoints only need the most recent window per zone; holding full
        # history was measured at ~412MB RSS against a 512MB platform limit.
        _state["features"] = (
            full_df.sort_values("timestamp")
            .groupby("zone", group_keys=False)
            .tail(ROWS_PER_ZONE_BUFFER)
            .reset_index(drop=True)
        )
        del full_df
        log.info("loaded %d feature rows (last %d/zone)", len(_state["features"]), ROWS_PER_ZONE_BUFFER)
    else:
        log.warning("no feature data found — run features/build_features.py first")


@asynccontextmanager
async def lifespan(_app: FastAPI):
    _load_artifacts()
    yield


app = FastAPI(title="GridSense AI API", version="1.1", lifespan=lifespan)

_default_cors_origin = "https://grid-senseai.vercel.app"
_cors_origins = [
    origin.strip()
    for origin in os.environ.get("CORS_ORIGINS", _default_cors_origin).split(",")
    if origin.strip()
] or [_default_cors_origin]
app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins,
    allow_methods=["GET"],
    allow_headers=["*"],
)


def _now_utc() -> datetime:
    return datetime.now(timezone.utc)


def _utc_now_iso() -> str:
    return _now_utc().isoformat()


def _zone_frame(zone: str) -> pd.DataFrame:
    df = _state["features"]
    zone_df = df[df["zone"] == zone].sort_values("timestamp")
    if zone_df.empty:
        raise HTTPException(404, f"No data for zone '{zone}'. Known zones: {list(ENTSOE_ZONES.keys())}")
    return zone_df


def _representative_snapshot(zone_df: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    """Latest complete row matching the current UTC weekday+hour (replay of a historical snapshot)."""
    complete_rows = zone_df.dropna(subset=feature_cols).sort_values("timestamp")
    if complete_rows.empty:
        return complete_rows

    now = datetime.now(timezone.utc)
    current_slot = complete_rows[
        (complete_rows["timestamp"].dt.dayofweek == now.weekday())
        & (complete_rows["timestamp"].dt.hour == now.hour)
    ]
    if not current_slot.empty:
        return current_slot.tail(1)

    current_hour = complete_rows[complete_rows["timestamp"].dt.hour == now.hour]
    if not current_hour.empty:
        return current_hour.tail(1)

    return complete_rows.tail(1)


# ------------------------------------------------------------
# Response schemas

class ForecastPoint(BaseModel):
    timestamp: str
    step: int
    p10: float
    p50: float
    p90: float


class ForecastResponse(BaseModel):
    zone: str
    generated_at: str
    origin_timestamp: str
    horizon_hours: int
    interval_method: str
    weather_assumption: str
    points: list[ForecastPoint]


class RiskFactor(BaseModel):
    feature: str
    contribution: float


class RiskResponse(BaseModel):
    zone: str
    asset_id: str
    risk_score: float
    status: str
    snapshot_timestamp: str
    predicted_load_mw: float
    capacity_mw: float
    top_factors: list[RiskFactor]
    is_proxy_capacity: bool


class AssetSummary(BaseModel):
    asset_id: str
    zone: str
    risk_score: float
    status: str  # "critical" | "elevated" | "normal"


class ShedRecommendation(BaseModel):
    zone: str
    asset_id: str
    priority: int
    reasoning: str
    predicted_margin_mw: float


class HealthResponse(BaseModel):
    status: str  # "ok" | "degraded"
    generated_at: str
    artifacts: dict[str, bool]
    zones: list[str]
    serving_window_start: Optional[str]
    serving_window_end: Optional[str]
    interval_method: str


class ValidationGate(BaseModel):
    id: str
    family: str
    label: str
    direction: str
    target: float | list[float]
    floor: float | list[float]
    value: Optional[float]
    result: str  # "PASS" | "WARN" | "FAIL"


class ValidationResponse(BaseModel):
    method: str
    generated_at: str
    runtime_seconds: float
    # Deep metric blocks stay untyped so the validator can grow without
    # breaking the API contract; the gates below are the release decision.
    window: dict
    artifacts: dict
    metrics: dict
    gates: list[ValidationGate]
    overall: str
    limitations: list[str]
    staleness_hours: float


# ------------------------------------------------------------
# Endpoints

@app.get("/")
def root():
    return {"service": "GridSense AI API", "status": "ok", "zones": list(ENTSOE_ZONES.keys())}


@app.get("/health", response_model=HealthResponse)
def health():
    artifacts = {
        "forecast_model": _state["forecast_model"] is not None,
        "risk_model": _state["risk_model"] is not None,
        "risk_factors": _state["risk_factors"] is not None,
        "forecast_intervals": _state["intervals"] is not None,
        "features": _state["features"] is not None,
        # Evidence, not a serving dependency: its absence must not mark the API degraded.
        "closed_loop_report": _state["closed_loop"] is not None,
    }
    df = _state["features"]
    required = ("forecast_model", "risk_model", "features")
    return HealthResponse(
        status="ok" if all(artifacts[k] for k in required) else "degraded",
        generated_at=_utc_now_iso(),
        artifacts=artifacts,
        zones=sorted(df["zone"].unique().tolist()) if df is not None else [],
        serving_window_start=df["timestamp"].min().isoformat() if df is not None else None,
        serving_window_end=df["timestamp"].max().isoformat() if df is not None else None,
        interval_method=(_state["intervals"] or {}).get("method", "fixed ±8% fallback"),
    )


@app.get("/validation", response_model=ValidationResponse)
def get_validation():
    """
    Closed-loop validation: the held-out test window replayed hour by hour through
    this same serving path (forecast + calibrated band + risk score + shed rule)
    and scored against what actually happened, with PASS/WARN/FAIL release gates.

    Served from data/models/closed_loop_report.json as written by
    `python -m models.closed_loop_validation` — nothing is recomputed per request,
    and staleness_hours tells the caller how old that evidence is.
    """
    report = _state["closed_loop"]
    if report is None:
        raise HTTPException(
            503, "No closed-loop report. Run: python -m models.closed_loop_validation")

    generated = datetime.fromisoformat(report["generated_at"])
    if generated.tzinfo is None:
        generated = generated.replace(tzinfo=timezone.utc)
    staleness_hours = round((_now_utc() - generated).total_seconds() / 3600.0, 2)
    return ValidationResponse(**report, staleness_hours=staleness_hours)


@app.get("/forecast/{zone}", response_model=ForecastResponse)
def get_forecast(zone: str, horizon_hours: int = Query(default=24, ge=1, le=MAX_HORIZON_HOURS)):
    if _state["features"] is None or _state["forecast_model"] is None:
        raise HTTPException(503, "Models not loaded. Train models first — see README.")

    history = _zone_frame(zone)[["timestamp", "load_mw", "temperature_c", "humidity_pct", "wind_speed_ms"]].dropna()
    if len(history) < MIN_HISTORY_HOURS:
        raise HTTPException(503, f"Need {MIN_HISTORY_HOURS}h of history for zone '{zone}', have {len(history)}")

    forecast = recursive_forecast(_state["forecast_model"], history, horizon_hours, FORECAST_FEATURE_COLS)
    forecast = apply_intervals(forecast, _state["intervals"])
    intervals = _state["intervals"] or {}

    points = [
        ForecastPoint(
            timestamp=row.timestamp.isoformat(),
            step=int(row.step),
            p10=float(row.p10),
            p50=float(row.p50),
            p90=float(row.p90),
        )
        for row in forecast.itertuples()
    ]
    return ForecastResponse(
        zone=zone,
        generated_at=_utc_now_iso(),
        origin_timestamp=history["timestamp"].iloc[-1].isoformat(),
        horizon_hours=horizon_hours,
        interval_method=intervals.get("method", "fixed ±8% fallback"),
        weather_assumption=intervals.get("weather_assumption", "same hour on the most recent observed day"),
        points=points,
    )


@app.get("/risk/{zone}", response_model=RiskResponse)
def get_risk(zone: str):
    if _state["features"] is None or _state["risk_model"] is None:
        raise HTTPException(503, "Risk model not loaded. Train models first — see README.")

    latest_row = _representative_snapshot(_zone_frame(zone), RISK_FEATURE_COLS)
    if latest_row.empty:
        raise HTTPException(404, f"No complete feature row available for zone '{zone}'")

    risk_score = float(_state["risk_model"].predict_proba(latest_row[RISK_FEATURE_COLS])[0, 1])

    # Bucketed precomputed factors, not per-request SHAP — see risk_classifier.py.
    factors = (_state["risk_factors"] or {}).get(bucket_for_score(risk_score), [])

    return RiskResponse(
        zone=zone,
        asset_id=f"{zone}-FEEDER-01",
        risk_score=risk_score,
        status=dashboard_status(risk_score),
        snapshot_timestamp=latest_row["timestamp"].iloc[0].isoformat(),
        predicted_load_mw=float(latest_row["load_mw"].iloc[0]),
        capacity_mw=float(latest_row["capacity_mw"].iloc[0]),
        top_factors=[RiskFactor(**f) for f in factors],
        is_proxy_capacity=bool(latest_row["capacity_is_proxy"].iloc[0]),
    )


@app.get("/assets", response_model=list[AssetSummary])
def list_assets():
    """Ranked list for the dashboard's asset panel and the 3D topology view."""
    if _state["features"] is None or _state["risk_model"] is None:
        raise HTTPException(503, "Models not loaded. Train models first — see README.")

    df = _state["features"]
    summaries = []
    for zone in df["zone"].unique():
        latest = _representative_snapshot(df[df["zone"] == zone], RISK_FEATURE_COLS)
        if latest.empty:
            continue
        score = float(_state["risk_model"].predict_proba(latest[RISK_FEATURE_COLS])[0, 1])
        summaries.append(
            AssetSummary(
                asset_id=f"{zone}-FEEDER-01",
                zone=zone,
                risk_score=score,
                status=dashboard_status(score),
            )
        )

    return sorted(summaries, key=lambda a: -a.risk_score)


@app.get("/shed-schedule/{zone}", response_model=list[ShedRecommendation])
def get_shed_schedule(zone: str):
    """
    Rules layer, NOT a model — System Architecture doc Section 2.
    Only elevated/critical assets get a recommendation (priority 1 = critical,
    2 = elevated), using the same thresholds as the dashboard status so the
    two never disagree. Critical infrastructure is never recommended.
    Advisory only: an operator must approve any shed action.
    """
    risk = get_risk(zone)

    CRITICAL_ASSETS = {"HOSPITAL", "WATER_PUMP"}
    if any(tag in risk.asset_id for tag in CRITICAL_ASSETS):
        return []
    priority = shed_priority(risk.risk_score)
    if priority is None:
        return []

    margin = risk.capacity_mw - risk.predicted_load_mw
    reasoning = (
        f"Predicted load {risk.predicted_load_mw:.0f}MW against "
        f"{'estimated (proxy)' if risk.is_proxy_capacity else 'rated'} capacity "
        f"{risk.capacity_mw:.0f}MW — margin {margin:.0f}MW, risk score {risk.risk_score:.2f}"
    )

    return [
        ShedRecommendation(
            zone=zone,
            asset_id=risk.asset_id,
            priority=priority,
            reasoning=reasoning,
            predicted_margin_mw=margin,
        )
    ]
