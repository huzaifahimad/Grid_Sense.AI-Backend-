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
from api import live_refresh
from models.decision_rules import (
    MAX_HORIZON_HOURS,
    bucket_for_score,
    dashboard_status,
    shed_priority,
)
from models.forecasting import (
    MIN_HISTORY_HOURS, WEATHER_COLS, apply_intervals, recursive_forecast,
)
from models.risk_classifier import FEATURE_COLS as RISK_FEATURE_COLS
from models.train_baselines import (
    FEATURE_COLS as FORECAST_FEATURE_COLS,
    FEATURE_COLS_NO_WEATHER as FALLBACK_FEATURE_COLS,
)

# Recursive forecast needs 168h of history; the extra rows are headroom. Kept at 14
# days because the newest hours per zone can be load-only (weather not yet published),
# and the weather-aware path still needs a full 168h run behind that tail.
ROWS_PER_ZONE_BUFFER = 336

# Only answer from the load-only tail when it is meaningfully newer; below this the
# weather-aware model is served instead of trading accuracy for a few spare hours.
FALLBACK_MIN_GAIN_HOURS = 6

_state: dict = {
    "forecast_model": None,
    "fallback_model": None,
    "risk_model": None,
    "risk_factors": None,
    "intervals": None,
    "fallback_intervals": None,
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

    fallback_path = MODELS_DIR / "lightgbm_load_forecast_no_weather.joblib"
    if fallback_path.exists():
        _state["fallback_model"] = joblib.load(fallback_path)
        log.info("loaded no-weather fallback forecaster")
    else:
        log.info("no fallback forecaster — /forecast will wait for published weather "
                 "instead of answering from the newest load hour")

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

    fallback_intervals_path = MODELS_DIR / "forecast_intervals_no_weather.json"
    if fallback_intervals_path.exists():
        _state["fallback_intervals"] = json.loads(fallback_intervals_path.read_text())
        log.info("loaded fallback forecast intervals (%s)",
                 _state["fallback_intervals"].get("method"))
    elif _state["fallback_model"] is not None:
        # A reduced model scored under the weather-aware model's quantiles would publish
        # coverage it never earned, so the model is dropped along with its missing bands.
        _state["fallback_model"] = None
        log.warning("fallback model present but uncalibrated — not serving it")

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


def _publish_frame(frame: pd.DataFrame) -> None:
    """Swap the served frame in. A single dict write, so readers see old or new."""
    _state["features"] = frame


def _refresh_served_frame() -> None:
    """
    Non-blocking: starts a background refresh if one is due, returns immediately.

    Called from the data endpoints rather than a middleware so docs pages and
    health probes cannot spend the egress budget, and so a request never waits on
    the network — the refreshed frame lands on the caller's next poll.
    """
    live_refresh.kick_if_due(_state["features"], _publish_frame)


@asynccontextmanager
async def lifespan(_app: FastAPI):
    _load_artifacts()
    # Kick once at boot so the frame is already chasing the feed before the first
    # dashboard request; the endpoint hooks below then keep it there.
    _refresh_served_frame()
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

# Set on every response. Verified absent from the deployed API before this existed
# (a live probe returned only Server/Date/Content-Type/Content-Length/vary/
# cf-cache-status). `no-store` matters specifically because the dashboard reports how
# old the served frame is: a copy held by any intermediary would make that number a
# lie. No CSP `default-src` because /docs and /redoc are served by this same app and
# need inline scripts; `frame-ancestors` covers the clickjacking case that matters for
# JSON. No HSTS `includeSubDomains`/`preload` because this host sits on a platform
# domain this project does not control — claiming those would be overreach.
_RESPONSE_HEADERS = {
    "X-Content-Type-Options": "nosniff",
    "Referrer-Policy": "no-referrer",
    "X-Frame-Options": "DENY",
    "Content-Security-Policy": "frame-ancestors 'none'",
    "Strict-Transport-Security": "max-age=31536000",
    "Cache-Control": "no-store",
}


@app.middleware("http")
async def _response_hygiene(request, call_next):
    response = await call_next(request)
    for key, value in _RESPONSE_HEADERS.items():
        response.headers[key] = value
    return response


def _now_utc() -> datetime:
    return datetime.now(timezone.utc)


def _utc_now_iso() -> str:
    return _now_utc().isoformat()


# Zone codes are a closed set, so an unmatched segment is always caller noise rather
# than a near-miss worth echoing back in full. Cap the echo: an unbounded one writes
# arbitrary attacker-chosen text into every log line and response body downstream.
MAX_ZONE_ECHO_CHARS = 32


def _zone_frame(zone: str) -> pd.DataFrame:
    if zone not in ENTSOE_ZONES:
        echo = zone if len(zone) <= MAX_ZONE_ECHO_CHARS else zone[:MAX_ZONE_ECHO_CHARS] + "…"
        raise HTTPException(
            404, f"No data for zone '{echo}'. Known zones: {list(ENTSOE_ZONES.keys())}"
        )
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


def _contiguous_hourly_tail(df: pd.DataFrame) -> pd.DataFrame:
    """
    Longest suffix of rows spaced exactly one hour apart.

    recursive_forecast indexes history positionally (lag 1h/24h/168h = -1/-24/-168
    rows) but labels its output from the final timestamp alone. A missing hour
    therefore shifts every lag feature out of true while the returned timestamps
    still look plausible — a silently wrong forecast rather than a loud failure.
    """
    n = len(df)
    if n < 2:
        return df
    ordered = df.sort_values("timestamp").reset_index(drop=True)
    gap_h = ordered["timestamp"].diff().dt.total_seconds() / 3600.0
    start = n - 1
    while start > 0 and abs(float(gap_h.iloc[start]) - 1.0) < 1e-9:
        start -= 1
    return ordered.iloc[start:]


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
    # Which of the two forecasters answered. The fallback exists because NASA POWER
    # publishes weather roughly two days after the hour, while ENTSO-E publishes load
    # far sooner; forecasting from those load hours needs a model with no weather input.
    served_by: str = "weather_aware"
    # Measured, not asserted: origin_timestamp alone makes a caller do the subtraction
    # it is supposed to be warned by.
    origin_age_hours: float = 0.0
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
    # Observed load on the historical row the score was computed from. Deliberately
    # NOT called "predicted": /risk scores a real snapshot, while /forecast produces
    # the forward-looking numbers. Naming them alike made two dashboard panels
    # disagree about "the load" by ~800MW.
    snapshot_load_mw: float
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
    # Where the served numbers actually come from. The synthetic bootstrap and
    # the real ingestion write the same file names, so this is the only field
    # that lets a reader tell them apart without opening the parquet.
    data_provenance: dict
    # What the request-time refresher is doing about the age reported above:
    # whether egress is enabled, when it last ran, how many hours it added, and the
    # reason it last fell back to the committed frame.
    data_refresh: dict = {}


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

    provenance: dict = {}
    if df is not None:
        def _counts(col: str) -> dict:
            # A frame built before provenance columns existed must not read as
            # though it had been verified as real.
            if col not in df.columns:
                return {"unstamped": int(len(df))}
            return {str(k): int(v) for k, v in df[col].value_counts().items()}

        load_src, weather_src = _counts("load_source"), _counts("weather_source")
        if "capacity_is_proxy" in df.columns:
            # A frame with even one proxy row must not report itself as rated.
            capacity_basis = (
                "p99-of-load proxy" if bool(df["capacity_is_proxy"].any()) else "rated"
            )
        else:
            capacity_basis = "unknown"

        # Currency per feed: the newest hour each input genuinely covers. Load
        # from ENTSO-E lands ~24h late and NASA POWER reanalysis lags further,
        # so these two can differ by days — which is what decides whether the
        # weather-aware forecaster can serve at all.
        load_latest = df["timestamp"].max()
        weather_latest = None
        if "temperature_c" in df.columns:
            weather_rows = df.loc[df["temperature_c"].notna(), "timestamp"]
            weather_latest = weather_rows.max() if len(weather_rows) else None
        lag_h = (
            round((load_latest - weather_latest).total_seconds() / 3600, 1)
            if weather_latest is not None
            else None
        )

        provenance = {
            "load": load_src,
            "weather": weather_src,
            "capacity": capacity_basis,
            "load_latest_timestamp": load_latest.isoformat(),
            "weather_latest_timestamp": (
                weather_latest.isoformat() if weather_latest is not None else None
            ),
            "weather_lag_hours": lag_h,
            "is_synthetic": any(
                "synthetic" in s for group in (load_src, weather_src) for s in group
            ),
        }

    return HealthResponse(
        status="ok" if all(artifacts[k] for k in required) else "degraded",
        generated_at=_utc_now_iso(),
        artifacts=artifacts,
        zones=sorted(df["zone"].unique().tolist()) if df is not None else [],
        serving_window_start=df["timestamp"].min().isoformat() if df is not None else None,
        serving_window_end=df["timestamp"].max().isoformat() if df is not None else None,
        interval_method=(_state["intervals"] or {}).get("method", "fixed ±8% fallback"),
        data_provenance=provenance,
        data_refresh=live_refresh.status(df),
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
    _refresh_served_frame()
    if _state["features"] is None or _state["forecast_model"] is None:
        raise HTTPException(503, "Models not loaded. Train models first — see README.")

    zone_df = _zone_frame(zone)
    weather_history = _contiguous_hourly_tail(
        zone_df[["timestamp", "load_mw", *WEATHER_COLS]].dropna()
    )
    load_history = _contiguous_hourly_tail(zone_df[["timestamp", "load_mw"]].dropna())

    # Serve the fallback only when it genuinely answers sooner, and only with its own
    # bands. Weather-blind load hours otherwise cost the operator a day and a half of
    # foresight, while the weather-aware model fed those NaNs would be guessing.
    gain_h = 0.0
    if len(load_history) >= MIN_HISTORY_HOURS and len(weather_history) >= 2:
        gain_h = float((load_history["timestamp"].iloc[-1]
                        - weather_history["timestamp"].iloc[-1]).total_seconds() // 3600)
    use_fallback = (
        _state["fallback_model"] is not None
        and _state["fallback_intervals"] is not None
        and gain_h >= FALLBACK_MIN_GAIN_HOURS
    )

    if use_fallback:
        history, model = load_history, _state["fallback_model"]
        feature_cols, intervals = FALLBACK_FEATURE_COLS, _state["fallback_intervals"]
        served_by = "no_weather_fallback"
    else:
        history, model = weather_history, _state["forecast_model"]
        feature_cols, intervals = FORECAST_FEATURE_COLS, _state["intervals"]
        served_by = "weather_aware"

    if len(history) < MIN_HISTORY_HOURS:
        raise HTTPException(
            503,
            f"Need {MIN_HISTORY_HOURS}h of gap-free history for zone '{zone}', "
            f"have {len(history)}",
        )

    forecast = recursive_forecast(model, history, horizon_hours, feature_cols)
    forecast = apply_intervals(forecast, intervals)

    origin = history["timestamp"].iloc[-1]
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
        origin_timestamp=origin.isoformat(),
        horizon_hours=horizon_hours,
        interval_method=intervals.get("method", "fixed ±8% fallback") if intervals else "fixed ±8% fallback",
        weather_assumption=(intervals or {}).get(
            "weather_assumption", "same hour on the most recent observed day"),
        served_by=served_by,
        origin_age_hours=round((_now_utc() - origin).total_seconds() / 3600, 2),
        points=points,
    )


@app.get("/risk/{zone}", response_model=RiskResponse)
def get_risk(zone: str):
    _refresh_served_frame()
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
        snapshot_load_mw=float(latest_row["load_mw"].iloc[0]),
        capacity_mw=float(latest_row["capacity_mw"].iloc[0]),
        top_factors=[RiskFactor(**f) for f in factors],
        is_proxy_capacity=bool(latest_row["capacity_is_proxy"].iloc[0]),
    )


@app.get("/assets", response_model=list[AssetSummary])
def list_assets():
    """Ranked list for the dashboard's asset panel and the 3D topology view."""
    _refresh_served_frame()
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
    two never disagree. Advisory only: an operator must approve any shed action.
    """
    risk = get_risk(zone)

    priority = shed_priority(risk.risk_score)
    if priority is None:
        return []

    margin = risk.capacity_mw - risk.snapshot_load_mw
    snapshot_hour = risk.snapshot_timestamp[:16].replace("T", " ")
    reasoning = (
        f"Load {risk.snapshot_load_mw:.0f}MW at {snapshot_hour}Z against "
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
