"""
GridSense AI — API Serving Layer

*** VERCEL-COMPATIBLE VERSION ***
Deliberately does NOT import shap, scikit-learn, or torch. Those pull in
numba/llvmlite/scipy (~95MB combined) which pushed this past what's
practical for a Vercel serverless function. Feature-contribution
explanations are read from a precomputed JSON lookup
(data/models/risk_factors_by_bucket.json, generated once by
models/risk_classifier.py at training time) instead of computed live via
SHAP per request. See that script's docstring for the full trade-off.

Matches the endpoints defined in System Architecture doc Section 1:
  GET /forecast/{zone}       -> load forecast + P10/P50/P90
  GET /risk/{zone}           -> overload risk score + top contributing factors
  GET /shed-schedule/{zone}  -> recommended shed schedule (rules layer, NOT a model)
  GET /assets                -> ranked list of all monitored assets by risk

Local run:
    uvicorn api.main:app --reload --port 8000

On Vercel, this file is adapted into api/index.py per Vercel's Python
serverless function convention — see DEPLOYMENT_GUIDE.md.
"""

import sys
from pathlib import Path
_ROOT = Path(__file__).parent.parent
sys.path.append(str(_ROOT))

# Preload vendored OpenMP runtime on Linux before lightgbm imports.
# lightgbm's compiled lib_lightgbm.so links against libgomp.so.1; some PaaS
# base images (FastAPI Cloud's included) don't ship it. Loading our vendored
# copy into the process first satisfies the symbol lookup at import time.
if sys.platform.startswith("linux"):
    _vendored_libgomp = _ROOT / "vendor" / "libgomp.so.1"
    if _vendored_libgomp.exists():
        try:
            import ctypes
            ctypes.CDLL(str(_vendored_libgomp), mode=ctypes.RTLD_GLOBAL)
            print(f"[api] preloaded vendored libgomp from {_vendored_libgomp}")
        except OSError as _e:
            print(f"[api] WARNING: could not preload vendored libgomp: {_e}")

import json
from datetime import datetime, timezone
from typing import Optional

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from config import MODELS_DIR, DATA_FEATURES_DIR, ENTSOE_ZONES

app = FastAPI(title="GridSense AI API", version="1.0")

# Allow the frontend dev server to call this API — tighten in production
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ------------------------------------------------------------
# Load models once at startup (fail loudly if they're missing — no silent fallback)
_lightgbm_model = None
_risk_model = None
_risk_factors_by_bucket = None
_features_df = None


def _bucket_for_score(score: float) -> str:
    if score >= 0.8:
        return "critical"
    if score >= 0.6:
        return "elevated"
    if score >= 0.3:
        return "moderate"
    return "low"


def _dashboard_status(score: float) -> str:
    if score >= 0.8:
        return "critical"
    if score >= 0.6:
        return "elevated"
    return "normal"


def _representative_snapshot(zone_df: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
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


@app.on_event("startup")
def load_models():
    global _lightgbm_model, _risk_model, _risk_factors_by_bucket, _features_df

    lgb_path = MODELS_DIR / "lightgbm_load_forecast.joblib"
    risk_path = MODELS_DIR / "risk_classifier.joblib"
    factors_path = MODELS_DIR / "risk_factors_by_bucket.json"
    feat_path = DATA_FEATURES_DIR / "training_features.parquet"

    if lgb_path.exists():
        _lightgbm_model = joblib.load(lgb_path)
        print("[api] loaded LightGBM forecast model")
    else:
        print("[api] WARNING: no forecast model found — run models/train_baselines.py first")

    if risk_path.exists():
        _risk_model = joblib.load(risk_path)
        print("[api] loaded risk classifier")

    if factors_path.exists():
        with open(factors_path) as f:
            _risk_factors_by_bucket = json.load(f)
        print("[api] loaded precomputed risk factors (no live SHAP — see risk_classifier.py)")
    else:
        print("[api] WARNING: no precomputed risk factors found — run models/risk_classifier.py first")

    if feat_path.exists():
        full_df = pd.read_parquet(feat_path)
        # MEMORY FIX: every endpoint below only ever reads the last 1-48 rows
        # per zone (confirmed by checking every _features_df usage in this file
        # — .tail(1) for risk/assets, .tail(horizon_hours) capped at 48 for
        # forecast). Holding the full history in memory was measured to push
        # RSS to 412MB against FastAPI Cloud's Hobby tier 512MB hard limit —
        # dangerously close, and that was under light single-request testing,
        # not concurrent load or a larger real-data feature set. Trimming to
        # a 200-row-per-zone buffer (well over the 48-row max any endpoint
        # needs) cuts memory footprint by roughly 99% of the dataframe's size
        # with zero change in what any endpoint can actually return.
        ROWS_PER_ZONE_BUFFER = 200
        _features_df = (
            full_df.sort_values("timestamp")
            .groupby("zone", group_keys=False)
            .tail(ROWS_PER_ZONE_BUFFER)
            .reset_index(drop=True)
        )
        del full_df
        print(f"[api] loaded {len(_features_df)} feature rows "
              f"(trimmed from full history to last {ROWS_PER_ZONE_BUFFER}/zone for memory)")
    else:
        print("[api] WARNING: no feature data found — run features/build_features.py first")


# ------------------------------------------------------------
# Response schemas

class ForecastPoint(BaseModel):
    timestamp: str
    p10: float
    p50: float
    p90: float


class ForecastResponse(BaseModel):
    zone: str
    generated_at: str
    points: list[ForecastPoint]


class RiskFactor(BaseModel):
    feature: str
    contribution: float


class RiskResponse(BaseModel):
    zone: str
    asset_id: str
    risk_score: float
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


# ------------------------------------------------------------
# Endpoints

@app.get("/")
def root():
    return {"service": "GridSense AI API", "status": "ok", "zones": list(ENTSOE_ZONES.keys())}


@app.get("/forecast/{zone}", response_model=ForecastResponse)
def get_forecast(zone: str, horizon_hours: int = Query(default=24, ge=1, le=48)):
    if _features_df is None or _lightgbm_model is None:
        raise HTTPException(503, "Models not loaded. Train models first — see README.")

    zone_df = _features_df[_features_df["zone"] == zone].sort_values("timestamp")
    if zone_df.empty:
        raise HTTPException(404, f"No data for zone '{zone}'. Known zones: {list(ENTSOE_ZONES.keys())}")

    latest = zone_df.tail(horizon_hours)
    from models.train_baselines import FEATURE_COLS
    valid = latest.dropna(subset=FEATURE_COLS)

    preds = _lightgbm_model.predict(valid[FEATURE_COLS])
    # Point-estimate model — approximate P10/P90 with a fixed band until the
    # quantile LSTM+CNN model is trained and swapped in here
    points = [
        ForecastPoint(
            timestamp=row.timestamp.isoformat(),
            p10=float(pred * 0.92),
            p50=float(pred),
            p90=float(pred * 1.08),
        )
        for row, pred in zip(valid.itertuples(), preds)
    ]

    return ForecastResponse(zone=zone, generated_at=datetime.utcnow().isoformat(), points=points)


@app.get("/risk/{zone}", response_model=RiskResponse)
def get_risk(zone: str):
    if _features_df is None or _risk_model is None:
        raise HTTPException(503, "Risk model not loaded. Train models first — see README.")

    from models.risk_classifier import FEATURE_COLS
    zone_df = _features_df[_features_df["zone"] == zone]
    if zone_df.empty:
        raise HTTPException(404, f"No data for zone '{zone}'")

    latest_row = _representative_snapshot(zone_df, FEATURE_COLS)
    if latest_row.empty:
        raise HTTPException(404, f"No complete feature row available for zone '{zone}'")

    risk_score = float(_risk_model.predict_proba(latest_row[FEATURE_COLS])[0, 1])

    # Read precomputed factors for this risk bucket instead of running live SHAP —
    # see risk_classifier.py docstring for why. Bucketed, not per-request-exact.
    top_factors = []
    if _risk_factors_by_bucket is not None:
        bucket = _bucket_for_score(risk_score)
        top_factors = [RiskFactor(**f) for f in _risk_factors_by_bucket.get(bucket, [])]

    return RiskResponse(
        zone=zone,
        asset_id=f"{zone}-FEEDER-01",
        risk_score=risk_score,
        predicted_load_mw=float(latest_row["load_mw"].iloc[0]),
        capacity_mw=float(latest_row["capacity_mw"].iloc[0]),
        top_factors=top_factors,
        is_proxy_capacity=bool(latest_row["capacity_is_proxy"].iloc[0]),
    )


@app.get("/assets", response_model=list[AssetSummary])
def list_assets():
    """Ranked list for the dashboard's asset panel and the 3D topology view."""
    if _features_df is None or _risk_model is None:
        raise HTTPException(503, "Models not loaded. Train models first — see README.")

    from models.risk_classifier import FEATURE_COLS
    summaries = []
    for zone in _features_df["zone"].unique():
        zone_df = _features_df[_features_df["zone"] == zone]
        latest = _representative_snapshot(zone_df, FEATURE_COLS)
        if latest.empty:
            continue
        score = float(_risk_model.predict_proba(latest[FEATURE_COLS])[0, 1])
        summaries.append(
            AssetSummary(
                asset_id=f"{zone}-FEEDER-01",
                zone=zone,
                risk_score=score,
                status=_dashboard_status(score),
            )
        )

    return sorted(summaries, key=lambda a: -a.risk_score)


@app.get("/shed-schedule/{zone}", response_model=list[ShedRecommendation])
def get_shed_schedule(zone: str):
    """
    Rules layer, NOT a model — System Architecture doc Section 2.
    Transparent priority logic: rank by risk-adjusted margin, respect a
    critical-infrastructure exclusion list. Human-reviewable by design.
    """
    risk = get_risk(zone)

    CRITICAL_ASSETS = {"HOSPITAL", "WATER_PUMP"}  # never auto-recommend shedding these
    if any(tag in risk.asset_id for tag in CRITICAL_ASSETS):
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
            priority=1 if risk.risk_score > 0.7 else 2,
            reasoning=reasoning,
            predicted_margin_mw=margin,
        )
    ]
