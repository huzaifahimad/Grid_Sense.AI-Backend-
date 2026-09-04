"""
GridSense AI — Central Configuration

This is the ONLY file you need to touch to plug in your ENTSO-E API key.
Everything else in the pipeline reads from here.
"""

import os
from pathlib import Path

# ============================================================
#  >>> ENTSO-E API TOKEN (set via environment, never here) <<<
# ============================================================
# Get this from: transparency.entsoe.eu -> My Account -> Web API Security Token
# (only appears after ENTSO-E approves your "Restful API access" request).
# Set it as ENTSOE_API_KEY in your shell / .env / Render dashboard / Vercel env.
# Do not paste the token into this file — it is committed to git.
# If ENTSOE_API_KEY is not set in the environment, leave this as None.
# entsoe_client.get_client() will raise a clear error telling you to set it —
# never paste a real token here, since this file is committed to git.
ENTSOE_API_KEY = os.environ.get("ENTSOE_API_KEY") or None
# ============================================================

# NASA POWER needs no key — public API
NASA_POWER_BASE_URL = "https://power.larc.nasa.gov/api/temporal/hourly/point"

# Kaggle credentials (only needed if using kaggle_ingest.py)
# Get from kaggle.com -> Account -> Create New API Token (downloads kaggle.json)
# Place kaggle.json at ~/.kaggle/kaggle.json — the kaggle library reads it automatically,
# no need to paste it here.

# ------------------------------------------------------------
# Zones to pull for prototype training (hot-summer-peak climates,
# closer analogue to Pakistan's AC-driven demand than northern Europe)
ENTSOE_ZONES = {
    "ES": "10YES-REE------0",   # Spain
    "GR": "10YGR-HTSO-----Y",   # Greece
    "IT_SOUTH": "10Y1001A1001A788",  # Italy South
}

# Matching lat/lon per zone for weather join (approx. population-weighted center)
ZONE_COORDINATES = {
    "ES": {"lat": 40.4168, "lon": -3.7038},   # Madrid
    "GR": {"lat": 37.9838, "lon": 23.7275},   # Athens
    "IT_SOUTH": {"lat": 40.8518, "lon": 14.2681},  # Naples
}

# ------------------------------------------------------------
# Paths — matches the System Architecture doc's OSS layout conceptually,
# using local disk for the prototype (swap for OSS SDK calls later)
BASE_DIR = Path(__file__).parent
DATA_RAW_DIR = BASE_DIR / "data" / "raw"
DATA_FEATURES_DIR = BASE_DIR / "data" / "features"
MODELS_DIR = BASE_DIR / "data" / "models"

for d in (DATA_RAW_DIR, DATA_FEATURES_DIR, MODELS_DIR):
    d.mkdir(parents=True, exist_ok=True)

# ------------------------------------------------------------
# Model config
LOOKBACK_HOURS = 168        # 7 days
FORECAST_HORIZONS = [1, 6, 24, 48]   # hours ahead
OVERLOAD_THRESHOLD_RATIO = 0.9       # proxy label: load > 0.9 * capacity = overload risk
