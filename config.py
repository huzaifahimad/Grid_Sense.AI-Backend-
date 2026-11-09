"""
GridSense AI — Central Configuration

This is the ONLY file you need to touch to plug in your ENTSO-E API key.
Everything else in the pipeline reads from here.
"""

import os
import re
from pathlib import Path

# ============================================================
#  >>> ENTSO-E API TOKEN (set via environment, never here) <<<
# ============================================================
# Get this from: transparency.entsoe.eu -> My Account -> Web API Security Token
# (only appears after ENTSO-E approves your "Restful API access" request).
# Set it as ENTSOE_API_KEY in your shell / .env / Render dashboard / Vercel env.
# This file is committed to git, so the token lives in the environment only;
# entsoe_client.get_client() raises a clear error when it is missing.
ENTSOE_API_KEY = os.environ.get("ENTSOE_API_KEY") or None
# ============================================================


# ------------------------------------------------------------
# Log/response hygiene
# ------------------------------------------------------------
# requests/urllib3 exception messages quote the request URL, and entsoe-py carries
# the token inside that URL's query string. The request-time refresher publishes
# exception text on /health, so the rule has to exist once, in the one module both
# the ingestion client and the API already import.
_CREDENTIAL_IN_URL = re.compile(
    r"(?i)(security[_-]?token|token|api[_-]?key)=([^&\s'\"]+)")


def redact_secrets(text: str) -> str:
    """Strip credential-shaped query parameters from arbitrary message text."""
    return _CREDENTIAL_IN_URL.sub(r"\1=[REDACTED]", str(text))


def error_text(value, limit: int) -> str:
    """
    "ExceptionType: message" for a log line or /health, credential stripped.

    Redaction comes first, never last: slicing can drop the "&" that terminates the
    token parameter, and then the pattern no longer matches and part of the
    credential is published anyway.

    Over-cap text loses its middle rather than its tail because a real requests/
    urllib3 message carries the request URL mid-string and the actual cause at the
    end — "... (read timeout=12)", "(caused by NewConnectionError('Connection
    refused'))". Head-truncating it keeps the URL and throws away the one part that
    says what to fix. This is the shared rule: ingestion and the API both format
    failure text here so neither can publish a quietly mangled reason.
    """
    message = redact_secrets(str(value))
    if len(message) > limit:
        head = (limit - 3) // 2
        message = message[:head] + "..." + message[-(limit - head - 3):]
    return f"{type(value).__name__}: {message}"


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


def _dir_from_env(var: str, default: Path) -> Path:
    """
    Redirectable data directory.

    A retrain must be able to write its candidates somewhere other than the
    committed artifacts the live API serves, so every writer below resolves its
    output through this hook instead of hardcoding BASE_DIR. Unset in the
    environment means exactly the historical layout, so nothing changes for the
    deployed service or the test suite.
    """
    value = os.environ.get(var)
    return Path(value).expanduser().resolve() if value else default


DATA_RAW_DIR = _dir_from_env("GRIDSENSE_DATA_RAW_DIR", BASE_DIR / "data" / "raw")
# Written by features/build_features.py; overridden by scripts/run_retrain_chain.sh
# --out-dir so a candidate feature table never lands on the served one.
DATA_FEATURES_DIR = _dir_from_env("GRIDSENSE_FEATURES_DIR", BASE_DIR / "data" / "features")
# The served artifact set (api/main.py loads it at startup). Training stages read
# this as INPUT and write GRIDSENSE_MODELS_OUT_DIR instead; only
# scripts/promote_models.py writes here, deliberately and from a gated candidate.
MODELS_DIR = _dir_from_env("GRIDSENSE_MODELS_DIR", BASE_DIR / "data" / "models")
# Where a retrain writes its models. Defaults to MODELS_DIR so the standalone
# trainers behave as they always did; the chain sets it to the dated candidate dir.
MODELS_OUT_DIR = _dir_from_env("GRIDSENSE_MODELS_OUT_DIR", MODELS_DIR)

# Only the data dirs the API and tests read are created eagerly; MODELS_OUT_DIR is
# created by whoever writes to it (scripts/run_retrain_chain.sh does it up front), so
# importing config never litters the repo with an empty candidate directory.
for d in (DATA_RAW_DIR, DATA_FEATURES_DIR, MODELS_DIR):
    d.mkdir(parents=True, exist_ok=True)

# ------------------------------------------------------------
# Model config
LOOKBACK_HOURS = 168        # 7 days
FORECAST_HORIZONS = [1, 6, 24, 48]   # hours ahead
OVERLOAD_THRESHOLD_RATIO = 0.9       # proxy label: load > 0.9 * capacity = overload risk
