"""
GridSense AI — ENTSO-E Ingestion

Pulls historical load data from the ENTSO-E Transparency Platform and writes
clean Parquet files to data/raw/. Requires ENTSOE_API_KEY set in config.py.

Usage:
    python -m ingestion.entsoe_client
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import pandas as pd
from config import ENTSOE_API_KEY, ENTSOE_ZONES, DATA_RAW_DIR

try:
    from entsoe import EntsoePandasClient
except ImportError:
    raise ImportError(
        "Missing dependency. Run: pip install entsoe-py --break-system-packages"
    )


def get_client() -> "EntsoePandasClient":
    if not ENTSOE_API_KEY:
        raise ValueError(
            "ENTSO-E API key not set. Set the ENTSOE_API_KEY environment variable "
            "(or paste it into config.py for local-only use — never commit it)."
        )
    return EntsoePandasClient(api_key=ENTSOE_API_KEY)


def pull_load(zone_code: str, country_code: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    """Pull actual load for one zone and return a clean DataFrame."""
    client = get_client()
    print(f"[entsoe] pulling load for {country_code} ({zone_code}): {start.date()} -> {end.date()}")

    # BUG FIX (found during code review, never live-tested against the real API):
    # this previously called query_load(country_code, ...) — using the internal
    # config label (e.g. "IT_SOUTH", not a real ISO code) instead of the actual
    # EIC bidding-zone code stored in zone_code. That would have silently
    # discarded zone_code entirely and failed outright for IT_SOUTH, since
    # entsoe-py has no built-in alias for a made-up label like "IT_SOUTH" — it
    # needs the real EIC code (e.g. "10Y1001A1001A788"). Spain/Greece might have
    # accidentally worked since entsoe-py has built-in aliases for standard ISO
    # country codes, which masked this bug for those two zones specifically.
    series = client.query_load(zone_code, start=start, end=end)

    df = series.reset_index()
    df.columns = ["timestamp", "load_mw"]
    df["zone"] = country_code
    df["source"] = "entsoe"

    # Basic validation — do not silently trust the API
    n_before = len(df)
    df = df.dropna(subset=["load_mw"])
    df = df.drop_duplicates(subset=["timestamp"])
    n_after = len(df)
    if n_after < n_before:
        print(f"[entsoe]   WARNING: dropped {n_before - n_after} rows (nulls/dupes) for {country_code}")

    return df


def pull_all_zones(start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    """Pull load for every zone in config.ENTSOE_ZONES, concat, and save."""
    frames = []
    for country_code, zone_code in ENTSOE_ZONES.items():
        try:
            df = pull_load(zone_code, country_code, start, end)
            frames.append(df)
        except Exception as e:
            print(f"[entsoe]   FAILED for {country_code}: {e}")
            continue

    if not frames:
        raise RuntimeError("No zones pulled successfully — check API key and zone codes.")

    combined = pd.concat(frames, ignore_index=True)
    out_path = DATA_RAW_DIR / "entsoe_load.parquet"
    combined.to_parquet(out_path, index=False)
    print(f"[entsoe] wrote {len(combined)} rows -> {out_path}")
    return combined


if __name__ == "__main__":
    # Default: last 2 years, adjust as needed once you confirm access works
    end = pd.Timestamp.now(tz="Europe/Brussels").floor("h")
    start = end - pd.DateOffset(years=2)
    pull_all_zones(start, end)
