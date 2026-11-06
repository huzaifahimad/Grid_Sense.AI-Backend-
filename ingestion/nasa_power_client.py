"""
GridSense AI — NASA POWER Weather Ingestion

Pulls hourly temperature/humidity/wind data for each grid zone. No API key
needed — this can run today regardless of ENTSO-E approval status.

Usage:
    python -m ingestion.nasa_power_client
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import requests
import pandas as pd
from config import NASA_POWER_BASE_URL, ZONE_COORDINATES, DATA_RAW_DIR

# NASA POWER parameter codes:
# T2M = temperature at 2m (C), RH2M = relative humidity (%), WS10M = wind speed at 10m (m/s)
PARAMETERS = "T2M,RH2M,WS10M,PRECTOTCORR"


def pull_weather(zone_name: str, lat: float, lon: float, start: str, end: str) -> pd.DataFrame:
    """
    Pull hourly weather for one zone. start/end format: 'YYYYMMDD'

    Chunks the request by calendar year rather than one large multi-year call.
    NASA POWER's hourly endpoint has practical limits on request size/duration
    for large date ranges that aren't precisely documented, so chunking is a
    defensive fix that removes the risk regardless of where the real limit
    actually is, rather than betting a 2-year single request against an
    unconfirmed ceiling. The chunked path itself is verified against the live
    endpoint; the unchunked ceiling is still undocumented.
    """
    start_dt = pd.to_datetime(start, format="%Y%m%d")
    end_dt = pd.to_datetime(end, format="%Y%m%d")

    chunks = []
    chunk_start = start_dt
    while chunk_start <= end_dt:
        chunk_end = min(pd.Timestamp(year=chunk_start.year, month=12, day=31), end_dt)
        print(f"[nasa_power] pulling weather for {zone_name} ({lat}, {lon}): "
              f"{chunk_start.date()} -> {chunk_end.date()}")

        params = {
            "parameters": PARAMETERS,
            "community": "RE",
            "longitude": lon,
            "latitude": lat,
            "start": chunk_start.strftime("%Y%m%d"),
            "end": chunk_end.strftime("%Y%m%d"),
            "format": "JSON",
        }

        resp = requests.get(NASA_POWER_BASE_URL, params=params, timeout=90)
        resp.raise_for_status()
        data = resp.json()

        param_data = data["properties"]["parameter"]
        chunk_df = pd.DataFrame(param_data)
        chunk_df.index = pd.to_datetime(chunk_df.index, format="%Y%m%d%H")
        chunks.append(chunk_df)

        chunk_start = chunk_end + pd.Timedelta(days=1)

    df = pd.concat(chunks).sort_index()
    df = df[~df.index.duplicated(keep="first")]
    df = df.reset_index().rename(columns={
        "index": "timestamp",
        "T2M": "temperature_c",
        "RH2M": "humidity_pct",
        "WS10M": "wind_speed_ms",
        "PRECTOTCORR": "precip_mm",
    })

    # NASA POWER hourly point data is UTC. Verified empirically against the live
    # API (2026-11-03, Sep 2026): the mean diurnal temperature peak lands at
    # 14:00 for Madrid (-3.7 deg lon -> solar noon ~12:15 UTC), 12:00 for Naples
    # (~11:00 UTC) and 13:00 for Athens (~10:25 UTC) — i.e. 1-2 h after solar
    # noon in *UTC*. If these stamps were local civil time, all three would peak
    # at the same local hour instead, offset by each zone's UTC+2 DST.
    # The load frame is tz-aware UTC, and merging tz-aware against naive keys
    # raises rather than warn, so the weather frame has to carry the same stamp.
    df["timestamp"] = df["timestamp"].dt.tz_localize("UTC")
    df["zone"] = zone_name
    # Provenance travels with the rows, not the filename: ingestion/
    # synthetic_bootstrap.py writes this same file name with made-up values, so
    # a reader cannot tell real from stand-in by looking at the path. /health
    # surfaces this column so the deployed API states its own data source.
    df["source"] = "nasa_power"

    # NASA POWER uses -999 as a missing-value sentinel — must be handled explicitly,
    # never left as a valid-looking number
    n_missing = (df[["temperature_c", "humidity_pct", "wind_speed_ms", "precip_mm"]] < -900).sum().sum()
    if n_missing:
        print(f"[nasa_power]   WARNING: {n_missing} missing-value sentinels (-999) found for {zone_name}")
        df = df.replace(-999, pd.NA)

    return df


def pull_all_zones(start: str, end: str) -> pd.DataFrame:
    frames = []
    for zone_name, coords in ZONE_COORDINATES.items():
        df = pull_weather(zone_name, coords["lat"], coords["lon"], start, end)
        frames.append(df)

    combined = pd.concat(frames, ignore_index=True)
    out_path = DATA_RAW_DIR / "nasa_power_weather.parquet"
    combined.to_parquet(out_path, index=False)
    print(f"[nasa_power] wrote {len(combined)} rows -> {out_path}")
    return combined


if __name__ == "__main__":
    end = pd.Timestamp.now().strftime("%Y%m%d")
    start = (pd.Timestamp.now() - pd.DateOffset(years=2)).strftime("%Y%m%d")
    pull_all_zones(start, end)
