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
    for large date ranges that aren't precisely documented — this couldn't be
    live-verified from this environment (no network access to power.larc.nasa.gov
    in this sandbox), so chunking is a defensive fix that removes the risk
    entirely regardless of where the real limit actually is, rather than betting
    a 2-year single request against an unconfirmed ceiling.
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

    df = pd.concat(chunks)
    df = df.reset_index().rename(columns={
        "index": "timestamp",
        "T2M": "temperature_c",
        "RH2M": "humidity_pct",
        "WS10M": "wind_speed_ms",
        "PRECTOTCORR": "precip_mm",
    })
    df["zone"] = zone_name

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
