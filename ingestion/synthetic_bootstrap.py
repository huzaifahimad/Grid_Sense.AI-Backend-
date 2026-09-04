"""
GridSense AI — Synthetic Data Generator (PIPELINE VALIDATION ONLY)

*** THIS IS NOT REAL DATA. ***
This exists solely to prove the ingestion->features->models->API pipeline
works mechanically, end to end, without waiting on real API pulls. It
produces files in the EXACT schema that ingestion/entsoe_client.py and
ingestion/nasa_power_client.py produce, so features/build_features.py and
every model script downstream run completely unmodified against it.

Once your real ENTSO-E pull finishes, DELETE these files and re-run the
real ingestion scripts. Nothing in this file is used elsewhere except as
a stand-in for real load_mw/weather values.

Usage:
    python -m ingestion.synthetic_bootstrap
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
from config import DATA_RAW_DIR, ENTSOE_ZONES, ZONE_COORDINATES

np.random.seed(42)


def generate_load(zone: str, start: pd.Timestamp, end: pd.Timestamp, base_load: float) -> pd.DataFrame:
    timestamps = pd.date_range(start, end, freq="h", tz="UTC")
    n = len(timestamps)

    hour = timestamps.hour.values
    dow = timestamps.dayofweek.values
    doy = timestamps.dayofyear.values

    # diurnal pattern: morning + evening peaks, night trough
    diurnal = 0.15 * np.sin((hour - 6) / 24 * 2 * np.pi) + 0.10 * np.sin((hour - 18) / 12 * 2 * np.pi)
    # weekly pattern: lower on weekends
    weekly = np.where(dow >= 5, -0.08, 0.0)
    # seasonal: summer AC peak (approximates the AC-driven demand this project cares about)
    seasonal = 0.20 * np.sin((doy - 172) / 365 * 2 * np.pi)  # peak near day 172 (~summer solstice)
    # slow multi-year growth trend
    trend = np.linspace(0, 0.04, n)
    # autocorrelated noise (AR(1)) instead of pure white noise — real load isn't that clean
    noise = np.zeros(n)
    for i in range(1, n):
        noise[i] = 0.85 * noise[i - 1] + np.random.normal(0, 0.02)

    load = base_load * (1 + diurnal + weekly + seasonal + trend + noise)
    load = np.clip(load, base_load * 0.4, base_load * 1.6)

    df = pd.DataFrame({
        "timestamp": timestamps,
        "load_mw": load.round(1),
        "zone": zone,
        "source": "synthetic_bootstrap",
    })
    return df


def generate_weather(zone: str, start: pd.Timestamp, end: pd.Timestamp, base_temp: float) -> pd.DataFrame:
    timestamps = pd.date_range(start, end, freq="h", tz="UTC")
    n = len(timestamps)
    hour = timestamps.hour.values
    doy = timestamps.dayofyear.values

    diurnal_temp = 5 * np.sin((hour - 9) / 24 * 2 * np.pi)
    seasonal_temp = 12 * np.sin((doy - 172) / 365 * 2 * np.pi)
    temp_noise = np.random.normal(0, 1.5, n)
    temperature = base_temp + diurnal_temp + seasonal_temp + temp_noise

    humidity = np.clip(55 - 0.6 * (temperature - base_temp) + np.random.normal(0, 5, n), 15, 95)
    wind = np.clip(np.abs(np.random.normal(3.5, 2.0, n)), 0, 25)
    precip = np.clip(np.random.exponential(0.15, n) * (np.random.random(n) < 0.08), 0, None)

    df = pd.DataFrame({
        "timestamp": timestamps,
        "temperature_c": temperature.round(1),
        "humidity_pct": humidity.round(1),
        "wind_speed_ms": wind.round(1),
        "precip_mm": precip.round(2),
        "zone": zone,
    })
    return df


def run():
    end = pd.Timestamp.now(tz="UTC").floor("h")
    start = end - pd.DateOffset(years=2)

    base_loads = {"ES": 5200, "GR": 2900, "IT_SOUTH": 3800}
    base_temps = {"ES": 15, "GR": 18, "IT_SOUTH": 16}

    load_frames, weather_frames = [], []
    for zone in ENTSOE_ZONES.keys():
        print(f"[synthetic] generating load for {zone} ({start.date()} -> {end.date()})")
        load_frames.append(generate_load(zone, start, end, base_loads[zone]))
        print(f"[synthetic] generating weather for {zone}")
        weather_frames.append(generate_weather(zone, start, end, base_temps[zone]))

    load_df = pd.concat(load_frames, ignore_index=True)
    weather_df = pd.concat(weather_frames, ignore_index=True)

    load_path = DATA_RAW_DIR / "entsoe_load.parquet"
    weather_path = DATA_RAW_DIR / "nasa_power_weather.parquet"
    load_df.to_parquet(load_path, index=False)
    weather_df.to_parquet(weather_path, index=False)

    print(f"\n[synthetic] wrote {len(load_df)} load rows -> {load_path}")
    print(f"[synthetic] wrote {len(weather_df)} weather rows -> {weather_path}")
    print("[synthetic] *** THIS IS SYNTHETIC DATA — replace with real entsoe_client.py / "
          "nasa_power_client.py output before any real accuracy claim ***")

    return load_df, weather_df


if __name__ == "__main__":
    run()
