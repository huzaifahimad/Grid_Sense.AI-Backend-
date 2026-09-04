"""
GridSense AI — Feature Engineering

Joins load + weather data, builds lag/rolling/cyclic features per the
ML Model Design doc (Section 1.2), and writes a versioned Parquet feature
table ready for training.

Usage:
    python -m features.build_features
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
from config import DATA_RAW_DIR, DATA_FEATURES_DIR, OVERLOAD_THRESHOLD_RATIO


def load_raw_data(load_source: str = "entsoe") -> pd.DataFrame:
    """load_source: 'entsoe' or 'kaggle' — whichever ingestion has run."""
    load_path = DATA_RAW_DIR / f"{load_source}_load.parquet"
    weather_path = DATA_RAW_DIR / "nasa_power_weather.parquet"

    if not load_path.exists():
        raise FileNotFoundError(
            f"{load_path} not found. Run the {load_source} ingestion script first."
        )
    if not weather_path.exists():
        raise FileNotFoundError(
            f"{weather_path} not found. Run ingestion/nasa_power_client.py first "
            f"(no API key required, do this now if you haven't)."
        )

    load_df = pd.read_parquet(load_path)
    weather_df = pd.read_parquet(weather_path)
    return load_df, weather_df


def add_cyclic_time_features(df: pd.DataFrame, ts_col: str = "timestamp") -> pd.DataFrame:
    ts = df[ts_col]
    df["hour"] = ts.dt.hour
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)

    df["dow"] = ts.dt.dayofweek
    df["dow_sin"] = np.sin(2 * np.pi * df["dow"] / 7)
    df["dow_cos"] = np.cos(2 * np.pi * df["dow"] / 7)
    df["is_weekend"] = (df["dow"] >= 5).astype(int)

    df["month"] = ts.dt.month
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)

    return df


def add_lag_and_rolling_features(df: pd.DataFrame, group_col: str = "zone",
                                  target_col: str = "load_mw") -> pd.DataFrame:
    df = df.sort_values([group_col, "timestamp"])
    g = df.groupby(group_col)[target_col]

    df["load_lag_1h"] = g.shift(1)
    df["load_lag_24h"] = g.shift(24)
    df["load_lag_168h"] = g.shift(168)

    df["load_roll_mean_24h"] = g.transform(lambda s: s.shift(1).rolling(24).mean())
    df["load_roll_std_24h"] = g.transform(lambda s: s.shift(1).rolling(24).std())
    df["load_roll_mean_168h"] = g.transform(lambda s: s.shift(1).rolling(168).mean())

    return df


def add_overload_proxy_label(df: pd.DataFrame, capacity_col: str = None) -> pd.DataFrame:
    """
    Builds the documented proxy label for the risk classifier (TRD Section 6.2 /
    Data Sourcing Plan Section 3): overload = load > threshold * capacity.
    If no real capacity data is available, uses each zone's historical 99th
    percentile load as a stand-in capacity estimate — THIS IS A PROXY, flagged
    in every model card, never presented as ground truth.
    """
    if capacity_col and capacity_col in df.columns:
        df["capacity_mw"] = df[capacity_col]
        df["capacity_is_proxy"] = False
    else:
        p99 = df.groupby("zone")["load_mw"].transform(lambda s: s.quantile(0.99))
        df["capacity_mw"] = p99
        df["capacity_is_proxy"] = True

    df["load_to_capacity_ratio"] = df["load_mw"] / df["capacity_mw"]
    df["overload_risk_label"] = (df["load_to_capacity_ratio"] > OVERLOAD_THRESHOLD_RATIO).astype(int)
    return df


def build(load_source: str = "entsoe") -> pd.DataFrame:
    load_df, weather_df = load_raw_data(load_source)

    # Round weather timestamps to hour to guarantee join alignment
    weather_df["timestamp"] = weather_df["timestamp"].dt.floor("h")
    load_df["timestamp"] = pd.to_datetime(load_df["timestamp"]).dt.floor("h")

    # Weather is keyed by a coordinate-derived zone label; load is keyed by country/zone code.
    # For the prototype these are aligned 1:1 via config.ZONE_COORDINATES / ENTSOE_ZONES keys.
    merged = load_df.merge(weather_df, on=["timestamp", "zone"], how="left")

    n_missing_weather = merged["temperature_c"].isna().sum()
    if n_missing_weather:
        print(f"[features] WARNING: {n_missing_weather} rows missing weather match — "
              f"check zone key alignment between load and weather sources")

    merged = add_cyclic_time_features(merged)
    merged = add_lag_and_rolling_features(merged)
    merged = add_overload_proxy_label(merged)

    # Drop rows where lag features are NaN (start of each zone's series) — model can't
    # use them anyway, and silently leaving NaNs in is how you get a training crash later
    before = len(merged)
    merged = merged.dropna(subset=["load_lag_168h"])
    print(f"[features] dropped {before - len(merged)} warm-up rows (insufficient lag history)")

    out_path = DATA_FEATURES_DIR / "training_features.parquet"
    merged.to_parquet(out_path, index=False)
    print(f"[features] wrote {len(merged)} rows, {merged.shape[1]} columns -> {out_path}")
    return merged


if __name__ == "__main__":
    build(load_source="entsoe")
