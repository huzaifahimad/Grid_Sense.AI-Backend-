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


def reindex_hourly_grid(df: pd.DataFrame) -> pd.DataFrame:
    """
    One row per UTC hour per zone, with unpublished hours present but empty.

    groupby().shift(N) counts *rows*, not hours. ENTSO-E withholds some hours
    (58 in Spain, 23 each in Greece and southern Italy over this two-year pull),
    so without this step the row right after a hole carries load_lag_1h from the
    hour before the hole — a 2-hour-old value labelled as 1 hour old — and every
    lag feature is quietly wrong for the N rows that follow each gap. Making the
    grid explicit turns those into NaN, which the existing dropna removes.
    Cost measured on the real pull: ~0.1-0.3% of rows per zone.
    """
    parts = []
    for zone, g in df.groupby("zone"):
        g = g.sort_values("timestamp")
        tz = str(g["timestamp"].dt.tz)
        full = pd.date_range(g["timestamp"].min(), g["timestamp"].max(), freq="h", tz=tz)
        n_missing = len(full) - len(g)
        if n_missing:
            print(f"[features] {zone}: adding {n_missing} unpublished hour(s) as empty rows "
                  f"so lag shifts stay time-correct")
        g = g.set_index("timestamp").reindex(full)
        g.index.name = "timestamp"
        g["zone"] = zone
        for col in ("load_source", "weather_source"):
            if col in g and g[col].notna().any():
                g[col] = g[col].fillna(g[col].dropna().iloc[0])
        parts.append(g.reset_index())
    return pd.concat(parts, ignore_index=True).sort_values(["zone", "timestamp"]).reset_index(drop=True)


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
    # Both frames now carry a "source" stamp; rename before the merge so provenance survives
    # as two explicit columns instead of collapsing into source_x/source_y.
    for name, frame in (("load", load_df), ("weather", weather_df)):
        if "source" not in frame.columns:
            raise ValueError(
                f"{name} frame has no 'source' column — an unstamped frame cannot be told "
                f"apart from the synthetic bootstrap stand-in that writes the same file name. "
                f"Re-run the ingestion script."
            )
    load_df = load_df.rename(columns={"source": "load_source"})
    weather_df = weather_df.rename(columns={"source": "weather_source"})

    merged = load_df.merge(weather_df, on=["timestamp", "zone"], how="left")

    n_missing_weather = merged["temperature_c"].isna().sum()
    if n_missing_weather:
        print(f"[features] WARNING: {n_missing_weather} rows missing weather match — "
              f"check zone key alignment between load and weather sources")

    merged = reindex_hourly_grid(merged)
    merged = add_cyclic_time_features(merged)
    merged = add_lag_and_rolling_features(merged)
    merged = add_overload_proxy_label(merged)

    # Drop rows that are not usable samples at all: series warm-up, the grid
    # placeholder rows reindex_hourly_grid adds for unpublished hours, and the rows
    # immediately after a hole (their lag would straddle the gap).
    before = len(merged)
    merged = merged.dropna(subset=["load_mw", "load_lag_1h", "load_lag_24h", "load_lag_168h"])
    print(f"[features] dropped {before - len(merged)} rows (lag warm-up, unpublished hours)")

    # NASA POWER publishes roughly two days behind, so the freshest hours of real
    # ENTSO-E load arrive with no weather row. Those observations are kept: deleting
    # them would delete real data and push the served forecast origin hours further
    # into the past. What they must not do is feed the *weather-aware* model —
    # LightGBM never errors on a NaN, it routes it down a learned default direction
    # and returns a confident-looking number whose accuracy was never measured. So
    # the rows are stamped and the serving path chooses the model whose inputs exist.
    blind = merged["temperature_c"].isna()
    merged.loc[blind, "weather_source"] = "unavailable"
    if blind.any():
        print(f"[features] {int(blind.sum())} row(s) have load but no published weather yet "
              f"(earliest such hour {merged.loc[blind, 'timestamp'].min()}) — served by the "
              f"no-weather fallback, never by the weather-aware model")
    print(f"[features] last weather-complete hour {merged.loc[~blind, 'timestamp'].max()}; "
          f"last observed load hour {merged['timestamp'].max()}")

    out_path = DATA_FEATURES_DIR / "training_features.parquet"
    merged.to_parquet(out_path, index=False)
    print(f"[features] wrote {len(merged)} rows, {merged.shape[1]} columns -> {out_path}")
    return merged


if __name__ == "__main__":
    build(load_source="entsoe")
