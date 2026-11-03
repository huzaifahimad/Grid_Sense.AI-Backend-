"""
GridSense AI — Recursive multi-step load forecasting (serving-safe).

The LightGBM forecaster is a 1-step-ahead model (its strongest feature is
load_lag_1h). To produce a genuine H-hour-ahead forecast from the latest
observation, each step's prediction is fed back as the lag input for the next
step. Feature construction mirrors features/build_features.py exactly
(shift(1) lags, rolling windows over the previous N hours, same cyclic
encodings) so the model sees train-time-consistent inputs.

Future weather is unknown to this system (no NWP feed yet), so weather for a
future hour uses the observed value at the same hour on the most recent
available day ("weather persistence"). This is a stated assumption, not a
forecast — replacing it with a real NWP feed is the first upgrade path.

Imports only numpy/pandas so the live API stays lean.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

WEATHER_COLS = ["temperature_c", "humidity_pct", "wind_speed_ms"]
MIN_HISTORY_HOURS = 168


def _calendar_matrix(ts: pd.DatetimeIndex) -> dict:
    hour = ts.hour.to_numpy()
    dow = ts.dayofweek.to_numpy()
    month = ts.month.to_numpy()
    return {
        "hour_sin": np.sin(2 * np.pi * hour / 24),
        "hour_cos": np.cos(2 * np.pi * hour / 24),
        "dow_sin": np.sin(2 * np.pi * dow / 7),
        "dow_cos": np.cos(2 * np.pi * dow / 7),
        "is_weekend": (dow >= 5).astype(int),
        "month_sin": np.sin(2 * np.pi * month / 12),
        "month_cos": np.cos(2 * np.pi * month / 12),
    }


def recursive_forecast_batch(model, loads: np.ndarray, weather: np.ndarray,
                             last_ts: pd.DatetimeIndex, horizon: int,
                             feature_cols: list[str]) -> np.ndarray:
    """
    loads:   (n_origins, n_hist) hourly load history ending at each origin
    weather: (n_origins, n_hist, len(WEATHER_COLS)) aligned with loads
    last_ts: (n_origins,) timestamp of the last observed hour per origin
    Returns (n_origins, horizon) point forecasts for steps 1..horizon.
    """
    n_origins, n_hist = loads.shape
    if n_hist < MIN_HISTORY_HOURS:
        raise ValueError(f"need >= {MIN_HISTORY_HOURS}h of history, got {n_hist}")

    buf = np.concatenate([loads.astype(float), np.empty((n_origins, horizon))], axis=1)
    out = np.empty((n_origins, horizon))
    for step in range(1, horizon + 1):
        end = n_hist + step - 1
        prev24 = buf[:, end - 24:end]
        days_back = -(-step // 24)
        w = weather[:, end - 24 * days_back, :]
        ts = pd.DatetimeIndex(last_ts) + pd.Timedelta(hours=step)
        feats = {
            "load_lag_1h": buf[:, end - 1],
            "load_lag_24h": buf[:, end - 24],
            "load_lag_168h": buf[:, end - 168],
            "load_roll_mean_24h": prev24.mean(axis=1),
            "load_roll_std_24h": prev24.std(axis=1, ddof=1),
            "load_roll_mean_168h": buf[:, end - 168:end].mean(axis=1),
            **{c: w[:, i] for i, c in enumerate(WEATHER_COLS)},
            **_calendar_matrix(ts),
        }
        pred = model.predict(pd.DataFrame(feats)[feature_cols])
        buf[:, end] = pred
        out[:, step - 1] = pred
    return out


def recursive_forecast(model, history: pd.DataFrame, horizon: int, feature_cols: list[str]) -> pd.DataFrame:
    """Single-origin wrapper. history: one zone, contiguous hourly, sorted."""
    history = history.sort_values("timestamp")
    loads = history["load_mw"].to_numpy(dtype=float)[None, :]
    weather = history[WEATHER_COLS].to_numpy(dtype=float)[None, :, :]
    last_ts = pd.DatetimeIndex([history["timestamp"].iloc[-1]])
    preds = recursive_forecast_batch(model, loads, weather, last_ts, horizon, feature_cols)[0]
    steps = np.arange(1, horizon + 1)
    return pd.DataFrame({
        "timestamp": last_ts[0] + pd.to_timedelta(steps, unit="h"),
        "step": steps,
        "p50": preds,
    })


def apply_intervals(forecast: pd.DataFrame, intervals: dict | None) -> pd.DataFrame:
    """
    intervals["horizons"]: {"1": {"q10": -0.03, "q90": 0.03}, ...} holding
    relative-error quantiles ((actual - pred) / pred) per horizon step,
    calibrated on a held-out window by models/calibrate_forecast_intervals.py.
    Falls back to a fixed ±8% band only if no calibration artifact exists.
    """
    out = forecast.copy()
    if not intervals:
        out["p10"] = out["p50"] * 0.92
        out["p90"] = out["p50"] * 1.08
        return out
    table = intervals["horizons"]
    max_h = max(int(k) for k in table)
    key = out["step"].clip(upper=max_h).astype(str)
    out["p10"] = out["p50"] * (1 + key.map(lambda k: table[k]["q10"]))
    out["p90"] = out["p50"] * (1 + key.map(lambda k: table[k]["q90"]))
    return out
