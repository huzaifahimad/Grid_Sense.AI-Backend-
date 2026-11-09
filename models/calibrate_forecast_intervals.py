"""
GridSense AI — Calibrate horizon-dependent forecast intervals.

Runs the exact recursive forecaster the API serves (models/forecasting.py)
from many origins inside the VALIDATION window, collects relative errors
(actual - pred) / pred per horizon step, and stores their 10th/90th
percentiles. This is split-conformal-style calibration: the band widens with
lead time because recursive errors compound, instead of a fixed ±8%.

Coverage and multi-step accuracy are then measured once on the TEST window
and written alongside the quantiles so the artifact carries its own evidence.

Usage (after models/train_baselines.py):
    python -m models.calibrate_forecast_intervals
"""

import argparse
import sys
import json
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import joblib
import numpy as np
import pandas as pd

from config import DATA_FEATURES_DIR, MODELS_DIR, MODELS_OUT_DIR
from models.train_baselines import (
    FEATURE_COLS, FEATURE_COLS_NO_WEATHER, train_val_test_split,
)
from models.forecasting import WEATHER_COLS, recursive_forecast_batch

HORIZON = 48
HISTORY = 168
REPORT_STEPS = [1, 6, 12, 24, 48]


def build_origins(zone_df: pd.DataFrame, window_start, window_end, stride: int,
                  horizon: int = HORIZON):
    """Origins whose history and full horizon of actuals lie inside contiguous data."""
    zone_df = zone_df.dropna(subset=["load_mw", *WEATHER_COLS])
    zone_df = zone_df.sort_values("timestamp").reset_index(drop=True)
    loads = zone_df["load_mw"].to_numpy(float)
    weather = zone_df[WEATHER_COLS].to_numpy(float)
    ts = zone_df["timestamp"]
    idx = np.where((ts >= window_start) & (ts <= window_end))[0]
    idx = idx[(idx >= HISTORY - 1) & (idx + horizon < len(zone_df))][::stride]

    # recursive_forecast_batch derives load_lag_1h/24h/168h by *row* offset and
    # stamps its output from the origin timestamp alone. Across a missing hour
    # those two stop agreeing: the features are hours stale while the labels look
    # right, so the forecast is quietly wrong instead of failing. Reject any
    # origin whose history block or horizon of actuals straddles a gap.
    if len(idx):
        steps = ts.diff().dt.total_seconds() / 3600.0
        is_gap = (steps != 1.0).to_numpy(copy=True)
        is_gap[0] = True  # a first row has no predecessor to measure against
        prefix = np.concatenate([[0], np.cumsum(is_gap)])
        lo = idx - HISTORY + 2
        hi = idx + horizon
        n_gaps = prefix[hi + 1] - prefix[lo]
        dropped = int((n_gaps != 0).sum())
        if dropped:
            print(f"[calibrate] {dropped} origin(s) skipped at "
                  f"{str(pd.Timestamp(ts.iloc[idx[0]]))[:13]}Z+ for straddling a time gap")
        idx = idx[n_gaps == 0]

    if not len(idx):
        empty = np.empty((0, horizon))
        return (np.empty((0, HISTORY)), np.empty((0, HISTORY, len(WEATHER_COLS))),
                pd.DatetimeIndex([]), empty, empty)

    hist_l = np.stack([loads[i - HISTORY + 1:i + 1] for i in idx])
    hist_w = np.stack([weather[i - HISTORY + 1:i + 1] for i in idx])
    actual = np.stack([loads[i + 1:i + 1 + horizon] for i in idx])
    seasonal = np.stack([loads[i + 1 - 168:i + 1 - 168 + horizon] for i in idx])
    return hist_l, hist_w, pd.DatetimeIndex(ts.iloc[idx]), actual, seasonal


def run_window(model, df, start, end, stride, horizon: int = HORIZON,
               feature_cols: list[str] = FEATURE_COLS):
    parts = [build_origins(g, start, end, stride, horizon) for _, g in df.groupby("zone")]
    hist_l, hist_w, last_ts, actual, seasonal = (
        np.concatenate([p[k] for p in parts]) if k != 2
        else pd.DatetimeIndex(np.concatenate([p[2].to_numpy() for p in parts]))
        for k in range(5)
    )
    preds = recursive_forecast_batch(model, hist_l, hist_w, last_ts, horizon, feature_cols)
    return preds, actual, seasonal


def calibrate(model, df, val, test, feature_cols, label: str,
              models_dir: Path = MODELS_DIR) -> dict:
    """
    Bands for one forecaster, from that forecaster's own held-out errors. A model with
    fewer inputs is less accurate, so it gets wider bands: reusing the primary model's
    quantiles would claim coverage the fallback does not have. Origins here are required
    to have weather so an actual exists to score against — the weather-blind tail the
    fallback will serve is the last thing in the dataset and has no future hours yet —
    and the reduced model ignores those columns, so this is a faithful surrogate.
    """
    preds, actual, _ = run_window(model, df, val["timestamp"].min(), val["timestamp"].max(),
                                  stride=6, feature_cols=feature_cols)
    rel_err = (actual - preds) / preds
    horizons = {
        str(h): {"q10": round(float(np.quantile(rel_err[:, h - 1], 0.10)), 5),
                 "q90": round(float(np.quantile(rel_err[:, h - 1], 0.90)), 5)}
        for h in range(1, HORIZON + 1)
    }
    print(f"[calibrate] {label}: {len(preds)} calibration origins (validation window)")

    t_preds, t_actual, t_seasonal = run_window(
        model, df, test["timestamp"].min(), test["timestamp"].max(),
        stride=12, feature_cols=feature_cols)
    q10 = np.array([horizons[str(h)]["q10"] for h in range(1, HORIZON + 1)])
    q90 = np.array([horizons[str(h)]["q90"] for h in range(1, HORIZON + 1)])
    lo, hi = t_preds * (1 + q10), t_preds * (1 + q90)
    covered = (t_actual >= lo) & (t_actual <= hi)
    mape = np.abs(t_actual - t_preds) / t_actual * 100
    mape_seasonal = np.abs(t_actual - t_seasonal) / t_actual * 100

    evaluation = {
        "test_origins": int(len(t_preds)),
        "p10_p90_coverage_overall": round(float(covered.mean()), 4),
        "per_step": {
            str(h): {
                "coverage": round(float(covered[:, h - 1].mean()), 4),
                "mape_recursive_lgbm": round(float(mape[:, h - 1].mean()), 3),
                "mape_seasonal_naive": round(float(mape_seasonal[:, h - 1].mean()), 3),
            }
            for h in REPORT_STEPS
        },
    }
    print(f"[calibrate] {label}: test origins {len(t_preds)}, nominal coverage 0.80, "
          f"empirical {evaluation['p10_p90_coverage_overall']:.3f}")
    for h in REPORT_STEPS:
        s = evaluation["per_step"][str(h)]
        print(f"[calibrate]   h={h:2d}  band=[{horizons[str(h)]['q10']:+.3f}, {horizons[str(h)]['q90']:+.3f}]  "
              f"coverage={s['coverage']:.3f}  MAPE lgbm={s['mape_recursive_lgbm']:.2f}%  "
              f"seasonal-naive={s['mape_seasonal_naive']:.2f}%")

    return {
        "method": f"split-conformal relative-error quantiles of the recursive {label} forecaster, "
                  "calibrated on the validation window, evaluated once on the test window",
        "nominal_coverage": 0.8,
        "weather_assumption": "persistence (same hour, most recent observed day)"
                              if len(feature_cols) == len(FEATURE_COLS)
                              else "none — this model has no weather inputs",
        "horizons": horizons,
        "evaluation": evaluation,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--models-dir", type=Path, default=MODELS_OUT_DIR,
                    help="where the forecaster(s) are read from and the interval JSON(s) "
                         "are written. Defaults to the retrain output dir; falls back to "
                         "the served artifact dir when nothing has been retrained. Never "
                         "point it at data/models unless you mean to overwrite it.")
    args = ap.parse_args()

    models_dir = args.models_dir
    primary_path = models_dir / "lightgbm_load_forecast.joblib"
    if not primary_path.exists():
        models_dir = MODELS_DIR
        primary_path = models_dir / "lightgbm_load_forecast.joblib"
    print(f"[calibrate] models dir: {models_dir}")

    df = pd.read_parquet(DATA_FEATURES_DIR / "training_features.parquet")
    _, val, test = train_val_test_split(df)

    model = joblib.load(primary_path)
    artifact = calibrate(model, df, val, test, FEATURE_COLS, "weather-aware LightGBM",
                         models_dir=models_dir)
    out = models_dir / "forecast_intervals.json"
    out.write_text(json.dumps(artifact, indent=2))
    print(f"[calibrate] saved -> {out}")

    fallback_path = models_dir / "lightgbm_load_forecast_no_weather.joblib"
    if fallback_path.exists():
        fb = joblib.load(fallback_path)
        fb_artifact = calibrate(fb, df, val, test, FEATURE_COLS_NO_WEATHER,
                                "no-weather fallback", models_dir=models_dir)
        fb_out = models_dir / "forecast_intervals_no_weather.json"
        fb_out.write_text(json.dumps(fb_artifact, indent=2))
        print(f"[calibrate] saved -> {fb_out}")
        primary = artifact["evaluation"]["p10_p90_coverage_overall"]
        gained = fb_artifact["evaluation"]["p10_p90_coverage_overall"]
        print(f"[calibrate] fallback coverage {gained:.3f} vs weather-aware {primary:.3f}")
    else:
        print(f"[calibrate] no fallback model at {fallback_path} — "
              f"run models.train_baselines first; /forecast will stay weather-gated")


if __name__ == "__main__":
    main()
