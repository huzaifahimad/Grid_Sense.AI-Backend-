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

import sys
import json
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import joblib
import numpy as np
import pandas as pd

from config import DATA_FEATURES_DIR, MODELS_DIR
from models.train_baselines import FEATURE_COLS, train_val_test_split
from models.forecasting import WEATHER_COLS, recursive_forecast_batch

HORIZON = 48
HISTORY = 168
REPORT_STEPS = [1, 6, 12, 24, 48]


def build_origins(zone_df: pd.DataFrame, window_start, window_end, stride: int):
    """Origins whose history and full horizon of actuals lie inside the data."""
    zone_df = zone_df.sort_values("timestamp").reset_index(drop=True)
    loads = zone_df["load_mw"].to_numpy(float)
    weather = zone_df[WEATHER_COLS].to_numpy(float)
    ts = zone_df["timestamp"]
    idx = np.where((ts >= window_start) & (ts <= window_end))[0]
    idx = idx[(idx >= HISTORY - 1) & (idx + HORIZON < len(zone_df))][::stride]
    hist_l = np.stack([loads[i - HISTORY + 1:i + 1] for i in idx])
    hist_w = np.stack([weather[i - HISTORY + 1:i + 1] for i in idx])
    actual = np.stack([loads[i + 1:i + 1 + HORIZON] for i in idx])
    seasonal = np.stack([loads[i + 1 - 168:i + 1 - 168 + HORIZON] for i in idx])
    return hist_l, hist_w, pd.DatetimeIndex(ts.iloc[idx]), actual, seasonal


def run_window(model, df, start, end, stride):
    parts = [build_origins(g, start, end, stride) for _, g in df.groupby("zone")]
    hist_l, hist_w, last_ts, actual, seasonal = (
        np.concatenate([p[k] for p in parts]) if k != 2
        else pd.DatetimeIndex(np.concatenate([p[2].to_numpy() for p in parts]))
        for k in range(5)
    )
    preds = recursive_forecast_batch(model, hist_l, hist_w, last_ts, HORIZON, FEATURE_COLS)
    return preds, actual, seasonal


def main():
    df = pd.read_parquet(DATA_FEATURES_DIR / "training_features.parquet")
    model = joblib.load(MODELS_DIR / "lightgbm_load_forecast.joblib")
    _, val, test = train_val_test_split(df)

    preds, actual, _ = run_window(model, df, val["timestamp"].min(), val["timestamp"].max(), stride=6)
    rel_err = (actual - preds) / preds
    horizons = {
        str(h): {"q10": round(float(np.quantile(rel_err[:, h - 1], 0.10)), 5),
                 "q90": round(float(np.quantile(rel_err[:, h - 1], 0.90)), 5)}
        for h in range(1, HORIZON + 1)
    }
    print(f"[calibrate] calibration origins: {len(preds)} (validation window)")

    t_preds, t_actual, t_seasonal = run_window(model, df, test["timestamp"].min(), test["timestamp"].max(), stride=12)
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
    print(f"[calibrate] test origins: {len(t_preds)}  nominal coverage 0.80, "
          f"empirical {evaluation['p10_p90_coverage_overall']:.3f}")
    for h in REPORT_STEPS:
        s = evaluation["per_step"][str(h)]
        print(f"[calibrate]   h={h:2d}  band=[{horizons[str(h)]['q10']:+.3f}, {horizons[str(h)]['q90']:+.3f}]  "
              f"coverage={s['coverage']:.3f}  MAPE lgbm={s['mape_recursive_lgbm']:.2f}%  "
              f"seasonal-naive={s['mape_seasonal_naive']:.2f}%")

    artifact = {
        "method": "split-conformal relative-error quantiles of the recursive LightGBM forecaster, "
                  "calibrated on the validation window, evaluated once on the test window",
        "nominal_coverage": 0.8,
        "weather_assumption": "persistence (same hour, most recent observed day)",
        "horizons": horizons,
        "evaluation": evaluation,
    }
    out = MODELS_DIR / "forecast_intervals.json"
    out.write_text(json.dumps(artifact, indent=2))
    print(f"[calibrate] saved -> {out}")


if __name__ == "__main__":
    main()
