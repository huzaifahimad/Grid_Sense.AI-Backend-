"""
GridSense AI — Baseline Models

TRD FR-12: every model must beat persistence and seasonal-naive baselines.
This script establishes those baselines PLUS a LightGBM benchmark before
anyone is allowed to justify the LSTM+CNN's added complexity.

Usage:
    python -m models.train_baselines
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import mean_absolute_percentage_error, mean_absolute_error, mean_squared_error
from config import DATA_FEATURES_DIR, MODELS_DIR

try:
    import lightgbm as lgb
except ImportError:
    raise ImportError("Missing dependency. Run: pip install lightgbm --break-system-packages")


FEATURE_COLS = [
    "load_lag_1h", "load_lag_24h", "load_lag_168h",
    "load_roll_mean_24h", "load_roll_std_24h", "load_roll_mean_168h",
    "temperature_c", "humidity_pct", "wind_speed_ms",
    "hour_sin", "hour_cos", "dow_sin", "dow_cos", "month_sin", "month_cos",
    "is_weekend",
]
TARGET_COL = "load_mw"


def walk_forward_split(df: pd.DataFrame, test_frac: float = 0.2):
    """
    Expanding-window split, NOT random k-fold (TRD FR-13). Sorted by time,
    last test_frac of each zone's history held out.
    """
    df = df.sort_values("timestamp")
    cutoff_idx = int(len(df) * (1 - test_frac))
    cutoff_time = df.iloc[cutoff_idx]["timestamp"]
    train = df[df["timestamp"] < cutoff_time]
    test = df[df["timestamp"] >= cutoff_time]
    print(f"[baselines] train: {len(train)} rows (< {cutoff_time}), test: {len(test)} rows (>= {cutoff_time})")
    return train, test


def train_val_test_split(df: pd.DataFrame, test_frac: float = 0.2, val_frac: float = 0.1):
    """
    Time-ordered three-way split. Early stopping uses `val` (the tail of the
    training window); `test` is scored exactly once. Early-stopping on the
    test set would select the iteration count on the very data used to report
    accuracy and bias the reported metrics optimistically.
    """
    train_full, test = walk_forward_split(df, test_frac)
    train_full = train_full.sort_values("timestamp")
    val_cut = train_full.iloc[int(len(train_full) * (1 - val_frac))]["timestamp"]
    fit = train_full[train_full["timestamp"] < val_cut]
    val = train_full[train_full["timestamp"] >= val_cut]
    print(f"[baselines] fit: {len(fit)} rows, val: {len(val)} rows (>= {val_cut})")
    return fit, val, test


def evaluate(y_true, y_pred, label: str) -> dict:
    mape = mean_absolute_percentage_error(y_true, y_pred) * 100
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    print(f"[baselines] {label:20s}  MAPE={mape:6.2f}%   MAE={mae:8.2f} MW   RMSE={rmse:8.2f} MW")
    return {"label": label, "mape": mape, "mae": mae, "rmse": rmse}


def persistence_baseline(test: pd.DataFrame) -> dict:
    """Prediction = load 24h ago (t-24h). The bar any real model must clear."""
    valid = test.dropna(subset=["load_lag_24h", TARGET_COL])
    return evaluate(valid[TARGET_COL], valid["load_lag_24h"], "Persistence (t-24h)")


def seasonal_naive_baseline(test: pd.DataFrame) -> dict:
    """Prediction = load same hour, same day-of-week, 1 week ago (t-168h)."""
    valid = test.dropna(subset=["load_lag_168h", TARGET_COL])
    return evaluate(valid[TARGET_COL], valid["load_lag_168h"], "Seasonal-naive (t-168h)")


def lightgbm_baseline(train: pd.DataFrame, val: pd.DataFrame, test: pd.DataFrame) -> dict:
    train_valid = train.dropna(subset=FEATURE_COLS + [TARGET_COL])
    val_valid = val.dropna(subset=FEATURE_COLS + [TARGET_COL])
    test_valid = test.dropna(subset=FEATURE_COLS + [TARGET_COL])

    model = lgb.LGBMRegressor(
        n_estimators=500,
        learning_rate=0.03,
        num_leaves=31,
        max_depth=-1,
        min_child_samples=20,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
    )
    model.fit(
        train_valid[FEATURE_COLS], train_valid[TARGET_COL],
        eval_set=[(val_valid[FEATURE_COLS], val_valid[TARGET_COL])],
        callbacks=[lgb.early_stopping(30, verbose=False)],
    )
    print(f"[baselines] LightGBM best_iteration={model.best_iteration_} (selected on val, not test)")

    preds = model.predict(test_valid[FEATURE_COLS])
    result = evaluate(test_valid[TARGET_COL], preds, "LightGBM")

    model_path = MODELS_DIR / "lightgbm_load_forecast.joblib"
    joblib.dump(model, model_path)
    print(f"[baselines] saved LightGBM model -> {model_path}")

    return result


def run_all():
    feat_path = DATA_FEATURES_DIR / "training_features.parquet"
    if not feat_path.exists():
        raise FileNotFoundError(f"{feat_path} not found. Run features/build_features.py first.")

    df = pd.read_parquet(feat_path)
    fit, val, test = train_val_test_split(df)

    print("\n=== Baseline Comparison (this table goes straight into your model card) ===")
    results = [
        persistence_baseline(test),
        seasonal_naive_baseline(test),
        lightgbm_baseline(fit, val, test),
    ]

    results_df = pd.DataFrame(results)
    results_path = MODELS_DIR / "baseline_results.csv"
    results_df.to_csv(results_path, index=False)
    print(f"\n[baselines] results saved -> {results_path}")
    print("[baselines] Your LSTM+CNN (models/train_lstm_cnn.py) must beat LightGBM's MAPE")
    print("[baselines] by a meaningful margin, or LightGBM ships instead — per ML Model Design Section 1.3")

    return results_df


if __name__ == "__main__":
    run_all()
