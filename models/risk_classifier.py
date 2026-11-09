"""
GridSense AI — Overload Risk Classifier

LightGBM classifier per ML Model Design Section 2. Deliberately NOT deep
learning — imbalanced tabular data, and trees give SHAP explainability
for free, which the shed-schedule recommendation requires (TRD Section 4).

*** VERCEL-COMPATIBLE VERSION ***
SHAP is used HERE at training time only, to precompute per-scenario
feature importances and save them as a small lookup table. The live API
does NOT import shap/numba/llvmlite at all — those packages alone add
~95MB to a serverless function (llvmlite 57MB + scipy 33MB + numba 3.7MB,
all pulled in transitively by shap), which is most of Vercel's 250MB
function size budget gone before your actual model code loads. Precomputing
importances here means the serving function only needs lightgbm+pandas.

Trade-off, stated plainly: explanations become "typical top factors for
this type of scenario" (bucketed by risk level) rather than a live,
request-specific SHAP computation. This is a real accuracy-of-explanation
cost, not free — flag it in the model card if this ships.

Usage:
    python -m models.risk_classifier
"""

import sys
import json
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import average_precision_score, brier_score_loss, precision_recall_curve
from config import DATA_FEATURES_DIR, MODELS_OUT_DIR

try:
    import lightgbm as lgb
except ImportError:
    raise ImportError("Missing dependency. Run: pip install lightgbm scikit-learn")

# NOTE: shap is imported lazily inside precompute_bucketed_shap — it is a
# training-only dependency and must NOT be imported at module level, because
# api/main.py imports FEATURE_COLS from this module on every request.


FEATURE_COLS = [
    "load_lag_1h", "load_lag_24h", "load_roll_mean_24h", "load_roll_std_24h",
    "temperature_c", "humidity_pct", "wind_speed_ms",
    # NOTE: load_to_capacity_ratio deliberately EXCLUDED — it's a near-direct copy of
    # the label (label = ratio > threshold), so including it causes the classifier to
    # read the answer off the feature instead of learning anything. Found via live
    # testing on 2026-08-27 (all zones saturated to identical ~0.999 risk score).
    # See docs/04_Pipeline_Documentation.md for the full trace.
    "hour_sin", "hour_cos", "dow_sin", "dow_cos", "month_sin", "month_cos",
    "is_weekend",
]
TARGET_COL = "overload_risk_label"

# Same reasoning that produced lightgbm_load_forecast_no_weather.joblib, applied to the
# alarm half of the product. NASA POWER reanalysis publishes an hour days after it
# happens while ENTSO-E publishes load far sooner, so every served frame carries a tail
# of load-only hours. This classifier's dropna(subset=FEATURE_COLS) deletes exactly that
# tail, which pins /risk to the last weather-published hour forever (measured: 90h on the
# committed parquet, 162h on a simulated refreshed frame). The reduced set below uses
# only features live_refresh fills from load alone plus the calendar, so a second
# artifact can score the blind tail. Trained by scripts/train_risk_no_weather.py.
WEATHER_FEATURE_COLS = ["temperature_c", "humidity_pct", "wind_speed_ms"]
FEATURE_COLS_NO_WEATHER = [c for c in FEATURE_COLS if c not in WEATHER_FEATURE_COLS]

# Risk buckets for precomputed explanation lookup — coarse but Vercel-cheap
RISK_BUCKETS = [
    ("low", 0.0, 0.3),
    ("moderate", 0.3, 0.6),
    ("elevated", 0.6, 0.8),
    ("critical", 0.8, 1.01),
]


def walk_forward_split(df: pd.DataFrame, test_frac: float = 0.2):
    df = df.sort_values("timestamp")
    cutoff_idx = int(len(df) * (1 - test_frac))
    cutoff_time = df.iloc[cutoff_idx]["timestamp"]
    return df[df["timestamp"] < cutoff_time], df[df["timestamp"] >= cutoff_time]


def precompute_bucketed_shap(model, test_valid: pd.DataFrame, probs: np.ndarray) -> dict:
    """
    Runs SHAP ONCE, here, at training time (not per-request in the API).
    Groups rows by risk bucket, averages absolute SHAP contribution per
    feature within each bucket, and saves the top-3 features per bucket as
    a small JSON lookup. This is what makes it safe to drop shap/numba/
    llvmlite from the live serving dependencies entirely.
    """
    try:
        import shap
    except ImportError:
        raise ImportError("shap is a training-only dependency. Run: pip install shap")
    print("[risk_classifier] computing SHAP values once at training time "
          "(this is the only place shap runs — not in the live API)...")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(test_valid[FEATURE_COLS])

    # SHAP's binary-classifier output format has changed between versions before
    # (sometimes a list of two arrays [neg_class, pos_class], sometimes a single
    # 2D array for the positive class). Fail loudly if it's neither shape we
    # know how to handle, rather than silently reading the wrong array — a
    # silent misread here would produce plausible-looking but wrong "top
    # factors" with no error anywhere.
    n_rows, n_features = len(test_valid), len(FEATURE_COLS)
    if isinstance(shap_values, list):
        assert len(shap_values) == 2, (
            f"Expected shap_values list of length 2 (neg/pos class), got {len(shap_values)}. "
            f"SHAP's output format has changed — inspect shap.__version__ and fix this branch."
        )
        vals = shap_values[1]
    else:
        vals = shap_values
    assert vals.shape == (n_rows, n_features), (
        f"Expected SHAP values shape ({n_rows}, {n_features}), got {vals.shape}. "
        f"SHAP's output format has changed in a way this code doesn't handle — "
        f"fix this before trusting the bucketed factors below."
    )

    bucketed = {}
    for label, lo, hi in RISK_BUCKETS:
        mask = (probs >= lo) & (probs < hi)
        if mask.sum() == 0:
            bucketed[label] = []
            continue
        mean_abs_shap = np.abs(vals[mask]).mean(axis=0)
        order = np.argsort(-mean_abs_shap)[:3]
        # keep sign from the mean signed contribution, magnitude from mean abs
        mean_signed = vals[mask].mean(axis=0)
        bucketed[label] = [
            {"feature": FEATURE_COLS[i], "contribution": round(float(mean_signed[i]), 4)}
            for i in order
        ]
        print(f"[risk_classifier]   bucket '{label}' ({mask.sum()} rows): "
              f"top factors = {[f['feature'] for f in bucketed[label]]}")

    return bucketed


# Serving gains below this are not worth trading weather-aware accuracy for, and the
# number is deliberately shared with /forecast's FALLBACK_MIN_GAIN_HOURS so the two
# panels cannot disagree about when "newest" means "weather-blind".
DEFAULT_FALLBACK_MIN_GAIN_HOURS = 6.0


def scorable_mask(zone_df: pd.DataFrame, feature_cols: list[str]) -> pd.Series:
    """
    Hours a model with exactly this feature set can be given as input.

    Missing columns count as missing values, not as an error: the point of this
    predicate is "can this row be scored", and a load-only frame without weather
    columns at all answers the same way a weather-NaN row does.
    """
    present = [c for c in feature_cols if c in zone_df.columns]
    if len(present) != len(feature_cols):
        return pd.Series(False, index=zone_df.index)
    return zone_df[present].notna().all(axis=1)


def weather_frontier(zone_df: pd.DataFrame) -> pd.Timestamp | None:
    """Newest hour whose weather features are all present, or None if there is none."""
    complete = zone_df.loc[scorable_mask(zone_df, FEATURE_COLS), "timestamp"]
    return complete.max() if len(complete) else None


def blind_tail_hours(zone_df: pd.DataFrame) -> float:
    """
    Hours between the newest row in the frame and the newest row the weather-aware
    classifier can still score. This is the staleness /risk carries on every request;
    it grows one hour per hour of uptime because NASA POWER never catches up to load.
    """
    frontier = weather_frontier(zone_df)
    if frontier is None or not len(zone_df):
        return 0.0
    return max(0.0, (zone_df["timestamp"].max() - frontier).total_seconds() / 3600.0)


def choose_risk_model(zone_df: pd.DataFrame, *, has_fallback: bool,
                      min_gain_hours: float = DEFAULT_FALLBACK_MIN_GAIN_HOURS,
                      ) -> tuple[str, str]:
    """
    Which risk artifact answers for this zone, and why, in one word the API can publish.

    Mirrors /forecast's gating rather than inventing a second policy: the reduced model
    is served only when its own scorable frontier is meaningfully newer than the
    incumbent's, and never when the artifact is absent — an unshipped fallback must
    leave the old behaviour intact, staleness and all.
    """
    if not has_fallback:
        return "weather_aware", "no weather-blind risk artifact loaded"
    incumbent = weather_frontier(zone_df)
    tail = zone_df.loc[scorable_mask(zone_df, FEATURE_COLS_NO_WEATHER), "timestamp"]
    if tail.empty:
        return "weather_aware", "no row either model can score"
    fallback = tail.max()
    if incumbent is None:
        return "no_weather_fallback", "incumbent has no scorable hour at all"
    gain_h = (fallback - incumbent).total_seconds() / 3600.0
    if gain_h >= min_gain_hours:
        return "no_weather_fallback", f"incumbent frontier {incumbent.isoformat()} is {gain_h:.1f}h behind"
    return "weather_aware", f"blind tail only {gain_h:.1f}h < {min_gain_hours:.0f}h minimum gain"


def train_and_evaluate():
    feat_path = DATA_FEATURES_DIR / "training_features.parquet"
    if not feat_path.exists():
        raise FileNotFoundError(f"{feat_path} not found. Run features/build_features.py first.")

    df = pd.read_parquet(feat_path)
    train_full, test_df = walk_forward_split(df)
    train_full = train_full.sort_values("timestamp")
    val_cut = train_full.iloc[int(len(train_full) * 0.9)]["timestamp"]
    train_df = train_full[train_full["timestamp"] < val_cut]
    val_df = train_full[train_full["timestamp"] >= val_cut]

    train_valid = train_df.dropna(subset=FEATURE_COLS + [TARGET_COL])
    val_valid = val_df.dropna(subset=FEATURE_COLS + [TARGET_COL])
    test_valid = test_df.dropna(subset=FEATURE_COLS + [TARGET_COL])

    pos_rate = train_valid[TARGET_COL].mean()
    print(f"[risk_classifier] positive class rate (train): {pos_rate:.2%}")
    if pos_rate < 0.001:
        print("[risk_classifier] WARNING: positive class is extremely rare — proxy threshold "
              "in config.OVERLOAD_THRESHOLD_RATIO may need adjustment, this data won't train a useful classifier")

    model = lgb.LGBMClassifier(
        n_estimators=500,
        learning_rate=0.03,
        num_leaves=31,
        is_unbalance=True,
        random_state=42,
    )
    model.fit(
        train_valid[FEATURE_COLS], train_valid[TARGET_COL],
        eval_set=[(val_valid[FEATURE_COLS], val_valid[TARGET_COL])],
        callbacks=[lgb.early_stopping(30, verbose=False)],
    )
    print(f"[risk_classifier] best_iteration={model.best_iteration_} (selected on val, not test)")

    probs = model.predict_proba(test_valid[FEATURE_COLS])[:, 1]
    pr_auc = average_precision_score(test_valid[TARGET_COL], probs)
    print(f"[risk_classifier] PR-AUC (test): {pr_auc:.4f}  <- optimize/report THIS, not ROC-AUC")

    y_test = test_valid[TARGET_COL].to_numpy()
    brier = brier_score_loss(y_test, probs)
    print(f"[risk_classifier] Brier score (test): {brier:.4f}  "
          f"(base-rate reference: {brier_score_loss(y_test, np.full_like(probs, y_test.mean())):.4f})")
    print("[risk_classifier] reliability (predicted bucket -> observed overload rate):")
    for label, lo, hi in RISK_BUCKETS:
        mask = (probs >= lo) & (probs < hi)
        if mask.sum():
            print(f"[risk_classifier]   {label:9s} n={mask.sum():5d}  "
                  f"mean_pred={probs[mask].mean():.3f}  observed={y_test[mask].mean():.3f}")

    precision, recall, thresholds = precision_recall_curve(test_valid[TARGET_COL], probs)
    target_precision = 0.5
    idx = np.argmin(np.abs(precision[:-1] - target_precision))
    print(f"[risk_classifier] at precision~{precision[idx]:.2f}, recall={recall[idx]:.2f}, "
          f"threshold={thresholds[idx]:.3f}  <- tune this operating point with the actual utility, not arbitrarily")

    model_path = MODELS_OUT_DIR / "risk_classifier.joblib"
    joblib.dump(model, model_path)
    print(f"[risk_classifier] saved model -> {model_path}")

    # SHAP runs ONCE here, output saved as plain JSON — the live API reads this
    # JSON file, it never imports shap/numba/llvmlite at request time
    bucketed_factors = precompute_bucketed_shap(model, test_valid, probs)
    factors_path = MODELS_OUT_DIR / "risk_factors_by_bucket.json"
    with open(factors_path, "w") as f:
        json.dump(bucketed_factors, f, indent=2)
    print(f"[risk_classifier] saved precomputed factors -> {factors_path}")
    print("[risk_classifier] *** shap/numba/llvmlite are NOT needed at serve time — "
          "requirements-api.txt should NOT include them ***")

    return model, pr_auc


if __name__ == "__main__":
    train_and_evaluate()
