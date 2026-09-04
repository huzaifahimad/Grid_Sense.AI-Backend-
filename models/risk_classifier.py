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
from sklearn.metrics import average_precision_score, precision_recall_curve
from config import DATA_FEATURES_DIR, MODELS_DIR

try:
    import lightgbm as lgb
    import shap
except ImportError:
    raise ImportError("Missing dependency. Run: pip install lightgbm shap scikit-learn --break-system-packages")


FEATURE_COLS = [
    "load_lag_1h", "load_lag_24h", "load_roll_mean_24h", "load_roll_std_24h",
    "temperature_c", "humidity_pct", "wind_speed_ms",
    # NOTE: load_to_capacity_ratio deliberately EXCLUDED — it's a near-direct copy of
    # the label (label = ratio > threshold), so including it causes the classifier to
    # read the answer off the feature instead of learning anything. Found via live
    # testing on 2026-08-27 (all zones saturated to identical ~0.999 risk score).
    # See PIPELINE_VALIDATION_REPORT.md for the full trace.
    "hour_sin", "hour_cos", "dow_sin", "dow_cos", "month_sin", "month_cos",
    "is_weekend",
]
TARGET_COL = "overload_risk_label"

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


def train_and_evaluate():
    feat_path = DATA_FEATURES_DIR / "training_features.parquet"
    if not feat_path.exists():
        raise FileNotFoundError(f"{feat_path} not found. Run features/build_features.py first.")

    df = pd.read_parquet(feat_path)
    train_df, test_df = walk_forward_split(df)

    train_valid = train_df.dropna(subset=FEATURE_COLS + [TARGET_COL])
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
        eval_set=[(test_valid[FEATURE_COLS], test_valid[TARGET_COL])],
        callbacks=[lgb.early_stopping(30, verbose=False)],
    )

    probs = model.predict_proba(test_valid[FEATURE_COLS])[:, 1]
    pr_auc = average_precision_score(test_valid[TARGET_COL], probs)
    print(f"[risk_classifier] PR-AUC (test): {pr_auc:.4f}  <- optimize/report THIS, not ROC-AUC")

    precision, recall, thresholds = precision_recall_curve(test_valid[TARGET_COL], probs)
    target_precision = 0.5
    idx = np.argmin(np.abs(precision[:-1] - target_precision))
    print(f"[risk_classifier] at precision~{precision[idx]:.2f}, recall={recall[idx]:.2f}, "
          f"threshold={thresholds[idx]:.3f}  <- tune this operating point with the actual utility, not arbitrarily")

    model_path = MODELS_DIR / "risk_classifier.joblib"
    joblib.dump(model, model_path)
    print(f"[risk_classifier] saved model -> {model_path}")

    # SHAP runs ONCE here, output saved as plain JSON — the live API reads this
    # JSON file, it never imports shap/numba/llvmlite at request time
    bucketed_factors = precompute_bucketed_shap(model, test_valid, probs)
    factors_path = MODELS_DIR / "risk_factors_by_bucket.json"
    with open(factors_path, "w") as f:
        json.dump(bucketed_factors, f, indent=2)
    print(f"[risk_classifier] saved precomputed factors -> {factors_path}")
    print("[risk_classifier] *** shap/numba/llvmlite are NOT needed at serve time — "
          "requirements-api.txt should NOT include them ***")

    return model, pr_auc


if __name__ == "__main__":
    train_and_evaluate()
