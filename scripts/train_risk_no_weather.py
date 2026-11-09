"""
GridSense AI — Weather-blind risk classifier (the blind-tail fallback)

WHY THIS EXISTS, MEASURED NOT ASSUMED
-------------------------------------
`models/risk_classifier.py` scores an hour only if its weather features are present:
its `dropna(subset=FEATURE_COLS)` deletes every hour NASA POWER has not published yet.
Reanalysis lags load by days while ENTSO-E publishes load far sooner, so the served
frame always ends in a tail of load-only hours and `/risk/{zone}` can never reach them.
Measured on this repo's committed artifacts (`/tmp/risknw/m2_api.py`):

    committed parquet : newest hour 2026-11-04T17:00Z, weather frontier 2026-11-01T00:00Z
                        -> 90h unscorable; /risk answers 2026-10-30T15:00Z (168.9h old)
    simulated refresh : newest hour 2026-11-07T17:00Z appended without weather
                        -> 162h unscorable; /risk still answers 2026-10-30T15:00Z

That is structural, not transient: the gap grows one hour per hour of uptime. The load
forecaster already solves the identical problem with a dedicated reduced-feature
artifact; the alarm half of the product had no equivalent.

DISCIPLINE THIS SCRIPT KEEPS
----------------------------
* Same honest temporal split as every other model here: `walk_forward_split` on the
  committed frame, contiguous future test window, early stopping on the validation
  slice of the training window. Never shuffled — docs/15 measured that shuffling before
  splitting puts ~80% of the test window into train and inflates accuracy 9-17 pp.
* Same class-imbalance approach as the incumbent: `is_unbalance=True`. Not SMOTE.
* PR-AUC as the primary metric, plus Brier/Brier skill and a reliability table,
  because an operator-facing score must mean the same thing as the incumbent's.
* NOTHING is written to data/models/ or data/features/. Output goes to --out-dir
  (default /tmp/gridsense_risk_no_weather), exactly like the retrain chain's candidate
  dirs. Promotion remains scripts/promote_models.py's gated, human-run decision.

THE QUESTION THIS SCRIPT ANSWERS (not "is it accurate", but "may it serve")
---------------------------------------------------------------------------
A fallback whose score means something different from the incumbent's across the
fallback boundary would be a worse defect than the staleness it fixes: the status label
and the shed priority are hard cuts on the score (models/decision_rules.py). So the
agreement block below measures both models on the SAME rows the incumbent can score,
inside the honest test window, and applies explicit ship gates.

Usage:
    python -m scripts.train_risk_no_weather [--features PATH] [--out-dir PATH]
                                             [--min-pr-auc 0.85] [--max-mean-shift 0.05]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score, brier_score_loss, precision_recall_curve, roc_auc_score,
)

from config import DATA_FEATURES_DIR, MODELS_DIR, OVERLOAD_THRESHOLD_RATIO
from models.decision_rules import ALARM_THRESHOLD, CRITICAL_THRESHOLD, ELEVATED_THRESHOLD
from models.risk_classifier import (
    FEATURE_COLS, FEATURE_COLS_NO_WEATHER, RISK_BUCKETS, TARGET_COL, walk_forward_split,
)

try:
    import lightgbm as lgb
except ImportError:  # pragma: no cover
    raise ImportError("Missing dependency. Run: pip install lightgbm scikit-learn")

DEFAULT_OUT_DIR = Path("/tmp/gridsense_risk_no_weather").resolve()
MODEL_FILE = "risk_classifier_no_weather.joblib"
METRICS_FILE = "risk_classifier_no_weather_metrics.json"
FACTORS_FILE = "risk_factors_by_bucket_no_weather.json"

# Ship gates, stated before the numbers exist. Tuning them after seeing the results
# would make them a description of the result instead of a decision rule.
MIN_PR_AUC = 0.85          # the reduced set must stay near the incumbent's absolute bar
MAX_PR_AUC_DROP_PP = 5.0   # pp of PR-AUC the fallback may lose against the incumbent
MAX_MEAN_SHIFT = 0.05      # |mean(fallback - incumbent)| over jointly-scorable hours
MAX_STATUS_FLIP_RATE = 0.10
MAX_CRITICAL_FLIP_RATE = 0.05
MAX_RECALL_LOSS_PP = 10.0
MAX_ECE = 0.05             # expected calibration error, 10 equal-count bins


def load_frame(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    return df.sort_values("timestamp").reset_index(drop=True)


def train_val_split(train_full: pd.DataFrame, val_frac: float = 0.1):
    """Tail slice of the TRAINING window only. The test window is never used for
    early stopping — selecting iterations on the scored data biases every metric."""
    train_full = train_full.sort_values("timestamp")
    val_cut = train_full.iloc[int(len(train_full) * (1 - val_frac))]["timestamp"]
    return (train_full[train_full["timestamp"] < val_cut],
            train_full[train_full["timestamp"] >= val_cut])


def fit_lgbm(features: list[str], fit_df: pd.DataFrame, val_df: pd.DataFrame, log):
    """Identical hyperparameters to the incumbent, including is_unbalance=True. Only
    the feature count differs, so the head-to-head below measures the missing weather
    and nothing else."""
    model = lgb.LGBMClassifier(
        n_estimators=500,
        learning_rate=0.03,
        num_leaves=31,
        is_unbalance=True,
        random_state=42,
    )
    model.fit(
        fit_df[features], fit_df[TARGET_COL],
        eval_set=[(val_df[features], val_df[TARGET_COL])],
        callbacks=[lgb.early_stopping(30, verbose=False)],
    )
    log(f"  best_iteration={model.best_iteration_} (selected on val, not test)")
    return model


def expected_calibration_error(y: np.ndarray, p: np.ndarray, n_bins: int = 10) -> float:
    """Equal-count binning: |mean predicted - mean observed| weighted by bin size."""
    order = np.argsort(p)
    ys, ps = y[order], p[order]
    edges = np.array_split(np.arange(len(ys)), n_bins)
    ece = 0.0
    for idx in edges:
        if len(idx) == 0:
            continue
        ece += (len(idx) / len(ys)) * abs(float(ps[idx].mean()) - float(ys[idx].mean()))
    return float(ece)


def reliability_rows(y: np.ndarray, p: np.ndarray) -> list[dict]:
    out = []
    for label, lo, hi in RISK_BUCKETS:
        m = (p >= lo) & (p < hi)
        out.append({"bucket": label, "n": int(m.sum()),
                    "mean_pred": round(float(p[m].mean()), 4) if m.any() else None,
                    "observed": round(float(y[m].mean()), 4) if m.any() else None})
    return out


def alarm_counts(y: np.ndarray, p: np.ndarray, threshold: float = ALARM_THRESHOLD) -> dict:
    alarm = p >= threshold
    tp = int((alarm & (y == 1)).sum()); fp = int((alarm & (y == 0)).sum())
    fn = int((~alarm & (y == 1)).sum())
    return {"threshold": threshold, "alarm_hours": int(alarm.sum()), "tp": tp, "fp": fp,
            "fn": fn,
            "precision": round(tp / (tp + fp), 4) if tp + fp else None,
            "recall": round(tp / (tp + fn), 4) if tp + fn else None}


def status_of(p: np.ndarray) -> np.ndarray:
    """Same cuts as models/decision_rules.dashboard_status, imported not copied."""
    s = np.full(len(p), "normal", dtype=object)
    s[p >= ELEVATED_THRESHOLD] = "elevated"
    s[p >= CRITICAL_THRESHOLD] = "critical"
    return s


def operating_point(y: np.ndarray, p: np.ndarray, target_precision: float = 0.5) -> dict:
    precision, recall, thresholds = precision_recall_curve(y, p)
    if len(thresholds) == 0:
        return {}
    idx = int(np.argmin(np.abs(precision[:-1] - target_precision)))
    return {"target_precision": target_precision,
            "precision": round(float(precision[idx]), 4),
            "recall": round(float(recall[idx]), 4),
            "threshold": round(float(thresholds[idx]), 4)}


def score(model, df: pd.DataFrame, features: list[str]) -> np.ndarray:
    return np.asarray(model.predict_proba(df[features])[:, 1], dtype=float)


def blind_tail_slices(df: pd.DataFrame, log):
    """
    The tail the incumbent cannot see at all, per zone: everything past the last hour
    whose weather features are complete. On the committed frame those hours carry no
    label either (labels need capacity, which exists), so the true label is recomputed
    from load/capacity with the shipped config threshold rather than dropped.
    """
    from features.build_features import add_overload_proxy_label

    tails = []
    for zone, g in df.groupby("zone"):
        g = g.sort_values("timestamp")
        weather_ok = g[FEATURE_COLS].notna().all(axis=1)
        if not weather_ok.any():
            continue
        frontier = g.loc[weather_ok, "timestamp"].max()
        tail = g[g["timestamp"] > frontier].copy()
        if len(tail) == 0:
            continue
        tail = tail.dropna(subset=FEATURE_COLS_NO_WEATHER + ["load_mw", "capacity_mw"])
        if len(tail) == 0:
            continue
        tail = add_overload_proxy_label(tail)   # deterministic recompute of the label
        tails.append(tail)
        log(f"  blind tail {zone}: {len(tail)} hours past {frontier.isoformat()}, "
            f"positives={int(tail[TARGET_COL].sum())}")
    if not tails:
        return None
    out = pd.concat(tails, ignore_index=True)
    return out


def agreement_block(y: np.ndarray, p_inc: np.ndarray, p_fb: np.ndarray) -> dict:
    """The number the ship decision turns on: do the two scores mean the same thing?"""
    diff = p_fb - p_inc
    s_inc, s_fb = status_of(p_inc), status_of(p_fb)
    inc_elev = p_inc >= ELEVATED_THRESHOLD
    fb_elev = p_fb >= ELEVATED_THRESHOLD
    inc_crit = p_inc >= CRITICAL_THRESHOLD
    fb_crit = p_fb >= CRITICAL_THRESHOLD
    spearman = float(pd.Series(p_inc).corr(pd.Series(p_fb), method="spearman"))
    kendall_tau_b = float(pd.Series(p_inc).corr(pd.Series(p_fb), method="kendall"))
    return {
        "hours_compared": int(len(y)),
        "positive_rate": round(float(y.mean()), 4),
        "pr_auc_incumbent": round(float(average_precision_score(y, p_inc)), 4),
        "pr_auc_fallback": round(float(average_precision_score(y, p_fb)), 4),
        "mean_score_incumbent": round(float(p_inc.mean()), 4),
        "mean_score_fallback": round(float(p_fb.mean()), 4),
        "mean_signed_shift": round(float(diff.mean()), 4),
        "median_abs_shift": round(float(np.median(np.abs(diff))), 4),
        "p90_abs_shift": round(float(np.percentile(np.abs(diff), 90)), 4),
        "max_abs_shift": round(float(np.abs(diff).max()), 4),
        "spearman_rank_correlation": round(spearman, 4),
        "kendall_tau_b": round(kendall_tau_b, 4),
        "status_flip_rate": round(float((s_inc != s_fb).mean()), 4),
        "elevated_disagreement_rate": round(float((inc_elev != fb_elev).mean()), 4),
        "critical_disagreement_rate": round(float((inc_crit != fb_crit).mean()), 4),
        "alarm_precision_incumbent": alarm_counts(y, p_inc)["precision"],
        "alarm_precision_fallback": alarm_counts(y, p_fb)["precision"],
        "alarm_recall_incumbent": alarm_counts(y, p_inc)["recall"],
        "alarm_recall_fallback": alarm_counts(y, p_fb)["recall"],
    }


def decide(agreement: dict, fallback_test: dict, incumbent_test: dict,
           calib: dict, gates: dict) -> dict:
    checks = [
        ("pr_auc >= min", fallback_test["pr_auc"] >= gates["min_pr_auc"],
         f"{fallback_test['pr_auc']:.4f} >= {gates['min_pr_auc']}"),
        (f"PR-AUC drop vs incumbent <= {gates['max_pr_auc_drop_pp']}pp",
         (incumbent_test["pr_auc"] - fallback_test["pr_auc"]) * 100
         <= gates["max_pr_auc_drop_pp"],
         f"{incumbent_test['pr_auc']:.4f} -> {fallback_test['pr_auc']:.4f}"),
        ("mean score shift <= max", abs(agreement["mean_signed_shift"]) <= gates["max_mean_shift"],
         f"{agreement['mean_signed_shift']:+.4f} <= {gates['max_mean_shift']}"),
        ("status flip rate <= max",
         agreement["status_flip_rate"] <= gates["max_status_flip_rate"],
         f"{agreement['status_flip_rate']:.4f} <= {gates['max_status_flip_rate']}"),
        ("critical flip rate <= max",
         agreement["critical_disagreement_rate"] <= gates["max_critical_flip_rate"],
         f"{agreement['critical_disagreement_rate']:.4f} <= {gates['max_critical_flip_rate']}"),
        ("alarm recall loss <= max",
         (incumbent_test["alarm"]["recall"] or 0) - (fallback_test["alarm"]["recall"] or 0)
         <= gates["max_recall_loss_pp"] / 100.0,
         f"{incumbent_test['alarm']['recall']} -> {fallback_test['alarm']['recall']}"),
        ("ECE <= max", calib["ece"] <= gates["max_ece"], f"{calib['ece']:.4f} <= {gates['max_ece']}"),
    ]
    failed = [name for name, ok, _ in checks if not ok]
    return {"ship": not failed, "checks": [{"check": n, "ok": bool(o), "value": v}
                                           for n, o, v in checks],
            "failed": failed}


def bucketed_top_factors(model, features: list[str], probs: np.ndarray, log) -> dict:
    """
    Gain-based factor lookup in the same JSON shape as risk_factors_by_bucket.json.
    SHAP is deliberately not used here: it is a training-only dependency this run may
    not have installed, and the shipped explanations are already bucket-level, not
    per-request. Recorded as gain-derived so nobody mistakes it for SHAP output.
    """
    booster = model.booster_
    gains = dict(zip(features, np.asarray(booster.feature_importance("gain"), dtype=float)))
    total = sum(gains.values()) or 1.0
    ranked = sorted(gains.items(), key=lambda kv: -kv[1])[:3]
    out = {}
    for label, lo, hi in RISK_BUCKETS:
        m = (probs >= lo) & (probs < hi)
        log(f"  bucket '{label}': n={int(m.sum())}")
        out[label] = [{"feature": f, "contribution": round(g / total, 4),
                       "basis": "global_gain_share"} for f, g in ranked]
    return out


def run(features_path: Path, out_dir: Path, incumbent_path: Path, gates: dict,
        log=print) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    log(f"[risk_no_weather] features : {features_path}")
    log(f"[risk_no_weather] out-dir  : {out_dir}  (nothing under data/ is written)")

    df = load_frame(features_path)
    train_full, test_df = walk_forward_split(df)
    fit_df, val_df = train_val_split(train_full)
    log(f"[risk_no_weather] split    : fit {fit_df.timestamp.min()}..{fit_df.timestamp.max()} "
        f"({len(fit_df)}) | val ..{val_df.timestamp.max()} ({len(val_df)}) | "
        f"test {test_df.timestamp.min()}..{test_df.timestamp.max()} ({len(test_df)})")

    def scorable(d, feats):
        return d.dropna(subset=feats + [TARGET_COL])

    fb_fit, fb_val, fb_test = (scorable(fit_df, FEATURE_COLS_NO_WEATHER),
                               scorable(val_df, FEATURE_COLS_NO_WEATHER),
                               scorable(test_df, FEATURE_COLS_NO_WEATHER))
    inc_test = scorable(test_df, FEATURE_COLS)
    log(f"[risk_no_weather] rows     : fallback fit={len(fb_fit)} val={len(fb_val)} "
        f"test={len(fb_test)} | incumbent test={len(inc_test)} "
        f"| incumbent cannot score {len(fb_test) - len(inc_test)} of the fallback's test hours")

    log("[risk_no_weather] training weather-blind classifier (is_unbalance=True, no SMOTE)")
    model = fit_lgbm(FEATURE_COLS_NO_WEATHER, fb_fit, fb_val, log)

    incumbent = joblib.load(incumbent_path)
    y_fb = fb_test[TARGET_COL].to_numpy()
    p_fb = score(model, fb_test, FEATURE_COLS_NO_WEATHER)
    y_inc = inc_test[TARGET_COL].to_numpy()
    p_inc = score(incumbent, inc_test, FEATURE_COLS)

    fallback_test = {
        "n_hours": int(len(fb_test)), "positive_rate": round(float(y_fb.mean()), 4),
        "pr_auc": round(float(average_precision_score(y_fb, p_fb)), 4),
        "roc_auc": round(float(roc_auc_score(y_fb, p_fb)), 4),
        "brier": round(float(brier_score_loss(y_fb, p_fb)), 4),
        "brier_base_rate_reference": round(float(
            brier_score_loss(y_fb, np.full_like(p_fb, float(y_fb.mean())))), 4),
        "alarm": alarm_counts(y_fb, p_fb),
        "operating_point": operating_point(y_fb, p_fb),
        "reliability": reliability_rows(y_fb, p_fb),
        "ece": round(expected_calibration_error(y_fb, p_fb), 4),
    }
    incumbent_test = {
        "n_hours": int(len(inc_test)), "positive_rate": round(float(y_inc.mean()), 4),
        "pr_auc": round(float(average_precision_score(y_inc, p_inc)), 4),
        "brier": round(float(brier_score_loss(y_inc, p_inc)), 4),
        "alarm": alarm_counts(y_inc, p_inc),
        "ece": round(expected_calibration_error(y_inc, p_inc), 4),
    }
    log(f"[risk_no_weather] TEST(weather-blind ) PR-AUC={fallback_test['pr_auc']:.4f} "
        f"Brier={fallback_test['brier']:.4f} alarm={fallback_test['alarm']}")
    log(f"[risk_no_weather] TEST(incumbent     ) PR-AUC={incumbent_test['pr_auc']:.4f} "
        f"Brier={incumbent_test['brier']:.4f} alarm={incumbent_test['alarm']}")

    # ---- agreement on the hours BOTH models can score ----
    both_idx = inc_test.index.intersection(fb_test.index)
    inc_b, fb_b = inc_test.loc[both_idx], fb_test.loc[both_idx]
    y_b = inc_b[TARGET_COL].to_numpy()
    agreement = agreement_block(y_b, score(incumbent, inc_b, FEATURE_COLS),
                                score(model, fb_b, FEATURE_COLS_NO_WEATHER))
    log(f"[risk_no_weather] AGREEMENT on {agreement['hours_compared']} jointly-scorable hours:")
    for k in ("mean_score_incumbent", "mean_score_fallback", "mean_signed_shift",
              "median_abs_shift", "p90_abs_shift", "spearman_rank_correlation",
              "status_flip_rate", "elevated_disagreement_rate", "critical_disagreement_rate"):
        log(f"                 {k} = {agreement[k]}")

    # ---- leakage control: does the reduced model actually reach the blind tail? ----
    tail = blind_tail_slices(df, log)
    tail_probe = {}
    if tail is not None and len(tail):
        t = tail.dropna(subset=FEATURE_COLS_NO_WEATHER)
        p_tail = score(model, t, FEATURE_COLS_NO_WEATHER)
        tail_probe = {
            "hours": int(len(t)),
            "scored_fraction_incumbent": 0.0,  # structurally: dropna leaves zero rows
            "scored_fraction_fallback": round(float(np.isfinite(p_tail).mean()), 4),
            "mean_score": round(float(p_tail.mean()), 4),
            "alarm_hours_at_operating_point": int((p_tail >= ALARM_THRESHOLD).sum()),
        }
        log(f"[risk_no_weather] BLIND TAIL probe: {tail_probe}")

    gates_report = decide(agreement, fallback_test, incumbent_test,
                          {"ece": fallback_test["ece"]}, gates)
    metrics = {
        "generated_from": {"features": str(features_path), "incumbent": str(incumbent_path)},
        "feature_cols_fallback": FEATURE_COLS_NO_WEATHER,
        "feature_cols_incumbent": FEATURE_COLS,
        "split": {"method": "walk_forward_split (temporal, never shuffled)",
                  "test_start": str(test_df.timestamp.min()),
                  "test_end": str(test_df.timestamp.max()),
                  "val_used_for_early_stopping": True},
        "imbalance": {"method": "is_unbalance=True (LightGBM weighting, not SMOTE)"},
        "gates": gates,
        "test_fallback": fallback_test,
        "test_incumbent": incumbent_test,
        "agreement": agreement,
        "blind_tail_probe": tail_probe,
        "verdict": gates_report,
    }

    if gates_report["ship"]:
        path = out_dir / MODEL_FILE
        joblib.dump(model, path)
        factors = bucketed_top_factors(model, FEATURE_COLS_NO_WEATHER, p_fb, log)
        (out_dir / FACTORS_FILE).write_text(json.dumps(factors, indent=2))
        metrics["artifact"] = {"model": str(path), "factors": str(out_dir / FACTORS_FILE)}
        log(f"[risk_no_weather] SHIP-GATE PASS — artifact written to {path}")
    else:
        log(f"[risk_no_weather] SHIP-GATE FAIL — no artifact written. Failed: "
            f"{', '.join(gates_report['failed'])}")
    (out_dir / METRICS_FILE).write_text(json.dumps(metrics, indent=2))
    log(f"[risk_no_weather] metrics -> {out_dir / METRICS_FILE}")
    log(f"[risk_no_weather] VERDICT: {'SHIP' if gates_report['ship'] else 'DO NOT SHIP'}")
    return metrics


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--features", type=Path, default=DATA_FEATURES_DIR / "training_features.parquet")
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    ap.add_argument("--incumbent", type=Path, default=MODELS_DIR / "risk_classifier.joblib")
    ap.add_argument("--min-pr-auc", type=float, default=MIN_PR_AUC)
    ap.add_argument("--max-pr-auc-drop-pp", type=float, default=MAX_PR_AUC_DROP_PP)
    ap.add_argument("--max-mean-shift", type=float, default=MAX_MEAN_SHIFT)
    ap.add_argument("--max-status-flip-rate", type=float, default=MAX_STATUS_FLIP_RATE)
    ap.add_argument("--max-critical-flip-rate", type=float, default=MAX_CRITICAL_FLIP_RATE)
    ap.add_argument("--max-recall-loss-pp", type=float, default=MAX_RECALL_LOSS_PP)
    ap.add_argument("--max-ece", type=float, default=MAX_ECE)
    args = ap.parse_args(argv)

    for protected in (MODELS_DIR, DATA_FEATURES_DIR):
        if _same_or_child(args.out_dir.resolve(), protected.resolve()):
            raise SystemExit(f"[risk_no_weather] refusing to write into the served artifact "
                             f"directory {protected}; pass a scratch --out-dir")

    gates = {"min_pr_auc": args.min_pr_auc, "max_pr_auc_drop_pp": args.max_pr_auc_drop_pp,
             "max_mean_shift": args.max_mean_shift,
             "max_status_flip_rate": args.max_status_flip_rate,
             "max_critical_flip_rate": args.max_critical_flip_rate,
             "max_recall_loss_pp": args.max_recall_loss_pp, "max_ece": args.max_ece}
    metrics = run(args.features, args.out_dir, args.incumbent, gates)
    return 0 if metrics["verdict"]["ship"] else 1


def _same_or_child(path: Path, root: Path) -> bool:
    try:
        Path(path).relative_to(root)
        return True
    except ValueError:
        return False


if __name__ == "__main__":
    raise SystemExit(main())
