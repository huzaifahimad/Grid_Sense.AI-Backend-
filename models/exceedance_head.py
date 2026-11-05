"""
GridSense AI — Second-stage overload-exceedance head.

Why this exists, in one measured sentence: the point forecast is accurate (MAPE of a
few percent) but the warning line sits at 0.9 * capacity and more than half of the
overload hours clear it by less than 5%, so thresholding p50 cannot warn on the
marginal events at all — which is exactly the gate that fails on real data.

The fix is not a lower threshold and not a wider band. It is a different question.
"P(load > line)" is estimable even when "P(load)" is not sharp, because it only needs
the probability mass on the far side of the line, and the forecaster's own spread is
information about that mass. So this module trains a classifier on the *outputs of the
serving forecaster* over the validation window and scores it once on the test window:

    X = what the API already has at an origin (p50 and p90 at lead h, the zone's
        capacity, the load most recently observed, how close recent hours ran)
    y = did load at origin + h actually exceed OVERLOAD_THRESHOLD_RATIO * capacity

Nothing in X is measured after the origin, so it cannot leak: features come from the
forecast the API would have had in hand, plus history ending at the origin.

It ships in serving only if it beats the existing p50 rule on the test window at
matched alert volume. If it does not, it stays here as a measured negative — which is
a result, not a failure.

Usage (after train_baselines + calibrate_forecast_intervals):
    python -m models.exceedance_head
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

import joblib
import numpy as np
import pandas as pd

from config import DATA_FEATURES_DIR, MODELS_DIR, OVERLOAD_THRESHOLD_RATIO
from models.calibrate_forecast_intervals import HORIZON, build_origins
from models.forecasting import WEATHER_COLS, recursive_forecast_batch
from models.train_baselines import FEATURE_COLS, train_val_test_split

LEADS = [1, 6, 12, 24]
MODEL_PATH = MODELS_DIR / "exceedance_head.joblib"
METRICS_PATH = MODELS_DIR / "exceedance_metrics.json"


def _capacity_by_zone(df: pd.DataFrame) -> dict:
    return {z: float(g["capacity_mw"].max()) for z, g in df.groupby("zone")}


def stage_features(pred: np.ndarray, band_hi: np.ndarray, origin_ts: np.ndarray,
                   capacity: np.ndarray, last_load: np.ndarray,
                   recent_max_ratio: np.ndarray, lead: int) -> pd.DataFrame:
    """
    One row per (origin, lead). Every column is computable at the origin from data the
    API already serves — no look-ahead, no re-derivation of the recursion.
    """
    ts = pd.DatetimeIndex(origin_ts) + pd.Timedelta(hours=lead)
    return pd.DataFrame({
        "lead_hours": lead,
        "p50_over_capacity": pred / capacity,
        "p90_over_capacity": band_hi / capacity,
        # The head has to know how much the point estimate is worth at this lead; the
        # calibrated band is the system's own measured statement of that.
        "band_width": (band_hi - pred) / np.maximum(pred, 1.0),
        "origin_load_over_capacity": last_load / capacity,
        "recent_peak_over_capacity": recent_max_ratio,
        "target_hour_sin": np.sin(2 * np.pi * ts.hour / 24),
        "target_hour_cos": np.cos(2 * np.pi * ts.hour / 24),
        "target_dow_sin": np.sin(2 * np.pi * ts.dayofweek / 7),
        "target_dow_cos": np.cos(2 * np.pi * ts.dayofweek / 7),
    })


FEATURE_NAMES = list(stage_features(
    np.empty(0), np.empty(0), pd.DatetimeIndex([]),
    np.empty(0), np.empty(0), np.empty(0), 1).columns)


def build_dataset(df: pd.DataFrame, model, intervals: dict, start, end,
                  stride: int) -> tuple[pd.DataFrame, pd.Series, np.ndarray]:
    """Origins across [start, end] with the serving forecaster's own outputs at each
    lead, labelled by what actually happened. Returns X, y, and the origin id behind
    every row; both the incumbent rule and the head are scored on this exact matrix, so
    the comparison is on identical hours and leads."""
    q10, q90 = _q_tables(intervals)
    frames, labels, origin_ids = [], [], []
    capacity = _capacity_by_zone(df)
    next_id = 0

    for zone, g in df.groupby("zone"):
        hist_l, hist_w, origin_ts, actual, _seasonal = build_origins(g, start, end, stride)
        if not len(origin_ts):
            continue
        preds = recursive_forecast_batch(model, hist_l, hist_w, origin_ts, HORIZON, FEATURE_COLS)
        cap = capacity[zone]
        line = OVERLOAD_THRESHOLD_RATIO * cap

        # Same filter build_origins applies internally, so the origin timestamps must be
        # findable here. If they ever diverge, that is a silent mislabel — fail on it.
        gs = g.dropna(subset=["load_mw", *WEATHER_COLS]).sort_values("timestamp").reset_index(drop=True)
        loads = gs["load_mw"].to_numpy(float)
        pos = {t: i for i, t in enumerate(gs["timestamp"])}
        if any(t not in pos for t in origin_ts):
            raise RuntimeError(f"zone {zone}: origin timestamp not in the labelled frame")
        last_load = np.array([loads[pos[t]] for t in origin_ts])
        recent_max = np.array([loads[max(0, pos[t] - 23):pos[t] + 1].max() / cap for t in origin_ts])

        for lead in LEADS:
            # actual[:, k] is the outcome k+1 hours after the origin
            hi = preds[:, lead - 1] * (1 + q90[lead - 1])
            y = actual[:, lead - 1] > line
            frames.append(stage_features(preds[:, lead - 1], hi, np.asarray(origin_ts),
                                         np.full(len(origin_ts), cap), last_load,
                                         recent_max, lead))
            labels.append(pd.Series(y.astype(int)))
            origin_ids.append(next_id + np.arange(len(origin_ts)))
        next_id += len(origin_ts)

    X = pd.concat(frames, ignore_index=True)
    y = pd.concat(labels, ignore_index=True)
    return X, y, np.concatenate(origin_ids)


def _q_tables(intervals: dict) -> tuple[np.ndarray, np.ndarray]:
    table = intervals["horizons"]
    q10 = np.array([float(table[str(h)]["q10"]) for h in range(1, HORIZON + 1)])
    q90 = np.array([float(table[str(h)]["q90"]) for h in range(1, HORIZON + 1)])
    return q10, q90


def precision_recall(y_true: np.ndarray, y_flag: np.ndarray) -> tuple[float, float, float]:
    """(precision, recall, alerts_per_1000_hours). Guarded against empty denominators."""
    flagged = int(y_flag.sum())
    hits = int(np.sum(y_flag & y_true))
    positives = int(y_true.sum())
    precision = hits / flagged if flagged else 0.0
    recall = hits / positives if positives else 0.0
    return precision, recall, 1000.0 * flagged / max(len(y_flag), 1)


def main() -> int:
    df = pd.read_parquet(DATA_FEATURES_DIR / "training_features.parquet")
    model = joblib.load(MODELS_DIR / "lightgbm_load_forecast.joblib")
    intervals = json.loads((MODELS_DIR / "forecast_intervals.json").read_text(encoding="utf-8"))
    _, val, test = train_val_test_split(df)

    print("[exceedance] building validation-window dataset")
    X_val, y_val, val_origin = build_dataset(
        df, model, intervals, val["timestamp"].min(), val["timestamp"].max(), stride=3)
    print(f"[exceedance] {len(X_val)} origin-lead pairs, base rate {y_val.mean():.3f}")

    from lightgbm import LGBMClassifier, early_stopping  # noqa: E402 — training-only dep

    # Hold out 20% of the *origins*. One origin contributes all four leads, so a
    # row-level cut would put the same origin's leads on both sides of the boundary and
    # let the iteration count be chosen on data the fit already saw.
    rng = np.random.default_rng(7)
    unique_origins = np.unique(val_origin)
    held_out = rng.choice(unique_origins, size=int(0.2 * len(unique_origins)), replace=False)
    fit_mask = ~np.isin(val_origin, held_out)
    if fit_mask.all() or not fit_mask.any():
        raise RuntimeError("origin-level split degenerated — check build_dataset ids")
    clf = LGBMClassifier(
        n_estimators=400, learning_rate=0.05, num_leaves=31,
        subsample=0.8, colsample_bytree=0.8, random_state=42)
    clf.fit(
        X_val.loc[fit_mask, FEATURE_NAMES], y_val[fit_mask],
        eval_set=[(X_val.loc[~fit_mask, FEATURE_NAMES], y_val[~fit_mask])],
        callbacks=[early_stopping(30, verbose=False)],
    )

    print("[exceedance] scoring once on the test window")
    X_test, y_test, _ = build_dataset(
        df, model, intervals, test["timestamp"].min(), test["timestamp"].max(), stride=3)
    prob = clf.predict_proba(X_test[FEATURE_NAMES])[:, 1]

    # The incumbent rule: warn when the point forecast itself is over the line.
    p50_over = X_test["p50_over_capacity"].to_numpy() >= OVERLOAD_THRESHOLD_RATIO
    incumbent_p, incumbent_r, incumbent_rate = precision_recall(
        y_test.to_numpy().astype(bool), p50_over)

    # Compare at exactly matched volume: the head gets the same number of alerts the
    # incumbent issued, chosen by descending probability. A recall gain here cannot have
    # been bought by paging more often.
    k = int(p50_over.sum())
    y_bool = y_test.to_numpy().astype(bool)
    if k == 0:
        raise RuntimeError("incumbent rule issued no alerts on the test window — there is "
                           "no alert volume to match, so no honest comparison exists")
    head_flag = np.zeros(len(prob), dtype=bool)
    head_flag[np.argsort(-prob, kind="stable")[:k]] = True
    head_p, head_r, head_rate = precision_recall(y_bool, head_flag)

    metrics = {
        "leads": LEADS,
        "rows": {"fit": int(fit_mask.sum()), "eval": int((~fit_mask).sum()),
                 "test": int(len(X_test))},
        "base_rate_test": round(float(y_test.mean()), 4),
        "incumbent_p50_rule": {"precision": round(incumbent_p, 4), "recall": round(incumbent_r, 4),
                               "alerts_per_1000_origin_leads": round(incumbent_rate, 2)},
        "exceedance_head": {"alerts_matched_to_incumbent": k,
                            "precision": round(head_p, 4), "recall": round(head_r, 4),
                            "alerts_per_1000_origin_leads": round(head_rate, 2)},
        "verdict": "BEATS" if head_r > incumbent_r and head_p >= incumbent_p else "DOES NOT BEAT",
    }
    print(f"[exceedance] incumbent p50 rule : precision {incumbent_p:.3f} recall {incumbent_r:.3f} "
          f"@ {incumbent_rate:.1f} alerts/1000")
    print(f"[exceedance] exceedance head    : precision {head_p:.3f} recall {head_r:.3f} "
          f"@ {head_rate:.1f} alerts/1000  -> {metrics['verdict']}")

    if metrics["verdict"] == "BEATS":
        joblib.dump(clf, MODEL_PATH)
        METRICS_PATH.write_text(json.dumps({**metrics, "feature_names": FEATURE_NAMES}, indent=2))
        print(f"[exceedance] saved -> {MODEL_PATH}")
    else:
        METRICS_PATH.write_text(json.dumps({**metrics, "feature_names": FEATURE_NAMES,
                                           "shipped": False}, indent=2))
        print(f"[exceedance] not shipped: it does not beat the rule at matched alert volume. "
              f"Metrics kept at {METRICS_PATH}")
        return 0
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
