"""
tests/test_risk_no_weather_live.py — the fallback measured against the committed data.

Skipped automatically if the scratch artifact is absent, because it is produced by
`python -m scripts.train_risk_no_weather --out-dir <scratch>` and deliberately never
written under data/models/. Run that first and these become real-data assertions:

  * the hours /risk structurally cannot reach are exactly the ones this model scores;
  * on hours both can score, the two agree closely enough that an operator crossing
    the fallback boundary is not handed a different meaning for the same number;
  * the honest walk-forward metrics beat the base rate by a margin worth serving.

Nothing here touches the network or the served artifacts.
"""

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from config import DATA_FEATURES_DIR, MODELS_DIR
from models.decision_rules import ALARM_THRESHOLD, ELEVATED_THRESHOLD
from models.risk_classifier import (
    FEATURE_COLS, FEATURE_COLS_NO_WEATHER, TARGET_COL, blind_tail_hours, walk_forward_split,
)

SCRATCH = Path(os.environ.get("GRIDSENSE_RISK_NW_SCRATCH",
                              "/tmp/gridsense_risk_no_weather")).expanduser()
MODEL_PATH = SCRATCH / "risk_classifier_no_weather.joblib"
METRICS_PATH = SCRATCH / "risk_classifier_no_weather_metrics.json"

pytestmark = pytest.mark.skipif(
    not (MODEL_PATH.exists() and METRICS_PATH.exists()),
    reason=f"no scratch fallback artifact at {SCRATCH}; run scripts/train_risk_no_weather")


@pytest.fixture(scope="module")
def trained():
    import joblib
    df = pd.read_parquet(DATA_FEATURES_DIR / "training_features.parquet")
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    return {"model": joblib.load(MODEL_PATH),
            "incumbent": joblib.load(MODELS_DIR / "risk_classifier.joblib"),
            "df": df,
            "metrics": json.loads(METRICS_PATH.read_text())}


def test_blind_tail_is_scorable_by_the_fallback_only(trained):
    df = trained["df"]
    _, test_df = walk_forward_split(df)
    inc_ok = test_df[FEATURE_COLS].notna().all(axis=1)
    fb_ok = test_df[FEATURE_COLS_NO_WEATHER].notna().all(axis=1)
    unreachable = int((fb_ok & ~inc_ok).sum())
    assert unreachable == len(test_df) - int(inc_ok.sum()) > 0, \
        "the weather-lag tail must exist in the committed frame, or the premise is gone"
    rows = test_df[fb_ok & ~inc_ok]
    p = trained["model"].predict_proba(rows[FEATURE_COLS_NO_WEATHER])[:, 1]
    assert np.isfinite(p).all() and len(p) == len(rows)


def test_per_zone_blind_tail_matches_the_measured_staleness(trained):
    """90h/zone was the audited figure on the committed parquet; the helper must keep
    reporting it, otherwise the fix is being measured against a different defect."""
    df = trained["df"]
    for zone, g in df.groupby("zone"):
        assert blind_tail_hours(g.sort_values("timestamp")) == pytest.approx(90.0)


def test_scores_agree_with_the_incumbent_where_both_apply(trained):
    m = trained["metrics"]["agreement"]
    assert m["hours_compared"] > 5_000
    # Drift in the mean is what silently moves an operator's status label.
    assert abs(m["mean_signed_shift"]) <= 0.05, m
    assert m["status_flip_rate"] <= 0.10, m
    assert m["critical_disagreement_rate"] <= 0.05, m
    assert m["spearman_rank_correlation"] >= 0.85, m


def test_honest_test_window_beats_the_base_rate(trained):
    m = trained["metrics"]["test_fallback"]
    assert m["pr_auc"] >= 0.85
    assert m["brier"] < trained["metrics"]["test_fallback"]["brier_base_rate_reference"]
    # Not shipped as a calibration layer, so the raw score has to carry its own weight.
    assert m["ece"] <= 0.05
    alarm = m["alarm"]
    assert alarm["precision"] >= 0.7 and alarm["recall"] >= 0.6, alarm


def test_verdict_and_ship_gates_are_recorded_in_the_artifact_metadata(trained):
    m = trained["metrics"]
    assert m["imbalance"]["method"].startswith("is_unbalance")
    assert m["split"]["method"].startswith("walk_forward_split")
    assert m["verdict"]["ship"] is True
    assert all(c["ok"] for c in m["verdict"]["checks"]), m["verdict"]["failed"]


def test_reproducing_the_headline_numbers_from_the_frame_itself(trained):
    """The JSON must not be the only place the numbers live — recompute PR-AUC from the
    committed parquet so a stale metrics file cannot pass the suite."""
    from sklearn.metrics import average_precision_score

    df = trained["df"]
    _, test_df = walk_forward_split(df)
    te = test_df.dropna(subset=FEATURE_COLS_NO_WEATHER + [TARGET_COL])
    p = trained["model"].predict_proba(te[FEATURE_COLS_NO_WEATHER])[:, 1]
    pr_auc = float(average_precision_score(te[TARGET_COL], p))
    assert pr_auc == pytest.approx(trained["metrics"]["test_fallback"]["pr_auc"], abs=1e-4)
