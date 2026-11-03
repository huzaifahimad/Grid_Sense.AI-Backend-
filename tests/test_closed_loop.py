"""
Closed-loop validation: the replay's own arithmetic, and proof that its release
gates reject a broken model.

The negative controls matter more than the passing ones. A validator that
reports PASS no matter what is worse than no validator, because it launders a
bad model into a "validated" badge on the dashboard.
"""

import json

import joblib
import numpy as np
import pytest

from config import MODELS_DIR
from models.closed_loop_validation import (
    GATES,
    _last_true,
    _prefix_true,
    _runs,
    grade,
    run,
)
from models.decision_rules import (
    ALARM_THRESHOLD,
    CRITICAL_THRESHOLD,
    ELEVATED_THRESHOLD,
    MODERATE_THRESHOLD,
    dashboard_status,
    shed_priority,
)

REPORT_PATH = MODELS_DIR / "closed_loop_report.json"


@pytest.fixture(scope="module")
def report():
    if not REPORT_PATH.exists():
        pytest.skip("closed-loop report not built; run python -m models.closed_loop_validation")
    return json.loads(REPORT_PATH.read_text())


class InflatedForecast:
    """Multiplies every prediction by 1.15 — moves the whole P10-P90 band with it."""

    def __init__(self, inner, factor=1.15):
        self.inner = inner
        self.factor = factor

    def predict(self, X):
        return np.asarray(self.inner.predict(X), dtype=float) * self.factor


class AlwaysShedRisk:
    """Says "shed" every hour, forever."""

    def predict_proba(self, X):
        n = len(X)
        return np.column_stack([np.zeros(n), np.ones(n)])


# ---------------------------------------------------------------- gate helpers

def test_grade_directions():
    assert grade(0.9, "high", 0.5, 0.35) == "PASS"
    assert grade(0.4, "high", 0.5, 0.35) == "WARN"
    assert grade(0.1, "high", 0.5, 0.35) == "FAIL"
    assert grade(1.0, "low", 5.0, 10.0) == "PASS"
    assert grade(7.0, "low", 5.0, 10.0) == "WARN"
    assert grade(11.0, "low", 5.0, 10.0) == "FAIL"
    assert grade(0.8, "between", [0.7, 0.95], [0.6, 0.98]) == "PASS"
    assert grade(0.65, "between", [0.7, 0.95], [0.6, 0.98]) == "WARN"
    assert grade(0.99, "between", [0.7, 0.95], [0.6, 0.98]) == "FAIL"


def test_grade_rejects_missing_values():
    assert grade(None, "high", 0.5, 0.35) == "FAIL"
    assert grade(float("nan"), "high", 0.5, 0.35) == "FAIL"


def test_prefix_true_requires_an_unbroken_warning_chain():
    assert _prefix_true([True, True, False, True]) == 2
    assert _prefix_true([False, True, True]) == 0
    assert _prefix_true([True, True]) == 2
    # A lead whose origin was not replayed is absence of evidence, not silence.
    assert _prefix_true([None, True, None, True, False]) == 4


def test_last_true_is_the_optimistic_glimpse():
    assert _last_true([True, False, False, True]) == 4
    assert _last_true([None, False, None]) == 0
    assert _last_true([False, False]) == 0


def test_runs_splits_on_missing_hours_not_just_labels():
    assert _runs(np.array([True, True, True]), np.array([0, 1, 5])) == [(0, 1), (2, 2)]
    assert _runs(np.array([True, False, True]), np.array([0, 1, 2])) == [(0, 0), (2, 2)]
    assert _runs(np.array([False, False]), np.array([0, 1])) == []


def test_decision_rules_match_the_alarm_definition():
    """The replay's "alarm" must mean exactly "the API would recommend shedding"."""
    for score in np.linspace(0, 1, 41):
        assert (score >= ALARM_THRESHOLD) == (shed_priority(score) is not None)
    assert shed_priority(ELEVATED_THRESHOLD) == 2
    assert shed_priority(CRITICAL_THRESHOLD) == 1
    assert shed_priority(MODERATE_THRESHOLD) is None
    assert dashboard_status(CRITICAL_THRESHOLD) == "critical"
    assert dashboard_status(ELEVATED_THRESHOLD) == "elevated"
    assert dashboard_status(CRITICAL_THRESHOLD - 0.01) == "elevated"
    assert dashboard_status(ELEVATED_THRESHOLD - 0.01) == "normal"
    assert dashboard_status(MODERATE_THRESHOLD) == "normal"


# -------------------------------------------------------------------- report

def test_shipped_report_has_no_failing_gate(report):
    failing = [g["id"] for g in report["gates"] if g["result"] == "FAIL"]
    assert failing == [], f"failing gates: {failing}"
    assert report["overall"] in ("PASS", "WARN")


def test_report_covers_a_real_held_out_window(report):
    w = report["window"]
    assert w["hours"] > 720, "replayed window is too short to conclude anything"
    assert w["rows_scored"] >= w["forecast_origins_replayed"]
    assert w["forecast_origins_replayed"] >= 500
    assert set(w["zones"]) == {"ES", "GR", "IT_SOUTH"}
    assert w["alarm_threshold"] == ALARM_THRESHOLD
    assert sum(w["hourly_gaps_per_zone"].values()) == 0, "gaps would fake contiguous event runs"


def test_forecast_beats_the_baseline_at_every_step(report):
    ps = report["metrics"]["forecast"]["per_step"]
    assert ps, "no per-step forecast metrics"
    for h, m in ps.items():
        assert m["mape_recursive"] < m["mape_seasonal_naive"], f"h+{h} loses to seasonal-naive"
        assert 0 < m["coverage_p10_p90"] <= 1
        assert abs(m["bias_pct"]) < 5, f"h+{h} is systematically biased"


def test_band_coverage_is_close_to_nominal(report):
    assert 0.70 <= report["metrics"]["forecast"]["coverage_p10_p90_overall"] <= 0.95


def test_alarm_counts_are_self_consistent(report):
    """Checks the validator's own arithmetic, not just the model's quality."""
    a = report["metrics"]["alarm"]
    tp, fp, fn = a["true_positive_hours"], a["false_alarm_hours"], a["missed_hours"]
    assert tp + fp == a["alarm_hours"]
    assert round(tp / (tp + fp), 4) == a["precision"]
    assert round(tp / (tp + fn), 4) == a["recall"]
    assert tp + fn == round(a["positive_rate"] * a["hours_scored"])
    assert a["brier"] < a["brier_base_rate"], "worse than always predicting the base rate"


def test_reliability_is_monotone_across_buckets(report):
    observed = [b["observed_overload_rate"] for b in report["metrics"]["alarm"]["reliability"]]
    assert observed == sorted(observed), "higher predicted risk must mean higher observed overload"


def test_event_metrics_are_internally_consistent(report):
    e = report["metrics"]["events"]
    assert e["n_events"] > 0
    assert round(e["n_detected"] / e["n_events"], 4) == e["detection_recall"]
    assert e["median_lead_hours_stable"] <= e["median_lead_hours_glimpse"], (
        "a stable warning chain cannot exceed the optimistic first glimpse")
    assert e["share_warned_3h"] <= e["share_warned_1h"]


def test_overall_field_is_derived_from_the_gates(report):
    results = {g["result"] for g in report["gates"]}
    assert {g["id"] for g in report["gates"]} == {g["id"] for g in GATES}
    expected = "FAIL" if "FAIL" in results else ("WARN" if "WARN" in results else "PASS")
    assert report["overall"] == expected


# ------------------------------------------------------------ negative controls

def test_negative_control_inflated_forecast_fails_the_gates():
    """
    Scaling every prediction up by 15% moves the calibrated band with it, so the
    band stops containing the truth. If the gates still say PASS here, they are
    decoration.
    """
    broken = {"forecast_model": InflatedForecast(
        joblib.load(MODELS_DIR / "lightgbm_load_forecast.joblib"))}
    result = run(stride=24, horizon=24, write=False, models=broken)

    coverage = next(g for g in result["gates"] if g["id"] == "band_coverage")
    assert coverage["result"] == "FAIL", f"band gate ignored a 15% shift: {coverage}"
    assert result["metrics"]["forecast"]["coverage_p10_p90_overall"] < 0.5
    assert result["overall"] == "FAIL"


def test_negative_control_always_shed_model_fails_the_gates():
    """
    A risk model that alarms every hour has recall 1.0 — exactly the number a
    naive demo would show off. Precision and alert burden must catch it.
    """
    result = run(stride=24, horizon=24, write=False, models={"risk_model": AlwaysShedRisk()})

    by_id = {g["id"]: g for g in result["gates"]}
    assert by_id["alarm_precision"]["result"] == "FAIL"
    assert by_id["false_alarms_per_zone_day"]["result"] == "FAIL"
    assert by_id["alarm_recall"]["result"] == "PASS", "recall alone rewards alarm fatigue"
    assert result["overall"] == "FAIL"


def test_negative_controls_do_not_overwrite_the_shipped_artifact(report):
    assert json.loads(REPORT_PATH.read_text())["overall"] == report["overall"]
    assert REPORT_PATH.stat().st_size > 1000


# ------------------------------------------------------------------- endpoint

def test_validation_endpoint_serves_the_report(client):
    resp = client.get("/validation")
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["overall"] in ("PASS", "WARN", "FAIL")
    assert body["staleness_hours"] >= 0
    assert len(body["gates"]) == len(GATES)
    assert set(body["metrics"]) == {"forecast", "alarm", "events"}
    assert body["limitations"], "an unqualified 'validated' claim is not acceptable"
    assert body["window"]["alarm_threshold"] == ALARM_THRESHOLD


def test_health_reports_the_closed_loop_artifact(client):
    body = client.get("/health").json()
    assert body["artifacts"]["closed_loop_report"] is True
    assert body["status"] == "ok"
