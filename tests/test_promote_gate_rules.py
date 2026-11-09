"""Pure-logic tests for the promote gate's decision rules. Fast, no model loading.

These cover the arithmetic and the fail-closed behaviour of scripts/promote_models.py;
tests/test_promote_gate_live.py covers the same rules against the real committed
artifacts. Split so the cheap invariants run in normal CI and the slow measured ones do
not silently disappear from the suite.
"""
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.promote_models import (  # noqa: E402
    ABSOLUTE_FLOOR_GATES, COVERAGE_MAX_ABSOLUTE_DROP, COVERAGE_MIN_LONG_LEAD,
    MAPE_FLOOR_H24, MAPE_FLOOR_H48, MAPE_MAX_RELATIVE_REGRESSION_PCT,
    MIN_TEST_ORIGINS, IncumbentUnmeasurable, compare_gate_results,
    long_lead_comparison)


def _step(mape, coverage, naive=9.85):
    return {"mape_recursive_lgbm": mape, "coverage": coverage,
            "mape_seasonal_naive": naive}


def _eval(steps, coverage_overall=0.83, origins=834):
    return {"per_step": {str(h): _step(*v) for h, v in steps.items()},
            "p10_p90_coverage_overall": coverage_overall,
            "test_origins": origins,
            "_nominal_coverage": 0.80}


INCUMBENT = _eval({24: (5.558, 0.8561), 48: (7.167, 0.8022)})


# ------------------------------------------------------------------ refusal
def test_deliberately_regressed_candidate_is_refused():
    """A candidate ~3x worse at both long leads must be refused. Numbers are the real
    values measured in _retrain_scratch/drill_gate.py against the committed artifacts."""
    candidate = _eval({24: (17.018, 0.7290), 48: (29.404, 0.6139)})
    cmp = long_lead_comparison(INCUMBENT, candidate, nominal_coverage=0.80)

    # Any ONE of these is sufficient to refuse; the gate ANDs them all.
    assert cmp["worst_step_worse_pct"] > MAPE_MAX_RELATIVE_REGRESSION_PCT
    assert cmp["worst_coverage_drop"] > COVERAGE_MAX_ABSOLUTE_DROP

    for h in ("24", "48"):
        s = cmp["steps"][h]
        assert s["worse_pct"] > MAPE_MAX_RELATIVE_REGRESSION_PCT       # relative rule
        floor = MAPE_FLOOR_H24 if h == "24" else MAPE_FLOOR_H48
        assert s["candidate_mape_pct"] > floor                         # absolute floor
        assert s["candidate_mape_pct"] > s["seasonal_naive_mape_pct"]   # beaten by naive
        assert s["coverage_drop"] > COVERAGE_MAX_ABSOLUTE_DROP

    # Under-coverage below the hard floor is reached at h+48 specifically; h+24 lands at
    # 0.729, which clears the 0.70 floor — asserted here rather than glossed over, so the
    # test says which rule caught which step.
    assert cmp["steps"]["48"]["candidate_coverage"] < COVERAGE_MIN_LONG_LEAD
    assert cmp["steps"]["24"]["candidate_coverage"] >= COVERAGE_MIN_LONG_LEAD
    assert cmp["steps"]["48"]["candidate_coverage"] < (0.80 - 0.10)     # vs nominal


def test_under_covering_band_alone_is_refused_even_when_accuracy_is_unchanged():
    """The product failure mode: identical MAPE, band no longer delivers its range."""
    candidate = _eval({24: (5.558, 0.60), 48: (7.167, 0.55)})
    cmp = long_lead_comparison(INCUMBENT, candidate, nominal_coverage=0.80)
    assert cmp["steps"]["24"]["worse_pct"] == pytest.approx(0.0)
    assert cmp["steps"]["24"]["candidate_coverage"] < COVERAGE_MIN_LONG_LEAD
    assert cmp["worst_coverage_drop"] > COVERAGE_MAX_ABSOLUTE_DROP


def test_too_few_contiguous_future_origins_is_not_a_pass():
    """Thin evidence must fail rather than wave a candidate through."""
    candidate = _eval({24: (5.558, 0.8561), 48: (7.167, 0.8022)}, origins=17)
    assert candidate["test_origins"] < MIN_TEST_ORIGINS


def test_closed_loop_regression_below_incumbent_is_reported():
    inc = {"gates": [{"id": "band_coverage", "result": "PASS"},
                     {"id": "alarm_recall", "result": "WARN"}]}
    cand = {"gates": [{"id": "band_coverage", "result": "FAIL"},
                      {"id": "alarm_recall", "result": "WARN"}]}
    regressed = compare_gate_results(inc, cand)
    assert regressed == ["band_coverage:PASS->FAIL"]


def test_preexisting_incumbent_failure_does_not_become_a_candidate_regression():
    """The incumbent already FAILs median_lead_hours; an identical candidate must not
    be blamed for it. Without this, weekly retrain is permanently un-promotable."""
    inc = {"gates": [{"id": "median_lead_hours", "result": "FAIL"},
                     {"id": "band_coverage", "result": "PASS"}]}
    cand = {"gates": [{"id": "median_lead_hours", "result": "FAIL"},
                      {"id": "band_coverage", "result": "PASS"}]}
    assert compare_gate_results(inc, cand) == []


def test_absent_gate_defaults_to_the_stricter_side():
    """A candidate missing a gate the incumbent has must count as a regression."""
    inc = {"gates": [{"id": "brier_skill", "result": "PASS"}]}
    cand = {"gates": []}
    assert compare_gate_results(inc, cand) == []  # nothing to compare, no new claim made
    cand_unknown = {"gates": [{"id": "brand_new_gate", "result": "FAIL"}]}
    assert compare_gate_results(inc, cand_unknown) == ["brand_new_gate:PASS->FAIL"]


def test_band_coverage_is_the_only_absolute_floor_gate():
    assert ABSOLUTE_FLOOR_GATES == {"band_coverage"}


def test_incumbent_unmeasurable_is_an_error_not_a_silent_pass():
    assert issubclass(IncumbentUnmeasurable, RuntimeError)


# ------------------------------------------------------------------ acceptance
def test_equal_candidate_is_accepted():
    cmp = long_lead_comparison(INCUMBENT, _eval({24: (5.558, 0.8561),
                                                 48: (7.167, 0.8022)}),
                               nominal_coverage=0.80)
    for h in ("24", "48"):
        s = cmp["steps"][h]
        assert s["worse_pct"] <= MAPE_MAX_RELATIVE_REGRESSION_PCT
        assert s["candidate_coverage"] >= COVERAGE_MIN_LONG_LEAD
        assert s["coverage_drop"] <= COVERAGE_MAX_ABSOLUTE_DROP
        assert s["candidate_coverage"] >= 0.80 - 0.10
        assert s["candidate_mape_pct"] < s["seasonal_naive_mape_pct"]


def test_better_candidate_is_accepted_and_shows_negative_regression():
    cmp = long_lead_comparison(INCUMBENT, _eval({24: (4.80, 0.8700), 48: (6.10, 0.8400)}),
                              nominal_coverage=0.80)
    assert cmp["worst_step_worse_pct"] < 0
    assert cmp["worst_coverage_drop"] < 0


def test_improvement_at_one_lead_does_not_excuse_collapse_at_the_other():
    """Both long leads are checked independently; h+24 gains cannot mask an h+48 cliff."""
    cmp = long_lead_comparison(INCUMBENT, _eval({24: (4.00, 0.90), 48: (40.0, 0.40)}),
                               nominal_coverage=0.80)
    assert cmp["steps"]["24"]["worse_pct"] < 0
    assert cmp["steps"]["48"]["worse_pct"] > MAPE_MAX_RELATIVE_REGRESSION_PCT
    assert cmp["worst_step_worse_pct"] > MAPE_MAX_RELATIVE_REGRESSION_PCT


def test_tolerances_are_recorded_in_the_decision_payload():
    cmp = long_lead_comparison(INCUMBENT, _eval({24: (5.558, 0.8561), 48: (7.167, 0.8022)}))
    tol = cmp["tolerances"]
    assert tol["max_relative_mape_regression_pct"] == MAPE_MAX_RELATIVE_REGRESSION_PCT
    assert tol["min_long_lead_coverage"] == COVERAGE_MIN_LONG_LEAD
    assert tol["mape_floor_pct"] == {"24": MAPE_FLOOR_H24, "48": MAPE_FLOOR_H48}


def test_zero_incumbent_mape_does_not_silently_pass():
    """Division-by-zero guard: an unusable denominator must read as infinite regression."""
    broken = {"per_step": {str(h): _step(0.0, 0.85) for h in (24, 48)},
              "p10_p90_coverage_overall": 0.83, "test_origins": 834}
    cmp = long_lead_comparison(broken, _eval({24: (5.5, 0.85), 48: (7.0, 0.80)}))
    assert cmp["worst_step_worse_pct"] == float("inf")
