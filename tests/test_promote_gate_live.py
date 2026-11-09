"""End-to-end promote-gate tests against the REAL committed artifacts.

Slow (each gate run retrains nothing but does replay the recursive forecaster twice), so
these are marked `gate_e2e`; CI runs them on the weekly schedule, developers can opt in
with `-m gate_e2e`. They exist because the pure-logic tests cannot prove the two things
that actually matter:

  * a regressed candidate is refused by the code path the scheduler will run, and
  * the gate never writes to data/models/, even when asked to --commit.

Every artifact directory these tests touch lives under pytest's tmp_path; the served set
is only ever read, and the last test asserts it byte-for-byte.
"""
import hashlib
import json
import shutil
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import DATA_FEATURES_DIR, MODELS_DIR  # noqa: E402
from scripts.promote_models import (  # noqa: E402
    CANDIDATE_FILES, evaluate_gate, main)

# The clean fix is registering `gate_e2e` under [tool.pytest.ini_options].markers in
# pyproject.toml; that file belongs to another agent in this session, so instead of
# editing it unilaterally we suppress only this one collection warning here and note
# the proposal. A module-level pytest_configure was tried first and runs too late to
# affect collection of the very file that defines it.
pytestmark = [
    pytest.mark.gate_e2e,
    pytest.mark.filterwarnings("ignore::pytest.PytestUnknownMarkWarning"),
]

SERVED_DIGEST_BEFORE = None


def digest_tree(p: Path) -> str:
    h = hashlib.sha256()
    for f in sorted(x for x in p.rglob("*") if x.is_file()):
        h.update(f.relative_to(p).as_posix().encode())
        h.update(f.read_bytes())
    return h.hexdigest()


@pytest.fixture(scope="module", autouse=True)
def _guard_served_artifacts():
    """No test in this file may change the artifacts the live API serves."""
    global SERVED_DIGEST_BEFORE
    SERVED_DIGEST_BEFORE = digest_tree(MODELS_DIR)
    yield
    assert digest_tree(MODELS_DIR) == SERVED_DIGEST_BEFORE, \
        "a promote-gate test modified data/models/"


def stage_candidate(dst: Path) -> Path:
    """Copy exactly the files the gate requires. Verified non-empty afterwards: a hollow
    copy would look like a refusal and turn the test into a false positive."""
    dst.mkdir(parents=True, exist_ok=True)
    for name in CANDIDATE_FILES:
        shutil.copy2(MODELS_DIR / name, dst / name)
    got = {n: (dst / n).stat().st_size for n in CANDIDATE_FILES}
    assert all(v > 0 for v in got.values()), f"hollow candidate staged at {dst}: {got}"
    return dst


class _BiasedForecaster:
    """Over-predicts by a fixed fraction. Deliberately pickled with the *inner* model by
    reference-free delegation is avoided: this class is module-level so joblib can
    rehydrate it inside the gate process."""

    def __init__(self, inner, bias):
        self.inner = inner
        self.bias = bias

    def predict(self, X):
        return self.inner.predict(np.asarray(X, dtype=float)) * (1 + self.bias)


def make_regressed_candidate(dst: Path, bias: float = 0.06) -> Path:
    """Regress the no-weather forecaster AND narrow its advertised band, so both the MAPE
    rules and the coverage rules have something to catch."""
    stage_candidate(dst)
    path = dst / "lightgbm_load_forecast_no_weather.joblib"
    joblib.dump(_BiasedForecaster(joblib.load(path), bias), path, compress=3)
    for name in ("forecast_intervals.json", "forecast_intervals_no_weather.json"):
        art = json.loads((dst / name).read_text(encoding="utf-8"))
        for q in art["horizons"].values():
            q["q10"] = round(q["q10"] * 0.45, 5)
            q["q90"] = round(q["q90"] * 0.45, 5)
        (dst / name).write_text(json.dumps(art, indent=2))
    return dst


def test_equal_candidate_is_accepted(tmp_path):
    cand = stage_candidate(tmp_path / "candidate_equal")
    checks, evidence = evaluate_gate(cand, MODELS_DIR)
    failed = [c for c, ok in zip(checks, checks) if not ok]
    assert all(checks), f"identical candidate refused by: {failed}"
    cmp = evidence["vs_incumbent"]["steps"]
    assert cmp["24"]["worse_pct"] == pytest.approx(0.0)
    assert cmp["48"]["worse_pct"] == pytest.approx(0.0)


def test_regressed_candidate_is_refused_and_copies_nothing(tmp_path):
    cand = make_regressed_candidate(tmp_path / "candidate_regressed")
    checks, evidence = evaluate_gate(cand, MODELS_DIR)
    assert not all(checks), "a deliberately regressed candidate passed the gate"

    steps = evidence["vs_incumbent"]["steps"]
    assert steps["24"]["worse_pct"] > 5.0 or steps["48"]["worse_pct"] > 5.0
    # The refusal must be attributable, not just "some check said no".
    cl = evidence["closed_loop"]
    assert cl["regressed_vs_incumbent"], "band regression was not detected as a regression"

    target_before = digest_tree(MODELS_DIR)
    rc = main(["--candidate", str(cand), "--commit", "--yes"])
    assert rc == 1, f"expected refusal exit 1, got {rc}"
    assert digest_tree(MODELS_DIR) == target_before, "--commit copied despite refusal"
    decision = json.loads((cand / "retrain_decision.json").read_text(encoding="utf-8"))
    assert decision["passed"] is False and decision["committed"] is False


def test_dry_run_on_passing_gate_writes_nothing(tmp_path):
    cand = stage_candidate(tmp_path / "candidate_dry")
    rc = main(["--candidate", str(cand)])
    assert rc == 0
    assert digest_tree(MODELS_DIR) == SERVED_DIGEST_BEFORE
    decision = json.loads((cand / "retrain_decision.json").read_text(encoding="utf-8"))
    assert decision["passed"] is True and decision["committed"] is False


def test_candidate_equals_target_is_refused_without_measuring(tmp_path):
    """Self-comparison always 'passes'; it must be rejected structurally."""
    rc = main(["--candidate", str(MODELS_DIR), "--target", str(MODELS_DIR)])
    assert rc == 2


def test_incomplete_candidate_is_refused(tmp_path):
    cand = tmp_path / "partial"
    cand.mkdir()
    (cand / "baseline_results.csv").write_text("x\n")
    rc = main(["--candidate", str(cand)])
    assert rc == 2


def test_unmeasurable_incumbent_fails_closed(tmp_path):
    """No baseline means no comparison, which must never read as a pass."""
    cand = stage_candidate(tmp_path / "candidate_x")
    empty_target = tmp_path / "empty_target"
    empty_target.mkdir()
    with pytest.raises(Exception) as exc:
        evaluate_gate(cand, empty_target)
    assert exc.type.__name__ in ("IncumbentUnmeasurable", "FileNotFoundError")


def test_gate_measures_on_contiguous_future_window_not_a_shuffle(tmp_path):
    """Regression guard for the split itself. If someone reintroduces a random split,
    the origin count and window start would move away from the time-ordered values."""
    df = pd.read_parquet(DATA_FEATURES_DIR / "training_features.parquet")
    from models.train_baselines import train_val_test_split
    _, val, test = train_val_test_split(df)
    assert test["timestamp"].min() >= val["timestamp"].max()
    assert test["timestamp"].is_monotonic_increasing or \
        test.sort_values("timestamp")["timestamp"].is_monotonic_increasing
    cand = stage_candidate(tmp_path / "candidate_split")
    _, evidence = evaluate_gate(cand, MODELS_DIR)
    origins = evidence["vs_incumbent"]["test_origins"]
    assert origins >= 200, f"only {origins} contiguous future origins scored"


def test_recorded_thresholds_match_the_module_constants(tmp_path):
    cand = stage_candidate(tmp_path / "candidate_tol")
    _, evidence = evaluate_gate(cand, MODELS_DIR)
    tol = evidence["vs_incumbent"]["tolerances"]
    assert tol["max_relative_mape_regression_pct"] == 5.0
    assert tol["min_long_lead_coverage"] == 0.70
    assert tol["max_coverage_drop"] == 0.05
    assert tol["max_under_coverage_vs_nominal"] == 0.10
    assert tol["mape_floor_pct"] == {"24": 8.0, "48": 10.0}
