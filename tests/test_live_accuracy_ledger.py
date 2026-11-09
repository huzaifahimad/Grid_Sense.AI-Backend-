"""
Tests for scripts/live_accuracy.py — specifically the claims it makes about itself.

The script's metrics are checked by running it (see the ledger output); what these tests
pin down is the part a reviewer cannot recompute: that the output can never be read as
live accuracy by accident. A banner in a docstring is not a control, because nobody who
screenshots the table reads the docstring.

Read-only with respect to committed artifacts: every case points the store at
_audit_out/ scratch paths and never runs the closed-loop validator, so nothing under
data/models/ or data/features/ is rewritten.
"""
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import scripts.live_accuracy as la  # noqa: E402

SCRATCH = ROOT / "_audit_out" / "live_accuracy"


@pytest.fixture(scope="module")
def ledger():
    """One real replay of the serving path over the shortest window the data supports."""
    return la.run(days=la.DEFAULT_DAYS, horizon=la.HORIZON, stride=6, quiet=True)


# ---------------------------------------------------------------- self-declared status
def test_banner_names_backtest_and_the_reason():
    text = la.BACKTEST_BANNER
    assert "BACKTEST" in text and "NOT LIVE ACCURACY" in text
    # The one-sentence reason has to be present, or the caveat is decorative: it must name
    # both the missing persistence and the store that would provide it.
    lowered = text.lower()
    assert "never persisted" in lowered
    assert "snapshot" in lowered and "store" in lowered


def test_banner_is_printed_notmerely_documented(capsys):
    """Requirement 2: the status must appear in the printed output itself."""
    la.run(days=la.DEFAULT_DAYS, horizon=la.HORIZON, stride=12)
    out = capsys.readouterr().out
    assert "THIS IS A BACKTEST / REPLAY -- NOT LIVE ACCURACY." in out
    assert out.index("BACKTEST / REPLAY") < out.index("MAE"), \
        "the caveat must precede the numbers; after them it is a footnote"
    assert "MODE=replay/backtest" in out


def test_provenance_agrees_with_the_banner(ledger):
    p = ledger["provenance"]
    assert p["live_or_backtest"] == "backtest"
    assert "snapshot" in p["why_not_live"]
    assert p["kind"].startswith("backtested serving path")


# ---------------------------------------------------------------- metrics are real
def test_metrics_are_real_numbers_from_the_honest_split(ledger):
    o = ledger["overall"]["overall"]
    assert o["cells"] > 100, "a ledger over a handful of cells proves nothing"
    for key in ("mae_mw", "rmse_mw", "mape_pct"):
        v = o[key]
        assert isinstance(v, float) and v == v and v >= 0.0, f"{key} is not a real number"
    # Bias is SIGNED -- it says which way the forecaster errs, so it goes both sides of
    # zero depending on the window (measured across runs: -1.16% and +3.18%). Asserting a
    # sign here would have encoded my guess about the data rather than the data.
    bias = o["bias_signed_pct"]
    assert isinstance(bias, float) and bias == bias and abs(bias) < 100.0
    # RMSE >= MAE always holds (Jensen); a violation means the two were not computed from
    # the same error array.
    assert o["rmse_mw"] >= o["mae_mw"], "RMSE < MAE means the metrics came from different data"
    # Errors on MW-scale load must not be ~0 (a stubbed predictor) nor absurd (>10x mean load).
    assert 0.5 < o["mae_mw"] < 50_000, o["mae_mw"]


def test_band_coverage_is_reported_at_every_level(ledger):
    """
    Coverage is the product contract for an alarm system. If it were only aggregated, a
    single under-covering day could hide inside a healthy overall number.
    """
    assert "coverage_p10_p90" in ledger["overall"]["overall"]
    assert ledger["per_day"], "no days rolled up"
    for d in ledger["per_day"]:
        cov = d["overall"]["coverage_p10_p90"]
        assert 0.0 <= cov <= 1.0
        for step in d["per_step"].values():
            assert 0.0 <= step["coverage_p10_p90"] <= 1.0
    for z in ledger["per_zone"]:
        assert 0.0 <= z["overall"]["coverage_p10_p90"] <= 1.0


def test_coverage_flags_under_covering_days(ledger):
    """A day below the alarm floor must be labelled, not left for a reader to notice."""
    flagged = [d["label"] for d in ledger["per_day"]
               if d["overall"]["coverage_p10_p90"] < la.COVERAGE_FLOOR]
    for d in ledger["per_day"]:
        if d["overall"]["coverage_p10_p90"] < la.COVERAGE_FLOOR:
            assert d["label"] in flagged
            assert any("UNDER-COVERED" == v["flag"] or "thin coverage" == v["flag"]
                       for v in d["per_step"].values())
    # The thresholds themselves, on synthetic inputs: which days exist depends on what the
    # feeds have published, but the rule is fixed, and 0.78 sitting between the floor and
    # the nominal band is exactly where a hardcoded guess went wrong first time.
    floor, thin_line = la.COVERAGE_FLOOR, la.NOMINAL_COVERAGE - 0.05
    assert la.flag({"coverage_p10_p90": floor - 0.15, "mape_pct": 1.0}, 1) == "UNDER-COVERED"
    assert la.flag({"coverage_p10_p90": (floor + thin_line) / 2, "mape_pct": 1.0}, 1) \
        == "thin coverage"
    assert la.flag({"coverage_p10_p90": thin_line, "mape_pct": 1.0}, 1) == ""
    assert la.flag({"coverage_p10_p90": 0.85, "mape_pct": 1.0}, 1) == ""


def test_day_buckets_partition_the_scored_cells(ledger):
    """No cell may be double counted across days, or totals stop meaning anything."""
    total = ledger["overall"]["overall"]["cells"]
    by_day = sum(d["overall"]["cells"] for d in ledger["per_day"])
    by_zone = sum(z["overall"]["cells"] for z in ledger["per_zone"])
    assert by_day == total == by_zone, (by_day, total, by_zone)


def test_negative_control_still_runs_through_the_same_engine():
    """
    The control must be scored by the identical accounting, otherwise it does not isolate
    the model as the thing being measured.
    """
    res = la.run(days=la.DEFAULT_DAYS, horizon=la.HORIZON, stride=12,
                 control="seasonal-naive", quiet=True)
    assert "negative control" in res["provenance"]["kind"]
    assert res["overall"]["overall"]["cells"] > 0


# ---------------------------------------------------------------- refusal behaviour
def test_live_mode_refuses_to_substitute_the_backtest(tmp_path):
    """
    Asking for --live and quietly getting replay numbers is the precise misreading this
    whole deliverable exists to prevent, so it must exit non-zero and say why.
    """
    SCRATCH.mkdir(parents=True, exist_ok=True)
    empty = tmp_path / "does_not_exist.jsonl"
    code = la.main(["--live", "--store", str(empty)])
    assert code == 2, f"expected a refusal (2), got {code}"


def test_live_mode_scores_a_populated_store(tmp_path):
    from datetime import datetime, timedelta, timezone

    from scripts.forecast_snapshot_store import SnapshotStore

    import pandas as pd

    store_path = tmp_path / "populated.jsonl"
    store = SnapshotStore(store_path,
                          now=lambda: datetime(2026, 11, 12, tzinfo=timezone.utc))
    origin = datetime(2026, 11, 10, 5, tzinfo=timezone.utc)
    pts = [{"timestamp": origin + timedelta(hours=s), "step": s,
            "p10": 900.0, "p50": 1000.0, "p90": 1120.0} for s in range(1, 13)]
    store.record(zone="ES", origin_timestamp=origin, points=pts,
                 model_artifact_id="testfp", served_by="no_weather_fallback")

    feats = tmp_path / "actuals.parquet"
    pd.DataFrame([{"timestamp": q["timestamp"], "load_mw": 1000.0, "zone": "ES"}
                  for q in pts]).to_parquet(feats, index=False)

    # Point the join at synthetic actuals rather than the committed parquet, so the test
    # cannot depend on which hours the live feeds happen to have published.
    code = la.run_live(store_path=str(store_path), features_path=feats)
    assert code == 0, "a populated store must produce live numbers, not a refusal"


def test_json_flag_writes_only_where_asked(tmp_path):
    """The script promises write-free runs; --json is the sole exception and it is explicit."""
    before = {p.name for p in (ROOT / "data" / "models").iterdir()}
    out = tmp_path / "ledger.json"
    rc = la.main(["--days", str(la.DEFAULT_DAYS), "--horizon", "6", "--stride", "12",
                  "--json", str(out)])
    assert rc == 0
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["provenance"]["live_or_backtest"] == "backtest"
    after = {p.name for p in (ROOT / "data" / "models").iterdir()}
    assert before == after, "committed artifact directory changed -- forbidden"
