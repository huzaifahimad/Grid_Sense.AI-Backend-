"""
Tests for scripts/forecast_snapshot_store.py and the honesty of scripts/live_accuracy.py.

These exist because the deliverable is an accuracy ledger whose whole value is saying
which kind of number it printed. A test that only checks "MAE is a float" would pass on a
ledger that mislabelled a backtest as live, which is the failure mode being guarded.

Everything writes to a scratch dir under the repo's ignored _audit_out/ tree -- never to
data/models/ or data/features/. Per project memory, /tmp is NOT shared between Git Bash
and Python on this host, so no test here relies on it.
"""
import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.forecast_snapshot_store import (  # noqa: E402
    SnapshotStore, _iso, _resolve, _split_scored, _utc_ts, artifact_fingerprint,
)

SCRATCH = ROOT / "_audit_out" / "live_accuracy"


@pytest.fixture
def scratch(tmp_path):
    d = SCRATCH if SCRATCH.exists() else tmp_path
    d.mkdir(parents=True, exist_ok=True)
    return d


def make_points(origin, steps=24, p50=1000.0, half=0.10):
    return [{"timestamp": origin + timedelta(hours=s), "step": s,
             "p10": p50 * (1 - half), "p50": p50, "p90": p50 * (1 + half)}
            for s in range(1, steps + 1)]


# ---------------------------------------------------------------- store writing
def test_record_persists_every_field_needed_to_score_later(scratch):
    """The row must be self-sufficient: zone, origin, issued-at, horizon, band, artifact."""
    p = scratch / "fields.jsonl"
    p.unlink(missing_ok=True)
    store = SnapshotStore(p, now=lambda: datetime(2026, 11, 7, 12, tzinfo=timezone.utc))
    origin = datetime(2026, 11, 5, 3, tzinfo=timezone.utc)
    res = store.record(zone="ES", origin_timestamp=origin, points=make_points(origin),
                       model_artifact_id="abc123", served_by="no_weather_fallback")
    assert res["error"] is None and res["written"] == 24

    row = json.loads(p.read_text(encoding="utf-8").splitlines()[0])
    for field in ("zone", "origin_timestamp", "issued_at", "target_timestamp", "step",
                  "p10", "p50", "p90", "model_artifact_id", "served_by"):
        assert field in row, f"{field} missing -- the daily join could not score this row"
    assert row["issued_at"].endswith("+00:00"), "issued_at must carry an explicit UTC offset"
    assert row["p10"] < row["p50"] < row["p90"], "band ordering inverted; alarms key off P90"
    # target hour is origin + step, which is what the actual joins against
    assert _utc_ts(row["target_timestamp"]) == _utc_ts(origin) + timedelta(hours=1)


def test_record_never_raises_even_on_garbage_points():
    """A forecast that cannot be logged must still reach the user, not become a 500."""
    store = SnapshotStore(":nonsense:/x/y.jsonl")
    res = store.record(zone="ES", origin_timestamp=datetime.now(timezone.utc),
                       points=[{"wrong": 1}], model_artifact_id="a")
    assert res["written"] == 0 and res["error"], "must report why, silently is the worst outcome"


def test_duplicate_poll_requests_are_suppressed_not_double_counted(scratch):
    """
    Chatty clients must not weight the retrain-deciding metric. Suppression keys on the
    identity of the scored cell, so a genuinely new origin always records.
    """
    p = scratch / "dedupe.jsonl"
    p.unlink(missing_ok=True)
    t0 = datetime(2026, 11, 7, 12, tzinfo=timezone.utc)
    clock = {"now": t0}
    store = SnapshotStore(p, now=lambda: clock["now"])
    origin = datetime(2026, 11, 5, 3, tzinfo=timezone.utc)
    first = store.record(zone="ES", origin_timestamp=origin, points=make_points(origin),
                         model_artifact_id="aaa")
    again = store.record(zone="ES", origin_timestamp=origin, points=make_points(origin),
                         model_artifact_id="aaa")
    assert first["written"] == 24 and again["written"] == 0 and again["suppressed"] == 24

    # same zone, later origin -> different cells, must record
    clock["now"] = t0 + timedelta(hours=1)
    newer = store.record(zone="ES", origin_timestamp=origin + timedelta(hours=1),
                         points=make_points(origin + timedelta(hours=1)), model_artifact_id="aaa")
    assert newer["written"] == 24, "suppression must not swallow a genuinely new origin"


def test_overlong_store_refuses_to_grow(scratch):
    p = scratch / "bounded.jsonl"
    p.unlink(missing_ok=True)
    store = SnapshotStore(p)
    origin = datetime(2026, 11, 5, 3, tzinfo=timezone.utc)
    store.record(zone="ES", origin_timestamp=origin, points=make_points(origin),
                 model_artifact_id="a")
    big = store.record(zone="ES", origin_timestamp=origin + timedelta(hours=1),
                       points=make_points(origin + timedelta(hours=1)),
                       model_artifact_id="a", max_bytes=10)
    assert big["written"] == 0 and "bytes" in big["error"]


# ---------------------------------------------------------------- join to actuals
def test_join_scores_issued_vs_actual_and_reports_coverage(scratch):
    """Half the hours outside the band must read as coverage 0.50, not be hidden by MAE."""
    p = scratch / "join.jsonl"
    p.unlink(missing_ok=True)
    now = datetime(2026, 11, 12, tzinfo=timezone.utc)
    store = SnapshotStore(p, now=lambda: now)
    origin = datetime(2026, 11, 10, 5, tzinfo=timezone.utc)
    pts = make_points(origin, p50=1000.0, half=0.10)
    store.record(zone="ES", origin_timestamp=origin, points=pts, model_artifact_id="fp")

    actuals = [{"timestamp": q["timestamp"],
                # even steps land above P90 (uncovered), odd steps exactly on P50
                "load_mw": 1200.0 if q["step"] % 2 == 0 else 1000.0, "zone": "ES"}
               for q in pts]
    res = store.join_to_actuals(pd.DataFrame(actuals))
    o = res["overall"]
    assert res["is_live_not_backtest"] is True
    assert o["cells"] == 24 and res["awaiting_actual"] == 0
    assert o["mae_mw"] == pytest.approx(100.0)          # mean(|200|,|0| alternating)
    assert o["coverage_p10_p90"] == pytest.approx(0.5)
    assert o["bias_signed_pct"] == pytest.approx(-8.333, abs=0.01)
    assert res["per_step"]["h+1"]["cells"] == 1


def test_join_counts_rows_whose_actual_has_not_arrived(scratch):
    """
    Unjoined rows are counted, never dropped: whether they are normal publication lag or
    a dead feed is exactly what the age distribution tells an operator.
    """
    p = scratch / "await.jsonl"
    p.unlink(missing_ok=True)
    clock = {"now": datetime(2026, 11, 10, 5, tzinfo=timezone.utc)}
    store = SnapshotStore(p, now=lambda: clock["now"])
    origin = datetime(2026, 11, 10, 5, tzinfo=timezone.utc)
    store.record(zone="ES", origin_timestamp=origin, points=make_points(origin),
                 model_artifact_id="fp")
    # Advance the clock: issued hours ago, actual still missing. Pinning one instant for
    # both write and join would make the age trivially 0.0 and assert nothing.
    clock["now"] = datetime(2026, 11, 20, 5, tzinfo=timezone.utc)
    res = store.join_to_actuals(pd.DataFrame(columns=["timestamp", "load_mw", "zone"]))
    assert res["scored_cells"] == 0 and res["awaiting_actual"] == 24
    assert res["oldest_unjoined_age_hours"] == pytest.approx(240.0, abs=1.0)


def test_join_matches_across_timezone_offset_spellings(scratch):
    """
    The same instant written with a different offset label must still join. String
    equality on ISO timestamps was the original bug: it silently dropped real joins and
    made coverage look better than it was because the hard cells vanished.
    """
    p = scratch / "tz.jsonl"
    p.unlink(missing_ok=True)
    store = SnapshotStore(p, now=lambda: datetime(2026, 11, 12, tzinfo=timezone.utc))
    origin = datetime(2026, 11, 10, 5, tzinfo=timezone.utc)
    store.record(zone="ES", origin_timestamp=origin, points=make_points(origin, steps=3),
                 model_artifact_id="fp")
    # actuals expressed as UTC-aware instants (identical moments, different rendering)
    act = pd.DataFrame([{"timestamp": origin + timedelta(hours=s), "load_mw": 1000.0,
                         "zone": "ES"} for s in (1, 2, 3)])
    assert store.join_to_actuals(act)["scored_cells"] == 3


# ---------------------------------------------------------------- compaction
def test_compaction_is_conservative_and_idempotent(scratch):
    """
    Nothing may vanish without a record. Measured during development: an earlier version
    deleted aged rows while reporting summarised=0, so the conservation check is the test.
    """
    p = scratch / "cmp.jsonl"
    sp = scratch / "cmp_sum.jsonl"
    p.unlink(missing_ok=True)
    sp.unlink(missing_ok=True)
    now = datetime(2026, 11, 7, tzinfo=timezone.utc)
    store = SnapshotStore(p, now=lambda: now)

    aged = []
    for d in (1, 2):                                    # >90 days old, actual present
        org = datetime(2026, 8, d, 3, tzinfo=timezone.utc)
        pts = make_points(org)
        store.record(zone="ES", origin_timestamp=org, issued_at=org, points=pts,
                     model_artifact_id="aaa")
        aged += [{"timestamp": q["timestamp"], "load_mw": 1000.0, "zone": "ES"} for q in pts]
    fresh_org = datetime(2026, 11, 5, 3, tzinfo=timezone.utc)
    store.record(zone="GR", origin_timestamp=fresh_org, issued_at=fresh_org,
                 points=make_points(fresh_org), model_artifact_id="bbb")
    aged += [{"timestamp": q["timestamp"], "load_mw": 990.0, "zone": "GR"}
             for q in make_points(fresh_org)]
    dead_org = datetime(2026, 6, 1, 3, tzinfo=timezone.utc)   # old, no actual, past expiry
    store.record(zone="IT_SOUTH", origin_timestamp=dead_org, issued_at=dead_org,
                 points=make_points(dead_org), model_artifact_id="ccc")

    total_in = store.stats()["rows"]
    res = store.compact(actuals=pd.DataFrame(aged), summary_path=sp)
    assert res["rewrote"] is True
    assert res["kept"] + res["summarised"] + res["expired"] == total_in
    assert res["summarised"] == 48, "aged-but-scored rows must be summarised, not deleted"
    assert res["expired"] == 24

    kinds = [json.loads(l)["kind"] for l in sp.read_text(encoding="utf-8").splitlines()]
    assert kinds.count("summary") == 48 and kinds.count("expired_tombstone") == 24, \
        "every deleted row needs its own tombstone, or the gap is unauditable"

    second = store.compact(actuals=pd.DataFrame(aged), summary_path=sp)
    assert second["rewrote"] is False and second["kept"] == res["kept"], \
        "compaction must be idempotent; a second run deleting detail is data loss"


def test_compact_without_actuals_deletes_nothing(scratch):
    """A stale row and a merely-lagging row are indistinguishable without actuals."""
    p = scratch / "nocut.jsonl"
    p.unlink(missing_ok=True)
    store = SnapshotStore(p, now=lambda: datetime(2026, 11, 7, tzinfo=timezone.utc))
    org = datetime(2026, 1, 1, 3, tzinfo=timezone.utc)
    store.record(zone="ES", origin_timestamp=org, issued_at=org, points=make_points(org),
                 model_artifact_id="a")
    before = store.stats()["rows"]
    assert store.compact()["kept"] == before
    assert store.stats()["rows"] == before


# ---------------------------------------------------------------- fingerprint
def test_artifact_fingerprint_tracks_content_not_filename(scratch, monkeypatch):
    """
    Retraining overwrites the joblib in place, so a filename is not an id: two rows could
    claim the same model while meaning different weights, and a retrain gate would then
    attribute one generation's error to another.
    """
    a = scratch / "m_a.joblib"
    b = scratch / "m_b.joblib"
    a.write_bytes(b"weights-v1")
    same = artifact_fingerprint(a)
    assert same == artifact_fingerprint(a)
    a.write_bytes(b"weights-v2")
    assert artifact_fingerprint(a) != same, "fingerprint must change when bytes change"
    b.write_bytes(b"weights-v1")
    assert artifact_fingerprint(a) != artifact_fingerprint(b)
    assert len(same) == 16 and "/" not in same and "\\" not in same, "id must not leak paths"


def test_fingerprint_absent_file_does_not_raise(scratch):
    assert len(artifact_fingerprint(scratch / "nope.joblib")) == 16


# ---------------------------------------------------------------- path handling
def test_resolve_maps_posix_tmp_onto_real_tempdir():
    """
    On Windows Path('/tmp/x') silently means C:\\tmp\\x -- outside every convention and
    outside the repo, so nothing cleans it up. The remap must land in the real temp dir.
    """
    import tempfile
    out = _resolve("/tmp/gridsense_probe.jsonl")
    assert out.parent == Path(tempfile.gettempdir()).resolve()
    assert out.name == "gridsense_probe.jsonl"
    assert str(out) != str(Path("\\tmp\\gridsense_probe.jsonl")), "naive drive-root path leaked"
    # A repo-relative or explicit path is honoured exactly, never remapped.
    rel = _resolve("data/forecasts/snapshots.jsonl")
    assert str(rel).replace("\\", "/").endswith("data/forecasts/snapshots.jsonl")


def test_tolerant_read_skips_torn_tail(scratch):
    p = scratch / "torn.jsonl"
    # Truncate explicitly: _audit_out/ survives between runs, so appending to a
    # half-remembered file from an earlier session would inflate every count below.
    p.unlink(missing_ok=True)
    origin = datetime(2026, 11, 5, 3, tzinfo=timezone.utc)
    store = SnapshotStore(p)
    store.record(zone="ES", origin_timestamp=origin, points=make_points(origin, steps=2),
                 model_artifact_id="a")
    with p.open("a", encoding="utf-8") as fh:      # simulate an interrupted write
        fh.write('{"zone": "ES", "step": 3, "p50": \n')
        fh.write("\n")
    assert len(store.read_rows()) == 2, "one torn line must not cost the whole ledger"
