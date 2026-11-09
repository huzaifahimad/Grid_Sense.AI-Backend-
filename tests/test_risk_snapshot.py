"""
Snapshot selection for the risk panel (api/main.py::_representative_snapshot).

Every test here uses synthetic frames and a stubbed risk model — no network, and
no dependence on which real hours happen to sit in the committed parquet. The
newest complete row and the weekday+hour lookalike are deliberately DIFFERENT
rows in every fixture, because the defect this covers was exactly the two panels
of one product claiming different "nows".
"""

from datetime import datetime, timezone

import numpy as np
import pandas as pd
import pytest

from api import main as api_main
from models.risk_classifier import FEATURE_COLS as RISK_FEATURE_COLS


@pytest.fixture
def clean_risk_state(monkeypatch):
    """Serving state with no refresher: these tests must never reach the network."""
    monkeypatch.setattr(api_main.live_refresh, "kick_if_due", lambda *a, **k: None)

    class _RiskModel:
        def predict_proba(self, X):
            # Scores off load_lag_1h alone, so each test can assert that a specific
            # row was the one scored rather than merely a timestamp.
            return np.column_stack([1.0 - X["load_lag_1h"], X["load_lag_1h"]])

    saved = dict(api_main._state)
    api_main._state.update({"features": None, "risk_model": _RiskModel(), "risk_factors": {}})
    try:
        yield api_main
    finally:
        api_main._state.clear()
        api_main._state.update(saved)


def _frame(now, newest_offset_hours, rows):
    """Hourly frame whose newest observed hour sits `newest_offset_hours` behind
    floor(now), tz-aware like both the committed parquet and a refreshed frame.

    Whole hours behind a floor, never ahead of it: an age must be positive to mean
    anything, and the reported live case was 1.42h. The same UTC weekday+hour then
    exists earlier in the window as a genuinely different row.
    """
    end = pd.Timestamp(now).floor("h") - pd.Timedelta(hours=newest_offset_hours)
    ts = pd.date_range(end=end, periods=rows, freq="h")
    return pd.DataFrame({
        "timestamp": ts,
        "zone": "ES",
        "load_mw": 2000.0 + np.arange(len(ts), dtype=float),
        "capacity_mw": 5000.0,
        "capacity_is_proxy": True,
        **{col: 0.5 for col in RISK_FEATURE_COLS},
    })


def _lookalike_pos(frame, now):
    """Negative row position of the newest weekday+hour match that is not the
    newest row — precisely the row the old heuristic preferred."""
    matches = [i for i, ts in enumerate(frame["timestamp"])
               if ts.dayofweek == now.weekday() and ts.hour == now.hour
               and i != len(frame) - 1]
    assert matches, "fixture must contain a weekday+hour lookalike besides the newest row"
    return matches[-1] - len(frame)


def _naive(frame):
    """The same frame with its UTC stamps stripped of tzinfo, as an external
    source or a partial merge would deliver them."""
    frame["timestamp"] = frame["timestamp"].dt.tz_localize(None)
    return frame


def _mark(frame, positions, score):
    frame.loc[[frame.index[p] for p in positions], "load_lag_1h"] = score
    return frame


def test_fresh_data_scores_the_newest_complete_row_not_the_weekday_lookalike(clean_risk_state):
    """The regression: live ENTSO-E load is ~1h old, so the same UTC weekday+hour
    row from a week earlier was scored while /forecast next to it reported an origin
    1.4 hours old — two panels of one product claiming different "nows".

    Mirrors the serving shape: ROWS_PER_ZONE_BUFFER rows, newest observed hour
    ~1.4h old, and the lookalike slot present a week earlier as a different row.
    """
    now = datetime.now(timezone.utc)
    frame = _frame(now, newest_offset_hours=1, rows=api_main.ROWS_PER_ZONE_BUFFER)
    lookalike_pos = _lookalike_pos(frame, now)
    lookalike_ts = frame["timestamp"].iloc[lookalike_pos]
    assert lookalike_ts != frame["timestamp"].iloc[-1]
    # Only the newest row can produce 0.9; the lookalike would score 0.2.
    _mark(frame, [lookalike_pos], 0.2)
    _mark(frame, [-1], 0.9)
    clean_risk_state._state["features"] = frame

    served = clean_risk_state.get_risk("ES")

    assert pd.Timestamp(served.snapshot_timestamp) == frame["timestamp"].iloc[-1]
    assert served.snapshot_timestamp != lookalike_ts.isoformat()
    assert served.risk_score == pytest.approx(0.9)
    assert served.snapshot_mode == "newest"
    assert served.snapshot_age_hours < api_main.RISK_SNAPSHOT_LOOKALIKE_AFTER_HOURS


def test_naive_timestamps_from_a_refresh_merge_select_the_newest_row_too(clean_risk_state):
    """A tz-naive hour compared against aware wall clock reads as a different
    instant rather than raising, so the freshness check must normalize before
    comparing. The committed parquet is aware UTC today; this guards the mixed
    frame a refresh merge can produce when one side arrives naive.
    """
    frame = _naive(_mark(_frame(datetime.now(timezone.utc), newest_offset_hours=1, rows=72), [-1], 0.9))

    picked, mode = clean_risk_state._representative_snapshot(frame, RISK_FEATURE_COLS)

    assert mode == "newest"
    assert pd.Timestamp(picked["timestamp"].iloc[0]) == frame["timestamp"].iloc[-1]


def test_stale_newest_row_falls_back_to_the_weekday_hour_lookalike_and_says_so(clean_risk_state):
    """When the newest hour really is older than the threshold, a same-weekday/
    same-hour replay carries more information than the stale newest row."""
    now = datetime.now(timezone.utc)
    frame = _frame(now, newest_offset_hours=24 * 7 + 1, rows=200)
    lookalike_pos = _lookalike_pos(frame, now)
    # Every other candidate row scores 0.1, so only a lookalike pick yields 0.8.
    _mark(frame, list(range(len(frame))), 0.1)
    frame.loc[frame.index[lookalike_pos], "load_lag_1h"] = 0.8
    lookalike_ts = frame["timestamp"].iloc[lookalike_pos]
    assert lookalike_ts.dayofweek == now.weekday() and lookalike_ts.hour == now.hour
    clean_risk_state._state["features"] = frame

    served = clean_risk_state.get_risk("ES")

    assert pd.Timestamp(served.snapshot_timestamp) == lookalike_ts
    assert served.risk_score == pytest.approx(0.8)
    assert served.snapshot_mode == "weekday_hour_lookalike"
    assert served.snapshot_age_hours > api_main.RISK_SNAPSHOT_LOOKALIKE_AFTER_HOURS


def test_incomplete_newest_rows_yield_the_newest_complete_row(clean_risk_state):
    """The newest hour at the ENTSO-E publication frontier has no lag features yet;
    scoring its NaN row would be worse than dropping it, and the age must report the
    row actually used rather than the hour that was skipped."""
    frame = _mark(_frame(datetime.now(timezone.utc), newest_offset_hours=1, rows=72), [-1], 0.7)
    frame.loc[frame.index[-1], "load_roll_mean_24h"] = np.nan
    expected_ts = frame["timestamp"].iloc[-2]

    picked, mode = clean_risk_state._representative_snapshot(frame, RISK_FEATURE_COLS)

    assert mode == "newest"
    assert len(picked) == 1
    assert pd.Timestamp(picked["timestamp"].iloc[0]) == expected_ts
    assert not picked[RISK_FEATURE_COLS].isna().any().any()

    clean_risk_state._state["features"] = frame
    served = clean_risk_state.get_risk("ES")
    assert served.snapshot_mode == "newest"
    assert served.snapshot_load_mw == pytest.approx(
        float(frame.loc[frame["timestamp"] == expected_ts, "load_mw"].iloc[0]))
    expected_age_h = (api_main._now_utc() - expected_ts).total_seconds() / 3600
    assert served.snapshot_age_hours == pytest.approx(expected_age_h, abs=0.1)


def test_assets_panel_shares_the_selection_with_risk(clean_risk_state):
    """/assets ranks the same snapshots /risk scores; a divergence between them
    would just move the two-panels-disagree bug instead of fixing it."""
    frame = _mark(_frame(datetime.now(timezone.utc), newest_offset_hours=1, rows=72), [-1], 0.6)
    clean_risk_state._state["features"] = frame

    assets = clean_risk_state.list_assets()

    assert [a.zone for a in assets] == ["ES"]
    assert assets[0].risk_score == pytest.approx(0.6)
