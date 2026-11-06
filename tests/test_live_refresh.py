"""
Tests for the request-time refresher (api/live_refresh.py).

No test here touches the network. Every egress path is monkeypatched at the
module the refresher imports from, which is also the check that the refresher
does not quietly reach the internet during a test run.
"""

import threading
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import pytest

from api import live_refresh as lr

ZONE = "ES"
COMMITTED_ROWS = 336


@pytest.fixture
def committed(features):
    """The last 336h of one real zone, exactly as api/main.py loads it."""
    return (features[features["zone"] == ZONE]
            .sort_values("timestamp")
            .tail(COMMITTED_ROWS)
            .reset_index(drop=True))


def make_tail(committed, hours=30, scale=1.01, start_offset_h=1):
    base = committed["load_mw"].tail(hours).to_numpy() * scale
    ts = pd.date_range(committed["timestamp"].max() + pd.Timedelta(hours=start_offset_h),
                       periods=hours, freq="h")
    return pd.DataFrame({"timestamp": ts, "load_mw": base,
                         "zone": ZONE, "source": "entsoe"})


def changed_committed_cells(before: pd.DataFrame, after: pd.DataFrame) -> int:
    """
    Cells that differ on timestamps both frames already had. NaN is compared to
    NaN as equal — a raw != counts every NaN pair as a change and hides real ones.
    """
    cols = [c for c in before.columns if c != "timestamp"]
    b = before.set_index("timestamp")[cols]
    a = after.set_index("timestamp").loc[b.index, cols]
    b_obj = b.astype(object).where(b.notna(), "<<NA>>")
    a_obj = a.astype(object).where(a.notna(), "<<NA>>")
    return int((b_obj.values != a_obj.values).sum())


# ------------------------------------------------------------
# build_candidate: the merge itself


def test_merge_appends_only_and_leaves_committed_rows_identical(committed):
    tail = make_tail(committed)
    candidate, reason = lr.build_candidate(committed, tail)

    assert reason == "merged"
    assert len(candidate) == COMMITTED_ROWS + len(tail)
    assert candidate["timestamp"].max() == tail["timestamp"].max()
    assert changed_committed_cells(committed, candidate) == 0


def test_merged_rows_carry_training_time_feature_formulas(committed):
    tail = make_tail(committed, hours=200)
    candidate, reason = lr.build_candidate(committed, tail)
    assert reason == "merged"

    last = candidate.iloc[-1]
    assert abs(last["load_lag_1h"] - candidate.iloc[-2]["load_mw"]) < 1e-6
    assert abs(last["load_lag_24h"] - candidate.iloc[-25]["load_mw"]) < 1e-6
    assert abs(last["load_lag_168h"] - candidate.iloc[-169]["load_mw"]) < 1e-6
    window = candidate.iloc[-25:-1]["load_mw"]
    assert abs(last["load_roll_mean_24h"] - window.mean()) < 1e-6
    assert abs(last["load_roll_std_24h"] - window.std()) < 1e-6
    assert abs(last["hour_sin"] - np.sin(2 * np.pi * last["timestamp"].hour / 24)) < 1e-9
    assert last["load_source"] == "entsoe"


def test_capacity_basis_is_frozen_not_recomputed(committed):
    """
    Appended load above the p99 must NOT move capacity_mw. Recomputing it would
    redefine overload_risk_label and invalidate every published accuracy figure.
    """
    frozen = float(committed["capacity_mw"].iloc[0])
    tail = make_tail(committed, hours=40, scale=1.6)
    candidate, reason = lr.build_candidate(committed, tail)

    assert reason == "merged"
    assert candidate["capacity_mw"].nunique() == 1
    assert float(candidate["capacity_mw"].iloc[-1]) == pytest.approx(frozen)
    assert bool(candidate["capacity_is_proxy"].iloc[-1]) is True
    ratio = candidate["load_to_capacity_ratio"].iloc[-1]
    assert ratio == pytest.approx(candidate["load_mw"].iloc[-1] / frozen)
    assert int(candidate["overload_risk_label"].iloc[-1]) == 1


def test_republished_history_never_rewrites_a_committed_hour(committed):
    """ENTSO-E republishes. A refresh may add hours, never revise one already served."""
    newest = committed["timestamp"].max()
    dupe = pd.DataFrame({"timestamp": [newest], "load_mw": [1.0],
                         "zone": [ZONE], "source": ["entsoe"]})
    candidate, reason = lr.build_candidate(committed, dupe)

    assert reason == "no_new_hours"
    assert candidate.empty
    assert float(committed.loc[committed["timestamp"] == newest, "load_mw"].iloc[0]) != 1.0


def test_implausible_unit_change_is_rejected(committed):
    """A 40x tail reads as MW->kWh, not a grid event: reject rather than score it."""
    for scale, expect in [(40.0, "implausible_load"), (0.0, "implausible_load")]:
        candidate, reason = lr.build_candidate(committed, make_tail(committed, scale=scale))
        assert reason.startswith(expect), (scale, reason)
        assert candidate.empty


def test_gap_in_the_new_hours_cannot_shorten_history_below_the_forecast_floor():
    """
    A hole between the committed tail and the new rows would leave a contiguous
    suffix shorter than MIN_HISTORY_HOURS, turning a working zone into a 503 on
    /forecast. The merge must refuse instead.
    """
    ts = pd.date_range("2026-01-01", periods=COMMITTED_ROWS, freq="h", tz="UTC")
    loads = 30000 + 2000 * np.sin(np.arange(COMMITTED_ROWS) / 12.0)
    zone = pd.DataFrame({"timestamp": ts, "load_mw": loads, "zone": ZONE,
                         "load_source": "entsoe", "capacity_mw": 37000.0,
                         "capacity_is_proxy": True})
    for col in [*lr.FRAME_COLS]:
        zone[col] = 1.0
    zone["weather_source"] = "nasa_power"
    _, reason = lr.build_candidate(zone, pd.DataFrame(
        {"timestamp": pd.date_range(ts[-1] + pd.Timedelta(hours=300), periods=5, freq="h"),
         "load_mw": [31000.0] * 5, "zone": ZONE, "source": "entsoe"}))
    assert reason.startswith("short_history")


def test_weather_join_stamps_matched_hours_and_leaves_the_rest_blind(committed):
    tail = make_tail(committed, hours=6)
    weather = pd.DataFrame({
        "timestamp": tail["timestamp"][:3],
        "temperature_c": [20.0, 21.0, 22.0],
        "humidity_pct": [40.0, 41.0, 42.0],
        "wind_speed_ms": [3.0, 3.1, 3.2],
        "precip_mm": [0.0, 0.0, 0.0],
    })
    candidate, reason = lr.build_candidate(committed, tail, weather)

    assert reason == "merged"
    added = candidate.tail(6)
    assert added["weather_source"].tolist() == ["nasa_power"] * 3 + ["unavailable"] * 3
    assert added["temperature_c"].head(3).notna().all()
    assert added["temperature_c"].tail(3).isna().all()
    assert candidate["temperature_c"].dtype == "float64"


def test_merged_dtypes_match_the_committed_frame(committed):
    candidate, reason = lr.build_candidate(committed, make_tail(committed))
    assert reason == "merged"
    for col in committed.columns:
        assert str(candidate[col].dtype) == str(committed[col].dtype), col


# ------------------------------------------------------------
# Gating: what may start an egress call, and how often


@pytest.fixture
def clean_state(monkeypatch):
    monkeypatch.setattr(lr, "ENTSOE_API_KEY", "test-token-not-a-secret")
    lr.reset_state()
    lr.config.stale_after_hours = 3.0
    lr.config.cache_seconds = 900.0
    lr.config.quiet_after_no_gain_seconds = 3600.0
    yield lr
    lr.reset_state()


def _stale_frame(committed, age_hours=40):
    frame = committed.copy()
    frame["timestamp"] = frame["timestamp"] - pd.Timedelta(hours=age_hours)
    return frame


def test_no_egress_without_an_api_key(committed, monkeypatch):
    monkeypatch.setattr(lr, "ENTSOE_API_KEY", None)
    lr.reset_state()
    calls = []
    monkeypatch.setattr(lr, "run_refresh", lambda *a, **k: calls.append(1))

    lr.kick_if_due(_stale_frame(committed), lambda f: None)

    assert calls == []
    assert "ENTSOE_API_KEY" in lr.status(committed)["last_error"]


def test_fresh_frame_is_not_refreshed(committed, clean_state, monkeypatch):
    calls = []
    monkeypatch.setattr(clean_state, "run_refresh", lambda *a, **k: calls.append(1))
    now = datetime.now(timezone.utc)
    frame = committed.copy()
    frame["timestamp"] = frame["timestamp"] + (
        pd.Timestamp(now).floor("h") - frame["timestamp"].max())

    clean_state.kick_if_due(frame, lambda f: None, now=now)

    assert calls == []
    assert clean_state.status(frame)["last_attempt_at"] is not None


def test_stale_frame_kicks_exactly_once_per_cache_window(clean_state, committed, monkeypatch):
    started = []
    running = threading.Event()
    hold = threading.Event()

    def fake_run(frame, publish, **kwargs):
        started.append(len(frame))
        running.set()
        assert hold.wait(5), "the refresh thread outlived the test's hold window"

    monkeypatch.setattr(clean_state, "run_refresh", fake_run)
    frame = _stale_frame(committed)

    clean_state.kick_if_due(frame, lambda f: None)
    assert running.wait(3), "a stale frame must start exactly one background refresh"

    # A refresh is in flight here: further requests must not spawn another.
    clean_state.kick_if_due(frame, lambda f: None)
    clean_state.kick_if_due(frame, lambda f: None)
    assert clean_state.status()["in_flight"] is True
    hold.set()

    assert len(started) == 1, "a second request must not start a second refresh"
    assert clean_state.status()["attempts"] == 1


def test_cache_window_blocks_the_next_kick(clean_state, committed, monkeypatch):
    """
    The gate that keeps egress off a hot route: one completed refresh per cache
    window, however many requests arrive in between. Run synchronously against
    the real run_refresh so last_attempt_at is written by the code that writes
    it in production.
    """
    import ingestion.entsoe_client as ec

    monkeypatch.setattr(ec, "get_client", lambda **kw: object())
    calls = []

    def fake_pull(zone_code, country_code, start, end, **kwargs):
        calls.append(start)
        return make_tail(committed, hours=6)

    monkeypatch.setattr(ec, "pull_load", fake_pull)
    clean_state.config.weather_enabled = False
    frame = _stale_frame(committed, age_hours=6)

    clean_state.kick_if_due(frame, lambda f: None, background=False)
    clean_state.kick_if_due(frame, lambda f: None, background=False)

    assert len(calls) == 1, "the second kick must not reach ENTSO-E"


def test_disabled_switch_prevents_everything(clean_state, committed, monkeypatch):
    calls = []
    clean_state.config.enabled = False
    monkeypatch.setattr(clean_state, "run_refresh", lambda *a, **k: calls.append(1))

    clean_state.kick_if_due(_stale_frame(committed), lambda f: None)

    assert calls == []


def test_no_gain_attempt_backs_off_longer_than_the_cache(clean_state, committed, monkeypatch):
    """
    A zone sitting at ENTSO-E's publication frontier must not be re-requested
    every cache window forever.
    """
    import ingestion.entsoe_client as ec

    monkeypatch.setattr(ec, "get_client", lambda **kw: object())
    monkeypatch.setattr(ec, "pull_load",
                        lambda *a, **k: pd.DataFrame(columns=["timestamp", "load_mw"]))
    published = []
    clean_state.run_refresh(_stale_frame(committed), published.append)

    assert published == []
    zone_state = clean_state.status()["zones"][ZONE]
    retry_after = datetime.fromisoformat(zone_state["retry_after"])
    assert retry_after - datetime.now(timezone.utc) > timedelta(
        seconds=clean_state.config.cache_seconds)


# ------------------------------------------------------------
# run_refresh: the egress boundary, fully mocked


def test_run_refresh_publishes_a_newer_frame(clean_state, committed, monkeypatch):
    import ingestion.entsoe_client as ec

    stale = _stale_frame(committed, age_hours=6)
    seen = {}

    def fake_pull_load(zone_code, country_code, start, end, **kwargs):
        seen["zone_code"] = zone_code
        seen["start"] = start
        seen["backfill"] = kwargs.get("backfill")
        return make_tail(stale, hours=12)

    monkeypatch.setattr(ec, "get_client", lambda **kw: object())
    monkeypatch.setattr(ec, "pull_load", fake_pull_load)
    clean_state.config.weather_enabled = False

    published = []
    outcomes = clean_state.run_refresh(stale, published.append)

    assert outcomes[ZONE]["outcome"] == "merged"
    assert seen["zone_code"] == "10YES-REE------0"
    assert seen["backfill"] is False, "the day-by-day backfill must not run in-request"
    assert seen["start"] == stale["timestamp"].max() + pd.Timedelta(hours=1)
    assert len(published) == 1
    new_frame = published[0]
    assert len(new_frame) == len(stale) + 12
    assert new_frame["timestamp"].max() == make_tail(stale, hours=12)["timestamp"].max()
    status = clean_state.status(new_frame)
    assert status["rows_added_total"] == 12
    assert status["last_error"] is None
    assert status["last_success_at"] is not None


def test_egress_failure_leaves_the_frame_serving(clean_state, committed, monkeypatch):
    import ingestion.entsoe_client as ec

    def boom(*a, **k):
        raise RuntimeError("ENTSO-E security token is not valid")

    monkeypatch.setattr(ec, "get_client", lambda **kw: object())
    monkeypatch.setattr(ec, "pull_load", boom)

    published = []
    stale = _stale_frame(committed)
    clean_state.run_refresh(stale, published.append)

    assert published == [], "a failed refresh must not publish anything"
    status = clean_state.status(stale)
    assert status["zones"][ZONE]["last_error"].startswith("RuntimeError")
    assert "token is not valid" in status["zones"][ZONE]["last_error"], \
        "the upstream reason is what an operator needs to see"
    # The reported reason must not carry the credential, whatever the upstream says.
    assert lr.ENTSOE_API_KEY not in str(status)
    assert status["last_success_at"] is None


def test_unparseable_tail_is_rejected_not_published(clean_state, committed, monkeypatch):
    import ingestion.entsoe_client as ec

    monkeypatch.setattr(ec, "get_client", lambda **kw: object())
    monkeypatch.setattr(ec, "pull_load",
                        lambda *a, **k: make_tail(_stale_frame(committed), scale=99.0))
    published = []
    clean_state.run_refresh(_stale_frame(committed), published.append)

    assert published == []
    assert clean_state.status()["last_success_at"] is None


# ------------------------------------------------------------
# The real entsoe-py client, with only the socket stubbed
#
# Every test above replaces pull_load wholesale, which is precisely the blind spot
# that let a dead configuration reach a commit: entsoe-py's @retry runs
# `for _ in range(self.retry_count)` with a bare `raise error` in the for/else, so
# retry_count=0 performs no request at all and raises
# "TypeError: exceptions must derive from BaseException". Our own code looks
# perfectly healthy from the outside — the API is simply never called.


def test_the_refresh_client_really_calls_the_http_layer(monkeypatch):
    import requests
    from ingestion.entsoe_client import get_client, pull_load

    monkeypatch.setattr("ingestion.entsoe_client.ENTSOE_API_KEY", "test-token")
    client = get_client(timeout=1, retry_count=lr.ENTSOE_ATTEMPTS,
                        retry_delay=lr.ENTSOE_ATTEMPT_DELAY_SECONDS)
    seen = []

    def fake_get(url, params=None, **kwargs):
        seen.append(params)
        raise requests.ConnectionError(f"stub: refused to reach {url}")

    monkeypatch.setattr(client.session, "get", fake_get)
    start = pd.Timestamp("2026-11-05", tz="UTC")
    with pytest.raises(RuntimeError) as exc:
        pull_load("10YES-REE------0", "ES", start, start + pd.Timedelta(hours=6),
                  client=client, backfill=False, verbose=False)

    assert len(seen) == 1, "the refresh configuration produced zero HTTP attempts"
    assert "securityToken" in seen[0], "the token belongs in the query string"
    cause = str(exc.value)
    assert "ConnectionError" in cause, cause
    assert "TypeError" not in cause, "entsoe-py never reached the network"


# ------------------------------------------------------------
# /health publishes failure reasons — never the credential


def test_published_error_never_carries_the_token(clean_state, committed, monkeypatch):
    """
    entsoe-py sends the token as a query parameter and requests/urllib3 quote the
    full URL into the exception text this module stores, so the natural place an
    operator looks for a failing pull is exactly where the credential appears.
    The string below is the shape actually captured from the running process on
    2026-11-06; the fake key is the value that must disappear, which makes the
    assertion about the credential rather than about a substring picked to pass.
    """
    import ingestion.entsoe_client as ec

    secret = "9f3c1a77e0b54d2f"
    monkeypatch.setattr(lr, "ENTSOE_API_KEY", secret)
    monkeypatch.setattr(ec, "get_client", lambda **kw: object())

    def refused(*a, **k):
        raise RuntimeError(
            "ENTSO-E returned no data at all for ES (10YES-REE------0); first failure: "
            "ConnectionError: HTTPSConnectionPool(host='web-api.tp.entsoe.eu', "
            "port=443): Max retries exceeded with url: /api?domain=10YES-REE------0"
            f"&processType=A01&securityToken={secret}&offset=0&limit=1000 (Caused by "
            "NewConnectionError('<urllib3.connection.HTTPSConnection object at 0x7f8'>: "
            "Failed to establish a new connection: [Errno 111] Connection refused'))")

    monkeypatch.setattr(ec, "pull_load", refused)

    published = []
    clean_state.run_refresh(_stale_frame(committed), published.append)

    assert secret not in str(clean_state.status()), \
        "the refresh reason reached /health verbatim"
    reason = clean_state.status()["zones"][ZONE]["last_error"]
    assert "Connection refused" in reason, "the cap kept the URL and dropped the cause"
    assert len(reason) <= len("RuntimeError: ") + 300
    assert published == []


def test_redaction_runs_before_the_length_cap():
    """
    Truncate first and the '&' that terminates the token parameter can fall off the
    end, leaving a value the pattern no longer recognises. Asserting the wrong order
    leaks is what keeps this test from passing for no reason.
    """
    head = "https://web-api.tp.entsoe.eu/api?offset=0&securityToken=" + "Z" * 40
    assert "Z" in head[:96], "premise: a raw slice keeps part of the token"
    assert "Z" not in lr._error_text(RuntimeError(head), 96)


@pytest.mark.parametrize("param", ["securityToken", "security_token", "api_key", "TOKEN"])
def test_credential_pattern_covers_the_parameter_spellings_used(param):
    from config import redact_secrets

    out = redact_secrets(f"GET /api?domain=X&{param}=super-secret-value&offset=0")
    assert "super-secret-value" not in out
    assert out.endswith("&offset=0")


def test_offline_pull_stdout_never_carries_the_token(monkeypatch, capsys):
    """
    pull_all_zones prints each zone failure, and stdout is the log stream of a
    deployed process. The text it prints happens to be built redacted today, but
    this site catches *any* exception out of pull_load — so it applies the rule
    itself rather than depending on the callee to have redacted before raising.
    """
    import ingestion.entsoe_client as ec

    secret = "7b41d9c0e2af4855"

    def refused(zone_code, country_code, start, end, *a, **k):
        raise RuntimeError(f"Max retries exceeded with url: /api?documentType=A75"
                           f"&securityToken={secret}&offset=0")

    monkeypatch.setattr(ec, "pull_load", refused)
    with pytest.raises(RuntimeError):
        ec.pull_all_zones(pd.Timestamp("2026-11-01", tz="UTC"),
                          pd.Timestamp("2026-11-02", tz="UTC"))

    out = capsys.readouterr().out
    assert secret not in out, "the zone failure was printed verbatim"
    assert "securityToken=[REDACTED]" in out
    assert "documentType=A75" in out, "redaction should not eat the rest of the message"


# ------------------------------------------------------------
# Through the API


def test_health_reports_the_refresher(client):
    body = client.get("/health").json()
    refresh = body["data_refresh"]
    assert set(refresh) >= {"enabled", "stale_after_hours", "cache_seconds",
                            "last_attempt_at", "last_success_at", "attempts",
                            "rows_added_total", "last_error", "in_flight",
                            "has_api_key", "zones"}
    assert refresh["stale_after_hours"] > 0
    assert refresh["cache_seconds"] >= 60


def test_forecast_origin_advances_after_a_refresh(client, monkeypatch, features):
    """
    End to end through the real serving path: the exact pair api/main.py wires
    together (kick_if_due + _publish_frame) appends hours in memory, and
    /forecast then answers from the newer origin. This is the user-visible thing
    the module exists to fix, so it is checked through the endpoint, not around it.
    """
    import ingestion.entsoe_client as ec
    import api.main as m

    served = m._state["features"]
    zone_frame = served[served["zone"] == ZONE].sort_values("timestamp")
    newest = zone_frame["timestamp"].max()
    before = client.get(f"/forecast/{ZONE}").json()
    origin_before = pd.Timestamp(before["origin_timestamp"])

    tail = make_tail(zone_frame.reset_index(drop=True), hours=24)
    lr.reset_state()
    monkeypatch.setattr(lr, "ENTSOE_API_KEY", "test-token")
    monkeypatch.setattr(lr.config, "weather_enabled", False)
    monkeypatch.setattr(ec, "get_client", lambda **kw: object())
    monkeypatch.setattr(ec, "pull_load", lambda *a, **k: tail)
    # Registers the undo: the publish below overwrites this entry.
    monkeypatch.setitem(m._state, "features", served)

    lr.kick_if_due(m._state["features"], m._publish_frame, background=False)

    assert m._state["features"] is not served, "the refresher must publish into the API state"
    after = client.get(f"/forecast/{ZONE}").json()
    origin_after = pd.Timestamp(after["origin_timestamp"])

    assert origin_after > origin_before
    assert origin_after - origin_before == pd.Timedelta(hours=24)
    assert origin_after == newest + pd.Timedelta(hours=24)
    assert after["origin_age_hours"] < before["origin_age_hours"]
    assert len(after["points"]) == len(before["points"]), \
        "a refreshed frame must still answer with a full forecast"
    assert client.get("/health").json()["data_refresh"]["rows_added_total"] == 24
