import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

ZONES = ["ES", "GR", "IT_SOUTH"]


def test_health_ready(client):
    body = client.get("/health").json()
    assert body["status"] == "ok"
    assert all(body["artifacts"].values()), body["artifacts"]
    assert body["zones"] == sorted(ZONES)
    assert "conformal" in body["interval_method"]


def test_health_declares_real_data_provenance(client):
    """The API must state where its numbers come from — a stand-in dataset must not
    be servable silently as if it were a real pull."""
    health = client.get("/health").json()
    prov = health["data_provenance"]
    assert set(prov) == {
        "load",
        "weather",
        "capacity",
        "load_latest_timestamp",
        "weather_latest_timestamp",
        "weather_lag_hours",
        "is_synthetic",
    }
    assert list(prov["load"]) == ["entsoe"], prov["load"]
    assert set(prov["weather"]) <= {"nasa_power", "unavailable"}, prov["weather"]
    assert "nasa_power" in prov["weather"], prov["weather"]
    assert prov["capacity"] == "p99-of-load proxy"
    assert prov["is_synthetic"] is False


def test_health_declares_per_feed_currency(client):
    """Reanalysis weather lags real-time load by days, so the two feeds must report
    separate horizons — a single 'last updated' would misstate what the models saw."""
    health = client.get("/health").json()
    prov = health["data_provenance"]
    import pandas as pd

    load_latest = pd.Timestamp(prov["load_latest_timestamp"])
    assert load_latest == pd.Timestamp(health["serving_window_end"])
    weather_latest = pd.Timestamp(prov["weather_latest_timestamp"])
    assert weather_latest <= load_latest
    expected_lag = (load_latest - weather_latest).total_seconds() / 3600
    assert prov["weather_lag_hours"] == round(expected_lag, 1)


def test_root(client):
    assert client.get("/").json()["status"] == "ok"


@pytest.mark.parametrize("zone", ZONES)
def test_forecast_is_forward_looking_and_ordered(client, zone):
    r = client.get(f"/forecast/{zone}", params={"horizon_hours": 48})
    assert r.status_code == 200
    body = r.json()
    pts = body["points"]
    assert len(pts) == 48
    origin = pd.Timestamp(body["origin_timestamp"])
    ts = pd.to_datetime([p["timestamp"] for p in pts])
    assert ts[0] == origin + pd.Timedelta(hours=1)
    assert (ts.to_series().diff().dropna() == pd.Timedelta(hours=1)).all()
    assert [p["step"] for p in pts] == list(range(1, 49))
    for p in pts:
        assert 0 < p["p10"] <= p["p50"] <= p["p90"]


def test_forecast_default_horizon_is_24(client):
    assert len(client.get("/forecast/ES").json()["points"]) == 24


@pytest.mark.parametrize("horizon", [0, 49])
def test_forecast_rejects_out_of_range_horizon(client, horizon):
    assert client.get("/forecast/ES", params={"horizon_hours": horizon}).status_code == 422


@pytest.mark.parametrize("path", ["/forecast/XX", "/risk/XX", "/shed-schedule/XX"])
def test_unknown_zone_404(client, path):
    assert client.get(path).status_code == 404


@pytest.mark.parametrize("zone", ZONES)
def test_risk_contract(client, zone):
    body = client.get(f"/risk/{zone}").json()
    assert 0.0 <= body["risk_score"] <= 1.0
    assert body["status"] in {"critical", "elevated", "normal"}
    assert body["capacity_mw"] > 0 and body["snapshot_load_mw"] > 0
    assert body["is_proxy_capacity"] is True
    assert body["asset_id"] == f"{zone}-FEEDER-01"
    pd.Timestamp(body["snapshot_timestamp"])
    assert len(body["top_factors"]) <= 3


def test_assets_sorted_and_status_consistent_with_risk(client):
    assets = client.get("/assets").json()
    assert {a["zone"] for a in assets} == set(ZONES)
    scores = [a["risk_score"] for a in assets]
    assert scores == sorted(scores, reverse=True)
    for a in assets:
        assert client.get(f"/risk/{a['zone']}").json()["status"] == a["status"]


@pytest.mark.parametrize("zone", ZONES)
def test_shed_schedule_matches_status(client, zone):
    risk = client.get(f"/risk/{zone}").json()
    shed = client.get(f"/shed-schedule/{zone}").json()
    if risk["status"] == "normal":
        assert shed == []
    else:
        assert len(shed) == 1
        assert shed[0]["priority"] == (1 if risk["status"] == "critical" else 2)
        expected_margin = risk["capacity_mw"] - risk["snapshot_load_mw"]
        assert shed[0]["predicted_margin_mw"] == pytest.approx(expected_margin)
        # The snapshot load is observed, not forecast — the reasoning must not claim otherwise.
        assert "Predicted load" not in shed[0]["reasoning"]
        assert risk["snapshot_timestamp"][:16].replace("T", " ") in shed[0]["reasoning"]


def test_contiguous_hourly_tail_drops_history_before_a_gap():
    """A hole in the series must truncate the usable window, never shift the lag features.

    recursive_forecast reads lags by row offset (-1/-24/-168) but stamps its output
    from the final timestamp only, so a gap would produce plausible, misaligned
    numbers instead of an error.
    """
    from api.main import _contiguous_hourly_tail
    from models.forecasting import MIN_HISTORY_HOURS

    idx = pd.date_range("2026-01-01", periods=200, freq="h", tz="UTC")
    df = pd.DataFrame({"timestamp": idx, "load_mw": 1000.0})

    intact = _contiguous_hourly_tail(df)
    assert len(intact) == 200
    assert intact["timestamp"].iloc[0] == idx[0]

    gapped = df.drop(df.index[37:120])
    trimmed = _contiguous_hourly_tail(gapped)
    assert len(trimmed) == 80
    assert trimmed["timestamp"].iloc[0] == idx[120]
    assert trimmed["timestamp"].diff().dt.total_seconds().max() == 3600.0

    # Losing everything before the gap is what pushes the zone under the serving
    # minimum, so the endpoint rejects it instead of forecasting off misaligned lags.
    assert len(trimmed) < MIN_HISTORY_HOURS

    assert len(_contiguous_hourly_tail(df.head(1))) == 1
    assert len(_contiguous_hourly_tail(df.head(0))) == 0


def test_serving_path_does_not_import_training_only_deps():
    root = Path(__file__).parent.parent
    code = (
        "import sys; sys.path.insert(0, r'%s'); import api.main; "
        "bad = [m for m in ('shap', 'numba', 'llvmlite', 'torch') if m in sys.modules]; "
        "print(','.join(bad))" % root
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "", f"serving path imported: {out.stdout.strip()}"


def _stub_state(blind_tail_hours: int):
    """Serving state for a zone whose newest hours have load but no published weather."""
    import numpy as np
    import pandas as pd
    from models.forecasting import WEATHER_COLS

    ts = pd.date_range("2026-01-01", periods=200, freq="h", tz="UTC")
    df = pd.DataFrame({"timestamp": ts, "zone": "ES", "load_mw": 1000.0 + np.arange(len(ts))})
    for col in WEATHER_COLS:
        df[col] = 20.0
    if blind_tail_hours:
        df.loc[df["timestamp"] >= ts[-blind_tail_hours], list(WEATHER_COLS)] = np.nan

    class _Stub:
        def __init__(self, value):
            self._value = value

        def predict(self, X):
            return np.full(len(X), float(self._value))

    bands = lambda tag: {  # noqa: E731 - tiny local helper for two near-identical artifacts
        "method": f"{tag} bands",
        "horizons": {str(h): {"q10": -0.05, "q90": 0.05} for h in range(1, 49)},
    }
    return df, _Stub(1500.0), _Stub(1700.0), bands("weather-aware"), bands("fallback")


def test_forecast_answers_from_the_no_weather_model_when_weather_is_not_published_yet(monkeypatch):
    """
    NASA publishes an hour roughly two days late while ENTSO-E publishes load far
    sooner. Waiting for weather would strand the forecast origin in the past, so the
    reduced model answers from the newest load hour — but only with its own bands, and
    the weather-aware model must never be fed the NaN tail.
    """
    from api import main as api_main

    df, full_model, fb_model, full_bands, fb_bands = _stub_state(blind_tail_hours=40)
    saved = dict(api_main._state)
    try:
        api_main._state.update({
            "features": df, "forecast_model": full_model, "fallback_model": fb_model,
            "intervals": full_bands, "fallback_intervals": fb_bands,
        })
        served = api_main.get_forecast("ES", 24)
        assert served.served_by == "no_weather_fallback"
        assert pd.Timestamp(served.origin_timestamp) == df["timestamp"].iloc[-1]
        assert served.interval_method == "fallback bands"
        assert served.origin_age_hours > 0
    finally:
        api_main._state.clear()
        api_main._state.update(saved)


def test_forecast_keeps_the_weather_aware_model_when_the_tail_is_not_worth_it(monkeypatch):
    """A two-hour gain does not justify trading measured accuracy for freshness."""
    from api import main as api_main

    df, full_model, fb_model, full_bands, fb_bands = _stub_state(blind_tail_hours=2)
    saved = dict(api_main._state)
    try:
        api_main._state.update({
            "features": df, "forecast_model": full_model, "fallback_model": fb_model,
            "intervals": full_bands, "fallback_intervals": fb_bands,
        })
        served = api_main.get_forecast("ES", 24)
        assert served.served_by == "weather_aware"
        assert served.interval_method == "weather-aware bands"
        assert pd.Timestamp(served.origin_timestamp) == df["timestamp"].iloc[-3]

        # With no fallback artifacts loaded the behaviour must be exactly what it was
        # before the fallback existed, bands and all.
        api_main._state["fallback_model"] = None
        assert api_main.get_forecast("ES", 24).served_by == "weather_aware"
    finally:
        api_main._state.clear()
        api_main._state.update(saved)


def test_every_response_carries_security_and_nocache_headers(client):
    """Measured on the deployed API before this existed: /health returned only
    Server/Date/Content-Type/Content-Length/vary/cf-cache-status. `no-store` is
    load-bearing rather than cosmetic here — the dashboard publishes how old the
    served frame is, and a copy held by any intermediary makes that published age wrong.
    """
    for path in ("/", "/health", "/validation", "/assets", "/forecast/ES", "/risk/ES",
                 "/shed-schedule/ES", "/docs"):
        r = client.get(path)
        assert r.headers["X-Content-Type-Options"] == "nosniff", path
        assert r.headers["Referrer-Policy"] == "no-referrer", path
        assert r.headers["X-Frame-Options"] == "DENY", path
        assert r.headers["Content-Security-Policy"] == "frame-ancestors 'none'", path
        assert r.headers["Strict-Transport-Security"] == "max-age=31536000", path
        assert r.headers["Cache-Control"] == "no-store", path


@pytest.mark.parametrize("path", ["/forecast/{z}", "/risk/{z}", "/shed-schedule/{z}"])
def test_unknown_zone_rejection_echoes_only_a_bounded_prefix(client, path):
    """A live probe sent 4,000 characters as a zone and got the entire string back in
    the 404 body. Zone codes are a closed set, so nothing that long can be a near-miss
    worth reflecting; an unbounded echo also writes caller-chosen text into logs.
    """
    from api import main as api_main

    r = client.get(path.format(z="A" * 4000))
    assert r.status_code == 404, r.text
    detail = r.json()["detail"]
    assert "A" * (api_main.MAX_ZONE_ECHO_CHARS + 1) not in detail
    assert "A" * api_main.MAX_ZONE_ECHO_CHARS in detail
    # The zone list is already public on /health, so naming it in the error costs nothing.
    assert "Known zones" in detail
    assert len(detail) < 200, detail

