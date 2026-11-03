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
