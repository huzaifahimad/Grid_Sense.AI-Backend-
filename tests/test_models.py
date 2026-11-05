import json

import joblib
import numpy as np
import pytest

import models.forecasting as F
from config import MODELS_DIR
from models.risk_classifier import FEATURE_COLS as RISK_COLS
from models.train_baselines import FEATURE_COLS as FORECAST_COLS

WEATHER = set(F.WEATHER_COLS)


def _trailing_window_not_served(features, window: int):
    """
    True for rows whose trailing `window` hours are not all present in the served
    frame — series warm-up or a load hole. Derived from timestamps only, so it is
    an independent read of the same fact the rolling step depends on.
    """
    import pandas as pd

    mask = pd.Series(False, index=features.index)
    hour = pd.Timedelta(hours=1)
    for _, g in features.groupby("zone"):
        g = g.sort_values("timestamp")
        ts = pd.to_datetime(g["timestamp"].to_numpy())
        lo = ts - window * hour
        hi = ts - hour
        # searchsorted needs nanoseconds; DatetimeIndex already stores those.
        counts = (np.searchsorted(ts, hi, side="right")
                  - np.searchsorted(ts, lo, side="left"))
        mask.loc[g.index] = counts < window
    return mask


def test_risk_features_exclude_label_leak():
    assert "load_to_capacity_ratio" not in RISK_COLS
    assert "overload_risk_label" not in RISK_COLS
    assert "capacity_mw" not in RISK_COLS


class _Spy:
    def predict(self, X):
        self.X = X.copy()
        return np.zeros(len(X))


@pytest.mark.parametrize("cut", [500, 9000, -1])
def test_recursive_step1_features_match_training_pipeline(features, cut):
    """Train/serve skew guard: every non-weather feature the recursive forecaster
    builds for step 1 must equal what features/build_features.py produced."""
    for zone in features["zone"].unique():
        zd = features[features["zone"] == zone].sort_values("timestamp").reset_index(drop=True)
        idx = cut if cut >= 0 else len(zd) - 1
        spy = _Spy()
        fc = F.recursive_forecast(spy, zd.iloc[:idx], 1, FORECAST_COLS)
        expected = zd.iloc[idx]
        assert fc["timestamp"].iloc[0] == expected["timestamp"]
        for col in FORECAST_COLS:
            if col in WEATHER:
                continue
            assert spy.X[col].iloc[0] == pytest.approx(float(expected[col]), abs=1e-6), (zone, col)


def test_recursive_forecast_requires_week_of_history(features):
    zd = features[features["zone"] == "ES"].sort_values("timestamp")
    with pytest.raises(ValueError):
        F.recursive_forecast(_Spy(), zd.iloc[:100], 1, FORECAST_COLS)


def test_interval_artifact_is_calibrated_and_monotone():
    data = json.loads((MODELS_DIR / "forecast_intervals.json").read_text())
    h = data["horizons"]
    assert set(h) == {str(i) for i in range(1, 49)}
    for v in h.values():
        assert v["q10"] < 0 < v["q90"]
    cov = data["evaluation"]["p10_p90_coverage_overall"]
    assert 0.75 <= cov <= 0.92, cov


def test_forecaster_beats_seasonal_naive_at_every_reported_horizon():
    ev = json.loads((MODELS_DIR / "forecast_intervals.json").read_text())["evaluation"]
    for step, row in ev["per_step"].items():
        assert row["mape_recursive_lgbm"] < row["mape_seasonal_naive"], step


def test_apply_intervals_fallback_band():
    import pandas as pd

    out = F.apply_intervals(pd.DataFrame({"step": [1], "p50": [100.0]}), None)
    assert out["p10"].iloc[0] == pytest.approx(92.0)
    assert out["p90"].iloc[0] == pytest.approx(108.0)


def test_artifacts_load_and_predict(features):
    fm = joblib.load(MODELS_DIR / "lightgbm_load_forecast.joblib")
    rm = joblib.load(MODELS_DIR / "risk_classifier.joblib")
    row = features.dropna(subset=FORECAST_COLS).tail(5)
    assert np.all(fm.predict(row[FORECAST_COLS]) > 0)
    p = rm.predict_proba(row[RISK_COLS])[:, 1]
    assert np.all((p >= 0) & (p <= 1))


def test_served_feature_table_is_real_data_not_the_stand_in(features):
    """Every served number traces back to a real pull, not synthetic_bootstrap."""
    assert set(features["load_source"].unique()) == {"entsoe"}
    # "unavailable" is the honest stamp on hours whose weather has not been published
    # yet (NASA POWER lags ~2 days); no row may be unstamped.
    assert set(features["weather_source"].unique()) <= {"nasa_power", "unavailable"}
    assert "nasa_power" in set(features["weather_source"].unique())
    assert not any("synthetic" in str(v) for v in features["load_source"].unique())
    assert bool(features["capacity_is_proxy"].all())


def test_rows_without_published_weather_are_contained_and_declared(features):
    """
    LightGBM does not raise on NaN — it routes the missing value down its learned
    default direction and returns a confident number nobody measured. NASA POWER
    publishes an hour roughly two days late, so the newest load hours arrive with no
    weather. Deleting them would delete real observations and strand the forecast
    origin further in the past, so they are kept, but they have to be:
      * confined to one contiguous tail per zone — a weather hole in the middle of
        history is a join bug, not a publication lag;
      * declared in weather_source, so /health and any caller can see them;
      * unusable by the weather-aware forecaster, which is why its feature columns are
        the thing checked for completeness, not the weather columns alone.
    """
    from models.forecasting import WEATHER_COLS

    assert not features["load_mw"].isna().any(), "served row without observed load"

    blind = features[WEATHER_COLS].isna().any(axis=1)
    assert set(features.loc[blind, "weather_source"].unique()) == {"unavailable"}

    for zone, g in features.groupby("zone"):
        zone_blind = g.sort_values("timestamp")[WEATHER_COLS].isna().any(axis=1).to_numpy()
        if zone_blind.any():
            first = int(zone_blind.argmax())
            assert zone_blind[first:].all(), (
                f"{zone}: weather reappears at row {zone_blind[first:first + 50].argmax()} "
                "after going missing — a publication lag is a tail, a join bug is not")

    # Anything the weather-aware model could be asked to consume must be complete.
    lag_warmup = features[["load_lag_1h", "load_lag_24h", "load_lag_168h"]].isna().any(axis=1)
    incomplete = features[FORECAST_COLS].isna().any(axis=1)

    # A trailing rolling window is also short whenever the served frame does not
    # carry all 168 hours behind the row: build_features() drops the unpublished
    # placeholder hours *and* the real-load rows whose lag straddles a hole.
    short_window = _trailing_window_not_served(features, window=168)
    explained = blind | lag_warmup | short_window
    assert not (incomplete & ~explained).any(), (
        f"{int((incomplete & ~explained).sum())} row(s) carry a NaN the model would read "
        "silently, with no publication lag, no warm-up and no gap behind them")

    # Exact equality is not derivable here: a row whose window lost only dropped-but-
    # observed hours was averaged over a full 168 real values and is legitimately
    # complete. Whether a hole must force NaN is a property of the rolling step itself,
    # pinned at the mechanism level in
    # test_rolling_window_is_nan_across_a_hole_never_a_partial_average.
    outside = features[~explained]
    assert not outside[FORECAST_COLS].isna().any().any(), (
        "an explained-complete row still carries NaN inputs")


def test_rolling_window_is_nan_across_a_hole_never_a_partial_average():
    """
    build_features computes the rolling windows on the explicit hourly grid, where an
    unpublished hour is an empty row. pandas rolling defaults to min_periods == window,
    so a window that straddles a hole must come out NaN instead of an average over
    whichever hours happened to be there. Lowering min_periods would let the model train
    and get scored on a shorter history than its features claim, silently.
    """
    import pandas as pd

    from features.build_features import add_lag_and_rolling_features, reindex_hourly_grid

    hours = pd.date_range("2025-01-01", periods=400, freq="h", tz="UTC")
    hole = np.arange(200, 224)
    df = pd.DataFrame({
        "timestamp": hours.drop(hours[hole]),
        "zone": "T",
        "load_mw": np.arange(len(hours) - len(hole), dtype=float),
    })
    out = add_lag_and_rolling_features(reindex_hourly_grid(df))
    ts = pd.to_datetime(out["timestamp"])
    hole_start, hole_end = hours[hole[0]], hours[hole[-1]]

    for window, col in ((168, "load_roll_mean_168h"), (24, "load_roll_mean_24h")):
        expected = ((ts < ts.min() + pd.Timedelta(hours=window))
                    | ((ts >= hole_start + pd.Timedelta(hours=1))
                       & (ts <= hole_end + pd.Timedelta(hours=window))))
        actual = out[col].isna().to_numpy()
        assert (actual == expected.to_numpy()).all(), (
            f"{col}: {int((actual != expected.to_numpy()).sum())} row(s) disagree with the "
            "window-coverage rule — a hole was averaged over instead of declared unusable")

    # Once 168 real hours have accumulated behind the hole the mean must be the plain
    # mean of those hours: this is what a partial average would get wrong.
    ref = ts[ts > hole_end + pd.Timedelta(hours=168)].min()
    row = out.loc[ts == ref].iloc[0]
    behind = out.loc[(out["timestamp"] >= ref - pd.Timedelta(hours=168))
                     & (out["timestamp"] <= ref - pd.Timedelta(hours=1)), "load_mw"]
    assert not behind.isna().any(), "reference window still contains an unpublished hour"
    assert len(behind) == 168
    assert row["load_roll_mean_168h"] == pytest.approx(float(behind.mean()))


def test_reindex_hourly_grid_makes_lag_shifts_time_correct():
    """
    groupby().shift(1) counts rows, not hours. Across a hole it silently labels an
    old value as fresh, so the grid has to be made explicit before the shift.
    """
    import pandas as pd
    from features.build_features import reindex_hourly_grid

    ts = pd.date_range("2026-01-01", periods=400, freq="h", tz="UTC")
    have = ts.delete([200, 201, 202])
    df = pd.DataFrame({"timestamp": have, "zone": "ES", "load_mw": np.arange(len(have), 0, -1)})
    naive = df.assign(load_lag_1h=df["load_mw"].shift(1))
    first_after_hole = naive[naive["timestamp"] == ts[203]]["load_lag_1h"].iloc[0]
    stale = df[df["timestamp"] == ts[199]]["load_mw"].iloc[0]
    assert first_after_hole == stale  # the bug: a 4-hour-old value labelled 1 hour old

    fixed = reindex_hourly_grid(df.assign(load_source="entsoe", weather_source="nasa_power"))
    assert len(fixed) == 400
    assert pd.DatetimeIndex(fixed["timestamp"]).is_monotonic_increasing
    assert set(fixed["timestamp"].diff().dropna()) == {pd.Timedelta(hours=1)}
    shifted = fixed.assign(load_lag_1h=fixed["load_mw"].shift(1))
    after_hole = shifted[shifted["timestamp"] == ts[203]].iloc[0]
    placeholder = shifted[shifted["timestamp"] == ts[201]].iloc[0]
    assert pd.notna(after_hole["load_mw"]) and pd.isna(after_hole["load_lag_1h"])
    assert pd.isna(placeholder["load_mw"])

    ok = shifted.dropna(subset=["load_mw", "load_lag_1h"])
    by_ts = dict(zip(fixed["timestamp"], fixed["load_mw"]))
    for _, r in ok.iterrows():
        assert r["load_lag_1h"] == by_ts[r["timestamp"] - pd.Timedelta(hours=1)]
    assert (fixed["load_source"] == "entsoe").all()
    assert (fixed["weather_source"] == "nasa_power").all()


def test_build_origins_rejects_origins_that_straddle_a_gap():
    """A conformal band calibrated on row-offset history is calibrated on the wrong hours."""
    import pandas as pd
    from models.calibrate_forecast_intervals import HISTORY, build_origins
    from models.forecasting import WEATHER_COLS

    rng = np.random.default_rng(0)
    ts = pd.date_range("2026-01-01", periods=600, freq="h", tz="UTC")
    zone_df = pd.DataFrame({"timestamp": ts, "zone": "ES", "load_mw": 1000 + rng.normal(0, 10, len(ts))})
    for c in WEATHER_COLS:
        zone_df[c] = rng.normal(20, 2, len(ts))
    zone_df = zone_df.drop(index=300).reset_index(drop=True)
    have = set(zone_df["timestamp"])

    _, _, origins, _, _ = build_origins(
        zone_df, ts[200], ts[500], stride=1, horizon=48
    )
    candidates = zone_df[(zone_df["timestamp"] >= ts[200]) & (zone_df["timestamp"] <= ts[500])]
    assert len(origins) < len(candidates)  # the guard removed straddling origins
    assert len(origins) > 0
    for t in origins:
        needed = pd.date_range(t - pd.Timedelta(hours=HISTORY - 1), t, freq="h")
        assert set(needed) <= have, t


def test_build_origins_returns_empty_shape_when_nothing_qualifies():
    import pandas as pd
    from models.calibrate_forecast_intervals import build_origins
    from models.forecasting import WEATHER_COLS

    ts = pd.date_range("2026-01-01", periods=200, freq="h", tz="UTC")
    zone_df = pd.DataFrame({"timestamp": ts, "zone": "ES", "load_mw": np.linspace(1, 2, len(ts))})
    for c in WEATHER_COLS:
        zone_df[c] = 1.0
    hist_l, hist_w, origin_ts, actual, seasonal = build_origins(zone_df, ts[0], ts[-1], stride=1)
    assert hist_l.shape == (0, 168) and hist_w.shape == (0, 168, 3)
    assert actual.shape == seasonal.shape == (0, 48)
    assert len(origin_ts) == 0
