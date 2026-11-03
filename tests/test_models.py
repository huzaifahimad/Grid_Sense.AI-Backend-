import json

import joblib
import numpy as np
import pytest

import models.forecasting as F
from config import MODELS_DIR
from models.risk_classifier import FEATURE_COLS as RISK_COLS
from models.train_baselines import FEATURE_COLS as FORECAST_COLS

WEATHER = set(F.WEATHER_COLS)


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
