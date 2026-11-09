"""
tests/test_risk_no_weather.py — model-side contract of the weather-blind risk fallback.

Two questions, both answered without a network and without ever writing into
data/models/ or data/features/:

  1. Does the reduced feature set actually leave the blind tail scorable, where the
     incumbent's dropna(subset=FEATURE_COLS) deletes it? (the defect itself)
  2. Does the trainer keep the repo's split discipline and imbalance approach, and
     refuse to write onto the served artifact directories?

Frames are synthetic and built by hand so an assertion failure points at the code,
not at whatever the last data pull happened to look like.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from models.risk_classifier import (
    DEFAULT_FALLBACK_MIN_GAIN_HOURS, FEATURE_COLS, FEATURE_COLS_NO_WEATHER, TARGET_COL,
    WEATHER_FEATURE_COLS, blind_tail_hours, choose_risk_model, scorable_mask,
    walk_forward_split, weather_frontier,
)

H0 = pd.Timestamp("2026-01-01T00:00:00+00:00")


def zone_frame(n_weather: int, n_blind: int, zone: str = "ES", seed: int = 0) -> pd.DataFrame:
    """n_weather complete hours followed by n_blind load-only hours."""
    rng = np.random.RandomState(seed)
    rows = []
    for i in range(n_weather + n_blind):
        ts = H0 + pd.Timedelta(hours=i)
        row = {
            "timestamp": ts, "zone": zone, "load_mw": 1000.0 + 10 * rng.rand(),
            "load_lag_1h": 1000.0, "load_lag_24h": 1000.0,
            "load_roll_mean_24h": 1000.0, "load_roll_std_24h": 12.0,
            "hour_sin": np.sin(2 * np.pi * ts.hour / 24), "hour_cos": np.cos(2 * np.pi * ts.hour / 24),
            "dow_sin": np.sin(2 * np.pi * ts.dayofweek / 7), "dow_cos": np.cos(2 * np.pi * ts.dayofweek / 7),
            "month_sin": np.sin(2 * np.pi * ts.month / 12), "month_cos": np.cos(2 * np.pi * ts.month / 12),
            "is_weekend": int(ts.dayofweek >= 5),
            "capacity_mw": 1500.0, TARGET_COL: 0,
        }
        if i < n_weather:
            row.update({"temperature_c": 15.0 + rng.rand(), "humidity_pct": 60.0,
                        "wind_speed_ms": 3.0})
        else:
            row.update({c: np.nan for c in WEATHER_FEATURE_COLS})
        rows.append(row)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- feature sets

def test_reduced_set_is_the_incumbent_minus_weather_and_order_preserved():
    assert WEATHER_FEATURE_COLS == ["temperature_c", "humidity_pct", "wind_speed_ms"]
    assert [c for c in FEATURE_COLS if c in WEATHER_FEATURE_COLS] == WEATHER_FEATURE_COLS
    assert FEATURE_COLS_NO_WEATHER == [c for c in FEATURE_COLS if c not in WEATHER_FEATURE_COLS]
    # Serving must never reorder: the committed joblib records its column names and
    # lightgbm matches them positionally against what api/main.py passes in.
    assert len(FEATURE_COLS_NO_WEATHER) == len(FEATURE_COLS) - 3
    assert not set(FEATURE_COLS_NO_WEATHER) & set(WEATHER_FEATURE_COLS)


def test_capacity_like_and_label_columns_stay_out_of_both_feature_sets():
    """The leak that saturated every zone at ~0.999 was load_to_capacity_ratio being a
    copy of the label. A reduced set is still a feature set and needs the same rule."""
    for banned in ("load_to_capacity_ratio", "overload_risk_label", "capacity_mw",
                   "precip_mm", "weather_source"):
        assert banned not in FEATURE_COLS_NO_WEATHER


# ------------------------------------------------------------- blind-tail geometry

def test_incumbent_cannot_score_the_blind_tail_but_the_reduced_set_can():
    df = zone_frame(n_weather=48, n_blind=90)
    inc = scorable_mask(df, FEATURE_COLS)
    fb = scorable_mask(df, FEATURE_COLS_NO_WEATHER)
    assert inc.sum() == 48
    assert fb.sum() == 138          # the whole frame, including every appended hour
    assert not inc.iloc[48:].any()  # exactly the tail the auditor called unscorable
    assert blind_tail_hours(df) == 90.0


def test_scorable_mask_treats_a_missing_column_as_unscorable_not_keyerror():
    df = zone_frame(24, 24).drop(columns=["temperature_c"])
    assert not scorable_mask(df, FEATURE_COLS).any()
    assert scorable_mask(df, FEATURE_COLS_NO_WEATHER).all()
    assert weather_frontier(zone_frame(0, 10)) is None


def test_blind_tail_hours_zero_when_weather_matches_load():
    df = zone_frame(72, 0)
    assert blind_tail_hours(df) == 0.0
    assert weather_frontier(df) == df.timestamp.max()


# ------------------------------------------------------------ fallback selection

def test_fallback_only_answers_when_it_is_meaningfully_newer():
    df = zone_frame(n_weather=48, n_blind=90)
    mode, reason = choose_risk_model(df, has_fallback=True)
    assert mode == "no_weather_fallback"
    assert "90.0h behind" in reason

    thin = zone_frame(n_weather=48, n_blind=int(DEFAULT_FALLBACK_MIN_GAIN_HOURS) - 1)
    assert choose_risk_model(thin, has_fallback=True)[0] == "weather_aware"

    exact = zone_frame(n_weather=48, n_blind=int(DEFAULT_FALLBACK_MIN_GAIN_HOURS))
    assert choose_risk_model(exact, has_fallback=True)[0] == "no_weather_fallback"


def test_selection_degrades_to_today_behaviour_without_the_artifact():
    """An unshipped fallback must change nothing: the old stale answer, not a 500."""
    df = zone_frame(48, 90)
    mode, reason = choose_risk_model(df, has_fallback=False)
    assert mode == "weather_aware"
    assert "no weather-blind risk artifact loaded" in reason


def test_selection_survives_a_frame_with_no_scorable_row_at_all():
    """Neither artifact can answer: the route must keep its existing behaviour rather
    than inventing a mode."""
    empty = zone_frame(0, 24).drop(columns=["load_roll_mean_24h"])
    assert not scorable_mask(empty, FEATURE_COLS_NO_WEATHER).any()
    mode, reason = choose_risk_model(empty, has_fallback=True)
    assert mode == "weather_aware"
    assert reason == "no row either model can score"


# ------------------------------------------------------------------- trainer

def _tiny_frame(tmp_path: Path) -> Path:
    """A frame with enough hours to train on and a real weather frontier at the end."""
    parts = []
    for z, seed in (("ES", 0), ("GR", 1)):
        df = zone_frame(n_weather=200, n_blind=30, zone=z, seed=seed)
        rng = np.random.RandomState(seed)
        # Make the label learnable from load alone, so the metric means something.
        df[TARGET_COL] = (df["load_mw"].to_numpy() > 1006.0).astype(int)
        df.loc[df.index[:5], TARGET_COL] = 0
        df["load_roll_mean_24h"] = df.groupby("zone")["load_mw"].transform(
            lambda s: s.shift(1).rolling(24).mean())
        df["load_roll_std_24h"] = 1.0
        df["load_lag_1h"] = df.groupby("zone")["load_mw"].shift(1)
        df["load_lag_24h"] = df["load_lag_1h"]
        df["temperature_c"] = np.where(df.index % 230 < 200, 15.0 + rng.rand(), np.nan)
        df["humidity_pct"] = df["temperature_c"]
        df["wind_speed_ms"] = df["temperature_c"]
        parts.append(df)
    out = pd.concat(parts, ignore_index=True)
    assert len(out) >= 400 and out[TARGET_COL].nunique() == 2, \
        "fixture must give the trainer real rows of both classes"
    path = tmp_path / "training_features.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(path)
    return path


class StubIncumbent:
    """
    Deterministic stand-in for risk_classifier.joblib.

    Deliberately scores *lower* than StubModel on the same row (load scaled by 0.5):
    identical stubs would make every agreement metric perfect no matter what the code
    does, and the drift gate would then be untestable.
    """

    def predict_proba(self, X):
        load = np.asarray(X[FEATURE_COLS[0]], dtype=float)
        p = np.clip((load * 0.5 - 1000.0) / 20.0 + 0.30, 0.01, 0.99)
        return np.column_stack([1 - p, p])


class _FakeBooster:
    def feature_importance(self, kind):
        assert kind == "gain", "factors are written from gain importance"
        return np.ones(len(FEATURE_COLS_NO_WEATHER))


class StubModel:
    """
    Module-level on purpose: run() joblib.dumps the model whenever the ship gates
    pass, and pickle cannot reach a function-local class. Records the columns it was
    fitted on in _TRAIN_CALLS so a test can assert the trainer never handed it weather.
    """

    def __init__(self):
        self.booster_ = _FakeBooster()
        self.best_iteration_ = 10

    def fit(self, X, y, eval_set=None, callbacks=None):
        _TRAIN_CALLS.append(list(X.columns))
        return self

    def predict_proba(self, X):
        load = np.asarray(X[FEATURE_COLS_NO_WEATHER[0]], dtype=float)
        p = np.clip((load - 1000.0) / 20.0, 0.01, 0.99)
        return np.column_stack([1 - p, p])


_TRAIN_CALLS: list[list[str]] = []


def _stub_fit_lgbm(features, fit_df, val_df, log):
    assert features == FEATURE_COLS_NO_WEATHER, "fallback must not be handed weather cols"
    log("  [stub] trained")
    return StubModel().fit(fit_df[features], fit_df[TARGET_COL])


@pytest.fixture
def stub_incumbent(monkeypatch, tmp_path):
    """Replaces the real LightGBM fit so these tests stay in milliseconds and off the
    committed data; everything else (metrics, agreement, gates, writing) runs for real."""
    import joblib
    import scripts.train_risk_no_weather as t

    _TRAIN_CALLS.clear()
    monkeypatch.setattr(t, "fit_lgbm", _stub_fit_lgbm)
    path = tmp_path / "incumbent.joblib"
    joblib.dump(StubIncumbent(), path)
    return t, path


def _permissive_gates():
    return {"min_pr_auc": 0.0, "max_pr_auc_drop_pp": 100.0, "max_mean_shift": 1.0,
            "max_status_flip_rate": 1.0, "max_critical_flip_rate": 1.0,
            "max_recall_loss_pp": 100.0, "max_ece": 1.0}


def test_trainer_writes_only_to_scratch_and_reports_agreement(stub_incumbent, tmp_path):
    t, incumbent_path = stub_incumbent
    features_path = _tiny_frame(tmp_path / "in")
    out_dir = tmp_path / "scratch"

    metrics = t.run(features_path, out_dir, incumbent_path,
                    _permissive_gates(), log=lambda *a: None)

    assert _TRAIN_CALLS == [FEATURE_COLS_NO_WEATHER], "trainer must fit on exactly the reduced set"
    for key in ("test_fallback", "test_incumbent", "agreement", "blind_tail_probe",
                "verdict", "gates", "split", "imbalance"):
        assert key in metrics, f"metrics must carry {key}"
    assert metrics["imbalance"]["method"].startswith("is_unbalance")
    assert metrics["split"]["method"].startswith("walk_forward_split")
    # The property a shuffled split would break, checked directly rather than by
    # string-matching the method name: no scored hour may predate the last fitted one.
    assert pd.Timestamp(metrics["split"]["test_start"]) > \
        t.load_frame(features_path).pipe(
            lambda d: t.walk_forward_split(d)[0]["timestamp"].max())
    ag = metrics["agreement"]
    for key in ("mean_signed_shift", "status_flip_rate", "elevated_disagreement_rate",
                "critical_disagreement_rate", "spearman_rank_correlation"):
        assert key in ag
    assert ag["hours_compared"] > 0
    # The point of the whole exercise: the hours the incumbent structurally cannot
    # reach are exactly the ones the reduced model is there to score.
    probe = metrics["blind_tail_probe"]
    assert probe["scored_fraction_incumbent"] == 0.0
    assert probe["scored_fraction_fallback"] == 1.0
    assert probe["hours"] > 0

    written = sorted(p.name for p in out_dir.iterdir())
    assert t.METRICS_FILE in written
    json.loads((out_dir / t.METRICS_FILE).read_text())


@pytest.mark.parametrize("dir_attr", ["MODELS_DIR", "DATA_FEATURES_DIR"])
def test_trainer_refuses_a_served_directory_as_output(monkeypatch, tmp_path, dir_attr):
    """The one hard rule of this script: a retrain can never land on what is live."""
    import config
    import scripts.train_risk_no_weather as t

    protected = str(Path(getattr(config, dir_attr)).resolve())
    assert protected != str(tmp_path), "fixture must not hand the guard a scratch path"
    monkeypatch.setattr(sys, "argv", ["train_risk_no_weather", "--out-dir", protected])
    with pytest.raises(SystemExit, match="refusing to write"):
        t.main()
    assert not list(Path(protected).glob("risk_classifier_no_weather*"))


def test_ship_gate_blocks_on_agreement_not_just_accuracy(stub_incumbent, tmp_path):
    """
    A fallback whose score means something different across the boundary must not ship
    even when its own accuracy is fine — that drift is the worse defect.

    Proven by contrast, not by a single run: with permissive gates the same candidate
    ships, then zeroing ONLY the three agreement gates blocks it, and the failed list
    names agreement and nothing else.
    """
    t, incumbent_path = stub_incumbent
    features_path = _tiny_frame(tmp_path / "in")

    ok = t.run(features_path, tmp_path / "scratch_ok", incumbent_path,
               _permissive_gates(), log=lambda *a: None)
    assert ok["verdict"]["ship"] is True, ok["verdict"]["failed"]

    strict = dict(_permissive_gates(), max_mean_shift=0.0,
                  max_status_flip_rate=0.0, max_critical_flip_rate=0.0)
    metrics = t.run(features_path, tmp_path / "scratch_strict", incumbent_path,
                    strict, log=lambda *a: None)
    assert metrics["verdict"]["ship"] is False
    failed = metrics["verdict"]["failed"]
    assert any("score shift" in f or "flip rate" in f for f in failed), failed
    assert all("pr_auc" not in f and "ECE" not in f and "recall" not in f
               for f in failed), failed
    assert not (tmp_path / "scratch_strict" / t.MODEL_FILE).exists(), \
        "a failing candidate must not leave an artifact someone could promote by accident"


def test_walk_forward_split_never_puts_future_hours_into_train():
    df = zone_frame(200, 40)
    train, test = walk_forward_split(df, test_frac=0.2)
    assert test["timestamp"].min() > train["timestamp"].max()
    assert len(train) + len(test) == len(df)
