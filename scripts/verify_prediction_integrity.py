"""
GridSense AI — Prediction integrity check: is every served number *computed*?

Answers one question with evidence: are the values the live API returns the
output of the committed model artifacts applied to the committed feature data,
or are they stored/hard-coded constants?

For each zone it
  1. downloads the live /forecast and /risk payloads,
  2. picks the artifact set the API says it used (response field "served_by"),
  3. re-implements the recursive forecast from scratch (this file does NOT
     import models.forecasting or api.main — a shared helper would only prove
     the same code run twice),
  4. re-runs the risk classifier on the historical row the API says it scored,
  5. checks the P10/P90 band arithmetic against that profile's intervals artifact,
  6. runs perturbation and echo probes that a fabricated constant would fail.

Run against the deployed API (default) or a local server:
    python scripts/verify_prediction_integrity.py
    python scripts/verify_prediction_integrity.py --base-url http://127.0.0.1:8001

Exit code 0 = every served number is reproduced by the artifacts.
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
import joblib

ROOT = Path(__file__).resolve().parent.parent
FEATURES = ROOT / "data" / "features" / "training_features.parquet"
RISK_MODEL = ROOT / "data" / "models" / "risk_classifier.joblib"

# /forecast reports which artifact answered (field "served_by"), and each feature set
# has its own calibrated bands. Reproducing a served number therefore means reproducing
# it with the model that actually produced it, not with whichever one this script
# happens to default to.
PROFILES = {
    "weather_aware": {
        "forecast_model": ROOT / "data" / "models" / "lightgbm_load_forecast.joblib",
        "intervals": ROOT / "data" / "models" / "forecast_intervals.json",
        "needs_weather": True,
    },
    "no_weather_fallback": {
        "forecast_model": ROOT / "data" / "models" / "lightgbm_load_forecast_no_weather.joblib",
        "intervals": ROOT / "data" / "models" / "forecast_intervals_no_weather.json",
        "needs_weather": False,
    },
}

WEATHER_COLS = ["temperature_c", "humidity_pct", "wind_speed_ms"]
HORIZON = 24

results: list[tuple[str, bool, str]] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    results.append((name, ok, detail))
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f" — {detail}" if detail else ""))


def get_json(url: str) -> dict | list:
    req = urllib.request.Request(url, headers={"User-Agent": "gridsense-integrity-check"})
    with urllib.request.urlopen(req, timeout=90) as resp:
        return json.load(resp)


# --------------------------------------------------------------------------
# Independent re-implementation of the recursion.
# Deliberately written from the feature definitions, not copied from
# models/forecasting.py, so agreement is real cross-validation of both.
# --------------------------------------------------------------------------
def calendar_features(ts: pd.Timestamp) -> dict:
    h, d, m = ts.hour, ts.dayofweek, ts.month
    return {
        "hour_sin": np.sin(2 * np.pi * h / 24),
        "hour_cos": np.cos(2 * np.pi * h / 24),
        "dow_sin": np.sin(2 * np.pi * d / 7),
        "dow_cos": np.cos(2 * np.pi * d / 7),
        "month_sin": np.sin(2 * np.pi * m / 12),
        "month_cos": np.cos(2 * np.pi * m / 12),
        "is_weekend": 1 if d >= 5 else 0,
    }


def recursive_forecast(model, history: pd.DataFrame, horizon: int) -> np.ndarray:
    """
    history is gap-free hourly, ascending. Indexing convention under test:
    lag N hours == N rows back, because rows are one hour apart.
    Future weather is persistence: same hour-of-day, most recent full day back.
    """
    loads = history["load_mw"].to_numpy(dtype=float)
    n = len(loads)
    tail = np.concatenate([loads, np.full(horizon, np.nan)])
    feats = model.feature_name_
    # The fallback model never asks for weather, so its history does not carry it.
    use_weather = any(col in feats for col in WEATHER_COLS)
    weather = history[WEATHER_COLS].to_numpy(dtype=float) if use_weather else None
    preds = []
    origin = pd.Timestamp(history["timestamp"].iloc[-1])

    for step in range(1, horizon + 1):
        idx = n + step - 1
        buf = tail[:idx]
        prev24 = buf[-24:]
        prev168 = buf[-168:]
        days_back = int(np.ceil(step / 24))
        row = {
            "load_lag_1h": buf[-1],
            "load_lag_24h": buf[-24],
            "load_lag_168h": buf[-168],
            "load_roll_mean_24h": prev24.mean(),
            "load_roll_std_24h": prev24.std(ddof=1),
            "load_roll_mean_168h": prev168.mean(),
            **calendar_features(origin + pd.Timedelta(hours=step)),
        }
        if use_weather:
            # Row whose timestamp is exactly 24*days_back hours before the target
            # hour, so it carries the same hour-of-day: the persistence assumption.
            w_row = weather[n - 1 + step - 24 * days_back]
            row.update(temperature_c=w_row[0], humidity_pct=w_row[1], wind_speed_ms=w_row[2])
        p = float(model.predict(pd.DataFrame([[row[f] for f in feats]], columns=feats))[0])
        tail[idx] = p
        preds.append(p)
    return np.array(preds)


def contiguous_tail(df: pd.DataFrame) -> pd.DataFrame:
    ordered = df.sort_values("timestamp").reset_index(drop=True)
    gaps = ordered["timestamp"].diff().dt.total_seconds() / 3600.0
    start = len(ordered) - 1
    while start > 0 and abs(float(gaps.iloc[start]) - 1.0) < 1e-9:
        start -= 1
    return ordered.iloc[start:].reset_index(drop=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="https://grid-senseai-backend.fastapicloud.dev")
    ap.add_argument("--zones", default="ES,GR,IT_SOUTH")
    args = ap.parse_args()

    print(f"Artifacts: {FEATURES.name}, {RISK_MODEL.name}, "
          + ", ".join(f"{p['forecast_model'].name}/{p['intervals'].name}" for p in PROFILES.values()))
    print(f"Live API : {args.base_url}\n")

    features = pd.read_parquet(FEATURES)
    features["timestamp"] = pd.to_datetime(features["timestamp"], utc=True)
    rmodel = joblib.load(RISK_MODEL)
    loaded = {
        name: (joblib.load(p["forecast_model"]),
               json.loads(p["intervals"].read_text())["horizons"])
        for name, p in PROFILES.items()
    }
    profiles_used: list[str] = []

    max_abs_p50_err = 0.0
    band_violations = 0

    for zone in args.zones.split(","):
        print(f"=== {zone} ===")
        live_fc = get_json(f"{args.base_url}/forecast/{zone}?horizon_hours={HORIZON}")
        live_risk = get_json(f"{args.base_url}/risk/{zone}")

        profile = live_fc.get("served_by") or "weather_aware"
        if profile not in PROFILES:
            check(f"{zone}: served_by names a profile this script can reproduce",
                  False, f"API reported {profile!r}")
            continue
        fmodel, intervals = loaded[profile]
        profiles_used.append(f"{zone}={profile}")
        print(f"  served_by={profile}")

        # Match what the serving path actually reads: the fallback scores from load
        # history alone, so requiring weather rows here would compare against a
        # different origin than the one the API used.
        cols = ["timestamp", "load_mw"]
        if PROFILES[profile]["needs_weather"]:
            cols += WEATHER_COLS
        history = contiguous_tail(features[features["zone"] == zone][cols].dropna())

        check(
            f"{zone}: served origin is the true end of gap-free history",
            pd.Timestamp(live_fc["origin_timestamp"]) == history["timestamp"].iloc[-1],
            f"API {live_fc['origin_timestamp']} / local {history['timestamp'].iloc[-1]}",
        )

        mine = recursive_forecast(fmodel, history, HORIZON)
        points = [p for p in live_fc["points"] if p["step"] <= HORIZON]
        theirs = np.array([p["p50"] for p in points])
        theirs_steps = [p["step"] for p in points]
        check(
            f"{zone}: steps 1..{HORIZON} present and ordered",
            theirs_steps == list(range(1, HORIZON + 1)),
            f"{theirs_steps[0]}..{theirs_steps[-1]}" if theirs_steps else "no steps returned",
        )
        # Never compare arrays of different length: this checker once asked for 48
        # steps with a parameter the API ignored, silently got 24, and only looked
        # green because the local horizon happened to equal the truncation.
        if len(theirs) != len(mine):
            check(
                f"{zone}: served point count matches the recomputed horizon",
                False,
                f"recomputed {len(mine)} steps, API served {len(theirs)}",
            )
            continue
        err = float(np.max(np.abs(mine - theirs)))
        rel = err / float(np.mean(np.abs(theirs)))
        max_abs_p50_err = max(max_abs_p50_err, err)
        check(
            f"{zone}: independent recursion reproduces served p50",
            rel < 1e-9,
            f"max abs diff {err:.3e} MW ({rel:.2e} relative) across {HORIZON} steps",
        )
        check(
            f"{zone}: p50 is not a seasonal-naive echo",
            bool(np.all(np.abs(theirs - history['load_mw'].iloc[-1]) > 1.0)),
            f"last observed load {history['load_mw'].iloc[-1]:.1f} MW vs forecast "
            f"{theirs.min():.1f}-{theirs.max():.1f} MW",
        )

        # Band arithmetic straight from the calibration artifact.
        worst_band = 0.0
        for p in live_fc["points"]:
            q = intervals[str(min(p["step"], len(intervals)))]
            e10 = abs(p["p10"] - p["p50"] * (1 + q["q10"]))
            e90 = abs(p["p90"] - p["p50"] * (1 + q["q90"]))
            worst_band = max(worst_band, e10, e90)
        if worst_band > 1e-6:
            band_violations += 1
        check(
            f"{zone}: P10/P90 = P50*(1+q10/q90) from {PROFILES[profile]['intervals'].name}",
            worst_band < 1e-6,
            f"max deviation {worst_band:.3e} MW",
        )
        check(
            f"{zone}: bands are ordered and non-negative",
            all(p["p10"] <= p["p50"] <= p["p90"] and p["p10"] > 0 for p in live_fc["points"]),
            "p10<=p50<=p90 for all points",
        )

        # Risk: re-score the exact historical row the API claims to have used.
        snap = pd.Timestamp(live_risk["snapshot_timestamp"])
        row = features[(features["zone"] == zone) & (features["timestamp"] == snap)]
        check(
            f"{zone}: snapshot row exists in committed features",
            len(row) == 1,
            f"{snap} -> {len(row)} row(s)",
        )
        if len(row) == 1:
            cols = rmodel.feature_name_
            score = float(rmodel.predict_proba(row[cols].astype(float))[0, 1])
            check(
                f"{zone}: re-run risk classifier matches served risk_score",
                abs(score - live_risk["risk_score"]) < 1e-12,
                f"local {score:.12f} vs API {live_risk['risk_score']:.12f}",
            )
            check(
                f"{zone}: served snapshot_load_mw is the observed load of that row",
                abs(float(row["load_mw"].iloc[0]) - live_risk["snapshot_load_mw"]) < 1e-6,
                f"{float(row['load_mw'].iloc[0]):.1f} MW",
            )
            # Perturbation probe: a hard-coded score would not move at any magnitude.
            # One +10% rung is not enough — a tree ensemble routes a low-risk row to the
            # same leaf until a split boundary is crossed, and GR's served row measured
            # flat at +10% (4.5762e-05) while moving to 0.053 at +50%. Scaling every
            # load-derived feature across a ladder keeps the constant detector honest
            # without letting leaf quantization look like a stored value.
            load_cols = [c for c in cols if c.startswith("load_")]
            trajectory = [score]
            for k in (1.10, 1.50, 2.00):
                pert = row[cols].astype(float).copy()
                for c in load_cols:
                    pert[c] = pert[c] * k
                trajectory.append(float(rmodel.predict_proba(pert)[0, 1]))
            moved = [f"{k:+.0%}" for k, (a, b) in
                     zip((1.10, 1.50, 2.00), zip(trajectory[:-1], trajectory[1:]))
                     if abs(b - a) > 1e-12]
            check(
                f"{zone}: risk score responds to load perturbation",
                bool(moved),
                f"base {score:.6g} -> " + " -> ".join(f"{v:.6g}" for v in trajectory[1:])
                + (f" (moved at {', '.join(moved)})" if moved else " (frozen at every rung)"),
            )

        # Forecast sensitivity probe.
        shuffled = history.copy()
        shuffled["load_mw"] = shuffled["load_mw"] * 1.05
        pert_fc = recursive_forecast(fmodel, shuffled, HORIZON)
        check(
            f"{zone}: forecast responds to a +5% shift in history",
            float(np.mean(pert_fc - mine)) > 1.0,
            f"mean p50 moved {float(np.mean(pert_fc - mine)):+.1f} MW",
        )
        print()

    print("=== Summary ===")
    print(f"serving profiles                     : {', '.join(profiles_used) or 'none resolved'}")
    print(f"largest p50 mismatch across zones : {max_abs_p50_err:.3e} MW")
    print(f"band arithmetic violations        : {band_violations}")
    failed = [n for n, ok, _ in results if not ok]
    print(f"checks run {len(results)} | passed {len(results) - len(failed)} | failed {len(failed)}")
    if failed:
        print("FAILED: " + "; ".join(failed))
        return 1
    print(
        "VERDICT: every served forecast and risk value is reproduced by the committed "
        "LightGBM artifacts applied to the committed feature parquet — none is a stored constant."
    )
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except urllib.error.HTTPError as exc:
        print(f"HTTP {exc.code} from {exc.url}: {exc.read()[:300]!r}")
        sys.exit(2)
