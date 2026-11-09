"""
Accuracy ledger: replay the serving path over the most recent window of committed
actuals and score it day by day.

Why this exists: /validation proves accuracy on one frozen held-out window, computed once
at train time. That is a release gate, not an operational control. The question a live
forecaster has to answer every morning is different — *how accurate were the forecasts
issued recently, measured against the actuals that have since arrived?* — and between
retrains nothing in this repo currently answers it. This ledger does, for the model that
is actually serving users: the no-weather artifact loaded the way api/main.py's
`_load_artifacts()` loads it (data/models/lightgbm_load_forecast_no_weather.joblib +
forecast_intervals_no_weather.json + FEATURE_COLS_NO_WEATHER), replayed through
models/forecasting.recursive_forecast_batch from origins built by the existing
models/calibrate_forecast_intervals.build_origins — which is also what keeps origins whose
168h history window straddles a data gap out of the numbers.

WHAT THIS IS NOT, AND WHY THE NAME HAS TO SAY SO: it is a **backtest of the serving path
over the recent window**, not a live forecast-vs-actual join. No forecast is ever written
down when it is issued. `/forecast/{zone}` computes its points inside the request and
returns them (api/main.py `get_forecast`, the `points = [...]` build currently at
api/main.py:606-615); the only module-level state is the in-process `_state["features"]`
frame (api/main.py:94-103), overwritten from the parquet at boot by `_load_artifacts()`
and swapped wholesale by `_publish_frame()` on refresh (api/main.py:181). A scan of api/
for any write of an issued forecast finds none — no joblib.dump, to_csv, to_parquet,
open(..., 'w') or json.dump anywhere in api/main.py or api/live_refresh.py — so there is
no snapshot store to join issued forecasts to arriving actuals. What is scored below is
therefore "what the served model would have predicted from history ending at origin T",
recomputed now, not "what the API answered at T". For a product whose deliverable is an
alarm the difference matters: a real join would also capture the band and `served_by` each
user actually saw. To make this live, persist each issued response (zone,
origin_timestamp, step, p10/p50/p90, served_by) as an append-only row at request time and
score those rows once the corresponding hours land in the feature frame.

Two properties of the committed data constrain the table, and both are reported rather
than hidden. First, `build_origins` requires a weather-complete row per origin because the
horizon's actuals must exist to be scored; since NASA POWER publishes ~5.5 days late while
load trails by hours, replayable origins stop at the last published weather hour even
though the served fallback runs off load alone. Second, that same rule means the newest
requested days come back empty until their horizon of actuals exists — which is exactly
the lag a persisted-snapshot join would remove.

    python scripts/live_accuracy.py
    python -m scripts.live_accuracy --days 30
    python -m scripts.live_accuracy --days 14 --json ledger.json   # writes only there
    python -m scripts.live_accuracy --control seasonal-naive       # negative control

Like scripts/model_head_to_head.py this writes NOTHING into the repo unless --json names
a path; in particular it never touches data/models/.
"""
from __future__ import annotations

import argparse
import contextlib
import io
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Windows consoles default to cp1252 and raise UnicodeEncodeError on the signs below.
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")

import joblib
import numpy as np
import pandas as pd

from config import DATA_FEATURES_DIR, MODELS_DIR  # noqa: E402
from models.calibrate_forecast_intervals import build_origins  # noqa: E402
from models.forecasting import WEATHER_COLS, recursive_forecast_batch  # noqa: E402
from models.train_baselines import FEATURE_COLS_NO_WEATHER  # noqa: E402

HORIZON = 24                      # the longest step /forecast is routinely asked for
REPORT_STEPS = [1, 6, 12, 24]
DEFAULT_DAYS = 14
NOMINAL_COVERAGE = 0.80           # what these bands claim; under-coverage is a defect
COVERAGE_FLOOR = 0.70             # below this the alarm's uncertainty envelope misleads
MAPE_REF_PP = 2.0                 # flag a step this far above the train-time test window
CELLS_PER_DAY = 24                # hourly actuals in one day, the denominator of "complete"
# MAPE % of this same artifact on the shipped test window, read from the committed
# forecast_intervals_no_weather.json evaluation block. A reference point for "does the
# recent window behave like the window the bands were calibrated on", not a pass mark.
TEST_WINDOW_MAPE_PCT = {1: 2.168, 6: 5.421, 12: 5.302, 24: 5.558}

SERVED_MODEL_FILE = "lightgbm_load_forecast_no_weather.joblib"
SERVED_INTERVALS_FILE = "forecast_intervals_no_weather.json"
CELL_KEYS = ("preds", "actual", "seasonal", "lo", "hi")

# Requirement: the honesty cannot live only in a docstring nobody opens. This prints
# ahead of every number the ledger emits, so a screenshot of the table alone can never
# be forwarded as evidence of live accuracy.
BACKTEST_BANNER = """
================================================================================
  THIS IS A BACKTEST / REPLAY -- NOT LIVE ACCURACY.
================================================================================
  Every number below is the served model RECOMPUTED NOW over historical hours.
  It is NOT a join of forecasts the service actually issued against the actuals
  that later arrived.

  WHY IT CANNOT BE LIVE, IN ONE SENTENCE:
      a forecast is never persisted when it is issued -- there is no snapshot
      store to join predictions against once their actuals land -- so origins
      are re-derived from committed history instead of read back.

  WHAT WOULD CHANGE THAT: scripts/forecast_snapshot_store.py records each issued
  forecast at request time and scores those rows as their hours arrive; see the
  ready-to-apply api/main.py patch in that module's header. Until that patch is
  applied and the store has rows in it, `live` in a filename here means
  "recent window", not "as served".
================================================================================"""


class _SeasonalNaiveStub:
    """
    A LightGBM-shaped stand-in whose 1-step prediction is load_lag_168h — last week's same
    hour. Handed to recursive_forecast_batch, the negative control is produced by the same
    engine and the same feature row the API uses, where each step's output becomes the next
    step's lag input. So the control degrades the way a deployed model degrades when it
    learns nothing beyond the weekly cycle, instead of bypassing the machinery under test.
    """

    def predict(self, frame):
        return np.asarray(frame["load_lag_168h"], dtype=float)


def load_served_model(models_dir: Path = MODELS_DIR):
    """
    The committed artifact the API serves, loaded with api/main.py's convention:
    joblib.load of the file, no re-fit, no substitute.

    The whole value of this ledger is that it scores the thing answering users. If its
    files are absent, a freshly fitted lookalike measures a different model, so fail loudly.
    """
    model_path = models_dir / SERVED_MODEL_FILE
    intervals_path = models_dir / SERVED_INTERVALS_FILE
    missing = [p.name for p in (model_path, intervals_path) if not p.exists()]
    if missing:
        raise FileNotFoundError(
            "cannot score the served model: missing "
            + ", ".join(missing)
            + f" in {models_dir}. "
            + f"{SERVED_MODEL_FILE} is what api/main.py:_load_artifacts loads as "
            "_state['fallback_model'] and /forecast reports as served_by="
            "'no_weather_fallback'; "
            f"{SERVED_INTERVALS_FILE} holds the bands calibrated for that artifact alone. "
            "Refusing to fit a substitute — regenerate them with `python -m "
            "models.train_baselines` then `python -m models.calibrate_forecast_intervals`."
        )
    model = joblib.load(model_path)
    intervals = json.loads(intervals_path.read_text())

    # The fallback's bands belong to one feature set. Scoring the weather-aware artifact
    # under them — or vice versa — would publish coverage it never earned, which is also
    # why api/main.py drops the fallback entirely when its intervals are absent.
    art_cols = list(getattr(model, "feature_name_", []))
    if sorted(art_cols) != sorted(FEATURE_COLS_NO_WEATHER):
        raise RuntimeError(
            f"{SERVED_MODEL_FILE} carries {len(art_cols)} features {art_cols}, which is not "
            f"the serving feature set ({len(FEATURE_COLS_NO_WEATHER)} cols). The artifact has "
            "drifted from the serving path; fix the artifact rather than rescore it here."
        )
    return model, intervals, model_path, intervals_path


def band_table(intervals: dict, horizon: int) -> tuple[np.ndarray, np.ndarray]:
    """Per-step relative-error quantiles, clamped at the calibrated max horizon."""
    table = intervals["horizons"]
    max_h = max(int(k) for k in table)
    steps = [min(h, max_h) for h in range(1, horizon + 1)]
    return (np.array([table[str(s)]["q10"] for s in steps]),
            np.array([table[str(s)]["q90"] for s in steps]))


def metrics(actual: np.ndarray, pred: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> dict:
    """MAE / RMSE / MAPE / band coverage / signed bias over aligned 1-D arrays of scored cells."""
    err = pred - actual
    rel = err / actual
    covered = (actual >= lo) & (actual <= hi)
    return {
        "cells": int(len(actual)),
        "mae_mw": float(np.mean(np.abs(err))),
        "rmse_mw": float(np.sqrt(np.mean(err ** 2))),
        "mape_pct": float(np.mean(np.abs(rel)) * 100),
        "coverage_p10_p90": float(np.mean(covered)),
        "bias_signed_pct": float(np.mean(rel) * 100),
    }


def metrics_for(cells: dict, mask: np.ndarray) -> dict:
    return metrics(cells["actual"][mask], cells["preds"][mask],
                   cells["lo"][mask], cells["hi"][mask])


def _fmt(block: dict) -> dict:
    """Round at presentation only, so a day row over its steps reconciles with its own cells."""
    return {"cells": int(block["cells"]),
            "mae_mw": round(block["mae_mw"], 1),
            "rmse_mw": round(block["rmse_mw"], 1),
            "mape_pct": round(block["mape_pct"], 3),
            "coverage_p10_p90": round(block["coverage_p10_p90"], 4),
            "bias_signed_pct": round(block["bias_signed_pct"], 3)}


def actual_dates(df: pd.DataFrame) -> list[pd.Timestamp]:
    return sorted(pd.DatetimeIndex(df["timestamp"]).normalize().unique())


def window_start(df: pd.DataFrame, days: int, emit=print) -> pd.Timestamp:
    """
    Start of exactly the last `days` calendar dates that carry actuals, walking back from
    the newest date one day at a time. Contiguity is verified rather than assumed: a date
    missing from the parquet is not a day of history, and silently substituting synthetic
    future dates would report "no origins" as though the forecaster had failed instead of
    the data being absent.

    The window is `days` long and no longer. An earlier version walked back until it hit a
    gap, which on a complete feed never stops — `--days 1` then silently replayed two years.
    """
    dates = actual_dates(df)
    if len(dates) < days:
        raise ValueError(f"asked for {days} days of actuals, the parquet holds "
                         f"{len(dates)} distinct dates")
    newest = dates[-1]
    start = newest - pd.Timedelta(days=days - 1)
    expected = pd.date_range(start, newest, freq="D")
    have = set(dates)
    missing = [d for d in expected if d not in have]
    if missing:
        emit(f"[ledger] {len(missing)} of the {len(expected)} requested day(s) carry no "
             f"actuals at all ({', '.join(str(m.date()) for m in missing[:5])}"
             f"{' ...' if len(missing) > 5 else ''}); they will simply score no cells "
             "rather than being invented")
    in_window = df[(df["timestamp"] >= start) & (df["timestamp"] <= newest)]
    filled = in_window.groupby(in_window["timestamp"].dt.normalize()).size()
    emit(f"[ledger] window {start.date()} .. {newest.date()} ({days} day(s)) | "
         f"hourly rows/day min {int(filled.min())}, median {int(filled.median())}, "
         f"expected {len(sorted(df['zone'].unique())) * CELLS_PER_DAY} for a complete day")
    return start


def flatten(block: dict) -> dict:
    """(n_origins, horizon) cells as parallel 1-D arrays plus their timestamps/zones."""
    h = block["actual"].shape[1]
    origin_ns = (block["origin_ts"].tz_localize(None) if block["origin_ts"].tz is not None
                 else block["origin_ts"]).to_numpy().astype("datetime64[ns]").astype(np.int64)
    offsets = np.arange(1, h + 1) * 3_600_000_000_000
    n = len(origin_ns)
    return {"cell_ts": pd.DatetimeIndex(np.repeat(origin_ns, h)
                                        + np.tile(offsets, n)),
            "origin_ns": np.repeat(origin_ns, h),
            "step": np.tile(np.arange(1, h + 1), n),
            "zone": np.repeat(block["zone"], h),
            **{k: block[k].reshape(-1) for k in CELL_KEYS}}


def rollup(cells: dict, mask: np.ndarray, label: str, horizon: int) -> dict:
    """Metrics for one bucket of scored cells, plus the per-horizon rows drift is read from."""
    overall = _fmt(metrics_for(cells, mask))
    steps_present = sorted(set(cells["step"][mask].tolist()))
    per_step = {}
    for h in [s for s in REPORT_STEPS if s <= horizon]:
        hs = mask & (cells["step"] == h)
        if not hs.any():
            continue          # absent rather than zero — see "steps_absent"
        step = _fmt(metrics_for(cells, hs))
        step["naive_mape_pct"] = round(float(np.mean(
            np.abs(cells["actual"][hs] - cells["seasonal"][hs])
            / cells["actual"][hs]) * 100), 3)
        step["flag"] = flag(step, h)
        per_step[f"h+{h}"] = step
    return {"label": label, "cells": overall["cells"],
            "origins": int(len(np.unique(cells["origin_ns"][mask]))),
            "actual_hours_scored": len(steps_present),
            "steps_absent": [s for s in range(1, horizon + 1) if s not in steps_present],
            "overall": overall, "per_step": per_step}


def flag(step: dict, h: int) -> str:
    cov, mape = step["coverage_p10_p90"], step["mape_pct"]
    ref = TEST_WINDOW_MAPE_PCT.get(h)
    if cov < COVERAGE_FLOOR:
        return "UNDER-COVERED"
    if cov < NOMINAL_COVERAGE - 0.05:
        return "thin coverage"
    if ref is not None and mape > ref + MAPE_REF_PP:
        return "MAPE drift"
    return ""


def print_table(rows: list[dict], title: str, steps: list[int]) -> None:
    print(f"\n{title}")
    head = (f"{'bucket':<19}{'cells':>6}{'MAE MW':>9}{'RMSE MW':>10}{'MAPE%':>8}"
            + "".join(f"{'h+' + str(h) + '%':>9}" for h in steps)
            + f"{'cov80':>8}{'bias%':>8}  flag")
    print(head)
    print("-" * len(head))
    for r in rows:
        o, ps = r["overall"], r["per_step"]
        cells = "".join(f"{ps[f'h+{h}']['mape_pct']:9.2f}" if f"h+{h}" in ps else f"{'-':>9}"
                        for h in steps)
        flags = "/".join(v["flag"] for v in ps.values() if v["flag"])
        print(f"{r['label']:<19}{o['cells']:>6}{o['mae_mw']:>9.1f}{o['rmse_mw']:>10.1f}"
              f"{o['mape_pct']:>8.2f}{cells}{o['coverage_p10_p90']:>8.3f}"
              f"{o['bias_signed_pct']:>+8.2f}  {flags or '-'}")


def gap_skipped_origins(zone_df: pd.DataFrame, start, end, stride: int, horizon: int,
                        history: int = 168) -> int:
    """
    How many candidate origins in this window were dropped for straddling an hourly gap —
    the same rejection build_origins makes internally, recounted here from the timestamps
    so the ledger can print the number instead of trusting another module's log line.
    """
    d = zone_df.dropna(subset=["load_mw", *WEATHER_COLS]).sort_values("timestamp")
    ts = d["timestamp"]
    idx = np.where((ts >= start) & (ts <= end))[0]
    idx = idx[(idx >= history - 1) & (idx + horizon < len(d))][::stride]
    if not len(idx):
        return 0
    steps = (ts.diff().dt.total_seconds() / 3600.0 != 1.0).to_numpy(copy=True)
    steps[0] = True                       # a first row has no predecessor to measure against
    prefix = np.concatenate([[0], np.cumsum(steps)])
    lo, hi = idx - history + 2, idx + horizon
    return int(((prefix[hi + 1] - prefix[lo]) != 0).sum())


def run(days: int = DEFAULT_DAYS, horizon: int = HORIZON, stride: int = 1,
        control: str = "none", quiet: bool = False) -> dict:
    t0 = time.time()
    model, intervals, model_path, intervals_path = load_served_model()
    q10, q90 = band_table(intervals, horizon)
    naive = control == "seasonal-naive"

    def emit(*a) -> None:
        if not quiet:
            print(*a)

    df = pd.read_parquet(DATA_FEATURES_DIR / "training_features.parquet")
    start = window_start(df, days, emit=emit)
    end = df["timestamp"].max()
    zones = sorted(df["zone"].unique())

    if not quiet:
        print(BACKTEST_BANNER)
    emit(f"[ledger] MODE=replay/backtest | "
         f"{'seasonal-naive, negative control' if naive else 'served no-weather LightGBM'}"
         f" | last {days} day(s) of actuals | {horizon}-step recursive path | stride {stride}h")
    emit(f"[ledger] artifacts {model_path.name} + {intervals_path.name} "
         f"| nominal band coverage {intervals.get('nominal_coverage', NOMINAL_COVERAGE)}")

    # Origins are collected before anything is scored, because "the most recent N days"
    # means different things to the three feeds: ES loses hours it never had, and each zone
    # has its own weather-publication frontier. Cutting at the EARLIEST last-replayable hour
    # across zones keeps every reported day three-zone complete, instead of a rollup whose
    # MAPE is quietly computed over whichever zones happened to be replayable that day.
    candidates = {}
    for zone in zones:
        g = df[df["zone"] == zone].sort_values("timestamp").reset_index(drop=True)
        with contextlib.redirect_stdout(io.StringIO()):   # build_origins prints its own notices
            hist_l, hist_w, origin_ts, actual, seasonal = build_origins(
                g, start, end, stride=stride, horizon=horizon)
        candidates[zone] = {"hist_l": hist_l, "hist_w": hist_w, "origin_ts": origin_ts,
                            "actual": actual, "seasonal": seasonal, "frame": g,
                            "gap_skipped": gap_skipped_origins(g, start, end, stride, horizon)}

    def _last_replayable(g: pd.DataFrame) -> pd.Timestamp:
        """Newest hour that can anchor an origin: weather-complete and with `horizon` actuals after it."""
        ok = g.dropna(subset=["load_mw", *WEATHER_COLS]).sort_values("timestamp")
        if len(ok) <= horizon:
            # Must be tz-aware: this value is compared against each zone's origin_ts index,
            # and pandas raises rather than coercing when a naive bound meets an aware one.
            # Returning the tz-naive pd.Timestamp.min crashed the whole ledger the moment one
            # requested day had no weather-complete row at all.
            return pd.Timestamp.min.tz_localize("UTC")
        stamps = pd.DatetimeIndex(ok["timestamp"]).tz_localize(None)[:-horizon]
        return pd.Timestamp(stamps[-1]).tz_localize("UTC")

    cut = min([_last_replayable(c["frame"]) for c in candidates.values()]
              + [end - pd.Timedelta(hours=horizon)])
    blocks, diag = [], []
    for zone, cand in candidates.items():
        # An empty origin_ts comes back from build_origins as a NAIVE datetime64[ns] index
        # whose .tz reads None, so it cannot be compared to an aware bound at all -- pandas
        # raises on dtype, not value. Normalise when non-empty, short-circuit when empty,
        # so "no replayable origins here" surfaces as the actionable message below instead
        # of a traceback about dtypes.
        origin_ts = cand["origin_ts"]
        if len(origin_ts) == 0:
            keep = np.zeros(0, bool)
        else:
            if origin_ts.tz is None:
                origin_ts = origin_ts.tz_localize("UTC")
            keep = np.asarray(origin_ts <= cut)
        if not keep.any():
            raise SystemExit(
                f"[ledger] zone {zone} has no origin at or before the common replay cutoff "
                f"{cut} ({origin_ts.min() if len(origin_ts) else 'no origins built'} .. "
                f"{origin_ts.max() if len(origin_ts) else 'n/a'} before the cut). "
                "Widen --days; the committed parquet cannot support this window yet."
            )
        preds = recursive_forecast_batch(_SeasonalNaiveStub() if naive else model,
                                         cand["hist_l"][keep], cand["hist_w"][keep],
                                         origin_ts[keep], horizon,
                                         FEATURE_COLS_NO_WEATHER)
        g = cand["frame"]
        weather_end_zone = g.loc[g[WEATHER_COLS].notna().all(axis=1), "timestamp"].max()
        blocks.append({"zone": zone, "origin_ts": origin_ts[keep],
                       "actual": cand["actual"][keep], "seasonal": cand["seasonal"][keep],
                       "preds": preds, "lo": preds * (1 + q10), "hi": preds * (1 + q90)})
        diag.append({"zone": zone,
                     "origins_replayed": int(keep.sum()),
                     "origins_after_common_cutoff": int((~keep).sum()),
                     "gap_skipped": int(cand["gap_skipped"]),
                     "newest_actual": str(g["timestamp"].max()),
                     "last_weather_published_hour": str(weather_end_zone),
                     "newest_origin": str(cand["origin_ts"][keep].max()),
                     "load_only_hours_after_that": int(max(
                         0, (g["timestamp"].max() - weather_end_zone).total_seconds() // 3600))})
    score_cutoff = cut

    cells = flatten({**{k: np.concatenate([b[k] for b in blocks])
                        for k in ("actual", "seasonal", "preds", "lo", "hi")},
                     "origin_ts": pd.DatetimeIndex(np.concatenate(
                         [b["origin_ts"].to_numpy() for b in blocks])),
                     "zone": np.concatenate([np.full(len(b["origin_ts"]), b["zone"])
                                             for b in blocks])})
    cell_day = cells["cell_ts"].normalize().strftime("%Y-%m-%d").to_numpy()
    # A scored step belongs to the calendar day of the *actual* it is compared against, not
    # the day the origin sat in: an origin at 22:00 whose h+2 lands at 00:00 forecasts the
    # next day, and that next day is what an operator holds the forecaster to. Each cell
    # therefore lands in exactly one day bucket, so nothing is double counted.
    weather_end = df.loc[df[WEATHER_COLS].notna().all(axis=1), "timestamp"].max()

    total = rollup(cells, np.ones(len(cell_day), bool), "ALL ZONES ALL DAYS", horizon)
    per_day = [rollup(cells, cell_day == d, d, horizon) for d in sorted(set(cell_day))]
    per_zone = [rollup(cells, cells["zone"] == z, f"zone {z}", horizon) for z in zones]

    dates_in_frame = {d.strftime("%Y-%m-%d") for d in actual_dates(df)}
    window_days = [d.strftime("%Y-%m-%d") for d in
                   pd.date_range(start, end.normalize(), freq="D")]
    replayed = set(cell_day)
    waiting_for_actuals = [d for d in window_days
                           if d not in replayed and d > score_cutoff.strftime("%Y-%m-%d")[:10]]
    waiting_for_weather = [d for d in window_days
                           if d not in replayed and d not in waiting_for_actuals
                           and d > weather_end.strftime("%Y-%m-%d")[:10]]
    absent_from_frame = [d for d in window_days if d not in dates_in_frame]
    partial_days = [d for d in per_day if d["actual_hours_scored"] < CELLS_PER_DAY]

    steps_reported = [s for s in REPORT_STEPS if s <= horizon]
    print_table([total] + per_day, "=== ledger, per UTC day of the scored actual ===",
                steps_reported)
    print_table(per_zone, "=== per-zone rollup over the same cells ===", steps_reported)

    print(f"\n{'zone':<11}{'origins':>8}{'gap-skip':>9}"
          f"{'last weather hr':>22}{'load-only hrs after':>21}{'newest origin':>20}")
    print("-" * 91)
    for d in diag:
        print(f"{d['zone']:<11}{d['origins_replayed']:>8}{d['gap_skipped']:>9}"
              f"{d['last_weather_published_hour'][:16]:>22}"
              f"{d['load_only_hours_after_that']:>21}{d['newest_origin'][:16]:>20}")
    print(f"[ledger] days requested {len(window_days)}, replayed {len(per_day)}"
          + (f"; waiting on their horizon of actuals: {', '.join(waiting_for_actuals)}"
             if waiting_for_actuals else "")
          + (f"; past the published-weather frontier: {', '.join(waiting_for_weather)}"
             if waiting_for_weather else "")
          + (f"; absent from the parquet: {', '.join(absent_from_frame)}"
             if absent_from_frame else ""))
    if partial_days:
        print("[ledger] partial days: "
              + "; ".join(f"{d['label']} {d['actual_hours_scored']}/{CELLS_PER_DAY} zone-hours "
                          f"(no origin reaches h+{max(d['steps_absent'], default=0)} there)"
                          for d in partial_days))
    print(f"[ledger] why the newest rows are missing, in two measured parts: an origin needs "
          f"its own {horizon} future hours of actuals, so nothing after {score_cutoff} can be "
          f"scored at all; and build_origins additionally needs a weather-complete row at the "
          f"origin, published through {weather_end} — the "
          f"{int((end - weather_end).total_seconds() // 3600)}h beyond it is exactly the "
          "load-only tail the served fallback forecasts into. Both close on their own as the "
          "feeds arrive; neither is a bug in this ledger.")

    worst = min((d["overall"]["coverage_p10_p90"] for d in per_day), default=1.0)
    under = [d["label"] for d in per_day if d["overall"]["coverage_p10_p90"] < COVERAGE_FLOOR]
    overall_cov = total["overall"]["coverage_p10_p90"]
    # Coverage is not a secondary stat here. The band IS the alarm contract: a shed
    # decision triggers on P90, so an interval that covers 68% instead of 80% is silently
    # discarding roughly one warning in six -- it manufactures missed warnings while every
    # point-estimate metric still looks healthy. MAE alone cannot see this failure.
    print(f"[ledger] BAND COVERAGE (P10-P90) — reported because the band is the product "
          f"contract, not a garnish: nominal {NOMINAL_COVERAGE:.2f} / alarm floor "
          f"{COVERAGE_FLOOR:.2f} | observed {overall_cov:.3f} "
          f"({(overall_cov - NOMINAL_COVERAGE) * 100:+.1f} pp vs nominal)")
    print(f"[ledger]   worst day {worst:.3f} — "
          + ("; ".join(f"{u} UNDER-COVERED" for u in under) if under
             else "no scored day under-covered"))
    # A per-zone coverage spread is easy to misread as three separate calibration defects.
    # It is mostly a scale effect: the committed table holds RELATIVE-error quantiles, so the
    # same percentage is a wide MW envelope on a 26 GW zone and a narrow one on a 2 GW zone.
    # Translating the band into MW makes that visible instead of leaving a bare number.
    emit(f"[ledger] bands are relative, so h+{horizon}'s "
         f"{(q90[horizon - 1] - q10[horizon - 1]) * 100:.2f}% is a different MW envelope per zone:")
    for b in blocks:
        lvl = float(np.mean(b["preds"]))
        cov = float(metrics_for(cells, cells["zone"] == b["zone"])["coverage_p10_p90"])
        emit(f"[ledger]   {b['zone']:<10} mean served {lvl:8.0f} MW -> band half-width "
             f"~{(q90[horizon - 1] - q10[horizon - 1]) * lvl / 2:6.0f} MW at h+{horizon}"
             f" | realized coverage {cov:.3f}")
    key = f"h+{max(steps_reported)}"
    vals = [(d["label"], d["per_step"][key]["mape_pct"]) for d in per_day if key in d["per_step"]]
    if len(vals) > 1:
        slope = float(np.polyfit(range(len(vals)), [v for _, v in vals], 1)[0])
        trend = ("degrading" if slope > 0.02 else "improving" if slope < -0.02 else "flat")
        print(f"[ledger] {key} MAPE across days: min {min(v for _, v in vals):.2f}% "
              f"max {max(v for _, v in vals):.2f}% slope {slope:+.3f} pp/day ({trend}). "
              "One bad day is weather; a slope is the model going stale between retrains.")

    provenance = {
        "kind": ("negative control: seasonal-naive through the identical accounting" if naive
                 else "backtested serving path over the recent window"),
        "live_or_backtest": "backtest",
        "why_not_live": "no forecast snapshot is persisted when it is issued: /forecast "
                        "(api/main.py `get_forecast`) computes points inside the request and "
                        "returns them, and no file in api/ writes an issued forecast anywhere. "
                        "Without that store there is nothing to join arriving actuals to, so "
                        "origins are recomputed here from committed history rather than read "
                        "back.",
        "what_would_make_it_live": "append zone, origin_timestamp, step, p10/p50/p90 and "
                                   "served_by to a store at issue time, then score those rows "
                                   "against actuals as the corresponding hours land",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "runtime_seconds": round(time.time() - t0, 1),
        "requested_days": int(days),
        "day_accounting": {
            "in_window": len(window_days),
            "replayed": len(per_day),
            "waiting_for_horizon_of_actuals": waiting_for_actuals,
            "past_published_weather_frontier": waiting_for_weather,
            "absent_from_parquet": absent_from_frame,
        },
        "partial_days": {d["label"]: {"origins": d["origins"],
                                      "actual_hours_scored": d["actual_hours_scored"],
                                      "steps_absent_max": max(d["steps_absent"], default=0)}
                         for d in partial_days},
        "actuals_window": {"start": str(start), "newest_actual": str(end)},
        "replayable_origins": {
            "count": int(total["origins"]),
            "scored_cells": int(total["cells"]),
            "start": str(min(b["origin_ts"].min() for b in blocks)),
            "end": str(max(b["origin_ts"].max() for b in blocks)),
            "last_weather_published_hour": str(weather_end),
            "note": f"an origin is replayable only once its {horizon} steps of actuals and a "
                    "weather-complete row exist, so this ledger always trails wall clock",
        },
        "zones_replayed": sorted({b["zone"] for b in blocks}),
        "replayability_by_zone": diag,
        "newest_origin_scoreable_by_actuals_alone": str(score_cutoff),
        "horizon_hours": int(horizon),
        "origin_stride_hours": int(stride),
        "feature_cols": FEATURE_COLS_NO_WEATHER,
        "artifacts": {"model": model_path.name, "intervals": intervals_path.name,
                      "interval_method": intervals.get("method"),
                      "nominal_band_coverage": intervals.get("nominal_coverage")},
        "test_window_mape_pct_reference": TEST_WINDOW_MAPE_PCT,
        "total_cells": int(total["overall"]["cells"]),
    }
    return {"provenance": provenance, "overall": total, "per_day": per_day,
            "per_zone": per_zone}


def run_live(store_path: str | None = None, features_path: Path | None = None,
             compact: bool = False) -> int:
    """
    True live accuracy: forecasts the service actually issued, joined to arriving actuals.

    Refuses to fall back to the replay. A caller who asks for `--live` and silently gets
    backtest numbers would be handed the exact confusion this script's banner exists to
    prevent, so an empty or absent store is an error with a reason, not a substitution.
    """
    from scripts.forecast_snapshot_store import EMPTY_BANNER, LEDGER_BANNER, SnapshotStore

    feats = pd.read_parquet(features_path or (DATA_FEATURES_DIR / "training_features.parquet"))
    feats = feats[["timestamp", "load_mw", "zone"]]
    store = SnapshotStore(store_path) if store_path else SnapshotStore()
    res = store.join_to_actuals(feats)
    st = store.stats()
    print(f"[ledger] MODE=live | store {st['path']} exists={st['exists']} rows={st['rows']}")
    if not res.get("scored_cells"):
        print(EMPTY_BANNER)
        print(f"[ledger] rows in store: {res.get('rows_in_store', 0)}; scored cells: 0")
        print("[ledger] refusing to print the backtest under a --live flag. Apply the "
              "api/main.py patch in scripts/forecast_snapshot_store.py, serve some "
              "requests, then re-run.")
        return 2
    print(LEDGER_BANNER)
    o = res["overall"]
    print(f"[ledger] LIVE overall cells={o['cells']} MAE={o['mae_mw']}MW RMSE={o['rmse_mw']}MW "
          f"MAPE={o['mape_pct']}% cov(P10-P90)={o['coverage_p10_p90']:.3f} "
          f"bias={o['bias_signed_pct']}%")
    for z, b in res["per_zone"].items():
        print(f"[ledger]   zone {z:<10} cells={b['cells']:>6} MAE={b['mae_mw']:>7} "
              f"cov={b['coverage_p10_p90']:.3f}")
    for s, b in sorted(res["per_step"].items()):
        print(f"[ledger]   {s:<5} cells={b['cells']:>6} MAPE={b['mape_pct']:>6}% "
              f"cov={b['coverage_p10_p90']:.3f}")
    print(f"[ledger] awaiting actual: {res['awaiting_actual']} rows, oldest unjoined "
          f"{res['oldest_unjoined_age_hours']}h | artifacts seen: "
          f"{', '.join(res['artifact_ids'])}")
    if compact:
        print(f"[ledger] compact -> {store.compact(actuals=feats)}")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Serving-path accuracy ledger (backtest by default)")
    ap.add_argument("--days", type=int, default=DEFAULT_DAYS,
                    help=f"contiguous days of actuals at the tail to replay (default {DEFAULT_DAYS})")
    ap.add_argument("--horizon", type=int, default=HORIZON,
                    help=f"recursive steps forecast per origin (default {HORIZON})")
    ap.add_argument("--stride", type=int, default=1, help="hours between forecast origins")
    ap.add_argument("--control", choices=["none", "seasonal-naive"], default="none",
                    help="score seasonal-naive through the same accounting (negative control)")
    ap.add_argument("--json", metavar="PATH",
                    help="write these numbers to PATH; nothing else is written anywhere")
    ap.add_argument("--live", action="store_true",
                    help="score persisted issued forecasts instead of replaying; exits 2 if "
                         "the snapshot store has no joinable rows (never falls back)")
    ap.add_argument("--store", metavar="PATH",
                    help="snapshot store path, for use with --live")
    ap.add_argument("--compact", action="store_true",
                    help="with --live, bound the store after reporting")
    args = ap.parse_args(argv)

    if args.live:
        return run_live(store_path=args.store, compact=args.compact)

    result = run(days=args.days, horizon=args.horizon, stride=args.stride,
                 control=args.control)
    if args.json:
        out = Path(args.json)
        out.write_text(json.dumps(result, indent=1))
        print(f"[ledger] wrote {out.resolve()}")
    print("[ledger] nothing written to data/models/ or anywhere in the repo"
          + (f", except {args.json}" if args.json else "; this run is write-free"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
