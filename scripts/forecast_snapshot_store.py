"""
Forecast snapshot store — the missing piece that turns the accuracy ledger from a
backtest into a measurement of what was actually served.

WHY THIS EXISTS
---------------
`/forecast/{zone}` computes its points inside the request and returns them; nothing in
api/ ever writes an issued forecast down (verified by grep over api/main.py and
api/live_refresh.py: no joblib.dump / to_csv / to_parquet / json.dump / open(...,'w')).
So there is no record of what a user was told at time T, and "how accurate were the
forecasts we issued?" can only be answered by recomputing them from committed history —
which is a backtest, not live accuracy. scripts/live_accuracy.py has to say so in its own
header for exactly this reason.

This module is the store that would remove that sentence. Three parts:

1. `SnapshotStore.record()` — called by the API at issue time. Appends one JSONL row per
   (issued response, step): zone, origin timestamp, issued-at UTC wall clock, target hour,
   step, p10/p50/p90, which model answered (`served_by`), and a content fingerprint of the
   artifact pair that produced it.

2. `SnapshotStore.join_to_actuals()` — called by the daily job. Joins those rows against
   observed actual load on `target_timestamp` and emits MAE / RMSE / MAPE / signed bias
   plus P10-P90 band coverage per zone, per horizon step and per day. Coverage is
   first-class: the band is the alarm system's product contract, and a band that
   under-covers manufactures missed warnings, so a ledger reporting only MAE would hide
   the failure mode that matters most.

3. `SnapshotStore.compact()` — keeps the file bounded without editing history. Scored rows
   older than a retention window are replaced by aggregate summary rows in a sibling
   file; unscored rows past their expiry are counted as a feed gap. The rewrite is atomic
   (temp + os.replace) so an interrupted compaction cannot truncate the log.

WHERE IT LIVES
--------------
Default `<repo>/data/forecasts/snapshots.jsonl`, overridable with GRIDSENSE_SNAPSHOT_DIR.
It deliberately sits OUTSIDE data/models/ and data/features/, because those hold committed
artifacts the deployed service reads at boot — a runtime write there would dirty the repo
on every request and could overwrite a model. On an ephemeral PaaS filesystem the
directory does not survive restarts, which is why the layout is one env-var redirect to a
mounted volume rather than a hardcoded path, and why `stats()` reports what is actually on
disk instead of assuming.

WHY ONE ROW PER STEP RATHER THAN PER RESPONSE
---------------------------------------------
Scoring joins on the hour being predicted. A per-response blob would force the joiner to
explode horizons anyway, and would make bounded-size accounting depend on whatever
horizon each caller asked for. Per-step rows also make duplicate suppression exact: two
identical requests inside a 15-minute poll loop score the identical cell, and counting
them twice would weight a chatty client more heavily than a quiet one in the very metric
used to decide whether to retrain.

WHY THE ARTIFACT ID IS A CONTENT FINGERPRINT AND NOT A FILENAME
---------------------------------------------------------------
Retraining overwrites lightgbm_load_forecast_no_weather.joblib in place, so two rows
sharing a filename can come from different weights. Scoring a pre-retrain forecast against
post-retrain behaviour would attribute one generation's error to another — which is
precisely the confusion that makes a retrain gate meaningless.

Usage (daily job):
    python scripts/forecast_snapshot_store.py --store /tmp/snapshots.jsonl
    python -c "from scripts.forecast_snapshot_store import SnapshotStore"  # import path
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

def _resolve(path) -> Path:
    """
    Resolve a store path without letting a POSIX-looking path become stray disk litter.

    Two measured traps on this host:
      * On Windows Path("/tmp/x.jsonl") is "\\tmp\\x.jsonl", i.e. C:\\tmp\\x.jsonl — it
        works but lands outside every convention and outside the repo, so nothing cleans
        it up. Such paths are remapped onto the real temp directory, which is what the
        name means.
      * Git Bash and Python do NOT share /tmp here (each sees its own directory), so a
        file one writes the other cannot find. Tests therefore pass an explicit
        in-workspace scratch path rather than relying on /tmp at all.

    Every other path is honoured verbatim; a bare filename resolves against the cwd.
    """
    s = str(path)
    if os.name == "nt" and (s.startswith("/tmp/") or s.startswith("/var/")):
        return (Path(tempfile.gettempdir()) / Path(s).name).resolve()
    return Path(s)


# ------------------------------------------------------------------ knobs
SNAPSHOT_DIR = _resolve(os.environ.get("GRIDSENSE_SNAPSHOT_DIR",
                                       str(ROOT / "data" / "forecasts")))
SNAPSHOT_FILE = SNAPSHOT_DIR / "snapshots.jsonl"
SUMMARY_FILE = SNAPSHOT_DIR / "ledger_summary.jsonl"

# Rows older than this are eligible for compaction once scored. Hourly origins x 3 zones
# x 24 steps is ~700 rows/day, so 90 days keeps the drift slope readable and bounds the
# file to a few MB.
RETENTION_DAYS = 90
# A row still unscored after this long will never be scored: ENTSO-E publishes load
# within hours-to-days, so anything older than this is a gap in the feed, not lag.
UNSIGHTED_EXPIRY_DAYS = 14
# Suppress exact duplicates issued within this window (poll-loop chatter). A genuinely
# new origin always gets recorded — dedupe keys on the identity of the scored cell.
DUPLICATE_SUPPRESS_SECONDS = 900
MAX_BYTES_DEFAULT = 64 * 1024 * 1024      # hard ceiling before compaction is forced


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _iso(dt) -> str:
    """UTC ISO-8601 with explicit offset; accepts naive, aware, or pd.Timestamp."""
    ts = pd.Timestamp(dt)
    return (ts if ts.tz is not None else ts.tz_localize("UTC")).isoformat()


def _utc_ts(dt) -> pd.Timestamp:
    """
    Same normalization as _iso, kept as a Timestamp for arithmetic.

    pd.Timestamp raises if a tz-aware datetime is passed together with tz=, and an
    injected test clock may be either naive (meant as UTC) or aware, so the conversion
    has to branch rather than assume — silently treating a naive clock as local time
    would shift every retention boundary in compaction.
    """
    ts = pd.Timestamp(dt)
    return ts if ts.tz is not None else ts.tz_localize("UTC")


def _file_digest(path) -> str:
    p = Path(path)
    if not p.exists():
        return "<absent>"
    hh = hashlib.sha256()
    with p.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            hh.update(chunk)
    return hh.hexdigest()


def artifact_fingerprint(*paths) -> str:
    """
    Short sha256 over the sorted (name, content-hash) of the contributing artifacts.

    Never a secret and never a path: only filenames and their digests go in, so the id
    can be logged and returned to clients without leaking deployment layout. Note the
    digest is taken at call time — the API caches it once at boot so that a retrain
    mid-life cannot silently relabel forecasts already issued.
    """
    h = hashlib.sha256()
    for entry in sorted(f"{Path(x).name}|{_file_digest(x)}" for x in paths):
        h.update(entry.encode())
    return h.hexdigest()[:16]


def _key(row: dict) -> tuple:
    """Identity of a scored cell: who, from which origin, for which future hour, under which model."""
    return (row["zone"], row["origin_timestamp"], row["target_timestamp"],
            row.get("model_artifact_id", ""))


class SnapshotStore:
    """Append-only JSONL writer + actuals joiner + bounder for issued forecasts."""

    def __init__(self, path: Path | str = SNAPSHOT_FILE, now=None):
        self.path = _resolve(path)
        self._now = now or utc_now
        self._recent: dict[tuple, float] = {}   # key -> issued epoch, in-process dedupe

    # ------------------------------------------------------------------ introspection
    def stats(self) -> dict:
        if not self.path.exists():
            return {"path": str(self.path), "exists": False, "rows": 0, "bytes": 0}
        n = 0
        with self.path.open("r", encoding="utf-8") as fh:
            for line in fh:
                if line.strip():
                    n += 1
        return {"path": str(self.path), "exists": True, "rows": n,
                "bytes": self.path.stat().st_size}

    def read_rows(self) -> list[dict]:
        """Tolerant read: blank lines and a torn final line are skipped, never fatal."""
        if not self.path.exists():
            return []
        rows = []
        with self.path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(obj, dict) and obj.get("kind") != "summary":
                    rows.append(obj)
        return rows

    # ------------------------------------------------------------------ write path
    def record(self, *, zone: str, origin_timestamp, issued_at=None,
               points, model_artifact_id: str, served_by: str = "",
               horizon_hours: int | None = None,
               max_bytes: int = MAX_BYTES_DEFAULT) -> dict:
        """
        Persist one issued forecast. `points` is the ForecastResponse point list (or a
        DataFrame with step/timestamp/p10/p50/p90), so the API call site stays three lines.

        Returns {"written": n, "suppressed": m, "error": str|None} and NEVER raises: a
        forecast that cannot be logged must still reach the user. The reason surfaces to
        /health as last_error instead of as a 500.
        """
        try:
            issued_iso = _iso(issued_at if issued_at is not None else self._now())
            issued_epoch = pd.Timestamp(issued_iso).timestamp()
            df = points if isinstance(points, pd.DataFrame) else pd.DataFrame(list(points))
            missing = {"step", "timestamp", "p10", "p50", "p90"} - set(df.columns)
            if missing:
                return {"written": 0, "suppressed": 0,
                        "error": f"points missing columns {sorted(missing)}"}

            self.path.parent.mkdir(parents=True, exist_ok=True)
            if self.path.exists() and self.path.stat().st_size > max_bytes:
                return {"written": 0, "suppressed": 0,
                        "error": f"snapshot store exceeded {max_bytes} bytes; run compact()"}

            written = suppressed = 0
            with self.path.open("a", encoding="utf-8") as fh:
                for rec in df.itertuples(index=False):
                    row = {
                        "kind": "forecast",
                        "zone": str(zone),
                        "origin_timestamp": _iso(origin_timestamp),
                        "issued_at": issued_iso,
                        "target_timestamp": _iso(rec.timestamp),
                        "step": int(rec.step),
                        "horizon_hours": int(horizon_hours if horizon_hours is not None
                                            else len(df)),
                        "p10": round(float(rec.p10), 3),
                        "p50": round(float(rec.p50), 3),
                        "p90": round(float(rec.p90), 3),
                        "model_artifact_id": str(model_artifact_id),
                        "served_by": str(served_by),
                    }
                    k = _key(row)
                    prior = self._recent.get(k)
                    if prior is not None and issued_epoch - prior < DUPLICATE_SUPPRESS_SECONDS:
                        suppressed += 1
                        continue
                    self._recent[k] = issued_epoch
                    fh.write(json.dumps(row, sort_keys=True) + "\n")
                    written += 1
            return {"written": written, "suppressed": suppressed, "error": None}
        except Exception as exc:  # noqa: BLE001 - logging must never break serving
            return {"written": 0, "suppressed": 0,
                    "error": f"{type(exc).__name__}: {exc}"}

    # ------------------------------------------------------------------ join
    def join_to_actuals(self, actuals: pd.DataFrame) -> dict:
        """
        Score issued forecasts against observed load on the hour they predicted.

        `actuals` needs [timestamp, load_mw, zone]. Rows whose target hour is absent stay
        `awaiting_actual` and are COUNTED, not dropped: an unjoinable row is either normal
        publication lag or a broken feed, and only its age distribution says which.
        """
        rows = self.read_rows()
        if not rows:
            return {"kind": "live_forecast_vs_actual_join", "rows_in_store": 0,
                    "scored_cells": 0, "awaiting_actual": 0, "overall": None,
                    "per_zone": {}, "per_step": {}, "per_day": {},
                    "note": "snapshot store is empty - nothing served-and-recorded yet"}

        act = actuals.dropna(subset=["load_mw", "zone"]).copy()
        # Match on normalised UTC instants, not ISO strings — see _split_scored for why
        # string equality is the wrong identity here: it silently drops real joins.
        truth = {(z, _utc_ts(t)): v for z, t, v in
                 act[["zone", "timestamp", "load_mw"]].itertuples(index=False, name=None)}

        matched, unmatched, oldest = [], 0, 0.0
        for r in rows:
            v = truth.get((r["zone"], _utc_ts(r["target_timestamp"])))
            if v is None:
                unmatched += 1
                age = (_utc_ts(self._now())
                       - pd.Timestamp(r["issued_at"])).total_seconds() / 3600.0
                oldest = max(oldest, age)
                continue
            matched.append({**r, "actual_mw": float(v)})
        if not matched:
            return {"kind": "live_forecast_vs_actual_join", "rows_in_store": len(rows),
                    "scored_cells": 0, "awaiting_actual": unmatched,
                    "oldest_unjoined_age_hours": round(oldest, 1), "overall": None,
                    "per_zone": {}, "per_step": {}, "per_day": {},
                    "note": "no issued forecast has a matching actual hour yet"}

        j = pd.DataFrame(matched)
        j["day"] = pd.to_datetime(j["target_timestamp"]).dt.strftime("%Y-%m-%d")

        def block(g: pd.DataFrame) -> dict:
            err = (g["p50"] - g["actual_mw"]).to_numpy()
            rel = err / g["actual_mw"].to_numpy()
            cov = ((g["actual_mw"] >= g["p10"]) & (g["actual_mw"] <= g["p90"])).to_numpy()
            return {"cells": int(len(g)),
                    "mae_mw": round(float(np.mean(np.abs(err))), 1),
                    "rmse_mw": round(float(np.sqrt(np.mean(err ** 2))), 1),
                    "mape_pct": round(float(np.mean(np.abs(rel)) * 100), 3),
                    "coverage_p10_p90": round(float(np.mean(cov)), 4),
                    "bias_signed_pct": round(float(np.mean(rel) * 100), 3),
                    "distinct_origins": int(g["origin_timestamp"].nunique()),
                    "zones": sorted(g["zone"].unique().tolist())}

        return {
            "kind": "live_forecast_vs_actual_join",
            "is_live_not_backtest": True,
            "why": ("every scored row was written by the service at the moment a user was "
                    "answered, and joined to the ENTSO-E actual for the hour it predicted"),
            "computed_at": _iso(self._now()),
            "rows_in_store": len(rows),
            "scored_cells": int(len(j)),
            "awaiting_actual": int(unmatched),
            "oldest_unjoined_age_hours": round(oldest, 1),
            "window": {"first_issued": str(j["issued_at"].min()),
                       "last_issued": str(j["issued_at"].max()),
                       "first_target": str(j["target_timestamp"].min()),
                       "last_target": str(j["target_timestamp"].max())},
            "artifact_ids": sorted(j["model_artifact_id"].unique().tolist()),
            "served_by_mix": {str(k): int(v) for k, v in j["served_by"].value_counts().items()},
            "overall": block(j),
            "per_zone": {str(z): block(g) for z, g in j.groupby("zone")},
            "per_step": {f"h+{int(s)}": block(g) for s, g in j.groupby("step")},
            "per_day": {str(d): block(g) for d, g in j.groupby("day")},
        }

    # ------------------------------------------------------------------ bounding
    def compact(self, *, retention_days: int = RETENTION_DAYS,
                expiry_days: int = UNSIGHTED_EXPIRY_DAYS,
                actuals: pd.DataFrame | None = None,
                summary_path: Path | str | None = None) -> dict:
        """
        Bound the store: scored rows older than `retention_days` collapse into aggregate
        summary rows in a sibling file; joinable-but-unjoinable rows past `expiry_days`
        are counted as expired and dropped.

        If `actuals` is None nothing is deleted — without actuals a stale row and a
        merely-lagging row are indistinguishable, and an unjoinable row is evidence of a
        feed gap, so the safe default is to keep it.
        """
        rows = self.read_rows()
        if not rows:
            return {"kept": 0, "summarised": 0, "expired": 0, "summaries_written": 0}

        j = pd.DataFrame(rows)
        j["issued_ts"] = pd.to_datetime(j["issued_at"], utc=True)
        cut = _utc_ts(self._now()) - pd.Timedelta(days=retention_days)
        old, keep_rows = j[j["issued_ts"] < cut], j[j["issued_ts"] >= cut]

        summarised = expired = 0
        summaries: list[dict] = []
        if len(old) and actuals is not None:
            # POSITIONAL CONTRACT, checked by the conservation assert below:
            #   scored      - has an actual now -> collapses into summary rows
            #   waiting     - no actual yet, still inside the feed window -> KEPT
            #   past_expiry - no actual and older than expiry_days -> counted, dropped
            scored, waiting, past_expiry = _split_scored(old, actuals, self._now, expiry_days)
            summarised, expired = int(len(scored)), int(len(past_expiry))
            # Rows whose actual has simply not arrived yet are KEPT: they are the only
            # evidence a stalled ENTSO-E feed leaves behind, and deleting them would make
            # awaiting_actual fall for the wrong reason.
            keep_rows = pd.concat([keep_rows, waiting], ignore_index=True)
            for (day, zone, step), g in scored.groupby(["day", "zone", "step"]):
                summaries.append({"kind": "summary", "day": day, "zone": zone,
                                  "step": int(step), "cells": int(len(g)), **_agg(g)})
            if expired:
                # One tombstone PER ROW, not an aggregate count. Deleting past-expiry rows
                # behind a single number leaves no way to answer the question a stale feed
                # actually raises — which zone, which hour, how many in a row — and an
                # accuracy ledger whose gaps are unauditable cannot be trusted to be
                # complete. The tombstone carries just enough to reconstruct the loss.
                summaries.extend({"kind": "expired_tombstone",
                                  "zone": r["zone"], "target_timestamp": r["target_timestamp"],
                                  "step": int(r["step"]),
                                  "expiry_days": int(expiry_days)}
                                 for _, r in past_expiry.iterrows())
        elif len(old):
            keep_rows = j                    # no actuals -> delete nothing

        conserved = int(len(keep_rows)) + summarised + expired
        if conserved != len(rows):
            raise RuntimeError(
                f"compact() would lose rows: {len(rows)} in, {len(keep_rows)} kept + "
                f"{summarised} summarised + {expired} expired = {conserved} out. Refusing "
                "to rewrite the store; a ledger that silently drops issued forecasts is "
                "worse than an unbounded one.")

        # Nothing aged out -> do not touch the file. Rewriting is where the risk lives:
        # `_plain` projects onto the on-disk schema, so an unconditional rewrite would
        # strip columns a later step still needs and make a second compaction destructive
        # rather than idempotent. A no-op must stay a no-op.
        if not summaries and len(keep_rows) == len(rows):
            return {"kept": int(len(rows)), "summarised": 0, "expired": 0,
                    "summaries_written": 0, "rewrote": False}

        tombstones = [s for s in summaries if s["kind"] == "expired_tombstone"]
        covered = int(sum(s["cells"] for s in summaries if s["kind"] == "summary")) \
            + len(tombstones)
        if len(tombstones) != expired or covered != summarised + expired:
            raise RuntimeError(
                f"compact() accounting does not close: {expired} expired rows vs "
                f"{len(tombstones)} tombstones, and {covered} rows covered by summaries "
                f"against {summarised} scored + {expired} expired expected. Refusing to "
                "rewrite — an issued forecast deleted without a record would make the "
                "ledger silently incomplete.")

        text = "".join(json.dumps(_plain(r), sort_keys=True) + "\n"
                       for r in keep_rows.to_dict("records"))
        tmp = self.path.with_name(self.path.name + ".tmp")
        tmp.parent.mkdir(parents=True, exist_ok=True)
        tmp.write_text(text, encoding="utf-8")
        os.replace(tmp, self.path)           # atomic: a crash leaves the old file whole

        if summaries:
            sp = _resolve(summary_path or SUMMARY_FILE)
            sp.parent.mkdir(parents=True, exist_ok=True)
            with sp.open("a", encoding="utf-8") as fh:
                for s in summaries:
                    fh.write(json.dumps(s, sort_keys=True) + "\n")
        return {"kept": int(len(keep_rows)), "summarised": summarised,
                "expired": expired, "summaries_written": len(summaries), "rewrote": True}


STORE_SCHEMA = ("kind", "zone", "origin_timestamp", "issued_at", "target_timestamp",
               "step", "horizon_hours", "p10", "p50", "p90", "model_artifact_id",
               "served_by")


def _plain(rec: dict) -> dict:
    """
    Project one joined frame row back onto the on-disk schema.

    Whitelist, not blacklist: compact() concat-enriches rows with working columns
    (`_target_ts`, `day`, `load_mw`) and pandas fills missing ones with NaT, which is not
    JSON-serialisable. Writing it anyway would raise mid-rewrite or, if coerced naively,
    put `null` where a bound must be — so only known fields survive, each coerced to its
    natural scalar type.
    """
    out = {}
    for k in STORE_SCHEMA:
        if k not in rec or pd.isna(rec[k]):
            continue
        v = rec[k]
        if isinstance(v, (np.integer,)):
            v = int(v)
        elif isinstance(v, (np.floating,)):
            v = float(v)
        out[k] = v
    return out


def _split_scored(old: pd.DataFrame, actuals: pd.DataFrame, now,
                  expiry_days: int) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    (scored, awaiting-actual, expired-past-the-feed-window) for a batch of aged rows.

    The three-way split is the point: "no actual yet" means two different things hours
    apart — minutes old it is normal ENTSO-E publication lag and the row must survive,
    weeks old it is a hole in the feed and it is dead weight. Collapsing them into one
    bucket would either delete live evidence or grow the store without bound.

    Matching is done on normalised UTC Timestamps, never on ISO strings: re-rendering an
    actual through `_iso()` assumes whatever offset the frame happens to carry, so a row
    written by a differently-clocked caller (any test with an injected `now`) silently
    stopped joining even though its actual was present. A minute of offset mismatch is
    invisible to string equality and fatal to the ledger.
    """
    act = actuals.dropna(subset=["load_mw", "zone"]).copy()
    act["_target_ts"] = [_utc_ts(t) for t in act["timestamp"]]
    old = old.copy()
    old["_target_ts"] = [_utc_ts(t) for t in old["target_timestamp"]]
    keys = set(zip(act["zone"], act["_target_ts"]))
    hit = old.apply(lambda r: (r["zone"], r["_target_ts"]) in keys, axis=1)
    picked = old[hit]
    if picked.empty:                 # merging an empty selection loses every column
        scored = pd.DataFrame(columns=[*old.columns, "load_mw", "day"])
    else:
        scored = picked.merge(act[["zone", "_target_ts", "load_mw"]],
                              on=["zone", "_target_ts"], how="left")
        scored["day"] = pd.to_datetime(scored["_target_ts"]).dt.strftime("%Y-%m-%d")
    waiting = old[~hit]
    ref = now() if callable(now) else now
    dead_line = _utc_ts(ref) - pd.Timedelta(days=int(expiry_days))
    # Age from `target_timestamp`, not an `issued_ts` column: that one is added by
    # compact() after this helper has already been handed the frame, so relying on it
    # made the function work only when called from one specific place. The target hour
    # is what actually bounds staleness anyway — a row whose hour is months gone and
    # still unmatched is a feed hole regardless of when it was written.
    ages = [_utc_ts(t) for t in waiting["target_timestamp"]]
    too_old = pd.Series([a < dead_line for a in ages], index=waiting.index)
    return scored, waiting[~too_old], waiting[too_old]


def _agg(g: pd.DataFrame) -> dict:
    err = (g["p50"] - g["load_mw"]).to_numpy()
    rel = err / g["load_mw"].to_numpy()
    cov = ((g["load_mw"] >= g["p10"]) & (g["load_mw"] <= g["p90"])).to_numpy()
    return {"mae_mw": round(float(np.mean(np.abs(err))), 1),
            "rmse_mw": round(float(np.sqrt(np.mean(err ** 2))), 1),
            "mape_pct": round(float(np.mean(np.abs(rel)) * 100), 3),
            "coverage_p10_p90": round(float(np.mean(cov)), 4)}


LEDGER_BANNER = (
    "[snapshots] These numbers join FORECASTS ACTUALLY ISSUED BY THE SERVICE against\n"
    "[snapshots] OBSERVED ENTSO-E ACTUALS. That is true live accuracy, not a replay:\n"
    "[snapshots] every scored row below was written at the moment a user was answered.\n"
)
EMPTY_BANNER = (
    "[snapshots] NO live accuracy number can be printed: the store holds no joinable\n"
    "[snapshots] rows. Until /forecast records each issued snapshot, the only accuracy\n"
    "[snapshots] this repo can produce is scripts/live_accuracy.py's BACKTEST, for one\n"
    "[snapshots] reason - no snapshot store exists, so there is nothing to join issued\n"
    "[snapshots] predictions to once the corresponding actuals arrive.\n"
)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Join issued forecast snapshots to observed actuals")
    ap.add_argument("--store", default=str(SNAPSHOT_FILE))
    ap.add_argument("--features",
                    default=str(ROOT / "data" / "features" / "training_features.parquet"))
    ap.add_argument("--compact", action="store_true", help="bound the store after reporting")
    ap.add_argument("--out", metavar="PATH", help="write the joined ledger JSON here")
    args = ap.parse_args(argv)

    store = SnapshotStore(args.store)
    st = store.stats()
    print(f"[snapshots] store {st['path']} exists={st['exists']} "
          f"rows={st['rows']} bytes={st['bytes']}")

    feats = pd.read_parquet(args.features)[["timestamp", "load_mw", "zone"]]
    res = store.join_to_actuals(feats)
    if res.get("scored_cells"):
        print(LEDGER_BANNER)
        o = res["overall"]
        print(f"[snapshots] overall cells={o['cells']} MAE={o['mae_mw']}MW "
              f"RMSE={o['rmse_mw']}MW MAPE={o['mape_pct']}% "
              f"cov(P10-P90)={o['coverage_p10_p90']:.3f} bias={o['bias_signed_pct']}%")
        for z, b in res["per_zone"].items():
            print(f"[snapshots]   zone {z:<10} cells={b['cells']:>6} MAE={b['mae_mw']:>7} "
                  f"cov={b['coverage_p10_p90']:.3f}")
        print(f"[snapshots] awaiting actual: {res['awaiting_actual']} rows, oldest "
              f"unjoined {res['oldest_unjoined_age_hours']}h")
    else:
        print(EMPTY_BANNER)
        print(f"[snapshots] rows in store: {res.get('rows_in_store', 0)}")

    if args.compact:
        print(f"[snapshots] compact -> {store.compact(actuals=feats)}")
    if args.out:
        Path(args.out).write_text(json.dumps(res, indent=1), encoding="utf-8")
        print(f"[snapshots] wrote {Path(args.out).resolve()}")
    print("[snapshots] nothing written to data/models/ or data/features/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
