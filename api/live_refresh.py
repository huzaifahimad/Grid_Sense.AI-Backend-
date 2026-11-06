"""
GridSense AI — request-time refresh of the served frame

Why this exists, measured rather than assumed: the API serves
data/features/training_features.parquet exactly as committed to git. Probed on
2026-11-06T10:18Z the newest hour in that file was 2026-11-04T17:00Z and the
commit that shipped it was made 2026-11-05T18:52Z, so the dashboard's own age
readout said 41.2h and grew one hour per hour of uptime. Nothing in the serving
path reached the network — that was a deliberate tradeoff ("runtime holds no
third-party credentials, so an upstream outage cannot fail the demo"), reversed
on 2026-11-06 because a dashboard labelled live and serving 41-hour-old data
fails a different way.

How it works
------------
`kick_if_due(current, publish)` is called at the top of the data routes. It
returns immediately and starts one background thread when a refresh is due, so no
request ever waits on the network; the newer frame lands on the caller's next
poll. Egress runs at most once per cache window, and a zone that returns nothing
new goes quiet for longer than a zone that gained rows.

Invariants this module holds, and why each one matters
------------------------------------------------------
* Feature parity. Lag/rolling/cyclic columns for appended hours come from the
  same functions features/build_features.py used for the committed frame,
  computed over the merged hourly series — a refreshed row must be
  indistinguishable from a trained row or the served models get inputs they
  never saw.
* Committed rows are never rewritten and their stored features are never
  recomputed. The serving frame keeps only the last 336h/zone, so recomputing
  load_roll_mean_168h over it would blank that column for about the oldest half
  of the window — exactly the rows /risk picks as its representative snapshot.
  New rows are computed *from* the committed history and grafted on alongside.
* Capacity stays frozen. capacity_mw is the per-zone p99 of load over the
  training history (measured: one distinct value per zone). Recomputing it over
  a rolling window would silently redefine overload_risk_label and invalidate
  every figure in docs/15 and /validation, so the committed value is carried
  forward and /health keeps declaring "p99-of-load proxy".
* History never gets shorter. A merged zone is adopted only if its contiguous
  hourly observed-load tail is still >= MIN_HISTORY_HOURS, the floor /forecast
  enforces. A refresh that would turn a working zone into a 503 is rejected.
* Any egress failure leaves the committed frame serving, with the reason
  reported in /health rather than swallowed.
"""

from __future__ import annotations

import logging
import os
import sys
import threading
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from config import (
    ENTSOE_API_KEY,
    ENTSOE_ZONES,
    OVERLOAD_THRESHOLD_RATIO,
    ZONE_COORDINATES,
)
from features.build_features import (
    add_cyclic_time_features,
    add_lag_and_rolling_features,
)
from models.forecasting import MIN_HISTORY_HOURS, WEATHER_COLS

log = logging.getLogger("gridsense.refresh")

# build_features writes all four; the first three are model inputs and precip_mm
# only rides along to keep the appended rows schema-identical to committed ones.
PRECIP_COL = "precip_mm"
FRAME_COLS = [*WEATHER_COLS, PRECIP_COL]

# Tripwire on an appended hour: a national bidding zone does not quadruple in one
# hour, but a source that changed units (MW -> kWh) or echoed a different
# measurement does. Serving that would push load_to_capacity_ratio past every
# label threshold, so an out-of-band tail is rejected and reported instead.
PLAUSIBLE_MIN_MULTIPLE = 0.25
PLAUSIBLE_MAX_MULTIPLE = 4.0


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return default
    try:
        value = float(raw)
    except ValueError:
        log.warning("%s=%r is not a number — using %s", name, raw, default)
        return default
    return value if value > 0 else default


@dataclass
class RefreshConfig:
    enabled: bool = True
    # A zone is only worth an egress call once its newest hour is older than this.
    # ENTSO-E publishes actual load behind wall clock, so a small number here buys
    # no fresher data and costs one request per cache window for nothing.
    stale_after_hours: float = 3.0
    cache_seconds: float = 900.0
    # After an attempt that found nothing new, wait this long before asking again.
    # Without it a zone genuinely sitting at ENTSO-E's publication frontier would
    # re-request every cache window forever.
    quiet_after_no_gain_seconds: float = 3600.0
    max_tail_hours: float = 336.0
    timeout_seconds: float = 12.0
    # Rows kept per zone after a merge. Above the 336 the API loads at startup, so
    # a refresh never drops history another endpoint is relying on, while still
    # bounding memory on a long-lived process.
    keep_rows_per_zone: int = 500
    # NASA POWER reanalysis publishes roughly two days behind, so this usually
    # cannot fill the blind tail — the no-weather fallback model is the shipped
    # answer for that. It does recover hours that were simply never pulled.
    weather_enabled: bool = True
    weather_max_days: float = 4.0

    @classmethod
    def from_env(cls) -> "RefreshConfig":
        return cls(
            enabled=_env_bool("GRIDSENSE_REFRESH", True),
            stale_after_hours=_env_float("GRIDSENSE_REFRESH_STALE_HOURS", 3.0),
            cache_seconds=_env_float("GRIDSENSE_REFRESH_CACHE_SECONDS", 900.0),
            quiet_after_no_gain_seconds=_env_float(
                "GRIDSENSE_REFRESH_QUIET_SECONDS", 3600.0),
            max_tail_hours=_env_float("GRIDSENSE_REFRESH_MAX_TAIL_HOURS", 336.0),
            timeout_seconds=_env_float("GRIDSENSE_REFRESH_TIMEOUT_SECONDS", 12.0),
            keep_rows_per_zone=int(_env_float("GRIDSENSE_REFRESH_KEEP_ROWS", 500.0)),
            weather_enabled=_env_bool("GRIDSENSE_REFRESH_WEATHER", True),
            weather_max_days=_env_float("GRIDSENSE_REFRESH_WEATHER_MAX_DAYS", 4.0),
        )


config = RefreshConfig.from_env()

_lock = threading.Lock()

_status: dict = {
    "last_attempt_at": None,      # ISO, any attempt
    "last_success_at": None,      # ISO, only when rows were actually added
    "attempts": 0,
    "rows_added_total": 0,
    "last_error": None,           # "ExceptionType: message" — never the token
    "in_flight": False,
    "zones": {},                  # zone -> per-zone bookkeeping
}


def _iso(value) -> Optional[str]:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    return pd.Timestamp(value).isoformat()


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _as_utc(value) -> pd.Timestamp:
    """
    UTC Timestamp from anything hour-shaped.

    Branching on tzinfo is required, not optional: pd.Timestamp(aware, tz="UTC")
    raises "Cannot pass a datetime or Timestamp with tzinfo with the tz
    parameter", and entsoe-py needs an *aware* value, so a naive frame and an
    aware frame both have to land on aware UTC rather than one of them crashing.
    """
    ts = pd.Timestamp(value)
    return ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")


def reset_state() -> None:
    """Re-read the environment and clear bookkeeping. Used by tests."""
    global config
    config = RefreshConfig.from_env()
    with _lock:
        _status.update({
            "last_attempt_at": None,
            "last_success_at": None,
            "attempts": 0,
            "rows_added_total": 0,
            "last_error": None,
            "in_flight": False,
            "zones": {},
        })


def status(current: Optional[pd.DataFrame] = None) -> dict:
    """Snapshot for /health: what the refresher is configured to do and what it did."""
    with _lock:
        snapshot = {
            "enabled": config.enabled,
            "stale_after_hours": config.stale_after_hours,
            "cache_seconds": config.cache_seconds,
            "last_attempt_at": _status["last_attempt_at"],
            "last_success_at": _status["last_success_at"],
            "attempts": _status["attempts"],
            "rows_added_total": _status["rows_added_total"],
            "last_error": _status["last_error"],
            "in_flight": _status["in_flight"],
            "has_api_key": bool(ENTSOE_API_KEY),
            "zones": {z: dict(v) for z, v in _status["zones"].items()},
        }
    if current is not None and len(current):
        newest = current.groupby("zone")["timestamp"].max()
        for zone, row in snapshot["zones"].items():
            if zone in newest.index:
                row["serving_newest"] = _iso(newest[zone])
    return snapshot


def _newest_by_zone(current: pd.DataFrame) -> dict:
    return {str(z): ts for z, ts in current.groupby("zone")["timestamp"].max().items()}


def _zones_due(current: pd.DataFrame, now: datetime) -> list:
    """Zones older than the staleness cutoff and outside their quiet period."""
    cutoff = timedelta(hours=config.stale_after_hours)
    due = []
    for zone, newest in _newest_by_zone(current).items():
        if now - _as_utc(newest).to_pydatetime() < cutoff:
            continue
        retry_after = (_status["zones"].get(zone) or {}).get("retry_after")
        if retry_after and now < datetime.fromisoformat(retry_after):
            continue
        due.append(zone)
    return due


# ------------------------------------------------------------
# Building the appended hours


def _contiguous_tail_hours(timestamps: pd.Series) -> int:
    """Hours covered by the longest one-hour-spaced suffix of observed load."""
    ordered = pd.to_datetime(timestamps).sort_values().dropna().reset_index(drop=True)
    if len(ordered) < 2:
        return int(len(ordered))
    gaps = ordered.diff().dt.total_seconds() / 3600.0
    start = len(gaps) - 1
    while start > 0 and abs(float(gaps.iloc[start]) - 1.0) < 1e-9:
        start -= 1
    return int(len(gaps) - start)


def _pull_weather_rows(zone: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    """Best-effort NASA POWER rows for a short window. Empty frame if unavailable."""
    coords = ZONE_COORDINATES.get(zone)
    if not coords:
        return pd.DataFrame()
    from ingestion.nasa_power_client import pull_weather

    start = pd.Timestamp(start).tz_convert("UTC").floor("h")
    end = pd.Timestamp(end).tz_convert("UTC").floor("h")
    window_start = start.floor("D")
    window_end = min(end.floor("D"), window_start + pd.Timedelta(
        days=max(0, int(config.weather_max_days) - 1)))
    if window_end < window_start:
        return pd.DataFrame()
    df = pull_weather(zone, coords["lat"], coords["lon"],
                      window_start.strftime("%Y%m%d"), window_end.strftime("%Y%m%d"),
                      timeout=int(config.timeout_seconds))
    if df is None or not len(df):
        return pd.DataFrame()
    df = df.copy()
    ts = pd.to_datetime(df["timestamp"])
    df["timestamp"] = (ts.dt.tz_localize("UTC") if ts.dt.tz is None
                       else ts.dt.tz_convert("UTC")).dt.floor("h")
    keep = [c for c in FRAME_COLS if c in df.columns]
    out = df.dropna(subset=keep, how="all")[["timestamp", *keep]].copy()
    # pull_weather turns NASA POWER's -999 sentinel into pd.NA, which leaves the
    # column object-dtype. Grafting that onto the served frame would change the
    # dtype the pickled models are scored against.
    for col in keep:
        out[col] = pd.to_numeric(out[col], errors="coerce").astype("float64")
    return out


def build_candidate(current_zone: pd.DataFrame, new_load: pd.DataFrame,
                    weather: Optional[pd.DataFrame] = None) -> tuple:
    """
    Merge appended load hours into one zone's frame.

    Returns (candidate, reason). On rejection candidate is empty and reason says
    why; the caller keeps serving current_zone in that case.
    """
    current_zone = current_zone.sort_values("timestamp")
    committed_newest = pd.Timestamp(current_zone["timestamp"].max())

    tail = new_load.dropna(subset=["load_mw"]).copy()
    if tail.empty:
        return pd.DataFrame(), "no_new_hours"
    ts = pd.to_datetime(tail["timestamp"])
    tail["timestamp"] = (ts.dt.tz_localize("UTC") if ts.dt.tz is None
                         else ts.dt.tz_convert("UTC")).dt.floor("h")
    # Committed rows win on collision: an ENTSO-E republication must not rewrite
    # history the published metrics were measured against. Only newer hours are added.
    tail = tail[tail["timestamp"] > committed_newest]
    if tail.empty:
        return pd.DataFrame(), "no_new_hours"
    tail = tail.sort_values("timestamp").drop_duplicates("timestamp", keep="last")

    if (tail["load_mw"] <= 0).any():
        n = int((tail["load_mw"] <= 0).sum())
        return pd.DataFrame(), f"implausible_load: {n} non-positive hour(s)"

    median_load = float(current_zone["load_mw"].median())
    lo, hi = PLAUSIBLE_MIN_MULTIPLE * median_load, PLAUSIBLE_MAX_MULTIPLE * median_load
    n_out = int(((tail["load_mw"] < lo) | (tail["load_mw"] > hi)).sum())
    if n_out:
        return pd.DataFrame(), (f"implausible_load: {n_out} hour(s) outside "
                                f"[{lo:.0f}, {hi:.0f}] MW around a "
                                f"{median_load:.0f} MW median")

    base = current_zone[["timestamp", "load_mw"]].copy()
    base["timestamp"] = pd.to_datetime(base["timestamp"]).dt.floor("h")
    joined = (
        pd.concat([base, tail[["timestamp", "load_mw"]]], ignore_index=True)
        .drop_duplicates("timestamp", keep="first")
        .sort_values("timestamp")
        .reset_index(drop=True)
    )
    # Explicit hourly grid, so a withheld hour becomes a real empty row instead of
    # shifting every row-indexed lag by one (the reason build_features has
    # reindex_hourly_grid and /forecast has _contiguous_hourly_tail).
    grid = pd.date_range(joined["timestamp"].min(), joined["timestamp"].max(), freq="h")
    series = joined.set_index("timestamp").reindex(grid)
    series.index.name = "timestamp"
    series = series.reset_index()
    series["zone"] = str(current_zone["zone"].iloc[0])

    # Lags and rolling windows over the merged series; the committed rows' own
    # stored values are never written back, only used as history for these.
    series = add_lag_and_rolling_features(series)
    add = series[series["timestamp"] > committed_newest].reset_index(drop=True)
    if add.empty:
        return pd.DataFrame(), "no_new_hours"

    add = add_cyclic_time_features(add)
    add["load_source"] = "entsoe"
    add["weather_source"] = "unavailable"
    for col in FRAME_COLS:
        add[col] = np.nan
    if weather is not None and len(weather):
        merged = add[["timestamp"]].merge(weather, on="timestamp", how="left")
        have = merged[WEATHER_COLS[0]].notna().to_numpy()
        for col in FRAME_COLS:
            if col in merged.columns:
                add[col] = merged[col].to_numpy()
        add["weather_source"] = np.where(have, "nasa_power", "unavailable")

    # Capacity basis frozen — see the module docstring.
    add["capacity_mw"] = float(current_zone["capacity_mw"].iloc[0])
    add["capacity_is_proxy"] = bool(current_zone["capacity_is_proxy"].iloc[0])
    add["load_to_capacity_ratio"] = add["load_mw"] / add["capacity_mw"]
    add["overload_risk_label"] = (
        add["load_to_capacity_ratio"] > OVERLOAD_THRESHOLD_RATIO).astype("int64")

    missing = [c for c in current_zone.columns if c not in add.columns]
    if missing:
        return pd.DataFrame(), f"schema_mismatch: {missing[:6]}"

    candidate = pd.concat(
        [current_zone, add[list(current_zone.columns)]], ignore_index=True
    ).sort_values("timestamp").reset_index(drop=True)

    observed = candidate.loc[candidate["load_mw"].notna(), "timestamp"]
    tail_hours = _contiguous_tail_hours(observed)
    if tail_hours < MIN_HISTORY_HOURS:
        return pd.DataFrame(), (f"short_history: contiguous tail would be "
                                f"{tail_hours}h < {MIN_HISTORY_HOURS}h")

    return candidate.tail(config.keep_rows_per_zone).reset_index(drop=True), "merged"


# ------------------------------------------------------------
# The refresh itself


def _refresh_zone(zone: str, zone_frame: pd.DataFrame, end: datetime,
                  client) -> tuple:
    """Fetch one zone's missing hours and merge. Returns (candidate_or_None, entry)."""
    entry: dict = {}
    newest = _as_utc(zone_frame["timestamp"].max())
    end = _as_utc(end)
    start = _as_utc(newest + pd.Timedelta(hours=1))
    if (end - start) > timedelta(hours=config.max_tail_hours):
        start = _as_utc(end - timedelta(hours=config.max_tail_hours))
    entry["requested_from"] = start.isoformat()
    entry["requested_to"] = end.isoformat()
    if start >= end:
        entry["outcome"] = "nothing_to_fetch"
        return None, entry

    from ingestion.entsoe_client import pull_load

    new_load = pull_load(ENTSOE_ZONES[zone], zone, start, end,
                         client=client, backfill=False, verbose=False)
    weather = pd.DataFrame()
    if config.weather_enabled and len(new_load):
        try:
            weather = _pull_weather_rows(
                zone, new_load["timestamp"].min(), new_load["timestamp"].max())
        except Exception as e:
            # Weather is optional; the appended load hours land in the frame's
            # existing "unavailable" bucket and the no-weather fallback serves them.
            entry["weather_error"] = f"{type(e).__name__}: {str(e)[:120]}"
    candidate, reason = build_candidate(
        zone_frame, new_load, weather if len(weather) else None)
    entry["outcome"] = reason
    if reason != "merged":
        return None, entry
    entry["rows_added"] = int(len(candidate) - len(zone_frame))
    entry["newest"] = _iso(candidate["timestamp"].max())
    return candidate, entry


def _replace_zones(current: pd.DataFrame, updated: list) -> pd.DataFrame:
    """Swap refreshed zones in, leave every other zone's rows untouched."""
    changed = {str(frame["zone"].iloc[0]) for frame in updated}
    kept = current[~current["zone"].isin(changed)]
    return pd.concat([kept, *updated], ignore_index=True)


def run_refresh(current: pd.DataFrame, publish: Callable[[pd.DataFrame], None],
                *, now: Optional[datetime] = None) -> dict:
    """
    Fetch and merge every due zone. Returns the per-zone outcome map.

    Never raises: on any failure the caller's frame stays published, which is the
    whole point of keeping egress off the request path.
    """
    now = now or _now()
    outcomes: dict = {}
    try:
        from ingestion.entsoe_client import get_client

        # retry_count=0 because the fallback is a committed frame, not a retry:
        # entsoe-py's default 3 retries x 10s delay would hold a thread for half a
        # minute per zone to no extra purpose.
        client = get_client(timeout=int(config.timeout_seconds), retry_count=0)
        end = now.replace(minute=0, second=0, microsecond=0)
        frames = {str(z): f for z, f in current.groupby("zone")}
        budget = timedelta(seconds=config.timeout_seconds * 3)
        updated = []

        for zone in _zones_due(current, now):
            entry: dict
            try:
                if _now() - now > budget:
                    outcomes[zone] = {"outcome": "deadline",
                                      "note": f"skipped after {budget}"
                                              f" spent on earlier zones"}
                    continue
                if zone not in frames:
                    outcomes[zone] = {"outcome": "zone_absent"}
                    continue
                candidate, entry = _refresh_zone(zone, frames[zone], end, client)
                if candidate is not None:
                    updated.append(candidate)
                    entry["retry_after"] = (now + timedelta(
                        seconds=config.cache_seconds)).isoformat()
                else:
                    entry["retry_after"] = (now + timedelta(
                        seconds=config.quiet_after_no_gain_seconds)).isoformat()
            except Exception as e:
                entry = {"outcome": "failed",
                         "last_error": f"{type(e).__name__}: {str(e)[:160]}"}
                entry["retry_after"] = (now + timedelta(
                    seconds=config.quiet_after_no_gain_seconds)).isoformat()
                log.warning("live refresh failed for zone %s: %s", zone, entry["last_error"])
            outcomes[zone] = entry

        done_at = _now()
        added = 0
        if updated:
            publish(_replace_zones(current, updated))
            added = sum(int(o.get("rows_added", 0)) for o in outcomes.values())

        with _lock:
            _status["last_attempt_at"] = done_at.isoformat()
            _status["in_flight"] = False
            _status["rows_added_total"] += added
            for zone, entry in outcomes.items():
                _status["zones"].setdefault(zone, {}).update(entry)
            if added:
                _status["last_success_at"] = done_at.isoformat()
                _status["last_error"] = None
        log.info("live refresh: %d zone(s) updated, %d row(s) added",
                 len(updated), added)
        return outcomes
    except Exception as e:
        message = f"{type(e).__name__}: {str(e)[:200]}"
        with _lock:
            _status["in_flight"] = False
            _status["last_attempt_at"] = _now().isoformat()
            _status["last_error"] = message
        log.warning("live refresh aborted, serving the committed frame: %s", message)
        return {"_fatal": {"outcome": "failed", "last_error": message}}


def _background_refresh(frame: pd.DataFrame,
                        publish: Callable[[pd.DataFrame], None]) -> None:
    """
    Thread target. Going through a module-level call rather than handing
    run_refresh itself to Thread keeps the background and synchronous paths
    resolving the same name at the same moment.
    """
    run_refresh(frame, publish)


def kick_if_due(current: Optional[pd.DataFrame],
                publish: Callable[[pd.DataFrame], None],
                *,
                now: Optional[datetime] = None,
                background: bool = True) -> None:
    """
    Start a refresh when one is due. Never blocks and never raises: the caller
    keeps serving the frame it passed in until a newer one is published.
    """
    if not config.enabled or current is None or not len(current):
        return
    if not ENTSOE_API_KEY:
        with _lock:
            if not _status["last_error"]:
                _status["last_error"] = ("disabled: ENTSOE_API_KEY is not set in "
                                         "this environment")
        return

    now = now or _now()
    with _lock:
        last_attempt = _status["last_attempt_at"]
        if last_attempt and now - datetime.fromisoformat(last_attempt) < timedelta(
                seconds=config.cache_seconds):
            return
        if _status["in_flight"]:
            return
        due = _zones_due(current, now)
        if not due:
            # Nothing stale. Move the gate forward so the next request does not
            # re-scan every hit.
            _status["last_attempt_at"] = now.isoformat()
            return
        _status["in_flight"] = True
        _status["attempts"] += 1

    frame = current.copy()
    if not background:
        run_refresh(frame, publish, now=now)
        return
    try:
        threading.Thread(target=_background_refresh, args=(frame, publish),
                         name="gridsense-live-refresh", daemon=True).start()
    except Exception as e:
        with _lock:
            _status["in_flight"] = False
            _status["last_error"] = f"{type(e).__name__}: {str(e)[:160]}"
