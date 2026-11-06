"""
GridSense AI — ENTSO-E Ingestion

Pulls historical load data from the ENTSO-E Transparency Platform and writes
clean Parquet files to data/raw/. Requires ENTSOE_API_KEY set in config.py.

Usage:
    python -m ingestion.entsoe_client
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import pandas as pd
from config import ENTSOE_API_KEY, ENTSOE_ZONES, DATA_RAW_DIR

try:
    from entsoe import EntsoePandasClient
except ImportError:
    raise ImportError(
        "Missing dependency. Run: pip install entsoe-py --break-system-packages"
    )


# Ceiling on the late-publication retry pass below: one day-sized request per empty
# day, so a wholesale API outage cannot turn a pull into thousands of calls.
MAX_BACKFILL_DAYS = 120


def get_client(timeout: int | None = None,
               retry_count: int | None = None) -> "EntsoePandasClient":
    """
    Build an ENTSO-E client.

    timeout/retry_count exist for the request-time refresh path, which cannot
    inherit the offline pull's defaults: retry_count=3 with retry_delay=10 means
    a dead endpoint costs ~2 minutes before raising, and the serving process
    would rather fall back to the committed frame inside a few seconds.
    """
    if not ENTSOE_API_KEY:
        raise ValueError(
            "ENTSO-E API key not set. Set the ENTSOE_API_KEY environment variable "
            "(or paste it into config.py for local-only use — never commit it)."
        )
    kwargs = {}
    if timeout is not None:
        kwargs["timeout"] = timeout
    if retry_count is not None:
        kwargs["retry_count"] = retry_count
    return EntsoePandasClient(api_key=ENTSOE_API_KEY, **kwargs)


def pull_load(zone_code: str, country_code: str, start: pd.Timestamp, end: pd.Timestamp,
              client: "EntsoePandasClient | None" = None,
              backfill: bool = True, verbose: bool = True) -> pd.DataFrame:
    """
    Pull actual load for one zone, normalised to a gap-free-able hourly UTC series.

    backfill=False skips the late-publication retry pass below — required by the
    request-time refresh path, where one day-sized request per empty hour-span
    would blow the latency budget. verbose=False keeps that path's log readable.

    Live-probed behaviour of the SOP endpoint (2026-11-03, with a real token) that
    the original single-call version got wrong:
      * returns a *DataFrame* ("Actual Load"), not a Series;
      * resolution is 15-min for some zones (ES) and hourly for others (GR,
        IT_SOUTH) — passing that straight through made load_lag_1h mean
        "15 minutes ago" for Spain while meaning "1 hour ago" elsewhere;
      * the index is the zone's *civil* timezone with DST (+01:00 / +02:00),
        while the weather join and every lag feature assume one UTC hourly grid.
    """
    client = client if client is not None else get_client()
    if verbose:
        print(f"[entsoe] pulling load for {country_code} ({zone_code}): "
              f"{start.date()} -> {end.date()}")

    # BUG FIX (found during code review, never live-tested against the real API):
    # this previously called query_load(country_code, ...) — using the internal
    # config label (e.g. "IT_SOUTH", not a real ISO code) instead of the actual
    # EIC bidding-zone code stored in zone_code. That would have silently
    # discarded zone_code entirely and failed outright for IT_SOUTH, since
    # entsoe-py has no built-in alias for a made-up label like "IT_SOUTH" — it
    # needs the real EIC code (e.g. "10Y1001A1001A788"). Spain/Greece might have
    # accidentally worked since entsoe-py has built-in aliases for standard ISO
    # country codes, which masked this bug for those two zones specifically.
    _say = (lambda *a: print(*a)) if verbose else (lambda *a: None)
    chunks, failed = [], []
    chunk_start = start
    while chunk_start < end:
        chunk_end = min(chunk_start + pd.DateOffset(months=1) - pd.Timedelta(hours=1), end)
        try:
            result = client.query_load(zone_code, start=chunk_start, end=chunk_end)
        except Exception as e:
            failed.append((chunk_start.strftime("%Y-%m"), f"{type(e).__name__}: {str(e)[:80]}"))
            result = None
        if result is not None and len(result):
            series = result.iloc[:, 0] if hasattr(result, "columns") else result
            chunks.append(series)
        chunk_start = chunk_end + pd.Timedelta(hours=1)

    if not chunks:
        raise RuntimeError(f"ENSO-E returned no data at all for {country_code} ({zone_code})")

    # Convert before resampling: civil->UTC is injective, so the autumn DST
    # repeat hour becomes two distinct UTC hours instead of collapsing.
    combined = pd.concat(chunks).sort_index()
    combined = combined[~combined.index.duplicated(keep="first")]
    combined.index = combined.index.tz_convert("UTC")

    hourly = combined.resample("1h").mean()

    # Reindex onto the full hourly grid first, so an hour lost to a failed monthly
    # chunk becomes NaN and is retried exactly like an hour that came back empty.
    hourly = hourly.reindex(pd.date_range(hourly.index.min(), hourly.index.max(), freq="h"))

    # Actual load is republished: an hour that is empty on the day can be filled days
    # later. Measured on the 2026-11-03 pull — 58 ES / 23 GR / 23 IT_SOUTH hours came
    # back empty, and re-requesting one of those months on 2026-11-04 returned all of
    # them. Without this pass those hours stay missing forever and every lag feature
    # around them gets dropped.
    empty = hourly.index[hourly.isna()] if backfill else hourly.index[:0]
    recovered = 0
    if len(empty):
        days = pd.DatetimeIndex(empty).normalize().unique()
        if len(days) > MAX_BACKFILL_DAYS:
            _say(f"[entsoe]   {country_code}: {len(days)} empty day(s), backfill capped at "
                 f"{MAX_BACKFILL_DAYS} — {len(days) - MAX_BACKFILL_DAYS} day(s) not retried")
        for day in days[:MAX_BACKFILL_DAYS]:
            try:
                res = client.query_load(zone_code, start=day,
                                        end=day + pd.Timedelta(hours=23))
            except Exception:
                continue
            ser = res.iloc[:, 0] if hasattr(res, "columns") else res
            if not len(ser):
                continue
            ser = ser.copy()
            ser.index = ser.index.tz_convert("UTC")
            fill = ser.resample("1h").mean().reindex(hourly.index)
            gained = int((fill.notna() & hourly.isna()).sum())
            if gained:
                hourly = hourly.fillna(fill)
                recovered += gained
        if recovered:
            _say(f"[entsoe]   {country_code}: backfill recovered {recovered} late-published "
                 f"hour(s) across {min(len(days), MAX_BACKFILL_DAYS)} day request(s)")

    df = hourly.reset_index()
    df.columns = ["timestamp", "load_mw"]
    df["zone"] = country_code
    df["source"] = "entsoe"

    # Basic validation — do not silently trust the API
    n_before = len(df)
    df = df.dropna(subset=["load_mw"]).drop_duplicates(subset=["timestamp"])
    n_after = len(df)
    if n_after == 0:
        # Every requested hour came back unpublished. Normal for a tail fetch that
        # runs ahead of ENTSO-E's publication lag; must not read as a crash.
        _say(f"[entsoe]   {country_code}: no published hours in "
             f"{start.isoformat()} -> {end.isoformat()}")
        if failed:
            _say(f"[entsoe]   {len(failed)} monthly chunk(s) failed for {country_code}: "
                 f"{failed[:6]}")
        return df
    expected = int((df["timestamp"].iloc[-1] - df["timestamp"].iloc[0]) / pd.Timedelta(hours=1)) + 1
    _say(f"[entsoe]   {country_code}: {n_after} hourly rows, "
         f"{df['timestamp'].min()} -> {df['timestamp'].max()}")
    if n_before > n_after:
        _say(f"[entsoe]   dropped {n_before - n_after} empty hours for {country_code}")
    if expected - n_after > 0:
        _say(f"[entsoe]   NOTE: {expected - n_after} hour(s) missing inside the span for "
             f"{country_code} — build_features keeps them; the serving path drops "
             f"history before the last gap rather than forecasting on misaligned lags")
    if failed:
        _say(f"[entsoe]   {len(failed)} monthly chunk(s) failed for {country_code}: "
             f"{failed[:6]}")

    return df


def pull_all_zones(start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    """Pull load for every zone in config.ENTSOE_ZONES, concat, and save."""
    frames = []
    for country_code, zone_code in ENTSOE_ZONES.items():
        try:
            df = pull_load(zone_code, country_code, start, end)
            frames.append(df)
        except Exception as e:
            print(f"[entsoe]   FAILED for {country_code}: {e}")
            continue

    if not frames:
        raise RuntimeError("No zones pulled successfully — check API key and zone codes.")

    combined = pd.concat(frames, ignore_index=True)
    out_path = DATA_RAW_DIR / "entsoe_load.parquet"
    combined.to_parquet(out_path, index=False)
    print(f"[entsoe] wrote {len(combined)} rows -> {out_path}")
    return combined


if __name__ == "__main__":
    # Default: last 2 years, adjust as needed once you confirm access works
    end = pd.Timestamp.now(tz="Europe/Brussels").floor("h")
    start = end - pd.DateOffset(years=2)
    pull_all_zones(start, end)
