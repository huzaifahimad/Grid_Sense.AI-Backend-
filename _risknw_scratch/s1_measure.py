"""
Step 1 — MEASURE the defect mechanism (no writes anywhere, no training).

For each zone on the committed frame data/features/training_features.parquet:
  * newest hour in the frame
  * weather frontier = newest hour complete on models.risk_classifier.FEATURE_COLS
    (the set api/main.py's get_risk passes to _representative_snapshot)
  * no-weather frontier = newest hour complete on FEATURE_COLS_NO_WEATHER
  -> exactly what /risk/{zone} can reach vs what /forecast reaches today.

Also imports api.main, runs its own _representative_snapshot() against the committed
frame, and prints the resulting RiskResponse fields (snapshot_timestamp/mode/age) —
i.e. reproduces the served answer offline without touching the network or data/.
"""
import sys
from pathlib import Path
_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

import pandas as pd

from models.risk_classifier import FEATURE_COLS, FEATURE_COLS_NO_WEATHER, blind_tail_hours
from config import DATA_FEATURES_DIR

df = pd.read_parquet(DATA_FEATURES_DIR / "training_features.parquet")
df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
print(f"frame: {len(df)} rows, zones={sorted(df.zone.unique())}, "
      f"span {df.timestamp.min()} .. {df.timestamp.max()}")

rows = []
for zone, g in df.groupby("zone"):
    g = g.sort_values("timestamp")
    inc_ok = g[FEATURE_COLS].notna().all(axis=1)
    fb_ok = g[FEATURE_COLS_NO_WEATHER].notna().all(axis=1)
    newest = g.timestamp.max()
    inc_frontier = g.loc[inc_ok, "timestamp"].max() if inc_ok.any() else None
    fb_frontier = g.loc[fb_ok, "timestamp"].max() if fb_ok.any() else None
    rows.append({
        "zone": zone, "n_rows": len(g),
        "newest": newest, "n_scorable_inc": int(inc_ok.sum()),
        "weather_frontier(incumbent)": inc_frontier,
        "no_weather_frontier": fb_frontier,
        "blind_tail_h": round((newest - inc_frontier).total_seconds()/3600, 1) if inc_frontier else None,
        "gain_if_fallback_h": round((fb_frontier - inc_frontier).total_seconds()/3600, 1) if (fb_frontier and inc_frontier) else None,
        "blind_tail_hours_fn": blind_tail_hours(g),
    })
out = pd.DataFrame(rows)
pd.set_option("display.width", 250)
print(out.to_string(index=False))

# label availability in the blind tail (can the tail even be scored/verified?)
for zone, g in df.groupby("zone"):
    g = g.sort_values("timestamp")
    inc_ok = g[FEATURE_COLS].notna().all(axis=1)
    frontier = g.loc[inc_ok, "timestamp"].max()
    tail = g[g.timestamp > frontier]
    lab = tail["overload_risk_label"]
    lm = tail["load_mw"]
    print(f"{zone}: blind tail n={len(tail)}; label non-null={int(lab.notna().sum())}; "
          f"load_mw non-null={int(lm.notna().sum())}; "
          f"tail load range={lm.min():.0f}..{lm.max():.0f} MW" if len(tail) else f"{zone}: no tail")

# --- offline reproduction of the served /risk answer on the committed frame ---
print("\n=== api.main._representative_snapshot on the committed frame ===")
import joblib
import api.main as m

m._state["features"] = (df.sort_values("timestamp").groupby("zone", group_keys=False)
                        .tail(m.ROWS_PER_ZONE_BUFFER).reset_index(drop=True))
m._state["risk_model"] = joblib.load(Path(_ROOT) / "data/models/risk_classifier.joblib")
m._state["risk_factors"] = __import__("json").loads(
    (Path(_ROOT) / "data/models/risk_factors_by_bucket.json").read_text())
for zone in sorted(df.zone.unique()):
    try:
        r = m.get_risk(zone)
        print(f"{zone}: snapshot_timestamp={r.snapshot_timestamp} mode={r.snapshot_mode} "
              f"age={r.snapshot_age_hours}h score={r.risk_score:.6f} status={r.status}")
    except Exception as e:
        print(f"{zone}: EXCEPTION {type(e).__name__}: {e}")
