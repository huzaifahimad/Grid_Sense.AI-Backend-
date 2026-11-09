"""Scratch probe (not a shipped test): measure incumbent recursive MAPE/coverage on
contiguous FUTURE windows, and contrast with a shuffled split, to derive gate thresholds."""
import sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np, pandas as pd, joblib

from config import DATA_FEATURES_DIR, MODELS_DIR
from models.train_baselines import FEATURE_COLS_NO_WEATHER, train_val_test_split
from models.calibrate_forecast_intervals import calibrate, run_window, HORIZON

df = pd.read_parquet(DATA_FEATURES_DIR / "training_features.parquet")
_, val, test = train_val_test_split(df)
model = joblib.load(MODELS_DIR / "lightgbm_load_forecast_no_weather.joblib")

art = calibrate(model, df, val, test, FEATURE_COLS_NO_WEATHER, "no-weather fallback",
                models_dir=MODELS_DIR)
ev = art["evaluation"]
print("\n=== INCUMBENT, contiguous future test window (time-ordered) ===")
print("test origins:", ev["test_origins"], "band calib window:",
      str(val.timestamp.min())[:10], "->", str(val.timestamp.max())[:10])
for h in ("24", "48"):
    s = ev["per_step"][h]
    print(f"  h+{h}: mape_recursive={s['mape_recursive_lgbm']:.3f}% "
          f"mape_seasonal_naive={s['mape_seasonal_naive']:.3f}% coverage={s['coverage']:.4f}")
print("  coverage_overall:", ev["p10_p90_coverage_overall"])

# how many origins survive per zone, and the actual horizon span covered
ts_min, ts_max = test["timestamp"].min(), test["timestamp"].max()
p, a, seas = run_window(model, df, ts_min, ts_max, stride=12,
                        feature_cols=FEATURE_COLS_NO_WEATHER)
q10 = np.array([art["horizons"][str(h)]["q10"] for h in range(1, HORIZON + 1)])
q90 = np.array([art["horizons"][str(h)]["q90"] for h in range(1, HORIZON + 1)])
lo, hi = p * (1 + q10), p * (1 + q90)
cov = (a >= lo) & (a <= hi)
mape = np.abs(a - p) / a * 100
print(f"\norigins={len(p)}  h24 mape={mape[:,23].mean():.3f} cov={cov[:,23].mean():.4f}  "
      f"h48 mape={mape[:,47].mean():.3f} cov={cov[:,47].mean():.4f}")

# --- SHUFFLED-SPLIT CONTRAST: prove the leak magnitude, not assume it ---
shuf = df.sample(frac=1.0, random_state=0).reset_index(drop=True)
n = len(shuf)
cut_tr = int(n * 0.7); cut_va = int(n * 0.85)
fit_s, val_s, test_s = shuf.iloc[:cut_tr], shuf.iloc[cut_tr:cut_va], shuf.iloc[cut_va:]
# fit on the time-ordered train part but score on the shuffled tail -> same model, dishonest rows
preds_s, actual_s, _ = None, None, None
from models.forecasting import recursive_forecast_batch, WEATHER_COLS
HIST = 168
parts = []
for _, g in shuf.groupby("zone"):
    gg = g.sort_values("timestamp").reset_index(drop=True)
    loads = gg["load_mw"].to_numpy(float)
    w = gg[WEATHER_COLS].to_numpy(float)
    tsv = gg["timestamp"]
    idx = np.where((tsv >= test_s["timestamp"].min()) & (tsv <= test_s["timestamp"].max()))[0]
    idx = idx[(idx >= HIST - 1) & (idx + HORIZON < len(gg))][::12]
    if not len(idx):
        continue
    steps = tsv.diff().dt.total_seconds() / 3600.0
    is_gap = (steps != 1.0).to_numpy(copy=True); is_gap[0] = True
    pref = np.concatenate([[0], np.cumsum(is_gap)])
    ok = (pref[idx + HORIZON + 1] - pref[idx - HIST + 2]) == 0
    idx = idx[ok]
    if not len(idx):
        continue
    parts.append((np.stack([loads[i-HIST+1:i+1] for i in idx]),
                  np.stack([w[i-HIST+1:i+1] for i in idx]),
                  pd.DatetimeIndex(tsv.iloc[idx]),
                  np.stack([loads[i+1:i+1+HORIZON] for i in idx])))
if parts:
    hl = np.concatenate([p[0] for p in parts]); hw = np.concatenate([p[1] for p in parts])
    lt = pd.DatetimeIndex(np.concatenate([p[2].to_numpy() for p in parts]))
    ac = np.concatenate([p[3] for p in parts])
    pr = recursive_forecast_batch(model, hl, hw, lt, HORIZON, FEATURE_COLS_NO_WEATHER)
    m_shuf = float((np.abs(ac - pr) / ac * 100)[:, 23].mean())
    print(f"\n=== same model, same code, rows selected by shuffled tail === h+24 mape={m_shuf:.3f}%")
    print(f"    delta vs time-ordered contiguous future = {m_shuf - mape[:,23].mean():+.3f} pp")
