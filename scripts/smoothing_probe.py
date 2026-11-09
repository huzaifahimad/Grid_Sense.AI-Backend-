"""
Offline probe — does input smoothing help the served forecast?
Read-only: writes nothing into the repo and retrains no shipped artifact.

Recipe under test: "use average of last 10 values to make prediction stable, not just 1 value."
Measured as a preprocessing variant on the history the features are built from, holding
model / split / feature set / origins identical, replayed through models.forecasting.
recursive_forecast_batch exactly like scripts/model_head_to_head.py does.

V0 control (incumbent, re-measured)   V1 last-10h mean replaces load_lag_1h
V2 extra rolling-mean-10h feature     V3 whole 168h history smoothed by MA(10) first

Self-contained re-implementation of the recursive path (numerically identical to the
serving one for V0/V1/V3) so the buffered variants can be passed a widened feature list.
Nothing here writes into the repo. Run: python scripts/smoothing_probe.py  (~145 s)
"""
import os
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
os.chdir(REPO)

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import mean_absolute_percentage_error as mape_score

from config import DATA_FEATURES_DIR
from models.train_baselines import FEATURE_COLS_NO_WEATHER, TARGET_COL, train_val_test_split
from models.forecasting import WEATHER_COLS, MIN_HISTORY_HOURS, _calendar_matrix
from models.calibrate_forecast_intervals import build_origins

HORIZON = 24
STRIDE = 12
STEPS = [1, 6, 12, 24]
SMOOTH_W = 10
SEED_BOOT, N_BOOT = 42, 1000


def make_lightgbm():
    return lgb.LGBMRegressor(n_estimators=500, learning_rate=0.03, num_leaves=31,
                             max_depth=-1, min_child_samples=20, subsample=0.8,
                             colsample_bytree=0.8, random_state=42, verbosity=-1)


def fit_model(frame_fit, frame_val, cols):
    """Same mechanics as model_head_to_head.fit: early stopping on val only, never test."""
    tr = frame_fit.dropna(subset=cols + [TARGET_COL])
    va = frame_val.dropna(subset=cols + [TARGET_COL])
    m = make_lightgbm()
    m.fit(tr[cols], tr[TARGET_COL], eval_set=[(va[cols], va[TARGET_COL])],
          callbacks=[lgb.early_stopping(30, verbose=False)])
    return m, int(m.best_iteration_), len(tr), len(va)


# ---------------------------------------------------------------- recursive replay
def moving_mean_last(x, w):
    """Trailing window mean over axis 1; rows with < w valid values -> NaN."""
    s = np.cumsum(np.where(np.isnan(x), 0.0, x), axis=1)
    c = np.cumsum((~np.isnan(x)).astype(float), axis=1)
    tot = s[:, w - 1:] - np.concatenate([np.zeros((x.shape[0], 1)), s[:, :-w]], axis=1)
    cnt = c[:, w - 1:] - np.concatenate([np.zeros((x.shape[0], 1)), c[:, :-w]], axis=1)
    out = np.full(x.shape, np.nan)
    with np.errstate(invalid="ignore", divide="ignore"):
        out[:, w - 1:] = np.where(cnt == w, tot / w, np.nan)
    return out


def smooth_ma(x, w):
    """Causal trailing MA(w); the first w-1 values keep raw load so lag features stay finite."""
    mm = moving_mean_last(x, w)
    keep = np.arange(x.shape[1]) < w - 1
    return np.where(keep, x, mm)


def recursive_replay(model, loads, weather, last_ts, horizon, feature_cols,
                    mode, base_len):
    """
    Mirrors models.forecasting.recursive_forecast_batch (row-offset lags, prev24 rolling
    stats, calendar from origin ts + step) plus the variant's single change.
    buffer columns [0, base_len) = history, [base_len, base_len+horizon) = predictions.
    """
    n_origins, n_hist = loads.shape
    assert n_hist >= MIN_HISTORY_HOURS
    buf = np.concatenate([loads.astype(float), np.empty((n_origins, horizon))], axis=1)
    if mode == "V3":
        # whole history smoothed BEFORE lags/rolls are derived; predictions appended raw
        buf[:, :base_len] = smooth_ma(buf[:, :base_len], SMOOTH_W)
    out = np.empty((n_origins, horizon))
    for step in range(1, horizon + 1):
        end = n_hist + step - 1
        prev24 = buf[:, end - 24:end]
        days_back = -(-step // 24)
        w = weather[:, end - 24 * days_back, :]
        ts = pd.DatetimeIndex(last_ts) + pd.Timedelta(hours=step)
        lag1 = buf[:, end - 1]
        if mode == "V1" and end >= SMOOTH_W:
            # literal recipe: smoothed newest value replaces the raw newest hour
            lag1 = buf[:, end - SMOOTH_W:end].mean(axis=1)
        feats = {
            "load_lag_1h": lag1,
            "load_lag_24h": buf[:, end - 24],
            "load_lag_168h": buf[:, end - 168],
            "load_roll_mean_24h": prev24.mean(axis=1),
            "load_roll_std_24h": prev24.std(axis=1, ddof=1),
            "load_roll_mean_168h": buf[:, end - 168:end].mean(axis=1),
            **{c: w[:, i] for i, c in enumerate(WEATHER_COLS)},
            **_calendar_matrix(ts),
        }
        if mode == "V2":
            e = min(end, base_len)
            feats["load_roll_mean_10h"] = (buf[:, e - SMOOTH_W:e].mean(axis=1)
                                           if e >= SMOOTH_W else lag1)
        pred = model.predict(pd.DataFrame(feats)[feature_cols])
        buf[:, end] = pred
        out[:, step - 1] = pred
    return out


# ---------------------------------------------------------------- static row features
def add_roll_mean_10(df):
    out = df.sort_values(["zone", "timestamp"]).copy()
    rm = out.groupby("zone")["load_mw"].transform(lambda s: s.rolling(SMOOTH_W).mean())
    out["load_roll_mean_10h"] = rm.to_numpy()
    return out


# ---------------------------------------------------------------- measurement
def collect(origins, model, cols, mode, base_len):
    """Returns APE (n_origins, HORIZON), abs error (MW), zone labels."""
    ape, ae, zl = [], [], []
    for zone, hl, hw, lt, ac in origins:
        p = recursive_replay(model, hl, hw, lt, HORIZON, cols, mode, base_len)
        ape.append(np.abs(ac - p) / ac * 100)
        ae.append(np.abs(ac - p))
        zl.append(np.full(len(ac), zone))
    return np.concatenate(ape), np.concatenate(ae), np.concatenate(zl)


def main():
    t_all = time.time()
    df = pd.read_parquet(DATA_FEATURES_DIR / "training_features.parquet")
    fit_df, val_df, test_df = train_val_test_split(df)
    start, end = test_df["timestamp"].min(), test_df["timestamp"].max()

    # ONE fixed origin set, reused by every variant -> paired comparison
    origins = []
    for zone, g in sorted(test_df.groupby("zone")):
        hl, hw, lt, ac, _ = build_origins(g, start, end, STRIDE, HORIZON)
        origins.append((zone, hl, hw, lt, ac))
    n_or = sum(len(o[3]) for o in origins)
    print(f"[probe] {n_or} origins at stride {STRIDE}, horizon {HORIZON}, "
          f"test window {start} .. {end}", flush=True)

    results = {}
    for mode in ["V0", "V1", "V2", "V3"]:
        cols = list(FEATURE_COLS_NO_WEATHER)
        f_fit, f_val = fit_df, val_df
        if mode == "V2":
            f_fit, f_val = add_roll_mean_10(fit_df), add_roll_mean_10(val_df)
            cols = cols + ["load_roll_mean_10h"]
        t0 = time.time()
        m, best_it, n_tr, n_va = fit_model(f_fit, f_val, cols)
        # V0 control assertion: our replay must reproduce the serving path bit-for-bit
        if mode == "V0":
            from models.forecasting import recursive_forecast_batch
            z0, hl0, hw0, lt0, ac0 = origins[0]
            ref = recursive_forecast_batch(m, hl0, hw0, lt0, HORIZON, cols)
            mine = recursive_replay(m, hl0, hw0, lt0, HORIZON, cols, "V0", MIN_HISTORY_HOURS)
            mx = float(np.max(np.abs(ref - mine)))
            print(f"[probe] V0 vs shipped recursive_forecast_batch max|diff| = {mx:.3e}", flush=True)
            assert mx < 1e-9, "replay diverged from serving path"
        ape, ae, zl = collect(origins, m, cols, mode, MIN_HISTORY_HOURS)
        vv = f_val.dropna(subset=cols + [TARGET_COL])
        one = mape_score(vv[TARGET_COL], m.predict(vv[cols])) * 100
        results[mode] = {"ape": ape, "ae": ae, "zones": zl, "best_iter": best_it,
                         "fit_s": round(time.time() - t0, 1)}
        per = {f"h+{h}": round(float(ape[:, h - 1].mean()), 3) for h in STEPS}
        print(f"[probe] {mode} iter={best_it} fit={results[mode]['fit_s']}s "
              f"val_one_step={one:.3f}% | " + "  ".join(f"{k} {v:.3f}%" for k, v in per.items())
              + f" | MAE {float(ae.mean()):.1f} MW", flush=True)

    # per-zone h+24
    print("\n[probe] per-zone h+24 MAPE")
    print(f"{'variant':8s}" + "".join(f"{z:>12}" for z in sorted(set(results['V0']['zones']))))
    for mode, r in results.items():
        row = {z: float(r["ape"][r["zones"] == z][:, HORIZON - 1].mean())
               for z in sorted(set(r["zones"]))}
        r["per_zone_h24"] = row
        print(f"{mode:8s}" + "".join(f"{v:11.3f}%" for v in row.values()))

    # challenger = lowest pooled h+24 MAPE among V1..V3
    ch = min(["V1", "V2", "V3"], key=lambda k: results[k]["ape"][:, HORIZON - 1].mean())
    print(f"\n[probe] best-of-challengers = {ch}")

    # paired bootstrap over origins: delta = challenger - control (negative = smoother better)
    base = results["V0"]["ape"]
    rng = np.random.default_rng(SEED_BOOT)
    print(f"[probe] paired bootstrap over {n_or} origins, B={N_BOOT} "
          f"(delta = challenger - V0, negative = smoothed input better)")
    for mode in ["V1", "V2", "V3"]:
        tag = " <- best-of-challengers" if mode == ch else ""
        print(f"  --- {mode}{tag}")
        for h in STEPS:
            i = h - 1
            d = results[mode]["ape"][:, i] - base[:, i]
            boots = np.array([d[rng.integers(0, n_or, n_or)].mean() for _ in range(N_BOOT)])
            lo, hi = np.percentile(boots, [2.5, 97.5])
            sig = bool(lo > 0 or hi < 0)
            print(f"  h+{h:<2} V0 {base[:, i].mean():.3f}%  {mode} {results[mode]['ape'][:, i].mean():.3f}%"
                  f"  delta {d.mean():+.3f} pp  95% CI [{lo:+.3f}, {hi:+.3f}]"
                  f"  {'SIGNIFICANT' if sig else 'CI includes 0 (~noise)':22s}"
                  f"  better on {(d < 0).mean() * 100:.1f}% of origins")

    # redundancy check: correlation between the recipe's 10h mean and existing roll means
    z0, hl0 = origins[0][0], origins[0][1]
    last = hl0[:, -1]
    m10 = hl0[:, -SMOOTH_W:].mean(axis=1)
    r24 = hl0[:, -24:].mean(axis=1)
    r168 = hl0.mean(axis=1)
    print("\n[probe] redundancy of a last-10h mean with existing inputs (pooled origins)")
    for name, v in (("lag_1h(raw)", last), ("roll_mean_24h", r24), ("roll_mean_168h", r168)):
        print(f"  corr(mean10, {name:14s}) = {np.corrcoef(m10, v)[0, 1]:.4f}"
              f"   mean|diff| = {np.mean(np.abs(m10 - v)):.1f} MW")
    print(f"  mean|mean10 - lag_1h| = {np.mean(np.abs(m10 - last)):.1f} MW "
          f"({np.mean(np.abs(m10 - last)) / np.mean(last) * 100:.2f}% of level)")
    imp = None
    print(f"\n[probe] total wall {time.time() - t_all:.1f}s")


if __name__ == "__main__":
    main()
