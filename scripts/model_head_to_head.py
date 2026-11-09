"""
Forecaster head-to-head: LightGBM (incumbent) vs XGBoost vs Random Forest, measured on
the shipped split and the shipped serving path.

Why this exists: the deployed model choice was long defensible but unmeasured —
`models/train_baselines.py` benchmarks LightGBM against persistence and seasonal-naive
only, so "we use gradient boosting" was a reasoned claim, not a measured one. This
script closes that gap and, in the same run, measures what a conventional shuffled
80/10/10 split and a shuffled 5-fold CV would have reported instead, because the
difference between those two numbers is the entire argument for the temporal split.

It writes NOTHING into the repo: no model artifact, no interval JSON, no CSV. Pass a
path to dump the numbers as JSON.

    python scripts/model_head_to_head.py
    python scripts/model_head_to_head.py --json sweep.json

XGBoost is not a declared dependency (it is absent from every requirements file), so
its row is skipped with a notice on a clean clone rather than forcing an install.

Rough cost: a few minutes, single process, no network.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold, TimeSeriesSplit
from sklearn.metrics import mean_absolute_percentage_error as mape_score

from config import DATA_FEATURES_DIR
from models.train_baselines import (
    FEATURE_COLS, FEATURE_COLS_NO_WEATHER, TARGET_COL, train_val_test_split)
from models.forecasting import recursive_forecast_batch
from models.calibrate_forecast_intervals import build_origins, calibrate

N_BOOT = 2000
STEPS = [1, 6, 12, 24]


def make_lightgbm():
    return lgb.LGBMRegressor(n_estimators=500, learning_rate=0.03, num_leaves=31,
                             max_depth=-1, min_child_samples=20, subsample=0.8,
                             colsample_bytree=0.8, random_state=42, verbosity=-1)


def make_random_forest():
    return RandomForestRegressor(n_estimators=300, min_samples_leaf=5, max_features=0.6,
                                 n_jobs=-1, random_state=42)


def make_xgboost():
    try:
        from xgboost import XGBRegressor
    except ImportError:
        return None
    # capacity matched to the incumbent (num_leaves=31 ~ max_depth=5), same learning
    # rate and same early-stopping patience: the comparison is about the algorithm.
    return XGBRegressor(n_estimators=500, learning_rate=0.03, max_depth=5, subsample=0.8,
                        colsample_bytree=0.8, tree_method="hist", early_stopping_rounds=30,
                        random_state=42, verbosity=0)


def candidates():
    out = {"LightGBM": (make_lightgbm, "lgb"), "RandomForest": (make_random_forest, "rf")}
    xgb = make_xgboost()
    if xgb is None:
        print("[sweep] xgboost not installed (not a declared dependency) — row skipped")
    else:
        out["XGBoost"] = (make_xgboost, "xgb")
    return out


def fit(make, kind, train, eval_frame, cols):
    tr = train.dropna(subset=cols + [TARGET_COL])
    va = eval_frame.dropna(subset=cols + [TARGET_COL])
    model = make()
    if kind == "lgb":
        model.fit(tr[cols], tr[TARGET_COL], eval_set=[(va[cols], va[TARGET_COL])],
                  callbacks=[lgb.early_stopping(30, verbose=False)])
    elif kind == "xgb":
        model.fit(tr[cols], tr[TARGET_COL], eval_set=[(va[cols], va[TARGET_COL])], verbose=False)
    else:
        # no early stopping to borrow: RF is fit on the train window alone, so its
        # validation number stays an out-of-sample figure like the other two.
        model.fit(tr[cols], tr[TARGET_COL])
    return model


def one_step(model, frame, cols):
    part = frame.dropna(subset=cols + [TARGET_COL])
    pred = model.predict(part[cols])
    return {"mape_pct": round(float(mape_score(part[TARGET_COL], pred)) * 100, 3),
            "mae_mw": round(float(np.mean(np.abs(part[TARGET_COL] - pred))), 1),
            "rows": int(len(part))}


def section_a(df, fit_df, val_df, test_df, cols, feature_label, out):
    print(f"\n=== A. one-step h+1, time-ordered split, {feature_label} features ===")
    print(f"{'model':16s} {'train':>8} {'val':>8} {'test':>8} {'MAE MW':>9}   train->test gap")
    rows = {}
    for name, (make, kind) in candidates().items():
        t0 = time.time()
        m = fit(make, kind, fit_df, val_df, cols)
        r = {"train": one_step(m, fit_df, cols), "val": one_step(m, val_df, cols),
             "test": one_step(m, test_df, cols), "fit_seconds": round(time.time() - t0, 1)}
        r["gap_pp"] = round(r["test"]["mape_pct"] - r["train"]["mape_pct"], 3)
        rows[name] = r
        print(f"{name:16s} {r['train']['mape_pct']:7.3f}% {r['val']['mape_pct']:7.3f}% "
              f"{r['test']['mape_pct']:7.3f}% {r['test']['mae_mw']:9.1f}   {r['gap_pp']:+.3f} pp")
    out["A_one_step_" + feature_label] = rows
    return rows


def replay(model, df, start, end, horizon, stride, cols):
    """Per-origin absolute percentage error, with the zone label each origin belongs to."""
    ape, zlab = [], []
    for zone, g in sorted(df.groupby("zone"), key=lambda kv: kv[0]):
        hist_l, hist_w, last_ts, actual, _ = build_origins(g, start, end, stride, horizon)
        if not len(last_ts):
            continue
        preds = recursive_forecast_batch(model, hist_l, hist_w, last_ts, horizon, cols)
        ape.append(np.abs(actual - preds) / actual * 100)
        zlab.append(np.full(len(actual), zone))
    return np.concatenate(ape), np.concatenate(zlab)


def section_b(df, fit_df, val_df, test_df, cols, out, horizon=48):
    print(f"\n=== B. recursive serving path (each band calibrated on its own model, "
          f"nominal coverage 0.80) ===")
    print(f"{'model':16s}" + "".join(f"{'h+' + str(h):>10}" for h in STEPS)
          + f"{'cov all':>10}{'cov h48':>10}")
    start, end = test_df["timestamp"].min(), test_df["timestamp"].max()
    rows = {}
    for name, (make, kind) in candidates().items():
        m = fit(make, kind, fit_df, val_df, cols)
        art = calibrate(m, df, val_df, test_df, cols, name)
        e = art["evaluation"]
        rows[name] = e
        print(f"{name:16s}" + "".join(f"{e['per_step'][str(h)]['mape_recursive_lgbm']:9.3f}%"
                                      for h in STEPS if str(h) in e["per_step"])
              + f"{e['p10_p90_coverage_overall']:10.4f}"
              + f"{e['per_step'].get('48', {}).get('coverage', float('nan')):10.4f}")
    out["B_recursive"] = rows
    return rows


def section_c(models_ape, ref="LightGBM", n_boot=N_BOOT, seed=42):
    print(f"\n=== C. paired bootstrap over origins (negative = challenger better) ===")
    base = models_ape[ref]
    n = len(base)
    rng = np.random.default_rng(seed)
    out = {"origins": int(n), "reference": ref}
    for h in STEPS:
        i = h - 1
        print(f"  h+{h:<2} {ref} MAPE {base[:, i].mean():.3f}%")
        for name, ape in models_ape.items():
            if name == ref:
                continue
            d = ape[:, i] - base[:, i]
            boots = np.array([d[rng.integers(0, n, n)].mean() for _ in range(n_boot)])
            lo, hi = np.percentile(boots, [2.5, 97.5])
            sig = bool(lo > 0 or hi < 0)
            out.setdefault(f"h{h}", {})[name] = {
                "diff_pp": round(float(d.mean()), 4),
                "ci95": [round(float(lo), 4), round(float(hi), 4)],
                "wins_pct_origins": round(float((d < 0).mean()) * 100, 1),
                "significant": sig}
            print(f"       vs {name:14s} {d.mean():+.3f} pp  95% CI [{lo:+.3f}, {hi:+.3f}]"
                  f"  {'SIGNIFICANT' if sig else 'indistinguishable from 0':24s}"
                  f"  better on {(d < 0).mean() * 100:.1f}% of origins")
    return out


def section_d(df, test_df, cols, out):
    """What the recipe's shuffled split and shuffled CV would have claimed."""
    print("\n=== D. the shuffled-split and shuffled-CV counterfactual ===")
    sh = df.sample(frac=1.0, random_state=42).reset_index(drop=True)
    n = len(sh)
    s_train, s_val, s_test = sh[: int(n * .8)], sh[int(n * .8): int(n * .9)], sh[int(n * .9):]

    shuffled_train_keys = set(zip(s_train["zone"], s_train["timestamp"]))
    honest_test_keys = set(zip(test_df["zone"], test_df["timestamp"]))
    # Two denominators, because they mean different things and only one is the leak:
    # how much of the future window the shuffled model has already seen, versus how
    # much of its own training set happens to be future data.
    memorized = float(np.mean([(z, t) in shuffled_train_keys
                               for z, t in zip(test_df["zone"], test_df["timestamp"])]))
    contamination = float(np.mean([(z, t) in honest_test_keys
                                   for z, t in zip(s_train["zone"], s_train["timestamp"])]))
    train_ts = set(s_train.timestamp)
    neighbour = float(((s_test.timestamp - pd.Timedelta(hours=1)).isin(train_ts)).mean())
    print(f"  honest test-window rows inside the shuffled TRAIN partition: {memorized:.2%}"
          f"  ({int(memorized * len(test_df))} of {len(test_df)})")
    print(f"  shuffled TRAIN rows that are honest test-window rows:        {contamination:.2%}")
    print(f"  shuffled TEST rows whose own t-1h hour is in shuffled TRAIN: {neighbour:.2%}")

    m = fit(make_lightgbm, "lgb", s_train, s_val, cols)
    start, end = test_df["timestamp"].min(), test_df["timestamp"].max()
    art = calibrate(m, df, s_val, test_df, cols, "shuffled-split LightGBM")
    e = art["evaluation"]
    print("  that model scored on the contiguous FUTURE window:")
    print("        " + "  ".join(f"h+{h} {e['per_step'][str(h)]['mape_recursive_lgbm']:.3f}%"
                                 f"/cov {e['per_step'][str(h)]['coverage']:.3f}"
                                 for h in STEPS + [48] if str(h) in e["per_step"]))
    out["D_shuffle"] = {"test_rows_memorized_by_shuffled_train": round(memorized, 4),
                        "shuffled_train_share_that_is_honest_test": round(contamination, 4),
                        "t1h_neighbour_in_shuffled_train": round(neighbour, 4),
                        "evaluation": e}

    # Both configs, because the fold spread is a property of the row set each config
    # keeps (weather-complete rows are far sparser) and quoting one alone invites the
    # reader to treat a config difference as run-to-run noise.
    for tag, feature_cols in (("no-weather", FEATURE_COLS_NO_WEATHER),
                              ("weather-aware", FEATURE_COLS)):
        V = df.dropna(subset=feature_cols + [TARGET_COL])
        X, y = V[feature_cols].to_numpy(float), V[TARGET_COL].to_numpy(float)
        print(f"  CV over {len(V)} rows, {len(feature_cols)} features ({tag})")
        for label, splitter in (("shuffled 5-fold KFold", KFold(5, shuffle=True, random_state=42)),
                                ("time-ordered TimeSeriesSplit", TimeSeriesSplit(5))):
            ms = []
            for tr_i, te_i in splitter.split(X):
                mm = make_lightgbm()
                mm.fit(X[tr_i], y[tr_i], eval_set=[(X[te_i], y[te_i])],
                       callbacks=[lgb.early_stopping(30, verbose=False)])
                ms.append(float(mape_score(y[te_i], mm.predict(X[te_i])) * 100))
            print(f"    {label:30s} mean {np.mean(ms):6.3f}%  std {np.std(ms):5.3f}  folds "
                  f"{[round(v, 2) for v in ms]}")
            out.setdefault("D_cv", {})[f"{tag} | {label}"] = {
                "rows": int(len(V)), "mean_mape_pct": round(float(np.mean(ms)), 3),
                "std_mape_pct": round(float(np.std(ms)), 3),
                "folds": [round(v, 3) for v in ms]}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", help="write every number measured here to this path")
    args = ap.parse_args()

    df = pd.read_parquet(DATA_FEATURES_DIR / "training_features.parquet")
    fit_df, val_df, test_df = train_val_test_split(df)
    out = {"split": {"fit": int(len(fit_df)), "val": int(len(val_df)), "test": int(len(test_df)),
                     "test_start": str(test_df["timestamp"].min()),
                     "test_end": str(test_df["timestamp"].max())}}

    section_a(df, fit_df, val_df, test_df, FEATURE_COLS_NO_WEATHER, "no-weather (LIVE)", out)
    section_a(df, fit_df, val_df, test_df, FEATURE_COLS, "weather-aware", out)

    print("\n=== fitting once for the replay sections ===")
    horizon = 24
    start, end = test_df["timestamp"].min(), test_df["timestamp"].max()
    ape, zlab = {}, None
    for name, (make, kind) in candidates().items():
        m = fit(make, kind, fit_df, val_df, FEATURE_COLS_NO_WEATHER)
        a, z = replay(m, df, start, end, horizon, 1, FEATURE_COLS_NO_WEATHER)
        ape[name], zlab = a, z
        print(f"  [{name}] {len(a)} origins, h+24 MAPE {a[:, horizon - 1].mean():.3f}%")
    out["C_bootstrap"] = section_c(ape)
    out["C_per_zone_h24"] = {}
    for name, a in ape.items():
        row = {z: round(float(a[zlab == z][:, horizon - 1].mean()), 3) for z in sorted(set(zlab))}
        print(f"  per-zone h+24 {name:14s} " + "  ".join(f"{k} {v:.3f}%" for k, v in row.items()))
        out["C_per_zone_h24"][name] = row

    section_b(df, fit_df, val_df, test_df, FEATURE_COLS_NO_WEATHER, out)
    section_d(df, test_df, FEATURE_COLS_NO_WEATHER, out)

    if args.json:
        Path(args.json).write_text(json.dumps(out, indent=1))
        print(f"\n[sweep] wrote {args.json}")
    print("[sweep] nothing in this run was written to data/models/; the served artifacts "
          "are untouched.")


if __name__ == "__main__":
    main()
