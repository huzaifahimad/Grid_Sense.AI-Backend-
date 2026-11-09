"""
Step 3 — honest per-zone evaluation + Step 1's "what age does that force" simulation.

Reads ONLY: committed parquet, incumbent joblib, the scratch-trained weather-blind
joblib. Writes only under _risknw_scratch/. Never calls train_and_evaluate().

A) TEST-window metrics (same walk_forward_split convention, time-ordered):
   per zone and aggregate, weather-blind candidate vs incumbent on their own
   scorable rows, plus jointly-scorable agreement (shift, flips, recall).
   MAPE/MAE are regression metrics and do not apply to a classifier here; alarm
   calibration (reliability/ECE/threshold recall) is the analogue and is reported.
B) SERVING SIMULATION on a mock refreshed frame (the live /health shape): load to
   2026-11-07T06:00Z with weather NaN beyond 2026-11-04T23:00Z -> what snapshot
   does api.main._representative_snapshot pick for each model choice, at now=
   2026-11-07T08:00Z? That is the panel age the defect forces, measured.
C) THE HOURS THE FALLBACK WOULD ACTUALLY TAKE OVER: incumbent-unscorable hours in
   the test window — labels, blow-up check of features, both models' scores where
   possible, and per-zone counts.
"""
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss

from config import DATA_FEATURES_DIR, MODELS_DIR
from models.decision_rules import ALARM_THRESHOLD, CRITICAL_THRESHOLD, ELEVATED_THRESHOLD
from models.risk_classifier import (
    FEATURE_COLS, FEATURE_COLS_NO_WEATHER, TARGET_COL, walk_forward_split,
)

FALLBACK_JOB = _ROOT / "_risknw_scratch/out_v1/risk_classifier_no_weather.joblib"

df = pd.read_parquet(DATA_FEATURES_DIR / "training_features.parquet")
df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
df = df.sort_values("timestamp").reset_index(drop=True)

train_full, test_df = walk_forward_split(df)
fit_df = train_full[train_full.timestamp < train_full.iloc[int(len(train_full)*0.9)].timestamp]
val_df = train_full[train_full.timestamp >= fit_df.timestamp.max()]


def ready(d, feats):
    return d.dropna(subset=feats + [TARGET_COL])


def metrics_for(y, p):
    alarm = p >= ALARM_THRESHOLD
    tp = int((alarm & (y == 1)).sum()); fp = int((alarm & (y == 0)).sum())
    fn = int((~alarm & (y == 1)).sum())
    rec = tp / (tp + fn) if tp + fn else float("nan")
    prec = tp / (tp + fp) if tp + fp else float("nan")
    base = float(np.mean(y))
    return {"n": int(len(y)), "pos_rate": round(base, 4),
            "pr_auc": round(float(average_precision_score(y, p)), 4) if len(set(y.tolist())) > 1 else None,
            "brier": round(float(brier_score_loss(y, p)), 4),
            "brier_skill_vs_base": (round(1 - float(brier_score_loss(y, p)) /
                                          float(brier_score_loss(y, np.full_like(p, base))), 4)
                                    if base not in (0, 1) else None),
            "alarm_recall@0.6": round(rec, 4), "alarm_precision@0.6": round(prec, 4),
            "mean_score": round(float(p.mean()), 4),
            "frac>=elevated": round(float((p >= ELEVATED_THRESHOLD).mean()), 4),
            "frac>=critical": round(float((p >= CRITICAL_THRESHOLD).mean()), 4)}


inc_model = joblib.load(MODELS_DIR / "risk_classifier.joblib")
fb_model = joblib.load(FALLBACK_JOB)

def probs(model, d, feats):
    return np.asarray(model.predict_proba(d[feats])[:, 1], dtype=float)

print("=== A) TEST window (2026-06-13T10:00Z .. end), per zone ===")
rows = []
for zone in sorted(df.zone.unique()):
    t = test_df[test_df.zone == zone]
    inc_t, fb_t = ready(t, FEATURE_COLS), ready(t, FEATURE_COLS_NO_WEATHER)
    pi, pf = probs(inc_model, inc_t, FEATURE_COLS), probs(fb_model, fb_t, FEATURE_COLS_NO_WEATHER)
    mi = metrics_for(inc_t[TARGET_COL].to_numpy(), pi)
    mf = metrics_for(fb_t[TARGET_COL].to_numpy(), pf)
    # joint agreement on this zone's jointly-scorable hours
    both = inc_t.index.intersection(fb_t.index)
    ib, fb = inc_t.loc[both], fb_t.loc[both]
    yb = ib[TARGET_COL].to_numpy()
    pib, pfb = probs(inc_model, ib, FEATURE_COLS), probs(fb_model, fb, FEATURE_COLS_NO_WEATHER)
    s_inc = np.where(pib >= ELEVATED_THRESHOLD, "elev", "norm")
    s_fb = np.where(pfb >= ELEVATED_THRESHOLD, "elev", "norm")
    r_inc = ((pib >= ALARM_THRESHOLD) & (yb == 1)).sum() / max(((yb == 1)).sum(), 1)
    r_fb = ((pfb >= ALARM_THRESHOLD) & (yb == 1)).sum() / max(((yb == 1)).sum(), 1)
    rows.append({"zone": zone, **{f"inc_{k}": v for k, v in mi.items()},
                 **{f"fb_{k}": v for k, v in mf.items()},
                 "joint_n": len(both),
                 "joint_mean_shift": round(float((pfb - pib).mean()), 4),
                 "joint_status_flip_rate": round(float((s_inc != s_fb).mean()), 4),
                 "joint_recall_inc": round(float(r_inc), 4),
                 "joint_recall_fb": round(float(r_fb), 4)})
tbl = pd.DataFrame(rows)
pd.set_option("display.width", 400); pd.set_option("display.max_columns", 60)
print(tbl.to_string(index=False))
tbl.to_csv(_ROOT / "_risknw_scratch/per_zone_test_metrics.csv", index=False)

print("\n=== A') same on ALL rows both models can score across full history (context) ===")
inc_all, fb_all = ready(df, FEATURE_COLS), ready(df, FEATURE_COLS_NO_WEATHER)
pi = probs(inc_model, inc_all, FEATURE_COLS); pf = probs(fb_model, fb_all, FEATURE_COLS_NO_WEATHER)
both = inc_all.index.intersection(fb_all.index)
y = inc_all.loc[both][TARGET_COL].to_numpy()
print("incumbent-all:", metrics_for(inc_all[TARGET_COL].to_numpy(), pi))
print("fallback-all :", metrics_for(fb_all[TARGET_COL].to_numpy(), pf))

print("\n=== C) hours the fallback would take over (incumbent-unscorable, test window) ===")
for zone in sorted(df.zone.unique()):
    t = test_df[test_df.zone == zone]
    ok_i = t[FEATURE_COLS].notna().all(axis=1)
    tail = t[~ok_i]
    lab_ok = tail[TARGET_COL].notna()
    tail2 = tail[lab_ok]
    print(f"{zone}: unscorable-in-test n={len(tail)} label-available={len(tail2)} "
          f"positives={int(tail2[TARGET_COL].sum())}")
    if len(tail2):
        print(f"   span {tail2.timestamp.min()}..{tail2.timestamp.max()}")
        missing = tail2[FEATURE_COLS_NO_WEATHER].isna().sum().sum()
        print(f"   NaN in no-weather feature cols among those rows: {missing}")
        fin = tail2.dropna(subset=FEATURE_COLS_NO_WEATHER)
        if len(fin):
            p = probs(fb_model, fin, FEATURE_COLS_NO_WEATHER)
            print(f"   fallback scores finite={np.isfinite(p).mean():.2%} "
                  f"mean={p.mean():.4f} max={p.max():.4f} "
                  f"alarm_hours={int((p>=ALARM_THRESHOLD).sum())} "
                  f"(vs observed positives={int(fin[TARGET_COL].sum())})")

print("\n=== B) serving simulation: frame shaped like LIVE /health right now ===")
NEWEST = pd.Timestamp("2026-11-07T06:00:00+00:00")
WEATHER_FRONTIER = pd.Timestamp("2026-11-04T23:00:00+00:00")
NOW = pd.Timestamp("2026-11-07T08:00:00+00:00")

mock_parts = []
for zone, g in df.groupby("zone"):
    g = g.sort_values("timestamp")
    base = g.tail(336).copy()                      # served buffer size
    past = base[base.timestamp <= WEATHER_FRONTIER]
    fut_hours = pd.date_range(WEATHER_FRONTIER + pd.Timedelta(hours=1), NEWEST, freq="h")
    add = pd.DataFrame({"timestamp": fut_hours, "zone": zone})
    for c in base.columns:
        if c not in add.columns:
            src = past[c]
            # lag/rolling features from the zone's own recent observed values
            add[c] = src.iloc[-1] if c.startswith(("load_", "capacity")) else np.nan
    add["load_mw"] = past["load_mw"].iloc[-1]
    add["load_lag_1h"] = past["load_mw"].iloc[-1]
    add["load_lag_24h"] = past["load_mw"].iloc[-min(25, len(past))]
    add["load_roll_mean_24h"] = past["load_mw"].iloc[-24:].mean()
    add["load_roll_std_24h"] = past["load_mw"].iloc[-24:].std()
    add["capacity_mw"] = past["capacity_mw"].iloc[-1]
    add["capacity_is_proxy"] = past["capacity_is_proxy"].iloc[-1]
    add[[c for c in FEATURE_COLS if c in ("temperature_c", "humidity_pct", "wind_speed_ms")]] = np.nan
    hs, hc = np.sin(2*np.pi*add.timestamp.dt.hour/24), np.cos(2*np.pi*add.timestamp.dt.hour/24)
    ds, dc = np.sin(2*np.pi*add.timestamp.dt.dayofweek/7), np.cos(2*np.pi*add.timestamp.dt.dayofweek/7)
    ms, mc = np.sin(2*np.pi*add.timestamp.dt.month/12), np.cos(2*np.pi*add.timestamp.dt.month/12)
    add["hour_sin"], add["hour_cos"] = hs, hc
    add["dow_sin"], add["dow_cos"] = ds, dc
    add["month_sin"], add["month_cos"] = ms, mc
    add["is_weekend"] = (add.timestamp.dt.dayofweek >= 5).astype(int)
    keep = [c for c in base.columns if c in add.columns]
    mock_parts.append(pd.concat([past[keep], add[keep]], ignore_index=True))
mock = pd.concat(mock_parts, ignore_index=True)

import api.main as m
m._state["features"] = mock
real_now = m._now_utc
m._now_utc = lambda: NOW
try:
    for zone in sorted(mock.zone.unique()):
        zdf = mock[mock.zone == zone]
        r_new, mode_new = m._representative_snapshot(zdf, FEATURE_COLS)
        age_new = (NOW - pd.Timestamp(r_new.timestamp.iloc[0])).total_seconds()/3600 if len(r_new) else None
        r_fbrow, mode_fb = m._representative_snapshot(zdf, FEATURE_COLS_NO_WEATHER)
        age_fb = (NOW - pd.Timestamp(r_fbrow.timestamp.iloc[0])).total_seconds()/3600 if len(r_fbrow) else None
        sc_fb = float(probs(fb_model, r_fbrow, FEATURE_COLS_NO_WEATHER)[0]) if len(r_fbrow) else None
        print(f"{zone}: TODAY(weather-aware selection): ts={r_new.timestamp.iloc[0]} mode={mode_new} age={age_new:.1f}h")
        print(f"    WITH(weather-blind selection)     : ts={r_fbrow.timestamp.iloc[0]} mode={mode_fb} age={age_fb:.1f}h score={sc_fb:.4f}")
finally:
    m._now_utc = real_now
print("\nDONE")
