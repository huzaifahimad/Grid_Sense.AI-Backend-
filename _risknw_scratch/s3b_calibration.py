"""
Step 3b — threshold / P-R calibration table per zone, honest test window only.

For each zone and the aggregate: on hours BOTH models can score (so the comparison is
like-for-like), sweep candidate alarm thresholds and report precision/recall/F1 at each,
plus what a quantile-matched operating point implies. Also reliability of both models'
score deciles vs observed overload rate (the classifier analogue of P10-P90 bands).

No training, no writes outside _risknw_scratch/.
"""
import sys
from pathlib import Path
_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support

from config import DATA_FEATURES_DIR, MODELS_DIR
from models.risk_classifier import FEATURE_COLS, FEATURE_COLS_NO_WEATHER, TARGET_COL, walk_forward_split

df = pd.read_parquet(DATA_FEATURES_DIR / "training_features.parquet")
df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
train_full, test_df = walk_forward_split(df)

inc = joblib.load(MODELS_DIR / "risk_classifier.joblib")
fb = joblib.load(_ROOT / "_risknw_scratch/out_v1/risk_classifier_no_weather.joblib")

rows = []
for zone in sorted(df.zone.unique()) + ["ALL"]:
    t = test_df if zone == "ALL" else test_df[test_df.zone == zone]
    ti = t.dropna(subset=FEATURE_COLS + [TARGET_COL])
    tf_ = t.dropna(subset=FEATURE_COLS_NO_WEATHER + [TARGET_COL])
    both = ti.index.intersection(tf_.index)
    ti, tf_ = ti.loc[both], tf_.loc[both]
    y = ti[TARGET_COL].to_numpy()
    pi = np.asarray(inc.predict_proba(ti[FEATURE_COLS])[:, 1], float)
    pf = np.asarray(fb.predict_proba(tf_[FEATURE_COLS_NO_WEATHER])[:, 1], float)
    for thr in (0.4, 0.5, 0.6, ALARM := 0.6, 0.7, 0.8):
        pass
    for thr in (0.4, 0.5, 0.6, 0.7, 0.8):
        for name, p in (("weather_aware", pi), ("no_weather", pf)):
            pred = (p >= thr).astype(int)
            tp = int(((pred == 1) & (y == 1)).sum()); fp = int(((pred == 1) & (y == 0)).sum())
            fn = int(((pred == 0) & (y == 1)).sum())
            prec, rec, f1, _ = precision_recall_fscore_support(
                y, pred, average="binary", zero_division=np.nan)
            rows.append({"zone": zone, "model": name, "thr": thr,
                         "alarms": int(pred.sum()), "tp": tp, "fp": fp, "fn": fn,
                         "precision": round(float(prec), 4), "recall": round(float(rec), 4),
                         "f1": round(float(f1), 4)})
tbl = pd.DataFrame(rows)
tbl.to_csv(_ROOT / "_risknw_scratch/threshold_table.csv", index=False)
pd.set_option("display.width", 250)
print(tbl.pivot_table(index=["zone", "thr"], columns="model",
                      values=["precision", "recall"]).round(4).to_string())

print("\n=== reliability: score decile -> observed overload rate (test window, joint rows) ===")
t = test_df
ti = t.dropna(subset=FEATURE_COLS + [TARGET_COL])
tf_ = t.dropna(subset=FEATURE_COLS_NO_WEATHER + [TARGET_COL])
both = ti.index.intersection(tf_.index)
y = ti.loc[both][TARGET_COL].to_numpy()
pi = np.asarray(inc.predict_proba(ti.loc[both][FEATURE_COLS])[:, 1], float)
pf = np.asarray(fb.predict_proba(tf_.loc[both][FEATURE_COLS_NO_WEATHER])[:, 1], float)
for name, p in (("weather_aware", pi), ("no_weather", pf)):
    q = pd.qcut(p, 10, labels=False, duplicates="drop")
    rel = pd.DataFrame({"decile": q, "p": p, "y": y}).groupby("decile").agg(
        n=("y", "size"), mean_pred=("p", "mean"), observed=("y", "mean"))
    print(f"--- {name} ---")
    print(rel.round(4).to_string())

print("\n=== where fallback under-alarms vs incumbent (joint rows, thr=0.6) ===")
a_i, a_f = pi >= 0.6, pf >= 0.6
missed = (~a_f) & a_i
print(f"hours incumbent alarms and fallback does not: {int(missed.sum())} "
      f"({missed.mean():.2%}); of those truly overloaded: {int((missed & (y==1)).sum())}")
extra = a_f & (~a_i)
print(f"hours fallback alarms and incumbent does not: {int(extra.sum())}; "
      f"truly overloaded: {int((extra & (y==1)).sum())}")
print(f"max absolute score gap: {np.abs(pf-pi).max():.4f} "
      f"(fallback={pf[np.argmax(np.abs(pf-pi))]:.4f} vs incumbent={pi[np.argmax(np.abs(pf-pi))]:.4f})")
print("DONE3B")
