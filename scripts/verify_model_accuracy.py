"""
GridSense AI — model accuracy verification: are the reported metrics real, and
could any of them be leakage?

`scripts/verify_prediction_integrity.py` proves the served numbers are
*computed*. This script answers the other half: whether the accuracy figures
quoted in the README, docs/02 and `/validation` can be reproduced from the
committed artifacts with an honest split, and whether the feature set makes
that reproduction meaningless.

Seven sections. Sections 1-6 need only the committed artifacts; section 7 also
calls the live API and skips cleanly when it is unreachable.

  1. Label integrity   — the stored `overload_risk_label` equals the documented
                         rule `load_mw > OVERLOAD_THRESHOLD_RATIO * capacity_mw`.
  2. Leakage           — the artifact's own recorded `feature_name_` contains no
                         column that encodes the label (ratio, load_mw, capacity_mw).
  3. Split honesty     — the split is temporal, so the scored rows postdate every
                         row the fit step could have seen, plus a direct
                         measurement of how much a shuffled split would leak.
  4. Held-out metrics  — PR-AUC, ROC-AUC, Brier, precision/recall recomputed here.
  5. Reliability       — observed overload rate inside each predicted bucket.
  6. Response range    — the score as a function of utilisation, including where
                         the tree ensemble stops responding.
  7. Deployed agreement— `/validation` equals the on-disk report and its figures
                         equal the recomputation in 4/5, and each deployed risk
                         score equals this artifact re-scoring the same row.

Checks 1, 2 and 3 are correctness properties: they must hold exactly, and they
fail the run. The rest *report* performance and only gate on agreement between
two independent computations of the same quantity — never on a performance
target, so no result here can be improved by moving a threshold.

    python scripts/verify_model_accuracy.py
    python scripts/verify_model_accuracy.py --skip-live

Exit code 0 = properties hold and every reported figure was reproduced twice.
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.request
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# A Windows console defaults to cp1252 and raises UnicodeEncodeError on the
# arrows and inequality signs used throughout the report below.
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")

from config import DATA_FEATURES_DIR, MODELS_DIR, OVERLOAD_THRESHOLD_RATIO  # noqa: E402
from models.decision_rules import (  # noqa: E402
    CRITICAL_THRESHOLD,
    ELEVATED_THRESHOLD,
    MODERATE_THRESHOLD,
    dashboard_status,
    shed_priority,
)
from models.risk_classifier import FEATURE_COLS, TARGET_COL, walk_forward_split  # noqa: E402

# A column leaks the answer if the label rule can be evaluated from it alone.
LEAKAGE_SUSPECTS = {
    "overload_risk_label",
    "load_to_capacity_ratio",
    "load_mw",
    "capacity_mw",
}

# Predicted-score buckets the dashboard reports, straight from decision_rules so
# this check can never disagree with the app it is auditing.
BUCKETS = [
    ("low", 0.0, MODERATE_THRESHOLD),
    ("moderate", MODERATE_THRESHOLD, ELEVATED_THRESHOLD),
    ("elevated", ELEVATED_THRESHOLD, CRITICAL_THRESHOLD),
    ("critical", CRITICAL_THRESHOLD, 1.0 + 1e-9),
]

results: list[tuple[str, bool, str]] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    results.append((name, ok, detail))
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f" — {detail}" if detail else ""))


def report(name: str, detail: str) -> None:
    print(f"  [    ] {name} — {detail}")


def get_json(url: str, timeout: int = 120) -> dict:
    req = urllib.request.Request(url, headers={"User-Agent": "gridsense-accuracy-check"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read())


def load_frame() -> pd.DataFrame:
    df = pd.read_parquet(DATA_FEATURES_DIR / "training_features.parquet")
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    return df


# ---------------------------------------------------------------------- 1
def check_label_integrity(df: pd.DataFrame) -> None:
    print("\n=== 1. Label integrity ===")
    has_rule_cols = {"load_mw", "capacity_mw"}.issubset(df.columns)
    check("features frame carries load_mw and capacity_mw (the label rule inputs)",
          has_rule_cols,
          "present" if has_rule_cols else f"missing {( {'load_mw','capacity_mw'} - set(df.columns))}")
    if not has_rule_cols:
        return

    d = df.dropna(subset=["load_mw", "capacity_mw", TARGET_COL])
    ratio = d["load_mw"] / d["capacity_mw"]
    if "load_to_capacity_ratio" in d.columns:
        ratio = d["load_to_capacity_ratio"]
    recomputed = (ratio > OVERLOAD_THRESHOLD_RATIO).astype(int)
    stored = d[TARGET_COL].astype(int)
    agreement = float((recomputed == stored).mean())
    check(f"stored label == (load > {OVERLOAD_THRESHOLD_RATIO} × capacity)",
          agreement == 1.0,
          f"rows={len(d):,} | positive rate stored={stored.mean():.4f} "
          f"recomputed={recomputed.mean():.4f} | exact agreement={agreement:.6f}")


# ---------------------------------------------------------------------- 2
def check_leakage(model) -> None:
    print("\n=== 2. Leakage into the classifier ===")
    recorded = [str(c) for c in getattr(model, "feature_name_", [])]
    check("artifact records its own feature names", bool(recorded),
          f"{len(recorded)} features: {', '.join(recorded)}")
    check("artifact features == models.risk_classifier.FEATURE_COLS",
          recorded == list(FEATURE_COLS),
          "identical, in order" if recorded == list(FEATURE_COLS)
          else f"differ: extra={set(recorded) - set(FEATURE_COLS)}, "
               f"missing={set(FEATURE_COLS) - set(recorded)}")
    suspects = sorted(set(recorded) & LEAKAGE_SUSPECTS)
    check("no feature can evaluate the label rule on its own",
          not suspects,
          f"present in artifact: {', '.join(suspects)}" if suspects else "none present")


# ---------------------------------------------------------------------- 3
def check_split(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    print("\n=== 3. Split honesty (temporal, via the trainer's own function) ===")
    train_part, test_part = walk_forward_split(df)
    train_max = train_part["timestamp"].max()
    test_min = test_part["timestamp"].min()
    check("every scored row starts after the last fitted row",
          bool(test_min > train_max),
          f"train {len(train_part):,} rows ≤ {train_max} | test {len(test_part):,} "
          f"rows ≥ {test_min}")

    # The alternative this project was explicitly asked to use ("80/10/10, shuffle
    # before split"), done literally: shuffle the rows, then split positionally.
    shuffled_idx = np.random.RandomState(0).permutation(len(df))
    cut = int(len(df) * 0.8)
    r_train, r_test = df.iloc[shuffled_idx[:cut]], df.iloc[shuffled_idx[cut:]]
    inside = int((r_test["timestamp"] <= r_train["timestamp"].max()).sum())

    # The concrete leakage channel, measured in the direction it actually flows:
    # each fitted row's lag_1h / lag_24h / roll_mean_24h / roll_std_24h are built
    # from the 24 hours that precede it. Under a random split some of those hours
    # are held-out rows, so the fit sees test-set loads as inputs.
    HOUR_NS = 3_600 * 1_000_000_000  # datetime64[ns] arithmetic
    in_test = np.zeros(len(df), dtype=bool)
    in_test[shuffled_idx[cut:]] = True
    lut = dict(zip(zip(df["zone"], df["timestamp"].astype("int64")), in_test))
    tz = list(zip(r_train["zone"], r_train["timestamp"].astype("int64")))
    contaminated = np.zeros(len(tz), dtype=bool)
    for offset in range(1, 25):
        contaminated |= np.fromiter(
            (lut.get((z, t - offset * HOUR_NS), False) for z, t in tz),
            dtype=bool, count=len(tz))
    pct = contaminated.mean()

    report("why the shuffled 80/10/10 split is rejected (measured, not asserted)",
           f"random shuffle puts {inside:,} of {len(r_test):,} scored rows "
           f"({inside / len(r_test):.1%}) chronologically at or before the last fitted "
           f"hour, and {int(contaminated.sum()):,} of the {len(tz):,} fitted rows "
           f"({pct:.1%}) take their lag/rolling inputs directly from held-out rows. "
           f"A metric from that split describes a model that already saw the answers. "
           f"The trainer's walk_forward_split avoids it, which check 1 of this section "
           f"is what proves.")
    return train_part, test_part


# ---------------------------------------------------------------------- 4 + 5
def score_held_out(model, test_part: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    print("\n=== 4. Held-out metrics (recomputed here, never read from the report) ===")
    d = test_part.dropna(subset=FEATURE_COLS + [TARGET_COL]).reset_index(drop=True)
    probs = model.predict_proba(d[FEATURE_COLS])[:, 1]
    y = d[TARGET_COL].to_numpy().astype(int)
    base = float(y.mean())
    pred = (probs >= ELEVATED_THRESHOLD).astype(int)
    tp = int(((pred == 1) & (y == 1)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())

    pr_auc = float(average_precision_score(y, probs))
    roc_auc = float(roc_auc_score(y, probs))
    brier = float(brier_score_loss(y, probs))
    brier_ref = float(brier_score_loss(y, np.full_like(probs, base)))
    precision = tp / (tp + fp) if tp + fp else float("nan")
    recall = tp / (tp + fn) if tp + fn else float("nan")

    check("positives exist in the scored window (a 0-positive test set proves nothing)",
          tp + fn > 0, f"rows={len(d):,} | positives={int(y.sum()):,} ({base:.2%})")
    report("PR-AUC (test)", f"{pr_auc:.4f} — the metric the trainer tells us to report; "
                            f"chance level is the base rate {base:.4f}")
    report("ROC-AUC (test)", f"{roc_auc:.4f}")
    report("Brier (test)", f"{brier:.4f} vs base-rate reference {brier_ref:.4f} "
                           f"→ skill {1 - brier / brier_ref:.4f}")
    report(f"Alarm threshold {ELEVATED_THRESHOLD}",
           f"TP={tp:,} FP={fp:,} FN={fn:,} | precision={precision:.4f} recall={recall:.4f}")

    print("\n=== 5. Reliability by predicted bucket ===")
    for name, lo, hi in BUCKETS:
        m = (probs >= lo) & (probs < hi)
        if not m.any():
            report(f"{name}", "no rows scored into this bucket")
            continue
        mid = (lo + min(hi, 1.0)) / 2
        obs = float(y[m].mean())
        report(f"{name:9} n={int(m.sum()):6,}",
               f"predicted_mid={mid:.2f} observed_overload={obs:.4f} "
               f"gap={obs - mid:+.4f}")
    d = d.assign(_prob=probs, _y=y)
    metrics = {"precision": precision, "recall": recall, "pr_auc": pr_auc,
               "roc_auc": roc_auc, "brier": brier,
               "brier_skill": 1 - brier / brier_ref, "base_rate": base}
    return d, metrics


def check_per_zone(scored: pd.DataFrame) -> None:
    print("\n=== 5b. Per-zone split of the aggregate ===")
    if "zone" not in scored.columns:
        report("zone column", "not present — aggregate only")
        return
    agg = int(((scored["_prob"] >= ELEVATED_THRESHOLD) & (scored["_y"] == 1)).sum())
    report("aggregate recall", f"{agg / max(int(scored['_y'].sum()), 1):.4f} "
                               f"({agg:,} of {int(scored['_y'].sum()):,} positives)")
    for zone, g in scored.groupby("zone"):
        pos = int(g["_y"].sum())
        if pos == 0:
            report(str(zone), "no positives in the scored window")
            continue
        hit = int(((g["_prob"] >= ELEVATED_THRESHOLD) & (g["_y"] == 1)).sum())
        alarm = int((g["_prob"] >= ELEVATED_THRESHOLD).sum())
        prec = hit / alarm if alarm else float("nan")
        rec = hit / pos
        if rec < 0.6:
            report(f"{zone:9} n={len(g):6,} positives={pos:5,}",
                   f"precision={prec:.4f} recall={rec:.4f} ← weakest zone; the "
                   f"aggregate above hides this, so it is stated here instead")
        else:
            report(f"{zone:9} n={len(g):6,} positives={pos:5,}",
                   f"precision={prec:.4f} recall={rec:.4f}")


# ---------------------------------------------------------------------- 6
def check_response_range(model, df: pd.DataFrame) -> None:
    print("\n=== 6. Response range: what the model does with an overload it has never seen ===")
    ladder = [0.50, 0.70, 0.85, 0.95, 1.05, 1.15, 1.30]
    for zone in sorted(df["zone"].unique()):
        g = df[df["zone"] == zone].dropna(subset=FEATURE_COLS + ["capacity_mw"])
        g = g.sort_values("timestamp")
        if g.empty:
            report(zone, "no scorable rows")
            continue
        row = g.iloc[-1]
        cap = float(row["capacity_mw"])
        anchor_load = float(row["load_mw"])
        if not np.isfinite(anchor_load) or anchor_load <= 0:
            report(f"{zone}: response range", "skipped, anchor row has no usable load_mw")
            continue
        base = float(model.predict_proba(pd.DataFrame([row[FEATURE_COLS]],
                                                      columns=FEATURE_COLS, dtype=float))[:, 1][0])
        curve = []
        for util in ladder:
            # Scale the utilisation-bearing inputs together; the clock and weather hold.
            scale = util * cap / anchor_load
            v = {c: float(row[c]) * (scale if c.startswith("load_") else 1.0)
                 for c in FEATURE_COLS}
            x = pd.DataFrame([[v[c] for c in FEATURE_COLS]], columns=FEATURE_COLS, dtype=float)
            curve.append((util, float(model.predict_proba(x)[:, 1][0])))
        peak_util, peak = max(curve, key=lambda k: k[1])
        worst_drop = min((b - a, u0, u1) for (u0, a), (u1, b) in zip(curve, curve[1:]))
        reached = [u for u, c in curve if c >= ELEVATED_THRESHOLD]
        report(f"{zone} (row {row['timestamp']}, load {anchor_load:,.0f} MW / "
               f"cap {cap:,.0f} MW, base risk {base:.6f})",
               " | ".join(f"util {u:.2f} -> risk {c:.6f} [{dashboard_status(c)}"
                          f"{', shed P' + str(p) if (p := shed_priority(c)) else ''}]"
                          for u, c in curve))
        check(f"{zone}: 30% over capacity does not score below nominal utilisation",
              curve[-1][1] >= curve[0][1],
              f"util {ladder[-1]:.2f} -> {curve[-1][1]:.6f} vs util {ladder[0]:.2f} -> "
              f"{curve[0][1]:.6f}")
        report(f"{zone}: largest step down as utilisation rises",
               f"{worst_drop[0]:+.6f} between util {worst_drop[1]:.2f} and "
               f"{worst_drop[2]:.2f} — a tree ensemble is not monotone in load, so this "
               f"curve is not trustworthy rung by rung, only as a range.")
        if reached:
            report(f"{zone}: first alarm rung",
                   f"util {reached[0]:.2f} crosses {ELEVATED_THRESHOLD}")
        else:
            report(f"{zone}: extrapolation limit (measured over util "
                   f"{ladder[0]:.2f}-{ladder[-1]:.2f})",
                   f"no rung reaches the {ELEVATED_THRESHOLD} alarm line; the highest score "
                   f"in the ladder is {peak:.6f} at util {peak_util:.2f}. A tree ensemble "
                   f"cannot emit a leaf value it never trained on, so an overload worse "
                   f"than anything in the two recorded years is NOT flagged. Reported, not "
                   f"fixed: fixing it needs capacity as a feature (which is the label) or a "
                   f"monotonic constraint, and either would retrain the model and change "
                   f"every number above.")


# ---------------------------------------------------------------------- 7
def check_live(base_url: str, df: pd.DataFrame, model, local: dict,
               scored: pd.DataFrame) -> None:
    print(f"\n=== 7. Deployed agreement ({base_url}) ===")
    try:
        served = get_json(f"{base_url}/validation")
    except Exception as exc:  # noqa: BLE001 - report any transport failure as a skip
        report("live check", f"skipped, {type(exc).__name__}: {exc}")
        return

    disk_path = MODELS_DIR / "closed_loop_report.json"
    if disk_path.exists():
        disk = json.loads(disk_path.read_text())
        same = all(served.get(k) == disk.get(k) for k in ("gates", "metrics", "window", "overall"))
        check("served /validation == committed closed_loop_report.json", same,
              f"overall={served.get('overall')}")
        n_pass = sum(1 for g in served["gates"] if g["result"] == "PASS")
        n_warn = sum(1 for g in served["gates"] if g["result"] == "WARN")
        n_fail = sum(1 for g in served["gates"] if g["result"] == "FAIL")
        report("served gates", f"{n_pass} PASS / {n_warn} WARN / {n_fail} FAIL")

    al = served["metrics"]["alarm"]
    tol = 5e-4
    for label, key in [("alarm precision", "precision"),
                       ("alarm recall", "recall"),
                       ("Brier skill", "brier_skill")]:
        check(f"served {label} reproduced by this script",
              abs(float(al.get(key, float("nan"))) - local[key]) <= tol,
              f"served {al.get(key)} vs recomputed {local[key]:.4f}")
    check("served hours_scored == rows scored here",
          int(al.get("hours_scored", -1)) == len(scored),
          f"served {al.get('hours_scored')} vs local {len(scored):,}")
    check("served positive_rate == positives counted here",
          abs(float(al.get("positive_rate", -1)) - local["base_rate"]) <= 5e-5,
          f"served {al.get('positive_rate')} vs local {local['base_rate']:.4f}")

    pz = al.get("per_zone") or {}
    if isinstance(pz, dict) and "zone" in scored.columns:
        for zone, g in scored.groupby("zone"):
            pos = int(g["_y"].sum())
            if pos == 0 or zone not in pz:
                continue
            hit = int(((g["_prob"] >= ELEVATED_THRESHOLD) & (g["_y"] == 1)).sum())
            entry = pz[zone]
            served_rec = entry.get("recall") if isinstance(entry, dict) else None
            if served_rec is None:
                continue
            check(f"served {zone} recall reproduced",
                  abs(float(served_rec) - hit / pos) <= tol,
                  f"served {served_rec} vs local {hit / pos:.4f}")

    zones = sorted(df["zone"].unique()) if "zone" in df.columns else []
    for zone in zones:
        try:
            live = get_json(f"{base_url}/risk/{zone}")
        except Exception as exc:  # noqa: BLE001
            report(f"/risk/{zone}", f"skipped, {type(exc).__name__}")
            continue
        ts = pd.Timestamp(live["snapshot_timestamp"])
        row = df[(df["zone"] == zone) & (df["timestamp"] == ts)]
        if len(row) != 1:
            check(f"{zone}: served snapshot row is in the committed frame", False,
                  f"{ts} → {len(row)} rows")
            continue
        mine = float(model.predict_proba(row[FEATURE_COLS])[:, 1][0])
        delta = abs(mine - float(live["risk_score"]))
        check(f"{zone}: deployed risk_score reproduced by this artifact",
              delta < 1e-12,
              f"live {live['risk_score']:.12f} [{live['status']}] vs local "
              f"{mine:.12f} [{dashboard_status(mine)}] | |delta|={delta:.3e}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="https://grid-senseai-backend.fastapicloud.dev")
    ap.add_argument("--skip-live", action="store_true")
    args = ap.parse_args()

    df = load_frame()
    model = joblib.load(MODELS_DIR / "risk_classifier.joblib")
    print(f"Artifacts: training_features.parquet ({len(df):,} rows, "
          f"{df['timestamp'].min()} → {df['timestamp'].max()}) | risk_classifier.joblib "
          f"({type(model).__name__})")

    check_label_integrity(df)
    check_leakage(model)
    _, test_part = check_split(df)
    scored, local_metrics = score_held_out(model, test_part)
    check_per_zone(scored)
    check_response_range(model, df)
    if args.skip_live:
        report("live check", "skipped by --skip-live")
    else:
        check_live(args.base_url, df, model, local_metrics, scored)

    print("\n=== Summary ===")
    failed = [n for n, ok, _ in results if not ok]
    print(f"checks run {len(results)} | passed {len(results) - len(failed)} | failed {len(failed)}")
    if failed:
        print("FAILED: " + "; ".join(failed))
        return 1
    print("VERDICT: the label matches its documented rule exactly, no feature can read the "
          "label off the frame, the scored window strictly postdates the fitted window, and "
          "every figure above was recomputed independently of the served report.")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except urllib.error.HTTPError as exc:
        print(f"HTTP {exc.code} from {exc.url}: {exc.read()[:300]!r}")
        sys.exit(2)
