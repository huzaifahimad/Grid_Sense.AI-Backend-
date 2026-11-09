"""
GridSense AI — Promote a retrain candidate into the served artifact set.

The chain (`scripts/run_retrain_chain.sh`) trains into an isolated, dated candidate
directory. Nothing in it writes to `data/models/`, because that directory is what the
deployed API loads at startup: overwriting it in place with an unvalidated model is how
a scheduled job takes down a working system. Promotion is therefore a separate, gated
step, and this script is the only writer of `data/models/`.

    python scripts/promote_models.py --candidate data/models_retrain/2026-11-12        # gate + show
    python scripts/promote_models.py --candidate ... --commit                          # gate + copy
    python scripts/promote_models.py --candidate ... --commit --yes                    # non-interactive (CI)
    python scripts/promote_models.py --candidate ... --target <dir> --commit           # drill target

THE GATE — every check below refuses promotion (exit 1) and copies nothing:

  1. Baselines. On the honest time-ordered TEST window (`train_val_test_split` in
     models/train_baselines.py), the candidate's one-step predictions must beat both
     persistence (load t-24h) and seasonal-naive (load t-168h). Recomputed from the
     candidate itself — not read off a CSV someone trusted.
  2. Closed-loop release gates on the candidate artifacts. No FAIL gate, and no gate
     that the incumbent passes may drop to WARN or FAIL in the candidate: a scheduled
     job must not be able to trade away a capability the served model has.
  3. RECURSIVE multi-step MAPE at h+24 and h+48 on the CONTIGUOUS FUTURE window,
     measured by identical code for candidate and incumbent. This is the headline
     check. It is never a shuffled split: docs/02_ML_Model_Design.md §5.4 measured that
     a shuffled 80/10/10 split puts 80.53% of the honest test window inside training and
     inflates accuracy by 9-17% at every horizon, and I reproduced the direction of that
     error locally before writing this gate.
  4. P10-P90 BAND COVERAGE at long lead (h+24, h+48). An under-covering band is the
     actual product failure mode: the true load falls outside the interval more often
     than advertised, so warnings that should have fired do not. Coverage is scored
     against BOTH a floor and the incumbent-relative tolerance, so a candidate cannot
     pass by widening bands until it covers everything.
  5. Absolute floors derived from the incumbent's own measured performance.

TOLERANCES AND WHERE THEY COME FROM
-----------------------------------
Measured in this repo, on the committed artifacts, on the contiguous future window
(834 origins, 3 zones; command in README of the gate output):

    incumbent h+24 MAPE = 5.558 %   coverage = 0.8561
    incumbent h+48 MAPE = 7.167 %   coverage = 0.8022
    incumbent overall P10-P90 coverage = 0.8322 (nominal 0.80)

  * MAPE_MAX_RELATIVE_REGRESSION_PCT = 5.0 — a candidate may be up to 5% relatively
    worse than the incumbent at h+24/h+48. This is NOT a measured noise floor: I did not
    measure run-to-run retrain variance (no new ENTSO-E hours were pulled here), so this
    number is a stated engineering tolerance, deliberately tighter than the 10 pp the
    previous version of this file used, and it is applied only on top of the absolute
    floors. Change it knowingly, not silently.
  * COVERAGE_MIN_LONG_LEAD = 0.70 — the closed-loop release gate already uses 0.70 as its
    lower band for overall coverage (models/closed_loop_validation.py, gate `band_coverage`,
    target=[0.7, 0.95]). Reusing it keeps one definition of "unacceptable" across the repo
    rather than inventing a second.
  * COVERAGE_MAX_ABSOLUTE_DROP = 0.05 — same value the previous gate used, expressed now
    as an absolute coverage-point drop at each long lead step.
  * MAPE_FLOOR_H24 / H48 = 8.0 / 10.0 % — absolute ceilings independent of the incumbent,
    so a broken incumbent cannot make a broken candidate look acceptable. They are set
    just below the incumbent's own seasonal-naive reference at those steps (9.847 % /
    9.860 %), i.e. a promotable model must stay better than naive-by-a-margin.

Exit codes: 0 = gate passed (and promoted if --commit), 1 = gate refused,
2 = candidate incomplete / environment unusable.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

for _stream in (sys.stdout, sys.stderr):  # Windows console defaults to cp1252
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")

import joblib
import numpy as np
import pandas as pd

from config import DATA_FEATURES_DIR, MODELS_DIR
from models.calibrate_forecast_intervals import REPORT_STEPS, calibrate
from models.train_baselines import (
    FEATURE_COLS_NO_WEATHER, TARGET_COL, train_val_test_split)

# ---------------------------------------------------------------- tolerances
LONG_LEAD_STEPS = (24, 48)          # the two leads the product actually warns on
MAPE_MAX_RELATIVE_REGRESSION_PCT = 5.0
COVERAGE_MIN_LONG_LEAD = 0.70
COVERAGE_MAX_ABSOLUTE_DROP = 0.05
# How far empirical coverage may sit below the band's advertised nominal (0.80) before we
# call the band dishonest. Chosen so 0.80 - this = 0.70, which is exactly the lower bound
# models/closed_loop_validation.py already uses for its `band_coverage` FAIL: one shared
# definition of "unacceptable" instead of a second number invented here and dressed up as
# measured. For reference, the incumbent's own measured long-lead coverage is
# 0.8561 at h+24 and 0.8022 at h+48.
COVERAGE_MAX_UNDER_COVERAGE = 0.10
MAPE_FLOOR_H24 = 8.0
MAPE_FLOOR_H48 = 10.0
MIN_TEST_ORIGINS = 200              # below this the long-lead numbers are noise

# Closed-loop gates that must clear no matter what the incumbent itself does. Deliberately
# narrow: only the gate whose failure IS the product defect we are shipping against.
# `band_coverage` measures whether the P10-P90 interval actually contains the truth, so an
# under-covering band manufactures missed warnings; its bounds are the repo's own existing
# target range [0.70, 0.95] in models/closed_loop_validation.py, not a number invented here.
ABSOLUTE_FLOOR_GATES = {"band_coverage"}


class IncumbentUnmeasurable(RuntimeError):
    """The served artifacts could not be scored, so the candidate has nothing to be
    compared against. Promotion fails closed: absence of a baseline is not a pass."""


CANDIDATE_FILES = [
    "lightgbm_load_forecast.joblib",
    "lightgbm_load_forecast_no_weather.joblib",
    "risk_classifier.joblib",
    "risk_factors_by_bucket.json",
    "forecast_intervals.json",
    "forecast_intervals_no_weather.json",
    "baseline_results.csv",
    "closed_loop_report.json",
]


def _mape(actual: np.ndarray, pred: np.ndarray) -> float:
    return float(np.mean(np.abs(actual - pred) / actual) * 100)


def one_step_baselines(df: pd.DataFrame, test: pd.DataFrame, candidate_dir: Path) -> dict:
    """Gate 1: candidate vs persistence and seasonal-naive on the test window."""
    model = joblib.load(candidate_dir / "lightgbm_load_forecast_no_weather.joblib")
    cols = FEATURE_COLS_NO_WEATHER
    part = test.dropna(subset=cols + [TARGET_COL])
    # Scored with the SAME input shape the serving path uses (a named frame, not a
    # bare array): a model that reads its features by name — including a negative-
    # control stub — must be scorable here, or this gate only tests sklearn-shaped
    # candidates and a stub can pass it unread.
    pred = model.predict(part[cols])
    y = part[TARGET_COL].to_numpy(float)
    out = {"rows": int(len(part)),
           "candidate_mape_pct": round(_mape(y, pred), 3)}
    for label, col in (("persistence_t24h", "load_lag_24h"),
                       ("seasonal_naive_t168h", "load_lag_168h")):
        v = part.dropna(subset=[col])
        out[f"{label}_mape_pct"] = round(_mape(v[TARGET_COL].to_numpy(float),
                                               v[col].to_numpy(float)), 3)
    out["beats_persistence"] = bool(out["candidate_mape_pct"] < out["persistence_t24h_mape_pct"])
    out["beats_seasonal_naive"] = bool(out["candidate_mape_pct"] < out["seasonal_naive_t168h_mape_pct"])
    print(f"[promote] one-step test-window MAPE: candidate {out['candidate_mape_pct']:.3f}% "
          f"vs persistence {out['persistence_t24h_mape_pct']:.3f}% "
          f"vs seasonal-naive {out['seasonal_naive_t168h_mape_pct']:.3f}%")
    return out


def recursive_eval(model, df: pd.DataFrame, val: pd.DataFrame, test: pd.DataFrame,
                   bands_dir: Path, warn_fallback: bool = True) -> dict:
    """
    The serving-path measurement, identical code for candidate and incumbent.

    calibrate() computes and returns the evaluation; only its main() writes, so this
    cannot touch any artifact directory. `bands_dir` supplies the P10/P90 quantiles the
    band is scored with. A candidate mid-chain has no band of its own yet, and judging a
    fresh retrain against the incumbent's band would measure the wrong thing, so we fall
    back to the served dir and say so out loud.
    """
    if (bands_dir / "forecast_intervals.json").exists():
        band_source = bands_dir
    elif warn_fallback:
        print(f"[promote] NOTE: no forecast_intervals.json in {bands_dir}; scoring the "
              "band from the served dir so the comparison still has a denominator")
        band_source = MODELS_DIR
    else:
        band_source = bands_dir
    art = calibrate(model, df, val, test, FEATURE_COLS_NO_WEATHER,
                    "no-weather fallback", models_dir=band_source)
    art["evaluation"]["_nominal_coverage"] = art.get("nominal_coverage", 0.80)
    art["_band_source"] = str(band_source)
    return art["evaluation"]


def long_lead_comparison(eval_incumbent: dict, eval_candidate: dict,
                         nominal_coverage: float = 0.80) -> dict:
    """Gate 3/4: recursive MAPE + band coverage at h+24/h+48, candidate vs incumbent."""
    steps = {}
    worst_mape = -np.inf
    cov_drop = -np.inf
    for h in LONG_LEAD_STEPS:
        k = str(h)
        c = eval_candidate["per_step"][k]
        i = eval_incumbent["per_step"][k]
        c_mape, i_mape = c["mape_recursive_lgbm"], i["mape_recursive_lgbm"]
        rel = (c_mape - i_mape) / i_mape * 100 if i_mape else float("inf")
        drop = i["coverage"] - c["coverage"]
        worst_mape = max(worst_mape, rel)
        cov_drop = max(cov_drop, drop)
        floor = MAPE_FLOOR_H24 if h == 24 else MAPE_FLOOR_H48
        steps[k] = {
            "candidate_mape_pct": c_mape, "incumbent_mape_pct": i_mape,
            "worse_pct": round(rel, 2),
            "candidate_coverage": c["coverage"], "incumbent_coverage": i["coverage"],
            "coverage_drop": round(drop, 4),
            "mape_floor_pct": floor,
            "nominal_coverage": nominal_coverage,
            "seasonal_naive_mape_pct": c["mape_seasonal_naive"],
        }
    return {
        "steps": steps,
        "long_lead_steps": list(LONG_LEAD_STEPS),
        "worst_step_worse_pct": round(worst_mape, 2),
        "worst_coverage_drop": round(cov_drop, 4),
        "candidate_coverage_all_steps": eval_candidate["p10_p90_coverage_overall"],
        "incumbent_coverage_all_steps": eval_incumbent["p10_p90_coverage_overall"],
        "test_origins": eval_candidate.get("test_origins"),
        "tolerances": {
            "max_relative_mape_regression_pct": MAPE_MAX_RELATIVE_REGRESSION_PCT,
            "min_long_lead_coverage": COVERAGE_MIN_LONG_LEAD,
            "max_coverage_drop": COVERAGE_MAX_ABSOLUTE_DROP,
            "max_under_coverage_vs_nominal": COVERAGE_MAX_UNDER_COVERAGE,
            "mape_floor_pct": {"24": MAPE_FLOOR_H24, "48": MAPE_FLOOR_H48},
            "min_test_origins": MIN_TEST_ORIGINS,
        },
    }


def check(name: str, ok: bool, detail: str = "") -> bool:
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f" — {detail}" if detail else ""))
    return ok


def compare_gate_results(incumbent: dict, candidate: dict) -> list[str]:
    """Names of gates the incumbent clears but the candidate does not."""
    regressed = []
    inc_map = {g["id"]: g["result"] for g in incumbent["gates"]}
    rank = {"PASS": 2, "WARN": 1, "FAIL": 0}
    for g in candidate["gates"]:
        before = inc_map.get(g["id"], "PASS")
        if rank[g["result"]] < rank[before]:
            regressed.append(f"{g['id']}:{before}->{g['result']}")
    return regressed


def evaluate_gate(candidate: Path, target: Path) -> tuple[list[bool], dict]:
    """Run every gate check. Writes nothing outside `candidate`."""
    checks: list[bool] = []
    evidence: dict = {}

    missing = [f for f in CANDIDATE_FILES if not (candidate / f).exists()]
    if missing:
        raise FileNotFoundError(
            f"candidate is incomplete, missing: {', '.join(missing)}. Finish the chain "
            f"first: bash scripts/run_retrain_chain.sh --out-dir {candidate}")

    df = pd.read_parquet(DATA_FEATURES_DIR / "training_features.parquet")
    _, val, test = train_val_test_split(df)

    print("\n=== 1. baselines on the honest time-ordered test window ===")
    one_step = one_step_baselines(df, test, candidate)
    evidence["one_step_test_window"] = one_step
    checks.append(check("candidate beats persistence (t-24h)", one_step["beats_persistence"]))
    checks.append(check("candidate beats seasonal-naive (t-168h)",
                        one_step["beats_seasonal_naive"]))

    print("\n=== 2. closed-loop release gates: candidate vs incumbent ===")
    from models.closed_loop_validation import run as closed_loop_run
    cl_cand = closed_loop_run(stride=6, horizon=24, write=False, models_dir=candidate)
    # The incumbent MUST be measurable. Its committed report says WARN for
    # median_lead_hours (value 1.0), but re-running the same code against the served
    # artifacts gives FAIL at value 0.0, twice, deterministically. So "no FAIL gate"
    # measured in isolation would refuse even a candidate identical to the incumbent —
    # a permanently un-promotable model teaches operators to bypass the gate. Hence:
    # compare like with like, and keep absolute floors only for the gates that are the
    # product failure mode itself (an under-covering band manufacturing missed warnings).
    try:
        cl_inc = closed_loop_run(stride=6, horizon=24, write=False, models_dir=target)
    except Exception as exc:
        raise IncumbentUnmeasurable(
            f"cannot score the incumbent's closed-loop gates at {target}: "
            f"{type(exc).__name__}: {exc}") from exc

    inc_failed = [g["id"] for g in cl_inc["gates"] if g["result"] == "FAIL"]
    cand_failed = [g["id"] for g in cl_cand["gates"] if g["result"] == "FAIL"]
    regressed = compare_gate_results(cl_inc, cl_cand)
    checks.append(check("no closed-loop gate worse than the incumbent",
                        not regressed, ", ".join(regressed) or "none"))
    absolute_breaches = [g["id"] for g in cl_cand["gates"]
                         if g["id"] in ABSOLUTE_FLOOR_GATES and g["result"] == "FAIL"]
    checks.append(check(
        f"absolute-floor gates clear ({', '.join(sorted(ABSOLUTE_FLOOR_GATES))})",
        not absolute_breaches,
        f"FAIL={absolute_breaches or 'none'}"))
    new_failures = sorted(set(cand_failed) - set(inc_failed))
    checks.append(check("candidate introduces no new FAIL gate", not new_failures,
                        f"new={new_failures or 'none'}; incumbent already FAILs: "
                        f"{inc_failed or 'none'}"))
    warn = [g["id"] for g in cl_cand["gates"] if g["result"] == "WARN"]
    evidence["closed_loop"] = {
        "candidate_overall": cl_cand["overall"], "incumbent_overall": cl_inc["overall"],
        "candidate_failed_gates": cand_failed, "warn_gates": warn,
        "incumbent_failed_gates": inc_failed,
        "regressed_vs_incumbent": regressed,
        "preexisting_incumbent_defects": sorted(set(inc_failed) - set(cand_failed)),
    }

    print("\n=== 3/4. recursive long-lead MAPE + band coverage, same rows, same code ===")
    incumbent_model = joblib.load(target / "lightgbm_load_forecast_no_weather.joblib")
    candidate_model = joblib.load(candidate / "lightgbm_load_forecast_no_weather.joblib")
    eval_inc = recursive_eval(incumbent_model, df, val, test, target)
    eval_cand = recursive_eval(candidate_model, df, val, test, candidate)
    cmp = long_lead_comparison(eval_inc, eval_cand,
                               nominal_coverage=float(eval_cand.get("_nominal_coverage",
                                                                     0.80)))
    evidence["vs_incumbent"] = cmp
    evidence["band_sources"] = {"incumbent": eval_inc.get("_band_source"),
                               "candidate": eval_cand.get("_band_source")}

    origins = cmp["test_origins"] or 0
    checks.append(check(f"enough contiguous future origins (>= {MIN_TEST_ORIGINS})",
                        origins >= MIN_TEST_ORIGINS, f"got {origins}"))

    for h, s in cmp["steps"].items():
        checks.append(check(
            f"h+{h} recursive MAPE not materially worse (<= "
            f"{MAPE_MAX_RELATIVE_REGRESSION_PCT}% relative vs incumbent)",
            s["worse_pct"] <= MAPE_MAX_RELATIVE_REGRESSION_PCT,
            f"candidate {s['candidate_mape_pct']}% vs incumbent {s['incumbent_mape_pct']}% "
            f"({s['worse_pct']:+.2f}%)"))
        checks.append(check(
            f"h+{h} recursive MAPE under absolute floor {s['mape_floor_pct']}%",
            s["candidate_mape_pct"] <= s["mape_floor_pct"],
            f"candidate {s['candidate_mape_pct']}%"))
        checks.append(check(
            f"h+{h} P10-P90 coverage >= {COVERAGE_MIN_LONG_LEAD}",
            s["candidate_coverage"] >= COVERAGE_MIN_LONG_LEAD,
            f"candidate {s['candidate_coverage']} (incumbent {s['incumbent_coverage']})"))
        checks.append(check(
            f"h+{h} P10-P90 coverage drop <= {COVERAGE_MAX_ABSOLUTE_DROP}",
            s["coverage_drop"] <= COVERAGE_MAX_ABSOLUTE_DROP,
            f"drop {s['coverage_drop']:+.4f}"))
        # Measured against the band's OWN advertised nominal level, not only the incumbent:
        # an incumbent-relative rule alone would pass a candidate that matches an equally
        # badly calibrated incumbent, which is how silent degradation looks on the way in.
        checks.append(check(
            f"h+{h} P10-P90 coverage within {COVERAGE_MAX_UNDER_COVERAGE} of nominal "
            f"{s['nominal_coverage']} (band must deliver what it advertises)",
            s["candidate_coverage"] >= s["nominal_coverage"] - COVERAGE_MAX_UNDER_COVERAGE,
            f"candidate {s['candidate_coverage']} vs nominal {s['nominal_coverage']}"))
        checks.append(check(
            f"h+{h} recursive MAPE beats seasonal-naive",
            s["candidate_mape_pct"] < s["seasonal_naive_mape_pct"],
            f"candidate {s['candidate_mape_pct']}% vs naive "
            f"{s['seasonal_naive_mape_pct']}%"))

    return checks, evidence


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=(__doc__ or "").strip().splitlines()[0])
    ap.add_argument("--candidate", type=Path, required=True,
                    help="the dated retrain output dir produced by run_retrain_chain.sh")
    ap.add_argument("--target", type=Path, default=MODELS_DIR,
                    help="served artifact dir; only change it for a drill")
    ap.add_argument("--commit", action="store_true", help="actually copy on a passing gate")
    ap.add_argument("--yes", action="store_true", help="skip the interactive confirmation")
    args = ap.parse_args(argv)

    candidate, target = args.candidate.resolve(), args.target.resolve()
    if candidate == target:
        print("[promote] REFUSED — candidate and target are the same directory; a gate "
              "measured against itself proves nothing.")
        return 2
    if target == MODELS_DIR:
        print(f"[promote] TARGET IS THE LIVE SERVED SET: {target}")
    print(f"[promote] candidate: {candidate}")

    try:
        checks, evidence = evaluate_gate(candidate, target)
    except FileNotFoundError as exc:
        print(f"[promote] REFUSED — {exc}")
        return 2
    except IncumbentUnmeasurable as exc:
        # No baseline means no comparison, and a gate that cannot run must never be read
        # as a pass. Exit 2 distinguishes "could not measure" from "measured and refused".
        print(f"[promote] REFUSED (gate could not run) — {exc} Nothing was copied.")
        return 2

    decision = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "candidate": str(candidate), "target": str(target),
        "committed": False, "passed": all(checks),
        **evidence,
    }
    (candidate / "retrain_decision.json").write_text(json.dumps(decision, indent=2))
    print(f"\n[promote] evidence written -> {candidate / 'retrain_decision.json'}")

    if not decision["passed"]:
        print("[promote] REFUSED — the candidate is worse than the incumbent or fails a "
              "release gate. Nothing was copied.")
        return 1

    if not args.commit:
        print("[promote] gate PASSED. Dry run only — add --commit to promote.")
        return 0

    if not args.yes:
        reply = input(f"[promote] copy {len(CANDIDATE_FILES)} artifact(s) into {target}? "
                      "Type 'promote' to confirm: ")
        if reply.strip() != "promote":
            print("[promote] not confirmed, nothing copied.")
            return 1

    backup = target.parent / f"models_pre_promote_{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}"
    shutil.copytree(target, backup)
    print(f"[promote] incumbent backed up -> {backup}")
    for name in CANDIDATE_FILES:
        shutil.copy2(candidate / name, target / name)
    decision["committed"] = True
    decision["incumbent_backup"] = str(backup)
    (candidate / "retrain_decision.json").write_text(json.dumps(decision, indent=2))
    print(f"[promote] promoted {len(CANDIDATE_FILES)} artifact(s) into {target}.")
    print("[promote] redeploy so the API reloads them, then re-run "
          "python scripts/verify_model_accuracy.py --skip-live to confirm agreement.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
