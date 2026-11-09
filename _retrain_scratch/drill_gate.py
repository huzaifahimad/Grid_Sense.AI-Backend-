"""Scratch drill: prove the gate refuses a regressed candidate and copies nothing,
and that data/models/ is never written. Not a shipped test."""
import json, shutil, sys, hashlib
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import joblib, numpy as np

ROOT = Path(__file__).resolve().parent.parent
SERVED = ROOT / "data" / "models"
SCRATCH = ROOT / "_retrain_scratch"

def digest(p: Path) -> str:
    h = hashlib.sha256()
    for f in sorted(p.rglob("*")):
        if f.is_file():
            h.update(f.name.encode()); h.update(f.read_bytes())
    return h.hexdigest()[:16]

before = digest(SERVED)
print("served data/models sha before:", before)

# --- incumbent snapshot used as the "good" candidate base ---
base = SCRATCH / "drill_base"
if base.exists():
    shutil.rmtree(base)
shutil.copytree(SERVED, base)

# --- CANDIDATE A: identical to the incumbent (should be accepted) ---
cand_ok = SCRATCH / "drill_candidate_ok"
if cand_ok.exists():
    shutil.rmtree(cand_ok)
shutil.copytree(base, cand_ok)

# --- CANDIDATE B: deliberately degraded forecaster (must be REFUSED) ---
cand_bad = SCRATCH / "drill_candidate_regressed"
if cand_bad.exists():
    shutil.rmtree(cand_bad)
shutil.copytree(base, cand_bad)

m = joblib.load(cand_bad / "lightgbm_load_forecast_no_weather.joblib")
rng = np.random.default_rng(7)

class Degraded:
    """Wrap the real booster: add noise + bias so long-lead recursive error compounds."""
    def __init__(self, inner):
        self.inner = inner
    def predict(self, X):
        X = np.asarray(X, dtype=float)
        out = self.inner.predict(X)
        noise = rng.normal(0, 0.04, size=out.shape) * out   # ~4% i.i.d. per step
        return out * (1 + noise + 0.01)                     # plus a small persistent bias
    def __getattr__(self, name):
        return getattr(self.inner, name)

joblib.dump(Degraded(m), cand_bad / "lightgbm_load_forecast_no_weather.joblib", compress=3)
print("degraded candidate model written")

# Also degrade the band so it under-covers (the product failure mode) — scale q's inward.
for name in ("forecast_intervals.json", "forecast_intervals_no_weather.json"):
    art = json.loads((cand_bad / name).read_text(encoding="utf-8"))
    for h, q in art["horizons"].items():
        q["q10"] = round(q["q10"] * 0.45, 5)
        q["q90"] = round(q["q90"] * 0.45, 5)
    (cand_bad / name).write_text(json.dumps(art, indent=2))
print("narrowed P10-P90 bands on the degraded candidate")

after_run = {}
for label, cand in (("EQUAL-OR-BETTER", cand_ok), ("REGRESSED", cand_bad)):
    print("\n" + "#" * 70)
    print(f"# DRILL: {label}  candidate={cand}")
    print("#" * 70)
    from scripts.promote_models import evaluate_gate
    checks, ev = evaluate_gate(cand, SERVED)
    print(f"-> all passed = {all(checks)}  ({sum(checks)}/{len(checks)} checks)")
    after_run[label] = (all(checks), ev)

print("\n=== SUMMARY ===")
for label, (ok, ev) in after_run.items():
    cmp = ev.get("vs_incumbent", {})
    print(f"{label:18s} gate_passed={ok}")
    for h, s in (cmp.get("steps") or {}).items():
        print(f"   h+{h}: cand mape {s['candidate_mape_pct']}% vs inc {s['incumbent_mape_pct']}% "
              f"({s['worse_pct']:+.2f}%) | cov cand {s['candidate_coverage']} "
              f"inc {s['incumbent_coverage']} drop {s['coverage_drop']:+.4f}")

now = digest(SERVED)
print("\nserved data/models sha after :", now)
print("UNCHANGED:", now == before)
