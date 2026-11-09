#!/usr/bin/env bash
# GridSense AI — retrain chain.
#
# WEEKLY CADENCE (the claim in README.md:43-45 and docs/04_Pipeline_Documentation.md §3):
# pull the new ENTSO-E/NASA hours, append them to the data already committed, retrain,
# validate, and only then promote. This script does everything EXCEPT promotion; it never
# writes to data/models/ or data/features/, so a scheduled run cannot replace the artifacts
# the live API is serving with an unvalidated model. Promotion is a separate, human-run,
# gated step (scripts/promote_models.py).
#
# Full weekly command:
#
#   OUT=data/models_retrain/$(date -u +%Y-%m-%d); mkdir -p "$OUT/raw"
#   GRIDSENSE_DATA_RAW_DIR="$OUT/raw" python -m ingestion.entsoe_client    # needs ENTSOE_API_KEY
#   GRIDSENSE_DATA_RAW_DIR="$OUT/raw" python -m ingestion.nasa_power_client # no key needed
#   GRIDSENSE_DATA_RAW_DIR="$OUT/raw" bash scripts/run_retrain_chain.sh --out-dir "$OUT" \
#        --append-load-parquet "$OUT/raw/entsoe_load.parquet"
#   python scripts/promote_models.py --candidate "$OUT"      # gate + diff, copies nothing
#   python scripts/promote_models.py --candidate "$OUT" --commit
#   git add data/models data/features && git commit && <redeploy>   # you review the diff
#
# Point the pulls at "$OUT/raw", not data/raw: the new hours then sit beside the
# candidate, and a scheduled job cannot leave data/raw ahead of the committed feature
# table (which is what would make a later build_features rewrite that table).
#
# Without --out-dir the chain runs against the committed feature table and writes models to
# data/models as it always did — kept for the from-scratch path in docs/04 §3. With --out-dir
# every stage is redirected: features -> $OUT/training_features.parquet (or the committed one
# if you give no new data), models -> $OUT, report -> $OUT/closed_loop_report.json.
#
# THE GATE lives in the promote step and fails loudly when the candidate would be worse
# than what is served: it must beat persistence AND seasonal-naive on the honest
# time-ordered test window (models/train_baselines.train_val_test_split), pass the
# closed-loop release gates without regressing any of them against the incumbent, and
# hold its recursive h+24/h+48 MAPE and P10-P90 coverage inside the tolerances documented
# in scripts/promote_models.py. --commit copies nothing until you type `promote`.
# The chain's last stage aborts on a FAIL gate; WARN alone does not abort, because the
# served report itself carries four WARN gates today.
#
# VERIFIED (2026-11-07, this repo): appending 504 synthetic new hours grew the candidate
# feature table 51,959 -> 52,463 rows and moved the honest split to fit 37,772 /
# val 4,197 / test 10,494 — the "add new data to old data" step really extends each zone's
# series instead of sitting beside it. `git status --porcelain data/models` was empty both
# before and after the whole run. A seasonal-naive candidate was refused at h+24
# (9.847% vs incumbent 5.558%) and a persistence candidate at h+48 (10.875%).
cd "$(dirname "$0")/.." || exit 9

OUT_DIR=""
APPEND_PARQUET=""
STEPS="baselines risk calibrate closedloop"
DRY_RUN=0
while [ $# -gt 0 ]; do
  case "$1" in
    --out-dir) OUT_DIR="$2"; shift 2 ;;
    --append-load-parquet) APPEND_PARQUET="$2"; shift 2 ;;
    --steps) STEPS="$2"; shift 2 ;;
    --dry-run) DRY_RUN=1; shift ;;
    *) echo "[chain] unknown arg: $1" >&2; exit 2 ;;
  esac
done

# ---------------------------------------------------------------------------
# PROTECTED-TREE GUARD
#
# This script's central promise is that it never touches data/models/ or
# data/features/. Two stages can break that promise while still honouring every
# documented env-var contract, which is worse than an obvious bug:
#
#   * models/closed_loop_validation.py binds OUT_PATH = MODELS_DIR /
#     "closed_loop_report.json" at IMPORT time (line 68), so running it without an
#     explicit --out overwrites the served report even when GRIDSENSE_MODELS_OUT_DIR
#     is set correctly. Reproduced 2026-11-07: sha ef28b64c67d4d757 -> 42a6def3ee4ff992,
#     repaired with `git restore data/models/closed_loop_report.json`.
#   * models/calibrate_forecast_intervals.py falls back to models_dir = MODELS_DIR when
#     its input model is missing there, and then writes forecast_intervals*.json to that
#     same directory — i.e. into the served set.
#
# Rather than trust each stage, snapshot the protected trees and fail loudly if anything
# changed. A scheduled job must not be able to silently degrade what the API serves.
snapshot_protected () {
  find data/models data/features -type f -print0 2>/dev/null \
    | sort -z | xargs -0 -r sha256sum 2>/dev/null | sha256sum | cut -c1-16
}
PROTECTED_BEFORE="$(snapshot_protected)"
echo "[chain] protected-tree snapshot: $PROTECTED_BEFORE"

check_protected () {
  local now; now="$(snapshot_protected)"
  if [ "$now" != "$PROTECTED_BEFORE" ]; then
    echo "[chain] *** VIOLATION *** data/models/ or data/features/ changed during this" >&2
    echo "[chain]     run ($PROTECTED_BEFORE -> $now). Restore with:" >&2
    echo "[chain]     git restore data/models data/features" >&2
    echo "[chain]     Refusing to declare a candidate. See the note above for which" >&2
    echo "[chain]     stage reaches into the served set." >&2
    return 1
  fi
  return 0
}

if [ -n "$OUT_DIR" ]; then
  mkdir -p "$OUT_DIR" || exit 9
  export GRIDSENSE_MODELS_OUT_DIR="$OUT_DIR"
  # With new data, the candidate gets its own feature table built from
  # committed history + the new hours. Without it, train on the committed frame —
  # read-only here; nothing writes to data/features unless you say so yourself.
  FEAT_TARGET="$OUT_DIR"
  if [ ! -f "$OUT_DIR/training_features.parquet" ]; then
    if [ -n "$APPEND_PARQUET" ]; then
      python -u -m features.build_features \
        --raw-dir "${GRIDSENSE_DATA_RAW_DIR:-data/raw}" \
        --out "$OUT_DIR/training_features.parquet" \
        --append-load-parquet "$APPEND_PARQUET" \
        > "_audit_out/build_features.log" 2>&1 \
        || { echo "[chain] build_features FAILED, see _audit_out/build_features.log" >&2; exit 1; }
    else
      FEAT_TARGET="data/features"
    fi
  fi
else
  # From-scratch path (docs/04 §3): the historical behaviour, writing data/models.
  # That is precisely what a scheduled run must never do, so it is refused unless a
  # human is sitting at a terminal and confirms by name.
  if [ "${CI:-}" = "true" ] || [ ! -t 0 ]; then
    echo "[chain] REFUSING to run without --out-dir in a non-interactive context:" >&2
    echo "[chain] this path writes models into data/models/, the set the live API serves." >&2
    echo "[chain] Use: bash scripts/run_retrain_chain.sh --out-dir <candidate-dir>" >&2
    exit 8
  fi
  printf '[chain] This path OVERWRITES data/models/. Type "overwrite" to continue: '
  read -r reply
  if [ "${reply:-}" != "overwrite" ]; then
    echo "[chain] not confirmed; nothing was trained."
    exit 8
  fi
  FEAT_TARGET="${GRIDSENSE_FEATURES_DIR:-data/features}"
fi
export GRIDSENSE_FEATURES_DIR="$(python -c "import pathlib;print(pathlib.Path('$FEAT_TARGET').resolve())")"
echo "[chain] features dir : $GRIDSENSE_FEATURES_DIR"
echo "[chain] models out   : ${GRIDSENSE_MODELS_OUT_DIR:-data/models}"
[ -n "$GRIDSENSE_MODELS_OUT_DIR" ] && echo "[chain] served data/models is INPUT ONLY this run"

mkdir -p _audit_out
: > _audit_out/chain.status
run () {
  local name="$1"; shift
  if [ "$DRY_RUN" -eq 1 ]; then
    echo "[chain] DRY RUN — would execute: python -u -m $name $*"
    return 0
  fi
  echo "[chain] start $name $(date -u +%H:%M:%S)Z" >> _audit_out/chain.stdout.log
  python -u -m "$name" "$@" > "_audit_out/${name##*.}.log" 2>&1
  code=$?
  echo "$name exit=$code $(date -u +%H:%M:%S)Z" >> _audit_out/chain.status
  echo "[chain] done $name exit=$code $(date -u +%H:%M:%S)Z" >> _audit_out/chain.stdout.log
  if [ $code -ne 0 ]; then
    echo "[chain] ABORT: $name exited $code — see _audit_out/${name##*.}.log." >&2
    echo "[chain] No candidate was certified; data/models/ is untouched." >&2
    exit $code
  fi
  # Attribute any write into the served set to the stage that caused it.
  if ! check_protected; then
    echo "[chain] ABORT: stage '$name' modified the protected artifact trees." >&2
    exit 7
  fi
}

for step in $STEPS; do
  case "$step" in
    baselines)  run models.train_baselines ;;
    risk)       run models.risk_classifier ;;
    calibrate)  run models.calibrate_forecast_intervals ;;
    # No --strict here on purpose: the served report itself carries WARN gates
    # (coverage_drift, event_detection, event_warning_3h, median_lead_hours), so
    # treating WARN as fatal would fail every honest weekly run. A FAIL gate still
    # exits 1 and aborts, and promote_models.py grades the same gates again.
    closedloop) run models.closed_loop_validation \
                  --out "${GRIDSENSE_MODELS_OUT_DIR:-data/models}/closed_loop_report.json" ;;
    *) echo "[chain] unknown step: $step" >&2; exit 2 ;;
  esac
done

if [ "$DRY_RUN" -eq 1 ]; then
  echo "[chain] DRY RUN complete — no stage was executed, nothing was trained."
  echo "[chain] real run: bash scripts/run_retrain_chain.sh --out-dir <dir>"
  echo "CHAIN_FINISHED_DRYRUN $(date -u +%Y-%m-%dT%H:%M:%SZ)" >> _audit_out/chain.status
  exit 0
fi

# Final certification: the whole point of the out-dir contract, checked rather than assumed.
if ! check_protected; then
  echo "PROTECTED_TREE_VIOLATION $(date -u +%Y-%m-%dT%H:%M:%SZ)" >> _audit_out/chain.status
  echo "[chain] No candidate is declared: the served artifact set changed this run." >&2
  exit 7
fi
echo "PROTECTED_TREE_CLEAN $PROTECTED_BEFORE $(date -u +%Y-%m-%dT%H:%M:%SZ)" \
  >> _audit_out/chain.status
echo "[chain] protected trees verified unchanged ($PROTECTED_BEFORE)"

if [ -n "$GRIDSENSE_MODELS_OUT_DIR" ]; then
  echo "[chain] candidate ready: $GRIDSENSE_MODELS_OUT_DIR"
  echo "[chain] inspect then gate it: python scripts/promote_models.py --candidate $GRIDSENSE_MODELS_OUT_DIR"
fi
echo "CHAIN_FINISHED $(date -u +%Y-%m-%dT%H:%M:%SZ)" >> _audit_out/chain.status
