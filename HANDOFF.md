# HANDOFF — gridsense-backend

Written 2026-11-09. This file describes a **frozen, reviewed-by-nobody** working tree, not a release.

## STATE

`GridSense AI` backend: a FastAPI service that forecasts electricity load for three ENTSO-E
bidding zones (ES, GR, IT_SOUTH), scores overload risk, and serves a closed-loop validation
report. Real ENTSO-E load and real NASA POWER weather; `is_synthetic` is false. The live
deployment reads committed artifacts (`data/models/*.joblib`, `data/features/*.parquet`) and
appends fresh load hours in-request.

| | |
|---|---|
| Working branch | `wip/handoff-2026-11-09` @ **3a61618** (pre-HANDOFF.md) |
| Base / deployed | `main` @ **3853127**, identical to `origin/main` (verified: `rev-list --left-right --count` = `0 0`) |
| Divergence | Branch is 1 commit ahead of main, 35 files, +5,741/−95 |
| `main` pushed? | Yes. Nothing is stranded locally. The branch is the only unmerged work. |
| Commit, is it deployed? | **No.** Only `main` deploys. The live API still runs `3853127`. |

**Checks, as run on this tree (verbatim results):**

| Check | Result |
|---|---|
| `python -m pytest tests -q` | **157 passed, 6 skipped** in 187.83s (199 warnings) |
| `PYTHONPATH="$PWD" python scripts/verify_prediction_integrity.py` | **27 passed, 6 failed of 33** — see BROKEN below; not a model regression |
| Frontend `npm run build` | separate repo, **succeeded**, 865 modules, 14.62s |
| `git hash-object data/features/training_features.parquet` | equals `HEAD:data/features/training_features.parquet` (committed artifact intact) |

Reproduce exactly:

```bash
cd gridsense-backend
python -m pytest tests -q
PYTHONPATH="$PWD" python scripts/verify_prediction_integrity.py
```

Known warning, not fixed: `PytestUnknownMarkWarning: Unknown pytest.mark.gate_e2e` — the marker
is used in `tests/test_promote_gate_live.py:39` but never registered in pytest config.

## DONE

| Item | Evidence |
|---|---|
| Model choice measured, not argued: LightGBM served, RF/XGB head-to-head with paired bootstrap | `scripts/model_head_to_head.py`; `docs/02` §5; commit `3a61618` |
| Split-leak quantified instead of asserted: 80.53% of the honest test window sits in a shuffled 80% train partition, 99.06% have their own `t−1h` in training, inflating MAPE 9–17% | `docs/02` §5.4 |
| **Input smoothing (recipe step 9) measured as a negative result and rejected** — 4 variants, bootstrap CIs; V1 +7.23 pp worse at h+1, +2.89 pp at h+24 | `docs/02` §5.5; `scripts/smoothing_probe.py` (~145 s, read-only, reproduces) |
| **Train/serve parity proven and an audit "P0" refuted** — captured the exact frame the served estimator receives; 6 features × 4 targets, every difference `0.0` | `docs/02` §5.6 |
| `predict_live.py` CLI (recipe step 8) works end to end | ran `--zone ES --horizon 6` → exit 0, P10/P50/P90, contiguity disclosure |
| Daily accuracy ledger (recipe step 11) | `scripts/live_accuracy.py` + `scripts/forecast_snapshot_store.py`; `tests/test_live_accuracy_ledger.py`, `tests/test_live_accuracy_snapshots.py` |
| Weekly retrain + promote gate (recipe step 10) — trains into a dated candidate dir, never `data/models/`; promotion is the only writer and is gated on baselines, closed-loop gates, recursive h+24/h+48 MAPE and band coverage | `scripts/promote_models.py`, `scripts/run_retrain_chain.sh`, `tests/test_promote_gate_live.py`, `tests/test_promote_gate_rules.py` |
| Risk panel scores the newest complete row instead of a weekday+hour lookalike; `snapshot_mode` + `snapshot_age_hours` added additively | `api/main.py` (`_representative_snapshot`), `tests/test_risk_snapshot.py` |
| Weather-blind risk artifact trained (the real cure for the stale risk panel) | `scripts/train_risk_no_weather.py`, `tests/test_risk_no_weather*.py` — **trained, not wired into serving** |
| Weather-lag claims corrected to the measured mechanism (a sawtooth, not one-way drift) | `README.md`, `docs/04:28`, `docs/06:38`, `docs/07:113` — 90 h committed / 134.0 h at 2026-11-06T15:29Z / 55.0 h at 2026-11-07T07:52Z |
| Scratch outputs kept out of git; harness scripts kept as evidence | `.gitignore` (`_retrain_scratch/*`, `_risknw_scratch/*` with `!*.py`) |
| Secret scan of every changed and untracked file | no hits; no `.env` beyond `.env.example` |

## IN PROGRESS

1. **Weather-blind risk artifact → serving (largest user-visible defect).**
   Trained to `_risknw_scratch/out_v1/risk_classifier_no_weather.joblib` with metrics beside it.
   Live proof of the defect (browser, 2026-11-07): the RISK panel read `2026-10-31 07:00Z`,
   **7.0 days old**, next to a forecast origin 1.9 h old. Row selection cannot fix this: the risk
   feature set requires weather, and the refresher appends load-only hours, so the newest
   weather-complete hour is always days behind.
   **Next step:** load the `_no_weather` artifact in `api/main.py::_load_artifacts()` as a
   fallback exactly like `lightgbm_load_forecast_no_weather`, set `served_by`/`snapshot_mode` to
   say which artifact scored the row, and copy the artifact into `data/models/` **only** through
   the promote gate. I stopped before touching `api/main.py` for this.
2. **`/health` provenance scopes.** Whole-history counters are printed beside served-frame
   currency, so a reader cannot tell which object a number describes. A read-only diff was
   produced but not applied; `api/main.py:499-507` is where the lag is computed from the live frame.
3. **`/docs`, `/redoc`, `/openapi.json` are public.**
   **Next step:** `FastAPI(docs_url=None, redoc_url=None, openapi_url=None)` gated on an
   `EXPOSE_API_DOCS` env var at `api/main.py:206`, then hide the `/docs` link at
   `gridsense-frontend/src/components/Footer.jsx:44`. Nothing else was changed; it was deferred
   only because another agent held `api/main.py`.
4. **Flaky test.** `tests/test_risk_snapshot.py:153` asserts `'weekday_hour_lookalike' == 'newest'`
   and fails when the local parquet is older than the 3.0 h threshold (it is ~5 days old).
   **Next step:** inject a fixed "now" (`_now_utc`) rather than weakening the assertion.
5. **Retrain chain exit status.** `_audit_out/chain.status` recorded `models.train_baselines
   exit=1 15:34:06Z` and the cause was never explained. Re-run the chain against scratch and read
   the log; do not assume it is benign.

## PENDING

Effort: S (< 1 h), M (half day), L (multi-day). Risk = chance of making the live app worse.

| Task | Acceptance criteria | Effort | Risk | Depends on |
|---|---|---|---|---|
| Wire weather-blind risk into serving | `/risk/{zone}` scores the newest load hour, `snapshot_age_hours` < 2 h live, and `served_by` names the artifact | M | **High** (changes what users are told) | promote gate; decision to ship a second risk model |
| Apply `/health` provenance diff | every counter's scope is named in the response; existing field names preserved | S | Low | — |
| Gate API docs behind `EXPOSE_API_DOCS` | `/openapi.json`, `/docs`, `/redoc` return 404 when unset; frontend Footer link removed | S | Low | — |
| Fix the flaky snapshot test | 20 consecutive runs green with a monkeypatched clock | S | Low | — |
| Live accuracy dashboard surface | the ledger's MAE/RMSE/coverage visible to a user, not just a CLI | M | Low | live_accuracy.py |
| Persist issued forecasts | each `/forecast` response appended as a row so accuracy is a real forecast-vs-actual join, not a serving-path backtest | M | Medium | storage decision (Parquet/CSV/SQLite) |
| Schedule the weekly retrain | a documented scheduler (currently there is none; the chain is manual) | S | Low | promote gate |
| Frontend: re-verify the chart/header fix in a browser at 393×852 | screenshots at the fixed commit; no clipped labels, no caption overlap | S | Low | frontend branch |
| IT_SOUTH accuracy | h+24 MAPE materially below the current ~9.6% — needs a zone-identity or per-zone model, not a different learner | L | Medium | measured experiment (4 arms) |
| Extreme-overload ceiling | a model that can output above the observed p99; today the utilisation ladder tops out at 0.490872 against a 0.6 alarm line, so the alarm never fires | L | Medium | label/capacity design |
| Advance warning for marginal overloads | detect events ≥ 3 h ahead at a stated recall/precision | L | Medium | exceedance head was tried and rejected once |
| Middle-band recalibration / coverage drift | closed-loop gates return to PASS honestly, or the WARN stays documented | M | Medium | new validation window |
| Weather forecast (NWP) input | weather-aware model usable at the live origin instead of the fallback | L | Medium | a forecast feed |
| Rated capacity instead of p99-of-load | `capacity_mw` from a documented source | L | Low | data source |
| Per-zone UI and replay picker | each zone explorable; the closed-loop replay selectable | M | Low | frontend |
| Pakistani data | a real NTDC/DISCO feed or an explicit, dated statement that none exists | L | Low | data access |
| CI integrity job | `pytest` + the integrity verifier run on every push and fail the build | S | Low | fixing the 6 verifier failures |

## BROKEN OR RISKY

- **`verify_prediction_integrity.py` fails 6 of 33**, all from one cause: it compares the live API
  (origin `2026-11-09T08:00Z`) with local artifacts ending `2026-11-04T17:00Z`. The six are the
  three `served origin is the true end of gap-free history` and three `independent recursion
  reproduces served p50` checks. Until the verifier learns about the live/committed split, its
  live mode cannot be a CI gate. The other 27 — band arithmetic, snapshot membership, risk
  reproducibility, sensitivity — pass. I did not edit the check.
- **The risk panel is days stale in production** (see IN PROGRESS 1). The FEEDS rail states it
  honestly, so it is not a silent lie, but it is a product defect.
- `data/features/training_features.parquet` was clobbered twice by an agent's retrain run on
  2026-11-06 and restored; the clobbered copy is quarantined at
  `_audit_out/retrain_demo/quarantine/`. Any retrain must write to `out_path` /
  `GRIDSENSE_MODELS_OUT_DIR`, never in place.
- The committed parquet is now ~5 days behind the live frame by design. Any doc sentence that
  quotes one number for "the data" without naming which object is wrong.
- Two structural P1s above (`/health` scope, weather-pinned risk) are design issues, not typos.

## HOW TO RUN

```bash
# install
python -m venv .venv && . .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# test
python -m pytest tests -q

# run locally (never kill HP's uvicorn on 8123; use a spare port)
uvicorn api.main:app --port 8124
export GRIDSENSE_REFRESH=false        # offline: serve the committed frame only

# integrity
PYTHONPATH="$PWD" python scripts/verify_prediction_integrity.py

# forecast CLI (read-only, writes nothing)
python scripts/predict_live.py --zone ES --horizon 6

# deploy: NOT DONE HERE. Deployment happens only by pushing main.
```

## RULES THE NEXT AGENT MUST FOLLOW

1. **Never change a gate threshold, metric, label rule, or split to improve a number.** The four
   WARN gates stay WARN until the model earns them.
2. **Every figure in docs, UI, or a resume point must come from a committed artifact or a command
   you just ran**, and must name its object (committed parquet vs live served frame; 90 h vs
   134.0 h vs 55.0 h are all real and are not the same quantity).
3. **Stop and ask before retraining**, and never call `train_and_evaluate()` for measurement — it
   overwrites the committed `risk_classifier.joblib` and `risk_factors_by_bucket.json`.
4. **No pushes to main without review.** `main` auto-deploys; the live app is being shown to
   people. Work on a branch, open a draft PR.
5. **No invented claims**: not "live", not "Pakistan-ready", not "production-grade". Say what a
   command proved and when. Dated measurements stay; present-tense claims get re-measured.
6. Never kill the owner's uvicorn on port 8123. Probes use spare ports and shut down after.
7. `/tmp` is not shared between Git Bash and Windows Python; `C:\tmp` is a different directory.
