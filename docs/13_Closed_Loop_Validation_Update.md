# Engineering Update — Closed-Loop Validation

**Project:** GridSense AI — predictive grid overload risk and advisory load-shedding
**Author:** Huzaifa Himad
**Date:** 2026-11-03
**Scope of this update:** model-validation methodology, backend serving layer, operator dashboard, test suite
**Status:** implemented, tested (48 tests as of this date, CI green), **merged to `main` and deployed on both services** — `/validation` is live and the panel is rendering on the public dashboard

| Repository | `main` commit | CI | Live status |
|---|---|---|---|
| [Grid_Sense.AI-Backend-](https://github.com/huzaifahimad/Grid_Sense.AI-Backend-) | `af26428` | passed | `GET /validation` and `GET /health` verified 200 on `grid-senseai-backend.fastapicloud.dev` |
| [Grid_Sense.AI-frontend-](https://github.com/huzaifahimad/Grid_Sense.AI-frontend-) | `1c1a87e` | passed | Panel 05 rendering with live report data on `grid-senseai.vercel.app` |

> **Superseded as of 2026-11-06 — read this before quoting any figure below.**
>
> Everything in Sections 3–6 describes the replay as it stood on **2026-11-03**, when the
> feature frame was still the synthetic bootstrap. The frame was rebuilt on two years of
> real ENTSO-E and NASA POWER hours (commit `9c6a48e`), and `/validation` now returns a
> different result set. The current live report, fetched 2026-11-06:
>
> | Measure | 2026-11-03 (this doc) | Now (live `/validation`) |
> |---|---|---|
> | Overall verdict | PASS, 13/13 | **WARN — 9 PASS, 4 WARN, 0 FAIL** |
> | Window | synthetic replay | 2026-06-13 10:00 → 2026-11-04 17:00 UTC, 3,464 h |
> | Hours scored | — | 10,122 across ES / GR / IT_SOUTH |
> | +1 h forecast MAPE | 2.24 % | 2.12 % (seasonal-naive 9.27 %) |
> | Aggregate P10–P90 coverage | 84.3 % | 84.9 % |
> | Week-to-week coverage spread | 14.2 pts (PASS) | 19.5 pts (**WARN**) |
> | Alarm precision / recall | 0.877 / 0.905 | 0.884 / 0.784 |
> | Event detection recall | — | 0.785 (142 of 181 events) |
>
> The four WARN gates are `coverage_drift`, `event_detection`, `event_warning_3h` and
> `median_lead_hours`. No gate FAILs, so the release verdict is WARN rather than FAIL —
> and that is the honest state, not a pass to be read as one. The mechanism, not the
> number, is what this document still earns its place for.

---

## 1. Why this work was necessary

Before this update, the project's accuracy claims rested on **training-time metrics** — a
validation-set MAPE here, a confusion matrix there. That is the weakest possible evidence for
a decision-support system, because it measures the model in the environment it was graded in,
not the environment it operates in. Three specific gaps mattered from a grid-operations point
of view:

1. **No multi-horizon accountability.** A single "MAPE ≈ 4.3%" figure says nothing about
   whether the forecast degrades gracefully from +1 h to +24 h, which is exactly the axis a
   dispatcher uses to decide whether an alert is worth acting on.
2. **No proof the uncertainty band was real.** The dashboard draws a P10–P90 band. A band is
   only meaningful if the observed outcome falls inside it roughly 80 % of the time. That had
   never been measured on the served forecast.
3. **No measure of alert quality in operational terms.** Precision, recall, nuisance-alarm
   rate per zone-day, and — most importantly — *how much warning time an operator actually
   gets before an overload event*. Those are the numbers a control room cares about; none of
   them existed.

Closed-loop validation closes these gaps by replaying the **held-out test window through the
exact code path the API serves from** — recursive forecast, calibrated interval, risk
classifier, shedding rules — and grading the outcome against pre-committed release gates.
Nothing is measured on a training harness that the deployed system never touches.

## 2. Method

**Replay window.** The untouched test split, as published live by `GET /validation → window`:
2026-06-13 10:00 UTC → 2026-11-04 17:00 UTC — **10,122 scored rows** across the three zones
(ES · GR · IT_SOUTH) over **3,464 elapsed hours**. Origins are hourly at a 24-hour horizon
(**10,050 replayed**, 1-hour stride); the first origins begin before the window so that
in-window forecasts have full lag history and are not warm-starting. There are 2 hourly gaps
in ES and none in GR or IT_SOUTH. This window is the current one and moves whenever the feature
frame is rebuilt, so the endpoint is the authority rather than this paragraph — the earlier
2026-11-03 replay described in §3 ran over 2026-04-14 20:00 → 2026-09-06 10:00 UTC
(3,471 zone-hours, 10,413 scored rows, 10,341 origins) on the frame committed before the
real-data rebuild.

**Serving path, not a research path.** The replay calls the same `recursive_forecast`,
`apply_intervals`, and risk-classifier code that `api/main.py` imports. Feature construction,
lag windows and the label definition are identical, so a discrepancy between "what the
validator measured" and "what the API serves" would itself be a finding.

**Baseline discipline.** Every accuracy gate is scored as an *improvement over seasonal-naive*
(load from the same hour, one week earlier) rather than as an absolute error figure. Skill
against a trivial benchmark is the only accuracy claim that means anything for load
forecasting.

**Label-leak guards retained.** `load_to_capacity_ratio` stays excluded from the risk feature
set, and each origin only sees information available at that timestamp — no forward fill, no
future lag.

**Uncertainty.** Intervals are split-conformal, calibrated per horizon on the validation
split; the replay then checks realised coverage instead of assuming nominal coverage.

**Release gates.** Thirteen gates, each with a **target** and a lower **floor**, and a
direction. A metric at or above target scores PASS; between floor and target scores WARN;
below floor scores FAIL. Any FAIL makes the overall verdict FAIL, and `--strict` makes the
process exit non-zero so the gate can block a release.

## 3. Results — the 2026-11-03 run (13/13 gates PASS at that date; see the superseding block above for the current report)

### Forecast skill and interval honesty

| Horizon | Recursive MAPE | Seasonal-naive MAPE | Skill | Realised P10–P90 coverage | Bias |
|---|---|---|---|---|---|
| +1 h | 2.24 % | 5.21 % | **+56.9 %** | 81.4 % | +0.08 % |
| +6 h | 4.17 % | 5.20 % | +19.8 % | 85.8 % | +0.29 % |
| +12 h | 4.34 % | 5.20 % | +16.6 % | 85.3 % | +0.28 % |
| +24 h | 4.36 % | 5.20 % | **+16.1 %** | 84.9 % | +0.26 % |

Aggregate P10–P90 coverage is **84.3 %** against a nominal 80 % — the band is honest, and the
week-to-week coverage spread is 14.2 percentage points, inside the tolerance gate. Bias is
small and consistently slightly positive, i.e. the forecast under-predicts by roughly a quarter
of a percent; that is reported rather than smoothed away.

### Alarm quality at the operating threshold (risk ≥ 0.6)

| Measure | Value |
|---|---|
| Precision (alarmed hours that were genuinely overloaded) | **0.877** |
| Recall (overloaded hours that were alarmed) | **0.905** |
| F1 | 0.891 |
| Brier score vs. base-rate Brier | 0.0185 vs. 0.0975 → **skill 0.810** |
| True positives / false alarms / misses (hours) | 1,032 / 145 / 108 |
| **False alarms per zone-day** | **0.33** |
| Advisory shedding volume (hours) | 1,033 priority-1, 144 priority-2 |

Overload occurs in 10.95 % of scored hours, so the precision figure is meaningful against a
base rate of 0.110, not against 50/50. The nuisance cost — the number that decides whether
operators keep trusting the system — is roughly one false alarm per zone every three days.

### Event-level response (contiguous overload episodes)

| Measure | Value |
|---|---|
| Overload events in window | 206 |
| Events alarmed at least once while developing | **188 (91.3 %)** |
| **Median stable warning lead** | **3.0 h** |
| Events with ≥ 1 h unbroken warning | 60.2 % |
| Events with ≥ 3 h unbroken warning | **51.9 %** |
| Events with ≥ 6 h unbroken warning | 45.6 % |
| Longest event | 16 h |

## 4. The substantive finding: lead time was being over-claimed

The first version of this validator reported PASS with a **19-hour median lead**. That number
was wrong in an instructive way, and finding it changed the methodology.

Two independent problems:

- **Event onset was an artefact of the synthetic daily cycle.** Every onset clustered at 08:00
  UTC. A metric that averages over a distribution this narrow is not describing system
  behaviour.
- **Long horizons were getting credit for re-reading the peak.** A +24 h forecast reproduces
  the next day's morning peak from its own lag features. Measured as "the earliest horizon that
  glimpsed the event", that reads as a full day of warning — but at +24 h the model is not
  predicting *this* event; it is restating a daily pattern, and the warning evaporates and
  returns as the horizon advances. An operator who acted on it would have shed load on a signal
  that did not persist.

**Correction.** Lead time is now measured two ways and reported side by side:

- **Stable lead** — an *unbroken chain* of warnings from +1 h through the event onset. This is
  the only version an operator can act on. **3.0 h.**
- **Glimpse lead** — the first time any horizon saw it. **24.0 h**, retained deliberately so the
  gap between the flattering reading and the honest one is visible, not buried.

The release gates are scored on stable lead. Two supporting measures were added: warning-rule
precision evaluated over **all** origins rather than only onset hours (at +6 h: 1,042 warnings
issued, 874 correct, 0.839 precision), and an explicit P90-upper-band trigger variant so the
lead-versus-precision trade-off is quantified (95.6 % of events would be warned ≥ 3 h ahead
using the conservative band trigger, but at materially more false alarms). Had I kept the
original metric definition, the dashboard would have advertised a day of warning it does not
provide.

## 5. Proof that the gates can fail

A validator that always prints PASS is worse than no validator, because it launders an
unqualified model into a certification badge. Two negative controls run in the normal test
suite, injecting deliberately broken models into the same replay:

| Control | Injected fault | Gate response |
|---|---|---|
| Inflated forecaster | every prediction × 1.15 | P10–P90 coverage **0.8556 → 0.0003** on the same coarse-stride replay configuration, `band_coverage` **FAIL**, overall **FAIL** (the shipped hourly replay records 0.8492) |
| Always-shed classifier | alarms on **all 10,122** scored hours | precision collapses to **0.1265** — byte-identical to the shipped report's own overload base rate, which is the point: an always-alarming model buys exactly the base rate — burden rises to **20.97** false alarms per zone-day against a 5.0 gate target (the shipped hourly replay records 0.313), `alarm_precision` and the burden gate **FAIL** — while `alarm_recall` correctly still **PASSes** at 1.0, overall **FAIL** |

Re-measured on the current frame at 2026-11-06T13:47Z by
`python -m pytest tests/test_closed_loop.py -k negative_control -q`, which runs both injections
through the same replay the shipped report uses.

The second control is the one worth noting: recall stays perfect while precision and burden
fail. Scoring alarms on recall alone would have certified a device that cries wolf every hour —
the classic failure mode behind alert fatigue in control rooms.

**A configuration note that is itself a finding.** The controls run on a coarse 24-hour origin
stride so they complete inside the test suite, but the shipped report replays **every** hour.
`forecast_warning_precision_h6` and the event lead metrics are only meaningful on a contiguous
hourly origin grid: on a coarse stride the sampled lead hours stop coinciding with event onset,
and the warning-precision gate collapses to 0.0 for reasons that have nothing to do with model
quality. A metric that is silently stride-sensitive is a metric that must be reported with its
configuration — which is why the artifact carries its window, stride, horizon and runtime.

## 6. What changed in the running system

**Backend**
- `models/closed_loop_validation.py` (new) — replay engine, metric families, gate grading, provenance, `--strict` exit code.
- `models/decision_rules.py` (new) — critical 0.8 / elevated 0.6 / moderate 0.3 thresholds, dashboard bucketing and shed priority now live in **one module** read by both the API and the validator. Previously they were duplicated, so a threshold edit could have made the deployed alerts and the reported metrics describe different systems.
- `api/main.py` — new `GET /validation`; report loaded at startup alongside the model artifacts; `/health` now reports its load state. Threshold constants removed in favour of `decision_rules`.
- `models/calibrate_forecast_intervals.py` — made horizon-aware so the validator reuses the calibration machinery instead of reimplementing it.
- `data/models/closed_loop_report.json` (new artifact, 18 KB) — the committed, version-controlled evidence, with method, runtime, window, artifact provenance and stated limitations.
- `.fastapicloudignore` — the report is included in the deploy bundle; verified that the serving path still imports nothing beyond the lean dependency set (no `shap`, no `torch`).

**Dashboard**
- **Panel 05 — Closed-Loop Validation.** Overall verdict with the gate count, report age, the four operationally relevant headline numbers (precision, recall, false alarms per zone-day, stable warning lead), realised coverage, MAPE against seasonal-naive, events alarmed, an expander listing every gate's measured value versus target, and the stated limitations.
- It loads once on mount and is deliberately isolated from the shared error banner: a `404` reads as "this backend does not expose the endpoint yet", not as an outage. Verified against both a compliant and a non-compliant backend.

**Quality gates**
- Test suite **28 → 48 tests**; the additions cover gate grading, lead-time arithmetic, event-run segmentation, train/serve rule equivalence, report self-consistency, the endpoint contract, both negative controls, and a history-contiguity guard added during the post-deploy audit.
- GitHub Actions green on both repositories (backend `pytest`, frontend `npm run build`).
- Documentation updated to match the code, including two stale claims corrected in passing: the user manual still described the forecast band as a hardcoded ±8 % display band (it is conformal now), and endpoint counts were out of date in three documents — now all state the seven routes `api/main.py` defines.

## 7. Limitations, stated plainly

These are in the report artifact itself and should be read before citing any figure above.

1. **The data is real, but it is not Pakistani data.** This limitation used to read
   "training data is a documented synthetic bootstrap"; it no longer does — the frame was
   rebuilt on two years of measured ENTSO-E load and NASA POWER weather. What remains true
   is narrower: three European bidding zones (ES, GR, IT_SOUTH), no NTDC or DISCO hours,
   and therefore no evidence the learned distribution transfers to Pakistan's grid
   unchanged.
2. **Capacity is a proxy.** "Overload" means load exceeds 0.9 × a per-zone 99th-percentile
   historical load — it is a statistical ceiling, **not** a thermal or emergency (N-1) rating.
   Conductor ratings, transformer limits and contingency criteria are not in the loop.
3. **Weather persists beyond the observation window** in the replay, so forecast skill at long
   horizons is optimistic for the meteorological component.
4. **Zone-level aggregation.** ENTSO-E zones are control-area-scale, not feeder or transformer
   level; real shedding decisions are made on distribution assets with different granularity.
5. **Single replay window, and the report is a snapshot rather than a monitor.** `/validation`
   returns a committed artifact generated offline, so it certifies the models as of its
   timestamp; retraining does not regenerate it automatically, there is no scheduled
   out-of-time revalidation, and there is no drift monitor in production.

## 8. Next steps

| Priority | Work | Why it matters |
|---|---|---|
| 1 | Re-run the pipeline end-to-end on real ENTSO-E load once API approval lands, regenerate the report with `--strict` | Converts every number above from "pipeline demonstration" to "measured on the target distribution" |
| 2 | Acquire rated capacity (NTDC/DISCO or ENTSO-E asset data) to replace the p99 proxy | Turns a statistical ceiling into an engineering constraint |
| 3 | Add an explicit forecast-error decomposition at long horizons (weather vs. model) | Isolates how much of the +24 h error is genuinely meteorological |
| 4 | Feeder-level granularity and N-1 contingency screening | The level at which shed decisions are actually executed |
| 5 | Scheduled re-validation and drift monitoring in production | Turns a one-time gate into an ongoing claim |
| 6 | Enforce `--strict` inside the deploy pipeline, not just as a manual local step | A gate that only runs when someone remembers is a suggestion, not a release control |

## 9. Reproducing this work

```bash
git clone https://github.com/huzaifahimad/Grid_Sense.AI-Backend-.git
cd Grid_Sense.AI-Backend- && git checkout main
pip install -r requirements.txt

python -m models.closed_loop_validation --strict   # ~15 s, rewrites data/models/closed_loop_report.json
python -m pytest tests -q                          # 94 tests, incl. both negative controls
uvicorn api.main:app --port 8000
curl http://localhost:8000/validation
```

## 10. Post-deploy audit of the running system

After the merge, both repositories were re-audited line by line against the code, the
installed dependencies and the live services — not against the notes written earlier in
the project. Seven defects were confirmed and fixed; two reported defects were
disproved by measurement.

| # | Defect | Consequence | Fix and evidence |
|---|---|---|---|
| 1 | `src/config/tokens.js` emits modern space-separated `hsl(5 100% 65%)`, but three.js r185 `Color` only parses the legacy comma form and **silently** falls back to white | Every risk ring in the 3D topology lerped toward white, so a critical zone looked identical to a healthy one — the panel's core signal was absent | `toThreeColor()` normalises before `new THREE.Color()`. Measured against installed three: `#ffffff` → `#ff5b4d` / `#ffb338` / `#3dd687` |
| 2 | `/risk/{zone}` returned `predicted_load_mw` holding the **observed** load of the historical row it scored | The field name and the shed reasoning claimed a measurement was a forecast; the Zone Inspector showed 6,117 MW for ES while the forecast chart's first hour showed 6,905 MW | Renamed to `snapshot_load_mw`, reasoning reworded to "Load … at …Z", inspector relabels to *Load at Snapshot* and prints the scored hour |
| 3 | A `CRITICAL_ASSETS` guard tested for `HOSPITAL`/`WATER_PUMP` inside an `asset_id` that is always the synthetic `{zone}-FEEDER-01` | The branch could never run, yet its docstring asserted "critical infrastructure is never recommended" — a safety property the code did not provide | Removed the unreachable branch; the docstring now states only what the rules actually do |
| 4 | `/forecast` passed `dropna()`-thinned history to a forecaster that indexes lags **by row offset** while stamping output from the final timestamp alone | One missing hour would shift every lag feature and still return plausible-looking numbers — a silently wrong forecast rather than a visible failure | `_contiguous_hourly_tail()` keeps only the gap-free suffix; a new test proves an 83-row hole leaves an 80-row run under the 168 h minimum |
| 5 | `risk.top_factors.map(...)` was unguarded | A single absent field would throw and blank the whole inspector | `(risk.top_factors ?? [])`, plus `Number.isFinite` guards on the numeric metrics |
| 6 | The mock `/risk` payload omitted `snapshot_timestamp`, which the real API returns | Simulation mode had quietly drifted from the live contract | Mock now emits the same shape as the backend |
| 7 | Twelve documentation claims had drifted from the code | README quoted PR-AUC **0.9637**, a value no artifact produces while six other documents say **0.9595**; endpoint counts said 6 where `api/main.py` defines 7; the showcase `/risk` payload was invented; Three.js was listed as 0.128.0 against 0.185.1 installed (the same revision implicated in defect 1); alarm burden was divided by days instead of zone-days; a code comment and a test procedure both pointed at files that no longer exist | Corrected against `closed_loop_report.json`, `package.json`/lockfile and the actual route table |

**Two reported defects were false positives**, and were rejected only after measuring:
`colorWithAlpha("var(--color-critical)")` does *not* produce an invalid declaration — the
computed style resolves to `color(srgb 1 0.357 0.298 / 0.08)`, correctly red; and the
assets loader in `useGridData.js` already abort-guards its rejection, so the suspected
StrictMode error banner does not occur.

The validator itself was untouched by these fixes, so every §3 number still held as of
that date: 48 tests passed and the replayed report was 13/13 PASS. Re-measured on
2026-11-06 against the real-data artifacts, the suite is **59 passed** and the live
report is **9 PASS / 4 WARN, overall WARN**.

---

**Bottom line.** The system's claims are now falsifiable rather than asserted: every accuracy
number on the dashboard is backed by a replay of the served code path against a held-out
window, the gates are known to be capable of failing, and the one metric that mattered most —
usable warning lead time — was found to be over-claimed and corrected downward from 24 hours to
a defensible 3 hours.
