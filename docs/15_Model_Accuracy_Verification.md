# 15. Model Accuracy Verification — every figure recomputed

**Measured:** 2026-11-06, 09:35–09:42 UTC, re-run 09:47 UTC.
**Live target:** `https://grid-senseai-backend.fastapicloud.dev`.
**Repo state:** backend `main` at `dbd2e56`; frontend `main` at `8e01149`.
**Tool:** `scripts/verify_model_accuracy.py` — 22 checks, exit 0, added with this document.

The three runs printed the same values for all 22 checks, including the deployed
`risk_score` comparisons that reach `|delta| = 0.000e+00`; the pipeline is
deterministic across runs, not merely across code paths. A third pass at
09:54:50 UTC, after these commits reached `main` and FastAPI Cloud rolled, again
returned 22 passed / 0 failed against the deployed backend.

Every number below was printed by that script or by the single comparison run in
§4 on the date above, against the committed artifacts. Nothing is quoted from a
training log, and nothing is read out of `data/models/closed_loop_report.json`
and then presented as a result: section 4 recomputes each metric from the
artifact and section 7 only then checks that the served report agrees. Where the
two had disagreed, this document would say so.

```bash
python scripts/verify_model_accuracy.py              # 22 checks, live API included
python scripts/verify_model_accuracy.py --skip-live   # 10 checks, artifacts only
```

No key, no local server, and no network needed for `--skip-live`. The
developer's own uvicorn on port 8123 was not touched by any measurement here.

---

## 1. What was proven

| Claim | Measured | Script section |
|---|---|---|
| The label means what the docs say | stored `overload_risk_label` == `load > 0.9 × capacity` on **51,959 / 51,959** rows, exact agreement **1.000000**, positive rate 0.0589 both ways | 1 |
| The classifier cannot read the answer off the frame | artifact's own `feature_name_` = 14 features, identical to `FEATURE_COLS` in order, **zero** of `load_to_capacity_ratio` / `load_mw` / `capacity_mw` / the label present | 2 |
| The scored window is genuinely unseen | train 41,567 rows ≤ **2026-06-13 09:00**, test 10,392 rows ≥ **2026-06-13 10:00** — strict and temporal, using the trainer's own `walk_forward_split` | 3 |
| Held-out accuracy | **PR-AUC 0.9268** (chance = base rate 0.1265), ROC-AUC 0.9857, Brier 0.0309 vs 0.1105 reference → skill 0.7204 | 4 |
| Alarms are mostly right | at 0.6: TP 1,004, FP 132, FN 276 → precision **0.8838**, recall **0.7844** | 4 |
| The probabilities mean something | observed overload 0.0221 / 0.4682 / 0.6127 / 0.9225 in the low / moderate / elevated / critical buckets | 5 |
| The dashboard is not the model's best day | served `/validation` deep-equals the committed report, and served precision, recall, Brier skill, `hours_scored`, `positive_rate` and **all three per-zone recalls** equal this script's recomputation | 7 |
| The deployed model *is* this model | served `risk_score` reproduced to **\|delta\| = 0.000e+00** for ES, GR and IT_SOUTH | 7 |

Scored population: **10,122 zone-hours** across 3 zones over 3,464 elapsed hours
(`window.hours`), 12.65 % of them overload. 270 of the 10,392 test rows are not
scorable at all because their published weather is missing; they are dropped by
`dropna(subset=FEATURE_COLS)`, not silently imputed.

## 2. What was proven to be weak

Stating these is the purpose of the document.

**"Capacity" is a proxy, so "utilisation" is not what an operator would assume.**
`closed_loop_report.json → window.overload_definition` records it verbatim:
`load_mw > 0.9 * capacity_mw (capacity is a per-zone p99-of-load proxy, not
rated capacity)`. So every utilisation figure below means *relative to the 99th
percentile of that zone's own observed load*, not relative to installed plant.
A Pakistani DISCO reading "util 1.05" as "5 % above transformer rating" would be
wrong. Sourcing real rated MW is the single highest-value data change available
and is listed in §6.

**IT_SOUTH is not the same product as ES.** Per-zone recall on the identical
held-out window:

| zone | precision@0.6 | recall@0.6 |
|---|---|---|
| ES | 0.9238 | **0.9671** |
| GR | 0.8601 | 0.9318 |
| IT_SOUTH | 0.8130 | **0.4556** |

The aggregate 0.7844 is a blend that hides a zone where the system misses more
than half of real overloads. README.md:164 already disclosed 45.6 % for southern
Italy; script sections 5b and 7 now recompute it and confirm the served figure
matches. Any statement of the form "the model catches 78 % of overloads" should
be refused in favour of this table.

**The model does not detect an overload worse than anything it has seen.**
Section 6 drives the newest row of each zone across utilisation 0.50 → 1.30,
scaling every load-derived feature and holding clock and weather fixed:

| zone | util 0.50 | 0.85 | 0.95 | 1.15 | 1.30 |
|---|---|---|---|---|---|
| ES | 0.000018 | 0.000201 | 0.359110 | 0.479391 | **0.490872** |
| GR | 0.000002 | 0.000004 | 0.000030 | 0.000010 | 0.000010 |
| IT_SOUTH | 0.000000 | 0.000003 | 0.000033 | 0.000002 | 0.000001 |

No rung reaches the 0.6 alarm line in any zone; the highest score anywhere is
**0.490872**. A tree ensemble cannot emit a leaf value it never trained on, so a
zone running 30 % above its own historical p99 — a worse event than anything in
the two recorded years — is reported `normal`, and `GET /shed-schedule/{zone}`
returns `[]` for it.

This is not fixable by tuning. Capacity would have to become a feature, but
capacity is half the label, so that is leakage; or the model would need a
monotonic constraint, which is a retrain, which changes every number in this
document. It is recorded as an accepted limitation. What makes it survivable
today is that the system is advisory rather than a control loop, and the
forecast panel shows the MW trajectory an operator should read alongside the
score (that path is verified independently by
`scripts/verify_prediction_integrity.py`).

**The utilisation curve is not monotone.** The largest step *down* as load rises
is −0.000012 (ES, util 0.50→0.70), −0.000019 (GR, 0.95→1.05) and −0.000028
(IT_SOUTH, 0.95→1.05). Small in absolute terms, and the reason the ladder is
reported as a range rather than read rung by rung.

**The `low` bucket is over-confident in the safe direction.** 8,813 rows scored
into 0–0.3 show an observed overload rate of 0.0221, so **195** real overloads
were called low-risk. The remaining **81** of the 276 missed hours sit in the
`moderate` bucket (173 rows at 0.4682), which is also below the 0.6 alarm line.
195 + 81 = 276 = the served `missed_hours`, so the miss budget is fully
accounted for by these two buckets.

## 3. The four WARN gates, and why none of them was "fixed"

`/validation` serves **9 PASS / 4 WARN / 0 FAIL** over 2026-06-13 10:00 →
2026-11-04 17:00 UTC:

| gate | value | target | floor | reading |
|---|---|---|---|---|
| `coverage_drift` | 0.1948 | 0.15 | 0.30 | week-to-week spread of P10–P90 coverage; the weakest week measured 0.726 against 0.9208 in another |
| `event_detection` | 0.7845 | 0.80 | 0.60 | 142 of 181 events, 1.55 points short |
| `event_warning_3h` | 0.3425 | 0.50 | 0.25 | only a third of events get 3 h of warning |
| `median_lead_hours` | 1.0 | 3.0 | 1.0 | median lead is 1 h, sitting exactly at the floor |

Each of these is a *quantity the app reports honestly*: the system is a nowcast
with roughly one hour of lead, not a three-hour warning service. Every one would
become a PASS if its target moved. No target was moved, no definition changed and
no metric re-selected — which is precisely why the verdict served by the live API
is `WARN` and why this document repeats it.

## 4. The model comparison, and the split, measured rather than argued

A review prompt handed to this project specified: "80 % Train, 10 % Validation,
10 % Test. **Shuffle before split**", "compare Random Forest, XGBoost and
LightGBM", "use 5-Fold Cross-Validation", "pick the best model on TEST accuracy".
Three of those four instructions are addressed by measurement below. **XGBoost
was never in the stack** — `docs/02_ML_Model_Design.md:171` says so explicitly,
no source file imports it, it appears in no dependency manifest, and it is not
installed here. Adding a runtime dependency to satisfy a checklist item is not
something this project does silently, so the comparison was run with what is
actually available plus sklearn's own booster.

### 4a. Why the split is temporal

Section 3 of the script measures what the shuffle instruction would cost:

- **10,392 of 10,392** scored rows (**100.0 %**) would land chronologically at or
  before the last fitted hour.
- **41,349 of 41,567** fitted rows (**99.5 %**) would take `load_lag_1h`,
  `load_lag_24h`, `load_roll_mean_24h` or `load_roll_std_24h` directly from a
  held-out row.

A shuffled split therefore hands the model the answers as inputs and its metrics
describe nothing. `walk_forward_split` sorts by timestamp first, and the check
"every scored row starts after the last fitted row" is what proves the shipped
model avoided it.

### 4b. The comparison on the identical held-out window

Single run, 2026-11-06 09:42 UTC, same features, same temporal split, same
10 % validation slice carved from the **fit** window only
(fit 37,410 / val 4,157 / test 10,122):

| model | PR-AUC | Brier | precision@0.6 | recall@0.6 | fit |
|---|---|---|---|---|---|
| **LightGBM (shipped artifact)** | **0.9268** | **0.0309** | 0.8838 | **0.7844** | deployed |
| LightGBM (rerun, same config) | 0.9256 | 0.0312 | 0.8859 | 0.7828 | 2.1 s, best_iter 490 |
| Random Forest (500 trees, class_weight balanced) | 0.9030 | 0.0434 | **0.9528** | 0.5367 | 8.1 s |
| HistGradientBoosting (sklearn) | 0.8556 | 0.0472 | 0.9289 | 0.6023 | 1.9 s |
| Logistic regression on 3 load lags | 0.3567 | 0.1375 | 0.3743 | 0.4000 | 0.2 s |

Reading, in the order it matters:

1. **The selection is correct on TEST, not on train.** LightGBM wins PR-AUC and
   Brier and has by far the best recall. Random Forest is the more *precise*
   alarm (0.9528) but catches barely half of overloads (0.5367) — for a
   safety-advisory product, precision bought with missed events is the wrong
   trade. Picking on TEST PR-AUC, as the prompt requires, selects the model
   already shipped. No change made.
2. **Retraining reproduces the artifact.** PR-AUC 0.9256 vs 0.9268, recall 0.7828
   vs 0.7844 — the pipeline is deterministic enough to be re-run and audited.
   Note `best_iteration_ = 490` of the 500 cap: early stopping nearly did not
   trigger, so the next retrain could bind on the cap. Worth watching, not a
   defect today.
3. **A linear floor proves the trees earn their keep.** Logistic regression on the
   three load lags gets PR-AUC 0.3567 against a chance level of 0.1265. So the
   labelled events are learnable from load alone, and the 0.9268 headline is real
   signal on top of that floor rather than a trivial pattern.

### 4c. 5-fold cross-validation: deliberately not used

For an hourly forecast with lag and rolling features, k-fold CV on rows sampled
across the timeline re-introduces exactly the leakage §4a measures — a training
fold would sit chronologically after the fold it is scored on. The correct
equivalent here is a walk-forward or expanding-window evaluation, which is what
`models/closed_loop_validation.py` already does: **10,050 replayed forecast
origins** at a 1-hour stride across 2026-06-13 → 2026-11-04, scored per horizon
and per zone. The 10 % fold the prompt asks for is not discarded either: it is
used as the early-stopping validation slice of the fit window, which is the
honest placement for it. `random_state=42` is pinned throughout and the rerun in
§4b confirms the effect.

## 5. Scope: what this document does not cover

- **Forecast accuracy** — verified by a different script, written deliberately
  without importing `models/forecasting.py` so agreement is a real cross-check:
  `python scripts/verify_prediction_integrity.py`.
- **Security posture** — measured attack surface, response headers, CORS,
  dependency audit, accepted risks: `docs/14_Security_Posture.md`.
- **Live data provenance** — ENTSO-E and NASA POWER are a *measured stand-in* for
  the intended Pakistani NTDC/DISCO domain. That framing is unchanged. Every
  number above describes European zones and must not be quoted as Pakistan
  performance.
- **Wall-clock animation playback and the 3D render loop** were not sampled: the
  in-app viewport was unavailable during the 2026-11-06 probe, so no real frame
  timing was recorded. Motion was verified by driving the Web Animations timeline
  deterministically instead, all eight keyframe sets.

## 6. Open decisions this document depends on

| Item | Status |
|---|---|
| Real rated MW capacity per zone | **not sourced**; it is the proxy behind every figure in §2, and replacing it would change the label itself |
| Data refresh path for the served frame | not designed; the frame ends 2026-11-04 17:00 UTC and the served risk snapshot ages with weather-feed lag |
| IT_SOUTH recall 0.4556 | accepted limitation; per-zone models or recalibration is unstarted work |
| Extreme-overload blindness above the trained range | accepted limitation, §2 |
| Pickle artifacts execute code on load | accepted; a non-code artifact format is not started |
| 500-estimator cap nearly binding at retrain (`best_iter` 490) | observed, not yet acted on |
