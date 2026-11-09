# GridSense AI — ML/DL Model Design Document

**Purpose:** Concrete architecture, feature set, and training strategy for both models
defined in the TRD. This is written to be buildable, not aspirational.

---

## 1. Model 1 — Short-Term Load Forecasting (Regression)

### 1.1 Architecture: Hybrid LSTM + 1D-CNN (with a Transformer variant as stretch goal)

Why this specific architecture, not "deep learning" as a vague label:

- **1D-CNN layers first:** extract short-range local patterns (hourly ramp shapes, sudden
  spikes) from the raw sequence before it hits the recurrent layers. This mirrors the
  LSTM + modified split-convolution approach attributed to the closest real precedent
  study (candidate source and its unverified status are recorded in
  `docs/01_Data_Sourcing_Plan.md` §1.2), reported at MAPE 2.38% single-step and 3.72%
  multi-step on Pakistan grid data. Treat that range as an unconfirmed external
  reference, not a goal this project has measured itself against — the deployed LightGBM
  forecaster already reports 2.10% test MAPE on real European hours (TRD Section 4), so
  the headline figure here would not be an improvement if it were reproduced.
- **LSTM layers second:** capture longer sequence dependencies — daily/weekly cyclic
  patterns, multi-day heatwave persistence.
- **Why not a plain Transformer for v1:** Transformers need more data to outperform
  LSTM-hybrids on single-series/zone-level load forecasting at the data volumes available
  from open sources per zone. Revisit once you have 3+ years of multi-zone data. Don't
  reach for the fashionable architecture before the simpler one is exhausted.

```
Input: [batch, lookback_hours=168, n_features]
   │
   ├─ Conv1D(64, kernel=3) → ReLU → BatchNorm
   ├─ Conv1D(32, kernel=3) → ReLU → BatchNorm
   │
   ├─ LSTM(128, return_sequences=True) → Dropout(0.2)
   ├─ LSTM(64) → Dropout(0.2)
   │
   ├─ Dense(32) → ReLU
   └─ Dense(forecast_horizon) → linear output (+ separate quantile heads for P10/P50/P90)
```

- **Lookback window:** 168 hours (7 days) — captures weekly seasonality (weekday/weekend
  demand shift, critical in Pakistan due to industrial load patterns).
- **Forecast horizons:** train separate heads or a multi-horizon output for 1h, 6h, 24h,
  48h — do not train one model per horizon separately unless resources allow; multi-output
  is more efficient and keeps forecasts internally consistent.
- **Uncertainty:** quantile regression (pinball loss) for P10/P50/P90 bands, satisfying
  FR-5 in the TRD. Do not ship point forecasts alone.

### 1.2 Feature Set

| Feature | Source | Notes |
|---|---|---|
| Historical load (lagged 1h, 24h, 168h) | ENTSO-E / PJM | Core autoregressive signal |
| Temperature (current + lagged) | NASA POWER | Strongest exogenous driver in AC-heavy grids |
| Humidity | NASA POWER | Heat-index effect on AC load |
| Hour of day (sin/cos encoded) | derived | Cyclic encoding, not raw integer |
| Day of week (sin/cos + is_weekend flag) | derived | |
| Month / season (sin/cos) | derived | |
| Public holiday flag | derived calendar | Pakistan-specific calendar to be swapped in later |
| Ramadan flag (Pakistan-specific, future) | derived calendar | Load patterns shift materially during Ramadan — flag now as a required feature for the Pakistan retraining phase, even though it's irrelevant to the open-data prototype |
| Rolling load statistics (mean/std, 24h/7d windows) | derived | |

### 1.3 Training Strategy
- **Split:** walk-forward / expanding window, never random shuffle (TRD FR-13).
  E.g., train on years 1–3, validate on year 4, test on year 5, then roll forward.
- **Loss:** pinball/quantile loss for the P10/P50/P90 heads, MAE for the point estimate.
- **Baselines to beat (mandatory, not optional):** persistence (t = t-24h), seasonal-naive,
  and a simple gradient-boosted tree (LightGBM) on the same features. If the deep model
  doesn't clear LightGBM by a meaningful margin, ship LightGBM — simpler, faster, easier to
  explain to a utility, and just as defensible in a pitch. Don't use deep learning as a
  credibility flex if a boosted tree does the job better.
- **Cross-zone generalization test:** train on one country/zone, evaluate zero-shot on
  another, to directly measure the transfer-learning risk flagged in the TRD rather than
  assuming it away.

---

## 2. Model 2 — Overload / Outage Risk Classification

### 2.1 Architecture: Gradient-Boosted Trees (LightGBM/XGBoost) as primary, not deep learning

This is a deliberate choice, and it matters that you can defend it:

- Outage events are **rare and imbalanced** (typically <2–5% positive class). Tree-based
  models with proper class weighting / focal-loss-equivalent handling generally outperform
  deep nets on tabular, imbalanced, moderate-data-volume problems — this isn't a compromise,
  it's the correct tool. Reserve deep learning for the sequence-heavy load forecasting task
  where it earns its complexity.
- Trees also give you **built-in feature importance and SHAP compatibility out of the box**,
  which directly satisfies the explainability requirement (TRD Section 4) that a utility or
  regulator will demand before trusting a shed-schedule recommendation.

### 2.2 Feature Set (adds to Model 1's features)
- Model 1's predicted load (P50) and predicted margin vs. known/estimated asset capacity
- Asset age / last-maintenance-date proxy (where available; placeholder feature until real
  asset registry data exists — flag explicitly, do not silently impute a fake value)
- Historical fault frequency at the asset/zone level (from ENTSO-E unavailability feed or
  Kaggle fault dataset)
- Weather stress indicators: wind speed (line stress), extreme heat duration (multi-day)
- Time since last known outage at that asset

### 2.3 Handling Class Imbalance
- Class weighting (`scale_pos_weight` in XGBoost / `is_unbalance` in LightGBM) as the
  default approach — not SMOTE. Synthetic oversampling on time-series-adjacent tabular data
  risks creating physically implausible feature combinations; prefer weighting or
  threshold-moving on the real event distribution.
- Evaluate with **Precision-Recall AUC**, not ROC-AUC (TRD Section 5) — report both but PR-AUC
  is the one to optimize and the one to put in front of anyone technical.

### 2.4 Output
- Risk score per asset per hour, 0–1, with top-3 contributing factors surfaced in the
  dashboard (e.g., "Predicted load 94% of capacity; sustained heat >35°C for 3 days;
  2 faults in past 90 days").

### 2.5 Explainability — precomputed, not live (updated after deployment testing)

**Original design (as written in 2.4 above):** a live `shap.TreeExplainer` call per API
request, returning exact per-request feature contributions.

**What actually shipped, and why:** `shap` pulls in `numba` + `llvmlite` + `scipy`
transitively — roughly 95MB combined. That mattered concretely during a Vercel
serverless deployment attempt (see System Architecture doc), where it was a meaningful
contributor to exceeding the platform's function size limit, and it's unnecessary weight
on any platform for something computable once rather than per-request.

The fix: `models/risk_classifier.py` now runs SHAP exactly once, at training time. Test-
set rows are grouped into four risk buckets (low/moderate/elevated/critical), and the
top-3 features by mean absolute SHAP contribution are saved per bucket to a small JSON
file (`data/models/risk_factors_by_bucket.json`, ~800 bytes). The live API looks up the
bucket matching the current risk score and returns those precomputed factors — no shap,
no numba, no llvmlite, no scipy-beyond-what-lightgbm-already-needs, at serve time.

**Trade-off, stated honestly:** this trades per-request exactness for a large dependency-
weight and latency reduction. "Top factors" become "typical top factors for this risk
level" rather than a computation specific to the exact input row. Reasonable for a
pilot/demo; a production system with tighter explainability requirements should revisit
live SHAP with the dependency cost budgeted in from the start, on a platform with a
persistent process and looser size limits rather than a size-constrained function.

---

## 3. Model Card (required deliverable, keep this current)

Every model version ships with:
- Training data sources + date ranges + known gaps
- Metrics vs. baselines (Section 5 of TRD)
- Known failure modes (e.g., "underperforms during unprecedented weather events outside
  training distribution," "proxy fault labels reduce classifier reliability")
- Explicit statement: "Trained on non-Pakistani data. Not validated for direct operational
  use on NTDC/DISCO infrastructure until retrained on local data per Section 8 of the TRD."

This last line is not optional legal boilerplate — it is the single most important
sentence in the entire document set, because it's the one most tempting to quietly drop
once the demo looks good.

---

## 4. Tooling / Stack

Rewritten 2026-11-06 to describe what is actually installed and run, not the original
plan. Each line below was checked by importing the package or grepping the codebase.

- **Training and serving:** Python 3.12.10 (`requires-python = ">=3.12"`, CI pins
  `"3.12"`). Not 3.11, which the first draft of this doc claimed.
- **Load forecast:** LightGBM 4.7.0 gradient-boosted trees, two deployed artifacts —
  `lightgbm_load_forecast.joblib` (weather-aware) and `lightgbm_load_forecast_no_weather.joblib`
  (reduced-feature fallback for the weather-lag blind tail).
- **Overload risk:** LightGBM 4.7.0 inside a scikit-learn 1.8.0 pipeline, serialized as
  `risk_classifier.joblib`. **XGBoost is not a shipped dependency** — no served code path
  imports it and it appears in no dependency manifest; only
  `scripts/model_head_to_head.py`, an offline comparison, imports it and prints a notice
  when it is absent. The "LightGBM/XGBoost" phrasing in the original §2.1 design described
  a candidate comparison, and that comparison has now actually been run — see §5. The
  separate baseline sweep is `models/train_baselines.py`, whose committed
  `data/models/baseline_results.csv` has exactly four rows: Persistence (t-24h),
  Seasonal-naive (t-168h), LightGBM, and LightGBM no-weather.
- **CNN-LSTM:** designed (`models/train_lstm_cnn.py`) and deliberately *not* deployed.
  PyTorch is not installed on this machine, so that file has never executed here. Treat
  it as a proposal, and treat every accuracy number in this project as a LightGBM number.
- **Experiment tracking:** none in use. MLflow is not installed and no script references
  it. What actually records experiment outcomes is the committed artifact set under
  `data/models/` plus `data/models/closed_loop_report.json`, regenerated by
  `python -m models.closed_loop_validation`.
- **Feature store:** versioned Parquet files committed in this repo
  (`data/raw/`, `data/features/`), loaded directly by the models. No hosted store.
- **Serving:** FastAPI 0.141.1 + uvicorn 0.52.4, deployed via `fastapi deploy` to
  **FastAPI Cloud**; the SPA lives on **Vercel**. Memory ceiling, CORS allowlist, and the
  per-feed data provenance are verified live — see `docs/03_System_Architecture.md` §2.2
  and §2.3 for the measurements behind those choices.

---

## 5. Model choice, measured (run 2026-11-06, 14:26–14:32 UTC)

§1.3 said "if the deep model doesn't clear LightGBM, ship LightGBM", and §2.1 named
LightGBM/XGBoost as candidates. Both were defensible reasoning, and neither had been
measured: the only committed sweep, `models/train_baselines.py`, benchmarks LightGBM against
persistence and seasonal-naive and never pits one *tree learner* against another. That gap
is now closed. `scripts/model_head_to_head.py` is the durable version — it writes nothing
into `data/models/` and reproduces every number below with:

```
python scripts/model_head_to_head.py
```

**Method, and why it is fair:** all three models are fit on the shipped time-ordered split
(`train_val_test_split`: fit 37,409 h, val 4,158 h, test 10,392 h, test starting
2026-06-13T10:00Z) with identical capacity intent — 500 rounds at lr 0.03, early stopping on
the validation slice; Random Forest gets 300 trees with `min_samples_leaf=5` and also fits on
`fit` only, so its validation score stays out-of-sample. Challengers are injected straight
into `models.forecasting.recursive_forecast_batch`, i.e. the recursive multi-step path the
API actually calls, not a one-step reimplementation. And each model gets **its own**
split-conformal band, calibrated on the validation window by the pure
`calibrate_forecast_intervals.calibrate`, so coverage compares models rather than comparing
one model against another model's error quantiles.

### 5.1 One-step (h+1), MAPE on the honest split

| Config | Model | Train | Val | Test | MAE (MW) | Train→test gap |
|---|---|---|---|---|---|---|
| no-weather (**served**) | LightGBM | 2.020% | 2.476% | 2.122% | 180.0 | +0.102 pp |
| no-weather (**served**) | XGBoost | 2.034% | 2.594% | 2.109% | 185.2 | +0.075 pp |
| no-weather (**served**) | Random Forest | 1.357% | 2.425% | **2.006%** | 189.1 | **+0.649 pp** |
| weather-aware | LightGBM | 2.003% | 2.452% | 2.104% | 176.5 | +0.101 pp |
| weather-aware | XGBoost | 2.025% | 2.574% | 2.079% | 179.8 | +0.054 pp |
| weather-aware | Random Forest | 1.294% | 2.387% | **1.968%** | 184.4 | **+0.674 pp** |

Read the MAE column against the MAPE column and the first result is less impressive than it
looks: Random Forest wins test MAPE by 0.1–0.14 pp while losing MAE by 9–12 MW. MAPE rewards
accuracy on GR's small overnight hours; MAE rewards accuracy on ES's peaks, and an overload
alarm cares about MW, not percentages of tiny loads.

### 5.2 Recursive serving path (834 test origins, horizon 48, bands calibrated per model)

| Model | h+1 | h+6 | h+12 | h+24 | h+48 | Coverage (all) | Coverage (h+48) |
|---|---|---|---|---|---|---|---|
| LightGBM (served) | 2.168% | 5.421% | 5.302% | 5.558% | 7.167% | 0.8322 | 0.8022 |
| XGBoost | 2.188% | 5.099% | 5.426% | 5.723% | 7.319% | 0.8501 | 0.8237 |
| Random Forest | **1.926%** | **4.871%** | 5.314% | 5.739% | 7.397% | 0.8126 | **0.7458** |
| Seasonal-naive | 10.081% | 8.693% | 9.845% | 9.847% | 9.860% | — | — |

Nominal band coverage is 0.80. Random Forest's h+48 band delivers 0.746.

### 5.3 Is any of it real? (paired bootstrap over forecast origins)

Stride 1, horizon 24, 10,050 origins, B=2,000 resamples, LightGBM as reference. Negative =
challenger better. A MAPE table without this column is a coin toss dressed up as a result.

| Step | LightGBM | vs Random Forest | vs XGBoost |
|---|---|---|---|
| h+1 | 2.120% | −0.117 pp [−0.145, −0.091] **significant** | −0.014 pp [−0.036, +0.008] indistinguishable |
| h+6 | 5.005% | −0.237 pp [−0.304, −0.168] **significant** | −0.052 pp [−0.094, −0.010] **significant** |
| h+12 | 5.463% | −0.137 pp [−0.210, −0.059] **significant** | −0.013 pp [−0.056, +0.032] indistinguishable |
| h+24 | 5.781% | −0.038 pp [−0.121, +0.046] indistinguishable | −0.024 pp [−0.076, +0.027] indistinguishable |

Per-zone at h+24, where the near-tie turns out to hide a trade:

| Model | ES | GR | IT_SOUTH |
|---|---|---|---|
| LightGBM (served) | **3.256%** | 4.456% | 9.632% |
| XGBoost | 3.443% | 4.454% | 9.373% |
| Random Forest | 3.694% | **4.261%** | **9.273%** |

**Decision: LightGBM stays served.** Random Forest is genuinely better at short lead times
(h+1 to h+12, outside the bootstrap noise) and at 24 h it is not distinguishable from the
incumbent at all — while its h+48 band under-covers at 0.746 against a 0.80 nominal target.
For a product whose purpose is raising an alarm before an overload, an under-covering band is
a manufactured miss, so it loses on the criterion that matters, not on the criterion that
looks best in a table. XGBoost is a wash everywhere and would add an undeclared dependency to
buy it. The recipe's instruction to "pick the best model on TEST accuracy" is not wrong here,
it is just under-specified: best on which metric, at which horizon, significant against what?

What would actually move the needle is not the learner. IT_SOUTH sits at ~9.3–9.6% h+24 MAPE
while ES sits at ~3.3%, and no learner in the sweep changes that, because the pooled model
carries no zone-identity feature and IT_SOUTH's absolute load is far smaller — so the same
lag/roll/calendar features express it worse. Per-zone models, or a relative-scaled target, is
the next measurable step.

### 5.4 What the conventional shuffled split would have claimed instead

The design guidance this project started from said to split 80/10/10 and shuffle, then
cross-validate. Measured on the same parquet, same model, same metrics, that recipe is not
mildly optimistic — it is a different kind of wrong.

- **80.53%** of the honest test window's 10,392 rows (8,369 of them) land inside a shuffled
  80% training partition. Read from the other side, 20.13% of that partition's rows *are*
  rows from the future window. Either way the model has memorised most of what it is
  supposed to be tested on.
- Worse, and this is the mechanism: **99.06%** of the shuffled test rows have their own
  `t−1h` hour still sitting in the shuffled training set. `load_lag_1h` is nearly the answer,
  so the headline autoregressive feature does the leakage's work for it.
- Scored on the contiguous future window anyway, that shuffled-trained model reports
  **h+24 5.058% / coverage 0.849** and **h+48 6.485% / coverage 0.838**, against the honest
  5.558% / 0.856 and 7.167% / 0.802. Between **9% and 17% more accurate at every horizon,
  and apparently better calibrated** — entirely for the wrong reason. The leak does not just
  inflate the number; it inflates confidence in the intervals too.
- Cross-validation, same disease:

| Resampling | Features / rows | Mean MAPE | Std | Folds |
|---|---|---|---|---|
| Shuffled 5-fold KFold | 13 / 51,850 | 2.095% | 0.030 | 2.08, 2.10, 2.05, 2.14, 2.10 |
| Time-ordered TimeSeriesSplit | 13 / 51,850 | 8.001% | 6.095 | 5.11, 8.51, 2.84, 19.59, 3.96 |
| Shuffled 5-fold KFold | 16 / 51,580 | 2.085% | 0.026 | 2.11, 2.04, 2.12, 2.08, 2.08 |
| Time-ordered TimeSeriesSplit | 16 / 51,580 | 17.001% | 18.997 | 5.21, 52.85, 2.83, 20.13, 3.98 |

The shuffled variant is stable to within 0.03 pp and reports ~2%. The honest variant reports
somewhere between 2.8% and 52.9% depending on which era it tests, and both feature configs
agree that the era matters more than the model. A number that is this consistent is not
measuring the forecast — it is measuring how well a boosted tree memorises the row that came
before. This is the entire empirical case for the temporal split, and §1.3's one-line "never
random shuffle" is the whole reason the reported metrics mean anything.

### 5.5 Input smoothing (recipe step 9): measured, and it does not help — do not ship

The recipe this project was handed says: *"for live data, use average of last 10 values to make
prediction stable, not just 1 value."* Smoothing the input of a series whose own lagged values
are the dominant predictors is not a stabiliser, it is an information downgrade — so the claim
was measured rather than argued, on the same committed frame, same model, same split, replayed
through `scripts/smoothing_probe.py` (798 origins at stride 12, horizon 24, test window
2026-06-13 10:00 → 2026-11-04 17:00 UTC; V0 is the shipped path and its replay was asserted
bit-for-bit equal to `recursive_forecast_batch`, `max|diff| = 0.000e+00`).

| Variant | what changes | h+1 | h+6 | h+12 | h+24 | MAE |
|---|---|---|---|---|---|---|
| **V0** | incumbent control | **2.026%** | **5.004%** | **5.317%** | **5.683%** | **431.2 MW** |
| V1 | last-10h mean *replaces* `load_lag_1h` | 9.255% | 12.182% | 7.416% | 8.573% | 1096.2 MW |
| V2 | last-10h mean *added* as a feature | 2.041% | 5.152% | 6.029% | 5.983% | 476.2 MW |
| V3 | whole 168h history smoothed by MA(10) before features are derived | 11.080% | 9.043% | 7.657% | 8.410% | 963.3 MW |

Paired bootstrap over the same 798 origins, B=1000, delta = challenger − control (positive =
smoothing is worse):

- **V1 is catastrophic**: +7.23 pp at h+1 [6.88, 7.58], +7.18 pp at h+6, +2.10 pp at h+12,
  +2.89 pp at h+24 — all significant, and the control is better at 92%/83%/68%/67% of origins.
  Taking the recipe literally (average instead of the last value) more than doubles h+1 error.
- **V3 is bad everywhere too**: +9.05 pp at h+1 [8.67, 9.45], +2.73 pp at h+24.
- **V2**, the gentlest reading (keep the lag, add the 10h mean as one more column), is a
  **wash to a loss**: h+1 +0.015 pp with a CI that includes zero, then significantly *worse* at
  h+6 (+0.148 [0.005, 0.294]), h+12 (+0.712 [0.497, 0.923]) and h+24 (+0.300 [0.120, 0.475]).
  It wins on under half of origins at every horizon past h+1.

**Decision: no input smoothing ships, in any form.** The reason is visible in the redundancy
numbers: a 10-hour mean correlates 0.76 with `load_roll_mean_24h` and 0.65 with `load_lag_1h`
already in the feature table, so V2 buys a near-duplicate column and pays for it in variance,
while V1 and V3 delete the single most informative hour. The smoothed context the recipe asks
for has been present since the first feature build (`load_roll_mean_24h`, `load_roll_std_24h`,
`load_roll_mean_168h`) — §1.2 lists them for exactly this purpose.

*Not verified:* the probe holds stride at 12 rather than 1 (798 origins instead of 10,050) to
keep the four-variant sweep to ~145 s, so its confidence intervals are wider than §5.3's. That
does not threaten the conclusion — V1 and V3 miss by margins far larger than the interval
widths — but a reader should not reuse these MAPE figures as a substitute for §5.2's.

### 5.6 Train/serve feature parity (recipe step 3): measured, and an audit claim refuted

Recipe step 3 asks for "same preprocessing for train and live data: scaling + handling missing
values. Save the scaler." Two separate things have to be true for that to be satisfied, and
both are now measured rather than asserted.

**There is no scaler, deliberately.** The served models are gradient-boosted trees, which are
scale-invariant: a monotone rescaling of a column cannot change a split point's partition of the
data. §7 of the recipe ("save the scaler as scaler.pkl") has nothing to save here, and shipping
one would add a fitting-time state object to keep in sync for zero effect. Missing values are
handled structurally instead — LightGBM routes NaNs to a default child per split — and the
reduced-feature fallback artifact exists for the hours where weather is genuinely absent, rather
than imputing them.

**Parity is exact on the load features.** An audit agent claimed a P0 train/serve skew: that
`load_roll_std_24h` and `load_roll_mean_168h` in `models/forecasting.py` cover `[t−23, t]` and
`[t−167, t]` while `features/build_features.py` computes `shift(1).rolling(...)` over the hours
strictly before the target, supposedly differing by ~1.18 MW on average and up to 3.1 MW. That
claim is **false**, and the `docstring` at `models/forecasting.py:7` is accurate.

The refutation is direct rather than analytical: the real `recursive_forecast_batch` was run
with the served LightGBM model wrapped so that the exact feature frame it hands the model is
captured, and each captured value was compared against the committed parquet row **for the same
target timestamp**. Over four target hours spread across two years of GR data:

```
serving − training, per contested feature (0.0 = identical)
  load_lag_1h         0.0   0.0   0.0   0.0
  load_lag_24h        0.0   0.0   0.0   0.0
  load_lag_168h       0.0   0.0   0.0   0.0
  load_roll_mean_24h  0.0   0.0   0.0   0.0
  load_roll_std_24h   0.0  -0.0  -0.0   0.0
  load_roll_mean_168h 0.0   0.0   0.0   0.0
```

The audit's error is an anchoring one, and it is worth naming because it is the kind of mistake
that gets a healthy system "fixed": in `recursive_forecast_batch`, `end = n_hist + step − 1`
indexes the **target** hour, not the origin, so `buf[:, end−24:end]` is already the 24 hours
strictly before the target. The tell was inside the audit's own table — it listed `lag_1h` and
`roll_mean_24h` as matching while claiming `roll_std_24h` did not, yet all three are derived
from the same two slices, so they cannot disagree with each other. A claim that contradicts
itself on two features built by one expression should be re-measured, not patched.

*One real caveat, stated rather than hidden:* at step ≥ 2 the serving path feeds its **own
predictions** into the lag and rolling windows (unavoidable in any recursive multi-step
forecast), whereas the training rows were built from actuals. That is not an endpoint bug, it is
the definition of multi-step forecasting, and it is why §5.2 evaluates the recursive path
separately from the one-step numbers in §5.1 rather than quoting one-step accuracy for a 24-hour
product.

