# GridSense AI — ML/DL Model Design Document

**Purpose:** Concrete architecture, feature set, and training strategy for both models
defined in the TRD. This is written to be buildable, not aspirational.

---

## 1. Model 1 — Short-Term Load Forecasting (Regression)

### 1.1 Architecture: Hybrid LSTM + 1D-CNN (with a Transformer variant as stretch goal)

Why this specific architecture, not "deep learning" as a vague label:

- **1D-CNN layers first:** extract short-range local patterns (hourly ramp shapes, sudden
  spikes) from the raw sequence before it hits the recurrent layers. This mirrors the
  LSTM + modified split-convolution approach used in the published NTDC-data benchmark
  study, which is the closest real precedent available and reported strong results
  (MAPE 2.38% single-step, 3.72% multi-step) on actual Pakistan grid data — that's your
  target ballpark, not an arbitrary one.
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

- **Training:** Python 3.11, PyTorch (LSTM/CNN), LightGBM/XGBoost (risk classifier)
- **Experiment tracking:** MLflow (self-hosted on Alibaba Cloud ECS, or Alibaba PAI's
  built-in experiment tracking if adopted — verify current feature parity before committing)
- **Feature store:** simple versioned Parquet files on Alibaba Cloud OSS for prototype
  scale; revisit a proper feature store (Feast or similar) only once multi-zone,
  multi-source production data volume justifies the added complexity
- **Serving:** FastAPI wrapping the trained models, containerized, deployed on Alibaba Cloud
  ECS/Container Service — see System Architecture doc for full pipeline
