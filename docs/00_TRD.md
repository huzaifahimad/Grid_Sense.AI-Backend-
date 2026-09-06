# GridSense AI — Technical Requirements Document (TRD)

**Version:** 2.0 (updated post-pipeline-validation and deployment architecture finalization)
**Status:** Backend and frontend both live-tested; two-repo deployment architecture finalized

---

## 1. Purpose and Scope

GridSense AI predicts power grid overload and outage risk ahead of occurrence, and
outputs a recommended load-shedding schedule. Trained initially on open European grid
data (ENTSO-E) and weather data (NASA POWER), architected to retrain on Pakistani
NTDC/DISCO data once institutional access is secured.

**In scope, currently built and tested:**
- Short-term load forecasting via LightGBM (1–48h ahead)
- Overload risk classification via LightGBM with precomputed, bucketed explainability
- REST API serving both models plus a rules-based shed-schedule recommendation
- 3D-visualization React dashboard, deployable independently of the backend

**Explicitly out of scope:**
- Real-time SCADA integration (no live telemetry access)
- Automated control actions — system recommends only, never executes
- LSTM/CNN deep model — architecture designed in `models/train_lstm_cnn.py` but not
  trained/deployed in this version; LightGBM cleared the baselines by a wide enough
  margin on available data that the added complexity wasn't justified (see Section 5)

---

## 2. Problem Definition

Two distinct ML problems, deliberately kept as two separate models:

| Problem | Type | Output |
|---|---|---|
| Load forecasting | Regression | Predicted MW demand per zone, P10/P50/P90 |
| Overload risk | Classification (imbalanced) | P(overload) per asset, 0–1, with top contributing factors |

---

## 3. Architecture Decisions (finalized this iteration, with rationale)

### 3.1 Deployment split: two repositories, two platforms

- **Backend repo** → FastAPI Cloud (primary), Render (documented fallback)
- **Frontend repo** → Vercel

**Why not one repo, one platform:** an earlier attempt to deploy the Python ML backend
as Vercel serverless functions was tested directly and failed. Even the absolute minimum
dependency set (fastapi + uvicorn + pydantic + lightgbm + joblib, no pandas, no pyarrow,
no shap) measured 282MB uncompressed — over Vercel's ~250MB serverless function limit,
because `lightgbm` mandatorily requires `scipy` (109MB alone). This is a hard constraint,
not a configuration issue — there is no viable trim path that fits a LightGBM-serving
Python backend into a Vercel function. Vercel is frontend-only in this architecture.

**Why FastAPI Cloud over Render as the primary backend target:** Render requires a card
on file even for its free tier (identity/anti-abuse verification), which was a hard
blocker with no card available. FastAPI Cloud requires no card, is built by the FastAPI
framework's own creators, and its `fastapi deploy` workflow matches this project's simple
request/response API shape (no long-running jobs, no websockets — exactly what it's
built for). Render config is kept in the repo as a documented fallback.

### 3.2 SHAP explainability: precomputed at training time, not live

**Original design:** live `shap.TreeExplainer` call per API request.
**Problem found:** `shap` pulls in `numba` + `llvmlite` + `scipy` transitively — about
95MB combined — which was most of the reason the Vercel-serverless attempt failed, and
is unnecessary weight on any platform for what is fundamentally a training-time
computation.
**Fix:** `models/risk_classifier.py` now runs SHAP once, at training time, buckets rows
by risk level (low/moderate/elevated/critical), and saves the top-3 average feature
contributions per bucket to `data/models/risk_factors_by_bucket.json`. The live API reads
this small JSON file instead of computing SHAP per request.
**Trade-off, stated plainly:** explanations are now "typical top factors for this risk
level" rather than a live, request-specific computation. This is a real accuracy-of-
explanation cost. Acceptable for a pilot/demo; revisit if per-request precision matters
for a production deployment.

---

## 4. Success Metrics and Actual Results

**Important framing:** the numbers below were produced on SYNTHETIC data (see
`docs/06_Pipeline_Documentation.md` for why), generated to validate the pipeline
mechanically end to end while real ENTSO-E ingestion was pending. They prove the code
works. They do not prove real-world accuracy. Do not cite these numbers to anyone as
product performance without re-running on real data first.

### Load forecasting (on synthetic data)

| Model | MAPE | MAE (MW) | RMSE (MW) |
|---|---|---|---|
| Persistence (t-24h) | 6.59% | 221.09 | 288.03 |
| Seasonal-naive (t-168h) | 5.20% | 181.00 | 231.56 |
| **LightGBM** | **2.23%** | **76.67** | **97.08** |

LightGBM clears both baselines by a wide margin. Expect this margin to shrink and MAPE
to rise on real data — synthetic data is smoother and more learnable by construction.
Real published NTDC-data benchmark studies report 2.38–3.72% MAPE for LSTM-hybrid
models — that is the credible external reference point, not this synthetic result.

### Risk classification

PR-AUC: 0.9595 on the current synthetic pipeline run (after removing the label-leakage feature).

### 4.1 Bug found and fixed during live testing

An earlier validation run showed near-perfect PR-AUC with all three zones returning an identical risk score to 13
decimal places. Root cause: `load_to_capacity_ratio` was both an input feature AND the
basis of the proxy label (`overload = load/capacity > 0.9`), so the classifier was
reading the label off a near-copy of itself rather than learning anything. Fixed by
removing that feature from the risk classifier's `FEATURE_COLS` (it remains available
for the shed-schedule rules layer, which legitimately needs it). PR-AUC dropped to
0.9595 post-fix — a lower, honest number, which is the correct outcome of fixing real
leakage, not a regression.

---

## 5. Known Limitations (stated up front, not discovered later)

1. **Transfer-learning risk is real and unvalidated.** Trained on European load/climate
   patterns; Pakistan's AC-driven summer peak, monsoon patterns, and load-shedding-as-
   normal-operation are structurally different. Expect real retraining effort, not a
   config change, when real Pakistani data arrives.
2. **No real fault/outage event data.** The overload proxy label
   (`load > 0.9 × capacity`) is a standard technique but a real accuracy ceiling until
   genuine historical outage records are available.
3. **Capacity values are a proxy** (99th percentile of historical load per zone) unless
   real asset capacity data is supplied — `capacity_is_proxy: true` is returned on every
   `/risk` response until this changes.
4. **Precomputed SHAP factors are bucketed, not per-request-exact** (Section 3.2).
5. **Feature data is served from a static file, not a live database.** `api/main.py`
   reads `data/features/training_features.parquet` directly — fine for a pilot/demo,
   not how a real production system serving live utility data should work. See
   `docs/06_Pipeline_Documentation.md`.
6. **CORS is currently open** (`allow_origins=["*"]`) — tighten to the real Vercel domain
   before treating this as more than a pilot/demo.
7. **This system recommends; it never decides or executes.** No claim anywhere in this
   project should imply autonomous control over grid infrastructure.

---

## 6. Companion Documents

- `docs/01_Data_Sourcing_Plan.md` — real datasets, APIs, access steps, licensing
- `docs/02_ML_Model_Design.md` — model architecture, features, training strategy
- `docs/03_System_Architecture.md` — full infra/pipeline diagram, both platforms
- `docs/04_Pipeline_Documentation.md` — data flow, what's real vs. synthetic, run order
- `docs/05_Deployment_Guide.md` — exact steps, both repos, both platforms
- `docs/06_Project_Overview.md` — plain-language explanation of the whole project
- `docs/07_User_Manual.md` — how to operate the dashboard
- `docs/08_Testing_Validation_Prompt.md` — comprehensive Antigravity QA prompt
- `docs/09_Two_Repo_Setup_Guide.md` — exactly how to split and push two repos
