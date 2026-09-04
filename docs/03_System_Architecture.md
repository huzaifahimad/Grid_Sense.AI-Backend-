# GridSense AI — System Architecture Document

**Purpose:** How the pieces actually connect, end to end, across the finalized
two-repository, two-platform deployment — and the real testing that led to this
architecture, not a theoretical design.

---

## 1. High-Level Pipeline

```
[External Data Sources]
   ENTSO-E API / NASA POWER API
         │
         ▼
[Ingestion — local, gridsense-backend repo, ingestion/]
   - entsoe_client.py, nasa_power_client.py
   - Writes raw Parquet to data/raw/ (gitignored — regenerable, not versioned)
         │
         ▼
[Feature Engineering — local, features/build_features.py]
   - Joins load + weather, computes lags/cyclic features/proxy overload label
   - Writes data/features/training_features.parquet
   - COMMITTED to git — the live API reads this file directly (see Section 4)
         │
         ▼
[Model Training — local, models/]
   - train_baselines.py: persistence, seasonal-naive, LightGBM (run FIRST, mandatory)
   - risk_classifier.py: LightGBM + SHAP run ONCE here, bucketed factors saved to JSON
   - Outputs COMMITTED to git: data/models/*.joblib, risk_factors_by_bucket.json
         │
         ▼
[API — api/main.py, deployed via `fastapi deploy` to FastAPI Cloud]
   - Loads committed model files + feature snapshot at startup
   - /forecast/{zone}, /risk/{zone}, /assets, /shed-schedule/{zone}
   - NO shap/scikit-learn/numba/llvmlite in the deployed dependency set
         │
         ▼
[Frontend — separate repo, deployed to Vercel]
   - React + Three.js + Recharts, reads VITE_API_BASE_URL at build time
   - Calls the live FastAPI Cloud URL for real predictions
```

---

## 2. Deployment Architecture: Two Repos, Two Platforms

### 2.1 Why two repos

`gridsense-backend` (Python/FastAPI) and `gridsense-frontend` (React/Vite) are separate
repositories deploying to separate platforms. This isn't incidental — it was tested both
ways.

### 2.2 Why the backend is NOT on Vercel — tested, not assumed

An earlier attempt merged backend Python files (`api/`, `ml_models/`,
`requirements.txt`) into the frontend repo and configured `vercel.json` to build them as
Python serverless functions. That build failed. Root-caused and measured directly:

| Dependency stack tested | Uncompressed size |
|---|---|
| Full original stack (fastapi, pandas, pyarrow, lightgbm, shap, scikit-learn) | 512MB |
| Stripped to fastapi + uvicorn + pydantic + lightgbm + joblib only (no pandas/pyarrow/shap) | 282MB |
| Vercel serverless function limit | ~250MB |

Even the absolute minimum viable stack is over the limit. Root cause: `lightgbm`
mandatorily requires `scipy` (109MB alone) — there is no dependency trim that removes
this. **Conclusion: a LightGBM-serving Python backend does not fit in a Vercel
serverless function, under any configuration.** This is a platform mismatch, not a bug
to fix. Vercel is frontend-only in this architecture, permanently.

### 2.3 Why FastAPI Cloud over Render for the backend

Both were evaluated as legitimate options for a persistent-process Python host (which is
what this backend actually needs — it loads model files into memory at startup, unlike a
stateless function):

- **Render:** free tier available, `render.yaml` blueprint config written and tested
  successfully earlier in this project's development. Requires a credit card on file even
  for the $0 plan (stated anti-abuse policy) — this was a hard blocker with no card
  available.
- **FastAPI Cloud:** no card required, built by the creators of the FastAPI framework
  itself, `fastapi deploy` CLI workflow. Chosen as the primary target for this reason.
  Currently in Public Beta — a real, funded (Sequoia-backed), actively developed
  platform, not an abandoned side project, but newer than Render with a shorter track
  record.

`render.yaml` is kept in the backend repo as a documented fallback, not deleted — if a
card becomes available later or Render's persistent-process model is preferred for a
future production (non-free) deployment, it's ready to use.

---

## 3. Explainability Architecture: Precomputed SHAP

Covered in full in `docs/02_ML_Model_Design.md` Section 2.5. Summary for architecture
purposes: SHAP computation moved from live (per-API-request) to training-time
(once, bucketed, saved as a ~800-byte JSON file). This was necessary regardless of which
platform hosts the backend — it removes ~95MB of dependency weight
(`shap`+`numba`+`llvmlite`+`scipy`-beyond-lightgbm's-own-requirement) and removes a
live computation from the request path, improving both deployability and latency.

---

## 4. A Known Architectural Shortcut (stated plainly, not hidden)

`api/main.py` reads `data/features/training_features.parquet` directly from disk at
startup/request time, and this file is committed to git rather than generated live. This
means:

- Predictions reflect whatever data existed the last time the pipeline was run locally
  and the resulting file was committed and deployed — NOT live, continuously updating
  data.
- Updating predictions requires: re-run ingestion → features → training locally → commit
  the new files → redeploy.

This is a reasonable shortcut for a pilot/demo/portfolio deployment. It is not how a
production system serving a real utility should work — that needs either a scheduled
ingestion job writing to a real database, or the API computing features on-demand from a
live data source. Flagging this explicitly so it's a known, chosen trade-off, not a
silently discovered gap later.

---

## 5. Platform Service Mapping

| Component | Platform | Notes |
|---|---|---|
| Backend API | FastAPI Cloud | `fastapi deploy`, scale-to-zero when idle |
| Backend fallback | Render | `render.yaml` present, requires card, not primary |
| Frontend | Vercel | Standard Vite SPA build, `vercel.json` handles SPA routing |
| Secrets (ENTSO-E token) | FastAPI Cloud env (`fastapi cloud env set --secret`) | Never committed to git |
| Frontend env vars | Vercel dashboard | `VITE_API_BASE_URL`, `VITE_USE_MOCK_DATA` |

---

## 7. Memory Risk — Found During Testing, Partially Mitigated, Not Fully Resolved

Real, measured finding, not a theoretical concern: the deployed API's peak RAM usage
under light request load was measured at **412MB**, against FastAPI Cloud's Hobby tier
hard limit of **512MB**. That leaves roughly 100MB of headroom — under light,
single-request local testing, not concurrent load or a larger real-data feature set.

**Root cause, isolated precisely:** importing `pandas` alone costs ~88MB, `lightgbm`
another ~80MB, before a single row of data or a trained model is loaded. The dependency
libraries themselves, not the data volume, are the dominant memory cost.

**Mitigation applied:** `api/main.py` now trims the feature dataframe to the last 200
rows per zone at startup instead of holding the full history (every endpoint only ever
reads the last 1–48 rows per zone anyway — confirmed by checking every usage site). This
reduced measured RSS by less than 1% (410MB vs. 412MB) — it was directionally correct
practice and will matter more as real data volume grows, but it did NOT meaningfully
address the actual bottleneck, because the bottleneck is import-time library weight, not
this project's specific dataset size.

**What was NOT done, and why:** removing pandas/lightgbm entirely would require
rewriting the entire feature engineering and model-serving layer around lighter
alternatives — a substantial rewrite, not a tuning pass, and one that would need its own
careful validation. Not undertaken in this pass; flagged as a real open risk instead of
silently absorbed into "it works."

**What to actually do about this:** monitor real memory usage on the live FastAPI Cloud
deployment using their built-in metrics (their pricing page lists "Basic metrics
(CPU/Memory)" as an included feature) rather than trusting this sandbox's local Linux
numbers — actual container behavior on their infrastructure could differ. If real usage
approaches the limit or the service gets OOM-killed under real traffic, the fix options,
roughly in order of effort: (1) check whether FastAPI Cloud's Pro tier offers more RAM
(unclear from their public pricing table as of this writing — verify directly), (2) move
to Render or another persistent-process host with a higher memory ceiling, (3) undertake
the larger rewrite to reduce the pandas/lightgbm import footprint itself.

---

## 8. What Ships Now vs. Roadmap

**Ships now (tested, working):**
- Full pipeline, run and validated end to end on synthetic data (real ENTSO-E ingestion
  code exists and is correct, but requires the person's own token/machine to run — see
  Pipeline Documentation)
- Real trained LightGBM models for both forecasting and risk classification
- A found-and-fixed data leakage bug in the risk classifier (documented in TRD Section 4.1)
- Live-tested API, all 5 endpoints, real JSON responses verified via curl
- Frontend that builds successfully and connects to the real API via environment variable

**Explicit roadmap, not claimed as done:**
- Real ENTSO-E/Pakistani grid data replacing the synthetic validation dataset
- NTDC/DISCO data integration
- Real-time SCADA ingestion
- Live database instead of the static feature-snapshot shortcut (Section 4)
- Per-request live SHAP if a production deployment's explainability requirements demand it
- LSTM+CNN quantile forecaster (architecture designed, not trained/deployed)
