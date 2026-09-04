# GridSense AI — Tech Stack

**Purpose:** Every technology used, why it was chosen, and what was tested/measured
rather than assumed.

---

## Backend

| Component | Technology | Why |
|---|---|---|
| API framework | FastAPI 0.141.1 | Async-native, automatic OpenAPI docs, matches the deploy target (FastAPI Cloud is built by FastAPI's own creators) |
| ASGI server | Uvicorn 0.52.4 | Standard FastAPI pairing |
| Data validation | Pydantic 2.13.4 | FastAPI's native validation layer, used for every response schema |
| Data manipulation | Pandas 3.0.2 | Feature engineering, joins, rolling windows — industry standard despite real memory cost (see Database/Memory notes below) |
| Numerical computing | NumPy 2.4.4 | Underlying pandas, used directly for cyclic feature encoding |
| Columnar storage | PyArrow 25.0.1 | Parquet read/write — efficient storage for the feature snapshot |
| Gradient boosting | LightGBM 4.7.0 | Both the load forecaster and risk classifier. Chosen over deep learning for the risk classifier specifically because imbalanced tabular data favors trees, and trees give explainability (via SHAP) essentially for free |
| Explainability | SHAP 0.52.0 | Training-time only (see ML Model Design doc Section 2.5) — removed from the live serving path entirely after it was found to add ~95MB of dependency weight for no live benefit |
| Model persistence | joblib 1.5.3 | Standard for scikit-learn/LightGBM model serialization |
| Real grid data | entsoe-py 0.6.19 | Wraps ENTSO-E Transparency Platform's XML API into pandas DataFrames |
| Real weather data | requests 2.32.3 | Direct calls to NASA POWER's REST API (no client library needed, no auth required) |

## Frontend

| Component | Technology | Why |
|---|---|---|
| UI framework | React 18.3.1 | Standard, matches Vite's first-class support |
| Build tool | Vite 5.4.21 | Fast dev server, clean env-var handling (`import.meta.env`), straightforward Vercel deployment |
| 3D rendering | Three.js 0.128.0 | Powers the grid topology visualization — wireframe nodes, dynamic risk-color rings, dashed transmission lines |
| Charting | Recharts 2.12.7 | The 24h forecast chart with P10/P50/P90 confidence bands |

## Infrastructure / Deployment

| Component | Platform | Why (tested, not assumed) |
|---|---|---|
| Backend hosting | FastAPI Cloud | Persistent-process host (needed — the API loads trained models into memory at startup, a stateless function architecture doesn't fit this). Chosen over Render specifically because Render requires a card on file even for its free tier, which was an actual blocker. Chosen over Vercel because a Vercel serverless function was directly tested at 282MB minimum viable size against Vercel's ~250MB limit — genuinely doesn't fit, not a configuration problem |
| Frontend hosting | Vercel | Standard static/SPA hosting, zero issues, this is exactly the workload Vercel is built for |
| Backend fallback | Render (`render.yaml` kept in repo) | Documented alternative if a card becomes available or a production (non-free) deployment is preferred later |
| Version control | Git / GitHub, two repos | Backend and frontend deploy to fundamentally different platforms with different build systems — see Two-Repo Setup Guide for the full reasoning |

## Data Sources

| Source | What it provides | Access |
|---|---|---|
| ENTSO-E Transparency Platform | Real hourly grid load, 35 European countries | Free, requires approved API token (mandatory EU open-data regulation) |
| NASA POWER API | Hourly temperature, humidity, wind, precipitation, any coordinates | Free, no registration, no API key |
| (Bootstrap only) Synthetic generator | Schema-matched fake data with realistic diurnal/weekly/seasonal structure | Used to validate the pipeline end-to-end before real ENTSO-E access — see Pipeline Documentation for the honest real-vs-synthetic breakdown |

## Explicitly NOT in the stack, and why

| Rejected | Why |
|---|---|
| Deep learning (LSTM/CNN) for risk classification | Imbalanced tabular data favors gradient-boosted trees; architecture is designed (`models/train_lstm_cnn.py`) but not deployed, since LightGBM already cleared accuracy baselines by a wide margin and didn't need the added complexity justified |
| Live SHAP in the serving path | Tested and measured to add ~95MB of dependency weight (numba+llvmlite+scipy transitively) for a computation that doesn't need to happen per-request — moved to training-time, bucketed, precomputed |
| A database | Not needed at current scale/workflow — see `10_Database_Assessment.md` for the specific, identifiable trigger point where this changes |
| Vercel for the backend | Directly tested and measured to not fit — see System Architecture doc Section 2.2 |
| scikit-learn in the live API | Only needed for training-time metrics (PR-AUC, precision-recall curve), not for serving predictions |

## Version pinning policy

All deploy-critical dependencies are pinned to exact versions (`==`, not `>=`) in both
`requirements.txt` and `pyproject.toml`. This is a direct response to a real incident
during testing: an open `>=` constraint silently resolved to pandas 3.0.2 (a major
version with breaking changes from 2.x) and a LightGBM version that deprecated the
`eval_set=` API this project's training scripts used — both caught only by actually
running the pipeline, not by reading the code. See Pipeline Documentation Section 4 for
the full incident history.
