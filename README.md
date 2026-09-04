# GridSense AI — Backend

**AI-driven smart grid load forecasting and overload risk prediction API.**

FastAPI · LightGBM · ENTSO-E · NASA POWER · Deploys to FastAPI Cloud

[![Python](https://img.shields.io/badge/Python-3.12-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115+-009688.svg)](https://fastapi.tiangolo.com/)
[![LightGBM](https://img.shields.io/badge/LightGBM-4.0+-success.svg)](https://lightgbm.readthedocs.io/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](#license)

---

## What it does

GridSense AI predicts electricity grid overload **before it happens**. It ingests hourly
grid load data (ENTSO-E Transparency Platform) and weather data (NASA POWER), engineers
leak-free time-series features, and serves two trained models through a REST API:

- **Load Forecaster** (LightGBM regressor) — hourly demand forecast with P10/P50/P90
  uncertainty bands. MAPE **1.72%** vs. 5.52% persistence baseline.
- **Overload Risk Classifier** (LightGBM classifier) — per-zone risk score with
  precomputed SHAP explanations. PR-AUC **0.9683**.
- **Load-Shedding Recommender** (rules-based) — priority-ranked shedding schedule with
  human-readable reasoning. The system *recommends*; it never controls anything.

Zones covered: **Spain (ES)** · **Greece (GR)** · **Southern Italy (IT_SOUTH)** — with the
architecture designed to retrain on Pakistani NTDC/DISCO grid data once access is secured.

> **Honesty note:** the committed model artifacts are currently trained on realistic
> **synthetic bootstrap data** (documented, not hidden) so the entire pipeline could be
> built and verified end-to-end while real ENTSO-E API approval was pending. The
> real-data ingestion code is complete — see
> [`docs/04_Pipeline_Documentation.md`](docs/04_Pipeline_Documentation.md) before citing
> any accuracy number.

---

## Architecture

```
ENTSO-E API ──┐
              ├─► ingestion/ ─► features/build_features.py ─► models/ (training)
NASA POWER ───┘                        │                          │
                                       ▼                          ▼
                            training_features.parquet    *.joblib + SHAP JSON
                                       │                          │
                                       └────────► api/main.py ◄───┘
                                                     │
                                              FastAPI Cloud
                                                     │
                                              React frontend
                                          (separate repo, Vercel)
```

Training happens **locally**; only the trained artifacts (~3 MB) are committed and
deployed. The serving path deliberately excludes `shap`, `scikit-learn`, and `torch` to
keep the deployed footprint lean.

---

## API Endpoints

| Endpoint | Method | Description |
|---|---|---|
| `/` | GET | Health check — service status and configured zones |
| `/forecast/{zone}?horizon_hours=24` | GET | Load forecast with P10/P50/P90 bands (1–48h) |
| `/risk/{zone}` | GET | Overload risk score, bucket, top SHAP factors, load vs. capacity |
| `/assets` | GET | Risk-ranked list of all monitored assets across zones |
| `/shed-schedule/{zone}` | GET | Recommended load-shedding priorities with reasoning |

Example:

```bash
curl https://<your-app>.fastapicloud.dev/risk/ES
```

```json
{
  "zone": "ES",
  "risk_score": 0.9999,
  "predicted_load_mw": 7784.7,
  "capacity_mw": 7330.78,
  "top_factors": [
    {"feature": "load_lag_1h", "contribution": 4.743},
    {"feature": "temperature_c", "contribution": 2.0245}
  ],
  "is_proxy_capacity": true
}
```

---

## Quick Start (Local)

```bash
git clone https://github.com/huzaifahimad/Grid_Sense.AI-Backend-.git
cd Grid_Sense.AI-Backend-
pip install -r requirements-api.txt   # lean serving deps
uvicorn api.main:app --port 8000
curl http://localhost:8000/assets
```

Models are pre-trained and committed — the API works out of the box with zero training.

### Retraining with real data

```bash
pip install -r requirements.txt              # full training deps
export ENTSOE_API_KEY="your-token"           # from transparency.entsoe.eu
python -m ingestion.nasa_power_client        # weather (no key needed)
python -m ingestion.entsoe_client            # real load data
python -m features.build_features            # feature engineering
python -m models.train_baselines             # baselines + LightGBM (mandatory first)
python -m models.risk_classifier             # risk classifier + SHAP precompute
```

Full run order and real-vs-synthetic breakdown:
[`docs/04_Pipeline_Documentation.md`](docs/04_Pipeline_Documentation.md)

---

## Environment Variables

| Variable | Where | Required | Purpose |
|---|---|---|---|
| `ENTSOE_API_KEY` | FastAPI Cloud secret store / local shell | Only for real-data ingestion | ENTSO-E Transparency Platform API token. **Never commit it** — `config.py` reads it from the environment only. |

Set it on FastAPI Cloud:

```bash
fastapi cloud env set --secret ENTSOE_API_KEY "your-token"
```

The serving API itself runs **without** this key — predictions come from committed
model artifacts, not live ENTSO-E pulls.

---

## Deployment

Primary target: **[FastAPI Cloud](https://fastapicloud.com)** (persistent process,
models stay loaded in memory, no credit card required).

```bash
pip install fastapi-cli
fastapi login
fastapi deploy
```

Fallback: **Render** via the committed [`render.yaml`](render.yaml).

Why not Vercel? The ML dependency chain (lightgbm → scipy) measures 282 MB minimum —
over Vercel's 250 MB serverless limit. Tested, not assumed. Full story:
[`docs/03_System_Architecture.md`](docs/03_System_Architecture.md).

Step-by-step guide: [`docs/05_Deployment_Guide.md`](docs/05_Deployment_Guide.md)

---

## Project Structure

```
├── api/main.py              # FastAPI app — 5 endpoints, model loading
├── config.py                # Zones, paths, thresholds, env-only API key
├── ingestion/               # ENTSO-E, NASA POWER, Kaggle, synthetic bootstrap
├── features/                # Leak-free feature engineering (lags, rolling, cyclic)
├── models/                  # LightGBM training, risk classifier, LSTM-CNN (designed)
├── data/
│   ├── features/            # training_features.parquet (committed)
│   └── models/              # *.joblib + SHAP JSON + baselines CSV (committed)
├── docs/                    # 13-document suite (TRD → presentation source)
├── pyproject.toml           # FastAPI Cloud deploy manifest
├── requirements.txt         # Full training dependencies
└── requirements-api.txt     # Lean serving dependencies
```

## Documentation

| Doc | Covers |
|---|---|
| [`00_TRD.md`](docs/00_TRD.md) | Requirements, decisions, known limitations |
| [`02_ML_Model_Design.md`](docs/02_ML_Model_Design.md) | Model architectures, features, training strategy |
| [`03_System_Architecture.md`](docs/03_System_Architecture.md) | Deployment architecture and its rationale |
| [`04_Pipeline_Documentation.md`](docs/04_Pipeline_Documentation.md) | Data flow, real-vs-synthetic, bug post-mortems |
| [`05_Deployment_Guide.md`](docs/05_Deployment_Guide.md) | Step-by-step deployment |
| [`06_Project_Overview.md`](docs/06_Project_Overview.md) | Plain-language orientation — start here |
| [`08_Testing_Validation_Prompt.md`](docs/08_Testing_Validation_Prompt.md) | 6-phase QA verification prompt |

## Related

Frontend dashboard (React 18 + Three.js):
**[Grid_Sense.AI-frontend-](https://github.com/huzaifahimad/Grid_Sense.AI-frontend-)**

## License

MIT — free to use, modify, and distribute. ENTSO-E data usage is subject to the
[ENTSO-E Transparency Platform terms](https://transparency.entsoe.eu/).
