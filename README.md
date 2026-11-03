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

- **Load Forecaster** (LightGBM regressor) — recursive 1–48h-ahead demand forecast from
  the latest observation. P10/P90 are **split-conformal intervals** calibrated per horizon
  on a held-out validation window (test coverage **84.8%** vs. 80% nominal). Honest test
  MAPE: **2.24%** at 1h ahead, **~4.3%** at 6–48h ahead vs. **~5.1%** seasonal-naive and
  **6.59%** persistence. Future weather uses persistence (same hour, previous day) until
  an NWP feed is connected.
- **Overload Risk Classifier** (LightGBM classifier) — per-zone risk score with
  precomputed SHAP explanations. Test PR-AUC **0.9637**, Brier **0.0185** (base rate
  0.0975). Early stopping uses a validation slice, never the test set.
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

**Tests:** `python -m pytest tests -q` — 47 tests covering endpoint contracts, train/serve
feature parity, label-leak guard, interval calibration, the lean serving import path, and
the closed-loop validator (gate grading, lead-time math, and negative controls that prove
the gates fail when they should).

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
| `/` | GET | Liveness — service status and configured zones |
| `/health` | GET | Readiness — per-artifact load state, serving data window, interval method |
| `/forecast/{zone}?horizon_hours=24` | GET | Recursive forecast (1–48h) with calibrated P10/P50/P90, origin timestamp and step |
| `/risk/{zone}` | GET | Overload risk score, bucket, top SHAP factors, load vs. capacity |
| `/assets` | GET | Risk-ranked list of all monitored assets across zones |
| `/shed-schedule/{zone}` | GET | Advisory shedding recommendation for elevated/critical zones only (priority 1 = critical ≥0.8, 2 = elevated ≥0.6) |
| `/validation` | GET | Closed-loop release-gate report — 13 gates, walk-forward replay metrics, provenance and stated limitations, plus report age in hours |

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

## Closed-Loop Validation

Accuracy claims are checked by **replaying the held-out test window through the same code
path the API serves from** — recursive forecast, calibrated intervals, risk classifier,
shedding rules — not by reading a training-time score.

```bash
python -m models.closed_loop_validation          # writes data/models/closed_loop_report.json
python -m models.closed_loop_validation --strict # exits 1 if any gate is WARN/FAIL
```

The replay covers 3,471 zone-hours (10,413 scored rows, 10,341 hourly forecast origins,
24-hour horizon) across **ES · GR · IT_SOUTH**, and grades 13 gates against a target and a
hard floor. Current report: **overall PASS, 13/13**.

| Gate family | Measured |
|---|---|
| Forecast skill | h+1 MAPE 2.24% vs 5.21% seasonal-naive (**+56.9%**); h+24 4.36% vs 5.20% (**+16.1%**) |
| Interval honesty | P10–P90 coverage 84.3% (target 70–95%), week-to-week drift 14.2 pts |
| Forecast warnings | 6-hour-lead overload warnings **83.9% precision** / 77.4% recall over all origins |
| Alarm quality | precision 87.7%, recall 90.5%, F1 89.1%, Brier skill 0.81 against a 0.097 base rate |
| Alarm burden | 0.33 false alarms per zone-day (145 false-alarm hours across 145 days) |
| Event response | 188/206 overload events alarmed; **stable median lead 3.0 h**, 51.9% warned ≥3 h ahead |
| Replay integrity | 10,341 origins replayed, zero gaps in the hourly series |

Two lead-time numbers are reported on purpose: *stable* lead (an unbroken chain of
warnings from h+1) and *glimpse* lead (the earliest time any horizon saw it). Glimpse lead
reads 24 h here only because the longest forecast reproduces the daily peak — the gates are
scored on the stable number. Read the caveats in the report's `limitations` block before
quoting any figure: capacity is a **p99-of-load proxy**, not rated capacity, and the replay
uses **weather persistence** beyond the observation window.

The JSON artifact is committed, served at `GET /validation`, and rendered as the
**Closed-Loop Validation** panel in the dashboard, so a reviewer sees the evidence behind
each number rather than the number alone.

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
| `CORS_ORIGINS` | FastAPI Cloud environment | Production | Comma-separated exact frontend origins, for example `https://your-project.vercel.app`; unset defaults to open CORS for local/demo use. |

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
├── api/main.py              # FastAPI app — 6 endpoints, model loading
├── config.py                # Zones, paths, thresholds, env-only API key
├── ingestion/               # ENTSO-E, NASA POWER, Kaggle, synthetic bootstrap
├── features/                # Leak-free feature engineering (lags, rolling, cyclic)
├── models/                  # LightGBM training, risk classifier, decision rules,
│                            #   closed-loop validator, LSTM-CNN (designed)
├── data/
│   ├── features/            # training_features.parquet (committed)
│   └── models/              # *.joblib + SHAP JSON + baselines CSV + closed_loop_report.json
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
