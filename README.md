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
leak-free time-series features, and serves three trained models through a REST API:

- **Load Forecaster** (LightGBM regressor) — recursive ahead-of-origin demand forecast from
  the latest observation, served for 1-48 h (`MAX_HORIZON_HOURS`). P10/P90 are
  **split-conformal intervals** calibrated per horizon on a held-out validation window
  (closed-loop replay coverage **84.92%** against 80% nominal). Measured on the real test
  window: MAPE **2.10%** overall for the 16-feature model, versus **9.20%** seasonal-naive
  and **7.81%** persistence. Per horizon, replayed through the serving path: **2.12%** at
  h+1, **4.99%** at h+6, **5.45%** at h+12, **5.71%** at h+24. Future weather uses
  persistence (same hour, previous day) until an NWP feed is connected.
- **No-weather fallback forecaster** (LightGBM regressor, 13 features) — the same recursion
  without the three weather inputs, with its own separately calibrated bands. NASA POWER
  publishes an hour roughly two days late, so the freshest ENTSO-E load hours have no
  weather row; the weather-aware model is not fed invented inputs, and the API answers those
  origins from this model instead and says which one it used in `served_by`. Test MAPE
  **2.12%**, within 0.02 pp of the weather-aware model.
- **Overload Risk Classifier** (LightGBM classifier) — per-zone risk score with
  precomputed SHAP explanations. Test PR-AUC **0.9268**, Brier **0.0309** against a base-rate
  reference of **0.1105** (train positive rate 4.76%). Early stopping uses a validation
  slice, never the test set.
- **Load-Shedding Recommender** (rules-based) — priority-ranked shedding schedule with
  human-readable reasoning. The system *recommends*; it never controls anything.

Zones covered: **Spain (ES)** · **Greece (GR)** · **Southern Italy (IT_SOUTH)**. These are
European bidding zones used as a working proxy. **No Pakistani grid data has been used.**
The architecture is built so it can be retrained on NTDC/DISCO data once access is secured;
nothing in the current accuracy figures transfers to the Pakistani grid without that
retraining and a fresh validation run.

> **Honesty note:** the committed artifacts are trained on **real** hourly data from the
> ENTSO-E Transparency Platform (load) and NASA POWER (weather), 2024-11 through 2026-11,
> `/health` reports `is_synthetic: false` with per-source row counts. Three things are
> still proxies and are flagged as such everywhere: capacity is a **p99-of-load estimate**,
> not rated thermal capacity, so "overload" means unusually high for that zone; the labels
> derive from the same load series the models consume; and future weather is persistence,
> not an NWP feed. `GET /health → data_provenance` therefore reports one horizon per feed
> (`load_latest_timestamp`, `weather_latest_timestamp`, `weather_lag_hours`, `capacity`)
> rather than a single "last updated" — reanalysis reaches its newest hour roughly four
> days after load does, and at the time of writing it was 90 h behind, which is what pushes
> `/forecast` onto the no-weather fallback and `/risk` onto the newest hour every input
> covers. The closed-loop release gates currently grade **WARN**, 9 of 13 passing
> and 4 short (band coverage drift, event detection, 3 h warning share, median stable lead).
> See [`docs/04_Pipeline_Documentation.md`](docs/04_Pipeline_Documentation.md) and
> `GET /validation` before citing any accuracy number.

**Tests:** `python -m pytest tests -q` — 94 tests covering endpoint contracts, train/serve
feature parity, label-leak guard, interval calibration, the lean serving import path, the
request-time refresh path (credential redaction, the egress budget, and the fallback that keeps
the committed frame serving when a fetch fails), and the closed-loop validator (gate grading,
lead-time math, and negative controls that prove the gates fail when they should).

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
| `/risk/{zone}` | GET | Overload risk score, status, top SHAP factors, and the load vs. capacity of the scored snapshot |
| `/assets` | GET | Risk-ranked list of all monitored assets across zones |
| `/shed-schedule/{zone}` | GET | Advisory shedding recommendation — `[]` unless that zone's score clears 0.6, so most zones return an empty list (priority 1 = critical ≥0.8, 2 = elevated ≥0.6) |
| `/validation` | GET | Closed-loop release-gate report — 13 gates, walk-forward replay metrics, provenance and stated limitations, plus report age in hours |

Example:

```bash
curl https://<your-app>.fastapicloud.dev/risk/IT_SOUTH
```

```json
{
  "zone": "ES",
  "asset_id": "ES-FEEDER-01",
  "risk_score": 0.0016,
  "status": "normal",
  "snapshot_timestamp": "2026-10-29T18:00:00+00:00",
  "snapshot_load_mw": 32193.0,
  "capacity_mw": 37215.17,
  "top_factors": [
    {"feature": "hour_cos", "contribution": -0.153},
    {"feature": "load_lag_1h", "contribution": -0.4868},
    {"feature": "temperature_c", "contribution": 0.4009}
  ],
  "is_proxy_capacity": true
}
```

That reading is low because ES sits at 86% of its own p99 proxy, under the 0.9 overload
ratio — a genuine "normal", not an inert model. `scripts/verify_prediction_integrity.py`
re-derives every served score from the committed artifacts, and perturbing the load by
+110% moves this zone's score from 0.0003 to 0.66.

---

## Closed-Loop Validation

Accuracy claims are checked by **replaying the held-out test window through the same code
path the API serves from** — recursive forecast, calibrated intervals, risk classifier,
shedding rules — not by reading a training-time score.

```bash
python -m models.closed_loop_validation          # writes data/models/closed_loop_report.json
python -m models.closed_loop_validation --strict # exits 1 if any gate is WARN/FAIL
```

The replay covers 3,464 hours (10,122 scored rows, 10,050 hourly forecast origins at a 1-hour
stride, 24-hour horizon) across **ES · GR · IT_SOUTH**, and grades 13 gates against a target
and a hard floor. Current report: **overall WARN — 9 PASS, 4 WARN**, the four short gates being
band-coverage drift and the three event lead-time gates. It is reported as WARN rather than
having the targets lowered.

| Gate family | Measured | Grade |
|---|---|---|
| Forecast skill | h+1 MAPE 2.12% vs 9.27% seasonal-naive (**+77.2%**); h+24 5.71% vs 9.27% (**+38.4%**) | PASS |
| Interval honesty | P10–P90 coverage 84.92% (target 70–95%); week-to-week spread 19.5 pts (target ≤15) | PASS / **WARN** |
| Forecast warnings | p50-over-line at h+6: **76.4% precision**, 79.5% recall, 3.18 alerts per zone-day, over all 10,050 origins | PASS |
| Alarm quality | precision 88.38%, recall 78.44%, F1 83.11%, Brier skill 0.72 against a 0.1105 base rate | PASS |
| Alarm burden | 0.313 false alarms per zone-day (132 false-alarm hours, 1,004 true-alarm hours) | PASS |
| Event response | 142/181 overload events alarmed (**78.45%**, target 80%); **median stable lead 1.0 h** (target 3.0); 34.3% warned ≥3 h ahead (target 50%) | **WARN** ×3 |
| Replay integrity | 10,050 origins replayed, 10,122 rows scored; 2 hourly gaps in ES, none in GR or IT_SOUTH | PASS |

Per zone, because the aggregate hides a real difference: ES precision 92.4% / recall 96.7%,
GR 86.0% / 93.2%, IT_SOUTH 81.3% / **45.6%**. Southern Italy is the weakest zone and the
recall gap there is the main reason the event gates miss their target.

Two lead-time numbers are reported on purpose: *stable* lead (an unbroken chain of warnings
from h+1) and *glimpse* lead (the earliest time any horizon saw it). On real data the medians
are **1.0 h stable** and **5.0 h glimpse**. Scoring the upper band instead of the point
forecast would lift the median stable lead to 24 h and the ≥3 h warned share to 93.4%, but it
costs 76.4% → 40.2% precision at h+6 and 3.18 → 7.58 alerts per zone-day, which is the alert
fatigue the burden gate exists to prevent. A second-stage classifier was trained to get the
early warnings without that cost; on the test window it scored precision 0.602 / recall 0.622
against the incumbent rule's 0.777 / 0.804 at matched alert volume, so it is **not shipped**
and the lead-time gates stay WARN. That is a measured negative, kept in
`data/models/exceedance_metrics.json`, not a suppressed one. Read the caveats in the report's
`limitations` block before quoting any figure: capacity is a **p99-of-load proxy**, not rated
capacity, and the replay uses **weather persistence** beyond the observation window.

The JSON artifact is committed, served at `GET /validation`, and rendered as the
**Closed-Loop Validation** panel in the dashboard, so a reviewer sees the evidence behind
each number rather than the number alone.

Every number above is also re-derived by an independent script that does not read the report:

```bash
python scripts/verify_model_accuracy.py            # 22 checks, incl. a live API comparison
python scripts/verify_model_accuracy.py --skip-live # 10 checks, no server needed
```

It recomputes precision, recall, PR-AUC and Brier skill from `data/features/training_features.parquet`
and the committed classifier, then compares them digit-for-digit with what the deployed API
returns. It also asserts the properties a leakage bug would break: label agreement 1.000000 across
51,959 rows, no label-shaped column among the feature inputs, a strictly temporal split, and a
deployed `risk_score` reproduced to `|delta| = 0` for all three zones. A shuffle-before-split probe
is run and reported rather than assumed safe — see [`docs/15_Model_Accuracy_Verification.md`](docs/15_Model_Accuracy_Verification.md)
for what it measures, what it proves weak, and the model comparison it backs.

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
| `ENTSOE_API_KEY` | FastAPI Cloud secret store / local shell | Yes for live data; optional for predictions | ENTSO-E Transparency Platform API token. **Never commit it** — `config.py` reads it from the environment only. Since the request-time refresh landed, this key is used by the **serving** process too, not only by offline ingestion: `/health` and the forecast endpoints kick a background pull that extends the served frame. Without it the API still answers, from the committed frame. |
| `CORS_ORIGINS` | FastAPI Cloud environment | Optional | Comma-separated exact frontend origins, for example `https://your-project.vercel.app`. Unset does **not** mean open CORS: `api/main.py` falls back to the single hardcoded origin `https://grid-senseai.vercel.app`. Measured 2026-11-06 against both live and local with the variable unset — that origin is echoed in `access-control-allow-origin`, `https://evil.example.com` gets no CORS header at all, and `POST /health` returns 405 because only `GET` is allowed. No credentials are permitted. |

### Request-time refresh knobs

Every one of these is read **once, at import time** (`api/live_refresh.py`
evaluates `RefreshConfig.from_env()` at module level), so changing a value means
restarting the process — on FastAPI Cloud, a redeploy. Defaults are the values
the deployed app runs with. `/health` → `data_refresh` reports the live effective
values, so you can confirm what a deployment actually picked up instead of
inferring it from the env.

| Variable | Default | Purpose |
|---|---|---|
| `GRIDSENSE_REFRESH` | `true` | Master switch. `false` serves the committed frame only — no outbound ENTSO-E or weather calls at all. |
| `GRIDSENSE_REFRESH_STALE_HOURS` | `3.0` | A zone is only eligible for a pull once its newest observed hour is older than this (`_zones_due`). ENTSO-E publishes actual load behind wall clock, so a smaller number buys no fresher data and costs requests for nothing. Independent of the `cache_seconds` gate, which is process-wide. |
| `GRIDSENSE_REFRESH_CACHE_SECONDS` | `900` | Minimum seconds between refresh attempts. The gate is process-wide — `kick_if_due` compares `last_attempt_at` against this before checking which zones are due — so a busy page cannot hammer the SOP endpoint. Kept ≥60 deliberately; `/health` asserts it. |
| `GRIDSENSE_REFRESH_QUIET_SECONDS` | `3600` | Backoff after an attempt that added zero rows — usually means the upstream simply has not published yet, and retrying every 15 minutes would only burn requests. |
| `GRIDSENSE_REFRESH_MAX_TAIL_HOURS` | `336` | Cap on how many hours one refresh will try to append (14 days), so a large gap is closed in bounded work rather than in one unbounded call. |
| `GRIDSENSE_REFRESH_TIMEOUT_SECONDS` | `30.0` | Per-request socket timeout for the refresh path. It cannot inherit entsoe-py's defaults: 3 retries × 10 s would hold a thread ~2 min for nothing, since the fallback is the committed frame. The refresher makes **one** attempt with **zero** delay between attempts. 12 s was measured to be **below the endpoint's own response floor** — two direct probes of `web-api.tp.entsoe.eu` (2026-11-06T11:56:44Z, invalid token, so the server short-circuits before building the document) returned first byte at **15.1 s** and **5.4 s** — and every zone in the deployed process consequently died with `ReadTimeout` on ~42 h requests. 30 s leaves roughly 2× the worst observed TTFB. `/health` now publishes `timeout_seconds`, `last_attempt_seconds` and per-zone `load_seconds`, so the next sizing is read off the wall rather than guessed. |
| `GRIDSENSE_REFRESH_KEEP_ROWS` | `500` | Rows kept per zone after a merge. Deliberately above the 336 the API loads at startup, so a refresh never drops history another endpoint relies on, while still bounding memory on a long-lived process. Trims the in-memory frame only — committed Parquet is never rewritten. |
| `GRIDSENSE_REFRESH_WEATHER` | `true` | Whether the refresh also pulls NASA POWER weather. No key required. |
| `GRIDSENSE_REFRESH_WEATHER_MAX_DAYS` | `4.0` | Calendar days of weather per pull (the window is day-bounded). Reanalysis publishes roughly two days behind, so this usually cannot fill the blind tail — the no-weather fallback model is the shipped answer for that gap; the pull does recover hours that were simply never fetched. |

Numeric knobs are validated on read: a non-numeric or non-positive value logs a
warning and falls back to the default rather than disabling the gate or
spinning the refresher.

Set it on FastAPI Cloud:

```bash
fastapi cloud env set --secret ENTSOE_API_KEY "your-token"
```

With the key **unset**, the serving API still works — predictions come from
committed model artifacts and the committed data frame, and every endpoint keeps
answering — but the served frame stops advancing and `/health` reports the refresh
as disabled/failed rather than silently serving ever-older data. With the key
**set**, the serving process performs the outbound ENTSO-E pull itself, in a
background thread that never blocks a request; a failed pull falls back to the
committed frame and never publishes a partial one.

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
├── api/main.py              # FastAPI app — 7 endpoints, model loading
├── config.py                # Zones, paths, thresholds, env-only API key
├── ingestion/               # ENTSO-E, NASA POWER, Kaggle, synthetic bootstrap
├── features/                # Leak-free feature engineering (lags, rolling, cyclic)
├── models/                  # LightGBM training, risk classifier, decision rules,
│                            #   closed-loop validator, LSTM-CNN (designed)
├── data/
│   ├── features/            # training_features.parquet (committed)
│   └── models/              # *.joblib + SHAP JSON + baselines CSV + closed_loop_report.json
├── docs/                    # 15-document suite (TRD → presentation source)
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
| [`13_Closed_Loop_Validation_Update.md`](docs/13_Closed_Loop_Validation_Update.md) | Closed-loop replay method, gate results, and the lead-time correction |
| [`14_Security_Posture.md`](docs/14_Security_Posture.md) | Measured attack surface, response-header and CORS evidence, dependency audit results, accepted risks |
| [`15_Model_Accuracy_Verification.md`](docs/15_Model_Accuracy_Verification.md) | Independent re-computation of every accuracy claim, leakage probes, model comparison, and what remains unproven |

## Related

Frontend dashboard (React 18 + Three.js):
**[Grid_Sense.AI-frontend-](https://github.com/huzaifahimad/Grid_Sense.AI-frontend-)**

## License

MIT — free to use, modify, and distribute. ENTSO-E data usage is subject to the
[ENTSO-E Transparency Platform terms](https://transparency.entsoe.eu/).
