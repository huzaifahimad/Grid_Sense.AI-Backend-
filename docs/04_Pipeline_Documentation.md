# GridSense AI — Pipeline Documentation

**Purpose:** Exactly what data flows through this system, what's real, what's synthetic,
and the precise run order to regenerate everything from scratch.

---

## 1. Data Reality Check — read this before trusting any number in this project

| Stage | Current status in the committed repo |
|---|---|
| Raw ENTSO-E load data | Present locally, gitignored by design: `data/raw/entsoe_load.parquet`, 52,522 hourly rows, 3 zones, 2024-11-04 19:00 → 2026-11-04 17:00 UTC, pulled with a real API token |
| Raw NASA POWER weather data | Same: `data/raw/nasa_power_weather.parquet`, 52,560 hourly rows, 3 zones, 2024-11-04 → 2026-11-03 UTC |
| `data/features/training_features.parquet` | Present, committed — 51,959 rows × 29 columns, every row stamped `load_source = entsoe` and 51,689 stamped `weather_source = nasa_power` (270 `unavailable`). Rebuilt on real data in commit `9c6a48e` |
| `data/models/*.joblib` | Present, committed — trained on the real feature frame above |
| `data/models/risk_factors_by_bucket.json` | Present, committed — precomputed SHAP factors from the real-data-trained model |
| `data/models/baseline_results.csv` | Present, committed — real metrics, measured on real data |

**What this means practically:** the deployed API returns predictions from a model
trained on two years of measured ENTSO-E load and NASA POWER weather. The accuracy
numbers in this repo are therefore real measurements, not pipeline-mechanics proofs.

**Three caveats that survive the switch to real data,** and are the reason "real data" is
not the same as "validated product":

- The overload label is a proxy. `capacity_mw` is a per-zone p99 of observed load, so
  "overload" means unusually high for that zone, not thermally verified.
- Weather is a reanalysis product, roughly two days behind real time, which is why the
  fallback no-weather forecaster exists.
- The three zones are European. Nothing here has been run on Pakistani data.

`ingestion/synthetic_bootstrap.py` is no longer part of the path that produced the
committed artifacts. It remains in the repo as the way to exercise the pipeline on a
machine with no API credentials, and the features layer still refuses to label a
synthetic row as real.

---

## 2. Where the Synthetic Bootstrap Came From

`ingestion/synthetic_bootstrap.py` generates schema-correct load and weather data
matching exactly what `entsoe_client.py` and `nasa_power_client.py` produce — same
columns, same types, same join keys — with realistic structure baked in:

- Diurnal pattern (morning/evening demand peaks)
- Weekly pattern (lower weekend demand)
- Seasonal pattern (summer AC-driven peak, approximating what matters for the eventual
  Pakistan use case)
- AR(1)-correlated noise (not pure white noise — real load data isn't that clean)
- Matching weather (temperature/humidity/wind correlated with the same seasonal cycle)

This existed so the pipeline could be built and debugged before ENTSO-E API approval
came through. It served that purpose and was then displaced: Section 1 documents that
the committed artifacts come from real pulls, not from it. What it still buys you is a
way to run the whole chain on a machine with no credentials — which is why the code
stays, and why nothing synthetic gets a `load_source = entsoe` stamp.

---

## 3. Regenerating Everything From Scratch

Run this locally (not on any deploy platform — training happens on your machine, only
the trained artifacts get deployed):

```bash
cd gridsense-backend
pip install -r requirements.txt --break-system-packages

# Step 1 — weather, no API key needed, do this regardless of ENTSO-E status
python -m ingestion.nasa_power_client

# Step 2 — real load data, needs ENTSOE_API_KEY set in config.py
python -m ingestion.entsoe_client

# Step 3 — join + feature engineering
python -m features.build_features

# Step 4 — baselines FIRST, mandatory. Compare against these before trusting anything fancier.
python -m models.train_baselines

# Step 5 — risk classifier + precomputed SHAP factors
python -m models.risk_classifier

# Step 6 — (optional, heavier) LSTM+CNN quantile forecaster
python -m models.train_lstm_cnn

# Step 7 — closed-loop replay of the held-out test window through the serving path.
# Refuses to certify the models if any release gate drops below its floor (--strict).
python -m models.closed_loop_validation --strict
```

After this, `data/features/training_features.parquet`, `data/models/*.joblib`,
`data/models/risk_factors_by_bucket.json`, and `data/models/closed_loop_report.json` will
all reflect REAL data. Commit these files
and redeploy (`fastapi deploy`) to make the live API serve real predictions.

**Do not skip Step 4.** If LightGBM doesn't meaningfully beat persistence and
seasonal-naive baselines on real data, that's a signal to investigate features or data
quality — not something to paper over by moving straight to a fancier model.

---

## 4. Known Bug History (for context on why the code looks the way it does)

### 4.1 Label leakage in the risk classifier (found and fixed)

Original `FEATURE_COLS` for the risk classifier included `load_to_capacity_ratio` — the
exact quantity the proxy label is thresholded on. Result: near-perfect PR-AUC and every
zone returning an identical risk score to 13 decimal places, both symptoms of the model
reading the answer off a near-copy of the label rather than learning anything. Fixed by
removing that feature from the classifier's inputs. Post-fix, the synthetic bootstrap
run reported a more honest PR-AUC of 0.9595; after the rebuild on two years of real
ENTSO-E and NASA POWER data, the same code measures **0.9268** on the 10,122-row test
split (Brier 0.0309 against a 0.1105 base-rate reference).

**If you retrain from scratch and see PR-AUC suspiciously close to 1.0 again, or
identical scores across zones, this is the first thing to check.**

### 4.2 Dead `torch` import breaking API startup (found and fixed)

`api/main.py` originally imported `torch` at the top level despite never using it — the
LSTM model was never wired into serving, only LightGBM. This unused import crashed
server startup when torch wasn't fully installed. Fixed by removing the dead import.

### 4.3 SHAP dependency weight (found via Vercel deployment attempt, fixed)

Covered fully in System Architecture doc Section 3 and ML Model Design doc Section 2.5.

### 4.4 ENTSO-E client discarding the zone code it needed (found via code review, fixed)

`ingestion/entsoe_client.py` accepted both a bidding-zone EIC code (e.g.
`"10Y1001A1001A788"` for Italy South) and an internal label (e.g. `"IT_SOUTH"`) as
parameters, but called `client.query_load(country_code, ...)` — using the internal
label, not the EIC code. This never surfaced earlier because the pipeline was still
running on the synthetic bootstrap at the time, so the real ingestion path had never
been exercised. It would
likely have failed outright for `IT_SOUTH` specifically (not a real ISO country code —
entsoe-py has no built-in alias for it), while possibly working by accident for Spain and
Greece (entsoe-py has built-in ISO-code aliases for standard countries, which would have
masked the bug for those two zones only). Fixed to use the actual EIC code, and the fix
has since been exercised for real: the committed `data/raw/entsoe_load.parquet` is
52,522 rows pulled through this code path across all three zones, `IT_SOUTH` included.
**This is
exactly the kind of bug that only surfaces on first real run — test the real ingestion
carefully when you run it with your token, don't assume a clean run on synthetic data
means the real-data path is bug-free too.**

### 4.5 NASA POWER single large multi-year request (hardened defensively)

`ingestion/nasa_power_client.py` originally requested 2 full years of hourly data in one
HTTP call. NASA POWER's hourly endpoint has practical limits on request size/duration
for large date ranges that aren't precisely documented, and at the time this was written
neither the limit nor the behaviour could be checked from a machine without access to
power.larc.nasa.gov.
Rather than risk a real failure on first run, the ingestion now chunks requests by
calendar year — this removes the risk regardless of where the actual undocumented limit
is. The chunked path has since been run against the live endpoint and produced the
committed `data/raw/nasa_power_weather.parquet` (52,560 rows, 3 zones), so the fix is
verified as working; where the unchunked request would actually have broken is still
undocumented.

### 4.6 Unhandled forecast horizon input (found via edge-case testing, fixed)

`GET /forecast/{zone}?horizon_hours=0` crashed with an unhandled 500 error and exposed a
raw Python stack trace to the client. `horizon_hours=9999` silently capped output with no
indication to the caller. Fixed with proper FastAPI `Query(ge=1, le=48)` validation —
both cases now return a clean 422 with a clear message.

### 4.7 `/shed-schedule` endpoint built but never called by the frontend (found via integration check, fixed)

A fully working backend endpoint — one of the TRD's core deliverables (the "recommended
load-shedding schedule") — had zero references anywhere in the frontend code. Wired in
properly: fetch function, state, mock-mode equivalent, and a new dashboard section
showing the priority and reasoning for each recommendation.

### 4.8 API memory footprint close to FastAPI Cloud's Hobby tier limit (found, partially mitigated, NOT fully resolved)

Measured peak RAM usage at ~410-412MB against FastAPI Cloud Hobby's 512MB hard limit —
under light single-request local testing only, not concurrent load or real (likely
larger) production data. Root cause isolated precisely: importing `pandas` alone costs
~88MB, `lightgbm` another ~80MB, before any data or model is loaded — the dependency
libraries themselves are the dominant cost, not this project's dataset size. A dataframe-
trimming optimization was applied (keep only the last 200 rows per zone in memory instead
of full history) but only moved the measurement by ~2MB, confirming it wasn't the actual
bottleneck. **This is a real, open risk, not fully solved** — see System Architecture doc
Section 7 for the full write-up and next-step options if it causes problems after real
deployment (monitor FastAPI Cloud's built-in memory metrics; if the service gets
OOM-killed under real traffic, consider Render as a higher-memory-ceiling fallback, or a
larger rewrite to reduce the pandas/lightgbm import footprint itself).

---

## 5. Data Governance Notes

- `.gitignore` excludes raw ingestion data (`data/raw/*.parquet`) — bulky, regenerable,
  should never be committed.
- Feature and model artifacts ARE committed intentionally (see System Architecture doc
  Section 4 for why, and the trade-off this implies).
- `config.py`'s `ENTSOE_API_KEY` must remain the placeholder string in every commit.
  Real tokens go into FastAPI Cloud's encrypted secret store
  (`fastapi cloud env set --secret ENTSOE_API_KEY "..."`) or Render's environment
  variable dashboard if using the fallback — never into a committed file.
