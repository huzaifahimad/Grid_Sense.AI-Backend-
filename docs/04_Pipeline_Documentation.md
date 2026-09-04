# GridSense AI — Pipeline Documentation

**Purpose:** Exactly what data flows through this system, what's real, what's synthetic,
and the precise run order to regenerate everything from scratch.

---

## 1. Data Reality Check — read this before trusting any number in this project

| Stage | Current status in the committed repo |
|---|---|
| Raw ENTSO-E load data | NOT present (gitignored, regenerable) — the code to pull it is real and tested, but has not been run with real credentials in the committed artifacts |
| Raw NASA POWER weather data | Same as above |
| `data/features/training_features.parquet` | Present, committed — generated from SYNTHETIC data (see Section 2), not real ENTSO-E |
| `data/models/*.joblib` | Present, committed — trained on the synthetic feature set above |
| `data/models/risk_factors_by_bucket.json` | Present, committed — precomputed SHAP factors from the synthetic-trained model |
| `data/models/baseline_results.csv` | Present, committed — real metrics, but measured on synthetic data |

**What this means practically:** the deployed API right now will return real, working,
mechanically-correct predictions — but they're predictions from a model trained on
synthetic data designed to mimic realistic grid/weather patterns, not real ENTSO-E
history. This is a deliberate, documented bootstrap choice, not an oversight — it let
the entire pipeline (ingestion → features → training → serving → frontend) be built and
verified end to end without waiting on external API approval delays.

**Before treating any accuracy number here as real:** re-run the pipeline with real
ENTSO-E data (Section 3) and regenerate the model artifacts.

---

## 2. Why Synthetic Data Exists At All

`ingestion/synthetic_bootstrap.py` generates schema-correct load and weather data
matching exactly what `entsoe_client.py` and `nasa_power_client.py` produce — same
columns, same types, same join keys — with realistic structure baked in:

- Diurnal pattern (morning/evening demand peaks)
- Weekly pattern (lower weekend demand)
- Seasonal pattern (summer AC-driven peak, approximating what matters for the eventual
  Pakistan use case)
- AR(1)-correlated noise (not pure white noise — real load data isn't that clean)
- Matching weather (temperature/humidity/wind correlated with the same seasonal cycle)

This exists SOLELY to prove the pipeline works mechanically without waiting on external
API access. It is not a modeling shortcut meant to persist — replace it per Section 3.

---

## 3. Regenerating Everything With Real Data

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
```

After this, `data/features/training_features.parquet`, `data/models/*.joblib`, and
`data/models/risk_factors_by_bucket.json` will all reflect REAL data. Commit these files
and redeploy (`fastapi deploy`) to make the live API serve real predictions.

**Do not skip Step 4.** If LightGBM doesn't meaningfully beat persistence and
seasonal-naive baselines on real data, that's a signal to investigate features or data
quality — not something to paper over by moving straight to a fancier model.

---

## 4. Known Bug History (for context on why the code looks the way it does)

### 4.1 Label leakage in the risk classifier (found and fixed)

Original `FEATURE_COLS` for the risk classifier included `load_to_capacity_ratio` — the
exact quantity the proxy label is thresholded on. Result: PR-AUC of 0.9997 and every
zone returning an identical risk score to 13 decimal places, both symptoms of the model
reading the answer off a near-copy of the label rather than learning anything. Fixed by
removing that feature from the classifier's inputs. PR-AUC dropped to a more honest
0.9654 after the fix — confirming the leak was real, not noise.

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
label, not the EIC code. This never surfaced during testing because this sandbox has no
network access to the ENTSO-E API, so it was never actually run end to end. It would
likely have failed outright for `IT_SOUTH` specifically (not a real ISO country code —
entsoe-py has no built-in alias for it), while possibly working by accident for Spain and
Greece (entsoe-py has built-in ISO-code aliases for standard countries, which would have
masked the bug for those two zones only). Fixed to use the actual EIC code. **This is
exactly the kind of bug that only surfaces on first real run — test the real ingestion
carefully when you run it with your token, don't assume a clean run on synthetic data
means the real-data path is bug-free too.**

### 4.5 NASA POWER single large multi-year request (hardened defensively)

`ingestion/nasa_power_client.py` originally requested 2 full years of hourly data in one
HTTP call. NASA POWER's hourly endpoint has practical limits on request size/duration
for large date ranges that aren't precisely documented, and this couldn't be verified
live from this environment (no network access to power.larc.nasa.gov here either).
Rather than risk a real failure on first run, the ingestion now chunks requests by
calendar year — this removes the risk regardless of where the actual undocumented limit
is.

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
