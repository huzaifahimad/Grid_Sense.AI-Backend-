# PROMPT FOR ANTIGRAVITY — Testing & Validation Pass

You are doing final QA on GridSense AI before it's considered complete. Act as a senior
engineer reviewing a colleague's work before sign-off — verify everything by running it,
never by reading code and assuming it works. This project has TWO repositories:
`gridsense-backend` (Python/FastAPI, deploys to FastAPI Cloud) and `gridsense-frontend`
(React/Vite, deploys to Vercel). Full context is in each repo's `docs/` folder
(specifically `06_Project_Overview.md` for orientation and `04_Pipeline_Documentation.md`
for what's real vs. synthetic in the current state).

## Phase 1 — Audit current state, report before changing anything

1. Read `docs/06_Project_Overview.md` and `docs/04_Pipeline_Documentation.md` in the
   backend repo fully before touching code.
2. Check what's actually present: `data/raw/` (should be empty/gitignored),
   `data/features/training_features.parquet` (should exist, trained on synthetic data
   per the docs — confirm this claim is true, don't just trust the doc), `data/models/`
   (should have `lightgbm_load_forecast.joblib`, `risk_classifier.joblib`,
   `risk_factors_by_bucket.json`, `baseline_results.csv`).
3. Check `config.py` — confirm `ENTSOE_API_KEY` is the placeholder string, not a real
   token. If you find a real token in ANY committed file in either repo, stop and flag
   this immediately as a security issue before doing anything else.
4. Report back what you found, in the same real-vs-claimed format as the docs use —
   don't just say "looks fine."

## Phase 2 — Verify the backend runs correctly, locally, first

1. `cd gridsense-backend && pip install -r requirements.txt --break-system-packages`
2. Confirm the model files load without error:
   ```
   python -c "import joblib; joblib.load('data/models/lightgbm_load_forecast.joblib'); joblib.load('data/models/risk_classifier.joblib'); print('models load OK')"
   ```
3. Start the API: `fastapi dev` (or `uvicorn api.main:app --port 8000` as fallback).
   Confirm it starts without error and confirm the startup log shows all four expected
   loads: forecast model, risk classifier, precomputed factors, feature data.
4. Hit every endpoint with curl, read the actual JSON response body, don't just check
   status codes:
   - `GET /`
   - `GET /assets` — **specifically check whether the three zones return DIFFERENT risk
     scores.** If they're identical or near-identical, this is the label-leakage bug
     documented in `docs/04_Pipeline_Documentation.md` Section 4.1 — it should be fixed,
     but verify it actually is, don't assume the docs are still accurate.
   - `GET /risk/ES` (and other zones) — confirm `top_factors` is populated with real
     feature names and non-zero contributions, not an empty list.
   - `GET /forecast/ES?horizon_hours=24` — confirm 24 real data points with P10/P50/P90.
   - `GET /shed-schedule/ES`
5. Confirm the API does NOT import `shap`, `sklearn`, or `torch` at the top level of
   `api/main.py` — grep for these imports. If any are present, that's a regression from
   the documented lean-serving design (`docs/02_ML_Model_Design.md` Section 2.5) and
   needs to be fixed before deploying, since it will break FastAPI Cloud/Vercel size
   constraints.

## Phase 3 — Verify the frontend builds and connects correctly

1. `cd gridsense-frontend && npm install && npm run build`. Confirm zero errors. This
   was tested and confirmed working — if it fails now, something changed, investigate
   what.
2. Check `src/GridSenseDashboard.jsx` for `import.meta.env.VITE_API_BASE_URL` and
   `import.meta.env.VITE_USE_MOCK_DATA` — confirm these are still present and read
   correctly (Vite's env var convention, not `process.env`).
3. With the backend running locally from Phase 2, run `npm run dev`, open it, and set
   `VITE_USE_MOCK_DATA=false` with `VITE_API_BASE_URL=http://localhost:8000` in a local
   `.env.local` file to confirm the frontend can actually pull real data from the local
   backend before testing the live deployed version.
4. Confirm in the browser: no "API ERROR" banner, all three zones show different risk
   scores when clicked, the 3D scene renders without console errors, the forecast chart
   populates with real data.

## Phase 4 — Verify the LIVE deployed versions, not just local

1. Curl the live FastAPI Cloud URL's `/assets` and `/risk/{zone}` endpoints. Confirm
   real, differentiated data — same checks as Phase 2 step 4, but against the actual
   deployed instance, not localhost.
2. Open the live Vercel URL in a browser. Confirm `VITE_API_BASE_URL` in Vercel's
   environment variables actually matches the live FastAPI Cloud URL exactly (check for
   typos, trailing slashes, http vs https).
3. Confirm the live dashboard shows real data with no error banner, and that this
   matches what Phase 3's local test showed.

## Phase 5 — Documentation accuracy check

For each of these specific claims made in the docs, verify it's still true by checking
the actual code, not by trusting the document:

- TRD Section 4.1: "PR-AUC 0.9654 after fixing leak" — does `models/risk_classifier.py`
  still exclude `load_to_capacity_ratio` from `FEATURE_COLS`? Confirm by reading the
  actual list in the file.
- System Architecture Section 2.2: "282MB minimum for Vercel serverless, over the
  250MB limit" — this doesn't need re-testing (it's a settled architectural conclusion),
  but confirm no one has since tried to reintroduce Python backend code into the
  frontend repo (check for `requirements.txt`, `api/`, or `ml_models/` folders existing
  in `gridsense-frontend` — there should be NONE).
- Pipeline Documentation Section 1: confirm the data really is synthetic-trained, not
  real ENTSO-E, by checking whether `data/raw/` has ever had real ENTSO-E files
  (check git log for evidence, or just confirm current `data/raw/` is empty/gitignored
  and no real ingestion has been recorded as run).

## Phase 6 — Final report

Give a structured report:

- **Backend:** does it run locally, does it deploy, do all 5 endpoints return correct
  real data, is the leak fix still in place, are shap/sklearn/torch still absent from
  the live serving path?
- **Frontend:** does it build, does it connect to the real backend, does the dashboard
  render correctly with real differentiated data across zones?
- **Documentation accuracy:** for each claim checked in Phase 5, is it still true?
- **Anything found broken or drifted from what the docs describe** — call this out
  explicitly, don't smooth it over.
- **What you could NOT verify** (e.g. no access to the live deployed URLs, no
  FastAPI Cloud CLI auth available) — say so plainly rather than assuming success.

Do not conclude "everything works" unless you have direct evidence for every claim in
that sentence. A confident-sounding summary without evidence is worse than an honest
"I couldn't verify X" — say what you actually checked and what you didn't.
