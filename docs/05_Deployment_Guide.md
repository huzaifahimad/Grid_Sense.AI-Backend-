# GridSense AI — Deployment Guide

**Target architecture:** backend on FastAPI Cloud, frontend on Vercel, two separate
GitHub repositories. This is the tested, working path — see System Architecture doc
Section 2 for why the alternatives (Vercel-only, Render) were ruled out or de-prioritized.

---

## Before you push anything to GitHub — non-negotiable check

Open `gridsense-backend/config.py`. Confirm:
```python
ENTSOE_API_KEY = os.environ.get("ENTSOE_API_KEY") or None
```
There must be **no literal token string** in this file — the key is read only from
the environment. Your real token goes into FastAPI Cloud's secret store (Step 5
below), or Render's / Vercel's env-var panel. Never commit it.

---

## Step 1 — Create two separate GitHub repositories

Yes, two repos, not one. This matches the tested architecture (System Architecture doc
Section 2.1) — the backend is Python/FastAPI, the frontend is a separate React/Vite app,
and they deploy to entirely different platforms with different build systems.

```bash
# Repo 1: backend
cd gridsense-backend
git init
git add .
git commit -m "GridSense AI backend — FastAPI + LightGBM, FastAPI Cloud deploy target"
git branch -M main
git remote add origin https://github.com/<you>/gridsense-backend.git
git push -u origin main

# Repo 2: frontend
cd ../gridsense-frontend
git init
git add .
git commit -m "GridSense AI frontend — React + Three.js dashboard"
git branch -M main
git remote add origin https://github.com/<you>/gridsense-frontend.git
git push -u origin main
```

---

## Step 2 — Train models locally with real data BEFORE deploying

The deployed backend serves from committed model files — it does not train on deploy.
Follow `docs/04_Pipeline_Documentation.md` Section 3 in full, then:

```bash
cd gridsense-backend
git add data/features/training_features.parquet data/models/
git commit -m "Add models trained on real ENTSO-E data"
git push
```

---

## Step 3 — Install the FastAPI CLI

```bash
pip install "fastapi[standard]"
```

---

## Step 4 — Verify locally before deploying

```bash
cd gridsense-backend
fastapi dev
```
This must succeed and find `api/main.py`'s `app` object via the `pyproject.toml`
entrypoint config (`api.main:app`) without you passing a file path. If it fails, fix the
entrypoint config before proceeding — do not deploy something that doesn't run locally
first.

Test it while it's running:
```bash
curl http://localhost:8000/assets
curl http://localhost:8000/risk/ES
```
Confirm real JSON, with `top_factors` populated and different zones showing different
risk scores.

---

## Step 5 — Deploy the backend to FastAPI Cloud

```bash
fastapi login      # opens a browser for auth, no card required
fastapi deploy
```

Note the live URL — format: `https://<appname>.fastapicloud.dev`

Set your real ENTSO-E token as an encrypted secret (never in a committed file):
```bash
fastapi cloud env set --secret ENTSOE_API_KEY "your-real-token-here"
```

Verify the LIVE deployment, not just a successful deploy message:
```bash
curl https://<appname>.fastapicloud.dev/assets
curl https://<appname>.fastapicloud.dev/risk/ES
```

---

## Step 6 — Deploy the frontend to Vercel

1. vercel.com → New Project → import `gridsense-frontend` repo.
2. Framework preset: Vite (should auto-detect). Build command / output directory
   should auto-fill from `vercel.json` (`npm run build` / `dist`).
3. Before deploying, set Environment Variables:
   - `VITE_API_BASE_URL` = your FastAPI Cloud URL from Step 5
   - `VITE_USE_MOCK_DATA` = `false`
4. Deploy.

---

## Step 7 — Verify the full chain, live

Open your Vercel URL in a browser.

- No red "API ERROR" banner should appear.
- Click through all three zones — risk scores and forecasts should genuinely differ
  per zone (confirms the earlier data-leak bug, see Pipeline Documentation Section 4.1,
  is still fixed).
- The "Top Contributing Factors" panel should show real feature names
  (`load_lag_1h`, `temperature_c`, etc.) with real numeric contributions — these come
  from the precomputed bucket lookup, not a live SHAP call (Pipeline Documentation
  Section 4.3 / System Architecture Section 3).

If the API error banner appears, check in this order:
1. Is `VITE_API_BASE_URL` in Vercel's env vars an exact match to your FastAPI Cloud URL
   (including `https://`, no trailing slash)?
2. Did you redeploy on Vercel AFTER setting the env vars? (Vite bakes env vars in at
   build time — changing them requires a redeploy, they aren't read at runtime.)
3. Open browser DevTools → Network tab, refresh, find the failed request, check its
   actual status code and response.

---

## Known limitations after this deploy (carried from System Architecture doc)

- Predictions are served from a static feature snapshot committed at deploy time — they
  will not reflect new real-world data until you re-run the pipeline locally and push
  again. Not a live database.
- CORS is currently open (`allow_origins=["*"]` in `api/main.py`) — tighten to your real
  Vercel domain before treating this as beyond a pilot/demo.
- LSTM+CNN model was never trained/deployed — only LightGBM baseline and risk classifier
  are live, consistent with the rule that a simpler model ships unless a complex one
  proves it's earning its added cost.
- SHAP explanations are precomputed/bucketed, not per-request-exact (see Pipeline
  Documentation Section 4.3).
