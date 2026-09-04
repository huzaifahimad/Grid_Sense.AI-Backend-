# GridSense AI — Project Overview

**Read this first if you're new to the project, evaluating it, or returning to it after
time away.**

---

## What this is, in one paragraph

GridSense AI predicts electricity grid overload risk before it happens, using machine
learning trained on real grid load data (European transmission operators, via ENTSO-E)
and weather data (NASA POWER), with the architecture designed to retrain on Pakistani
NTDC/DISCO grid data once institutional access to that data is secured. It outputs a
risk score per grid zone, a short-term load forecast with uncertainty bands, and a
rules-based (not AI-generated) recommended load-shedding schedule — the system
recommends, it never autonomously controls anything.

## Why this exists

Grid overload prediction is a genuine, high-stakes engineering problem, especially in
countries where load-shedding is a routine operational reality rather than an emergency
exception. This project is a serious technical portfolio piece demonstrating an
end-to-end ML system — real data pipelines, honest baseline comparisons, found-and-fixed
bugs, tested deployment — built with the explicit goal of eventually working with real
Pakistani utility data, not just a hackathon demo.

## The two things anyone evaluating this should understand immediately

**1. It's trained on synthetic data right now, and that's documented, not hidden.**
The full pipeline — data ingestion, feature engineering, model training, API serving,
frontend — was built and verified end to end using realistic synthetic data as a
bootstrap, because real ENTSO-E API access has an approval delay. The code to pull real
data is complete and tested; running it just requires the person's own credentials. See
`docs/04_Pipeline_Documentation.md` for the exact breakdown of what's real vs. synthetic
right now.

**2. Real bugs were found and fixed by actually running the system, not just writing it.**
Most notably: the risk classifier initially had a label-leakage bug (a feature that was
near-identical to the target it was predicting), producing a suspiciously perfect
PR-AUC of 0.9997 and identical risk scores across every grid zone. This was caught by
live-testing the deployed API and noticing the identical scores, traced to the leak, and
fixed — dropping PR-AUC to an honest 0.9654. That's the kind of finding that only
surfaces from actually running a system, and it's documented rather than swept aside.

## What the system actually does, end to end

1. **Ingests** hourly grid load data and matching weather data for several zones.
2. **Engineers features**: lag values, rolling statistics, cyclic time encodings, and a
   proxy "overload" label based on load relative to estimated capacity.
3. **Trains two separate models**: a LightGBM regressor forecasting load 1–48 hours
   ahead with uncertainty bands, and a LightGBM classifier predicting overload risk,
   deliberately kept as two models because they're genuinely different problems.
4. **Precomputes explainability**: instead of an expensive live SHAP computation on
   every API request, feature-contribution explanations are computed once at training
   time and bucketed by risk level.
5. **Serves predictions** via a REST API (FastAPI), deployed as a persistent process
   (not a size-constrained serverless function — that distinction mattered, see below).
6. **Displays results** in a 3D dashboard (React + Three.js) showing a live grid
   topology visualization, ranked risk list, forecast chart with confidence bands, and
   the reasoning behind any shed-schedule recommendation.

## Why the deployment looks the way it does

Two separate GitHub repositories, two separate hosting platforms — this wasn't the
starting design, it's what testing led to:

- An attempt to run everything on Vercel (frontend platform) failed for a real,
  measured reason: the backend's ML dependencies (specifically `lightgbm`, which
  mandatorily requires `scipy`) don't fit within Vercel's serverless function size
  limit, even at the absolute minimum possible dependency set. This was tested directly,
  not assumed.
- The backend needs a platform that runs a persistent process (to keep trained models
  loaded in memory), which is a different kind of hosting than a frontend needs.
- FastAPI Cloud was chosen for the backend specifically because it required no credit
  card (a real constraint) and is purpose-built for exactly this kind of API.

Full reasoning: `docs/03_System_Architecture.md`.

## What's genuinely NOT done yet — the honest gap list

- Real Pakistani NTDC/DISCO grid data — not yet integrated, requires institutional
  access still being pursued.
- Real fault/outage event logs — the risk classifier currently uses a proxy label
  (load exceeding 90% of estimated capacity) rather than genuine historical outage
  records.
- A live database — predictions are currently served from a static file snapshot
  committed to the repo, not continuously updating data.
- The LSTM+CNN deep learning model — designed, but not trained or deployed, because the
  simpler LightGBM model already cleared the accuracy baselines by a wide margin and
  didn't need the added complexity justified.
- Live, per-request SHAP explanations — currently precomputed and bucketed for
  deployment-size reasons, not computed fresh per request.

## Where to go next depending on what you need

- Want to understand a specific technical decision and why? → `docs/00_TRD.md`
- Want to actually deploy this? → `docs/05_Deployment_Guide.md`
- Want to regenerate everything with real data? → `docs/04_Pipeline_Documentation.md`
- Want to use the live dashboard? → `docs/07_User_Manual.md`
- Want to verify a deployment is actually working, not just claiming to? →
  `docs/08_Testing_Validation_Prompt.md`
