# GridSense AI — Backend

Power grid overload risk prediction API. FastAPI + LightGBM, deploys to
[FastAPI Cloud](https://fastapicloud.com).

**Start here:** `docs/06_Project_Overview.md` for what this is,
`docs/05_Deployment_Guide.md` for how to deploy it.

## Quick local run

```bash
pip install -r requirements.txt --break-system-packages
fastapi dev
```

Then: `curl http://localhost:8000/assets`

## Important — before you touch anything

- `config.py`'s `ENTSOE_API_KEY` must stay as the placeholder in every commit. Real
  tokens go into FastAPI Cloud's secret store, never into this repo.
- The committed `data/features/` and `data/models/` files are trained on SYNTHETIC data
  right now — see `docs/04_Pipeline_Documentation.md` before citing any accuracy number.
- This repo is backend-only. If you ever see frontend/React files appear here, something
  has gone wrong — see `docs/09_Two_Repo_Setup_Guide.md`.

## Full documentation

| Doc | What it covers |
|---|---|
| `docs/00_TRD.md` | Technical requirements, what's built, known limitations |
| `docs/01_Data_Sourcing_Plan.md` | Real data sources, APIs, licensing |
| `docs/02_ML_Model_Design.md` | Model architecture, features, training strategy |
| `docs/03_System_Architecture.md` | Full deployment architecture and why it's shaped this way |
| `docs/04_Pipeline_Documentation.md` | Data flow, real-vs-synthetic breakdown, exact run order |
| `docs/05_Deployment_Guide.md` | Step-by-step deploy instructions |
| `docs/06_Project_Overview.md` | Plain-language explanation of the whole project |
| `docs/07_User_Manual.md` | How to use the dashboard (frontend-facing, included here too) |
| `docs/08_Testing_Validation_Prompt.md` | QA prompt for verifying a deployment actually works |
| `docs/09_Two_Repo_Setup_Guide.md` | Why two repos, exactly what goes where |
