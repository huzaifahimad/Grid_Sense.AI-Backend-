# GridSense AI — Two-Repo Setup Guide

**Direct answer to "do I need two repos or what": yes, two repos.** Here's exactly why
and exactly how, so there's no ambiguity when you push.

---

## Why two repos, not one

This was tested, not assumed. An earlier attempt put everything in one repo and tried to
deploy the Python ML backend as Vercel functions alongside the frontend. It failed —
measured directly: even the smallest possible version of the backend's dependencies
(282MB) is bigger than what Vercel allows for a single function (250MB), because the
`lightgbm` library needs `scipy`, which alone is 109MB. There's no way to shrink this
further and keep the model working. So the backend needs a completely different kind of
hosting than the frontend does — which means two separate deployable things, which is
cleanest as two separate repos.

## The two repos, exactly

### Repo 1: `gridsense-backend`
Contains: `api/`, `ingestion/`, `features/`, `models/`, `config.py`, `data/`,
`pyproject.toml`, `.python-version`, `.fastapicloudignore`, `requirements*.txt`,
`render.yaml` (fallback), `docs/` (all documentation lives here), `.gitignore`.

Deploys to: **FastAPI Cloud** (primary) via `fastapi deploy`. Render is a documented
fallback if a card ever becomes available.

### Repo 2: `gridsense-frontend`
Contains: `src/`, `public/`, `index.html`, `package.json`, `vite.config.js`,
`vercel.json`, `.env.example`, `.gitignore`.

Deploys to: **Vercel**.

**Nothing Python should ever appear in the frontend repo.** No `requirements.txt`, no
`api/` folder, no `.joblib` files. If you ever see these show up there, something has
gone wrong — go back to `docs/03_System_Architecture.md` Section 2.2 for why that
specific mistake happened once already and what it broke.

---

## Pushing both, step by step

```bash
# Backend
cd gridsense-backend
git init
git add .
git commit -m "GridSense AI backend"
git branch -M main
git remote add origin https://github.com/<you>/gridsense-backend.git
git push -u origin main

# Frontend
cd ../gridsense-frontend
git init
git add .
git commit -m "GridSense AI frontend"
git branch -M main
git remote add origin https://github.com/<you>/gridsense-frontend.git
git push -u origin main
```

Two GitHub repos, two Vercel/FastAPI Cloud projects, one connecting to the other via a
single environment variable (`VITE_API_BASE_URL` on the frontend, pointing at the
backend's live URL). That's the whole relationship between them — they don't share code,
don't share a build process, and don't need to live in the same place.

## The one thing that connects them

That's it — one string, one place it lives (Vercel's environment variables), pointing
from the frontend to wherever the backend ends up running. Everything else about the two
repos is fully independent.
