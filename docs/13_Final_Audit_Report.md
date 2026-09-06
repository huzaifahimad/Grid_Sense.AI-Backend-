# GridSense AI - Final Audit Report

**Audit date:** 2026-09-06
**Scope:** FastAPI + LightGBM backend and React + Vite + Three.js frontend
**Release commits:** backend `91da74d`, frontend `90a2bc3`

## Overall Status

**READY FOR HACKATHON SUBMISSION; NOT FULL PRODUCTION CERTIFIED**

The original uniform-risk deployment defect was fixed and the freshly retrained backend artifact was verified live. The final CORS fix is committed and pushed, but its post-redeploy browser result must still be confirmed after FastAPI Cloud restarts the service.

## Verified Fixes

- Removed `load_to_capacity_ratio` from the risk classifier feature list.
- Retrained the classifier and regenerated committed model artifacts.
- Added frontend request cancellation and stale-response protection.
- Added 60-second frontend polling.
- Separated asset and zone-detail error state.
- Removed the production `localhost:8000` API fallback.
- Made CORS configurable with `CORS_ORIGINS`.
- Added the verified Vercel origin as the backend CORS default:
  `https://grid-senseai.vercel.app`
- Corrected forecast documentation: P10/P90 are fixed +/-8% display bands, not learned quantiles.
- Updated duplicated documentation metrics.

## Model Evidence

Fresh local pipeline output:

```text
Persistence MAPE=  6.59%
Seasonal-naive MAPE=  5.20%
LightGBM MAPE=  2.23%
[risk_classifier] positive class rate (train): 10.73%
[risk_classifier] PR-AUC (test): 0.9595
```

The classifier audit confirmed:

```text
load_to_capacity_ratio in FEATURE_COLS False
```

Fresh local matched-snapshot scores:

```text
ES        prob=0.867338
GR        prob=0.000066
IT_SOUTH  prob=0.843588
```

## Live API Verification

The redeployed backend returned these results through the browser:

| Endpoint | Result |
|---|---|
| `/` | HTTP 200 |
| `/assets` | HTTP 200, 3 assets |
| `/forecast/ES` | HTTP 200, 24 points |
| `/risk/ES` | HTTP 200, risk 0.8673377959160332 |
| `/shed-schedule/ES` | HTTP 200, 1 item |
| `/forecast/GR` | HTTP 200, 24 points |
| `/risk/GR` | HTTP 200, risk 0.00006599940696357459 |
| `/shed-schedule/GR` | HTTP 200, 1 item |
| `/forecast/IT_SOUTH` | HTTP 200, 24 points |
| `/risk/IT_SOUTH` | HTTP 200, risk 0.8435876548093002 |
| `/shed-schedule/IT_SOUTH` | HTTP 200, 1 item |

The live `/assets` response showed ES and IT_SOUTH as critical and GR as normal, proving the deployed risk values are no longer uniform.

## CORS

Local configured-origin test passed:

```text
CORS_STATUS 200
CORS_ALLOW_ORIGIN https://grid-senseai.vercel.app
```

The exact Vercel browser request previously failed with:

```text
No 'Access-Control-Allow-Origin' header is present on the requested resource.
```

The backend fix is now committed in `91da74d`. A FastAPI Cloud restart/redeploy is required before claiming the browser CORS issue is resolved in production.

## Frontend Verification

```text
✓ built in 7.55s
No errors found.
```

The production bundle was checked for:

- Live FastAPI Cloud URL: present.
- `http://localhost:8000`: absent.
- Mock generator references: absent.

The frontend source has one dashboard route and consumes `/assets`, `/forecast/{zone}`, `/risk/{zone}`, and `/shed-schedule/{zone}`.

## Security

- No committed API keys, passwords, tokens, or private keys were found.
- `.env.local`, `node_modules`, `dist`, logs, and raw data are ignored.
- Python dependency check: `No broken requirements found.`
- Frontend audit: `found 0 vulnerabilities`.
- Authentication and authorization are not implemented.
- Rate limiting is not implemented.
- CORS is configurable and should be set to the exact Vercel origin in FastAPI Cloud.

## Data and Real-Time Status

The dashboard is **not genuine live telemetry**. It uses synthetic/static Parquet feature data and committed model artifacts. The frontend polls the REST API every 60 seconds. WebSockets and SSE are not implemented.

This must be described as a polling-based synthetic model snapshot, not a live utility telemetry system.

## Database and External Services

- Database: none present.
- ENTSO-E ingestion: implemented but requires an API token and was not verified in production.
- NASA POWER ingestion: implemented but not verified as a deployed refresh process.
- API serving: uses precomputed local/committed artifacts.

## Git and Deployment

Backend:

```text
91da74d Allow deployed Vercel origin by default
origin/main = 91da74d
```

Frontend:

```text
90a2bc3 Complete GridSense AI frontend release audit
origin/main = 90a2bc3
```

Both repositories were pushed successfully without force-pushing or changing remotes.

## Remaining Items

1. Redeploy/restart FastAPI Cloud with backend commit `91da74d`.
2. Reload `https://grid-senseai.vercel.app/` and confirm no CORS/API error appears.
3. Confirm `CORS_ORIGINS` in FastAPI Cloud is exactly:
   `https://grid-senseai.vercel.app`
4. The system remains synthetic/static with 60-second polling.
5. No automated unit or integration test suite exists.
6. Three.js reports a non-fatal `PCFSoftShadowMap` deprecation warning.
7. Authentication and rate limiting remain absent.

## Release Decision

The project is suitable for hackathon submission after the CORS redeploy and one browser refresh verification. It should not be presented as a production utility telemetry platform until real ingestion, authentication, rate limiting, automated tests, and deployment monitoring are added.
