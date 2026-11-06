# 14. Security Posture — measured, not asserted

Date of every measurement below: **2026-11-06**. Live target
`https://grid-senseai-backend.fastapicloud.dev`; local target
`http://127.0.0.1:8126` (a spare port — the developer's own server on 8123 was never
touched). Live probes need `user-agent: gridsense-verify`; the platform WAF returns 403 to
default curl agents, which is a protection working as intended, not a finding.

This document is written so a reviewer can re-run any line. Where a control is **absent**,
it says absent. Where a risk is **accepted**, it says accepted and why.

---

## 1. Attack surface, measured

| Item | Measured result | How to re-check |
|---|---|---|
| Routes | 7 read-only GET endpoints + `/docs`, `/redoc`, `/openapi.json` | `GET /openapi.json` |
| Methods | `POST /health` → **405** | `curl -X POST …/health` |
| Write paths | None. No upload, no admin, no auth, no user input persisted | read `api/main.py` route decorators |
| Request-time egress | **Bounded, no longer zero** (changed 2026-11-06, `f78b76e`). `api/live_refresh.py` fetches the missing ENTSO-E tail, plus NASA POWER when enabled, from a background thread it starts at most once per cache window. Two fixed destinations: the ENTSO-E host compiled into `entsoe-py` and `config.NASA_POWER_BASE_URL`. | `GET /health` → `data_refresh.{attempts,last_success_at,last_error,zones}` |
| Secrets in 38 commits of history | None matched AWS key, PEM header, JWT, or assigned-token patterns (re-run 2026-11-06T11:06Z, 0 matches) | `git log --all -p \| grep -icE "(AKIA[0-9A-Z]{16}\|BEGIN (RSA \|EC \|OPENSSH )?PRIVATE KEY\|eyJ\[A-Za-z0-9_-\]{20,\}\.eyJ\|securityToken=\[0-9a-f\]{32})"` |
| Path traversal | `/../etc/passwd` → 404, `/%2e%2e/%2e%2e/etc/passwd` → 404 | `curl -i` both |

The row above previously read "Zero" and was checked with
`grep -rnE "requests\|httpx\|urllib" api/ models/forecasting.py`. That command still
returns nothing after the change, which is exactly why it was dropped as evidence:
`api/live_refresh.py` imports the egress modules lazily, inside the functions that need
them, so a source-text grep over `api/` cannot see the outbound call. A check that passes
while the property it claims is false is worse than no check, because it reads like
verification.

What still holds, and what the new bound rests on:

* **No request input reaches the destination.** The threads fetch the zones already
  present in the served frame (`config.ENTSOE_ZONES` keys), never the `{zone}` from the
  request path, so there is no SSRF surface reachable over HTTP. This bound is weaker than
  the zero-egress property it replaces and its gap is listed in §5: entsoe-py resolves its
  endpoint as `os.getenv("ENTSOE_ENDPOINT_URL") or "https://web-api.tp.entsoe.eu/api"`, so
  the destination is fixed against network attackers but movable by anyone who can already
  set the process environment.
* **The token is sent to one destination, as a query parameter.** `ENTSOE_API_KEY` goes
  over HTTPS to the host above as `securityToken`; NASA POWER needs no key. That query
  parameter is the reason the environment-variable override matters and is recorded as an
  accepted risk in §5, not argued away here.
* **Request volume cannot amplify egress.** One refresh per process per
  `GRIDSENSE_REFRESH_CACHE_SECONDS` (default 900 s), one concurrent refresh per process,
  and a three-zone deadline of `timeout_seconds × 3`. Measured by
  `test_cache_window_blocks_the_next_kick`, which asserts the second request in the window
  never reaches `pull_load`.
* **`/health` cannot be used to trigger it.** The readiness route deliberately does not
  call the refresher, so an unauthenticated probe loop spends no egress budget.
* **A failed or blocked fetch is invisible to callers.** Every path falls back to the
  committed frame; the failure is recorded in `data_refresh`, not returned as a 5xx.
  Verified by `test_egress_failure_leaves_the_frame_serving`.

## 2. Fixed: reflected input was unbounded

`GET /risk/{zone}` and its siblings build their 404 message by interpolating the caller's
zone segment. Before the fix a 4,000-character segment came back verbatim, which writes
attacker-chosen text into the response body and every log line downstream of it.

Now capped at `MAX_ZONE_ECHO_CHARS = 32` (api/main.py:229). Measured on the patched local
server:

```
status: 404 | body length: 104
detail: No data for zone 'AAAA…'. Known zones: ['ES', 'GR', 'IT_SOUTH']
4000-A present: False
```

Covered by `test_unknown_zone_rejection_echoes_only_a_bounded_prefix` in
`tests/test_api.py`, parametrized across all three zone routes. Re-probed against the
deployed app after the push: identical 104-byte body, long run absent.

The zone set is closed (`ENTSOE_ZONES`), so a near-miss is never worth echoing in full —
the response already lists the valid zones.

## 3. Fixed: no response headers were set at all

Live baseline before the change (`GET /health`, 2026-11-06 08:55 UTC) returned only
`Content-Type`, `Content-Length`, `Server`, `Date`, `vary: Origin`, `cf-cache-status`,
`CF-RAY`, and `x-envoy-upstream-service-time`. No security header, and no `Cache-Control`.

The missing `Cache-Control` was the one with product consequences: the dashboard prints how
old the served frame is, and any intermediary holding a cached copy makes that number a lie.

Six headers are now set on every response by `_response_hygiene` (api/main.py:211, header map
at api/main.py:200):

| Header | Value | Why this value |
|---|---|---|
| `X-Content-Type-Options` | `nosniff` | blocks MIME-sniffing of JSON into script |
| `Referrer-Policy` | `no-referrer` | no URL-embedded context leaks outbound |
| `X-Frame-Options` | `DENY` | clickjacking; legacy fallback for the CSP below |
| `Content-Security-Policy` | `frame-ancestors 'none'` | modern framing denial |
| `Strict-Transport-Security` | `max-age=31536000` | justified by the measured 301 upgrade (§4) |
| `Cache-Control` | `no-store` | keeps the freshness claims true |

Verified present on the patched server at `/health`:

```
x-content-type-options: nosniff
referrer-policy: no-referrer
x-frame-options: DENY
content-security-policy: frame-ancestors 'none'
strict-transport-security: max-age=31536000
cache-control: no-store
```

`test_every_response_carries_security_and_nocache_headers` asserts all six on `/`, `/health`,
`/validation`, `/assets`, `/forecast/ES`, `/risk/ES`, `/shed-schedule/ES` and `/docs`.

**Verified in production, not just locally.** After the push rolled the deployment, the same
eight routes were re-probed live at 09:05 UTC on 2026-11-06: every one returned 200 with
`missing=none` for all six headers.

**Two deliberate omissions, stated rather than quietly skipped.** No CSP `default-src`:
`/docs` and `/redoc` are served by this same app and need inline scripts, so a strict
directive would break them — see §5. No HSTS `includeSubDomains` or `preload`: this host sits
on a `fastapicloud.dev` domain this project does not control, and preloading a host you do not
own pins every sibling subdomain to HTTPS for months on someone else's behalf.

## 4. Controls verified already correct

- **CORS is an exact allowlist, and it is not open by default.** With `CORS_ORIGINS` unset,
  both live and local echo `access-control-allow-origin: https://grid-senseai.vercel.app` and
  return **no** CORS header for `https://evil.example.com`. `vary: Origin` is present, which is
  the correct cache-variance consequence of an origin-dependent response.
  `README.md` previously claimed an unset value "defaults to open CORS" — measured false, corrected.
- **Transport upgrade.** `http://` → **301** with `Location: https://grid-senseai-backend.fastapicloud.dev/health`,
  and following it returns 200. This is what justifies sending HSTS at all.
- **Query bounds.** `horizon_hours` is `ge=1, le=48`; 0 and 49 both measured **422**. Bounded
  horizon bounds worst-case compute per request.
- **No auth to weaken.** Every endpoint is intentionally public read-only; there is no session,
  token, or password in the request path to leak, brute-force, or mis-validate.
- **Secret handling.** `config.py` reads `ENTSOE_API_KEY` from the environment only, and it is now
  used on the serving path too: `api/live_refresh.py` needs it to fetch the missing tail. Because
  that same module publishes exception text on `/health`, and entsoe-py carries the token in the
  query string that requests/urllib3 quote into those messages, `config.redact_secrets` strips
  credential-shaped parameters and every stored message passes through it **before** the length cap
  (a cap applied first can drop the `&` that terminates the parameter and leave half a token).
  The rule is applied at every site that converts an exception into text, not only where the text is
  stored — enumerated with `grep -rn "str(e)\|repr(e)\|exc_info=True\|format_exc\|{e}" api/
  ingestion/ models/ features/ config.py`. That sweep found one site still printing raw:
  `ingestion/entsoe_client.pull_all_zones`, whose per-zone failure goes to stdout, i.e. the log
  stream of a deployed process. It now calls `redact_secrets` itself instead of relying on
  `pull_load` having redacted before raising. Verified by
  `test_published_error_never_carries_the_token`, which asserts on the real token value inside a
  message copied from the live failure payload, by `test_redaction_runs_before_the_length_cap`, and
  by `test_offline_pull_stdout_never_carries_the_token` (negative control measured first: the same
  exception printed raw yields `leak in raw: True`, redacted yields `False`).

## 5. Accepted risks — explicit, with the condition that would change the answer

| Risk | Status | Reasoning |
|---|---|---|
| `/docs`, `/redoc`, `/openapi.json` live (all 200) | **Accepted** | The API is public, read-only and unauthenticated, so the schema is not a secret and a hostile reader gains nothing a `curl` cannot give them. The cost of closing it is a broken `/docs`. Revisit if the deployment ever gains a write path or private zone. |
| No CSP `default-src` | **Accepted** | Same reason: it would break `/docs` and `/redoc`. `frame-ancestors` covers the framing risk that matters for a JSON API. |
| Pickle deserialization of committed `.joblib` artifacts (api/main.py:105,112,119) | **Accepted, with a real caveat** | `joblib.load` executes arbitrary code contained in the file. There is no upload endpoint, so the only way to exploit it is to commit a malicious artifact — which means **write access to this repo is equivalent to code execution on the serving host**. Mitigation is review discipline: `scripts/verify_prediction_integrity.py` reproduces served values from the committed artifacts (33/33 checks), which catches a swapped artifact that predicts wrong but not one carrying a payload. Moving to a format without code execution (`safetensors`, ONNX, PMML) is the structural fix. |
| Rate limiting not implemented in app | **Accepted at this tier** | The platform sits a WAF in front (measured: it 403s unrecognized user agents). No app-level limit exists; a determined caller can force recursive 48-step forecasts. Add `slowapi` before exposing this to untrusted traffic at volume. |
| Capacity is an estimate, not metered rating | **Not a code risk, a claim risk** | `capacity_mw` is a p99-of-load proxy and is labelled as such on the wire (`is_proxy_capacity: true`). Documented in `docs/04` and `docs/13`. |
| The ENTSO-E request destination is environment-configurable | **Accepted, same boundary as the pickle row** | `entsoe/entsoe.py:32` evaluates `URL = os.getenv("ENTSOE_ENDPOINT_URL") or "https://web-api.tp.entsoe.eu/api"` at import time, and the token travels in that request's query string — so a process environment an attacker already controls also chooses where the credential is sent. That is not an added exposure: setting `ENTSOE_API_KEY` at all requires the same privilege, and environment write access on this host already equals code execution via `joblib.load` (row above). Structural fix if that boundary ever matters: pin the endpoint in code (`EntsoeRawClient` accepts a `session`, so a `requests.Session` with a transport-level host allowlist would make redirection impossible rather than unlikely). |

## 6. Dependency posture

`scripts/audit_dependencies.py` queries the OSV API for each declared pin. It exists instead
of `pip-audit` because `pip-audit` resolves and downloads distributions, while the question
that matters for a deployment is whether the exact versions we ship are affected.

Measured 2026-11-06, before and after remediation:

| Scope | Before | After |
|---|---|---|
| serving (`pyproject.toml`, 9 pins) | 0 findings | 0 findings |
| extras (`requirements.txt`, 14 pins) | 3 findings | 0 findings |

The three findings were `requests==2.32.3` (2 distinct CVEs — CVE-2024-47081, CVE-2026-25645 —
across 4 advisories; GHSA and PYSEC routinely double-file one CVE, so the count that matters is
distinct CVEs), and two `>=` specifiers in a file whose own header comment forbids exactly that.
Remediated: `requests==2.34.2`, `kaggle==2.2.5`, `torch==2.15.0` (2.5.1 scores 19 distinct CVEs),
`entsoe-py==0.8.0` to match the version that actually performed the 52,522-row pull. Each new pin
was confirmed to exist on PyPI before being written down.

CI now runs `python scripts/audit_dependencies.py --all` as a **blocking** step
(`.github/workflows/ci.yml`). Exit codes are distinct on purpose: 0 clean, 1 advisory or unpinned,
**2 unable to reach OSV** — a check that reports success because the database was unreachable
would earn a green badge while proving nothing, so that path fails the build.

### 6.1 Declared pins are not this machine's versions

`pyproject.toml` claimed its pins "are the ones the pipeline was actually verified against."
Measured against the local interpreter, **5 of 9 have drifted**: fastapi 0.104.1 (vs 0.141.1),
uvicorn 0.24.0, pydantic 2.5.0, pandas 2.3.3, pyarrow 23.0.1; numpy, lightgbm, joblib and
scikit-learn match. `torch` and `kaggle` are not installed at all.

What this changes: the claim was only true of the *deployed* app, which does install these pins and
was verified live. It is not true of artifact reproduction locally. Corrected in `pyproject.toml`
and `docs/02` §4, and made re-checkable with `python scripts/audit_dependencies.py --installed-drift`
(measured: 5 drifted serving, 13 including extras).

The one advisory that lands on a developer machine rather than production: installed
fastapi 0.104.1 carries **CVE-2024-24762**, network-exploitable with no privileges and no user
interaction, availability impact High. Local servers bind `127.0.0.1`, so exposure is limited to
processes already on this host. The deployed app is on 0.141.1 and is not affected. Upgrading the
local environment was **not** done here — it would move the framework under a server the developer
may be running — and is left as an explicit decision.

## 7. Honest limits of this review

- The dependency audit proves *declared versions are not affected by published advisories*. It does
  not prove the installed environment matches the declaration (§6.1 exists because of that gap), and
  it does not assess advisories not yet published.
- The secret scan is pattern-based over 28 commits of text. Tracked **binary** artifacts
  (`.parquet`, `.joblib`) were not string-scanned; a key could in principle be embedded in one.
  Spot-checking them would need a different tool than grep.
- No penetration test, no fuzzing, no load test was run. The bounds probes were hand-written
  single requests.
- This review covers the backend. The frontend (separate repo, Vercel) was not re-assessed here.

## 8. Test evidence

`python -m pytest tests -q` on 2026-11-06T11:21Z: **90 passed**, 1 warning (an httpx `app=`
deprecation in the test client, not in product code). The 63-test run recorded earlier the same day
has since grown by `tests/test_live_refresh.py` (27 tests), which is where the security-relevant
coverage of the new egress path lives:

- `test_the_refresh_client_really_calls_the_http_layer` — real entsoe-py client, only
  `requests.Session.get` stubbed, asserting an HTTP attempt actually happens.
- `test_published_error_never_carries_the_token` — the credential value must be absent from
  `/health` while the root cause is still present.
- `test_redaction_runs_before_the_length_cap` + `test_credential_pattern_covers_the_parameter_spellings_used`.
- `test_offline_pull_stdout_never_carries_the_token` — the ingestion script's own failure print is
  redacted, so the credential cannot reach the deployed log stream.
- `test_egress_failure_leaves_the_frame_serving` — a failed fetch changes nothing the caller sees.
- `test_cache_window_blocks_the_next_kick` — request volume cannot amplify egress.

**Limitation of the evidence above.** These runs execute on this machine's interpreter, whose
package versions differ from the pins in `requirements-api.txt` that the deployed process installs
(local pandas 2.3.3 / requests 2.31.0 / fastapi 0.104.1 / pyarrow 23.0.1 vs pinned
pandas 3.0.2 / requests 2.34.2 / fastapi 0.141.1 / pyarrow 25.0.1). Every pin was confirmed to exist
on PyPI with installable wheels and a compatible `requires-python` on 2026-11-06T11:14Z, so the
build itself is not at risk; what is not yet proven by these 90 tests is behaviour under the pinned
pandas major version. §10 item 5 is the ranked action for that drift.

CI on commit `b8c5651` completed with `conclusion=success`, and the per-step record proves the
new gate executed rather than being skipped:

```
4. Dependency advisory audit (OSV)      success
6. Run python -m pytest tests -q        success
```

The CI pytest pass is the stronger of the two results. That runner installs the declared pins
from scratch, so it is the first evidence that the suite passes against fastapi 0.141.1 and
pandas 3.0.2 rather than the drifted local versions in §6.1 — which is exactly the distinction
the drift check exists to make visible.

## 9. Deployed-bundle evidence (2026-11-06T11:15Z)

The failure class that bit this project once was a green build with 500 endpoints: the upload
ignore-list excluded a module the request path imports at request time. `.fastapicloudignore`
excludes `ingestion/` file-by-file rather than as a directory precisely so the two modules the
refresher needs survive. A list that long is easy to get wrong, so the claim was tested by
reconstructing the upload set locally (gitignore semantics: last matching rule wins, `!`
re-includes) and running the API out of that stripped directory.

- 14 endpoints requested from inside the bundle — `/`, `/health`, `/assets`, `/validation`,
  `/docs`, `/openapi.json`, three `/forecast/*`, three `/risk/*`, two `/shed-schedule/*` — all
  returned **200**, exit code 0. `ingestion/entsoe_client.py` resolved from the bundle.
- The refresher's ENTSO-E and weather imports are **lazy**, inside the functions
  (`api/live_refresh.py:298`, `:442`, `:483`), so a source grep for module-level imports cannot see
  them and a no-key start never executes them. Forcing the refresh inside the bundle with a fake
  token resolved all three.
- With that fake token the bundle reached the **real** SOP endpoint and got
  `HTTPError: 401 Client Error: Unauthorized for url: https://web-api.tp.entsoe.eu/api?...` —
  direct confirmation that the one-attempt configuration actually performs egress, which the
  previous `retry_count=0` did not. The zone recorded outcome `failed`, **0 rows published**, and
  the stored `last_error` (179 chars) contained neither the token nor any part of it.

## 10. What remains, in order of value

1. **Replace pickle loading** with a non-code-executing artifact format. This is the only accepted
   risk above that could turn into code execution.
2. **Forward-looking weather feed**, or keep the no-weather fallback as the freshness answer and stop
   describing reanalysis lag as a data problem.
3. **Rated MW capacity** from an operator source (NTDC for the current ENTSO-E stand-in zones), so
   `capacity_mw` stops being a p99 proxy.
4. **App-level rate limiting** before the endpoint faces untrusted traffic at volume.
5. **Upgrade the local dev environment** to the declared pins, or keep documenting the drift.
