# GridSense AI — Data Sourcing Plan

**Purpose:** Every source below is real, currently accessible, and verified as of Aug 2026.
No placeholder/synthetic sources are listed as if they were real. Where a source is
synthetic or a proxy, it's labeled as such.

---

## 1. Load / Demand Data (Primary — Regression Target)

### 1.1 ENTSO-E Transparency Platform (Europe) — **Primary training source**
- **What:** Actual load, generation by fuel type, cross-border flows, forecasts, outages,
  for 35 European countries, hourly resolution, from 2015 onward (some series from 2017).
- **Access:** Free. Register at transparency.entsoe.eu → email `transparency@entsoe.eu`
  with "Restful API access" in the subject line → token issued to your account within
  ~3 working days → generate a Web API Security Token from account settings.
- **Client:** Use the `entsoe-py` Python library (unofficial but reliable, wraps the raw
  XML API into pandas DataFrames) rather than parsing XML by hand.
- **License:** Mandated open publication under EU Regulation 543/2013. Commercial use
  permitted; attribution to ENTSO-E required.
- **Why this first:** Longest clean hourly history, most countries (lets you test cross-
  climate generalization — e.g., train on a hot-summer-peak country like Spain/Greece as a
  closer analogue to Pakistan's AC-driven demand than say Norway).

### 1.2 PJM Data Miner 2 (USA) — **Secondary training source / benchmark**
- **What:** Hourly metered load by zone (13 states + DC), day-ahead LMPs, load forecasts.
- **Access:** UI is public, no login needed for manual browsing. API access requires a free
  PJM Tools account. Non-members capped at 6 API calls/minute — plan batch pulls accordingly.
- **Why:** This is the exact type of dataset used as the generalization benchmark in the
  closest real precedent to this project — a published study trained an LSTM-hybrid model on
  actual NTDC (Pakistan) load data and validated generalization against AEP and ISO-NE
  (both PJM-adjacent US utility data), reporting MAPE of 2.38–3.72%. Replicate that
  benchmarking structure: train/validate cross-utility, not on one dataset alone.

### 1.3 Kaggle open grid datasets — **supplementary, for fault/anomaly modeling**
- Smart Grid Real-Time Load Monitoring Dataset (ziya07) — time-series load + fault flags
- Power Grid Dataset for Optimization and Fault Detection (ziya07)
- Power System Faults Dataset (ziya07) — fault type + environmental condition labels
- Grid Loss Time Series Dataset (Norway, TrønderEnergi Kraft) — real utility grid-loss data
- **Caveat:** Kaggle-sourced utility datasets vary in documentation quality — verify units,
  timezone, and sampling interval before use; do not assume metadata is complete or correct.

---

## 2. Weather Data (Feature Inputs)

### 2.1 NASA POWER API — **Primary weather source**
- **What:** Satellite/reanalysis-derived temperature, humidity, precipitation, wind speed,
  solar irradiance, at any global lat/lon, hourly/daily/monthly, from 1981 onward.
- **Resolution:** ~0.5° x 0.625° grid for meteorology (~50km) — coarse but consistent
  globally, meaning the same pipeline works unmodified when you later point it at Pakistani
  coordinates (Lahore, Karachi, Islamabad, Faisalabad grid zones).
- **Access:** No registration, no API key, free REST API, returns CSV/JSON directly.
- **License:** US federal open data — free for commercial use, no attribution legally
  required (courtesy acknowledgment requested).
- **Why this matters for accuracy:** Temperature is the single strongest external driver of
  short-term load in AC-heavy grids. This is the feature that will separate a mediocre model
  from a good one — do not treat weather as a nice-to-have.

### 2.2 Local/regional alternative (for later Pakistan-specific work)
- Pakistan Meteorological Department (PMD) publishes station data but not a clean public
  API — treat as a manual/negotiated data source later, not part of the open-data prototype.

---

## 3. Fault / Outage Event Data (Classification Labels)

This is the weakest link in open data, and I'm not going to pretend otherwise.

- **Kaggle "Power System Faults Dataset"** — real fault-type/environmental-condition
  records, usable for training the outage/risk classifier's feature relationships, but
  scale and geographic coverage are limited.
- **ENTSO-E "Unavailability" feeds** — transmission grid unavailability/outage domain data
  (planned + unplanned outages) is part of the Transparency Platform and is a legitimate
  real proxy for "asset was down" events in Europe.
- **Synthetic proxy label (documented, not hidden):** Where no real fault log exists,
  construct a proxy label: `overload_event = 1 if load > (0.9 × known_or_estimated_capacity)`.
  This is a standard technique but must be labeled as a proxy in every model card and
  dashboard footnote — never presented as ground truth.

---

## 4. Pakistan-Specific Sources (context data now, operational data later)

These will NOT give you training-ready time series today, but they matter for context,
credibility, and eventual retraining:

- **NEPRA State of Industry Reports** (annual, public PDF) — installed capacity, generation
  mix, transmission additions, documented overload incidents (e.g., FY2023 part-load
  adjustment charges of PKR 46.6B tied to system constraints — cite this in investor decks,
  it's real and it's exactly the cost your product claims to reduce).
- **NTDC IGCEP / TSEP planning documents** (public, via NEPRA filings) — generation and
  transmission expansion plans, useful for asset/topology context.
- **Precedent research:** A peer-reviewed study (PMC, 2023) used actual NTDC load data for
  an LSTM-based short-term load forecasting model — proof that academic-channel data access
  to NTDC has happened before. Worth tracking down the paper's authors/institution as a
  potential access pathway or collaboration lead.
- **Known infrastructure constraint:** NEPRA inspection reports document NTDC's SCADA system
  as having RTU shortages and portions dating to 1992 with reduced functionality. Design
  around this — do not assume clean, complete real-time data even after access is granted.

---

## 5. Data Governance Notes

- Every external dataset used must be logged with: source, license, access date, and
  known limitations — this becomes an appendix in the model card (see ML doc).
- No dataset should be used in a way that violates its license (e.g., PJM Data Miner data
  redistribution is restricted without a PJM redistribution license — fine for internal
  model training, not fine for republishing raw data in a public repo).
- Treat any future NTDC/DISCO data as confidential-by-default; do not commingle it with
  open-data training pipelines without an explicit data-handling agreement in place.
