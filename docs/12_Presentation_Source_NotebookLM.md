# GridSense AI — Presentation Source Document

**How to use this file:** Upload this single document to NotebookLM as a source, then
ask it to generate a slide deck or presentation. Each section below is written as one
self-contained idea with a clear takeaway, deliberately structured so each section maps
cleanly to one or two slides. Headings are the natural slide titles. Bullet points are
short and scannable on purpose — this is written to be presented, not read as a report.

---

# Slide 1: Title

**GridSense AI**
Power Grid Overload Risk Prediction for Pakistan
Built by Huzaifa Himad

---

# Slide 2: The Problem

Pakistan's power grid operates under chronic strain. Load-shedding — planned,
region-by-region power cuts — is a routine operational tool, not a rare emergency
measure, because utilities frequently cannot predict overload risk far enough in advance
to manage it proactively instead of reactively.

- NEPRA (Pakistan's energy regulator) reported PKR 46.6 billion in part-load adjustment
  charges in FY2023 alone — a direct financial cost tied to exactly this kind of system
  constraint.
- NTDC's SCADA monitoring infrastructure has documented gaps: RTU shortages, and
  portions of the system dating to 1992 with reduced functionality.
- The result: grid operators are often responding to overload after it starts, not
  predicting it before it happens.

**The core question this project answers:** can machine learning predict grid overload
risk hours in advance, using data that's actually available, in a way that's honest
about its own limitations?

---

# Slide 3: The Solution

GridSense AI is an end-to-end machine learning system that:

1. **Forecasts electricity load** 1–48 hours ahead, with honest uncertainty bands
   (not just a single number pretending to be certain)
2. **Scores overload risk** per grid zone, with plain-language explanations of what's
   driving that risk
3. **Recommends a load-shedding priority order** when risk is high — as a suggestion
   for a human operator to review, never as an automated action
4. **Visualizes all of this** in a live, 3D dashboard showing real-time grid topology

It's built on real, open grid and weather data today, with the explicit architecture to
retrain directly on Pakistani NTDC/DISCO data once institutional access is secured.

---

# Slide 4: Who This Is For

- **Grid operators and utility engineers** who need risk visibility hours in advance,
  not just real-time alarms
- **NTDC and DISCOs** as a decision-support layer that complements, not replaces,
  operator judgment
- **Anyone evaluating this as a technical portfolio piece** — the honest documentation
  of what's real, what's tested, and what's still a known gap is itself part of the
  demonstration of engineering maturity

---

# Slide 5: How It Works — The Data Pipeline

Real grid load data comes from **ENTSO-E**, the European transmission operators'
open-data platform — mandated by EU law to be public, hourly, and free. Real weather
data comes from **NASA POWER**, a free, no-registration-required satellite/reanalysis
API.

Why European data for a Pakistan-focused project: the goal is proving the forecasting
*approach* works on real, high-quality grid data first, with the explicit next step
being retraining on real Pakistani data once that access is secured — European
hot-summer-climate zones (Spain, Greece, southern Italy) were deliberately chosen as the
closer climate analogue to Pakistan's AC-driven demand pattern, rather than picking
countries at random.

**The pipeline, in order:** ingest load and weather data → engineer features (lag
values, rolling statistics, cyclic time encodings) → train two separate models → serve
predictions through an API → display in the dashboard.

---

# Slide 6: How It Works — Two Models, Not One

A common mistake in grid ML projects is treating load forecasting and overload risk as
one problem. They're genuinely different:

| | Load Forecasting | Overload Risk |
|---|---|---|
| **Type** | Regression (predict a number) | Classification (predict a probability) |
| **Question answered** | "How much power will be needed?" | "Is that amount dangerous for this asset?" |
| **Model** | LightGBM (gradient-boosted trees) | LightGBM (gradient-boosted trees) |
| **Output** | Predicted MW, with a confidence range | Risk score 0–1, with explanation |

Both use LightGBM rather than deep learning — a deliberate choice. Deep learning was
evaluated (the architecture exists in the codebase) but wasn't deployed, because
LightGBM already cleared standard accuracy baselines by a wide margin and didn't need
the added complexity justified. Simpler, faster, easier to explain to a utility, and
just as effective at this data scale.

---

# Slide 7: A Bug We Found — and Why That Matters

Early in testing, the overload risk model showed a suspiciously perfect accuracy score
(PR-AUC of 0.9997) — and every single grid zone returned an *identical* risk score, down
to thirteen decimal places.

That's not a good sign. It's the signature of **data leakage**: a feature that's a
near-copy of the answer the model is supposed to predict. Investigation confirmed it —
one input feature was mathematically almost identical to the label being predicted, so
the model was reading the answer off a hidden copy of itself rather than actually
learning anything.

**The fix:** remove that feature from the model's inputs. Accuracy dropped to a more
honest 0.9654 afterward.

**Why this is worth a slide of its own:** a lower, honest number after fixing a real bug
is a better outcome than a suspiciously perfect one — and this kind of issue only
surfaces by actually running and stress-testing a system, not by reading the code and
assuming it works. That discipline was applied throughout this project.

---

# Slide 8: Explainability — Why the Model Made That Call

Every risk score comes with the top contributing factors — for example: "predicted load
at 94% of capacity, temperature elevated for 3 consecutive days, moderate load
volatility." This isn't a black box.

**Engineering trade-off, stated honestly:** the explanations are computed once, at
training time, and grouped by risk level (low / moderate / elevated / critical) rather
than calculated fresh for every single request. This was a deliberate choice after
testing showed that live, per-request explanation calculations added significant
technical weight for a benefit that didn't justify the cost at this project's scale — the
explanations are "typical top factors for this risk level," not a bespoke calculation
for every exact moment. That's a real, acknowledged trade-off, not a shortcut hidden from
view.

---

# Slide 9: The Dashboard

A live, 3D visualization of the grid — not a decorative graphic, a functional one:

- Each monitored zone is a node in a 3D topology view, connected to a central hub
- Node color and pulse intensity are directly driven by that zone's real-time risk
  score — the visualization is data, not decoration
- A ranked list shows every monitored asset sorted by risk, highest first
- A forecast chart shows predicted load 24 hours ahead, with honest uncertainty bands
  (a range, not a false-precision single number)
- A recommended shed-schedule panel shows exactly which assets are flagged, in what
  priority, and why — always as a suggestion, never an automated action

---

# Slide 10: Engineering Discipline — What Sets This Apart

This project treats "it runs" and "it's correct" as two different bars, and clears both
deliberately:

- **Baselines came first.** Before any model was trusted, it had to beat a naive
  persistence forecast and a seasonal-naive forecast by a real, measured margin — not
  assumed to be better just because it's more sophisticated.
- **Walk-forward validation, not random splits.** Time-series data leaks badly under
  random train/test splits; this project uses expanding-window validation instead,
  which is the technically correct approach and a common mistake to get wrong.
- **Every claimed number was actually measured**, not estimated — including the ones
  that turned out to reveal bugs.
- **Deployment architecture was tested, not assumed.** An early attempt to run the
  Python ML backend as Vercel serverless functions was tested directly and found to
  measure over Vercel's size limit even at the absolute minimum viable dependency
  set — a real, load-bearing engineering finding, not a guess.

---

# Slide 11: Deployment Architecture

Two separate, purpose-fit platforms:

- **Backend (Python/FastAPI + LightGBM)** deploys to **FastAPI Cloud** — a platform
  built by FastAPI's own creators, chosen because it runs a persistent process (needed,
  since the models load into memory at startup) and requires no credit card.
- **Frontend (React/Three.js dashboard)** deploys to **Vercel** — the right tool for a
  static/SPA frontend, and genuinely not the right tool for a Python ML backend
  (confirmed by direct testing, not assumption).

This split wasn't the starting design — it's what real testing led to, and that
reasoning is documented in full rather than presented as if it were obvious from the
start.

---

# Slide 12: Honest Limitations

Stated plainly, not discovered later by someone else:

- **Currently trained on synthetic data**, not yet real Pakistani or even real ENTSO-E
  data in the deployed models — a deliberate bootstrap choice to validate the full
  pipeline end-to-end before waiting on external API approval delays, clearly documented
  as such rather than presented as more than it is.
- **No real fault/outage history yet** — overload risk currently uses a proxy label
  (load exceeding 90% of estimated capacity), a standard technique but a real accuracy
  ceiling until genuine outage records are available.
- **Capacity figures are currently estimated**, not real asset capacity data.
- **No live database yet** — predictions are served from a data snapshot, refreshed
  manually, not continuously updating. (A specific, deliberate engineering decision —
  not needed at current scale, with a clearly identified trigger point for when it will
  be.)

---

# Slide 13: What's Next

1. Secure real NTDC/DISCO institutional data access and retrain on real Pakistani grid
   data — the architecture is already built for this, it's a data-access step, not a
   rebuild.
2. Incorporate real historical fault/outage records to replace the current proxy risk
   label.
3. Move from a manually-refreshed data snapshot to scheduled, live ingestion — the point
   at which a real database becomes genuinely necessary, per this project's own
   documented assessment of that trade-off.
4. Pilot with a real utility partner, framed as low-risk R&D value in exchange for
   anonymized historical (not real-time) data access first — a realistic, incremental
   path to real-world validation.

---

# Slide 14: Closing

GridSense AI demonstrates that rigorous ML engineering — honest baselines, real bug
discovery, tested (not assumed) deployment architecture, and transparent limitations —
can be applied to a problem that matters: predicting grid overload before it happens, in
a country where that prediction gap has a real, measurable cost.

**Built by Huzaifa Himad.**
