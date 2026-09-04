# GridSense AI — User Manual

**Purpose:** How to read and use the dashboard. Written for someone who didn't build
this system but needs to operate or evaluate it.

---

## Opening the dashboard

Go to your deployed Vercel URL. The dashboard loads immediately — no login required in
the current pilot version.

**First thing to check:** is there a red banner at the top saying "API ERROR"? If yes,
the dashboard can't reach the backend — see Deployment Guide's troubleshooting section.
If no banner, you're looking at live (or demo-mode) data.

**How to tell if you're looking at real data or demo data:** the header shows either
"SIMULATION MODE" (demo/mock data, not connected to a real backend) or "LIVE ·
[your API URL]" (connected to the real deployed API). This is always visible, never
hidden.

---

## The five panels, what each one means

### 01 — Monitored Assets (left panel)
A ranked list of every grid zone being monitored, sorted highest risk first. Each row
shows:
- A colored status dot: green = normal, amber = elevated, red = critical
- The asset ID and zone name
- A risk percentage on the right

**Click any row to select that zone** — every other panel updates to show that zone's
detail.

### 02 — Grid Topology (top center, 3D visualization)
A live 3D scene showing each monitored zone as a node connected to a central hub.
- Each zone has its own fixed color (used consistently across the dashboard) so you can
  visually track "this is always Spain" regardless of its current risk level.
- A ring around each node pulses faster and glows brighter as that zone's risk score
  rises — this is directly tied to the real risk score, not decorative.
- **Click a node to select that zone**, same effect as clicking it in the asset list.
- The camera slowly orbits automatically — no interaction needed to see all zones.

### 03 — Risk Detail (right panel)
Detail for whichever zone is currently selected:
- A large percentage: the overload risk score.
- Predicted load vs. capacity, in megawatts. If capacity shows "· PROXY" next to it,
  that means real asset capacity data isn't available yet and this is an estimated
  figure (99th percentile of historical load) — see Pipeline Documentation for why.
- **Top Contributing Factors**: the model features that most influenced this risk score,
  with a bar showing relative magnitude and direction (red bar = pushes risk up, green
  bar = pushes risk down). These are precomputed for the zone's current risk bracket,
  not calculated fresh on every page load — see Pipeline Documentation Section 4.3 if
  you need the technical reason.
- A reminder at the bottom: this is decision-support only, no automated action is taken.

### 04 — 24H Load Forecast (bottom center, chart)
A forecast chart for the selected zone, showing predicted load for the next 24 hours.
- The solid line is the median forecast (P50).
- The shaded band and dashed lines above/below show the uncertainty range (P10 to P90)
  — the model's honest acknowledgment that it can't predict exactly, only a likely range.
- The chart automatically updates when you select a different zone.

### Footer bar
Shows deployment metadata: which model is running, whether the data source is real or
simulated, and a reminder that capacity figures are currently proxy estimates.

---

## What to do if something looks wrong

**All zones show the exact same risk score:** this was a real bug found once before
(data leakage in the risk classifier — see Pipeline Documentation Section 4.1). If you
see this again after a redeploy, it likely means the model was retrained without the
fix — check that `load_to_capacity_ratio` is still excluded from the risk classifier's
feature list.

**Numbers seem unrealistic or the dashboard is empty:** check whether you're in
"SIMULATION MODE" — if so, this is expected demo behavior, not a bug. If it says "LIVE"
and numbers still look wrong, the backend may be serving stale or synthetic training
data — see Pipeline Documentation Section 1 to check what data the deployed models were
actually trained on.

**Dashboard won't load at all / blank white page:** likely a frontend build or
deployment issue, not a data issue. Check the browser console (F12) for JavaScript
errors and check Vercel's deployment logs.

---

## Who this system is for, and who it isn't for (important boundary)

This dashboard is a decision-support tool for someone already qualified to make grid
operations decisions — it surfaces risk signals and reasoning, it does not replace
professional judgment, and it has no authority to execute any action on real
infrastructure. Nothing in this system should be treated as a substitute for a grid
operator's own assessment, especially given the documented limitations (proxy capacity
data, synthetic training data pending real ENTSO-E integration, no real fault/outage
history yet). See the TRD's Known Limitations section for the complete list.
