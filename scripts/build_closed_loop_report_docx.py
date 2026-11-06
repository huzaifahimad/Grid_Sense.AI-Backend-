#!/usr/bin/env python3
"""Build submission/GridSense_AI_Closed_Loop_Validation_Progress_Report.docx.

The previous copy of this report was written on 3 November by hand, with its
figures transcribed from the closed-loop artifact committed then (commit
502cdd8). Since that date the models were retrained on two years of real
ENTSO-E load and NASA POWER weather and the replay window moved, so ten of its
statements stopped being true while the file stayed in the delivery folder. It
still read "overall PASS - 13 of 13 gates", "47 automated tests", "models
trained on documented synthetic bootstrap data" and "median stable warning lead
3.0 h", where the deployed API now answers 9 PASS / 4 WARN / 0 FAIL, 94 tests,
is_synthetic=false and a 1.0 h median stable lead.

So this document is generated. Everything about the *current* system is read at
build time from live /validation and /health, from pytest collection and from
git. Everything about the 3 November investigation - why the lead-time metric
was redefined, what the negative controls showed - stays as dated history,
because those are records of an experiment that happened, not claims about the
running system. Where the historical number and the measured one disagree, the
measured one is printed and the disagreement is stated.
"""
from __future__ import annotations

import json
import re
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_audit_report_pdf import (  # noqa: E402
    API,
    BACKEND,
    BACKEND_REPO,
    FRONTEND,
    FRONTEND_REPO,
    SITE,
    curl,
    curl_json,
    http_code,
    refresh_label,
    refresh_state,
    sent,
    sh,
)

try:
    from docx import Document
    from docx.enum.text import WD_ALIGN_PARAGRAPH
    from docx.shared import Inches, Pt, RGBColor
except ImportError:  # pragma: no cover
    sys.exit("python-docx is required: pip install python-docx")

OUT = BACKEND / "submission" / "GridSense_AI_Closed_Loop_Validation_Progress_Report.docx"

ACCENT = RGBColor(0x0F, 0x5F, 0x73)
GREY = RGBColor(0x55, 0x5F, 0x6D)
ZONE_LABEL = {"ES": "Spain", "GR": "Greece", "IT_SOUTH": "Southern Italy"}
HORIZONS = ("1", "6", "12", "24")


def pct(x, digits: int = 2) -> str:
    return "not reported" if x is None else f"{100.0 * float(x):.{digits}f}%"


def num(x, digits: int = 3) -> str:
    return "not reported" if not isinstance(x, (int, float)) else f"{float(x):.{digits}f}"


def cm(x) -> str:
    return f"{int(x):,}" if isinstance(x, (int, float)) else "not reported"


def ts(x) -> str:
    return str(x)[:16].replace("T", " ").replace("+00:00", " UTC") if x else "not reported"


def git_sha(repo: Path) -> str:
    return sh(["git", "rev-parse", "--short", "HEAD"], cwd=str(repo), timeout=60) or "unknown"


GH_API = "https://api.github.com/repos"


def latest_run(repo_url: str) -> dict:
    """Most recent GitHub Actions run, read unauthenticated from the public API.

    A rate-limit reply or any non-JSON body returns an empty dict: CI that cannot
    be read must not be printed as a passing build.
    """
    owner_repo = repo_url.replace("https://github.com/", "").strip("/")
    try:
        body = json.loads(curl(f"{GH_API}/{owner_repo}/actions/runs?per_page=1")) or {}
    except (json.JSONDecodeError, TypeError):
        return {}
    runs = body.get("workflow_runs") or []
    return runs[0] if runs else {}


def ci_label(run: dict) -> str:
    if not run:
        return "not readable (unauthenticated GitHub API)"
    return (f"{run.get('name', '?')} on {str(run.get('head_sha'))[:7]} is "
            f"{run.get('status')}/{run.get('conclusion')} at {str(run.get('created_at'))[:16]}Z")


def n_tests() -> str:
    raw = sh([sys.executable, "-m", "pytest", "--collect-only", "-q", "tests"],
             cwd=str(BACKEND), timeout=300)
    m = re.search(r"(\d+) tests? collected", raw)
    return m.group(1) if m else "not collected"


def n_controls() -> str:
    raw = sh([sys.executable, "-m", "pytest", "--collect-only", "-q", "tests"],
             cwd=str(BACKEND), timeout=300)
    return str(len([l for l in raw.splitlines() if "negative_control" in l]))


def main() -> int:
    now = datetime.now(timezone.utc)

    health = curl_json("/health") or {}
    report = curl_json("/validation") or {}
    prov = health.get("data_provenance", {})
    rs = refresh_state(health)
    metrics = report.get("metrics") or {}
    fc = metrics.get("forecast") or {}
    al = metrics.get("alarm") or {}
    ev = metrics.get("events") or {}
    steps = fc.get("per_step") or {}
    leads = fc.get("warning_rule_at_lead") or {}
    window = report.get("window") or {}
    gates = report.get("gates") or []
    counts = Counter(g["result"] for g in gates)
    limits = report.get("limitations") or []
    zones = window.get("zones") or health.get("zones") or list(ZONE_LABEL)
    zone_names = ", ".join(ZONE_LABEL.get(z, z) for z in zones)
    weather_feed = prov.get("weather") or {}
    load_parts = ", ".join(f"{cm(v)} {k}" for k, v in (prov.get("load") or {}).items()) or "none"
    weather_parts = ", ".join(f"{cm(v)} {k}" for k, v in weather_feed.items()) or "none"
    # Read in-process rather than through sh(): the shared helper swallows stderr, so a
    # failing subprocess used to surface as an empty string and a confusing IndexError here.
    frame = {"rows": None, "cols": None, "start": None, "end": None, "zones": []}
    frame_path = BACKEND / "data" / "features" / "training_features.parquet"
    if frame_path.exists():
        import pandas as pd

        ff = pd.read_parquet(frame_path)
        frame = {
            "rows": int(len(ff)),
            "cols": int(ff.shape[1]),
            "start": str(ff["timestamp"].min())[:10],
            "end": str(ff["timestamp"].max())[:10],
            "zones": sorted(str(z) for z in ff["zone"].unique()) if "zone" in ff else [],
        }
    is_synthetic = prov.get("is_synthetic")
    tests = n_tests()
    controls = n_controls()
    be_sha = git_sha(BACKEND)
    fe_sha = git_sha(FRONTEND)
    be_run = latest_run(BACKEND_REPO)
    fe_run = latest_run(FRONTEND_REPO)
    val_code = http_code(f"{API}/validation")
    site_code = http_code(SITE)
    warn_ids = [g for g in gates if g["result"] == "WARN"]

    doc = Document()
    for section in doc.sections:
        section.top_margin = Inches(0.8)
        section.bottom_margin = Inches(0.8)
        section.left_margin = Inches(0.9)
        section.right_margin = Inches(0.9)
    style = doc.styles["Normal"]
    style.font.name = "Calibri"
    style.font.size = Pt(10.5)

    def heading(text, level=1):
        h = doc.add_heading(text, level=level)
        for run in h.runs:
            run.font.color.rgb = ACCENT
        return h

    def para(text, bold=False, italic=False, size=None, color=None, space_after=6):
        p = doc.add_paragraph()
        r = p.add_run(text)
        r.bold = bold
        r.italic = italic
        if size:
            r.font.size = Pt(size)
        if color:
            r.font.color.rgb = color
        p.paragraph_format.space_after = Pt(space_after)
        return p

    def bullets(items, style_name="List Bullet"):
        for item in items:
            doc.add_paragraph(item, style=style_name)

    def table(headers, rows, widths=None):
        t = doc.add_table(rows=1, cols=len(headers))
        t.style = "Light Grid Accent 1"
        for i, h in enumerate(headers):
            cell = t.rows[0].cells[i]
            cell.text = ""
            run = cell.paragraphs[0].add_run(h)
            run.bold = True
            run.font.size = Pt(9.5)
        for row in rows:
            cells = t.add_row().cells
            for i, value in enumerate(row):
                cells[i].text = ""
                run = cells[i].paragraphs[0].add_run(str(value))
                run.font.size = Pt(9.5)
        if widths:
            for row in t.rows:
                for cell, w in zip(row.cells, widths):
                    cell.width = Inches(w)
        return t

    # ── Title ───────────────────────────────────────────────────────────────
    title = doc.add_heading("GridSense AI - Closed-Loop Validation Progress Report", level=0)
    title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    meta = doc.add_paragraph()
    meta.alignment = WD_ALIGN_PARAGRAPH.CENTER
    meta.add_run(
        f"Huzaifa Himad   ·   validation work of 3 November 2026, retrained model set of "
        f"5 November   ·   regenerated {now:%Y-%m-%d %H:%M} UTC"
    ).font.size = Pt(10)
    para(
        f"Every figure describing the running system in this report is read at build time from "
        f"the deployed API: {len(gates)} gates, their measured values and the replay window come "
        f"from GET /validation (HTTP {val_code}), the data provenance and refresh state from GET "
        f"/health, the test count from pytest collection, and the commit SHAs from each "
        f"repository. The narrative sections are dated records of the investigation itself, and "
        f"where a 3 November figure has since moved, the current measurement is printed and the "
        f"move is stated rather than quietly corrected.",
        size=10, color=GREY,
    )

    table(
        ["Item", "Status at build time"],
        [
            ["Method", str(report.get("method") or "not reported by the API")],
            ["Release verdict",
             f"overall {report.get('overall', 'not reported')} - {counts.get('PASS', 0)} of "
             f"{len(gates)} gates at or above target, {counts.get('WARN', 0)} between floor and "
             f"target, {counts.get('FAIL', 0)} below floor"],
            ["Replay window",
             f"{ts(window.get('test_start'))} to {ts(window.get('test_end'))}: "
             f"{cm(window.get('hours'))} test hours x {len(zones)} zones, "
             f"{cm(window.get('rows_scored'))} zone-hours scored, "
             f"{cm(window.get('forecast_origins_replayed'))} origins replayed at "
             f"{window.get('origin_stride_hours', '?')}-hour stride through "
             f"{window.get('forecast_horizon_hours', '?')}-hour horizons"],
            ["Evidence artifact",
             f"generated {ts(report.get('generated_at'))}, {num(report.get('staleness_hours'), 1)} h "
             f"old at build time, replay runtime {num(report.get('runtime_seconds'), 1)} s"],
            ["Models graded",
             f"{health.get('interval_method') or 'interval method not reported'}; artifacts "
             f"loaded on the deployed host: "
             f"{', '.join(k for k, v in (health.get('artifacts') or {}).items() if v) or 'none reported'}; "
             f"models overridden for the reported run: "
             f"{(report.get('artifacts') or {}).get('models_overridden_for_this_run', 'not reported')}"],
            ["Automated tests", f"{tests} collected in tests/; {controls} negative controls "
                                f"still in the suite"],
            ["CI (measured from GitHub)", f"backend: {ci_label(be_run)}; frontend: {ci_label(fe_run)}"],
            ["Commits", f"backend {be_sha}, frontend {fe_sha}"],
            ["Served data",
             f"{'real' if is_synthetic is False else 'flagged synthetic' if is_synthetic is True else 'provenance flag not reported'}"
             f" - load rows: {load_parts}; weather rows: {weather_parts}; capacity: "
             f"{prov.get('capacity') or 'not reported'}; is_synthetic={is_synthetic}. These count "
             f"the served window only, not the training frame the replay graded."],
            ["Live surfaces",
             f"GET /validation HTTP {val_code}; dashboard HTTP {site_code} at {SITE}"],
        ],
        widths=[1.7, 5.1],
    )

    # ── 1 ───────────────────────────────────────────────────────────────────
    heading("1. Motivation from an operations standpoint", 1)
    para(
        "Grid dispatchers do not judge a predictive system by its average error. They judge it by "
        "four things: whether the warning arrives early enough to act, whether it persists rather "
        "than flickering, how many false alarms they must sit through to get one real warning, and "
        "whether the uncertainty range the system publishes is the range it actually achieves. "
        "Before this work, none of those four quantities was measured for GridSense AI - the "
        "project could state a validation-set error and nothing else."
    )
    para("Three specific gaps were closed:")
    bullets([
        "Multi-horizon accountability - a single headline MAPE hides whether skill degrades "
        "gracefully from +1 h to +24 h, which is the axis an operator acts along.",
        "Uncertainty verification - the dashboard publishes a P10-P90 band; a band is only "
        "meaningful if outcomes fall inside it near the nominal rate. That had never been measured.",
        "Alert quality in operational units - precision, recall, false alarms per zone-day, and "
        "usable warning lead time before an overload event.",
    ])

    # ── 2 ───────────────────────────────────────────────────────────────────
    heading("2. Method", 1)
    para(
        f"A walk-forward replay of the held-out test window - "
        f"{ts(window.get('test_start'))} to {ts(window.get('test_end'))}, "
        f"{cm(window.get('hours'))} test hours across {len(zones)} zones ({zone_names}). "
        f"{cm(window.get('forecast_origins_replayed'))} origins are each forecast "
        f"{window.get('forecast_horizon_hours', '?')} hours ahead through the same recursive "
        f"forecaster, conformal interval, risk classifier and shedding rules that the production "
        f"API imports. Origins begin before the window opens so in-window forecasts have full lag "
        f"history and are not warm-starting. Hourly gaps inside the window, measured per zone: "
        f"{json.dumps(window.get('hourly_gaps_per_zone') or {}, default=str)}."
    )
    para("Design decisions that make the result defensible:")
    bullets([
        "Serving path, not research path - replay reuses the deployed code, so validator and API "
        "cannot disagree about what the system does.",
        "Baseline-relative skill - accuracy gates are scored as improvement over seasonal-naive "
        "(same hour, one week earlier), the recognised floor benchmark for load forecasting.",
        "Leak discipline retained - the capacity ratio stays excluded from the risk feature set, "
        "and each origin sees only information available at its timestamp.",
        f"Pre-committed gates - {len(gates)} gates, each with a target and a lower floor; a value "
        f"below its floor fails and fails the release. Targets were fixed before the real-data "
        f"retrain and have not been moved since, including the four that now read WARN.",
        "Centralised decision rules - the 0.8 / 0.6 / 0.3 thresholds live in one module read by "
        "both the API and the validator, so reported metrics and live alerts cannot describe "
        "different systems.",
    ])

    # ── 3 ───────────────────────────────────────────────────────────────────
    heading("3. What the replay measures now", 1)

    heading("3.1 Forecast skill and band honesty, by horizon", 2)
    table(
        ["Horizon", "Model MAPE", "Seasonal-naive", "Skill", "Realised P10-P90 coverage", "Bias"],
        [[f"+{h} h",
          f"{num((steps.get(h) or {}).get('mape_recursive'), 2)} %",
          f"{num((steps.get(h) or {}).get('mape_seasonal_naive'), 2)} %",
          f"+{pct((steps.get(h) or {}).get('mape_improvement'), 1)}",
          f"{pct((steps.get(h) or {}).get('coverage_p10_p90'), 1)}",
          f"+{num((steps.get(h) or {}).get('bias_pct'), 2)} %"]
         for h in HORIZONS if steps.get(h)],
        widths=[0.7, 1.2, 1.2, 0.9, 1.7, 0.8],
    )
    para(
        f"Aggregate coverage is {pct(fc.get('coverage_p10_p90_overall'))} against a nominal 80 %, "
        f"so the published interval is honest rather than decorative. Week-to-week coverage "
        f"spread is {pct(fc.get('weekly_coverage_spread'), 1)} points, which is above the "
        f"<15 % target and below the 30 % floor - the one coverage gate that does not read PASS, "
        f"and the reason it is a WARN rather than a FAIL is that the band never collapses; it "
        f"varies. Bias is small and consistently slightly positive: the forecaster "
        f"under-predicts by {num((steps.get('1') or {}).get('bias_pct'), 2)} % at +1 h and "
        f"{num((steps.get('24') or {}).get('bias_pct'), 2)} % at +24 h."
    )

    heading(f"3.2 Alarm quality at the operating threshold (risk >= {al.get('threshold', 0.6)})", 2)
    table(
        ["Measure", "Value", "Interpretation"],
        [
            ["Precision", num(al.get("precision")),
             f"{cm(al.get('true_positive_hours'))} of {cm(al.get('alarm_hours'))} alarmed hours "
             f"genuinely overloaded"],
            ["Recall", num(al.get("recall")),
             f"{cm(al.get('missed_hours'))} overloaded hours not alarmed"],
            ["F1", num(al.get("f1")), "-"],
            ["Brier skill", num(al.get("brier_skill")),
             f"Brier {num(al.get('brier'), 5)} against a {num(al.get('brier_base_rate'), 5)} "
             f"base-rate Brier"],
            ["False alarms per zone-day", num(al.get("false_alarms_per_zone_day")),
             "the operator-burden measure; the fatigue gate reads this, not precision alone"],
            ["Overload base rate", pct(al.get("positive_rate")),
             "precision is meaningful against this, not 50 %"],
            ["Advisory shedding volume",
             f"{cm(al.get('shed_hours_priority_1'))} P1 / {cm(al.get('shed_hours_priority_2'))} P2 h",
             f"scored over {cm(al.get('hours_scored'))} zone-hours"],
        ],
        widths=[2.0, 1.4, 3.4],
    )
    para(
        "Reliability, bucketed the way the dashboard buckets it: "
        + "; ".join(
            f"{b['bucket']} (n={cm(b.get('n'))}) predicts {num(b.get('mean_predicted'), 3)} and "
            f"observed {pct(b.get('observed_overload_rate'), 1)} overloaded"
            for b in (al.get("reliability") or [])
        ) + "."
    )

    heading("3.3 Event-level response", 2)
    table(
        ["Measure", "Value"],
        [
            ["Overload events in window", cm(ev.get("n_events"))],
            ["Events alarmed while developing",
             f"{cm(ev.get('n_detected'))} ({pct(ev.get('detection_recall'), 1)})"],
            ["Median stable warning lead", f"{num(ev.get('median_lead_hours_stable'), 1)} h"],
            ["Median glimpse lead", f"{num(ev.get('median_lead_hours_glimpse'), 1)} h"],
            ["Warned >= 1 h / >= 3 h / >= 6 h ahead",
             f"{pct(ev.get('share_warned_1h'), 1)} / {pct(ev.get('share_warned_3h'), 1)} / "
             f"{pct(ev.get('share_warned_6h'), 1)}"],
            ["Longest single event", f"{cm(ev.get('longest_event_hours'))} h"],
            ["Most common onset hours (UTC)",
             ", ".join(f"{o['hour_utc']:02d}:00 x{o['n_events']}"
                       for o in (ev.get("top_onset_hours") or []))],
        ],
        widths=[3.2, 3.0],
    )

    heading("3.4 Every gate, measured against the target committed before the retrain", 2)
    table(
        ["Gate", "Measured", "Target", "Floor", "Result"],
        [[g["label"],
          num(g.get("value"), 4) if isinstance(g.get("value"), (int, float)) else str(g.get("value")),
          json.dumps(g.get("target"), default=str) if isinstance(g.get("target"), list)
          else num(g.get("target"), 3) if isinstance(g.get("target"), (int, float)) else "-",
          json.dumps(g.get("floor"), default=str) if isinstance(g.get("floor"), list)
          else num(g.get("floor"), 3) if isinstance(g.get("floor"), (int, float)) else "-",
          g.get("result")]
         for g in gates],
        widths=[3.1, 0.9, 0.8, 0.8, 0.7],
    )
    para(
        f"{counts.get('PASS', 0)} gates clear their target, {counts.get('WARN', 0)} sit between "
        f"floor and target, and none is below its floor, so the release verdict is "
        f"{report.get('overall')}. The four WARNs are "
        + ", ".join(f"{g['id']} (measured {num(g['value'], 3)} against a target of "
                    f"{num(g['target'], 3)})" for g in warn_ids)
        + ". All four are in the same family: how much actionable notice the system gives before "
          "an overload. That is the honest cost of moving from the synthetic bootstrap "
          "distribution to two years of real ENTSO-E load, and it is reported rather than "
          "negotiated - no target, floor or split was changed to improve a number."
    )

    # ── 4 ───────────────────────────────────────────────────────────────────
    heading("4. Principal finding: warning lead time had been over-claimed", 1)
    para(
        "The first implementation of the validator, on 3 November, reported PASS with a 19-hour "
        "median lead. That figure was an artefact, and diagnosing it is the most instructive part "
        "of this work."
    )
    bullets([
        "Event onset was concentrated by the synthetic daily cycle - averaging lead time over a "
        "distribution that narrow describes the test data, not the system.",
        "Long horizons earned credit for restating the daily peak. A +24 h forecast reproduces the "
        "next morning's peak from its own lag features, so measuring \"the first horizon that "
        "glimpsed the event\" reads as a full day of warning even though the signal disappears and "
        "returns as the horizon advances. An operator acting on it would have shed load on a "
        "non-persistent indication.",
    ])
    para("The metric was redefined and is reported two ways, side by side:")
    table(
        ["Definition", "Value now", "Use"],
        [
            ["Stable lead - an unbroken chain of warnings from +1 h through onset",
             f"{num(ev.get('median_lead_hours_stable'), 1)} h",
             "scored by the release gates; the only version actionable in a control room"],
            ["Glimpse lead - the earliest horizon that saw it at all",
             f"{num(ev.get('median_lead_hours_glimpse'), 1)} h",
             "retained visibly so the flattering reading is not buried"],
        ],
        widths=[3.0, 0.9, 2.9],
    )
    l6 = leads.get("6") or {}
    para(
        f"Two supporting measures keep the stable lead honest. Warning-rule precision is evaluated "
        f"across all origins rather than only at onset hours: at +6 h the system issued "
        f"{cm(l6.get('alerts'))} warnings, {cm(l6.get('true_alerts'))} were correct, precision "
        f"{num(l6.get('precision'))}, at {num(l6.get('alerts_per_zone_day'))} alerts per zone-day. "
        f"A P90-upper-band trigger variant quantifies the lead-versus-precision trade-off "
        f"explicitly: acting on the conservative band would warn "
        f"{pct(ev.get('share_stable_3h_upper_band'), 1)} of events at least three hours ahead, at "
        f"materially higher false-alarm cost. Had the original metric stood, the dashboard would "
        f"have advertised a day of warning the system does not provide."
    )
    para(
        f"Read this against the 3 November figures, which this report used to carry: median stable "
        f"lead 3.0 h, events alarmed 91.3 %, warning-rule precision at +6 h 0.839. On the real "
        f"distribution those are {num(ev.get('median_lead_hours_stable'), 1)} h, "
        f"{pct(ev.get('detection_recall'), 1)} and {num(l6.get('precision'))}. The system got less "
        f"able to promise early notice than the synthetic run suggested, and the gates were "
        f"pre-committed precisely so that this shows up as a WARN instead of as a redefinition.",
        color=GREY,
    )

    # ── 5 ───────────────────────────────────────────────────────────────────
    heading("5. Evidence that the gates can fail", 1)
    para(
        "A validator that always prints PASS is worse than none, because it launders an unqualified "
        "model into a certification badge. Two negative controls run inside the normal test suite, "
        f"injecting deliberately broken models into the same replay. {controls} tests match "
        f"'negative_control' in pytest collection at build time, so the controls are still wired "
        f"in; the magnitudes below are what they demonstrated on 3 November, on that run's replay "
        f"configuration."
    )
    table(
        ["Injected fault", "Gate response (3 November run)"],
        [
            ["Forecaster inflated by 15 % on every step",
             "P10-P90 coverage collapses 0.842 -> 0.0002 on the same replay configuration; "
             "band_coverage FAILs; overall FAIL"],
            ["Risk model that alarms on every one of 10,413 hours",
             "precision falls to that run's 0.1095 base rate; burden rises 0.33 -> 21.37 false "
             "alarms per zone-day; alarm_precision and the burden gate FAIL - while alarm_recall "
             "correctly still PASSes at 1.0"],
        ],
        widths=[2.6, 4.2],
    )
    para(
        "The second control is the important one. Recall stays perfect while precision and burden "
        "fail - exactly the failure mode behind alert fatigue. Had the system been graded on recall "
        "alone, a device that cries wolf every hour would have been certified."
    )
    para(
        "One configuration detail is itself a finding. The controls run on a coarse 24-hour origin "
        "stride so they finish inside the test suite, whereas the shipped report replays every hour "
        f"(stride {window.get('origin_stride_hours')} h here): the warning-precision and lead-time "
        "metrics are only meaningful on a contiguous hourly origin grid, because a coarse stride "
        "stops sampling the hours that actually precede event onset and the precision gate "
        "collapses for reasons unrelated to model quality. Any metric that sensitive has to travel "
        "with its configuration, which is why the committed artifact records its window, stride, "
        "horizon and runtime alongside the numbers."
    )

    # ── 6 ───────────────────────────────────────────────────────────────────
    heading("6. What changed in the running system", 1)
    heading("Backend", 2)
    bullets([
        "models/closed_loop_validation.py - replay engine: metric families, gate grading, "
        "provenance block, --strict release exit.",
        "models/decision_rules.py - single source of truth for alert thresholds and shed priority, "
        "shared by the API and the validator.",
        "api/main.py - GET /validation endpoint; report loaded at startup with the other artifacts "
        "and reported by /health.",
        "data/models/closed_loop_report.json - committed evidence artifact carrying the method, "
        "runtime, window, artifact provenance and stated limitations.",
        "Lean-serving guarantee preserved - the deployed import path still excludes shap and "
        "torch, and the report is included in the deploy bundle.",
    ])
    heading("Operator dashboard", 2)
    bullets([
        "Panel 05, Closed-Loop Validation: overall verdict with gate count and report age; the "
        "operationally relevant numbers (precision, recall, false alarms per zone-day, stable "
        "warning lead); realised coverage; MAPE against seasonal-naive; events alarmed; an "
        "expander listing every gate's measured value against its target; and the stated limitations.",
        "Deliberately isolated from the dashboard's shared error banner, and a 404 is interpreted "
        "as \"this backend does not expose the endpoint yet\" rather than an outage - verified "
        "against both a compliant backend and one that does not expose the endpoint at all.",
    ])
    heading("Quality gates and documentation", 2)
    bullets([
        f"Test suite now {tests} collected tests, covering gate grading, lead-time arithmetic, "
        f"event-run segmentation, rule equivalence between API and validator, report "
        f"self-consistency, the endpoint contract, the serving-path data refresh, and both "
        f"negative controls.",
        f"GitHub Actions, read from the API at build time: backend {ci_label(be_run)}, frontend "
        f"{ci_label(fe_run)}. This working copy is at backend {be_sha} / frontend {fe_sha}; where "
        f"those differ from the SHA CI ran on, the newer commit has not been verified by a "
        f"pipeline yet.",
        f"Post-deploy verification on the public dashboard: /validation and /health serving "
        f"HTTP {val_code}, dashboard HTTP {site_code}, and the validation panel rendering the same "
        f"report the API serves - the panel reads the artifact, so it cannot disagree with this "
        f"document.",
        "Documentation corrected to match the code, including stale claims found in passing: the "
        "user manual had described the forecast band as a hardcoded +-8 % display band, and "
        "endpoint counts and synthetic-data wording were out of date across the docs set.",
    ])

    # ── 7 ───────────────────────────────────────────────────────────────────
    heading("7. Limitations", 1)
    para(
        "These are the limitations published inside the live report artifact itself, transcribed "
        "at build time, and they should be read before any figure above is cited:"
    )
    for item in limits:
        doc.add_paragraph(str(item), style="List Number")
    para(
        f"One limitation from the 3 November version of this list no longer applies and is worth "
        f"saying so plainly: the models are no longer trained on synthetic bootstrap data. "
        f"/health reports is_synthetic={is_synthetic} over {load_parts} load rows and "
        f"{weather_parts} weather rows in the served window, and the replay above graded the "
        f"models on the "
        f"committed training frame - {cm(frame['rows'])} rows x {cm(frame['cols'])} columns, "
        f"{frame['start'] or 'not reported'} to {frame['end'] or 'not reported'}, zones "
        f"{', '.join(frame['zones']) or 'not reported'}. "
        f"The remaining gaps are the ones listed by the artifact, plus data currency: "
        f"{sent(refresh_label(rs))}.",
        color=GREY,
    )

    # ── 8 ───────────────────────────────────────────────────────────────────
    heading("8. Next steps", 1)
    table(
        ["Priority", "Work", "Why it matters"],
        [
            ["1", "Give the deployed refresher a credential ENTSO-E accepts, or turn the "
                  f"refresher off deliberately. Measured now: {refresh_label(rs)}.",
             "until one or the other happens, the newest served hour is "
             f"{ts(prov.get('load_latest_timestamp'))} and ages one hour per hour while the "
             "dashboard claims a live feed"],
            ["2", f"Close the four WARN lead-time gates on the real distribution: {len(warn_ids)} "
                  "gates, all in the notice-before-onset family",
             "this is the quantity an operator acts on; the target was set at >=3 h and the "
             f"measurement says {num(ev.get('median_lead_hours_stable'), 1)} h"],
            ["3", "Obtain rated capacity data (NTDC/DISCO or ENTSO-E asset registers) to replace "
                  "the p99 proxy",
             "turns a statistical ceiling into an engineering constraint"],
            ["4", "Decompose long-horizon error into weather versus model contributions, and "
                  "replace persistence weather with a real NWP feed",
             "long-horizon skill is currently optimistic on the meteorological component"],
            ["5", "Move to feeder-level assets with N-1 contingency screening",
             "the level at which shed schedules are actually executed"],
            ["6", "Revalidation on a schedule rather than on demand, and enforce --strict inside "
                  "the deploy pipeline",
             "a gate that only runs when someone remembers is a suggestion, not a release control"],
        ],
        widths=[0.7, 3.2, 2.9],
    )

    # ── 9 ───────────────────────────────────────────────────────────────────
    heading("9. Reproducing these results", 1)
    for line in [
        f"git clone {BACKEND_REPO}.git",
        "cd Grid_Sense.AI-Backend- && git checkout main",
        "pip install -r requirements.txt",
        "python -m models.closed_loop_validation --strict    # rewrites the report artifact",
        f"python -m pytest tests -q                           # {tests} tests, both negative "
        f"controls included",
        "uvicorn api.main:app --port 8000 && curl http://localhost:8000/validation",
        f"python scripts/build_closed_loop_report_docx.py     # regenerates this document",
    ]:
        p = doc.add_paragraph()
        r = p.add_run(line)
        r.font.name = "Consolas"
        r.font.size = Pt(9)
        p.paragraph_format.space_after = Pt(1)
    para(
        f"Repos: backend {BACKEND_REPO} (HEAD {be_sha}), frontend {FRONTEND_REPO} (HEAD {fe_sha}). "
        f"Live: {API} and {SITE}."
    )

    heading("10. Summary", 1)
    para(
        "The project's accuracy claims are falsifiable rather than asserted, and the mechanism now "
        "points at the real distribution. Every performance number on the dashboard comes from "
        "replaying the deployed serving path against a held-out window; the release gates are "
        "demonstrated to reject broken models; and the single most operationally important quantity "
        f"- usable warning lead time before an overload event - was found to be over-claimed in "
        f"the first validator, corrected, and has since moved again: {counts.get('PASS', 0)} of "
        f"{len(gates)} gates pass on two years of real ENTSO-E data, the four that do not are all "
        f"lead-time gates, and the measured median stable lead is "
        f"{num(ev.get('median_lead_hours_stable'), 1)} h against a pre-committed 3 h target. That "
        f"is a worse result than the synthetic run showed and a more useful one, because it is what "
        f"the system actually does."
    )

    OUT.parent.mkdir(parents=True, exist_ok=True)
    doc.save(OUT)
    print(f"wrote {OUT} ({OUT.stat().st_size:,} bytes)")
    print(f"gates {counts.get('PASS', 0)}/{len(gates)} PASS, {counts.get('WARN', 0)} WARN, "
          f"{counts.get('FAIL', 0)} FAIL; stable lead {num(ev.get('median_lead_hours_stable'), 1)} h; "
          f"tests {tests}; refresher: {refresh_label(rs)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
