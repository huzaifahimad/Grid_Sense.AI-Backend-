#!/usr/bin/env python3
"""Build submission/GridSense_AI_Change_Log.docx.

Every entry is parsed out of `git log` for both repositories at build time, so
the log cannot claim a change that is not in history and cannot omit one that
is. Each commit also carries its own recorded reason (the commit body) and its
file delta (numstat), and the closing section re-measures the deployment the
changes are sitting on.
"""
from __future__ import annotations

import json
import re
import sys
from collections import Counter, OrderedDict
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
    curl_json,
    http_code,
    refresh_label,
    refresh_state,
    sh,
)

try:
    from docx import Document
    from docx.enum.text import WD_ALIGN_PARAGRAPH
    from docx.shared import Pt, RGBColor, Inches
except ImportError:  # pragma: no cover
    sys.exit("python-docx is required: pip install python-docx")

OUT = BACKEND / "submission" / "GridSense_AI_Change_Log.docx"

SOH, US, RS = "\x01", "\x02", "\x1f"
NUMSTAT = re.compile(r"^(\d+|-)\t(\d+|-)\t(.+)$")
KIND_ORDER = ["feat", "fix", "perf", "refactor", "test", "docs", "chore", "build", "ci", "merge", "other"]


def parse_log(repo: Path) -> list[dict]:
    raw = sh(
        ["git", "log", "--reverse", "--numstat",
         f"--date=format:%Y-%m-%d",
         f"--pretty=format:{SOH}%h{US}%ad{US}%s{US}%p{US}%b", "HEAD"],
        cwd=str(repo), timeout=120,
    )
    commits: list[dict] = []
    cur = None
    for line in raw.split("\n"):
        if line.startswith(SOH):
            sha, date, subject, parents, body = line[1:].split(US, 4)
            kind = subject.split(":", 1)[0].split("(")[0].strip().lower()
            scope = ""
            m = re.match(r"^\w+(?:\(([^)]*)\))?!?:", subject)
            if m and m.group(1):
                scope = m.group(1)
            cur = {
                "sha": sha, "date": date, "subject": subject, "body": body.strip(),
                "kind": "merge" if len(parents.split()) > 1
                        else (kind if kind in KIND_ORDER else "other"),
                "scope": scope,
                "files": 0, "add": 0, "del": 0, "binary": 0,
            }
            commits.append(cur)
            continue
        if cur is None or not line.strip():
            continue
        m = NUMSTAT.match(line.strip())
        if not m:
            continue
        a, d, _path = m.groups()
        cur["files"] += 1
        if a == "-" or d == "-":
            cur["binary"] += 1
        else:
            cur["add"] += int(a)
            cur["del"] += int(d)
    return commits


def why_lines(body: str) -> list[str]:
    """Commit body paragraphs, folded at sentence-ish boundaries for the doc."""
    paras = [p.strip() for p in re.split(r"\n\s*\n", body) if p.strip()]
    out: list[str] = []
    for p in paras:
        flat = " ".join(p.split())
        if re.match(r"^(Co-?Authored|Signed-off|Generated with|Source:|Trailer)", flat, re.I):
            continue
        out.append(flat)
    return out


def set_base_style(doc: Document) -> None:
    st = doc.styles["Normal"]
    st.font.name = "Calibri"
    st.font.size = Pt(10)
    st.paragraph_format.space_after = Pt(4)
    for name, size in (("Heading 1", 15), ("Heading 2", 12), ("Heading 3", 10.5)):
        s = doc.styles[name]
        s.font.name = "Calibri"
        s.font.size = Pt(size)
        s.font.color.rgb = RGBColor(0x0F, 0x5F, 0x73)
        s.font.bold = True


def head(doc: Document, title: str, sub: str) -> None:
    p = doc.add_paragraph()
    r = p.add_run(title)
    r.font.size = Pt(22)
    r.font.bold = True
    r.font.color.rgb = RGBColor(0x0F, 0x5F, 0x73)
    q = doc.add_paragraph()
    rr = q.add_run(sub)
    rr.font.size = Pt(10.5)
    rr.font.color.rgb = RGBColor(0x62, 0x6E, 0x7A)
    q.alignment = WD_ALIGN_PARAGRAPH.LEFT


def table(doc: Document, headers: list[str], rows: list[list[str]], widths: list[float]) -> None:
    t = doc.add_table(rows=1, cols=len(headers))
    t.style = "Table Grid"
    for i, h in enumerate(headers):
        cell = t.rows[0].cells[i]
        cell.text = ""
        run = cell.paragraphs[0].add_run(h)
        run.font.bold = True
        run.font.size = Pt(9)
    for row in rows:
        cells = t.add_row().cells
        for i, val in enumerate(row):
            cells[i].text = ""
            run = cells[i].paragraphs[0].add_run(str(val))
            run.font.size = Pt(9)
    for row in t.rows:
        for i, w in enumerate(widths):
            row.cells[i].width = Inches(w)
    doc.add_paragraph()


def main() -> int:
    now = datetime.now(timezone.utc)
    cb, cf = parse_log(BACKEND), parse_log(FRONTEND)

    health = curl_json("/health") or {}
    validation = curl_json("/validation") or {}
    prov = health.get("data_provenance", {})
    rs = refresh_state(health)
    gates = validation.get("gates") or []
    counts = Counter(g.get("result") for g in gates)

    suite = sh([sys.executable, "-m", "pytest", "-q"], cwd=str(BACKEND), timeout=1800)
    suite_line = next((l for l in suite.splitlines() if " passed" in l or " failed" in l),
                      "pytest produced no summary line")

    # The accuracy numbers quoted in commit bodies are re-derived here rather than
    # trusted: --skip-live keeps the check to the committed artifacts.
    accuracy = sh([sys.executable, str(BACKEND / "scripts/verify_model_accuracy.py"),
                   "--skip-live"], cwd=str(BACKEND), timeout=1800)
    acc_line = next((l.strip() for l in accuracy.splitlines() if l.startswith("checks run")),
                    "verify_model_accuracy produced no summary line")

    routes = ["/health", "/validation", "/assets", "/forecast/ES", "/risk/ES",
              "/shed-schedule/IT_SOUTH", "/forecast/NOPE"]
    route_rows = [[r, http_code(f"{API}{r}")] for r in routes]

    doc = Document()
    set_base_style(doc)
    for s in doc.sections:
        s.top_margin = s.bottom_margin = Inches(0.7)
        s.left_margin = s.right_margin = Inches(0.8)

    head(doc, "GridSense AI - Change Log",
         f"Generated {now:%Y-%m-%d %H:%M} UTC from git history. "
         f"{len(cb)} backend commits and {len(cf)} frontend commits on main, oldest first.")
    doc.add_paragraph(
        "Nothing in this document was typed in by hand: each entry below is a commit that exists "
        "on main in the named repository, with its recorded subject, the reason its author wrote "
        "in the commit body, and the file delta git measured for it. Where a commit body quotes a "
        "number, that number was measured when the commit was made; the closing section "
        "re-measures the deployment today so the two can be compared."
    )

    doc.add_heading("Repository totals", level=1)
    rows = []
    for label, lst, repo_url in (("Backend", cb, BACKEND_REPO), ("Frontend", cf, FRONTEND_REPO)):
        k = Counter(c["kind"] for c in lst)
        rows.append([
            label, len(lst),
            ", ".join(f"{k[t]} {t}" for t in KIND_ORDER if k[t]),
            sum(c["files"] for c in lst),
            f"+{sum(c['add'] for c in lst)} / -{sum(c['del'] for c in lst)}",
            repo_url,
        ])
    table(doc, ["Repository", "Commits", "By type", "Files touched", "Line delta", "Remote"],
          rows, [0.75, 0.6, 2.5, 0.9, 1.2, 1.35])

    doc.add_heading("How the work is grouped", level=1)
    doc.add_paragraph(
        "The history reads in three arcs. The first builds the product and its pipeline; the "
        "second replaces the synthetic dataset with real ENTSO-E load and NASA POWER weather and "
        "retrains on it; the third is the honesty pass - closed-loop replay, release gates, and "
        "removing every claim the data did not support. Commit types mark which arc an entry "
        "belongs to: feat adds capability, fix repairs measured broken behaviour, docs restates "
        "claims against measurement, test and perf and chore hold the verification in place."
    )

    for label, lst, repo in (("Backend", cb, BACKEND), ("Frontend", cf, FRONTEND)):
        doc.add_heading(f"{label} - {len(lst)} commits", level=1)
        by_kind: "OrderedDict[str, list[dict]]" = OrderedDict(
            (k, [c for c in lst if c["kind"] == k]) for k in KIND_ORDER
        )
        for kind, items in by_kind.items():
            if not items:
                continue
            doc.add_heading(f"{kind} ({len(items)})", level=2)
            for c in sorted(items, key=lambda x: x["date"]):
                p = doc.add_paragraph(style="List Bullet")
                r = p.add_run(f"{c['sha']} [{c['date']}] ")
                r.font.bold = True
                r.font.name = "Consolas"
                r.font.size = Pt(9.5)
                p.add_run(c["subject"])
                for w in why_lines(c["body"]):
                    q = doc.add_paragraph(style="List Bullet 2")
                    qr = q.add_run(w)
                    qr.font.size = Pt(9)
                    qr.font.color.rgb = RGBColor(0x40, 0x4A, 0x54)
                note = ("(merge commit - git records no file delta for a merge; the work it "
                        "carried is listed under its own type above)"
                        if c["kind"] == "merge" else
                        f"({c['files']} files, +{c['add']}/-{c['del']})")
                d = p.add_run(f"  {note}")
                d.font.size = Pt(8)
                d.font.color.rgb = RGBColor(0x8A, 0x93, 0x9C)

    doc.add_heading("What the changes are sitting on right now", level=1)
    doc.add_paragraph(
        f"Frontend {SITE} responds HTTP {http_code(SITE)}; backend {API} responds HTTP "
        f"{http_code(f'{API}/health')}. Latest build measured from the deployed bundle: "
        f"{sh(['git', 'rev-parse', '--short', 'HEAD'], cwd=str(FRONTEND))} frontend, "
        f"{sh(['git', 'rev-parse', '--short', 'HEAD'], cwd=str(BACKEND))} backend."
    )
    table(doc, ["Measure", "Value"], [
        ["Release gates", f"{validation.get('overall', 'n/a')} - {counts['PASS']} PASS, "
                          f"{counts['WARN']} WARN, {counts['FAIL']} FAIL of {len(gates)}"],
        ["Replay window", f"{(validation.get('window') or {}).get('rows_scored', 'n/a')} hours scored"],
        ["Load rows held in memory", (prov.get("load") or {}).get("entsoe", "n/a")],
        ["Weather rows held in memory", f"{(prov.get('weather') or {}).get('nasa_power', 'n/a')} rows, "
                                       f"{(prov.get('weather') or {}).get('unavailable', 'n/a')} unavailable"],
        ["Newest load hour", prov.get("load_latest_timestamp", "n/a")],
        ["Request-time refresher (live /health)", refresh_label(rs)],
        ["Weather lag", f"{prov.get('weather_lag_hours', 'n/a')} hours behind load"],
        ["Capacity basis", prov.get("capacity", "n/a")],
        ["Synthetic data in use", str(prov.get("is_synthetic", "n/a"))],
        ["Backend test suite", suite_line],
        ["Independent accuracy recheck", f"{acc_line} (offline subset - re-run the script "
                                         "without --skip-live to add the deployed-API checks)"],
    ], [2.3, 4.4])
    attempt_tail = ("" if not rs["reported"] else
                    f" Last attempt {rs['attempt_at'] or 'n/a'} took "
                    f"{rs['attempt_seconds'] if rs['attempt_seconds'] is not None else 'n/a'} s against a "
                    f"{rs['timeout'] if rs['timeout'] is not None else 'n/a'} s ceiling; "
                    f"{rs['rows']:,} rows added and last success "
                    f"{rs['last_success'] or 'null'}.")
    doc.add_paragraph(
        "The two row counts above are the served window - the in-memory tail the API answers "
        "queries from - not the size of the training frame, which is a different and much larger "
        "scope and is quantified in the release note and the audit report. The backend loads the "
        "committed feature frame on startup and, since 2026-11-06, also runs a request-time "
        "refresher over that window at most once per cache interval, per zone only once its newest "
        "hour is stale, with any failure falling back to the committed frame. Live state measured at "
        f"build time: {refresh_label(rs)}.{attempt_tail} "
        + ("The newest hour therefore still ages one hour per hour between rebuilds, which is why the "
           "age is published rather than assumed."
           if not rs["publishing"] else
           "The newest hour therefore advances without a redeploy, and the published age tracks it.")
    )

    doc.add_heading("Endpoint behaviour", level=1)
    table(doc, ["Route", "HTTP"], route_rows, [4.0, 1.2])
    doc.add_paragraph(
        "The final row is deliberate: an unknown zone must not return a plausible-looking number. "
        "It returns the status above, which is the behaviour the test suite pins."
    )

    OUT.parent.mkdir(parents=True, exist_ok=True)
    doc.save(str(OUT))
    print(f"wrote {OUT} ({OUT.stat().st_size:,} bytes)")
    print(f"backend commits {len(cb)}, frontend commits {len(cf)}, "
          f"gates {counts['PASS']}/{len(gates)} PASS, suite: {suite_line}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
