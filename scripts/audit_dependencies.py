"""Dependency advisory audit against the OSV database.

Why this exists rather than `pip-audit`: `pip-audit` resolves and downloads
distributions, which needs the full install tree; this checks the exact pinned
versions this repo declares, which is the question that actually matters for a
deployment — "is the version we ship affected by a published advisory".

Exit codes, deliberately distinct:
  0  every package in scope was answered by OSV and none is affected
  1  an advisory, or a dependency specifier that is not an exact `==` pin
  2  the audit could not be completed (no network, unexpected response)

Exit 2 is not optional. A security check that reports success because it could not
reach the advisory database is worse than no check at all, because the CI badge would
read green while proving nothing.

Scope defaults to pyproject.toml — the set the deployed API installs. Pass --all to also
audit requirements.txt, which carries the training and ingestion extras that the serving
path never imports. CI runs the default scope as a blocking gate; --all is a local
sweep because requirements.txt is not installed in CI.

--installed-drift is a separate question, so it has its own mode and needs no network:
it compares the declared pins against what this interpreter actually has, which is the
only way to know whether committed artifacts were reproduced on the versions we claim.
Its exit code is 0 = no drift, 1 = drift present, and it is deliberately NOT wired into
CI — CI installs the pins, so it would always report a false clean.

Measured 2026-11-06, the day this was written: the serving set returned 0 advisories on all
9 packages. In the extras, `requests` scored 2 distinct CVEs across 4 advisories at the
then-declared 2.32.3, and 6 advisories at the 2.31.0 actually installed on the machine that
produced the committed artifacts. After the remediation in the same session — requests
2.34.2, torch 2.15.0 and kaggle 2.2.5 pinned exactly, entsoe-py 0.8.0 to match the version
that performed the real pull — both scopes re-run to 0 findings and exit 0.
"""
from __future__ import annotations

import argparse
import importlib.metadata
import json
import re
import sys
import tomllib
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OSV_URL = "https://api.osv.dev/v1/query"
PINNED = re.compile(r"([A-Za-z0-9_.-]+)(\[[^\]]*\])?==([A-Za-z0-9_.+!-]+)")


def parse_specifiers(text: str) -> tuple[list[tuple[str, str]], list[str]]:
    """Split `name==version` lines into pins, and everything else into findings."""
    pins, unpinned = [], []
    for raw in text.splitlines():
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        m = PINNED.fullmatch(line)
        if m:
            pins.append((m.group(1).lower(), m.group(3)))
        else:
            unpinned.append(line)
    return pins, unpinned


def osv_advisories(name: str, version: str) -> list[dict]:
    query = json.dumps(
        {"package": {"name": name, "ecosystem": "PyPI"}, "version": version}
    ).encode()
    req = urllib.request.Request(
        OSV_URL,
        data=query,
        headers={"content-type": "application/json", "user-agent": "gridsense-audit"},
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.loads(resp.read()).get("vulns") or []


def summarize(vuln: dict) -> str:
    cves = [a for a in (vuln.get("aliases") or []) if a.startswith("CVE-")]
    label = cves[0] if cves else vuln.get("id", "?")
    for aff in vuln.get("affected", []):
        fixed = (aff.get("ecosystem_specific") or {}).get("fixed_version")
        if fixed:
            return f"{label} (fixed in {fixed})"
    return label


def report_installed_drift(scopes: list[tuple[str, list[tuple[str, str]]]]) -> int:
    """Compare declared pins against what this interpreter actually resolved."""
    drifted = 0
    print(f"Installed-vs-declared drift - python {sys.version.split()[0]}\n")
    for label, pins in scopes:
        print(f"[{label}]")
        for name, version in sorted(pins):
            base = name.split("[", 1)[0]
            try:
                got = importlib.metadata.version(base)
            except importlib.metadata.PackageNotFoundError:
                got = None
            if got == version:
                print(f"  ok      {base}=={version}")
            else:
                drifted += 1
                print(f"  DRIFT   {base}: declared {version}, "
                      f"installed {got or 'ABSENT'}")
    print(f"\n{drifted} drifted package(s).")
    if drifted:
        print("Artifacts reproduced on this machine were built against the installed "
              "versions, NOT the declared pins.")
    return drifted


def main() -> int:
    ap = argparse.ArgumentParser(description="Audit declared dependency pins against OSV.")
    ap.add_argument("--all", action="store_true",
                    help="also audit requirements.txt (training/ingestion extras)")
    ap.add_argument("--unreachable-is-warning", action="store_true",
                    help="exit 0 on a network failure instead of 2, for local use only")
    ap.add_argument("--installed-drift", action="store_true",
                    help="compare declared pins against this interpreter; no network needed")
    args = ap.parse_args()

    deps = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]["dependencies"]
    scopes = [("serving (pyproject.toml)", *parse_specifiers("\n".join(deps)))]
    unpinned = list(scopes[0][2])
    if args.all:
        extra = parse_specifiers((ROOT / "requirements.txt").read_text(encoding="utf-8"))
        scopes.append(("extras (requirements.txt)", *extra))
        unpinned += extra[1]

    if args.installed_drift:
        return 1 if report_installed_drift([(label, pins) for label, pins, _ in scopes]) else 0

    findings = 0
    print(f"OSV dependency audit - scope: {'serving + extras' if args.all else 'serving'}")
    for label, pins, _ in scopes:
        print(f"\n[{label}]")
        for name, version in sorted(pins):
            try:
                vulns = osv_advisories(name, version)
            except (urllib.error.URLError, TimeoutError, OSError) as exc:
                print(f"\nCANNOT AUDIT {name}=={version}: {exc}")
                print("OSV was unreachable, so no advisory claim below is verified.")
                return 0 if args.unreachable_is_warning else 2
            if vulns:
                findings += 1
                # GHSA and PYSEC routinely publish separate entries for one CVE, so the
                # useful count is distinct vulnerabilities, not rows returned.
                distinct = sorted({summarize(v) for v in vulns})
                print(f"  ADVISORY {name}=={version}: "
                      f"{len(distinct)} distinct ({', '.join(distinct[:6])}) "
                      f"across {len(vulns)} advisories")
            else:
                print(f"  ok       {name}=={version}")

    for line in sorted(set(unpinned)):
        findings += 1
        print(f"\n  UNPINNED specifier: {line}  "
              "(requirements.txt states exact pins are required, on purpose)")

    print(f"\n{findings} finding(s).")
    return 1 if findings else 0


if __name__ == "__main__":
    sys.exit(main())
