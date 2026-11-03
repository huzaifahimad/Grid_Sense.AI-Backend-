"""
GridSense AI — Decision rules (single source of truth).

The thresholds below drive three things that must never disagree:
  * the dashboard status label,
  * the shed-schedule recommendation and its priority,
  * the closed-loop validator that scores alarms against what actually happened.

Keeping them in one module means a validator that disagrees with the API is a
bug in the rules, not a drifted copy of them.
"""

from __future__ import annotations

CRITICAL_THRESHOLD = 0.8
ELEVATED_THRESHOLD = 0.6
MODERATE_THRESHOLD = 0.3
MAX_HORIZON_HOURS = 48

# An operator is paged at this score and above — the same cut the shed
# recommendation uses, so "alarm" in the validator means "would have shed".
ALARM_THRESHOLD = ELEVATED_THRESHOLD


def bucket_for_score(score: float) -> str:
    """Key into data/models/risk_factors_by_bucket.json."""
    if score >= CRITICAL_THRESHOLD:
        return "critical"
    if score >= ELEVATED_THRESHOLD:
        return "elevated"
    if score >= MODERATE_THRESHOLD:
        return "moderate"
    return "low"


def dashboard_status(score: float) -> str:
    if score >= CRITICAL_THRESHOLD:
        return "critical"
    if score >= ELEVATED_THRESHOLD:
        return "elevated"
    return "normal"


def shed_priority(score: float) -> int | None:
    """1 = shed first, 2 = shed second, None = no shed action recommended."""
    if score >= CRITICAL_THRESHOLD:
        return 1
    if score >= ELEVATED_THRESHOLD:
        return 2
    return None
