"""The insurance scenario: which policyholders whose renewal is coming up
will cancel, and what might keep them? The agent gets two raw exports
that need cleaning first (the "clean" step, so it gets a shell). The files, the id
and outcome columns, and how the page names them (see ../../scenario.py).
"""

from __future__ import annotations

HISTORY = "renewals_2025_export.csv"  # last year's renewals, with the outcome
CURRENT = "renewals_due_export.csv"  # renewals due now, outcome withheld
ANSWERS = "current_outcomes.csv"  # under answers/insurance/
ID = "policy_id"  # the cleaned tables' names (PROMPT.md asks for them)
TARGET = "cancelled"
POSITIVE = "Yes"
NEGATIVE = "No"
PROBA = f"p_{TARGET}={POSITIVE}"  # the column predict writes

# The raw exports' own names for the id and the outcome.
RAW_ID = "Policy ID"
RAW_TARGET = "Status at Renewal"
# How the export spells each outcome (prepare_data.py writes these).
SPELLINGS = {
    POSITIVE: ("Cancelled", "LAPSED", "cxl"),
    NEGATIVE: ("Renewed", "RENEWED", "Active"),
}
POLICY_PREFIX = "MM-"  # real policies; the export also has test rows and a total


def positive_rate(history) -> float:
    """The share cancelled in the raw history (a pandas DataFrame), over
    its real policies, each counted once."""
    policies = history[history[RAW_ID].astype(str).str.startswith(POLICY_PREFIX)]
    status = policies.drop_duplicates(RAW_ID)[RAW_TARGET].astype(str).str.strip()
    return float(status.isin(SPELLINGS[POSITIVE]).mean())


# For the page: what a row is, what the positive outcome is called, and
# PROMPT.md's steps as icons (a tool's icon, "clean", or "report").
LABELS = {
    "entity": "policyholders",
    "outcome": "cancelled",
    "window": "at renewal",
    "history": "last year's renewals",
    "current": "renewals due",
    "steps": [
        ["clean", "Clean", "Clean and standardize both raw exports"],
        ["evaluate", "Compare", "Benchmark the models against gradient boosting"],
        ["predict", "Rank", "Predict every policyholder's chance of cancelling"],
        ["explain", "Drivers", "Find what drives cancellation"],
        ["what_if", "What-if", "Test retention offers on the likeliest to cancel"],
        ["report", "Report", "Write report.md"],
    ],
}
