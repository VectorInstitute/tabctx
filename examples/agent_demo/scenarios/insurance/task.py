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
# The page's "messy -> tidy" view: which raw column became which clean one,
# and the clean history the agent writes (PROMPT.md names it).
RAW_COLUMNS = {
    "Prov": "province",
    "Product Type": "product_type",
    "Annual Premium": "annual_premium",
    "Pay Method": "pay_method",
    "Advisor?": "has_advisor",
    "Age": "age",
    "Status at Renewal": "cancelled",
}
CLEAN_HISTORY = "renewals_2025.csv"
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
    "org": "Maple Mutual",
    "question": (
        "Clean up these two exports, find who is likely to cancel at renewal "
        "and why, and test what could keep them. Write it up for the head of "
        "retention."
    ),
    "audience": "the head of retention",
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
    # Column names as a retention team says them, and how a value reads
    # ("{}" is the value).
    "column_names": {
        "policy_id": ["Policy"],
        "years_as_customer": ["Years as a customer"],
        "age": ["Age"],
        "province": ["Province"],
        "product_type": ["Product"],
        "annual_premium": ["Annual premium", "${}"],
        "premium_change_pct": ["Premium increase", "+{}%"],
        "pay_method": ["Pays by"],
        "pay_freq": ["Payment frequency"],
        "missed_payments_12m": ["Missed payments"],
        "has_advisor": ["Has an advisor"],
        "online_account": ["Online account"],
        "smoker": ["Smoker"],
        "advisor_contacts_12m": ["Advisor contacts"],
        "other_products": ["Other products held"],
        "claims_12m": ["Recent claims"],
        "dependents": ["Dependents"],
        "cancelled": ["Cancelled?"],
    },
}
