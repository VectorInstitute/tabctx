"""The insurance scenario, in group benefits: which employers whose group
benefits plan renews soon will leave, and what might keep them? The
agent gets two raw exports that need cleaning first (the "clean" step, so
it gets a shell). The files, the id and outcome columns, and how the page
names them (see ../../scenario.py).
"""

from __future__ import annotations

HISTORY = "renewals_2025_export.csv"  # last year's plan renewals, with the outcome
CURRENT = "renewals_due_export.csv"  # renewals due now, outcome withheld
ANSWERS = "current_outcomes.csv"  # under answers/insurance/
ID = "plan_id"  # the cleaned tables' names (PROMPT.md asks for them)
TARGET = "terminated"
POSITIVE = "Yes"
NEGATIVE = "No"
PROBA = f"p_{TARGET}={POSITIVE}"  # the column predict writes

# The raw exports' own names for the id and the outcome.
RAW_ID = "Plan ID"
RAW_TARGET = "Renewal Status"
# How the export spells each outcome (prepare_data.py writes these).
SPELLINGS = {
    POSITIVE: ("Terminated", "TERM", "moved carrier"),
    NEGATIVE: ("Renewed", "RENEWED", "In force"),
}
# The page's "messy -> tidy" view: which raw column became which clean one,
# and the clean history the agent writes (PROMPT.md names it).
RAW_COLUMNS = {
    "Prov": "province",
    "Industry": "industry",
    "Plan Type": "plan_type",
    "Lives Covered": "lives_covered",
    "Annual Premium": "annual_premium",
    "Channel": "channel",
    "Renewal Status": "terminated",
}
CLEAN_HISTORY = "renewals_2025.csv"
PLAN_PREFIX = "GB-"  # real plans; the export also has test rows and a total


def positive_rate(history) -> float:
    """The share terminated in the raw history (a pandas DataFrame), over
    its real plans, each counted once."""
    plans = history[history[RAW_ID].astype(str).str.startswith(PLAN_PREFIX)]
    status = plans.drop_duplicates(RAW_ID)[RAW_TARGET].astype(str).str.strip()
    return float(status.isin(SPELLINGS[POSITIVE]).mean())


# For the page: what a row is, what the positive outcome is called, and
# PROMPT.md's steps as icons (a tool's icon, "clean", or "report").
LABELS = {
    "org": "Maple Mutual Group Benefits",
    "question": (
        "Clean up these two exports, find which employers are likely to leave "
        "at renewal and why, and test what could keep them. Write it up for "
        "the head of group retention."
    ),
    # Questions to start from (id, the card's words, the question). The
    # open questions that held up against the real outcomes in testing.
    "suggestions": [
        [
            "retention",
            "Who will leave, and what could keep them?",
            "Clean up these two exports, find which employers are likely to leave "
            "at renewal and why, and test what could keep them. Write it up for "
            "the head of group retention.",
        ],
        [
            "calls",
            "Which 50 employers should we call first?",
            "Which 50 employers renewing in the next three months should our "
            "account managers call first, and what should they talk about with "
            "each one?",
        ],
        [
            "premium",
            "How much premium is at risk?",
            "How much annual premium is at risk of walking out the door in the "
            "next three months, and where is it concentrated?",
        ],
        [
            "reviews",
            "Are account reviews worth it?",
            "We're thinking of giving every at-risk plan an account review "
            "meeting next quarter. Is it worth it, and which plans should get one?",
        ],
    ],
    "audience": "the head of group retention",
    "entity": "employers",
    "outcome": "left",
    "likely": "likely to leave",
    "window": "at renewal",
    "history": "last year's plan renewals",
    "current": "renewals due",
    "steps": [
        ["clean", "Clean", "Clean and standardize both raw exports"],
        ["evaluate", "Compare", "Benchmark the models against gradient boosting"],
        ["predict", "Rank", "Predict every plan's chance of leaving"],
        ["explain", "Drivers", "Find what drives plans to leave"],
        ["what_if", "What-if", "Test retention moves on the likeliest to leave"],
        ["report", "Report", "Write report.md"],
    ],
    # Column names as a group retention team says them, and how a value
    # reads ("{}" is the value).
    "column_names": {
        "plan_id": ["Plan"],
        "years_with_us": ["Years with us"],
        "province": ["Province"],
        "industry": ["Industry"],
        "lives_covered": ["Employees covered"],
        "plan_type": ["Coverage"],
        "funding": ["Funding"],
        "annual_premium": ["Annual premium", "${}"],
        # snake_case of the export's headers, as an agent tends to write them
        "renewal_increase_pct": ["Renewal increase", "+{}%"],
        "rate_increase_pct": ["Renewal increase", "+{}%"],
        "loss_ratio_12m_pct": ["Loss ratio", "{}%"],
        "loss_ratio_pct": ["Loss ratio", "{}%"],
        "channel": ["Sold through"],
        "billing": ["Pays by"],
        "late_payments_12m": ["Late payments"],
        "account_reviews_12m": ["Account reviews"],
        "service_issues_12m": ["Service issues"],
        "online_admin": ["Online plan admin"],
        "wellness_program": ["Wellness program"],
        "member_app_use_pct": ["Members on the app", "{}%"],
        "member_app_pct": ["Members on the app", "{}%"],
        "terminated": ["Left?"],
    },
}
