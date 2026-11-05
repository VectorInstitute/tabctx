"""Build the insurance scenario's data: a synthetic book of policies at
Maple Mutual, a fictional Canadian insurer, and whether each policyholder
cancelled at renewal.

    uv run --extra mcp python examples/agent_demo/prepare_data.py insurance

The data is made up (no real customers) but follows patterns insurers
report: a premium increase at renewal, missed payments and paying by
cheque raise the chance of cancelling; years as a customer, more products
and contact with an advisor lower it. An advisor's check-in also halves
the effect of a premium increase, so the what-if step has a real lever to
find. About one policyholder in seven cancels.

The agent gets the tables as a policy system exports them, with the mess
real exports have: header names with spaces, dates and amounts in several
formats, one province or product spelled five ways, yes/no as Y/1/TRUE,
blanks written as N/A or "-", impossible ages, duplicated rows, internal
test policies and a totals row. Writes, under examples/agent_demo/:

- workspace/insurance/renewals_2025_export.csv  last year's renewals, with
                                                "Status at Renewal"
- workspace/insurance/renewals_due_export.csv   renewals due now; the
                                                outcome withheld
- answers/insurance/current_outcomes.csv        who cancelled (for
                                                score.py; outside the
                                                agent's workspace)
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from . import task

if TYPE_CHECKING:
    from scenario import Scenario

HISTORY_ROWS = 6_000
CURRENT_ROWS = 1_200
SEED = 11
HISTORY_RENEWALS = ("2025-01-01", "2025-12-31")
CURRENT_RENEWALS = ("2026-11-15", "2027-02-15")

PROVINCES = {  # code: (share, spellings in the export)
    "ON": (0.39, ["ON", "Ontario", "ont", " on", "ONT."]),
    "QC": (0.22, ["QC", "Quebec", "Québec", "PQ", "qc"]),
    "BC": (0.14, ["BC", "British Columbia", "B.C.", "bc"]),
    "AB": (0.12, ["AB", "Alberta", "Alta", "ab"]),
    "MB": (0.04, ["MB", "Manitoba", "Man."]),
    "SK": (0.03, ["SK", "Saskatchewan", "Sask"]),
    "NS": (0.03, ["NS", "Nova Scotia", "N.S."]),
    "NB": (0.02, ["NB", "New Brunswick", "N.B."]),
    "NL": (0.01, ["NL", "Newfoundland", "Nfld"]),
}
PRODUCTS = {  # name: (share, export spellings, base premium, effect on cancelling)
    "Term life": (0.38, ["Term Life", "TERM", "term life 20", "Term-Life"], 420, 0.30),
    "Whole life": (0.20, ["Whole Life", "WHOLE", "whole life"], 2600, -0.70),
    "Health and dental": (
        0.27,
        ["Health & Dental", "Health and dental", "H&D", "HEALTH/DENTAL"],
        1900,
        0.10,
    ),
    "Critical illness": (0.15, ["Critical Illness", "CI", "crit. illness"], 980, 0.0),
}
PAYMENT = {  # name: (share, export spellings, effect on cancelling)
    "Pre-authorized debit": (
        0.62,
        ["PAD", "Pre-authorized debit", "pre-auth debit", "Auto debit"],
        0.0,
    ),
    "Credit card": (0.25, ["Credit Card", "CC", "credit", "VISA/MC"], 0.30),
    "Cheque": (0.13, ["Cheque", "Check", "cheque/invoice", "INVOICE"], 0.75),
}
YES = ["Y", "Yes", "1", "TRUE", "yes"]
NO = ["N", "No", "0", "FALSE", "no"]
BLANK = ["", "N/A", "-", "unknown", "n/a"]


def _within(draw, low: float, high: float) -> np.ndarray:
    """draw(n) redrawn until every value is in [low, high]: clipping would
    pile values up on the bounds, which an agent rightly flags as odd."""
    values = draw()
    while (bad := (values < low) | (values > high)).any():
        values[bad] = draw()[: bad.sum()]
    return values


def _pick(rng: np.random.Generator, table: dict, n: int) -> np.ndarray:
    names = list(table)
    shares = np.array([table[k][0] for k in names])
    return rng.choice(names, size=n, p=shares / shares.sum())


def book(n: int, renewals: tuple[str, str], rng: np.random.Generator) -> pd.DataFrame:
    """n policies up for renewal between the two dates, clean, with whether
    each cancelled (cancelled = Yes/No)."""
    start, end = (pd.Timestamp(d) for d in renewals)
    renewal = start + pd.to_timedelta(rng.integers(0, (end - start).days + 1, n), "D")
    age = _within(lambda: rng.normal(46, 13, n), 21, 84).round().astype(int)
    product = _pick(rng, PRODUCTS, n)
    years = _within(lambda: rng.gamma(1.6, 4.2, n), 0.6, 32)
    since = renewal - pd.to_timedelta((years * 365.25).round(), "D")
    payment = _pick(rng, PAYMENT, n)
    annual = rng.random(n) < 0.3
    advisor = rng.random(n) < 0.55
    contacts = np.where(advisor, rng.poisson(1.2, n), 0)
    other = np.minimum(rng.poisson(0.55, n), 4)
    claims = rng.poisson(np.where(product == "Health and dental", 1.4, 0.15))
    online = rng.random(n) < np.clip(1.15 - age / 70, 0.15, 0.9)
    smoker = rng.random(n) < 0.13
    dependents = np.minimum(rng.poisson(np.where(age < 60, 1.2, 0.3)), 5)
    late = rng.poisson(np.where(payment == "Cheque", 0.9, 0.25))
    hike = np.where(
        rng.random(n) < 0.22,
        rng.uniform(9, 24, n),  # a repricing year for this policy
        rng.uniform(0, 6, n),
    ).round(1)
    base = np.array([PRODUCTS[p][2] for p in product])
    premium = (base * (0.6 + age / 55) * np.where(smoker, 1.6, 1.0)).round(2)

    contacted = contacts >= 1
    logit = (
        -1.9
        + 0.11 * hike * np.where(contacted, 0.5, 1.0)  # a check-in softens a hike
        + 0.55 * late
        + np.array([PAYMENT[p][2] for p in payment])
        + 0.35 * annual  # one big bill is a moment to reconsider
        - 0.075 * np.minimum(years, 15)
        - 0.45 * other
        - 0.55 * contacted
        - 0.30 * online
        + 0.35 * (age < 35)
        + np.array([PRODUCTS[p][3] for p in product])
        - 0.20 * np.minimum(claims, 3)
        + rng.normal(0, 0.35, n)
    )
    cancelled = rng.random(n) < 1 / (1 + np.exp(-logit))
    return pd.DataFrame(
        {
            "renewal_date": renewal,
            "policy_start": since,
            "age": age,
            "province": _pick(rng, PROVINCES, n),
            "product": product,
            "annual_premium": premium,
            "premium_change_pct": hike,
            "payment_method": payment,
            "payment_frequency": np.where(annual, "Annual", "Monthly"),
            "missed_payments_12m": late,
            "has_advisor": np.where(advisor, "Yes", "No"),
            "advisor_contacts_12m": contacts,
            "other_products": other,
            "claims_12m": claims,
            "online_account": np.where(online, "Yes", "No"),
            "smoker": np.where(smoker, "Yes", "No"),
            "dependents": dependents,
            task.TARGET: np.where(cancelled, task.POSITIVE, task.NEGATIVE),
        }
    )


# The export's header for each clean column.
HEADERS = {
    task.ID: task.RAW_ID,
    "renewal_date": "Renewal Date",
    "policy_start": "Policy Start",
    "age": "Age",
    "province": "Prov",
    "product": "Product Type",
    "annual_premium": "Annual Premium",
    "premium_change_pct": "Premium Change",
    "payment_method": "Pay Method",
    "payment_frequency": "Pay Freq",
    "missed_payments_12m": "Missed Pmts (12m)",
    "has_advisor": "Advisor?",
    "advisor_contacts_12m": "Advisor Contacts (12m)",
    "other_products": "Other Products",
    "claims_12m": "Claims (12m)",
    "online_account": "Online Account",
    "smoker": "Smoker",
    "dependents": "Dependents",
    task.TARGET: task.RAW_TARGET,
}


def _spell(rng: np.random.Generator, values: pd.Series, spellings: dict) -> pd.Series:
    """Each value as one of its export spellings (the first most often)."""

    def one(v: str) -> str:
        options = spellings[v]
        weights = np.array([4.0] + [1.0] * (len(options) - 1))
        return options[rng.choice(len(options), p=weights / weights.sum())]

    return values.map(one)


def _dates(rng: np.random.Generator, values: pd.Series) -> pd.Series:
    formats = ["%Y-%m-%d", "%b %d, %Y", "%Y/%m/%d", "%d-%b-%Y"]
    which = rng.choice(len(formats), size=len(values), p=[0.55, 0.2, 0.15, 0.1])
    return pd.Series(
        [d.strftime(formats[k]) for d, k in zip(values, which, strict=True)],
        index=values.index,
    )


def _blank_some(rng: np.random.Generator, values: pd.Series, share: float) -> pd.Series:
    out = values.astype(object).copy()
    hit = rng.random(len(out)) < share
    out[hit] = rng.choice(BLANK, size=int(hit.sum()))
    return out


def export(clean: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """The table as the policy system exports it: same rows, messy cells,
    the export's headers (no duplicates or extra rows yet)."""
    raw = pd.DataFrame(index=clean.index)
    raw[task.ID] = clean[task.ID]
    raw["renewal_date"] = _dates(rng, clean["renewal_date"])
    raw["policy_start"] = _dates(rng, clean["policy_start"])
    age = clean["age"].astype(object)
    odd = rng.random(len(age)) < 0.006
    age[odd] = rng.choice(["999", "-1", "0", "199"], size=int(odd.sum()))
    raw["age"] = _blank_some(rng, age, 0.01)
    raw["province"] = _spell(
        rng, clean["province"], {k: v[1] for k, v in PROVINCES.items()}
    )
    raw["product"] = _spell(
        rng, clean["product"], {k: v[1] for k, v in PRODUCTS.items()}
    )
    money = rng.choice(3, size=len(clean), p=[0.5, 0.35, 0.15])
    raw["annual_premium"] = [
        (f"${p:,.2f}", f"{p:.2f}", f"CAD {p:,.0f}")[k]
        for p, k in zip(clean["annual_premium"], money, strict=True)
    ]
    percent = rng.random(len(clean)) < 0.6
    raw["premium_change_pct"] = [
        f"{v:.1f}%" if pc else f"{v:.1f}"
        for v, pc in zip(clean["premium_change_pct"], percent, strict=True)
    ]
    raw["payment_method"] = _spell(
        rng, clean["payment_method"], {k: v[1] for k, v in PAYMENT.items()}
    )
    raw["payment_frequency"] = _spell(
        rng,
        clean["payment_frequency"],
        {
            "Monthly": ["Monthly", "M", "monthly", "MTHLY"],
            "Annual": ["Annual", "A", "yearly", "ANNUAL"],
        },
    )
    raw["missed_payments_12m"] = clean["missed_payments_12m"]
    yes_no = {"Yes": YES, "No": NO}
    for name in ("has_advisor", "online_account", "smoker"):
        raw[name] = _spell(rng, clean[name], yes_no)
    raw["advisor_contacts_12m"] = _blank_some(rng, clean["advisor_contacts_12m"], 0.04)
    raw["other_products"] = clean["other_products"]
    raw["claims_12m"] = clean["claims_12m"]
    raw["dependents"] = _blank_some(rng, clean["dependents"], 0.05)
    if task.TARGET in clean:
        raw[task.TARGET] = _spell(
            rng,
            clean[task.TARGET],
            {
                "Yes": ["Cancelled", "LAPSED", "cxl"],
                "No": ["Renewed", "RENEWED", "Active"],
            },
        )
    return raw.rename(columns=HEADERS)


def _duplicate(
    rng: np.random.Generator, raw: pd.DataFrame, share: float
) -> pd.DataFrame:
    """Some rows exported twice, the copies spread through the file."""
    copies = raw.sample(frac=share, random_state=int(rng.integers(1 << 31)))
    out = pd.concat([raw, copies])
    return out.sample(frac=1, random_state=int(rng.integers(1 << 31)))


def _with_junk(raw: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """Internal test policies, and a totals row at the bottom."""
    tests = raw.sample(n=6, random_state=int(rng.integers(1 << 31))).copy()
    tests[task.RAW_ID] = [f"TEST-{i:04d}" for i in range(1, 7)]
    out = pd.concat([raw, tests]).sample(
        frac=1, random_state=int(rng.integers(1 << 31))
    )
    total = {c: "" for c in out.columns}
    total[task.RAW_ID] = "TOTAL"
    total["Annual Premium"] = f"${raw['Annual Premium'].map(_amount).sum():,.2f}"
    return pd.concat([out, pd.DataFrame([total])])


def _amount(text: str) -> float:
    return float(str(text).replace("CAD", "").replace("$", "").replace(",", ""))


def build(seed: int = SEED) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """(history export, current export, current outcomes)."""
    rng = np.random.default_rng(seed)
    history = book(HISTORY_ROWS, HISTORY_RENEWALS, rng)
    current = book(CURRENT_ROWS, CURRENT_RENEWALS, rng)
    ids = rng.choice(
        np.arange(100_000, 1_000_000), HISTORY_ROWS + CURRENT_ROWS, replace=False
    )
    history.insert(0, task.ID, [f"MM-{i}" for i in ids[:HISTORY_ROWS]])
    current.insert(0, task.ID, [f"MM-{i}" for i in ids[HISTORY_ROWS:]])
    answers = current[[task.ID, task.TARGET]]
    history_raw = _with_junk(_duplicate(rng, export(history, rng), 0.02), rng)
    current_raw = _duplicate(rng, export(current.drop(columns=task.TARGET), rng), 0.01)
    return history_raw, current_raw, answers


def main(scenario: Scenario) -> None:
    history, current, answers = build()
    scenario.workspace.mkdir(parents=True, exist_ok=True)
    scenario.answers.mkdir(parents=True, exist_ok=True)
    history.to_csv(scenario.workspace / task.HISTORY, index=False)
    current.to_csv(scenario.workspace / task.CURRENT, index=False)
    answers.to_csv(scenario.answers / task.ANSWERS, index=False)
    rate = (answers[task.TARGET] == task.POSITIVE).mean()
    print(
        f"{len(history)} raw history rows ({task.positive_rate(history):.1%} cancelled), "
        f"{len(current)} raw current rows for {len(answers)} policies "
        f"({rate:.1%} will cancel) -> {scenario.workspace}"
    )
