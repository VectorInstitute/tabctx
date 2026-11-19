"""Build the insurance scenario's data: a synthetic book of group benefits
plans at Maple Mutual Group Benefits, a fictional Canadian insurer, and
whether each employer left at its annual renewal.

    uv run --extra mcp python examples/agent_demo/prepare_data.py insurance

The data is made up (no real employers) but follows patterns group
benefits carriers describe: a big renewal increase (which follows a bad
claims year, a high loss ratio), service problems, late payments and
small plans raise the chance an employer leaves; plans sold through a
broker get shopped around more. Years with us, broader coverage, ASO
funding, online plan administration, a wellness program and members who
use the app lower it. An account review meeting also halves the effect
of a renewal increase, so the what-if step has a real lever to find.
About one plan in eight leaves.

The agent gets the tables as a plan administration system exports them,
with the mess real exports have: header names with spaces, dates and
amounts in several formats, one province or industry spelled five ways,
yes/no as Y/1/TRUE, a loss ratio sometimes a percent and sometimes a
fraction, blanks written as N/A or "-", impossible plan sizes,
duplicated rows, internal test plans and a totals row. Writes, under
examples/agent_demo/:

- workspace/insurance/renewals_2025_export.csv  last year's renewals, with
                                                "Renewal Status"
- workspace/insurance/renewals_due_export.csv   renewals due now; the
                                                outcome withheld
- answers/insurance/current_outcomes.csv        who left (for score.py;
                                                outside the agent's
                                                workspace)
"""

from __future__ import annotations

import os
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
INDUSTRIES = {  # name: (share, export spellings, effect on leaving)
    "Professional services": (
        0.22,
        ["Professional Services", "Prof. Services", "PROFESSIONAL SVCS"],
        0.0,
    ),
    "Manufacturing": (0.16, ["Manufacturing", "MFG", "Mfg."], 0.0),
    "Retail": (0.14, ["Retail", "RETAIL", "Retail Trade"], 0.25),
    "Construction": (0.12, ["Construction", "CONSTR", "construction"], 0.20),
    "Technology": (0.12, ["Technology", "Tech", "IT/Software"], 0.15),
    "Health care": (0.10, ["Health Care", "Healthcare", "HEALTHCARE"], -0.10),
    "Non-profit and public": (0.14, ["Non-Profit", "Public Sector/NFP", "NFP"], -0.35),
}
PLAN_TYPES = {  # name: (share, export spellings, premium per covered employee, effect)
    "Health and dental": (
        0.40,
        ["Health & Dental", "H&D", "HEALTH/DENTAL", "Health and dental"],
        2600,
        0.25,
    ),
    "Health, dental and life": (
        0.30,
        ["Health, Dental & Life", "HDL", "H/D/Life"],
        3100,
        0.0,
    ),
    "Full package with disability": (
        0.30,
        ["Full Package", "FULL", "Full benefits (incl. LTD)", "H/D/L/LTD"],
        4200,
        -0.40,
    ),
}
FUNDING = {  # name: (export spellings, effect)
    "Fully insured": (["Fully Insured", "INSURED", "Pooled"], 0.0),
    "ASO": (["ASO", "A.S.O.", "Admin Services Only"], -0.45),
}
CHANNELS = {  # name: (share, export spellings, effect: a broker shops the plan around)
    "Broker": (0.55, ["Broker", "BRKR", "broker"], 0.25),
    "Benefits consultant": (0.25, ["Consultant", "Benefits Consultant", "CONS."], 0.10),
    "Direct": (0.20, ["Direct", "DIRECT", "In-house"], 0.0),
}
BILLING = {  # name: (share, export spellings, effect)
    "Pre-authorized debit": (0.60, ["PAD", "Pre-authorized debit", "Auto debit"], 0.0),
    "EFT": (0.25, ["EFT", "Wire/EFT", "eft"], 0.10),
    "Cheque": (0.15, ["Cheque", "Check", "cheque/invoice", "INVOICE"], 0.45),
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
    """n plans up for renewal between the two dates, clean, with whether
    each employer left (terminated = Yes/No)."""
    start, end = (pd.Timestamp(d) for d in renewals)
    renewal = start + pd.to_timedelta(rng.integers(0, (end - start).days + 1, n), "D")
    industry = _pick(rng, INDUSTRIES, n)
    lives = (
        _within(lambda: rng.lognormal(np.log(38), 1.05, n), 3, 2500).round().astype(int)
    )
    plan = _pick(rng, PLAN_TYPES, n)
    aso = rng.random(n) < np.clip(
        (np.log(lives) - 3.2) / 4, 0.02, 0.85
    )  # big plans self-fund
    years = _within(lambda: rng.gamma(1.7, 4.0, n), 0.6, 35)
    since = renewal - pd.to_timedelta((years * 365.25).round(), "D")
    channel = _pick(rng, CHANNELS, n)
    billing = _pick(rng, BILLING, n)
    # A plan's claims year: most run 60-85% of premium, some much worse.
    loss = np.where(
        rng.random(n) < 0.2,
        rng.uniform(88, 125, n),
        rng.uniform(55, 86, n),
    ).round(1)
    increase = np.clip(
        np.where(loss > 86, (loss - 80) * 0.45, rng.uniform(0, 6, n))
        + rng.normal(0, 1.2, n),
        0,
        28,
    ).round(1)
    reviews = rng.poisson(np.where(lives >= 50, 1.4, 0.6), n)
    issues = rng.poisson(0.35, n)
    late = rng.poisson(np.where(billing == "Cheque", 0.9, 0.2))
    online = rng.random(n) < 0.6
    wellness = rng.random(n) < np.clip(0.15 + np.log(lives) / 12, 0.1, 0.7)
    app = _within(lambda: rng.normal(42, 20, n), 0, 95).round()
    per_life = np.array([PLAN_TYPES[p][2] for p in plan])
    premium = (lives * per_life * rng.uniform(0.85, 1.15, n)).round(2)

    reviewed = reviews >= 1
    logit = (
        -2.05
        + 0.13 * increase * np.where(reviewed, 0.5, 1.0)  # a review softens an increase
        + 0.55 * np.minimum(issues, 3)
        + 0.45 * late
        + np.array([BILLING[b][2] for b in billing])
        + np.array([CHANNELS[c][2] for c in channel])
        + np.array([INDUSTRIES[i][2] for i in industry])
        + np.array([PLAN_TYPES[p][3] for p in plan])
        + np.where(aso, FUNDING["ASO"][1], 0.0)
        + 0.55 * (lives < 20)
        - 0.35 * (lives >= 200)
        - 0.06 * np.minimum(years, 15)
        - 0.55 * reviewed
        - 0.30 * online
        - 0.25 * wellness
        - 0.008 * app
        + rng.normal(0, 0.35, n)
    )
    left = rng.random(n) < 1 / (1 + np.exp(-logit))
    return pd.DataFrame(
        {
            "renewal_date": renewal,
            "plan_start": since,
            "province": _pick(rng, PROVINCES, n),
            "industry": industry,
            "lives_covered": lives,
            "plan_type": plan,
            "funding": np.where(aso, "ASO", "Fully insured"),
            "annual_premium": premium,
            "rate_increase_pct": increase,
            "loss_ratio_pct": loss,
            "channel": channel,
            "billing": billing,
            "late_payments_12m": late,
            "account_reviews_12m": reviews,
            "service_issues_12m": issues,
            "online_admin": np.where(online, "Yes", "No"),
            "wellness_program": np.where(wellness, "Yes", "No"),
            "member_app_pct": app,
            task.TARGET: np.where(left, task.POSITIVE, task.NEGATIVE),
        }
    )


# The export's header for each clean column.
HEADERS = {
    task.ID: task.RAW_ID,
    "renewal_date": "Renewal Date",
    "plan_start": "Plan Effective Date",
    "province": "Prov",
    "industry": "Industry",
    "lives_covered": "Lives Covered",
    "plan_type": "Plan Type",
    "funding": "Funding",
    "annual_premium": "Annual Premium",
    "rate_increase_pct": "Renewal Increase",
    "loss_ratio_pct": "Loss Ratio (12m)",
    "channel": "Channel",
    "billing": "Billing",
    "late_payments_12m": "Late Pmts (12m)",
    "account_reviews_12m": "Account Reviews (12m)",
    "service_issues_12m": "Service Issues (12m)",
    "online_admin": "Online Admin?",
    "wellness_program": "Wellness Program?",
    "member_app_pct": "Member App Use %",
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
    """The table as the plan administration system exports it: same rows,
    messy cells, the export's headers (no duplicates or extra rows yet)."""
    raw = pd.DataFrame(index=clean.index)
    raw[task.ID] = clean[task.ID]
    raw["renewal_date"] = _dates(rng, clean["renewal_date"])
    raw["plan_start"] = _dates(rng, clean["plan_start"])
    raw["province"] = _spell(
        rng, clean["province"], {k: v[1] for k, v in PROVINCES.items()}
    )
    raw["industry"] = _spell(
        rng, clean["industry"], {k: v[1] for k, v in INDUSTRIES.items()}
    )
    lives = clean["lives_covered"].astype(object)
    odd = rng.random(len(lives)) < 0.006
    lives[odd] = rng.choice(["0", "-1", "99999", "999999"], size=int(odd.sum()))
    raw["lives_covered"] = _blank_some(rng, lives, 0.01)
    raw["plan_type"] = _spell(
        rng, clean["plan_type"], {k: v[1] for k, v in PLAN_TYPES.items()}
    )
    raw["funding"] = _spell(
        rng, clean["funding"], {k: v[0] for k, v in FUNDING.items()}
    )
    money = rng.choice(3, size=len(clean), p=[0.5, 0.35, 0.15])
    raw["annual_premium"] = [
        (f"${p:,.2f}", f"{p:.2f}", f"CAD {p:,.0f}")[k]
        for p, k in zip(clean["annual_premium"], money, strict=True)
    ]
    percent = rng.random(len(clean)) < 0.6
    raw["rate_increase_pct"] = [
        f"{v:.1f}%" if pc else f"{v:.1f}"
        for v, pc in zip(clean["rate_increase_pct"], percent, strict=True)
    ]
    # One system writes the loss ratio as a percent, another as a fraction.
    fraction = rng.random(len(clean)) < 0.35
    raw["loss_ratio_pct"] = [
        f"{v / 100:.3f}" if fr else f"{v:.1f}%"
        for v, fr in zip(clean["loss_ratio_pct"], fraction, strict=True)
    ]
    raw["channel"] = _spell(
        rng, clean["channel"], {k: v[1] for k, v in CHANNELS.items()}
    )
    raw["billing"] = _spell(
        rng, clean["billing"], {k: v[1] for k, v in BILLING.items()}
    )
    raw["late_payments_12m"] = clean["late_payments_12m"]
    raw["account_reviews_12m"] = _blank_some(rng, clean["account_reviews_12m"], 0.04)
    raw["service_issues_12m"] = clean["service_issues_12m"]
    yes_no = {"Yes": YES, "No": NO}
    for name in ("online_admin", "wellness_program"):
        raw[name] = _spell(rng, clean[name], yes_no)
    raw["member_app_pct"] = _blank_some(rng, clean["member_app_pct"], 0.05)
    if task.TARGET in clean:
        raw[task.TARGET] = _spell(
            rng,
            clean[task.TARGET],
            {k: list(v) for k, v in task.SPELLINGS.items()},
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
    """Internal test plans, and a totals row at the bottom."""
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


def build(
    seed: int = SEED,
    history_rows: int = HISTORY_ROWS,
    current_rows: int = CURRENT_ROWS,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """(history export, current export, current outcomes)."""
    rng = np.random.default_rng(seed)
    history = book(history_rows, HISTORY_RENEWALS, rng)
    current = book(current_rows, CURRENT_RENEWALS, rng)
    ids = rng.choice(
        np.arange(100_000, 1_000_000), history_rows + current_rows, replace=False
    )
    ids = [f"{task.PLAN_PREFIX}{i}" for i in ids]
    history.insert(0, task.ID, ids[:history_rows])
    current.insert(0, task.ID, ids[history_rows:])
    answers = current[[task.ID, task.TARGET]]
    history_raw = _with_junk(_duplicate(rng, export(history, rng), 0.02), rng)
    current_raw = _duplicate(rng, export(current.drop(columns=task.TARGET), rng), 0.01)
    return history_raw, current_raw, answers


def main(scenario: Scenario, **sizes: int) -> None:
    """Write the scenario's data; sizes are build()'s row counts.

    DEMO_SEED replaces the fixed seed. The hosted image draws one at build
    time and keeps it nowhere: this generator is public, so with the fixed
    seed an agent with a shell could rebuild the withheld outcomes."""
    seed = int(os.environ.get("DEMO_SEED") or SEED)
    history, current, answers = build(seed, **sizes)
    scenario.workspace.mkdir(parents=True, exist_ok=True)
    scenario.answers.mkdir(parents=True, exist_ok=True)
    history.to_csv(scenario.workspace / task.HISTORY, index=False)
    current.to_csv(scenario.workspace / task.CURRENT, index=False)
    answers.to_csv(scenario.answers / task.ANSWERS, index=False)
    rate = (answers[task.TARGET] == task.POSITIVE).mean()
    print(
        f"{len(history)} raw history rows ({scenario.positive_rate(history):.1%} left), "
        f"{len(current)} raw current rows for {len(answers)} plans "
        f"({rate:.1%} will leave) -> {scenario.workspace}"
    )
