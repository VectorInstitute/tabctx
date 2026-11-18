"""What the agent's cleaning did, for the page's "messy -> tidy" view: the
raw export the agent got, set beside the clean table it wrote, joined on
the id.

    {"file", "rows_before", "rows_after",
     "removed": [{"n", "what"}, ...],
     "columns": [{"raw", "clean", "kind", "before", "after", "example"}],
     "sample": {"columns": [[raw, clean], ...], "rows": [[[raw, clean], ...]]}}

A categorical column reports how many spellings became how many values
(and the value with the most spellings); a numeric one, how many values
were text (converted) or impossible (left empty). The sample is the rows
with the most fixed cells. Which raw column became which clean one is the
scenario's call (task.RAW_COLUMNS): headers alone don't say that "Prov"
is `province`.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import pandas as pd

SAMPLE_ROWS = 5
CLEAN = "clean:"
SPELLINGS_SHOWN = 5


def _numeric(clean: pd.Series) -> bool:
    return pd.api.types.is_numeric_dtype(clean)


def _show(value) -> str:
    """A cell as the page shows it: blanks as empty, 2384.0 as 2384."""
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return ""
    if isinstance(value, float):
        return f"{value:.2f}".rstrip("0").rstrip(".")
    return str(value)


def _changed(raw: str, clean) -> bool:
    shown = _show(clean)
    if raw == shown:
        return False
    try:  # "2384" and "2384.0", or "55" and "55.0", are the same number
        return float(raw) != float(shown)
    except ValueError:
        return True


def summarize(
    raw_file: Path,
    clean_file: Path,
    raw_id: str,
    clean_id: str,
    columns: dict[str, str],
) -> dict:
    raw = pd.read_csv(raw_file, dtype=str, keep_default_na=False)
    clean = pd.read_csv(clean_file)
    clean_ids = clean[clean_id].astype(str)
    duplicates = int(raw.duplicated(raw_id).sum())
    extra = int((~raw[raw_id].drop_duplicates().isin(clean_ids)).sum())
    removed = [
        {"n": duplicates, "what": "duplicate rows"},
        {"n": extra, "what": "rows that aren't records (tests, totals)"},
    ]
    pairs = [(r, c) for r, c in columns.items() if r in raw and c in clean]
    # Clean columns prefixed, so a raw header can't collide with one.
    joined = raw.drop_duplicates(raw_id).merge(
        clean.astype({clean_id: str}).add_prefix(CLEAN),
        left_on=raw_id,
        right_on=CLEAN + clean_id,
    )
    out = []
    for r, c in pairs:
        before, after = joined[r], joined[CLEAN + c]
        if _numeric(clean[c]):
            as_number = pd.to_numeric(before.str.strip(), errors="coerce")
            converted = int((as_number.isna() & after.notna()).sum())
            emptied = int((as_number.notna() & after.isna()).sum())
            out.append(
                {
                    "raw": r,
                    "clean": c,
                    "kind": "numeric",
                    "before": converted,
                    "after": emptied,
                }
            )
            continue
        spellings = before.groupby(after.astype(str)).unique()
        top = max(spellings.index, key=lambda v: len(spellings[v]))
        out.append(
            {
                "raw": r,
                "clean": c,
                "kind": "category",
                "before": int(before.nunique()),
                "after": int(after.nunique()),
                "example": {
                    "value": top,
                    "spellings": [str(s) for s in spellings[top][:SPELLINGS_SHOWN]],
                },
            }
        )
    return {
        "file": raw_file.name,
        "rows_before": len(raw),
        "rows_after": len(clean),
        "removed": [x for x in removed if x["n"]],
        "columns": out,
        "sample": _sample(joined, pairs),
    }


def _sample(joined: pd.DataFrame, pairs: list[tuple[str, str]]) -> dict:
    """The rows with the most fixed cells, raw beside clean."""
    rows = [
        [(raw, clean) for raw, clean in zip(cells[::2], cells[1::2], strict=True)]
        for cells in joined[[x for r, c in pairs for x in (r, CLEAN + c)]].itertuples(
            index=False
        )
    ]
    fixed = [sum(_changed(raw, clean) for raw, clean in row) for row in rows]
    best = sorted(range(len(rows)), key=lambda i: -fixed[i])[:SAMPLE_ROWS]
    return {
        "columns": [[r, c] for r, c in pairs],
        "rows": [[[raw, _show(clean)] for raw, clean in rows[i]] for i in best],
    }


SAVED = "cleaning.json"  # a showcase run's copy (make_showcase.py)


def for_run(run_dir: Path, scenario) -> dict | None:
    """A run's cleaning view: its saved copy, else from its own files, else
    None (no cleaning step, or the agent hasn't written the clean table)."""
    saved = run_dir / SAVED
    if saved.is_file():
        return json.loads(saved.read_text())
    task = scenario.task
    columns = getattr(task, "RAW_COLUMNS", None)
    clean = run_dir / getattr(task, "CLEAN_HISTORY", "")
    raw = run_dir / task.HISTORY
    if not columns or not clean.is_file() or not raw.is_file():
        return None
    try:
        return summarize(raw, clean, scenario.raw_id, task.ID, columns)
    except (KeyError, ValueError, pd.errors.ParserError):
        return None  # the agent's table isn't what the prompt asked for
