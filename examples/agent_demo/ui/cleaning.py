"""What the agent's cleaning did, for the page's "messy -> tidy" view: the
raw export the agent got, set beside the clean table it wrote, joined on
the id.

    {"file", "rows_before", "rows_after",
     "removed": [{"n", "what"}, ...],
     "columns": [{"raw", "clean", "kind", "before", "after", "example"}],
     "sample": {"columns": [[raw, clean], ...], "rows": [[[raw, clean], ...]]},
     "examples": {"rows", "features", "outcome": {"raw", "clean",
                  "classes": [{"value", "n", "spellings"}]}}}

The examples are the past rows the model will learn from: how many,
how many columns each, and how the export's outcome spellings were read
(which is the question's definition: what counts as cancelling). A
categorical column reports how many spellings became how many values
(and the value with the most spellings); a numeric one, how many values
were text (converted) or impossible (left empty). The sample is the rows
with the most fixed cells. Which raw column became which clean one is the
scenario's call (task.RAW_COLUMNS): headers alone don't say that "Prov"
is `province`.

The clean table is the agent's, so untrusted: for_run reads it only as a
plain file inside the run's folder and under a size cap, never a saved
view the agent could have written, and a table that isn't what the
prompt asked for gives no view rather than an error.
"""

from __future__ import annotations

import json
import math
from functools import lru_cache
from pathlib import Path

import pandas as pd

SAMPLE_ROWS = 5
CLEAN = "clean:"
SPELLINGS_SHOWN = 5
MAX_BYTES = 50_000_000  # a larger file isn't a table the page sets side by side


def _numeric(clean: pd.Series) -> bool:
    """A number column; yes/no written as True/False or 1/0 is a category."""
    if pd.api.types.is_bool_dtype(clean):
        return False
    numbers = clean.dropna()
    return pd.api.types.is_numeric_dtype(clean) and not numbers.isin([0, 1]).all()


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
    target: str | None = None,
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
    # Columns the agent left empty say nothing about what it fixed.
    pairs = [
        (r, c)
        for r, c in columns.items()
        if r in raw and c in clean and clean[c].notna().any()
    ]
    # Clean columns prefixed, so a raw header can't collide with one; one
    # row per id, should the agent have left duplicates.
    joined = raw.drop_duplicates(raw_id).merge(
        clean.drop_duplicates(clean_id).astype({clean_id: str}).add_prefix(CLEAN),
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
        # Spellings of a value, not of a blank ("N/A", "-" became empty).
        known = after.notna()
        if not known.any():
            continue
        before, after = before[known], after[known].astype(str)
        spellings = before.groupby(after).unique()
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
        "examples": _examples(clean, joined, columns, clean_id, target),
    }


def _examples(clean, joined, columns, clean_id, target) -> dict:
    """The rows the model learns from, and how the outcome was read."""
    out = {
        "rows": len(clean),
        "features": len([c for c in clean if c not in (clean_id, target)]),
    }
    raw_target = next((r for r, c in columns.items() if c == target), None)
    if target in clean and raw_target in joined:
        said = joined[raw_target].str.strip()
        read = joined[CLEAN + target].astype(str)
        counts = clean[target].astype(str).value_counts()
        out["outcome"] = {
            "raw": raw_target,
            "clean": target,
            "classes": [
                {
                    "value": value,
                    "n": int(n),
                    "spellings": sorted(said[read == value].unique().tolist()),
                }
                for value, n in counts.items()
            ],
        }
    return out


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


def for_run(run_dir: Path, scenario, saved: bool = False) -> dict | None:
    """A run's cleaning view, else None (no cleaning step, or no clean
    table the page can read). saved: the run is a showcase run, whose saved
    copy is read first; a visitor's agent could write that file itself."""
    if saved and (run_dir / SAVED).is_file():
        try:
            view = json.loads((run_dir / SAVED).read_text())
        except ValueError:
            return None
        return view if isinstance(view, dict) else None
    task = scenario.task
    columns = getattr(task, "RAW_COLUMNS", None)
    raw = _readable(run_dir, task.HISTORY)
    clean = _readable(run_dir, getattr(task, "CLEAN_HISTORY", None))
    if not columns or raw is None or clean is None:
        return None
    return _cached(
        raw,
        clean,
        scenario.raw_id,
        task.ID,
        tuple(columns.items()),
        getattr(task, "TARGET", None),
    )


def _readable(run_dir: Path, name: str | None) -> tuple[str, int, int] | None:
    """(path, size, mtime) of a plain file in run_dir under the size cap. The
    server may read as root: an agent's link to another file isn't followed."""
    if not name:
        return None
    path = run_dir / name
    try:
        if path.is_symlink() or not path.is_file():
            return None
        if not path.resolve().is_relative_to(run_dir.resolve()):
            return None
        stat = path.stat()
    except OSError:
        return None
    if stat.st_size > MAX_BYTES:
        return None
    return str(path), stat.st_size, stat.st_mtime_ns


@lru_cache(maxsize=32)
def _cached(raw, clean, raw_id, clean_id, columns, target=None) -> dict | None:
    """summarize(), once per version of the two files (their size and time
    are part of the key)."""
    try:
        return summarize(
            Path(raw[0]), Path(clean[0]), raw_id, clean_id, dict(columns), target
        )
    except (KeyError, ValueError, TypeError, pd.errors.ParserError):
        return None  # the agent's table isn't what the prompt asked for
