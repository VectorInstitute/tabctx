"""Reading table files, and describing them for an agent."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from tabctx.labels import infer_task

# Shown per column; enough for an agent to spot an id column or a typo.
_EXAMPLES = 3
# A text column with this share of distinct values (or more) is a
# likely id: worthless as a feature, and costly to encode.
_ID_LIKE = 0.95


def read_table(path: str | Path) -> pd.DataFrame:
    """A CSV, TSV (plain or gzip/zstd/bz2/xz) or Parquet file as a
    DataFrame. Blank text cells are missing, and text columns whose
    values are all numbers become numeric (a CSV export often writes
    " " for a missing number, which makes the whole column text)."""
    path = Path(path)
    if not path.is_file():
        raise ValueError(f"no such file: {path}")
    suffixes = [s.lower() for s in path.suffixes]
    if ".parquet" in suffixes or ".pq" in suffixes:
        frame = pd.read_parquet(path)
        # A saved pandas index is metadata, not a feature.
        frame = frame.reset_index(drop=True)
    elif ".tsv" in suffixes:
        frame = pd.read_csv(path, sep="\t")
    else:
        frame = pd.read_csv(path)
    return _clean(frame)


def _clean(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    for name in frame.columns:
        col = frame[name]
        if not (col.dtype == object or pd.api.types.is_string_dtype(col)):
            continue
        blank = col.map(lambda v: isinstance(v, str) and not v.strip())
        if blank.any():
            col = col.mask(blank, np.nan)
        numbers = pd.to_numeric(col, errors="coerce")
        if col.notna().any() and numbers.notna().sum() == col.notna().sum():
            col = numbers
        frame[name] = col
    return frame


def column_kind(col: pd.Series) -> str:
    if pd.api.types.is_bool_dtype(col):
        return "boolean"
    if pd.api.types.is_numeric_dtype(col):
        return "numeric"
    if pd.api.types.is_datetime64_any_dtype(col):
        return "datetime"
    return "text"


def describe_column(col: pd.Series) -> dict[str, Any]:
    present = col.dropna()
    kind = column_kind(col)
    info: dict[str, Any] = {
        "name": str(col.name),
        "kind": kind,
        "missing": int(col.isna().sum()),
        "distinct": int(present.nunique()),
    }
    if kind == "numeric" and len(present):
        info.update(
            min=_plain(present.min()),
            mean=round(float(present.mean()), 4),
            max=_plain(present.max()),
        )
    else:
        top = present.astype(str).value_counts().head(_EXAMPLES)
        info["top_values"] = {str(k): int(v) for k, v in top.items()}
    if kind == "text" and len(present) and info["distinct"] >= _ID_LIKE * len(present):
        info["note"] = "nearly every value is different: likely an id, drop it"
    return info


def describe_target(col: pd.Series) -> dict[str, Any]:
    present = col.dropna()
    if not len(present):
        raise ValueError(f"target column {col.name!r} has no values")
    task = infer_task(present.to_numpy())
    info: dict[str, Any] = {
        "name": str(col.name),
        "task": task,
        "missing": int(col.isna().sum()),
    }
    if task == "classification":
        counts = present.astype(str).value_counts()
        info["classes"] = {str(k): int(v) for k, v in counts.items()}
    else:
        info.update(
            min=_plain(present.min()),
            mean=round(float(present.mean()), 4),
            max=_plain(present.max()),
        )
    return info


def profile(
    frame: pd.DataFrame, target: str | None = None, max_columns: int = 60
) -> dict[str, Any]:
    """A summary of the table an agent can read in one go: size, each
    column's kind, missing values and typical values, and the target's
    task and class balance."""
    if target is not None:
        require_columns(frame, [target])
    names = [c for c in frame.columns if c != target]
    out: dict[str, Any] = {
        "rows": len(frame),
        "columns": len(frame.columns),
        "features": [describe_column(frame[c]) for c in names[:max_columns]],
    }
    if len(names) > max_columns:
        out["features_not_shown"] = len(names) - max_columns
    if target is not None:
        out["target"] = describe_target(frame[target])
    return out


def require_columns(frame: pd.DataFrame, names: list[str]) -> None:
    missing = [n for n in names if n not in frame.columns]
    if missing:
        shown = [str(c) for c in frame.columns[:40]]
        raise ValueError(f"no column(s) {missing} in the table; it has {shown}")


def _plain(v: Any) -> Any:
    """A numpy scalar as a JSON-friendly Python number."""
    return v.item() if isinstance(v, np.generic) else v
