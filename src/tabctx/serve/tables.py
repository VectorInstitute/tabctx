"""Inline JSON tables -> arrays for the engine.

JSON has no NaN, so a missing cell arrives as ``null`` (None) and becomes
NaN; both backends handle missing values natively. A cell may also be
text, which makes its column a text column (tabctx.schema). An all-number
table (the common case) converts in one numpy call, so large numeric
tables pay no per-cell Python cost.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from tabctx.schema import TableSchema

Rows = Sequence[Sequence[float | str | None]]


def rows_to_array(rows: Rows) -> np.ndarray | None:
    """An all-number table as float32 (null -> NaN), or None when it has
    text, isn't a rectangle, or is empty."""
    # numpy would read "007" and "nan" as numbers; a text cell makes a
    # text column whatever it says (~10 ns a cell).
    if any(str in map(type, row) for row in rows):
        return None
    try:
        # float32, as files are encoded: both backends compute in
        # float32/16, and it halves what crosses to the worker processes.
        arr = np.asarray(rows, dtype=np.float32)
    except (TypeError, ValueError):
        return None
    return arr if arr.ndim == 2 and arr.size else None


def learn_rows(rows: Rows) -> tuple[np.ndarray | Rows, TableSchema | None]:
    """A training table, encoded, and the schema to encode test rows with.
    A table that isn't a non-empty rectangle comes back as is, with no
    schema, for the engine to reject in its usual terms."""
    arr = rows_to_array(rows)
    if arr is not None:
        return arr, TableSchema((None,) * arr.shape[1])
    columns = _columns(rows)
    if columns is None:
        return rows, None
    schema, X = TableSchema.learn(columns)
    return X, schema


def encode_rows(rows: Rows, schema: TableSchema | None) -> np.ndarray | Rows:
    """Test rows encoded as the training table was."""
    arr = rows_to_array(rows)
    if arr is not None and (schema is None or not (schema.has_text or schema.offsets)):
        return arr
    columns = _columns(rows)
    if columns is None:
        return rows
    if schema is None:
        # A context fit before schemas existed: numbers only.
        schema = TableSchema((None,) * len(columns))
    return schema.encode(columns)


def _columns(rows: Rows) -> list[np.ndarray] | None:
    if not rows or not rows[0]:
        return None
    width = len(rows[0])
    if any(len(row) != width for row in rows):
        return None
    columns = []
    for values in zip(*rows, strict=True):
        col = np.empty(len(values), dtype=object)
        col[:] = values
        columns.append(col)
    return columns
