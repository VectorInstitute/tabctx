"""Tables in memory (pandas DataFrames, numpy arrays, lists of rows) as
Parquet files to send to a deployment.

Parquet keeps what the server needs to see: which columns are text (a
string column stays text even when it reads "007"), missing values as
nulls, and dates and durations as such. Each column is numbers or text,
by the server's rule (tabctx.schema): numbers when every present value
is a number, else text, unless the caller names it as text (a column
that was text in the training table must stay text at predict, even
when its new values all look like numbers). Numbers travel as float32,
the precision both models compute in; labels as given.

Requires pyarrow (the `client` extra).
"""

from __future__ import annotations

import io
import numbers
from collections.abc import Collection, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class Columns:
    """A table as named columns: float32 (NaN = missing), datetime64 or
    timedelta64, or object arrays of text and None."""

    names: list[str]
    arrays: list[np.ndarray]
    named: bool  # the names came from the caller (a DataFrame's columns)

    @property
    def n_rows(self) -> int:
        return len(self.arrays[0])

    @property
    def text_names(self) -> list[str]:
        return [
            n for n, a in zip(self.names, self.arrays, strict=True) if a.dtype == object
        ]

    def renamed(self, names: Sequence[str]) -> Columns:
        return Columns(list(names), self.arrays, self.named)


def columns(
    X: Any, text: Collection[str] = (), positions: Collection[int] = ()
) -> Columns:
    """X as columns. text / positions: columns (by name, or by position)
    to keep as text whatever their values look like."""
    if is_frame(X):
        names = [str(c) for c in X.columns]
        named = True
        raw = [X.iloc[:, j] for j in range(len(names))]
    else:
        arr = _as_2d(X)
        names = [f"x{j}" for j in range(arr.shape[1])]
        named = False
        raw = [arr[:, j] for j in range(arr.shape[1])]
    if not raw or not len(raw[0]):
        raise ValueError("X has no rows or no columns")
    duplicated = sorted({n for n in names if names.count(n) > 1})
    if duplicated:
        raise ValueError(f"X repeats column name(s) {duplicated}")
    arrays = [
        _column(col, as_text=names[j] in text or j in positions)
        for j, col in enumerate(raw)
    ]
    return Columns(names, arrays, named)


def to_parquet(cols: Columns, labels: tuple[str, np.ndarray] | None = None) -> bytes:
    """The columns (and a label column, as (name, values)) as Parquet."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    names, arrays = list(cols.names), list(cols.arrays)
    if labels is not None:
        name, values = labels
        if name in names:
            raise ValueError(f"X has a column named {name!r}, which is reserved")
        names.append(name)
        arrays.append(np.asarray(values))
    fields = [
        pa.array(list(a), pa.string()) if a.dtype == object else pa.array(a)
        for a in arrays
    ]
    buf = io.BytesIO()
    pq.write_table(pa.table(fields, names=names), buf, compression="zstd")
    return buf.getvalue()


def is_frame(X: Any) -> bool:
    return hasattr(X, "columns") and hasattr(X, "iloc")


def take_rows(cols: Columns, start: int, stop: int) -> Columns:
    return Columns(cols.names, [a[start:stop] for a in cols.arrays], cols.named)


# ---- columns ----------------------------------------------------------------


def _as_2d(X: Any) -> np.ndarray:
    if hasattr(X, "toarray") or hasattr(X, "tocsr"):
        raise ValueError("sparse X is not supported; pass a dense array or DataFrame")
    # Lists keep each cell's own type ("007" stays text); arrays keep theirs.
    arr = X if isinstance(X, np.ndarray) else np.asarray(X, dtype=object)
    if arr.ndim != 2:
        raise ValueError(f"X must be 2-dimensional, got shape {arr.shape}")
    if arr.dtype.kind == "c":
        raise ValueError("complex X is not supported")
    return arr


def _column(values: Any, as_text: bool) -> np.ndarray:
    if hasattr(values, "dtype") and hasattr(values, "to_numpy"):  # a Series
        values = _series_values(values, as_text)
    if values.dtype.kind in "Mm" and not as_text:
        return values  # Parquet timestamps and durations; the server takes seconds
    if values.dtype.kind in "biuf" and not as_text:
        return values.astype(np.float32)
    values = values.astype(object) if values.dtype != object else values
    present = [v for v in values if not _missing(v)]
    if not as_text and all(isinstance(v, numbers.Real) for v in present):
        return np.array(
            [np.nan if _missing(v) else float(v) for v in values], dtype=np.float32
        )
    out = np.empty(len(values), dtype=object)
    out[:] = [None if _missing(v) else _text(v) for v in values]
    return out


def _series_values(series, as_text: bool) -> np.ndarray:
    from pandas.api.types import (
        is_bool_dtype,
        is_datetime64_any_dtype,
        is_numeric_dtype,
        is_timedelta64_dtype,
    )

    if as_text:
        return series.to_numpy(dtype=object)
    if is_datetime64_any_dtype(series.dtype):
        if getattr(series.dt, "tz", None) is not None:
            series = series.dt.tz_convert("UTC").dt.tz_localize(None)
        return series.to_numpy(dtype="datetime64[us]")
    if is_timedelta64_dtype(series.dtype):
        return series.to_numpy(dtype="timedelta64[us]")
    if is_numeric_dtype(series.dtype) or is_bool_dtype(series.dtype):
        # Nullable dtypes (Int64, boolean) included: pd.NA -> NaN.
        return series.to_numpy(dtype=np.float32, na_value=np.nan)
    return series.to_numpy(dtype=object)


def _missing(v: object) -> bool:
    if v is None:
        return True
    try:
        return bool(v != v)  # NaN, NaT
    except (TypeError, ValueError):  # pandas.NA
        return True


def _text(v: object) -> str:
    # As the server compares a number with text: 3, not 3.0.
    if isinstance(v, bool | np.bool_):
        return str(bool(v))
    if isinstance(v, numbers.Integral):
        return str(int(v))
    if isinstance(v, numbers.Real) and float(v).is_integer():
        return str(int(v))
    return str(v)  # numpy floats print their shortest form, as Python's do
