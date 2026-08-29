"""CSV -> arrays for fit/predict-by-reference (see serve/uploads.py).

Parsing lives at the serving layer on purpose: the engine keeps taking
in-memory tables (ArrayLike), so transport format is a serving concern
and the engine/backends never learn about files.

Format contract (kept deliberately simple -- this is a numeric-table
transport, not a general CSV ingester):

- First row is a header of column names.
- Feature columns must be numeric; parsed as float32 (halves memory for
  the 10^5-10^6-row tables this path exists for; both backends compute
  in float32/16 internally anyway).
- For training CSVs, one column is the target (``target_column``, or the
  LAST column when unspecified). Target values parse as strings for
  classification and floats for regression.
- No quoted fields containing commas (numpy's C parser; numeric tables
  don't need them).

Malformed content raises InvalidInputError naming the problem -- a
caller's bad file must 422, never 500.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from tabctx.errors import InvalidInputError
from tabctx.types import Task


def _read_header(path: Path) -> list[str]:
    with open(path, encoding="utf-8-sig") as f:
        first = f.readline().strip()
    if not first:
        raise InvalidInputError("CSV is empty (expected a header row)")
    return [c.strip() for c in first.split(",")]


def _load_float_columns(path: Path, usecols: list[int], what: str) -> np.ndarray:
    try:
        arr = np.loadtxt(
            path,
            delimiter=",",
            skiprows=1,
            usecols=usecols,
            dtype=np.float32,
            ndmin=2,
        )
    except ValueError as e:
        raise InvalidInputError(f"{what}: non-numeric or ragged CSV data ({e})") from e
    if arr.shape[0] == 0:
        raise InvalidInputError(f"{what}: CSV has a header but no data rows")
    return arr


def parse_train_csv(
    path: Path, task: Task, target_column: str | None = None
) -> tuple[np.ndarray, list, list[str]]:
    """Returns (X float32 2D, y list, feature_names)."""
    header = _read_header(path)
    if target_column is None:
        target_column = header[-1]
    if target_column not in header:
        raise InvalidInputError(
            f"target_column {target_column!r} not in CSV header {header}"
        )
    target_idx = header.index(target_column)
    feature_idx = [i for i in range(len(header)) if i != target_idx]
    if not feature_idx:
        raise InvalidInputError(
            "training CSV needs at least one feature column besides the target"
        )
    feature_names = [header[i] for i in feature_idx]

    X = _load_float_columns(path, feature_idx, "train features")

    if task == "regression":
        y_arr = _load_float_columns(path, [target_idx], "train target")
        y: list = [float(v) for v in y_arr[:, 0]]
    else:
        try:
            y_arr = np.loadtxt(
                path,
                delimiter=",",
                skiprows=1,
                usecols=[target_idx],
                dtype=str,
                ndmin=2,
            )
        except ValueError as e:
            raise InvalidInputError(f"train target: unreadable CSV column ({e})") from e
        y = [str(v).strip() for v in y_arr[:, 0]]

    if len(y) != X.shape[0]:
        raise InvalidInputError(
            f"target column has {len(y)} values but features have {X.shape[0]} rows"
        )
    return X, y, feature_names


def parse_features_csv(
    path: Path, expected_features: list[str] | None = None
) -> np.ndarray:
    """Returns a float32 2D array of all columns. When expected_features
    is given (from the training upload), the header must match it -- a
    silently reordered or missing column would produce garbage
    predictions, which is far worse than a 422."""
    header = _read_header(path)
    if expected_features is not None and header != expected_features:
        raise InvalidInputError(
            f"test CSV columns {header} do not match the training features "
            f"{expected_features} (same names, same order, no target column)"
        )
    return _load_float_columns(path, list(range(len(header))), "test features")
