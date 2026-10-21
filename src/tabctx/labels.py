"""What the training labels say about the task.

A caller shouldn't need to name the task for the obvious cases, so when
it's omitted it comes from the labels (resolve_task): non-integer numbers,
or whole numbers with more distinct values than a class list would have
(prices, counts), are a continuous target; anything else is
classification. Only actual
numbers count: text labels are categories even when they look numeric
("250.01" as a diagnosis code), as they always were. CSV labels arrive as
text, so the CSV parser converts them first when the task is left open
(serve/csv_io.py).

When the caller does say classification but sends non-integer numbers,
that's a mistake worth a clear 422 naming the fix, not a backend error
about "573 classes" or a fit that predicts nonsense.
"""

from __future__ import annotations

import numbers

import numpy as np

from tabctx.errors import InvalidInputError
from tabctx.types import ArrayLike, Task

_REGRESSION_HINT = "if train_y is a continuous target, pass task='regression'"

# Whole-number labels with more distinct values than this are inferred to
# be a quantity, not class codes (an explicit task overrides).
MAX_INFERRED_CLASSES = 30


def _as_numbers(y: ArrayLike) -> np.ndarray | None:
    """Labels as a flat float array when every one is a number (bools
    included), else None -- text, even numeric-looking, is not a number."""
    arr = np.asarray(y)
    if arr.dtype.kind in "biuf":
        return arr.astype(float).ravel()
    if arr.dtype.kind != "O":  # text (U/S) or anything else
        return None
    flat = arr.ravel()
    if not all(isinstance(v, numbers.Real) for v in flat):
        return None
    try:
        return flat.astype(float)
    except OverflowError:  # a Python int beyond float range: whole anyway
        return None


def _check_finite(values: np.ndarray) -> None:
    if not np.isfinite(values).all():
        raise InvalidInputError(
            "train_y has missing or infinite labels (NaN/inf); every training "
            "row needs a label"
        )


def _fractional(values: np.ndarray) -> np.ndarray:
    return values != np.round(values)


def infer_task(y: ArrayLike) -> Task:
    """Regression when every label is a number and some aren't whole, or
    when whole numbers take more than MAX_INFERRED_CLASSES distinct values;
    classification otherwise, which keeps class codes (0, 1, 2, ...) and
    text labels classes."""
    values = _as_numbers(y)
    if values is not None:
        _check_finite(values)
        if _fractional(values).any():
            return "regression"
        if np.unique(values).size > MAX_INFERRED_CLASSES:
            return "regression"
    return "classification"


def resolve_task(y: ArrayLike, task: Task | None) -> Task:
    """The task to fit: the caller's, checked against the labels, or the
    one the labels imply when the caller didn't say."""
    if task is None:
        return infer_task(y)
    validate_labels(y, task)
    return task


def validate_labels(y: ArrayLike, task: Task) -> None:
    """Rejects numeric labels no fit can use (NaN/inf), and classification
    labels that can only be a continuous target: numbers that aren't whole.
    Integer-coded classes are left alone -- they're legitimate, and a
    backend's class limit catches the rest (with regression_hint in its
    message)."""
    values = _as_numbers(y)
    if values is None:
        return
    _check_finite(values)
    if task != "classification":
        return
    fractional = _fractional(values)
    if fractional.any():
        example = float(values[np.argmax(fractional)])
        raise InvalidInputError(
            f"classification needs discrete class labels, but train_y has "
            f"non-integer numbers (e.g. {example:g}); {_REGRESSION_HINT}"
        )


def regression_hint(y: ArrayLike, task: Task) -> str:
    """A suffix for a backend's rejection of a classification table whose
    labels are all numbers: the likeliest cause is a missing task."""
    if task == "classification" and _as_numbers(y) is not None:
        return f" ({_REGRESSION_HINT})"
    return ""
