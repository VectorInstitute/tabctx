"""A training table's columns, kept with its context so new rows can be
encoded the same way.

Both models take numbers. Real tables also have text columns ("gold",
"CA"), so at fit time each text column's values become categories, coded
0, 1, 2, ... in sorted order, and the context keeps that encoding. At
predict, the same text maps to the same codes; a value the training
table never had, or a missing one, becomes NaN, which both models treat
as missing.

A column is numeric when every present value is a number; any text makes
it a text column, and numbers in it are compared as text ("3" == 3).
Columns are matched by name when the table had a header (a file), and by
position otherwise (JSON rows).

The schema travels with the cached context (spill included) and between
processes, so it is plain data: names and category lists, JSON round
trip via to_json/from_json.
"""

from __future__ import annotations

import numbers
from collections.abc import Sequence
from dataclasses import dataclass
from functools import cached_property

import numpy as np

from tabctx.errors import InvalidInputError

# A table as columns: float arrays (NaN = missing) for numeric columns,
# object arrays of str/number/None for columns that may hold text.
Columns = Sequence[np.ndarray]


@dataclass(frozen=True)
class TableSchema:
    """Per feature column: None if numeric, else its categories in code
    order. names: the columns' names, when the table had a header.
    offsets: per column, a value subtracted before the model's float32,
    for columns given in float64 (times as seconds since the epoch, where
    float32 alone would round to 128 s): the training minimum."""

    categories: tuple[tuple[str, ...] | None, ...]
    names: tuple[str, ...] | None = None
    offsets: tuple[float | None, ...] | None = None

    @property
    def n_features(self) -> int:
        return len(self.categories)

    @property
    def has_text(self) -> bool:
        return any(c is not None for c in self.categories)

    @cached_property
    def approx_bytes(self) -> int:
        """Roughly what the schema takes in memory (its text dominates)."""
        text = [
            *(self.names or ()),
            *(c for cats in self.categories if cats for c in cats),
        ]
        return 64 * len(self.categories) + sum(len(t) + 56 for t in text)

    @property
    def text_names(self) -> tuple[str, ...]:
        """Names of the text columns (none when the table had no names)."""
        if self.names is None:
            return ()
        return tuple(
            n for n, c in zip(self.names, self.categories, strict=True) if c is not None
        )

    # ---- fit -----------------------------------------------------------

    @classmethod
    def learn(
        cls, columns: Columns, names: Sequence[str] | None = None
    ) -> tuple[TableSchema, np.ndarray]:
        """The schema of a training table, and the table encoded with it."""
        categories = tuple(_categories(col) for col in columns)
        offsets = tuple(_offset(col) for col in columns)
        schema = cls(
            categories,
            tuple(names) if names is not None else None,
            offsets if any(o is not None for o in offsets) else None,
        )
        return schema, schema.encode(columns)

    # ---- predict -------------------------------------------------------

    def encode(self, columns: Columns) -> np.ndarray:
        """Columns in schema order -> a float32 table for the model."""
        if len(columns) != self.n_features:
            raise InvalidInputError(
                f"the table has {len(columns)} columns; the training table had "
                f"{self.n_features}"
            )
        n_rows = len(columns[0]) if columns else 0
        out = np.empty((n_rows, self.n_features), dtype=np.float32)
        offsets = self.offsets or (None,) * self.n_features
        for j, (col, codes, offset) in enumerate(
            zip(columns, self._codes, offsets, strict=True)
        ):
            if codes is not None:
                out[:, j] = _encode(col, codes)
            elif offset is not None:
                out[:, j] = _numbers(col, self._label(j), np.float64) - offset
            else:
                out[:, j] = _numbers(col, self._label(j))
        return out

    @cached_property
    def _codes(self) -> tuple[dict[str, float] | None, ...]:
        """Per text column, value -> code (built once per schema)."""
        return tuple(
            None if cats is None else {c: float(i) for i, c in enumerate(cats)}
            for cats in self.categories
        )

    def __getstate__(self) -> dict:
        # Derived values are rebuilt on use, not shipped between processes.
        return {
            k: v
            for k, v in self.__dict__.items()
            if k not in ("_codes", "approx_bytes")
        }

    def select(self, names: Sequence[str], columns: Columns) -> Columns:
        """A named table's columns in schema order (by name when the
        training table had names, else by position). Extra columns, such
        as the target, are ignored; missing ones are an error."""
        if self.names is None:
            return columns
        index = {name: i for i, name in enumerate(names)}
        missing = [n for n in self.names if n not in index]
        if missing:
            raise InvalidInputError(
                f"the table is missing training column(s) {missing[:10]}; it has "
                f"{list(names)[:20]}"
            )
        return [columns[index[n]] for n in self.names]

    def _label(self, j: int) -> str:
        return repr(self.names[j]) if self.names is not None else f"{j}"

    # ---- storage -------------------------------------------------------

    def to_json(self) -> dict:
        return {
            "categories": [list(c) if c is not None else None for c in self.categories],
            "names": list(self.names) if self.names is not None else None,
            "offsets": list(self.offsets) if self.offsets is not None else None,
        }

    @classmethod
    def from_json(cls, data: dict) -> TableSchema:
        offsets = data.get("offsets")
        return cls(
            tuple(tuple(c) if c is not None else None for c in data["categories"]),
            tuple(data["names"]) if data.get("names") is not None else None,
            tuple(offsets) if offsets is not None else None,
        )


def _as_text(value: object) -> str:
    """A cell's text: numbers as written in a CSV (3, not 3.0)."""
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value)


def _present(value: object) -> bool:
    return value is not None and not (isinstance(value, float) and np.isnan(value))


def _categories(col: np.ndarray) -> tuple[str, ...] | None:
    if col.dtype.kind in "biuf":
        return None
    present = [v for v in col if _present(v)]
    if all(isinstance(v, numbers.Real) for v in present):
        return None
    return tuple(sorted({_as_text(v) for v in present}))


# float32 holds every integer up to 2**24 exactly; past it, values are
# stored from the training minimum so differences keep their resolution.
_FLOAT32_EXACT = float(2**24)


def _offset(col: np.ndarray) -> float | None:
    """For a float64 column with values past float32's exact range (times
    as seconds since the epoch): its training minimum."""
    if col.dtype != np.float64:
        return None
    finite = col[np.isfinite(col)]
    if not finite.size or np.abs(finite).max() < _FLOAT32_EXACT:
        return None
    return float(finite.min())


def _numbers(col: np.ndarray, label: str, dtype=np.float32) -> np.ndarray:
    if col.dtype.kind in "biuf":
        return col.astype(dtype, copy=False)
    try:
        return np.array([v if _present(v) else np.nan for v in col], dtype=dtype)
    except (TypeError, ValueError):
        text = next(v for v in col if isinstance(v, str))
        raise InvalidInputError(
            f"column {label} held numbers in the training table but has text "
            f"here ({text!r})"
        ) from None


def _encode(col: np.ndarray, code: dict[str, float]) -> np.ndarray:
    if col.dtype.kind == "f":  # a numeric column where training had text
        values = (None if np.isnan(v) else float(v) for v in col)
    else:
        values = iter(col)
    return np.array(
        [code.get(_as_text(v), np.nan) if _present(v) else np.nan for v in values],
        dtype=np.float32,
    )
