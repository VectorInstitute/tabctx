"""Table files -> columns: CSV (plain, gzip or zstd) and Parquet.

A fit or predict can carry its table as a file (a multipart upload, or
an earlier /v1/tabctx/upload referenced by id). The format comes from
the content, not the filename: Parquet starts with ``PAR1``, gzip and
zstd with their magic bytes, anything else is read as CSV with a header
row. Parquet keeps types and nulls and is several times smaller, so it
is the better way to send a large table.

Parsing lives at the serving layer on purpose: the engine keeps taking
in-memory arrays, so transport is a serving concern.

Cells:

- A CSV cell that is empty or ``NA``, ``N/A``, ``null``, ``None`` or
  ``NaN`` (any case, surrounding spaces ignored) is missing, as is a
  Parquet null. Numbers stay numbers; any other text makes its column a
  text column, encoded by tabctx.schema.
- In the target column, text is kept as written ("NA" can be a class);
  only an empty cell, or a missing value among numbers, is a missing
  label, which is an error: every training row needs one.

Malformed content raises InvalidInputError naming the problem, so a
caller's bad file is a 422, never a 500. pyarrow is imported on first
use, so deployments that never read files don't need it.
"""

from __future__ import annotations

from collections.abc import Collection
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from tabctx.errors import InvalidInputError
from tabctx.labels import infer_task
from tabctx.types import Task

# Spellings of a missing value, compared case-insensitively after strip.
_MISSING = frozenset({"", "na", "n/a", "nan", "null", "none"})

_PARQUET_MAGIC = b"PAR1"
_COMPRESSION_MAGIC = {b"\x1f\x8b": "gzip", b"\x28\xb5\x2f\xfd": "zstd"}


@dataclass(frozen=True)
class Table:
    """Columns as read: float arrays (NaN = missing), or object arrays of
    text as written (stripped; None only for a Parquet null)."""

    names: list[str]
    columns: list[np.ndarray]

    def features(self, text_columns: Collection[str] = ()) -> Table:
        """The columns as model features: missing spellings in text become
        None, and a text column whose values are all numbers or missing
        becomes numeric -- unless it's named in text_columns (a training
        text column must stay text: "007" is not 7)."""
        return Table(
            self.names,
            [
                _text_feature(col, name not in text_columns)
                if col.dtype == object
                else col
                for name, col in zip(self.names, self.columns, strict=True)
            ],
        )


def read_table(path: Path, text_columns: Collection[str] = ()) -> Table:
    """A table file's columns. text_columns: names to read as text
    whatever their values look like."""
    pa = _pyarrow()
    try:
        arrow = _read_arrow(path, text_columns)
    except (pa.ArrowInvalid, pa.ArrowNotImplementedError, OSError) as e:
        raise InvalidInputError(f"could not read the table file: {e}") from e
    names = [str(n) for n in arrow.column_names]
    if not names:
        raise InvalidInputError("the table file has no columns")
    duplicated = sorted({n for n in names if names.count(n) > 1})
    if duplicated:
        raise InvalidInputError(f"the table file repeats column name(s) {duplicated}")
    if arrow.num_rows == 0:
        raise InvalidInputError("the table file has a header but no rows")
    columns = [_column(arrow.column(i), names[i]) for i in range(len(names))]
    return Table(names, columns)


def split_target(
    table: Table, target: str | None, task: Task | None
) -> tuple[Table, list]:
    """(features, labels) of a training table. target: the label
    column's name, default the last column. Labels come back as floats
    for regression and text for classification."""
    target = table.names[-1] if target is None else target
    if target not in table.names:
        raise InvalidInputError(
            f"target column {target!r} is not in the table's columns {table.names[:20]}"
        )
    index = table.names.index(target)
    if len(table.names) < 2:
        raise InvalidInputError(
            "the training table needs at least one feature column besides the target"
        )
    features = Table(
        [n for i, n in enumerate(table.names) if i != index],
        [c for i, c in enumerate(table.columns) if i != index],
    )
    return features.features(), _labels(table.columns[index], task)


def _labels(col: np.ndarray, task: Task | None) -> list:
    if col.dtype.kind == "f":
        missing = np.isnan(col)
        if missing.any():
            _no_label(int(np.argmax(missing)))
        if task is None:
            task = infer_task(col)
        if task == "regression":
            return col.tolist()
        return [_whole_as_text(v) for v in col.tolist()]
    labels = list(col)
    for row, label in enumerate(labels):
        if label is None or label == "":
            _no_label(row)
    return labels


def _whole_as_text(v: float) -> str:
    return str(int(v)) if v.is_integer() else str(v)


def _no_label(row: int) -> None:
    raise InvalidInputError(
        f"train target: row {row + 1} has no label; every training row needs one"
    )


# ---- arrow ------------------------------------------------------------------


def _pyarrow():
    try:
        import pyarrow as pa
    except ImportError as e:  # pragma: no cover -- a deployment packaging error
        raise InvalidInputError(
            "this deployment can't read table files (pyarrow is not installed); "
            "send the table as JSON rows"
        ) from e
    return pa


def _read_arrow(path: Path, text_columns: Collection[str]):
    import pyarrow as pa
    import pyarrow.csv as csv
    import pyarrow.parquet as pq

    with open(path, "rb") as f:
        head = f.read(4)
    if head == _PARQUET_MAGIC:
        return pq.read_table(path)
    source = pa.OSFile(str(path))
    compression = _COMPRESSION_MAGIC.get(head[:2]) or _COMPRESSION_MAGIC.get(head)
    if compression is not None:
        source = pa.CompressedInputStream(source, compression)
    convert = csv.ConvertOptions(
        # Text stays text here (so the target keeps "NA" as written);
        # missing spellings in feature columns are mapped by _text_column.
        strings_can_be_null=False,
        column_types={name: pa.string() for name in text_columns},
    )
    with source:
        return csv.read_csv(source, convert_options=convert)


def _column(chunked, name: str) -> np.ndarray:
    import pyarrow as pa

    kind = chunked.type
    if pa.types.is_dictionary(kind):
        chunked = chunked.cast(kind.value_type)
        kind = chunked.type
    if pa.types.is_string(kind) or pa.types.is_large_string(kind):
        return np.array(
            [None if v is None else v.strip() for v in chunked.to_pylist()],
            dtype=object,
        )
    if pa.types.is_temporal(kind) and not pa.types.is_time(kind):
        # Dates and times as seconds since the epoch: ordered, numeric.
        chunked = chunked.cast(pa.timestamp("s")).cast(pa.int64())
    try:
        return chunked.cast(pa.float64()).to_numpy()
    except (pa.ArrowInvalid, pa.ArrowNotImplementedError) as e:
        raise InvalidInputError(
            f"column {name!r} has type {kind}, which is neither numbers nor text"
        ) from e


def _text_feature(col: np.ndarray, numeric_ok: bool) -> np.ndarray:
    """Missing spellings -> None; then, if allowed and every present value
    is a number (" 3", "NA" and "4.5" in a CSV that arrow couldn't type),
    a numeric column."""
    cells = [None if v is None or v.lower() in _MISSING else v for v in col]
    if numeric_ok:
        try:
            return np.array([np.nan if v is None else float(v) for v in cells])
        except ValueError:
            pass
    return np.array(cells, dtype=object)
