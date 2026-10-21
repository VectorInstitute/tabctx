"""Table files -> columns: CSV (plain, gzip or zstd) and Parquet.

A fit or predict can carry its table as a file (a multipart upload, or
an earlier /v1/tabctx/upload referenced by id). The format comes from
the content, not the filename: Parquet starts with ``PAR1``, gzip and
zstd with their magic bytes, anything else is read as UTF-8 CSV with a
header row. Parquet keeps types and nulls and is several times smaller,
so it is the better way to send a large table.

Parsing lives at the serving layer on purpose: the engine keeps taking
in-memory arrays, so transport is a serving concern.

Features:

- A missing value is a Parquet null, or a CSV cell that is empty or
  ``NA``, ``N/A``, ``null``, ``None`` or ``NaN`` (any case, surrounding
  spaces ignored).
- Numbers stay numbers; dates and times become seconds (since the epoch,
  or since midnight for a time of day); any other text makes its column
  a text column, encoded by tabctx.schema. A Parquet string column is
  text even when its values look like numbers ("007"); a CSV has no
  types, so there a column of number-like text is numbers.

Labels (the target column) keep their text as written: ``"1.0"`` and
``"01"`` are classes of their own, and ``"NA"`` can be a class among text
labels. A target whose present values are all numbers is numeric: a
missing value among them is a missing label (an error: every training
row needs one), and the numbers decide the task when none is given
(labels.infer_task). Regression needs numbers.

Malformed content raises InvalidInputError naming the problem, so a
caller's bad file is a 422, never a 500. pyarrow is imported on first
use, so deployments that never read files don't need it.
"""

from __future__ import annotations

import csv as stdlib_csv
from collections.abc import Collection
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from tabctx.errors import InvalidInputError
from tabctx.labels import infer_task
from tabctx.types import Task

# Spellings of a missing value, compared case-insensitively after strip.
_MISSING = frozenset({"", "na", "n/a", "nan", "null", "none"})
# The same for arrow's CSV parser, which compares exactly (no strip).
_ARROW_NULLS = sorted(
    {f(m) for m in _MISSING for f in (str.lower, str.upper, str.title)}
)

_PARQUET_MAGIC = b"PAR1"
_COMPRESSION_MAGIC = {b"\x1f\x8b": "gzip", b"\x28\xb5\x2f\xfd": "zstd"}


@dataclass(frozen=True)
class Table:
    """Feature columns: float32 arrays (NaN = missing), or object arrays
    of text and None."""

    names: list[str]
    columns: list[np.ndarray]


def read_table(path: Path, text_columns: Collection[str] = ()) -> Table:
    """A table file's columns as features. text_columns: names to keep as
    text whatever their values look like (a training text column must
    stay text at predict: "007" is not 7)."""
    names, arrow = _read(path, text_columns)
    keep = _kept_text(path, names, text_columns)
    return Table(names, list(_features(arrow, names, keep)))


def read_training_table(
    path: Path, target: str | None, task: Task | None
) -> tuple[Table, list]:
    """(features, labels) of a training table file. target: the label
    column's name, default the last column. Labels are floats for
    regression and text for classification."""
    pa = _pyarrow()
    if target is None and not _is_parquet(path):
        # Known before reading, so the label column is read as text.
        header = _csv_header(path)
        target = header[-1].strip() if header else None
    names, arrow = _read(path, (), raw_text=(target,) if target else ())
    target = names[-1] if target is None else target
    if target not in names:
        raise InvalidInputError(
            f"target column {target!r} is not in the table's columns {names[:20]}"
        )
    if len(names) < 2:
        raise InvalidInputError(
            "the training table needs at least one feature column besides the target"
        )
    index = names.index(target)
    labels = _labels(arrow.column(index), task, pa)
    arrow = arrow.remove_column(index)
    del names[index]
    keep = _kept_text(path, names, ())
    return Table(names, list(_features(arrow, names, keep))), labels


# ---- labels -----------------------------------------------------------------


def _labels(col, task: Task | None, pa) -> list:
    kind = col.type
    if pa.types.is_string(kind) or pa.types.is_large_string(kind):
        return _text_labels(
            [None if v is None else v.strip() for v in col.to_pylist()], task
        )
    if pa.types.is_dictionary(kind):
        return _text_labels(col.cast(kind.value_type).to_pylist(), task)
    values = col.to_pylist()
    missing = [i for i, v in enumerate(values) if v is None or v != v]
    if missing:
        _no_label(missing[0])
    numeric = (
        pa.types.is_integer(kind)
        or pa.types.is_floating(kind)
        or pa.types.is_decimal(kind)
    )
    if not numeric:
        if task == "regression":
            raise InvalidInputError(
                f"regression needs numeric labels; the target has {kind}"
            )
        return [str(v) for v in values]
    numbers = col.cast(pa.float64()).to_pylist()
    return _numeric_labels([str(v) for v in values], numbers, task)


def _text_labels(texts: list[str | None], task: Task | None) -> list:
    for row, text in enumerate(texts):
        if text is None or text == "":
            _no_label(row)
    present = [t for t in texts if t.lower() not in _MISSING]
    numbers = _floats_or_none(present)
    if numbers is None:
        if task == "regression":
            example = next(t for t in texts if _floats_or_none([t]) is None)
            raise InvalidInputError(
                f"regression needs numeric labels; the target has text ({example!r})"
            )
        return texts  # text classes: "NA" among them is a class
    if len(present) < len(texts):  # "NA" among numbers is a missing label
        _no_label(next(i for i, t in enumerate(texts) if t.lower() in _MISSING))
    return _numeric_labels(texts, numbers, task)


def _numeric_labels(texts: list[str], numbers: list[float], task: Task | None) -> list:
    if task is None:
        task = infer_task(numbers)
    return numbers if task == "regression" else texts


def _floats_or_none(texts: list[str]) -> list[float] | None:
    try:
        return [float(t) for t in texts]
    except ValueError:
        return None


def _no_label(row: int) -> None:
    raise InvalidInputError(
        f"train target: row {row + 1} has no label; every training row needs one"
    )


# ---- features ---------------------------------------------------------------


def _kept_text(path: Path, names: list[str], text_columns: Collection[str]):
    """Columns whose text stays text even when it looks like numbers: all
    of a Parquet file's (it has types: a string column is text), and in a
    CSV, which has none, the named ones."""
    return set(names) if _is_parquet(path) else set(text_columns)


def _features(arrow, names: list[str], text_columns: Collection[str]):
    """Each column converted in turn."""
    for name in names:
        col = arrow.column(0)
        arrow = arrow.remove_column(0)
        yield _feature(col, name, name in text_columns)


def _feature(col, name: str, keep_text: bool) -> np.ndarray:
    import pyarrow as pa

    kind = col.type
    try:
        if pa.types.is_dictionary(kind):
            col = col.cast(kind.value_type)
            kind = col.type
        if pa.types.is_string(kind) or pa.types.is_large_string(kind):
            return _text_feature(col.to_pylist(), numeric_ok=not keep_text)
        if pa.types.is_temporal(kind):
            # float64: the schema stores them from an offset (tabctx.schema).
            return _seconds(col, pa).to_numpy()
        return col.cast(pa.float32(), safe=False).to_numpy()
    except (pa.ArrowInvalid, pa.ArrowNotImplementedError, pa.ArrowTypeError) as e:
        what = "not UTF-8 text" if pa.types.is_binary(kind) else f"of type {kind}"
        raise InvalidInputError(
            f"column {name!r} is {what}, which is neither numbers nor text"
        ) from e


def _seconds(col, pa):
    """Dates and times as seconds: since the epoch, or since midnight for
    a time of day; durations as their length."""
    import pyarrow.compute as pc

    kind = col.type
    if pa.types.is_time(kind):
        micros = col.cast(pa.time64("us")).cast(pa.int64())
    elif pa.types.is_duration(kind):
        micros = col.cast(pa.duration("us"), safe=False).cast(pa.int64())
    else:
        micros = col.cast(pa.timestamp("us"), safe=False).cast(pa.int64())
    return pc.divide(micros.cast(pa.float64()), 1e6)


def _text_feature(values: list, numeric_ok: bool) -> np.ndarray:
    """Stripped, missing spellings -> None; then, if allowed and every
    present value is a number (" 3", "NA" and "4.5" in a CSV that arrow
    couldn't type), a numeric column."""
    cells = [
        None if v is None or v.strip().lower() in _MISSING else v.strip()
        for v in values
    ]
    if numeric_ok:
        try:
            return np.array(
                [np.nan if v is None else float(v) for v in cells], dtype=np.float32
            )
        except ValueError:
            pass
    out = np.empty(len(cells), dtype=object)
    out[:] = cells
    return out


# ---- reading ----------------------------------------------------------------


def _pyarrow():
    try:
        import pyarrow as pa
    except ImportError as e:  # pragma: no cover -- a deployment packaging error
        raise InvalidInputError(
            "this deployment can't read table files (pyarrow is not installed); "
            "send the table as JSON rows"
        ) from e
    return pa


def _is_parquet(path: Path) -> bool:
    with open(path, "rb") as f:
        return f.read(4) == _PARQUET_MAGIC


def _read(path: Path, text_columns: Collection[str], raw_text: Collection[str] = ()):
    """(column names, arrow table). raw_text: columns read as text with no
    missing-value spellings applied (labels)."""
    pa = _pyarrow()
    try:
        arrow = _read_arrow(path, text_columns, raw_text)
    except (pa.ArrowInvalid, pa.ArrowNotImplementedError, OSError) as e:
        raise InvalidInputError(f"could not read the table file: {e}") from e
    names = [str(n).strip() for n in arrow.column_names]
    if not names:
        raise InvalidInputError("the table file has no columns")
    duplicated = sorted({n for n in names if names.count(n) > 1})
    if duplicated:
        raise InvalidInputError(f"the table file repeats column name(s) {duplicated}")
    if arrow.num_rows == 0:
        raise InvalidInputError("the table file has a header but no rows")
    return names, arrow


def _open_csv(path: Path):
    """The file as a stream of CSV bytes, decompressed if it's gzip or zstd."""
    import pyarrow as pa

    with open(path, "rb") as f:
        head = f.read(4)
    source = pa.OSFile(str(path))
    compression = _COMPRESSION_MAGIC.get(head[:2]) or _COMPRESSION_MAGIC.get(head)
    if compression is not None:
        source = pa.CompressedInputStream(source, compression)
    return source


def _csv_header(path: Path) -> list[str]:
    """A CSV's column names as written (unstripped), read ahead of the
    table so columns can be typed by name."""
    line = b""
    with _open_csv(path) as source:
        while b"\n" not in line and (block := source.read(1 << 16)):
            line += block
    try:
        text = line.split(b"\n", 1)[0].rstrip(b"\r").decode("utf-8-sig")
    except UnicodeDecodeError:
        return []  # reading the table reports it
    return next(stdlib_csv.reader([text]), [])


def _read_arrow(path: Path, text_columns: Collection[str], raw_text: Collection[str]):
    import pyarrow as pa
    import pyarrow.csv as csv
    import pyarrow.parquet as pq

    if _is_parquet(path):
        return pq.read_table(path)
    # Arrow matches column_types by the header's own (unstripped) spelling.
    wanted = set(text_columns) | set(raw_text)
    convert = csv.ConvertOptions(
        null_values=_ARROW_NULLS,
        # Text stays text here: missing spellings in text feature columns
        # are mapped by _text_feature, and labels keep theirs.
        strings_can_be_null=False,
        column_types={
            raw: pa.string() for raw in _csv_header(path) if raw.strip() in wanted
        },
    )
    with _open_csv(path) as source:
        return csv.read_csv(source, convert_options=convert)
