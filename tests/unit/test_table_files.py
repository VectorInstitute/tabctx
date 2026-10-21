"""In-memory tables -> Parquet (tabctx/table_files.py), checked by
reading them back the way the server does (serve/files.py)."""

import datetime
import math

import numpy as np
import pytest

pa = pytest.importorskip("pyarrow")
pd = pytest.importorskip("pandas")

from tabctx.serve.files import read_table, read_training_table
from tabctx.table_files import columns, take_rows, to_parquet


def read_back(tmp_path, cols, labels=None):
    p = tmp_path / "t.parquet"
    p.write_bytes(to_parquet(cols, labels))
    if labels is None:
        table = read_table(p)
        return table.names, table.columns
    table, y = read_training_table(p, labels[0], None)
    return table.names, table.columns, y


def cells(col):
    return [None if isinstance(v, float) and math.isnan(v) else v for v in col.tolist()]


def test_dataframe_with_text_missing_and_pandas_dtypes(tmp_path):
    df = pd.DataFrame(
        {
            "age": [31.5, np.nan, 40.0],
            "visits": pd.array([1, None, 3], dtype="Int64"),
            "member": [True, False, True],
            "plan": pd.Categorical(["gold", None, "basic"]),
            "city": ["Toronto", "Montréal, QC", None],
            "zip": ["007", "010", None],
        }
    )
    cols = columns(df)
    assert cols.named and cols.text_names == ["plan", "city", "zip"]
    names, got = read_back(tmp_path, cols)
    assert names == list(df.columns)
    assert cells(got[0]) == [31.5, None, 40.0]
    assert cells(got[1]) == [1.0, None, 3.0]
    assert got[2].tolist() == [1.0, 0.0, 1.0]
    assert list(got[3]) == ["gold", None, "basic"]
    assert list(got[4]) == ["Toronto", "Montréal, QC", None]
    assert list(got[5]) == ["007", "010", None]  # Parquet strings stay text


def test_dates_and_durations_travel_as_such(tmp_path):
    df = pd.DataFrame(
        {
            "when": pd.to_datetime(["1970-01-01 00:00:01.5", None]),
            "utc": pd.to_datetime(["1970-01-01 01:00:00+01:00"] * 2),
            "took": pd.to_timedelta(["1s", "2.5s"]),
        }
    )
    _, got = read_back(tmp_path, columns(df))
    assert cells(got[0]) == [1.5, None]
    assert got[1].tolist() == [0.0, 0.0]
    assert got[2].tolist() == [1.0, 2.5]


def test_rows_keep_each_cells_type(tmp_path):
    rows = [[1, "a", None, "007"], [2.5, 3, "b", "010"]]
    cols = columns(rows)
    assert cols.names == ["x0", "x1", "x2", "x3"] and not cols.named
    _, got = read_back(tmp_path, cols)
    assert got[0].tolist() == [1.0, 2.5]
    assert list(got[1]) == ["a", "3"]  # a number in a text column is text
    assert list(got[2]) == [None, "b"]
    assert list(got[3]) == ["007", "010"]


def test_float32_arrays_travel_exactly(tmp_path):
    X = np.random.default_rng(0).normal(size=(50, 3)).astype(np.float32)
    _, got = read_back(tmp_path, columns(X))
    assert np.array_equal(np.column_stack(got), X)


def test_forced_text_keeps_numbers_as_written(tmp_path):
    df = pd.DataFrame({"c": [0.1, 2.5, 16777217]})
    _, got = read_back(tmp_path, columns(df, text=["c"]))
    assert list(got[0]) == ["0.1", "2.5", "16777217"]
    _, got = read_back(tmp_path, columns(df.to_numpy(), positions=[0]))
    assert list(got[0]) == ["0.1", "2.5", "16777217"]


def test_labels_column_keeps_full_precision(tmp_path):
    y = np.array([123456789.123, 2.0])
    _, _, got = read_back(tmp_path, columns([[1.0], [2.0]]), ("__y__", y))
    assert got == [123456789.123, 2.0]


def test_bad_tables_are_rejected():
    with pytest.raises(ValueError, match="2-dimensional"):
        columns([1.0, 2.0])
    with pytest.raises(ValueError, match="no rows"):
        columns(np.empty((0, 3)))
    with pytest.raises(ValueError, match="reserved"):
        to_parquet(columns(pd.DataFrame({"y": [1.0]})), ("y", np.array([0])))
    with pytest.raises(ValueError, match="repeats"):
        columns(pd.DataFrame([[1, 2]], columns=["a", "a"]))
    with pytest.raises(ValueError, match="complex"):
        columns(np.ones((2, 2), dtype=complex))


def test_take_rows():
    cols = columns(np.arange(10.0).reshape(5, 2))
    assert take_rows(cols, 3, 5).arrays[1].tolist() == [7.0, 9.0]


def test_a_dated_object_column_is_text():
    # datetime objects in an object column (not a datetime dtype) are text.
    cols = columns([[datetime.date(2020, 1, 1)], [None]])
    assert list(cols.arrays[0]) == ["2020-01-01", None]
