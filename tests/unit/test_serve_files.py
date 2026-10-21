"""Table files -> columns (serve/files.py): CSV, compressed CSV, Parquet."""

import gzip
import io
import math

import numpy as np
import pytest

pa = pytest.importorskip("pyarrow")
import pyarrow.parquet as pq

from tabctx.errors import InvalidInputError
from tabctx.serve.files import read_table, split_target


def write(tmp_path, data, name="t"):
    p = tmp_path / name
    p.write_bytes(data.encode() if isinstance(data, str) else data)
    return p


def train(tmp_path, text, task=None, target=None):
    return split_target(read_table(write(tmp_path, text)), target, task)


def parquet_bytes(table):
    buf = io.BytesIO()
    pq.write_table(table, buf, compression="zstd")
    return buf.getvalue()


class TestFormats:
    CSV = "x,band,label\n1.5,gold,yes\n2,basic,no\n"

    def check(self, table):
        assert table.names == ["x", "band", "label"]
        assert table.columns[0].tolist() == [1.5, 2.0]
        assert list(table.columns[1]) == ["gold", "basic"]

    def test_csv(self, tmp_path):
        self.check(read_table(write(tmp_path, self.CSV)))

    def test_gzip_csv_detected_from_content(self, tmp_path):
        self.check(read_table(write(tmp_path, gzip.compress(self.CSV.encode()))))

    def test_zstd_csv_detected_from_content(self, tmp_path):
        zstd = pa.compress(self.CSV.encode(), codec="zstd", asbytes=True)
        # pa.compress writes a raw zstd frame, which is what `zstd file.csv` makes.
        self.check(read_table(write(tmp_path, zstd)))

    def test_parquet_detected_from_content(self, tmp_path):
        table = pa.table(
            {"x": [1.5, 2.0], "band": ["gold", "basic"], "label": ["yes", "no"]}
        )
        self.check(read_table(write(tmp_path, parquet_bytes(table))))

    def test_parquet_nulls_ints_bools_dates_and_categoricals(self, tmp_path):
        import datetime

        table = pa.table(
            {
                "n": pa.array([1, None], pa.int64()),
                "b": [True, False],
                "d": [datetime.date(1970, 1, 2), None],
                "c": pa.array(["a", None]).dictionary_encode(),
            }
        )
        cols = read_table(write(tmp_path, parquet_bytes(table))).features().columns
        assert cols[0][0] == 1.0 and math.isnan(cols[0][1])
        assert cols[1].tolist() == [1.0, 0.0]
        assert cols[2][0] == 86400.0 and math.isnan(cols[2][1])
        assert list(cols[3]) == ["a", None]

    def test_nested_columns_are_rejected(self, tmp_path):
        table = pa.table({"l": [[1, 2], [3]]})
        with pytest.raises(InvalidInputError, match="neither numbers nor text"):
            read_table(write(tmp_path, parquet_bytes(table)))


class TestCells:
    def test_missing_spellings_in_any_case_with_spaces(self, tmp_path):
        features, _ = train(
            tmp_path, "a,b,c,label\n1,,NA,x\nnull, n/a ,2,y\n3,NaN,None,x\n"
        )
        X = np.column_stack(features.columns)
        assert np.isnan(X).tolist() == [
            [False, True, True],
            [True, True, False],
            [False, True, True],
        ]

    def test_text_features_keep_text_and_lose_missing_spellings(self, tmp_path):
        features, _ = train(tmp_path, "c,label\ngold,x\nNA,y\n basic ,x\n")
        assert list(features.columns[0]) == ["gold", None, "basic"]

    def test_forced_text_columns_stay_text(self, tmp_path):
        p = write(tmp_path, "zip,x\n007,1\n010,2\n")
        assert read_table(p).features().columns[0].tolist() == [7.0, 10.0]
        table = read_table(p, text_columns=["zip"]).features(["zip"])
        assert list(table.columns[0]) == ["007", "010"]


class TestTarget:
    def test_default_target_is_the_last_column(self, tmp_path):
        features, y = train(tmp_path, "f0,f1,label\n1,2,cat\n3,4,dog\n")
        assert features.names == ["f0", "f1"]
        assert y == ["cat", "dog"]

    def test_named_target_anywhere(self, tmp_path):
        features, y = train(
            tmp_path, "label,f0\nyes,1\nno,3\n", task="classification", target="label"
        )
        assert features.names == ["f0"] and y == ["yes", "no"]

    def test_numeric_target_regression_and_classes(self, tmp_path):
        assert train(tmp_path, "f,y\n1,0.5\n2,1.5\n")[1] == [0.5, 1.5]
        assert train(tmp_path, "f,y\n1,0\n2,1\n")[1] == ["0", "1"]
        assert train(tmp_path, "f,y\n1,0\n2,1\n", task="regression")[1] == [0.0, 1.0]

    def test_na_among_text_classes_is_a_class(self, tmp_path):
        assert train(tmp_path, "f,label\n1,cat\n2,NA\n3,dog\n")[1] == [
            "cat",
            "NA",
            "dog",
        ]

    def test_missing_labels_are_rejected(self, tmp_path):
        for text in ("f,y\n1,2.5\n2,NA\n3,1.5\n", "f,label\n1,x\n2,\n"):
            with pytest.raises(InvalidInputError, match="row 2 has no label"):
                train(tmp_path, text)

    def test_unknown_target_and_no_features_are_rejected(self, tmp_path):
        with pytest.raises(InvalidInputError, match="'nope' is not in"):
            train(tmp_path, "f0,f1\n1,2\n", target="nope")
        with pytest.raises(InvalidInputError, match="at least one feature"):
            train(tmp_path, "label\ncat\n")


class TestMalformed:
    @pytest.mark.parametrize(
        ("text", "match"),
        [
            ("", "could not read"),
            ("f0,label\n", "no rows"),
            ("a,b,label\n1,2,x\n3,4\n", "could not read"),
            ("a,a\n1,2\n", "repeats column"),
        ],
    )
    def test_rejected_with_a_reason(self, tmp_path, text, match):
        with pytest.raises(InvalidInputError, match=match):
            read_table(write(tmp_path, text))
