"""Table files -> columns (serve/files.py): CSV, compressed CSV, Parquet."""

import datetime
import gzip
import io
import math

import numpy as np
import pytest

pa = pytest.importorskip("pyarrow")
import pyarrow.parquet as pq

from tabctx.errors import InvalidInputError
from tabctx.serve.files import read_table, read_training_table


def write(tmp_path, data, name="t"):
    p = tmp_path / name
    p.write_bytes(data.encode() if isinstance(data, str) else data)
    return p


def train(tmp_path, data, task=None, target=None):
    return read_training_table(write(tmp_path, data), target, task)


def labels(tmp_path, data, task=None):
    return train(tmp_path, data, task)[1]


def parquet(table):
    buf = io.BytesIO()
    pq.write_table(table, buf, compression="zstd")
    return buf.getvalue()


def cells(col):
    return [None if isinstance(v, float) and math.isnan(v) else v for v in col.tolist()]


class TestFormats:
    CSV = "x,band,label\n1.5,gold,yes\n2,basic,no\n"

    def check(self, table, y):
        assert table.names == ["x", "band"]
        assert table.columns[0].tolist() == [1.5, 2.0]
        assert table.columns[0].dtype == np.float32
        assert list(table.columns[1]) == ["gold", "basic"]
        assert y == ["yes", "no"]

    def test_csv(self, tmp_path):
        self.check(*train(tmp_path, self.CSV))

    def test_gzip_csv_detected_from_content(self, tmp_path):
        self.check(*train(tmp_path, gzip.compress(self.CSV.encode())))

    def test_zstd_csv_detected_from_content(self, tmp_path):
        # A raw zstd frame, which is what `zstd file.csv` makes.
        self.check(
            *train(tmp_path, pa.compress(self.CSV.encode(), "zstd", asbytes=True))
        )

    def test_parquet_detected_from_content(self, tmp_path):
        table = pa.table(
            {"x": [1.5, 2.0], "band": ["gold", "basic"], "label": ["yes", "no"]}
        )
        self.check(*train(tmp_path, parquet(table)))

    def test_parquet_types(self, tmp_path):
        table = pa.table(
            {
                "n": pa.array([1, None], pa.int64()),
                "big": pa.array([2**63, 1], pa.uint64()),
                "b": [True, False],
                "d": [datetime.date(1970, 1, 2), None],
                "c": pa.array(["a", None]).dictionary_encode(),
            }
        )
        cols = read_table(write(tmp_path, parquet(table))).columns
        assert cells(cols[0]) == [1.0, None]
        assert cols[1][0] == np.float32(2**63)
        assert cols[2].tolist() == [1.0, 0.0]
        assert cells(cols[3]) == [86400.0, None]
        assert list(cols[4]) == ["a", None]

    def test_times_are_seconds_including_fractions(self, tmp_path):
        csv = "t,clock,y\n2020-01-01 10:00:00.5,10:30:00,1\n1970-01-01 00:00:01,00:00:02,2\n"
        cols = train(tmp_path, csv)[0].columns
        assert cols[0][1] == 1.0
        assert cols[0][0] == 1577872800.5  # float64 here; offset by the schema
        assert cols[1].tolist() == [37800.0, 2.0]
        table = pa.table(
            {
                "ts": pa.array([1_500_000], pa.timestamp("us")),
                "dur": pa.array([2_500_000], pa.duration("us")),
            }
        )
        cols = read_table(write(tmp_path, parquet(table))).columns
        assert cols[0].tolist() == [1.5] and cols[1].tolist() == [2.5]

    def test_nested_and_undecodable_columns_are_rejected(self, tmp_path):
        with pytest.raises(InvalidInputError, match="neither numbers nor text"):
            read_table(write(tmp_path, parquet(pa.table({"l": [[1, 2], [3]]}))))
        with pytest.raises(InvalidInputError, match="not UTF-8 text"):
            read_table(write(tmp_path, b"a,b\n1,caf\xe9\n"))


class TestFeatures:
    def test_missing_spellings_in_any_case_with_spaces(self, tmp_path):
        features, _ = train(
            tmp_path,
            "a,b,c,d,label\n1,,NA,x,y\nnull, n/a ,2,NONE,z\n3,NaN,None,Null,y\n",
        )
        assert [cells(c) for c in features.columns[:3]] == [
            [1.0, None, 3.0],
            [None, None, None],
            [None, 2.0, None],
        ]
        assert list(features.columns[3]) == ["x", None, None]

    def test_text_features_are_stripped(self, tmp_path):
        features, _ = train(tmp_path, "c,label\ngold,x\nNA,y\n basic ,x\n")
        assert list(features.columns[0]) == ["gold", None, "basic"]

    def test_header_names_are_stripped(self, tmp_path):
        features, y = train(tmp_path, " a , y \n1, 0\n2, 1\n", target="y")
        assert features.names == ["a"] and y == ["0", "1"]

    def test_text_columns_stay_text(self, tmp_path):
        p = write(tmp_path, "zip,x\n007,1\n010,2\n")
        assert read_table(p).columns[0].tolist() == [7.0, 10.0]
        assert list(read_table(p, text_columns=["zip"]).columns[0]) == ["007", "010"]

    def test_a_stripped_text_column_name_types_a_spaced_header(self, tmp_path):
        p = write(tmp_path, " zip ,x\n007,1\n")
        assert list(read_table(p, text_columns=["zip"]).columns[0]) == ["007"]


class TestLabels:
    def test_default_target_is_the_last_column(self, tmp_path):
        features, y = train(tmp_path, "f0,f1,label\n1,2,cat\n3,4,dog\n")
        assert features.names == ["f0", "f1"] and y == ["cat", "dog"]

    def test_named_target_anywhere(self, tmp_path):
        features, y = train(tmp_path, "label,f0\nyes,1\nno,3\n", target="label")
        assert features.names == ["f0"] and y == ["yes", "no"]

    def test_class_labels_keep_their_text(self, tmp_path):
        assert labels(tmp_path, "f,y\n1,1.0\n2,2.0\n") == ["1.0", "2.0"]
        assert labels(tmp_path, "f,y\n1,01\n2,02\n") == ["01", "02"]
        assert labels(tmp_path, "f,y\n1,true\n2,false\n") == ["true", "false"]

    def test_numbers_decide_the_task(self, tmp_path):
        assert labels(tmp_path, "f,y\n1,0.5\n2,1.5\n") == [0.5, 1.5]
        assert labels(tmp_path, "f,y\n1,0\n2,1\n") == ["0", "1"]
        assert labels(tmp_path, "f,y\n1,0\n2,1\n", task="regression") == [0.0, 1.0]
        assert labels(tmp_path, "f,y\n1,0.5\n2,1.5\n", task="classification") == [
            "0.5",
            "1.5",
        ]

    def test_na_among_text_classes_is_a_class(self, tmp_path):
        assert labels(tmp_path, "f,label\n1,cat\n2,NA\n3,dog\n") == ["cat", "NA", "dog"]

    @pytest.mark.parametrize("missing", ["", "NA", "None", "none", "null", "NaN"])
    def test_a_missing_label_among_numbers_is_rejected(self, tmp_path, missing):
        with pytest.raises(InvalidInputError, match="row 2 has no label"):
            labels(tmp_path, f"f,y\n1,2.5\n2,{missing}\n3,1.5\n")

    def test_a_blank_text_label_is_rejected(self, tmp_path):
        with pytest.raises(InvalidInputError, match="row 2 has no label"):
            labels(tmp_path, "f,label\n1,x\n2,\n")

    def test_regression_needs_numbers(self, tmp_path):
        with pytest.raises(InvalidInputError, match="regression needs numeric labels"):
            labels(tmp_path, "a,y\n1,a\n2,b\n", task="regression")

    def test_parquet_labels(self, tmp_path):
        def parquet_labels(values, task=None):
            table = pa.table({"x": [1.0] * len(values), "y": values})
            return read_training_table(write(tmp_path, parquet(table)), None, task)[1]

        assert parquet_labels([0, 1]) == ["0", "1"]
        assert parquet_labels([0.5, 1.25]) == [0.5, 1.25]
        assert parquet_labels([True, False]) == ["True", "False"]
        assert parquet_labels(pa.array(["a", "b"]).dictionary_encode()) == ["a", "b"]
        with pytest.raises(InvalidInputError, match="row 2 has no label"):
            parquet_labels([1.0, None])
        with pytest.raises(InvalidInputError, match="regression needs numeric"):
            parquet_labels([True, False], task="regression")

    def test_unknown_target_and_no_features_are_rejected(self, tmp_path):
        with pytest.raises(InvalidInputError, match="'nope' is not in"):
            train(tmp_path, "f0,f1\n1,2\n", target="nope")
        with pytest.raises(InvalidInputError, match="at least one feature"):
            train(tmp_path, "label\ncat\n")


@pytest.mark.parametrize(
    ("text", "match"),
    [
        ("", "could not read"),
        ("f0,label\n", "no rows"),
        ("a,b,label\n1,2,x\n3,4\n", "could not read"),
        ("a,a\n1,2\n", "repeats column"),
    ],
)
def test_malformed_files_are_rejected_with_a_reason(tmp_path, text, match):
    with pytest.raises(InvalidInputError, match=match):
        read_table(write(tmp_path, text))


def test_parquet_strings_stay_text_even_when_numeric(tmp_path):
    table = pa.table({"zip": ["007", "010"], "y": ["a", "b"]})
    features, _ = read_training_table(write(tmp_path, parquet(table)), "y", None)
    assert list(features.columns[0]) == ["007", "010"]


def test_a_long_csv_with_text_reads_its_header(tmp_path):
    # The header is read ahead to type the label column; a multi-byte
    # character anywhere later must not break that.
    rows = "".join(f"café{i % 7},0{i % 2 + 1}\n" for i in range(120_000))
    p = write(tmp_path, "town,y\n" + rows)
    assert p.stat().st_size > 1 << 20
    _, y = read_training_table(p, None, None)
    assert set(y) == {"01", "02"}


def test_crlf_headers_type_the_label_column(tmp_path):
    assert labels(tmp_path, "f,y\r\n1,01\r\n2,02\r\n") == ["01", "02"]


def test_decimal_labels_are_numbers(tmp_path):
    import decimal

    values = pa.array(
        [decimal.Decimal("1.5"), decimal.Decimal("2.5")], pa.decimal128(2, 1)
    )
    table = pa.table({"x": [1.0, 2.0], "y": values})
    path = write(tmp_path, parquet(table))
    assert read_training_table(path, None, "regression")[1] == [1.5, 2.5]
    assert read_training_table(path, None, None)[1] == [1.5, 2.5]


def test_times_keep_subsecond_resolution_through_the_schema(tmp_path):
    from tabctx.schema import TableSchema

    csv = "t,y\n2020-01-01 10:00:00.5,1\n2020-01-01 10:00:01,2\n2020-01-01 10:00:02,1\n"
    features, _ = train(tmp_path, csv)
    schema, X = TableSchema.learn(features.columns, features.names)
    assert X[:, 0].tolist() == [0.0, 0.5, 1.5]
    test = read_table(write(tmp_path, "t\n2020-01-01 10:00:03\n", name="test"))
    assert schema.encode(test.columns).tolist() == [[2.5]]
    assert TableSchema.from_json(schema.to_json()) == schema


class TestPandasIndex:
    """DataFrame.to_parquet() saves any index but the default 0..n-1 range
    as columns, listed in the file's pandas metadata; pandas reads them
    back as the index, and so must the server: they are not data."""

    @pytest.fixture
    def frame(self):
        pd = pytest.importorskip("pandas")
        return pd.DataFrame(
            {
                "x": [1.5, 2.0, 3.0],
                "band": ["gold", "basic", "gold"],
                "label": ["yes", "no", "yes"],
            },
            index=[7, 3, 5],  # e.g. after train_test_split
        )

    def path(self, tmp_path, frame, **kwargs):
        buf = io.BytesIO()
        frame.to_parquet(buf, **kwargs)
        return write(tmp_path, buf.getvalue())

    def test_a_saved_index_is_not_a_feature(self, tmp_path, frame):
        path = self.path(tmp_path, frame)
        assert (
            "__index_level_0__" in pq.read_table(path).column_names
        )  # the bug's cause
        table = read_table(path)
        assert table.names == ["x", "band", "label"]
        assert table.columns[0].tolist() == [1.5, 2.0, 3.0]

    def test_nor_the_default_target(self, tmp_path, frame):
        """The default target is the last column, where pandas puts the
        index: the labels must be the data's last column, not row numbers."""
        features, labels = read_training_table(self.path(tmp_path, frame), None, None)
        assert features.names == ["x", "band"]
        assert labels == ["yes", "no", "yes"]

    def test_named_and_multi_level_indexes(self, tmp_path, frame):
        named = frame.rename_axis("customer_id")
        assert read_table(self.path(tmp_path, named)).names == ["x", "band", "label"]
        multi = frame.set_index("band", append=True)
        assert read_table(self.path(tmp_path, multi)).names == ["x", "label"]

    def test_a_named_index_can_be_the_target(self, tmp_path, frame):
        indexed = frame.set_index("label")
        features, labels = read_training_table(
            self.path(tmp_path, indexed), "label", None
        )
        assert features.names == ["x", "band"] and labels == ["yes", "no", "yes"]

    def test_files_without_a_saved_index_are_unchanged(self, tmp_path, frame):
        default = frame.reset_index(drop=True)  # RangeIndex: metadata only
        assert read_table(self.path(tmp_path, default)).names == ["x", "band", "label"]
        assert read_table(self.path(tmp_path, frame, index=False)).names == [
            "x",
            "band",
            "label",
        ]

    def test_only_columns_the_metadata_lists_are_dropped(self, tmp_path):
        """A column that merely has an index-like name is data."""
        table = pa.table({"__index_level_0__": [1, 2], "x": [3.0, 4.0]})
        assert read_table(write(tmp_path, parquet(table))).names == [
            "__index_level_0__",
            "x",
        ]

    def test_unreadable_pandas_metadata_is_ignored(self, tmp_path):
        table = pa.table({"x": [1.0, 2.0]}).replace_schema_metadata(
            {b"pandas": b"not json"}
        )
        assert read_table(write(tmp_path, parquet(table))).names == ["x"]

    @pytest.mark.parametrize(
        "metadata",
        [b"[1]", b'"x"', b'{"index_columns": 5}', b'{"index_columns": "x"}', b"null"],
    )
    def test_metadata_of_another_shape_is_ignored(self, tmp_path, metadata):
        """JSON pandas didn't write: no 500, and no real column dropped (a
        string index_columns must not drop the column named "x")."""
        table = pa.table({"x": [1.0, 2.0], "y": [3.0, 4.0]})
        table = table.replace_schema_metadata({b"pandas": metadata})
        assert read_table(write(tmp_path, parquet(table))).names == ["x", "y"]

    @pytest.mark.parametrize("target", ["label", " label "])
    def test_names_are_compared_stripped(self, tmp_path, frame, target):
        indexed = frame.set_index("label").rename_axis(" label ")
        path = self.path(tmp_path, indexed)
        features, labels = read_training_table(path, target, None)
        assert features.names == ["x", "band"] and labels == ["yes", "no", "yes"]

    def test_errors_say_the_index_was_left_out(self, tmp_path, frame):
        """Labels saved as the index, with no target named: the index isn't
        read, which the error says, with what to do."""
        path = self.path(tmp_path, frame[["label"]].assign(x=1.0).set_index("label"))
        with pytest.raises(
            InvalidInputError, match=r"pandas index \['label'\].*reset_index"
        ):
            read_training_table(path, None, None)
        only_index = self.path(tmp_path, frame.set_index("x")[[]])
        with pytest.raises(
            InvalidInputError, match=r"no columns.*pandas index \['x'\]"
        ) as e:
            read_table(only_index)
        assert "target" not in str(e.value)  # a predict file has none

    def test_a_repeated_name_is_a_clear_error(self, tmp_path):
        """Repeated names are refused before the index is dropped (newer
        pyarrow refuses them while reading): a 422 naming the column."""
        table = pa.table([[1, 2], [3, 4], [5.0, 6.0]], names=["i", "i", "x"])
        table = table.replace_schema_metadata({b"pandas": b'{"index_columns": ["i"]}'})
        with pytest.raises(
            InvalidInputError, match=r"(repeats column name\(s\) \['i'\]|Name\(i\))"
        ):
            read_table(write(tmp_path, parquet(table)))
