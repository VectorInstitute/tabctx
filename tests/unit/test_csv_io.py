"""Unit tests for CSV parsing (serve/csv_io.py)."""

import numpy as np
import pytest

from tabctx.errors import InvalidInputError
from tabctx.serve.csv_io import parse_features_csv, parse_train_csv


def _write(tmp_path, text, name="t.csv"):
    p = tmp_path / name
    p.write_text(text)
    return p


class TestParseTrainCsv:
    def test_classification_default_target_is_last_column(self, tmp_path):
        p = _write(tmp_path, "f0,f1,label\n1,2,cat\n3,4,dog\n")
        X, y, names = parse_train_csv(p, "classification")
        assert X.dtype == np.float32 and X.shape == (2, 2)
        assert X.tolist() == [[1.0, 2.0], [3.0, 4.0]]
        assert y == ["cat", "dog"]
        assert names == ["f0", "f1"]

    def test_named_target_column_anywhere(self, tmp_path):
        p = _write(tmp_path, "label,f0,f1\nyes,1,2\nno,3,4\n")
        X, y, names = parse_train_csv(p, "classification", target_column="label")
        assert X.tolist() == [[1.0, 2.0], [3.0, 4.0]]
        assert y == ["yes", "no"]
        assert names == ["f0", "f1"]

    def test_regression_target_parsed_as_float(self, tmp_path):
        p = _write(tmp_path, "f0,y\n1,0.5\n2,1.5\n")
        _, y, _ = parse_train_csv(p, "regression")
        assert y == [0.5, 1.5]

    def test_missing_target_column_rejected(self, tmp_path):
        p = _write(tmp_path, "f0,f1\n1,2\n")
        with pytest.raises(InvalidInputError, match="nope"):
            parse_train_csv(p, "classification", target_column="nope")

    def test_no_feature_columns_rejected(self, tmp_path):
        p = _write(tmp_path, "label\ncat\n")
        with pytest.raises(InvalidInputError):
            parse_train_csv(p, "classification")

    def test_non_numeric_feature_rejected(self, tmp_path):
        p = _write(tmp_path, "f0,label\noops,cat\n")
        with pytest.raises(InvalidInputError):
            parse_train_csv(p, "classification")

    def test_empty_file_rejected(self, tmp_path):
        with pytest.raises(InvalidInputError):
            parse_train_csv(_write(tmp_path, ""), "classification")

    def test_header_only_rejected(self, tmp_path):
        with pytest.raises(InvalidInputError):
            parse_train_csv(_write(tmp_path, "f0,label\n"), "classification")

    def test_ragged_target_column_rejected(self, tmp_path):
        # Feature columns (0, 1) are present in every row, so they parse
        # cleanly; the target column (index 2) is missing from the second
        # data row -- must surface as a 422-worthy InvalidInputError, not
        # an unhandled numpy ValueError.
        p = _write(tmp_path, "a,b,label\n1,2,x\n3,4\n")
        with pytest.raises(InvalidInputError, match="train target"):
            parse_train_csv(p, "classification")

    def test_single_row_still_2d(self, tmp_path):
        X, y, _ = parse_train_csv(
            _write(tmp_path, "f0,f1,label\n1,2,cat\n"), "classification"
        )
        assert X.shape == (1, 2)
        assert y == ["cat"]


class TestParseFeaturesCsv:
    def test_roundtrip(self, tmp_path):
        p = _write(tmp_path, "f0,f1\n1,2\n3,4\n")
        X = parse_features_csv(p)
        assert X.shape == (2, 2) and X.dtype == np.float32

    def test_schema_match_enforced(self, tmp_path):
        p = _write(tmp_path, "f1,f0\n1,2\n")
        with pytest.raises(InvalidInputError, match="do not match"):
            parse_features_csv(p, expected_features=["f0", "f1"])

    def test_schema_match_ok(self, tmp_path):
        p = _write(tmp_path, "f0,f1\n1,2\n")
        assert parse_features_csv(p, expected_features=["f0", "f1"]).shape == (1, 2)
