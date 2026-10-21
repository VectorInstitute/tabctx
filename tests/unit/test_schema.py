"""Text columns and the training table's schema (tabctx.schema)."""

import math

import numpy as np
import pytest

from tabctx.errors import InvalidInputError
from tabctx.schema import TableSchema


def obj(*values):
    col = np.empty(len(values), dtype=object)
    col[:] = values
    return col


def nan_mask(X):
    return np.isnan(X).tolist()


class TestLearn:
    def test_numbers_stay_numbers_and_text_becomes_sorted_codes(self):
        schema, X = TableSchema.learn(
            [np.array([1.5, 2.5, np.nan]), obj("gold", "basic", "gold")]
        )
        assert schema.categories == (None, ("basic", "gold"))
        assert X.dtype == np.float32
        assert X[:, 1].tolist() == [1.0, 0.0, 1.0]
        assert math.isnan(X[2, 0])

    def test_missing_text_is_nan_not_a_category(self):
        schema, X = TableSchema.learn([obj("a", None, "b", float("nan"))])
        assert schema.categories == (("a", "b"),)
        assert nan_mask(X) == [[False], [True], [False], [True]]

    def test_numbers_in_a_text_column_are_text(self):
        schema, X = TableSchema.learn([obj(3, "x", 3.0, 2.5)])
        # 3 and 3.0 are the same category, as a CSV writes them.
        assert schema.categories == (("2.5", "3", "x"),)
        assert X[:, 0].tolist() == [1.0, 2.0, 1.0, 0.0]

    def test_an_object_column_of_numbers_is_numeric(self):
        schema, X = TableSchema.learn([obj(1, None, 2.5)])
        assert schema.categories == (None,)
        assert nan_mask(X) == [[False], [True], [False]]


class TestEncode:
    def setup_method(self):
        self.schema, _ = TableSchema.learn(
            [np.array([1.0, 2.0]), obj("lo", "hi")], names=["x", "band"]
        )

    def test_same_text_same_codes_and_unseen_is_missing(self):
        X = self.schema.encode([np.array([5.0, 6.0, 7.0]), obj("hi", "mid", None)])
        assert X[:, 0].tolist() == [5.0, 6.0, 7.0]
        assert X[0, 1] == 0.0  # "hi" sorts before "lo"
        assert nan_mask(X) == [[False, False], [False, True], [False, True]]

    def test_numbers_where_training_had_text_are_matched_as_text(self):
        schema, _ = TableSchema.learn([obj("1", "2", "x")])
        X = schema.encode([np.array([2.0, 1.0, 9.0])])
        assert X[:2, 0].tolist() == [1.0, 0.0]
        assert math.isnan(X[2, 0])

    def test_text_where_training_had_numbers_is_rejected(self):
        with pytest.raises(InvalidInputError, match="'x' held numbers"):
            self.schema.encode([obj("oops", 1.0), obj("lo", "hi")])

    def test_wrong_column_count_is_rejected(self):
        with pytest.raises(InvalidInputError, match="training table had 2"):
            self.schema.encode([np.array([1.0])])


class TestSelect:
    def test_by_name_in_any_order_ignoring_extras(self):
        schema = TableSchema((None, ("a",)), ("x", "band"))
        cols = schema.select(["band", "label", "x"], ["B", "L", "X"])
        assert cols == ["X", "B"]

    def test_missing_names_are_reported(self):
        schema = TableSchema((None, None), ("x", "y"))
        with pytest.raises(
            InvalidInputError, match=r"missing training column\(s\) \['y'\]"
        ):
            schema.select(["x"], ["X"])

    def test_unnamed_schemas_match_by_position(self):
        schema = TableSchema((None, None))
        assert schema.select(["a", "b"], ["A", "B"]) == ["A", "B"]


def test_text_names():
    assert TableSchema((None, ("a",)), ("x", "c")).text_names == ("c",)
    assert TableSchema((None, ("a",))).text_names == ()


def test_json_round_trip():
    for schema in (
        TableSchema((None, ("a", "b"))),
        TableSchema((("x",), None), ("c", "n")),
    ):
        assert TableSchema.from_json(schema.to_json()) == schema
