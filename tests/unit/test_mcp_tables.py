"""tabctx-mcp's table reading and profiling (mcp_tools/tables.py)."""

import pytest

pd = pytest.importorskip("pandas")
pytest.importorskip("pyarrow")

import numpy as np

from tabctx.mcp_tools.tables import profile, read_table


def test_blank_cells_are_missing_and_numeric_text_becomes_numbers(tmp_path):
    path = tmp_path / "t.csv"
    path.write_text("id,charges,plan\nA,10.5,gold\nB, ,\nC,7,basic\n")
    frame = read_table(path)
    assert frame["charges"].dtype.kind == "f"
    assert frame["charges"].isna().tolist() == [False, True, False]
    assert pd.isna(frame["plan"][1])


def test_mixed_text_column_stays_text(tmp_path):
    path = tmp_path / "t.csv"
    path.write_text("code\n007\nA1\n12\n")
    assert read_table(path)["code"].tolist() == ["007", "A1", "12"]


def test_parquet_index_is_not_a_column(tmp_path):
    frame = pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]}, index=[7, 3, 5])
    frame.to_parquet(tmp_path / "t.parquet")
    assert list(read_table(tmp_path / "t.parquet").columns) == ["a", "b"]


def test_tsv_and_gzip(tmp_path):
    (tmp_path / "t.tsv").write_text("a\tb\n1\tx\n")
    pd.DataFrame({"a": [1]}).to_csv(tmp_path / "t.csv.gz", index=False)
    assert list(read_table(tmp_path / "t.tsv").columns) == ["a", "b"]
    assert read_table(tmp_path / "t.csv.gz")["a"].tolist() == [1]


def test_missing_file():
    with pytest.raises(ValueError, match="no such file"):
        read_table("nope.csv")


def test_profile_flags_ids_and_describes_a_classification_target():
    frame = pd.DataFrame(
        {
            "id": [f"c{i}" for i in range(40)],
            "x": np.arange(40.0),
            "y": ["yes"] * 10 + ["no"] * 30,
        }
    )
    out = profile(frame, target="y")
    by_name = {f["name"]: f for f in out["features"]}
    assert set(by_name) == {"id", "x"}
    assert "likely an id" in by_name["id"]["note"]
    assert by_name["x"]["kind"] == "numeric" and by_name["x"]["max"] == 39.0
    assert out["target"] == {
        "name": "y",
        "task": "classification",
        "missing": 0,
        "classes": {"no": 30, "yes": 10},
    }


def test_profile_regression_target_and_column_cap():
    frame = pd.DataFrame({f"c{i}": [1.0, 2.0] for i in range(5)})
    frame["price"] = [10.25, 99.5]
    out = profile(frame, target="price", max_columns=2)
    assert out["target"]["task"] == "regression"
    assert len(out["features"]) == 2 and out["features_not_shown"] == 3


def test_profile_unknown_target():
    with pytest.raises(ValueError, match="no column"):
        profile(pd.DataFrame({"a": [1]}), target="b")
