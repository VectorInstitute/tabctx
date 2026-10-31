"""tabctx-mcp's tools (mcp_tools/tools.py) against an in-memory fake
deployment. The fake predicts a row's class from its first encoded
feature (regression: returns it), so in these tables the first column,
x, decides the answer and the others are noise."""

import json

import numpy as np
import pytest

pd = pytest.importorskip("pandas")
pytest.importorskip("sklearn")
pytest.importorskip("pyarrow")

from fake_deployment import FakeDeployment

from tabctx.mcp_tools.store import ModelStore
from tabctx.mcp_tools.tools import BASELINE, TabularTools


def classification_table(n=80, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.integers(0, 2, n)
    return pd.DataFrame(
        {
            "x": x.astype(float),
            "plan": rng.choice(["basic", "gold", None], n),
            "noise": rng.normal(size=n),
            "customer_id": [f"c{i}" for i in range(n)],
            "churn": np.where(x == 1, "yes", "no"),
        }
    )


@pytest.fixture
def deployment(tmp_path):
    return FakeDeployment(tmp_path)


@pytest.fixture
def tools(tmp_path, deployment):
    return TabularTools(ModelStore(tmp_path / "store"), client=deployment)


@pytest.fixture
def history(tmp_path):
    path = tmp_path / "history.csv"
    classification_table().to_csv(path, index=False)
    return str(path)


def test_fit_keeps_the_model_and_its_description(tools, history, deployment):
    out = tools.fit_model(history, "churn", name="churn", drop=["customer_id"])
    assert out["task"] == "classification" and out["classes"] == ["no", "yes"]
    assert out["features"] == ["x", "plan", "noise"]
    assert out["numeric_features"] == ["x", "noise"]
    assert out["train_rows"] == 80 and "note" not in out
    assert deployment.fits[-1]["model"] == "tabpfn-3.5"
    assert [m["name"] for m in tools.list_models()["models"]] == ["churn"]
    json.dumps(out)  # what the MCP server and CLI hand back


def test_fit_default_name_and_unlabelled_rows(tools, tmp_path):
    frame = classification_table()
    frame.loc[:4, "churn"] = None
    path = tmp_path / "my data.csv"
    frame.to_csv(path, index=False)
    out = tools.fit_model(str(path), "churn", model="tabicl-v2")
    assert out["name"] == "my-data-churn-tabicl-v2"
    assert out["train_rows"] == 75 and "5 rows" in out["note"]


def test_predict_writes_the_table_with_predictions(tools, history, tmp_path):
    tools.fit_model(history, "churn", name="churn", drop=["customer_id"])
    new = classification_table(20, seed=1).drop(columns=["churn"])
    new.to_csv(tmp_path / "new.csv", index=False)
    out = tools.predict("churn", str(tmp_path / "new.csv"))
    written = pd.read_csv(out["output"])
    assert out["output"].endswith("new.churn.predictions.csv")
    assert list(written.columns) == [
        *new.columns,
        "predicted_churn",
        "p_churn=no",
        "p_churn=yes",
    ]
    expected = np.where(new["x"] == 1, "yes", "no")
    assert written["predicted_churn"].tolist() == expected.tolist()
    assert out["summary"]["predicted_counts"] == {
        k: int(v) for k, v in pd.Series(expected).value_counts().items()
    }
    assert len(out["top_rows"]) == 10 and "score_against_target" not in out
    json.dumps(out)


def test_predict_top_rows_rank_by_the_rarer_class_with_ids(tools, tmp_path):
    frame = classification_table()
    frame["churn"] = np.where(frame.index < 20, "yes", "no")  # yes is rarer
    frame["x"] = (frame.index < 20).astype(float)
    frame.to_csv(tmp_path / "h.csv", index=False)
    tools.fit_model(str(tmp_path / "h.csv"), "churn", name="c", drop=["customer_id"])
    out = tools.predict("c", str(tmp_path / "h.csv"), top=3, show=["plan"])
    assert out["ranked_by"] == "p_churn=yes"
    assert [r["customer_id"] for r in out["top_rows"]] == ["c0", "c1", "c2"]
    assert set(out["top_rows"][0]) == {
        "customer_id",
        "plan",
        "predicted_churn",
        "p_churn=no",
        "p_churn=yes",
    }

    some = tools.predict(
        "c", str(tmp_path / "h.csv"), where={"plan": "gold"}, rank_by="no", top=100
    )
    golds = int((frame["plan"] == "gold").sum())
    assert out["ranked_by"] != some["ranked_by"] == "p_churn=no"
    assert some["rows_matching_where"] == golds == len(some["top_rows"])
    with pytest.raises(ValueError, match="rank_by 'maybe'"):
        tools.predict("c", str(tmp_path / "h.csv"), rank_by="maybe")
    with pytest.raises(ValueError, match=r"no column\(s\) \['nope'\]"):
        tools.predict("c", str(tmp_path / "h.csv"), show=["nope"])
    assert tools.predict("c", str(tmp_path / "h.csv"), top=0)["top_rows"] == []


def test_predict_top_rows_for_regression(tools, tmp_path):
    frame = pd.DataFrame({"x": [3.0, 9.0, 1.0, 5.0], "price": [3.0, 9.0, 1.0, 5.0]})
    frame.to_csv(tmp_path / "r.csv", index=False)
    tools.fit_model(str(tmp_path / "r.csv"), "price", name="r", task="regression")
    out = tools.predict("r", str(tmp_path / "r.csv"), top=2)
    assert out["ranked_by"] == "predicted_price"
    assert [r["predicted_price"] for r in out["top_rows"]] == [9.0, 5.0]


def test_predict_scores_against_the_target_when_present(tools, history, tmp_path):
    tools.fit_model(history, "churn", name="churn", drop=["customer_id"])
    out = tools.predict("churn", history, output=str(tmp_path / "o" / "p.csv"))
    assert out["output"] == str(tmp_path / "o" / "p.csv")
    assert out["score_against_target"]["accuracy"] == 1.0


def test_predict_needs_the_features(tools, history, tmp_path):
    tools.fit_model(history, "churn", name="churn", drop=["customer_id"])
    pd.DataFrame({"x": [1.0]}).to_csv(tmp_path / "few.csv", index=False)
    with pytest.raises(ValueError, match=r"no column\(s\) \['plan', 'noise'\]"):
        tools.predict("churn", str(tmp_path / "few.csv"))


def test_predict_refits_after_the_server_dropped_the_context(
    tools, history, deployment, tmp_path
):
    tools.fit_model(history, "churn", name="churn", drop=["customer_id"])
    deployment.contexts.clear()  # evicted, or a restart
    out = tools.predict("churn", history, output=str(tmp_path / "p.csv"))
    assert out["rows"] == 80 and len(deployment.fits) == 2


def test_what_if_rows(tools, history):
    tools.fit_model(history, "churn", name="churn", drop=["customer_id"])
    out = tools.what_if("churn", [{"x": 1, "plan": "gold"}, {"x": "0"}])
    first, second = out["results"]
    assert first["prediction"] == "yes"
    assert first["probabilities"] == {"no": 0.0, "yes": 1.0}
    assert first["missing_columns"] == ["noise"]
    assert second["prediction"] == "no"  # "0" read as the number 0
    assert "base_row" not in out
    json.dumps(out)


def test_what_if_from_a_base_row(tools, history):
    tools.fit_model(history, "churn", name="churn", drop=["customer_id"])
    frame = pd.read_csv(history)
    customer = frame[frame.x == 1].iloc[0]
    out = tools.what_if(
        "churn",
        [{}, {"x": 0}],
        base_file=history,
        base_where={"customer_id": customer.customer_id},
    )
    assert out["base_row"]["x"] == 1.0 and "customer_id" not in out["base_row"]
    assert [r["prediction"] for r in out["results"]] == ["yes", "no"]
    assert out["results"][1]["changes"] == {"x": 0}
    json.dumps(out)


@pytest.mark.parametrize(
    ("rows", "kwargs", "message"),
    [
        ([], {}, "rows is empty"),
        ([{"nope": 1}], {}, r"unknown column\(s\) \['nope'\]"),
        ([{"x": "high"}], {}, "column 'x' is numeric, but got 'high'"),
        ([{}], {"base_where": {"customer_id": "c1"}}, "together"),
        ([{}], {"base_where": {"plan": "gold"}}, "must match one"),
    ],
)
def test_what_if_errors(tools, history, rows, kwargs, message):
    tools.fit_model(history, "churn", name="churn", drop=["customer_id"])
    if "plan" in kwargs.get("base_where", {}):
        kwargs["base_file"] = history
    with pytest.raises(ValueError, match=message):
        tools.what_if("churn", rows, **kwargs)


def test_evaluate_compares_models_and_the_baseline(tools, history):
    out = tools.evaluate(history, "churn", drop=["customer_id"])
    assert out["task"] == "classification"
    assert out["train_rows"] == 60 and out["test_rows"] == 20
    assert [r["model"] for r in out["results"]] == ["tabpfn-3.5", "tabicl-v2", BASELINE]
    for r in out["results"]:
        assert set(r["metrics"]) == {"accuracy", "roc_auc", "log_loss"}
    assert out["results"][0]["metrics"]["accuracy"] == 1.0
    assert out["best_by"] == "roc_auc" and out["best"] in {"tabpfn-3.5", "tabicl-v2"}
    json.dumps(out)


def test_evaluate_regression_without_baseline(tools, tmp_path):
    rng = np.random.default_rng(0)
    x = rng.normal(size=60) * 10
    frame = pd.DataFrame({"x": x, "z": rng.normal(size=60), "price": x})
    frame.to_csv(tmp_path / "r.csv", index=False)
    out = tools.evaluate(
        str(tmp_path / "r.csv"), "price", models=["tabicl-v2"], baseline=False
    )
    assert out["task"] == "regression" and len(out["results"]) == 1
    assert out["results"][0]["metrics"]["r2"] == pytest.approx(1.0, abs=1e-4)
    assert out["best_by"] == "r2"


def test_baseline_handles_text_dates_and_many_categories(tools, tmp_path):
    n = 600
    rng = np.random.default_rng(0)
    x = rng.integers(0, 2, n)
    frame = pd.DataFrame(
        {
            "x": x.astype(float),
            "when": pd.date_range("2024-01-01", periods=n, freq="D").astype(str),
            "many": [f"v{i % 300}" for i in range(n)],  # > 255 categories
            "flag": rng.random(n) < 0.5,
            "y": np.where(x == 1, "a", "b"),
        }
    )
    frame.to_csv(tmp_path / "t.csv", index=False)
    out = tools.evaluate(str(tmp_path / "t.csv"), "y", models=[])
    assert out["results"][0]["model"] == BASELINE
    assert out["results"][0]["metrics"]["accuracy"] > 0.9


def test_explain_ranks_the_column_the_model_uses_first(tools, history, deployment):
    out = tools.explain(history, "churn", drop=["customer_id"], repeats=2)
    assert out["metric"] == "roc_auc" and out["score_with_all_columns"] == 1.0
    ranked = [i["column"] for i in out["importance"]]
    assert ranked[0] == "x" and set(ranked) == {"x", "plan", "noise"}
    assert out["importance"][0]["score_drop"] > 0.2
    assert {i["column"]: i["score_drop"] for i in out["importance"]}["noise"] == 0.0
    # The original rows plus 2 shuffled copies per column, in one request.
    assert out["rows_predicted"] == 20 * (1 + 3 * 2)
    assert len(deployment.predict_calls) == 1


def test_explain_regression_and_multiclass(tools, tmp_path):
    rng = np.random.default_rng(0)
    x = rng.integers(0, 3, 90).astype(float)
    frame = pd.DataFrame({"x": x, "z": rng.normal(size=90), "y": x * 1.5})
    frame.to_csv(tmp_path / "r.csv", index=False)
    reg = tools.explain(str(tmp_path / "r.csv"), "y", task="regression")
    assert reg["metric"] == "r2" and reg["importance"][0]["column"] == "x"
    frame["k"] = x.astype(int).astype(str)
    frame.drop(columns="y").to_csv(tmp_path / "m.csv", index=False)
    multi = tools.explain(str(tmp_path / "m.csv"), "k", top=1)
    assert multi["metric"] == "accuracy" and len(multi["importance"]) == 1


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"target": "nope"}, r"no column\(s\) \['nope'\]"),
        ({"target": "churn", "drop": ["churn"]}, "cannot also be dropped"),
        ({"target": "churn", "task": "ranking"}, "task 'ranking'"),
        (
            {"target": "churn", "drop": ["x", "plan", "noise", "customer_id"]},
            "no feature columns",
        ),
        ({"target": "churn", "name": "../x"}, "model name"),
    ],
)
def test_fit_errors(tools, history, kwargs, message):
    with pytest.raises(ValueError, match=message):
        tools.fit_model(history, **kwargs)


def test_inspect_table(tools, history):
    out = tools.inspect_table(history, "churn")
    assert out["rows"] == 80 and out["file"] == history
    counts = pd.read_csv(history)["churn"].value_counts()
    assert out["target"]["task"] == "classification"
    assert out["target"]["classes"] == {k: int(v) for k, v in counts.items()}
