"""tabctx-mcp against a real local 2-replica deployment (fake backend:
every row gets the training table's class fractions, or the mean target):
the tools over HTTP, and the MCP server as an agent runs it, a stdio
subprocess."""

import asyncio
import os
import sys

import numpy as np
import pytest

pytest.importorskip("ray")
pytest.importorskip("sklearn")
pytest.importorskip("mcp")
pd = pytest.importorskip("pandas")

from tabctx.client import TabctxClient  # noqa: E402
from tabctx.mcp_tools.store import HOME_ENV, ModelStore  # noqa: E402
from tabctx.mcp_tools.tools import BASELINE, TabularTools  # noqa: E402

MODEL = "fake"


def customers(n=60, seed=0):
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "customer_id": [f"C{i:03d}" for i in range(n)],
            "tenure": rng.integers(1, 72, n),
            "contract": rng.choice(["monthly", "yearly", None], n),
            "charges": np.where(rng.random(n) < 0.1, np.nan, rng.normal(70, 20, n)),
            "churn": ["yes"] * (n // 4) + ["no"] * (n - n // 4),
        }
    )


@pytest.fixture
def files(tmp_path):
    history = tmp_path / "history.csv"
    current = tmp_path / "current.parquet"
    customers().to_csv(history, index=False)
    customers(12, seed=1).drop(columns="churn").to_parquet(current)
    return str(history), str(current)


@pytest.fixture
def tools(two_replica_service, tmp_path):
    return TabularTools(
        ModelStore(tmp_path / "store"), client=TabctxClient(two_replica_service)
    )


def test_fit_predict_what_if(tools, files):
    history, current = files
    fit = tools.fit_model(
        history, "churn", model=MODEL, name="churn", drop=["customer_id"]
    )
    assert fit["classes"] == ["no", "yes"] and fit["train_rows"] == 60

    out = tools.predict("churn", current)
    written = pd.read_csv(out["output"])
    assert len(written) == 12 and "customer_id" in written.columns
    assert written["p_churn=yes"].tolist() == [0.25] * 12  # the class fraction
    assert out["summary"]["predicted_counts"] == {"no": 12}

    what = tools.what_if(
        "churn",
        [{}, {"contract": "yearly", "tenure": "48"}],
        base_file=current,
        base_where={"customer_id": "C003"},
    )
    assert [r["probabilities"]["yes"] for r in what["results"]] == [0.25, 0.25]


def test_evaluate_and_explain(tools, files):
    history, _ = files
    out = tools.evaluate(history, "churn", models=[MODEL], drop=["customer_id"])
    assert [r["model"] for r in out["results"]] == [MODEL, BASELINE]
    assert out["results"][0]["metrics"]["accuracy"] == 0.7333  # always "no"
    expl = tools.explain(history, "churn", model=MODEL, drop=["customer_id"], repeats=2)
    # The fake ignores the features: shuffling any column changes nothing.
    assert {i["score_drop"] for i in expl["importance"]} == {0.0}


def test_a_model_survives_the_server_dropping_its_context(tools, files):
    history, current = files
    tools.fit_model(history, "churn", model=MODEL, name="churn", drop=["customer_id"])
    # The saved model is fitted under a different id on a fresh store: a
    # predict against an unknown context re-fits from the kept table.
    info, estimator = tools.store.load("churn")
    estimator.dataset_id_ = "never-fitted-" + estimator.dataset_id_
    tools.store.save(info, estimator)
    assert tools.predict("churn", current)["rows"] == 12


def test_the_mcp_server_over_stdio(two_replica_service, files, tmp_path):
    """The way Claude Code, Codex or Gemini CLI run it: a subprocess
    speaking MCP on stdin/stdout, configured by environment variables."""
    from mcp.client import Client
    from mcp.client.stdio import StdioServerParameters

    _, current = files
    params = StdioServerParameters(
        command=sys.executable,
        args=["-m", "tabctx.mcp_tools.cli", "serve"],
        env={
            **os.environ,
            "TABCTX_BASE_URL": two_replica_service,
            HOME_ENV: str(tmp_path / "mcp-store"),
        },
        cwd=str(tmp_path),
    )

    async def session():
        async with Client(params) as client:
            names = {t.name for t in (await client.list_tools()).tools}
            fit = await client.call_tool(
                "fit_model",
                {
                    "path": "history.csv",
                    "target": "churn",
                    "model": MODEL,
                    "name": "churn",
                    "drop": ["customer_id"],
                },
            )
            pred = await client.call_tool("predict", {"name": "churn", "path": current})
            bad = await client.call_tool("predict", {"name": "nope", "path": current})
            return names, fit, pred, bad

    names, fit, pred, bad = asyncio.run(session())
    assert {"fit_model", "predict", "what_if", "evaluate", "explain"} <= names
    assert not fit.is_error and fit.structured_content["classes"] == ["no", "yes"]
    assert pred.structured_content["rows"] == 12
    assert (tmp_path / "mcp-store" / "churn.pkl").is_file()  # TABCTX_MCP_HOME
    assert bad.is_error and "no model named 'nope'" in bad.content[0].text
