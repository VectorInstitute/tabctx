"""tabctx-mcp's MCP server (mcp_tools/server.py) and command line
(mcp_tools/cli.py), over the same fake deployment as test_mcp_tools."""

import asyncio
import json

import pytest

pd = pytest.importorskip("pandas")
pytest.importorskip("sklearn")
pytest.importorskip("mcp")

from fake_deployment import FakeDeployment
from mcp.server.mcpserver.exceptions import ToolError
from test_mcp_tools import classification_table

from tabctx.errors import DatasetNotFoundError
from tabctx.mcp_tools import cli
from tabctx.mcp_tools.server import build_server, tool_specs
from tabctx.mcp_tools.store import ModelStore
from tabctx.mcp_tools.tools import TabularTools

TOOLS = {
    "inspect_table",
    "evaluate",
    "fit_model",
    "predict",
    "what_if",
    "explain",
    "list_models",
}


@pytest.fixture
def tools(tmp_path):
    return TabularTools(ModelStore(tmp_path / "store"), client=FakeDeployment(tmp_path))


@pytest.fixture
def history(tmp_path):
    path = tmp_path / "history.csv"
    classification_table().to_csv(path, index=False)
    return str(path)


def call(server, name, args):
    result = asyncio.run(server.call_tool(name, args))
    # (content, structured) or a CallToolResult, depending on the SDK path
    if isinstance(result, tuple):
        return result[1]
    return result.structured_content


def test_tools_are_listed_with_descriptions(tools):
    listed = asyncio.run(build_server(tools).list_tools())
    assert {t.name for t in listed} == TOOLS
    for t in listed:
        assert t.description and t.input_schema["type"] == "object"
    fit = next(t for t in listed if t.name == "fit_model")
    assert set(fit.input_schema["required"]) == {"path", "target"}
    assert "tabicl-v2" in fit.input_schema["properties"]["model"]["description"]


def test_fit_predict_what_if_through_the_server(tools, history, tmp_path):
    server = build_server(tools)
    fit = call(
        server,
        "fit_model",
        {"path": history, "target": "churn", "name": "c", "drop": ["customer_id"]},
    )
    assert fit["classes"] == ["no", "yes"]
    pred = call(
        server,
        "predict",
        {"name": "c", "path": history, "output": str(tmp_path / "p.csv")},
    )
    assert pred["score_against_target"]["accuracy"] == 1.0
    what = call(server, "what_if", {"name": "c", "rows": [{"x": 1}]})
    assert what["results"][0]["prediction"] == "yes"
    assert [m["name"] for m in call(server, "list_models", {})["models"]] == ["c"]


def test_explain_and_evaluate_through_the_server(tools, history):
    server = build_server(tools)
    args = {"path": history, "target": "churn", "drop": ["customer_id"]}
    assert call(server, "explain", {**args, "top": 1})["importance"][0]["column"] == "x"
    out = call(server, "evaluate", {**args, "models": ["tabicl-v2"], "baseline": False})
    assert [r["model"] for r in out["results"]] == ["tabicl-v2"]
    assert call(server, "inspect_table", {"path": history})["rows"] == 80


@pytest.mark.parametrize(
    ("raised", "message"),
    [
        (ValueError("no model named 'nope'"), "no model named 'nope'"),
        (DatasetNotFoundError("model 'x' not found"), "model 'x' not found"),
        (ConnectionRefusedError(61, "Connection refused"), "Connection refused"),
    ],
)
def test_expected_errors_reach_the_agent_with_their_message(
    tools, monkeypatch, raised, message
):
    def fail(*args):
        raise raised

    monkeypatch.setattr(tools, "list_models", fail)
    with pytest.raises(ToolError, match=message):
        call(build_server(tools), "list_models", {})


def test_tool_error_reaches_the_agent_over_the_protocol(tools, history):
    """What an MCP client gets: is_error and the message as text."""
    from mcp.client import Client

    async def go():
        async with Client(build_server(tools)) as client:
            return await client.call_tool("predict", {"name": "nope", "path": history})

    result = asyncio.run(go())
    assert result.is_error
    assert "no model named 'nope'" in result.content[0].text


def test_cli_commands(tools, history, capsys):
    assert (
        cli.main(
            [
                "fit",
                history,
                "--target",
                "churn",
                "--drop",
                "customer_id",
                "--name",
                "c",
            ],
            tools,
        )
        == 0
    )
    assert json.loads(capsys.readouterr().out)["name"] == "c"

    assert cli.main(["what-if", "c", '{"x": 0}'], tools) == 0
    assert json.loads(capsys.readouterr().out)["results"][0]["prediction"] == "no"

    customer = pd.read_csv(history).iloc[0].customer_id
    args = [
        "what-if",
        "c",
        "[{}]",
        "--base-file",
        history,
        "--base-where",
        json.dumps({"customer_id": customer}),
    ]
    assert cli.main(args, tools) == 0
    assert json.loads(capsys.readouterr().out)["base_row"]

    assert (
        cli.main(
            [
                "evaluate",
                history,
                "--target",
                "churn",
                "--models",
                "tabicl-v2",
                "--no-baseline",
                "--drop",
                "customer_id",
            ],
            tools,
        )
        == 0
    )
    assert [r["model"] for r in json.loads(capsys.readouterr().out)["results"]] == [
        "tabicl-v2"
    ]

    assert (
        cli.main(
            [
                "explain",
                history,
                "--target",
                "churn",
                "--drop",
                "customer_id",
                "--top",
                "1",
            ],
            tools,
        )
        == 0
    )
    assert json.loads(capsys.readouterr().out)["importance"][0]["column"] == "x"

    assert cli.main(["inspect", history], tools) == 0
    assert json.loads(capsys.readouterr().out)["rows"] == 80

    assert cli.main(["models"], tools) == 0
    assert len(json.loads(capsys.readouterr().out)["models"]) == 1


def test_cli_errors_are_json_with_exit_1(tools, capsys):
    assert cli.main(["predict", "nope", "x.csv"], tools) == 1
    assert "no model named" in json.loads(capsys.readouterr().out)["error"]
    assert cli.main(["what-if", "c", "{not json"], tools) == 1
    assert "error" in json.loads(capsys.readouterr().out)


def test_cli_skill_is_packaged(capsys):
    assert cli.main(["skill"]) == 0
    text = capsys.readouterr().out
    assert text.startswith("---\nname: tabctx") and "tabctx-mcp predict" in text


def test_descriptions_have_no_docstring_indentation(tools):
    for spec in tool_specs(build_server(tools)):
        assert "\n    " not in spec["description"], spec["name"]


def test_cli_tools_and_call_go_through_the_server(tools, history, capsys):
    assert cli.main(["tools"], tools) == 0
    specs = json.loads(capsys.readouterr().out)["tools"]
    assert {s["name"] for s in specs} == TOOLS
    assert all(s["input_schema"]["type"] == "object" for s in specs)

    args = {"path": history, "target": "churn", "name": "c", "drop": ["customer_id"]}
    assert cli.main(["call", "fit_model", json.dumps(args)], tools) == 0
    assert json.loads(capsys.readouterr().out)["name"] == "c"
    assert cli.main(["call", "list_models"], tools) == 0
    assert len(json.loads(capsys.readouterr().out)["models"]) == 1


@pytest.mark.parametrize(
    ("argv", "message"),
    [
        (["call", "nope"], "no tool named 'nope'"),
        (["call", "predict", '{"name": "x", "path": "y.csv"}'], "no model named 'x'"),
        (["call", "predict", '{"nme": "x"}'], "Field required"),
    ],
)
def test_cli_call_errors(tools, capsys, argv, message):
    assert cli.main(argv, tools) == 1
    assert message in json.loads(capsys.readouterr().out)["error"]


def test_cli_pi_extension_is_packaged(capsys):
    assert cli.main(["pi-extension"]) == 0
    text = capsys.readouterr().out
    assert "pi.registerTool" in text and '"call"' in text and '["tools"]' in text
