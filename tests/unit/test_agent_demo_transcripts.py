"""The agent demo UI's transcript parsers (examples/agent_demo/ui/
transcripts.py): each agent's JSON lines become the same events."""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(
    0, str(Path(__file__).resolve().parents[2] / "examples" / "agent_demo" / "ui")
)
from transcripts import parse_lines, tool_name  # noqa: E402

RESULT = {"rows": 10, "columns": 3}


def lines(*objs):
    return [json.dumps(o) for o in objs]


def kinds(events):
    return [(e["type"], e.get("tool")) for e in events]


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("mcp__tabctx__predict", ("predict", True)),
        ("tabctx_what_if", ("what_if", True)),
        ("predict", ("predict", True)),
        ("Write", ("Write", False)),
        ("tabctx_unknown", ("tabctx_unknown", False)),
    ],
)
def test_tool_names(raw, expected):
    assert tool_name(raw) == expected


def test_claude():
    events = parse_lines(
        "claude",
        lines(
            {"type": "system", "subtype": "init"},
            {
                "type": "assistant",
                "message": {
                    "content": [
                        {"type": "text", "text": "Looking at the data."},
                        {
                            "type": "tool_use",
                            "id": "t1",
                            "name": "mcp__tabctx__inspect_table",
                            "input": {"path": "a.csv"},
                        },
                    ]
                },
            },
            {
                "type": "user",
                "message": {
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": "t1",
                            "content": json.dumps(RESULT),
                        },
                    ]
                },
            },
            {
                "type": "assistant",
                "message": {
                    "content": [
                        {
                            "type": "tool_use",
                            "id": "t2",
                            "name": "mcp__tabctx__predict",
                            "input": {},
                        },
                    ]
                },
            },
            {
                "type": "user",
                "message": {
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": "t2",
                            "is_error": True,
                            "content": [{"type": "text", "text": "no model named 'x'"}],
                        },
                    ]
                },
            },
            {
                "type": "result",
                "duration_ms": 12345,
                "num_turns": 3,
                "total_cost_usd": 0.2,
            },
        ),
    )
    assert kinds(events) == [
        ("message", None),
        ("tool_call", "inspect_table"),
        ("tool_result", "inspect_table"),
        ("tool_call", "predict"),
        ("tool_result", "predict"),
        ("done", None),
    ]
    assert events[2]["result"] == RESULT and events[2]["tabctx"]
    assert events[4]["error"] and events[4]["text"] == "no model named 'x'"
    assert events[5]["stats"] == {"seconds": 12.3, "turns": 3, "cost_usd": 0.2}


def test_codex():
    call = {
        "id": "i1",
        "type": "mcp_tool_call",
        "server": "tabctx",
        "tool": "fit_model",
        "arguments": {"name": "c"},
    }
    events = parse_lines(
        "codex",
        lines(
            {
                "type": "item.completed",
                "item": {"id": "i0", "type": "agent_message", "text": "Fitting."},
            },
            {"type": "item.started", "item": {**call, "status": "in_progress"}},
            {
                "type": "item.completed",
                "item": {
                    **call,
                    "result": {
                        "content": [{"type": "text", "text": json.dumps(RESULT)}]
                    },
                },
            },
            {
                "type": "item.started",
                "item": {
                    "id": "i2",
                    "type": "command_execution",
                    "command": "cat > report.md",
                },
            },
            {
                "type": "item.completed",
                "item": {
                    "id": "i2",
                    "type": "command_execution",
                    "aggregated_output": "",
                    "exit_code": 0,
                },
            },
            {"type": "error", "message": "Reconnecting... 1/5"},
            {"type": "turn.completed", "usage": {"output_tokens": 99}},
        ),
    )
    assert kinds(events) == [
        ("message", None),
        ("tool_call", "fit_model"),
        ("tool_result", "fit_model"),
        ("tool_call", "shell"),
        ("tool_result", "shell"),
        ("done", None),
    ]
    assert events[2]["result"] == RESULT and not events[4]["error"]


def test_codex_failure():
    events = parse_lines(
        "codex", lines({"type": "turn.failed", "error": {"message": "422 tool_search"}})
    )
    assert events == [{"type": "error", "text": "422 tool_search"}]


def test_pi():
    events = parse_lines(
        "pi",
        lines(
            {
                "type": "tool_execution_start",
                "toolCallId": "p1",
                "toolName": "tabctx_explain",
                "args": {"path": "a.csv"},
            },
            {
                "type": "tool_execution_end",
                "toolCallId": "p1",
                "toolName": "tabctx_explain",
                "result": {
                    "content": [{"type": "text", "text": "{}"}],
                    "details": RESULT,
                },
                "isError": False,
            },
            {
                "type": "message_end",
                "message": {
                    "role": "user",
                    "content": [{"type": "text", "text": "the task"}],
                },
            },
            {
                "type": "message_end",
                "message": {
                    "role": "assistant",
                    "content": [{"type": "text", "text": "Done."}],
                },
            },
            {"type": "agent_end"},
        ),
    )
    assert kinds(events) == [
        ("tool_call", "explain"),
        ("tool_result", "explain"),
        ("message", None),
        ("done", None),
    ]
    assert events[1]["result"] == RESULT


def test_gemini():
    events = parse_lines(
        "gemini",
        lines(
            {"type": "init", "model": "gemini"},
            {
                "type": "message",
                "role": "assistant",
                "content": "Let me ",
                "delta": True,
            },
            {"type": "message", "role": "assistant", "content": "look.", "delta": True},
            {
                "type": "tool_use",
                "tool_name": "what_if",
                "tool_id": "g1",
                "parameters": {"name": "c"},
            },
            {
                "type": "tool_result",
                "tool_id": "g1",
                "status": "success",
                "output": json.dumps(RESULT),
            },
            {"type": "result", "status": "success", "stats": {"duration_ms": 2000}},
        ),
    )
    assert kinds(events) == [
        ("message", None),
        ("tool_call", "what_if"),
        ("tool_result", "what_if"),
        ("done", None),
    ]
    assert events[0]["text"] == "Let me look." and events[2]["result"] == RESULT


def test_junk_lines_are_skipped():
    assert (
        parse_lines("claude", ["", "not json", "[1, 2]", json.dumps({"type": "other"})])
        == []
    )
