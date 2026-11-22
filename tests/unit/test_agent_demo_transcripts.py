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


def test_claude_skips_a_system_notice_whose_message_is_text():
    events = parse_lines(
        "claude",
        lines(
            {
                "type": "system",
                "subtype": "permission_denied",
                "tool_name": "Bash",
                "message": "Heredoc with unquoted delimiter undergoes shell expansion",
            },
            {
                "type": "assistant",
                "message": {"content": [{"type": "text", "text": "Ok."}]},
            },
        ),
    )
    assert [e["type"] for e in events] == ["message"]


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


def test_antigravity():
    def step(i, state, kind, **extra):
        return {
            "event": "step_update",
            "step_update": {
                "conversation_id": "c",
                "step_index": i,
                "state": state,
                "step_type": kind,
                **extra,
            },
        }

    mcp = {
        "name": "call_mcp_tool",
        "parameters": {
            "Arguments": {"name": "m"},
            "ServerName": "tabctx",
            "ToolName": "predict",
        },
    }
    shell = {"name": "run_command", "parameters": {"CommandLine": "wc -l x.csv"}}
    events = parse_lines(
        "antigravity",
        lines(
            {"event": "init", "init": {"model": "gemini-3.8-flash-high"}},
            step(1, "ACTIVE", "agent_response", text_delta="Let me "),
            step(1, "DONE", "agent_response", text_delta="look."),
            step(2, "ACTIVE", "tool", tool_name="run_command", tool_info=shell),
            step(
                2,
                "DONE",
                "tool",
                tool_name="run_command",
                tool_info={
                    **shell,
                    "output": "Created At: x\nCommand exited with code 0.\n\nCommand output:\n 2 x.csv\n",
                },
            ),
            step(3, "ACTIVE", "tool", tool_name="call_mcp_tool", tool_info=mcp),
            step(
                3,
                "DONE",
                "tool",
                tool_name="call_mcp_tool",
                tool_info={**mcp, "output": json.dumps(RESULT)},
            ),
            step(
                4,
                "DONE",
                "tool",
                tool_name="run_command",
                tool_info={
                    "name": "run_command",
                    "parameters": {"CommandLine": "false"},
                    "output": "Command exited with code 1.\n\nNo output from the command.",
                },
            ),
            step(
                5,
                "DONE",
                "tool",
                tool_name="write_file",
                tool_info={
                    "name": "write_file",
                    "parameters": {"Path": "/w/report.md"},
                },
            ),
            {
                "event": "result",
                "result": {
                    "status": "SUCCESS",
                    "duration_seconds": 12.34,
                    "num_turns": 1,
                },
            },
        ),
    )
    assert kinds(events) == [
        ("message", None),
        ("tool_call", "run_command"),
        ("tool_result", "run_command"),
        ("tool_call", "predict"),
        ("tool_result", "predict"),
        ("tool_call", "run_command"),
        ("tool_result", "run_command"),
        ("tool_call", "write_file"),
        ("tool_result", "write_file"),
        ("done", None),
    ]
    assert events[0]["text"] == "Let me look."
    assert events[1]["args"] == {"command": "wc -l x.csv"}
    assert events[2]["text"] == " 2 x.csv\n" and not events[2]["error"]
    assert events[3]["tabctx"] and events[3]["args"] == {"name": "m"}
    assert events[4]["result"] == RESULT
    assert events[6]["error"]
    assert events[7]["args"] == {"file_path": "/w/report.md"}
    assert events[9]["stats"] == {"seconds": 12.3, "turns": 1}


def test_antigravity_failure():
    events = parse_lines(
        "antigravity",
        lines({"event": "result", "result": {"status": "ERROR", "error": "quota"}}),
    )
    assert events[0]["type"] == "error" and "quota" in events[0]["text"]


def _agy_step(i, state, kind="tool", **extra):
    return {
        "event": "step_update",
        "step_update": {
            "conversation_id": "c",
            "step_index": i,
            "state": state,
            "step_type": kind,
            **extra,
        },
    }


def _shell(i, state, output=None, cmd="wc -l x.csv"):
    info = {"name": "run_command", "parameters": {"CommandLine": cmd} if cmd else {}}
    if output is not None:
        info["output"] = output
    return _agy_step(i, state, tool_name="run_command", tool_info=info)


def test_antigravity_ends_a_step_once_and_waits_for_its_arguments():
    events = parse_lines(
        "antigravity",
        lines(
            _shell(1, "PENDING", cmd=None),  # no arguments yet: no call
            _shell(1, "ACTIVE"),
            _shell(1, "WAITING"),  # still going: no result
            _shell(
                1, "DONE", "Command exited with code 0.\n\nCommand output:\n2 x.csv\n"
            ),
            _shell(
                1, "DONE", "Command exited with code 0.\n\nCommand output:\n2 x.csv\n"
            ),
        ),
    )
    assert kinds(events) == [
        ("tool_call", "run_command"),
        ("tool_result", "run_command"),
    ]
    assert events[0]["args"] == {"command": "wc -l x.csv"}
    assert events[1]["text"] == "2 x.csv\n" and not events[1]["error"]


@pytest.mark.parametrize(
    ("state", "output", "text", "error"),
    [
        ("DONE", "plain output, no header", "plain output, no header\n", False),
        (
            "DONE",
            "Created At: x\nCommand exited with code 0.\n\nNo output from the command.",
            "",
            False,
        ),
        (
            "DONE",
            "Command exited with code 2.\n\nCommand output:\nboom\n",
            "boom\n",
            True,
        ),
        (
            "DONE",
            "Command exited with code 1.\n\nNo output from the command.",
            "",
            True,
        ),
        ("CANCELLED", "", "", True),
    ],
)
def test_antigravity_shell_results(state, output, text, error):
    events = parse_lines("antigravity", lines(_shell(1, state, output)))
    assert kinds(events) == [
        ("tool_call", "run_command"),
        ("tool_result", "run_command"),
    ]
    assert events[1]["text"] == text and events[1]["error"] is error


def test_antigravity_pairs_steps_without_an_index_and_flushes_a_stopped_run():
    def step(state, out=None):
        e = _shell(0, state, out)
        del e["step_update"]["step_index"]
        return e

    events = parse_lines(
        "antigravity",
        lines(
            step("ACTIVE"),
            step("DONE", "Command output:\na\n"),
            step("ACTIVE"),
            step("DONE", "Command output:\nb\n"),
            _agy_step(9, "ACTIVE", "agent_response", text_delta="Half a thought"),
        ),
    )
    ids = [e["id"] for e in events if e["type"] in ("tool_call", "tool_result")]
    assert ids[0] == ids[1] and ids[2] == ids[3] and ids[0] != ids[2]
    assert events[-1] == {
        "type": "message",
        "text": "Half a thought",
    }  # no result event came


def test_antigravity_other_mcp_servers_are_not_tabctx():
    info = {
        "name": "call_mcp_tool",
        "parameters": {"ServerName": "other", "ToolName": "predict", "Arguments": {}},
    }
    events = parse_lines(
        "antigravity",
        lines(
            _agy_step(
                1, "DONE", tool_name="call_mcp_tool", tool_info={**info, "output": "{}"}
            )
        ),
    )
    assert events[0]["tool"] == "call_mcp_tool" and not events[0]["tabctx"]
