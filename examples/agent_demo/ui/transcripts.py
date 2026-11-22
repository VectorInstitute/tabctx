"""Each agent's transcript (JSON lines) as one stream of events the page
draws, whatever the agent:

    {"type": "message", "text": ...}                      the agent talking
    {"type": "tool_call", "id", "tool", "tabctx", "args"} a tool starts
    {"type": "tool_result", "id", "tool", "tabctx",
     "result": dict | None, "text": str, "error": bool}   it finished
    {"type": "done", "stats": {...}}                      the agent finished
    {"type": "error", "text": ...}                        the agent failed

`tabctx` is true for the tabctx tools, whose names are normalized
(mcp__tabctx__predict, tabctx_predict -> predict) and whose results are
parsed into dicts. Parsers are incremental: feed() takes one line and
returns its events, so a live transcript can be streamed as it grows.
"""

from __future__ import annotations

import json
import re
from typing import Any

TABCTX_TOOLS = {
    "inspect_table",
    "evaluate",
    "fit_model",
    "predict",
    "what_if",
    "explain",
    "list_models",
}
_PREFIXES = ("mcp__tabctx__", "mcp_tabctx_", "tabctx__", "tabctx_", "tabctx.")

Event = dict[str, Any]


def tool_name(raw: str) -> tuple[str, bool]:
    """(name, is a tabctx tool)."""
    for prefix in _PREFIXES:
        if raw.startswith(prefix) and raw[len(prefix) :] in TABCTX_TOOLS:
            return raw[len(prefix) :], True
    return raw, raw in TABCTX_TOOLS


def _text(content: Any) -> str:
    """Text from a string or a list of {"type": "text", "text": ...} parts."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(
            p.get("text", "")
            for p in content
            if isinstance(p, dict) and p.get("type") == "text"
        )
    return json.dumps(content)


def _as_dict(text: str) -> dict | None:
    try:
        value = json.loads(text)
    except (TypeError, ValueError):
        return None
    return value if isinstance(value, dict) else None


class Parser:
    """Turns transcript lines into events; one subclass per agent."""

    def __init__(self) -> None:
        self._tools: dict[str, tuple[str, bool]] = {}

    def feed(self, line: str) -> list[Event]:
        line = line.strip()
        if not line:
            return []
        try:
            raw = json.loads(line)
        except ValueError:
            return []  # a CLI's own log line among the JSON
        return self._events(raw) if isinstance(raw, dict) else []

    def _events(self, raw: dict) -> list[Event]:
        raise NotImplementedError

    def finish(self) -> list[Event]:
        """Events held back when the transcript ends (a stopped run)."""
        return []

    def _call(self, call_id: str, raw_name: str, args: Any) -> Event:
        name, tabctx = tool_name(raw_name)
        self._tools[call_id] = (name, tabctx)
        return {
            "type": "tool_call",
            "id": call_id,
            "tool": name,
            "tabctx": tabctx,
            "args": args or {},
        }

    def _result(
        self, call_id: str, text: str, error: bool, result: dict | None = None
    ) -> Event:
        name, tabctx = self._tools.get(call_id, ("tool", False))
        if result is None and tabctx and not error:
            result = _as_dict(text)
        return {
            "type": "tool_result",
            "id": call_id,
            "tool": name,
            "tabctx": tabctx,
            "result": result,
            "text": text if result is None else "",
            "error": error,
        }


class ClaudeParser(Parser):
    """claude -p --output-format stream-json --verbose"""

    def _events(self, raw: dict) -> list[Event]:
        kind = raw.get("type")
        if kind == "result":
            if raw.get("is_error"):
                return [{"type": "error", "text": str(raw.get("result", ""))}]
            return [
                {
                    "type": "done",
                    "stats": {
                        "seconds": round(raw.get("duration_ms", 0) / 1000, 1),
                        "turns": raw.get("num_turns"),
                        "cost_usd": raw.get("total_cost_usd"),
                    },
                }
            ]
        # Only the assistant's and the tools' messages are objects; a system
        # notice (e.g. "permission_denied") carries its message as text.
        message = raw.get("message")
        content = message.get("content") if isinstance(message, dict) else None
        if not isinstance(content, list):
            return []
        events = []
        for part in content:
            if (
                kind == "assistant"
                and part.get("type") == "text"
                and part["text"].strip()
            ):
                events.append({"type": "message", "text": part["text"]})
            elif kind == "assistant" and part.get("type") == "tool_use":
                events.append(self._call(part["id"], part["name"], part.get("input")))
            elif kind == "user" and part.get("type") == "tool_result":
                events.append(
                    self._result(
                        part["tool_use_id"],
                        _text(part.get("content")),
                        bool(part.get("is_error")),
                    )
                )
        return events


class CodexParser(Parser):
    """codex exec --json"""

    def _events(self, raw: dict) -> list[Event]:
        kind = raw.get("type", "")
        if kind == "turn.completed":
            usage = raw.get("usage") or {}
            return [
                {"type": "done", "stats": {"output_tokens": usage.get("output_tokens")}}
            ]
        if kind in ("turn.failed", "error"):
            message = raw.get("message") or (raw.get("error") or {}).get("message", "")
            if kind == "error" and str(message).startswith("Reconnecting"):
                return []
            return [{"type": "error", "text": str(message)}]
        item = raw.get("item") or {}
        started, completed = kind == "item.started", kind == "item.completed"
        if item.get("type") == "agent_message" and completed:
            return [{"type": "message", "text": item.get("text", "")}]
        if item.get("type") == "mcp_tool_call":
            if started:
                return [
                    self._call(
                        item["id"],
                        f"mcp__{item.get('server')}__{item['tool']}",
                        item.get("arguments"),
                    )
                ]
            if completed:
                result = item.get("result") or {}
                error = item.get("error")
                text = (
                    (error or {}).get("message", "")
                    if isinstance(error, dict)
                    else str(error or "")
                )
                return [
                    self._result(
                        item["id"],
                        text or _text(result.get("content")),
                        bool(error) or item.get("status") == "failed",
                        result.get("structured_content") if not error else None,
                    )
                ]
        if item.get("type") == "command_execution":
            if started:
                return [
                    self._call(
                        item["id"], "shell", {"command": item.get("command", "")}
                    )
                ]
            if completed:
                return [
                    self._result(
                        item["id"],
                        item.get("aggregated_output", ""),
                        item.get("exit_code") not in (0, None),
                    )
                ]
        return []


class PiParser(Parser):
    """pi -p --mode json"""

    def _events(self, raw: dict) -> list[Event]:
        kind = raw.get("type")
        if kind == "tool_execution_start":
            return [self._call(raw["toolCallId"], raw["toolName"], raw.get("args"))]
        if kind == "tool_execution_end":
            result = raw.get("result") or {}
            details = result.get("details")
            return [
                self._result(
                    raw["toolCallId"],
                    _text(result.get("content")),
                    bool(raw.get("isError")),
                    details
                    if isinstance(details, dict) and not raw.get("isError")
                    else None,
                )
            ]
        if kind == "message_end":
            message = raw.get("message") or {}
            if message.get("role") != "assistant":
                return []
            text = _text(message.get("content"))
            events = [{"type": "message", "text": text}] if text.strip() else []
            if message.get("stopReason") == "error" or message.get("errorMessage"):
                events.append(
                    {
                        "type": "error",
                        "text": message.get("errorMessage", "the model failed"),
                    }
                )
            return events
        if kind == "agent_end":
            return [{"type": "done", "stats": {}}]
        return []


class GeminiParser(Parser):
    """gemini -p -o stream-json"""

    def __init__(self) -> None:
        super().__init__()
        self._message = ""

    def _flush(self) -> list[Event]:
        text, self._message = self._message, ""
        return [{"type": "message", "text": text}] if text.strip() else []

    def _events(self, raw: dict) -> list[Event]:
        kind = raw.get("type")
        if kind == "message" and raw.get("role") == "assistant":
            self._message += raw.get("content", "")
            return []
        events = self._flush()
        if kind == "tool_use":
            events.append(
                self._call(raw["tool_id"], raw["tool_name"], raw.get("parameters"))
            )
        elif kind == "tool_result":
            error = raw.get("status") != "success"
            text = raw.get("output") or (raw.get("error") or {}).get("message", "")
            events.append(self._result(raw["tool_id"], _text(text), error))
        elif kind == "result":
            if raw.get("status") not in (None, "success"):
                events.append(
                    {"type": "error", "text": json.dumps(raw.get("error") or raw)}
                )
            else:
                stats = raw.get("stats") or {}
                events.append(
                    {
                        "type": "done",
                        "stats": {
                            "seconds": round(stats.get("duration_ms", 0) / 1000, 1)
                        },
                    }
                )
        elif kind == "error":
            events.append({"type": "error", "text": raw.get("message", "")})
        return events


class AntigravityParser(Parser):
    """agy -p --output-format stream-json (the Antigravity CLI)"""

    # Its tools' parameters, as the other agents name them (the page reads
    # command and file_path).
    _FILE_TOOLS = {"write_file", "edit_file", "view_file", "read_file"}
    # States a step ends in; any other (ACTIVE, PENDING, ...) is still going.
    _ENDED = {"DONE", "ERROR", "FAILED", "CANCELLED", "CANCELED", "SKIPPED"}
    _EXIT = re.compile(r"Command exited with code (-?\d+)\.")

    def __init__(self) -> None:
        super().__init__()
        self._message = ""
        self._ended: set[str] = set()
        self._open: str | None = None  # a step without an index (to pair its end)
        self._n = 0

    def _flush(self) -> list[Event]:
        text, self._message = self._message, ""
        return [{"type": "message", "text": text}] if text.strip() else []

    def finish(self) -> list[Event]:
        return self._flush()  # a run stopped before its result event

    def _tool(self, name: str, params: dict) -> tuple[str, dict]:
        if name == "call_mcp_tool" and params.get("ServerName") == "tabctx":
            return f"tabctx_{params.get('ToolName', '')}", params.get("Arguments") or {}
        if name == "run_command":
            return name, {"command": params.get("CommandLine", "")}
        if name in self._FILE_TOOLS:
            return name, {"file_path": params.get("Path", "")}
        return name, params

    def _id(self, step: dict) -> str:
        if step.get("step_index") is not None:
            return f"{step.get('conversation_id', '')}:{step['step_index']}"
        # No index: a step's updates come in a row, so pair them in order.
        if self._open is None:
            self._n += 1
            self._open = f"step-{self._n}"
        return self._open

    def _command(self, output: str) -> tuple[str, bool]:
        """A run_command's output without its header, and whether it failed
        (a non-zero exit code; no code found is not a failure)."""
        code = self._EXIT.search(output)
        body = (
            output.partition("Command output:\n")[2]
            if "Command output:\n" in output
            else output
        )
        if code and body is output:
            body = (
                "" if "No output from the command." in output else output[code.end() :]
            )
        return body.strip("\n") + ("\n" if body.strip() else ""), bool(
            code and code.group(1) != "0"
        )

    def _events(self, raw: dict) -> list[Event]:
        kind = raw.get("event")
        if kind == "step_update":
            step = raw.get("step_update") or {}
            if step.get("step_type") == "agent_response":
                self._message += step.get("text_delta") or ""
                return self._flush() if step.get("state") == "DONE" else []
            if step.get("step_type") != "tool":
                return []
            events = self._flush()
            call_id = self._id(step)
            state = str(step.get("state", ""))
            ended = state in self._ENDED
            if call_id in self._ended:
                return events  # a repeated end
            info = step.get("tool_info") or {}
            name = step.get("tool_name") or info.get("name") or "tool"
            params = info.get("parameters") or {}
            # Wait for the arguments: an early update may not have them yet.
            if call_id not in self._tools and (params or ended):
                raw_name, args = self._tool(name, params)
                events.append(self._call(call_id, raw_name, args))
            if not ended:
                return events
            self._ended.add(call_id)
            if self._open == call_id:
                self._open = None
            output = _text(info.get("output") or "")
            error = state != "DONE"
            if name == "run_command":
                output, failed = self._command(output)
                error = error or failed
            events.append(self._result(call_id, output, error))
            return events
        events = self._flush()
        if kind == "result":
            result = raw.get("result") or {}
            if result.get("status") not in (None, "SUCCESS"):
                events.append({"type": "error", "text": json.dumps(result)[:2000]})
            else:
                events.append(
                    {
                        "type": "done",
                        "stats": {
                            "seconds": round(result.get("duration_seconds", 0), 1),
                            "turns": result.get("num_turns"),
                        },
                    }
                )
        elif kind == "error":
            events.append({"type": "error", "text": json.dumps(raw)[:2000]})
        return events


PARSERS: dict[str, type[Parser]] = {
    "claude": ClaudeParser,
    "codex": CodexParser,
    "gemini": GeminiParser,
    "antigravity": AntigravityParser,
    "pi": PiParser,
}


def parse_lines(agent: str, lines) -> list[Event]:
    parser = PARSERS[agent]()
    return [event for line in lines for event in parser.feed(line)] + parser.finish()
