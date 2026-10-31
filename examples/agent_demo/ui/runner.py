"""Starting agents (through ../run.sh) and reading their runs back."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

DEMO = Path(__file__).resolve().parent.parent
RUNS = DEMO / "runs"
PROMPTS = RUNS / ".prompts"

sys.path.insert(0, str(DEMO))
from score import score  # noqa: E402

AGENTS = {
    "claude": {
        "label": "Claude Code",
        "binary": "claude",
        "llm": "Claude (your login)",
        "tools": "MCP · file tools · no shell",
    },
    "codex": {
        "label": "Codex",
        "binary": "codex",
        "llm": "gpt-5 via Vector gateway",
        "tools": "MCP · sandboxed shell",
    },
    "gemini": {
        "label": "Gemini CLI",
        "binary": "gemini",
        "llm": "Gemini (your login)",
        "tools": "MCP · file tools · no shell",
    },
    "pi": {
        "label": "pi",
        "binary": "pi",
        "llm": "Sonnet 5.5 via Vector gateway",
        "tools": "pi extension · file tools · no bash",
    },
}


def available(agent: str) -> bool:
    binary = os.environ.get("PI_BIN") if agent == "pi" else None
    return (
        bool(binary and Path(binary).is_file())
        or shutil.which(AGENTS[agent]["binary"]) is not None
    )


@dataclass
class Run:
    agent: str
    process: subprocess.Popen | None = None
    started: float = field(default_factory=time.time)
    log: list[str] = field(default_factory=list)

    @property
    def dir(self) -> Path:
        return RUNS / self.agent

    @property
    def transcript(self) -> Path:
        return self.dir / "transcript.jsonl"

    @property
    def running(self) -> bool:
        return self.process is not None and self.process.poll() is None

    @property
    def failed(self) -> bool:
        return self.process is not None and self.process.poll() not in (None, 0)


class Runner:
    """One agent run at a time (they share the gateway key and the GPU)."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.current: Run | None = None

    def start(self, agent: str, prompt: str) -> Run:
        if agent not in AGENTS:
            raise ValueError(f"unknown agent {agent!r}")
        if not available(agent):
            raise ValueError(f"{AGENTS[agent]['label']} is not installed")
        if not os.environ.get("TABCTX_API_KEY"):
            raise ValueError("set TABCTX_API_KEY before starting the UI")
        with self._lock:
            if self.current and self.current.running:
                raise RuntimeError(f"{self.current.agent} is still running")
            PROMPTS.mkdir(parents=True, exist_ok=True)
            prompt_file = PROMPTS / f"{agent}.md"
            prompt_file.write_text(prompt)
            run = Run(agent)
            run.process = subprocess.Popen(
                [str(DEMO / "run.sh"), agent],
                env={**os.environ, "PROMPT_FILE": str(prompt_file)},
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
            )
            threading.Thread(target=_drain, args=(run,), daemon=True).start()
            self.current = run
            return run

    def get(self, agent: str) -> Run:
        if self.current and self.current.agent == agent:
            return self.current
        return Run(agent)  # a finished run on disk


def _drain(run: Run) -> None:
    """Keep run.sh's own output (its errors), without blocking it."""
    assert run.process and run.process.stdout
    for line in run.process.stdout:
        run.log.append(line.rstrip())
        del run.log[:-50]


def past_runs() -> list[dict]:
    out = []
    for agent in AGENTS:
        run = Run(agent)
        if not run.transcript.is_file():
            continue
        entry = {"agent": agent, "finished": run.transcript.stat().st_mtime}
        try:
            entry["score"] = score(run.dir)
        except SystemExit:
            entry["score"] = None
        out.append(entry)
    return out


def outcome(run: Run) -> dict:
    """The run's report and its score against the real outcomes."""
    report = run.dir / "report.md"
    try:
        scored = score(run.dir)
    except SystemExit as e:
        scored = {"error": str(e)}
    return {
        "report": report.read_text() if report.is_file() else None,
        "score": scored,
        "log": run.log[-20:] if run.failed else [],
    }
