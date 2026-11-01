"""Starting agents (through ../run.sh) and reading their runs back.

A run lives in a folder: run.sh writes transcript.jsonl, report.md and the
predictions there. Locally that is ../runs/<agent>; in the hosted demo each
visitor has their own folder, and ../showcase/<agent> holds the finished
runs anyone can replay (with score.json in place of the predictions).
"""

from __future__ import annotations

import json
import os
import shutil
import signal
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

DEMO = Path(__file__).resolve().parent.parent
RUNS = Path(os.environ.get("DEMO_RUNS", DEMO / "runs"))
SHOWCASE = DEMO / "showcase"

sys.path.insert(0, str(DEMO))
from score import score  # noqa: E402

AGENTS = {
    "claude": {
        "label": "Claude Code",
        "binary": "claude",
        "llm": "Claude (your login)",
        "tools": "MCP · no shell",
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
        "tools": "MCP · no shell",
    },
    "pi": {
        "label": "pi",
        "binary": "pi",
        "llm": "Qwen3.8 27B, on-prem via Vector gateway",
        "tools": "pi extension · no shell",
    },
}


def installed(agent: str) -> bool:
    binary = os.environ.get("PI_BIN") if agent == "pi" else None
    return (
        bool(binary and Path(binary).is_file())
        or shutil.which(AGENTS[agent]["binary"]) is not None
    )


@dataclass
class Run:
    agent: str
    dir: Path
    process: subprocess.Popen | None = None
    started: float = field(default_factory=time.time)
    log: list[str] = field(default_factory=list)
    timed_out: bool = False
    uid: int | None = None

    @property
    def transcript(self) -> Path:
        return self.dir / "transcript.jsonl"

    @property
    def running(self) -> bool:
        return self.process is not None and self.process.poll() is None

    @property
    def failed(self) -> bool:
        return self.timed_out or (
            self.process is not None and self.process.poll() not in (None, 0)
        )

    @property
    def exists(self) -> bool:
        return self.running or self.transcript.is_file()


class BusyError(RuntimeError):
    """A run can't start now (one is going, or the demo is at capacity)."""


class Runner:
    def __init__(
        self,
        max_running: int = 3,
        timeout_seconds: float = 15 * 60,
        uids: range | None = None,
    ) -> None:
        """max_running: agent runs at once across all visitors (they share
        the instance's CPU). timeout_seconds: a run is stopped after this.
        uids: when set (the hosted demo, running as root), each run gets a
        user id of its own from this range, and its folder is that user's
        alone: one visitor's agent can't read another's files or the
        environment (and so the key) of another's processes."""
        self.max_running = max_running
        self.timeout_seconds = timeout_seconds
        self.uids = uids
        self._lock = threading.Lock()
        self._runs: dict[Path, Run] = {}

    def start(self, agent: str, folder: Path, prompt: str, env: dict[str, str]) -> Run:
        """Run run.sh for agent in folder (replacing what was there), with
        the prompt and extra environment (the key, the models)."""
        with self._lock:
            live = [r for r in self._runs.values() if r.running]
            if any(r.dir.parent == folder.parent for r in live):
                raise BusyError("a run is already going: wait for it to finish")
            if len(live) >= self.max_running:
                raise BusyError(
                    "the demo is busy with other runs; try again in a few minutes"
                )
            prompt_file = folder.parent / f".{agent}.prompt.md"
            folder.parent.mkdir(parents=True, exist_ok=True)
            prompt_file.write_text(prompt)
            user = {}
            if self.uids is not None:
                uid = self._free_uid(live)
                _own(folder.parent, uid)
                user = {"user": uid, "group": uid, "extra_groups": []}
            run = Run(agent, folder)
            run.uid = user.get("user")
            run.process = subprocess.Popen(
                [str(DEMO / "run.sh"), agent],
                env={
                    **os.environ,
                    **env,
                    "PROMPT_FILE": str(prompt_file),
                    "RUN_DIR": str(folder),
                },
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                start_new_session=True,  # its own process group, to stop it whole
                **user,
            )
            self._runs[folder] = run
        threading.Thread(target=self._watch, args=(run,), daemon=True).start()
        return run

    def _free_uid(self, live: list[Run]) -> int:
        taken = {r.uid for r in live}
        uid = next((u for u in self.uids if u not in taken), None)
        if uid is None:
            raise BusyError(
                "the demo is busy with other runs; try again in a few minutes"
            )
        return uid

    def get(self, agent: str, folder: Path) -> Run:
        """The live run in folder, else the finished run on disk."""
        run = self._runs.get(folder)
        return run if run and run.agent == agent else Run(agent, folder)

    def stop_in(self, parent: Path) -> None:
        """Stop every run under parent and wait for it to end."""
        for run in [r for r in self._runs.values() if r.dir.parent == parent]:
            self._stop(run, reason="stopped: signed out")
            if run.process:
                try:
                    run.process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    os.killpg(run.process.pid, signal.SIGKILL)

    def running_in(self, parent: Path) -> Run | None:
        return next(
            (r for r in self._runs.values() if r.running and r.dir.parent == parent),
            None,
        )

    def _watch(self, run: Run) -> None:
        """Keep run.sh's own output (its errors) and stop it at the timeout."""
        assert run.process and run.process.stdout
        timer = threading.Timer(
            self.timeout_seconds,
            self._stop,
            args=(run, f"stopped after {self.timeout_seconds / 60:.0f} minutes"),
        )
        timer.daemon = True
        timer.start()
        for line in run.process.stdout:
            run.log.append(line.rstrip())
            del run.log[:-50]
        run.process.wait()
        timer.cancel()

    def _stop(self, run: Run, reason: str) -> None:
        if run.running and run.process:
            run.timed_out = True
            run.log.append(reason)
            try:
                os.killpg(run.process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass


def _own(folder: Path, uid: int) -> None:
    """Give folder and everything in it to uid, readable by it alone (the
    server, as root, still reads it)."""
    for root, dirs, files in os.walk(folder):
        for name in (*dirs, *files):
            os.lchown(os.path.join(root, name), uid, uid)
    os.chown(folder, uid, uid)
    os.chmod(folder, 0o700)


def outcome(run: Run) -> dict:
    """The run's report and its score against the real outcomes."""
    report = run.dir / "report.md"
    return {
        "report": report.read_text() if report.is_file() else None,
        "score": run_score(run.dir),
        "log": run.log[-20:] if run.failed else [],
    }


def run_score(folder: Path) -> dict | None:
    """score.py on the run's predictions, or the stored score.json of a
    showcase run; None when there is nothing to score yet."""
    stored = folder / "score.json"
    if stored.is_file():
        return json.loads(stored.read_text())
    try:
        return score(folder)
    except SystemExit as e:
        return {"error": str(e)}


def finished(folder: Path) -> dict[str, dict | None]:
    """Agents with a finished run under folder, and their scores."""
    if not folder.is_dir():
        return {}
    return {
        agent: run_score(folder / agent)
        for agent in AGENTS
        if (folder / agent / "transcript.jsonl").is_file()
    }
