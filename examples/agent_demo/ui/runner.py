"""Starting agents (through ../run.sh) and reading their runs back.

A run lives in a folder: run.sh writes transcript.jsonl, report.md and the
predictions there. Locally that is ../runs/<scenario>/<agent>; in the
hosted demo each visitor has their own folder, and
../showcase/<scenario>/<agent> holds the finished runs anyone can replay
(with score.json in place of the predictions).
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

sys.path.insert(0, str(DEMO))
import scenario as scenarios  # noqa: E402
from score import score  # noqa: E402

SCENARIO = scenarios.load()
RUNS = Path(os.environ.get("DEMO_RUNS", SCENARIO.runs))
SHOWCASE = SCENARIO.showcase
# What run.sh gives an agent besides the tabctx tools: a shell only when
# the scenario's task has it run code.
_SHELL = SCENARIO.needs_shell
# The hosted demo runs Claude Code's shell as the run's own user, without
# its sandbox (run.sh, CLAUDE_SANDBOX=off).
_CLAUDE_SHELL = (
    "MCP · shell"
    if os.environ.get("CLAUDE_SANDBOX") == "off"
    else "MCP · sandboxed shell"
)

AGENTS = {
    "claude": {
        "label": "Claude Code",
        "binary": "claude",
        "llm": "Claude (your login)",
        "tools": _CLAUDE_SHELL if _SHELL else "MCP · no shell",
    },
    "codex": {
        "label": "Codex",
        "binary": "codex",
        "llm": "gpt-5 via Vector gateway",
        "tools": (
            "MCP · shell"
            if os.environ.get("CODEX_SANDBOX") == "danger-full-access"
            else "MCP · sandboxed shell"
        ),
    },
    "gemini": {
        "label": "Gemini CLI",
        "binary": "gemini",
        "llm": "Gemini (your login)",
        "tools": "MCP · shell" if _SHELL else "MCP · no shell",
    },
    "pi": {
        "label": "pi",
        "binary": "pi",
        "llm": "Qwen3.8 27B, on-prem via Vector gateway",
        "tools": "pi extension · bash" if _SHELL else "pi extension · no shell",
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
    # The question and plan the run was given, kept here: the files beside
    # the run are in a folder its agent can write.
    ask_data: dict | None = None
    discard: bool = False  # stopped: delete the half-done run once it ends
    done: threading.Event = field(default_factory=threading.Event)

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

    @property
    def ask_file(self) -> Path:
        """Where the question and approved plan are kept on disk: inside a
        showcase run; beside any other (run.sh replaces its folder)."""
        if self.dir.resolve().is_relative_to(SHOWCASE.resolve()):
            return self.dir / "ask.json"
        return self.dir.parent / f".{self.agent}.ask.json"

    @property
    def ask(self) -> dict | None:
        if self.ask_data is not None:
            return self.ask_data
        try:
            ask = json.loads(self.ask_file.read_text())
        except (OSError, ValueError):
            return None
        ok = isinstance(ask, dict) and isinstance(ask.get("plan"), list)
        return ask if ok else None


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

    def start(
        self,
        agent: str,
        folder: Path,
        prompt: str,
        env: dict[str, str],
        ask: dict | None = None,
    ) -> Run:
        """Run run.sh for agent in folder (replacing what was there), with
        the prompt and extra environment (the key, the models); ask, an open
        question's question and plan, is kept for the page."""
        with self._lock:
            # Live until its watcher has cleaned up after it (killed what it
            # left behind, taken its folder back): its uid and folder stay taken.
            live = [r for r in self._runs.values() if not r.done.is_set()]
            if any(r.dir.parent == folder.parent for r in live):
                raise BusyError("a run is already going: wait for it to finish")
            if len(live) >= self.max_running:
                raise BusyError(
                    "the demo is busy with other runs; try again in a few minutes"
                )
            prompt_file = folder.parent / f".{agent}.prompt.md"
            folder.parent.mkdir(parents=True, exist_ok=True)
            prompt_file.write_text(prompt)
            ask_file = folder.parent / f".{agent}.ask.json"
            if ask:
                ask_file.write_text(json.dumps(ask))
            else:
                ask_file.unlink(missing_ok=True)
            user = {}
            if self.uids is not None:
                uid = self._free_uid(live)
                _own(folder.parent, uid)
                user = {"user": uid, "group": uid, "extra_groups": []}
            run = Run(agent, folder, ask_data=ask)
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
        """Stop every run under parent and wait until each is cleaned up."""
        for run in [r for r in self._runs.values() if r.dir.parent == parent]:
            self._end(run, "stopped: signed out")
            run.done.wait(timeout=30)

    def stop(self, folder: Path) -> bool:
        """Stop the run in folder (the viewer pressed Stop) and wait until it
        is cleaned up, so a new one can start at once. False if none was
        going. A half-done run is no use to replay, and (hosted) it would
        hide the recorded run of the same agent, so its folder goes too."""
        run = self._runs.get(folder)
        if not run or run.done.is_set() or run.process is None:
            return False
        run.discard = True
        self._end(run, "stopped by the presenter")
        run.done.wait(timeout=30)
        return True

    def _end(self, run: Run, reason: str) -> None:
        self._stop(run, reason)
        if run.process:
            try:
                run.process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(run.process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                run.process.wait()

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
        try:
            if run.uid is not None:
                # Anything the agent left running (detached from its process
                # group) dies before the uid goes to another visitor's run.
                _kill_uid(run.uid)
            if run.discard:
                shutil.rmtree(run.dir, ignore_errors=True)
                with self._lock:
                    if self._runs.get(run.dir) is run:
                        del self._runs[run.dir]
            if run.uid is not None:
                # Back to root: the next run given this uid (another
                # visitor's) can't read this one's folder.
                _own(run.dir.parent, 0)
        finally:
            run.done.set()  # only now may its uid and folder be used again

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
    server, as root, still reads it). uid 0 takes it back once a run ends."""
    for root, dirs, files in os.walk(folder):
        for name in (*dirs, *files):
            try:
                os.lchown(os.path.join(root, name), uid, uid)
            except FileNotFoundError:
                pass  # gone since the walk listed it
    os.chown(folder, uid, uid)
    os.chmod(folder, 0o700)


PROC = Path("/proc")


def _pids_of(uid: int) -> list[int]:
    """The processes running as uid (Linux: /proc; elsewhere none)."""
    pids = []
    for entry in PROC.glob("[0-9]*"):
        try:
            status = (entry / "status").read_text()
        except OSError:
            continue  # gone, or not ours to read
        for line in status.splitlines():
            if line.startswith("Uid:"):
                if uid in {int(x) for x in line.split()[1:]}:
                    pids.append(int(entry.name))
                break
    return pids


def _kill_uid(uid: int, rounds: int = 5) -> None:
    """Kill every process running as uid, again while new ones appear (a
    process can fork while its siblings are being killed)."""
    for _ in range(rounds):
        pids = _pids_of(uid)
        if not pids:
            return
        for pid in pids:
            try:
                os.kill(pid, signal.SIGKILL)
            except (ProcessLookupError, PermissionError):
                pass
        time.sleep(0.05)


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
