"""Copy finished runs into showcase/, the runs anyone can replay in the
hosted demo: the transcript (without the session details an agent CLI
logs at start, with local paths made relative and the login name masked), the report, and the
score against the real outcomes (so the predictions file isn't needed),
and, for a task with a cleaning step, the page's view of what it cleaned.

    uv run --extra mcp python examples/agent_demo/make_showcase.py claude codex pi

DEMO_SCENARIO picks the scenario (runs/<scenario>/ -> showcase/<scenario>/).
A recording of one of the scenario's suggested questions goes under
showcase/<scenario>/questions/<id>/, from a run folder given with --from:

    make_showcase.py --question calls --from <run dir> claude
"""

from __future__ import annotations

import argparse
import getpass
import json
import re
import shutil
import sys
from pathlib import Path

import scenario as scenarios
from score import score

sys.path.insert(0, str(Path(__file__).resolve().parent / "ui"))
from cleaning import SAVED, for_run  # noqa: E402

HERE = Path(__file__).resolve().parent
# Events that record the machine, not the run (the session's tools, cwd,
# ids and account details), and pi's token-by-token streaming, which the
# UI doesn't read (it uses each message's end).
_DROP = {
    ("claude", "system"),
    ("codex", "thread.started"),
    ("pi", "session"),
    ("pi", "message_update"),
    ("pi", "message_start"),
}


def sanitize(agent: str, line: str, run_dir: Path) -> str | None:
    event = json.loads(line)
    if (agent, event.get("type")) in _DROP:
        return None
    text = json.dumps(event)
    for path in sorted({str(run_dir), str(run_dir.resolve())}, key=len, reverse=True):
        text = text.replace(path + "/", "").replace(path, ".")
    text = text.replace(str(HERE.parent.parent), "<repo>")
    text = re.sub(r"/(Users|home)/[^/\"\\\\]+", "~", text)
    # The agent's sandbox temp folder names the local user id.
    # Claude Code's temp folders, which name the user id and, by project,
    # the home folder ("claude-501/-Users-jdoe-src-x").
    text = re.sub(r"claude-\d+/-(Users|home)-[A-Za-z0-9._-]+", "claude", text)
    text = re.sub(r"(/private)?/tmp/claude-\d+", "/tmp", text)
    # The login as the owner, and on Linux often the group, in `ls -l`
    # output ("1 jdoe  staff", "1 jdoe jdoe").
    login = re.escape(getpass.getuser())
    text = re.sub(rf"(\s\d+ ){login}(\s+){login}(\s)", r"\1user\2user\3", text)
    return re.sub(rf"(\s\d+ ){login}(\s+\w+\s)", r"\1user\2", text)


def main(
    agents: list[str], question: str | None = None, source: Path | None = None
) -> None:
    scenario = scenarios.load()
    if question:
        ids = [s[0] for s in scenario.task.LABELS.get("suggestions", [])]
        if question not in ids:
            sys.exit(f"no suggested question {question!r}: one of {ids}")
    for agent in agents:
        run_dir = source or scenario.runs / agent
        out = scenario.showcase / agent
        if question:
            out = scenario.showcase / "questions" / question / agent
        shutil.rmtree(out, ignore_errors=True)
        out.mkdir(parents=True)
        lines = (run_dir / "transcript.jsonl").read_text().splitlines()
        kept = [
            s
            for s in (sanitize(agent, line, run_dir) for line in lines if line.strip())
            if s
        ]
        (out / "transcript.jsonl").write_text("\n".join(kept) + "\n")
        shutil.copy(run_dir / "report.md", out / "report.md")
        # The run's question and plan: inside a run folder given with
        # --from (only there: its parent may hold another run's), else
        # beside it, where the page keeps it.
        ask = run_dir / "ask.json" if source else run_dir.parent / f".{agent}.ask.json"
        if ask.is_file():  # an open question: what was asked, the plan
            shutil.copy(ask, out / "ask.json")
        (out / "score.json").write_text(
            json.dumps(score(run_dir, scenario), indent=2) + "\n"
        )
        view = for_run(run_dir, scenario)
        if view:  # the page's messy -> tidy view, without the raw files
            (out / SAVED).write_text(json.dumps(view, indent=1) + "\n")
        print(agent, len(kept), "events ->", out)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("agents", nargs="*", default=["claude", "codex", "pi"])
    parser.add_argument("--question", help="a suggested question's id")
    parser.add_argument("--from", dest="source", type=Path, help="the run folder")
    args = parser.parse_args()
    if args.source and len(args.agents) != 1:
        parser.error("--from takes one agent")
    main(args.agents, args.question, args.source)
