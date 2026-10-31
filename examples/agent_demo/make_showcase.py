"""Copy finished runs into showcase/, the runs anyone can replay in the
hosted demo: the transcript (without the session details an agent CLI
logs at start, with local paths made relative and the login name masked), the report, and the
score against the real outcomes (so the predictions file isn't needed).

    uv run --extra mcp python examples/agent_demo/make_showcase.py claude codex pi
"""

from __future__ import annotations

import getpass
import json
import re
import shutil
import sys
from pathlib import Path

from score import score

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
    # The login as the owner, and on Linux often the group, in `ls -l`
    # output ("1 jdoe  staff", "1 jdoe jdoe").
    login = re.escape(getpass.getuser())
    text = re.sub(rf"(\s\d+ ){login}(\s+){login}(\s)", r"\1user\2user\3", text)
    return re.sub(rf"(\s\d+ ){login}(\s+\w+\s)", r"\1user\2", text)


def main(agents: list[str]) -> None:
    for agent in agents:
        run_dir = HERE / "runs" / agent
        out = HERE / "showcase" / agent
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
        (out / "score.json").write_text(json.dumps(score(run_dir), indent=2) + "\n")
        print(agent, len(kept), "events ->", out)


if __name__ == "__main__":
    main(sys.argv[1:] or ["claude", "codex", "pi"])
