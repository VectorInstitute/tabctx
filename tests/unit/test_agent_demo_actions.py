"""The agent demo's pictures of the agent's own commands (examples/agent_demo/
ui/static/actions.js), run under node on the recorded Claude Code run: each
command gets a plain-English title, and a picture shows only what the
command or its output says."""

import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

DEMO = Path(__file__).resolve().parents[2] / "examples" / "agent_demo"
sys.path.insert(0, str(DEMO / "ui"))
from transcripts import parse_lines  # noqa: E402

ACTIONS = DEMO / "ui" / "static" / "actions.js"
SHOWCASE = DEMO / "showcase" / "insurance" / "claude" / "transcript.jsonl"

pytestmark = pytest.mark.skipif(shutil.which("node") is None, reason="needs node")

# For each command: its kind, title and picture, from actions.js.
HARNESS = """
const Actions = require(process.argv[1]);
const calls = JSON.parse(require("fs").readFileSync(0, "utf8"));
console.log(JSON.stringify(calls.map(([tool, args, text]) => {
  const c = Actions.classify(tool, args);
  return { ...c, html: Actions.picture(c.kind, text, args) };
})));
"""


def run(calls):
    out = subprocess.run(
        ["node", "-e", HARNESS, str(ACTIONS)],
        input=json.dumps(calls),
        capture_output=True,
        text=True,
        check=True,
    )
    return json.loads(out.stdout)


@pytest.fixture(scope="module")
def showcase():
    with SHOWCASE.open() as f:
        events = parse_lines("claude", f)
    results = {e["id"]: e for e in events if e["type"] == "tool_result"}
    calls = [
        (e["tool"], e["args"] or {}, results.get(e["id"], {}).get("text") or "")
        for e in events
        if e["type"] == "tool_call" and not e["tabctx"]
    ]
    return calls, run(calls)


def test_every_command_gets_a_plain_title(showcase):
    _, out = showcase
    assert [o["kind"] for o in out] == [
        "look",
        "profile",
        "profile",
        "profile",
        "profile",
        "script",
        "edit",
        "run",
        "analyze",
        "prepare",
        "analyze",
        "edit",
        "verify",
        "edit",
        "verify",
        "report",
    ]
    assert out[0]["title"] == "Opened the files"
    assert out[5]["title"] == "Wrote the cleaning script"
    assert out[7]["title"] == "Ran the cleaning"
    assert all("python" not in o["title"].lower() for o in out)


def test_opening_the_files_shows_their_rows(showcase):
    html = showcase[1][0]["html"]
    # wc -l counts the header; the picture counts rows.
    assert "renewals_2025_export.csv" in html and "6,127 rows" in html
    assert "renewals_due_export.csv" in html and "1,212 rows" in html


def test_a_check_shows_what_it_found(showcase):
    html = showcase[1][1]["html"]
    assert "120 duplicate rows" in html
    assert "columns written many ways" in html


def test_running_the_cleaning_draws_its_own_printout(showcase):
    html = showcase[1][7]["html"]
    assert "6,127" in html and "6,000" in html
    assert "−120 duplicate rows" in html and "−6 test policies" in html
    assert "−1 TOTAL row" in html
    # province: 33 spellings collapse into the 9 codes the script printed
    assert "33&nbsp;&rarr;&nbsp;9" in html
    assert "454 placeholders emptied" in html
    assert "36 impossible ages emptied" in html


def test_the_cleaning_script_lists_only_what_its_code_does():
    script = "df = df.drop_duplicates()\ndf['d'] = pd.to_datetime(df['d'])\n"
    (o,) = run([["Write", {"file_path": "clean.py", "content": script}, ""]])
    assert "remove copies" in o["html"] and "read every date format" in o["html"]
    assert "drop test rows" not in o["html"]


def test_output_it_cannot_read_draws_nothing():
    (o,) = run([["Bash", {"command": "python3 clean.py"}, "done"]])
    assert o["kind"] == "run" and o["html"] == ""
    (o,) = run([["Bash", {"command": "make all"}, ""]])
    assert o["kind"] == "other" and o["title"] == "Ran a command"


def test_codex_wrapped_commands_are_unwrapped():
    (o,) = run([["shell", {"command": "/bin/zsh -lc 'python3 clean.py'"}, ""]])
    assert o["kind"] == "run"
