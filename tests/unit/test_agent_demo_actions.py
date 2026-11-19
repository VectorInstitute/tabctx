"""The agent demo's pictures of the agent's own commands (examples/agent_demo/
ui/static/actions.js), run under node on the recorded Claude Code run: each
command gets a plain-English title, and a picture shows only what the
command or its output says."""

import json
import shutil
import subprocess
import sys
import time
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
        "script",
        "run",
        "edit",
        "run",
        "analyze",
        "prepare",
        "prepare",
        "analyze",
        "report",
    ]
    assert out[0]["title"] == "Opened the files"
    assert out[4]["title"] == "Wrote the cleaning script"
    assert out[5]["title"] == "Ran the cleaning"
    assert all("python" not in o["title"].lower() for o in out)


def test_opening_the_files_shows_their_rows(showcase):
    html = showcase[1][0]["html"]
    # wc -l counts the header; the picture counts rows.
    assert "renewals_2025_export.csv" in html and "6,127 rows" in html
    assert "renewals_due_export.csv" in html and "1,212 rows" in html


def test_a_check_shows_what_it_found(showcase):
    first, second = showcase[1][1]["html"], showcase[1][2]["html"]
    assert "120 duplicate rows" in first and "a totals row" in first
    # value counts printed as dicts: Prov {'ON': 1131, ' on': 321, ...}
    assert "columns written many ways" in first
    # describe() output: a min of -1 and a max of 999999
    assert "impossible values" in second and "placeholders for blanks" in second


def test_running_the_cleaning_draws_its_own_printout(showcase):
    html = showcase[1][5]["html"]
    # "6127 rows in, 6000 plans out", first file only
    assert "6,127" in html and "6,000" in html and "1,212" not in html
    assert "−7 non-plan rows" in html and "−120 duplicate rows" in html
    assert "−12 duplicate rows" not in html
    assert "43 impossible lives_covered emptied" in html


def test_a_cleaning_printout_that_lists_its_merges_draws_them():
    printout = """renewals_2025_export.csv
  6127 rows read
  - 454 placeholder values ('n/a', 'N/A', 'unknown', '-') set to empty
  - dropped 1 'TOTAL' summary row (sum of premiums, not a policy)
  - dropped 120 exact duplicate rows
  - province: 33 spellings -> ['AB', 'BC', 'MB', 'NB', 'NL', 'NS', 'ON', 'QC', 'SK']
  - age: 36 impossible values ([-1, 0, 199, 999]) set to empty
  - dates in 4 formats parsed; replaced by years_as_customer
  6000 policies written

renewals_due_export.csv
  1212 rows read
"""
    (o,) = run([["Bash", {"command": "python3 clean.py"}, printout]])
    html = o["html"]
    assert "6,127" in html and "6,000" in html
    assert "−1 TOTAL row" in html and "−120 duplicate rows" in html
    assert "33&nbsp;&rarr;&nbsp;9" in html  # 33 spellings collapse into 9
    assert "454 placeholders emptied" in html
    assert "36 impossible ages emptied" in html
    assert "4 date formats read" in html


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
    out = subprocess.run(
        ["node", "-e", UNWRAP, str(ACTIONS)], capture_output=True, text=True, check=True
    ).stdout
    assert json.loads(out) == ["wc -l a.csv", "ls", "head -3 x.csv"]


UNWRAP = """
const A = require(process.argv[1]);
console.log(JSON.stringify([
  A.commandOf("shell", { command: "/bin/zsh -lc 'wc -l a.csv'" }),
  A.commandOf("shell", { command: ["bash", "-lc", "ls"] }).replace(/^bash -lc /, ""),
  A.commandOf("bash", { command: "head -3 x.csv" }),
]));
"""


@pytest.mark.parametrize(
    ("tool", "args", "kind", "title"),
    [
        # Gemini CLI
        (
            "replace",
            {"file_path": "clean.py", "old_string": "a", "new_string": "b"},
            "edit",
            "Adjusted clean.py",
        ),
        ("read_file", {"absolute_path": "/w/notes.txt"}, "look", "Read notes.txt"),
        ("list_directory", {"path": "/w"}, "look", "Looked around the folder"),
        ("search_file_content", {"pattern": "x"}, "look", "Looked around the folder"),
        # pi
        (
            "edit",
            {"path": "clean.py", "oldText": "a", "newText": "b\nc"},
            "edit",
            "Adjusted clean.py",
        ),
        # a flag before the script
        ("bash", {"command": "python3 -u clean.py"}, "run", "Ran the cleaning"),
    ],
)
def test_each_agents_tool_shapes(tool, args, kind, title):
    (o,) = run([[tool, args, ""]])
    assert (o["kind"], o["title"]) == (kind, title)


def test_pi_edits_show_lines_in_and_out():
    (o,) = run([["edit", {"path": "clean.py", "oldText": "a", "newText": "b\nc"}, ""]])
    assert "−1 line" in o["html"] and "+2 lines" in o["html"]


@pytest.mark.parametrize(
    "command",
    [
        "python3 -c \"import pandas as pd; d=pd.read_csv('renewals_2025.csv'); print(d.x.diff())\"",
        "python3 -c \"import pandas as pd; d=pd.read_csv('a.csv'); print(d.price_diff)\"",
    ],
)
def test_a_pandas_diff_is_not_a_re_run_check(command):
    (o,) = run([["bash", {"command": command}, ""]])
    assert o["kind"] != "verify"


def test_hostile_output_is_escaped():
    bad = "<img src=x onerror=alert(1)>"
    out = run(
        [
            ["bash", {"command": "wc -l *.csv"}, f"  12 {bad}.csv\n"],
            [
                "bash",
                {"command": "python3 - <<EOF\npd.read_csv('a_export.csv')\nEOF"},
                f"-- {bad} nunique 7\n",
            ],
            [
                "bash",
                {"command": "python3 clean.py"},
                f"  10 rows read\n  - dropped 2 {bad}\n  - {bad}: 9 spellings -> [a, b]\n  8 rows written\n",
            ],
        ]
    )
    for o in out:
        assert "<img" not in o["html"]


def test_long_output_cannot_hang_the_page():
    huge = "dup" * 100_000 + "\n" * 50_000
    started = time.monotonic()
    run(
        [
            [
                "bash",
                {"command": "python3 - <<EOF\npd.read_csv('a_export.csv')\nEOF"},
                huge,
            ],
            ["bash", {"command": "wc -l a.csv"}, huge],
            ["bash", {"command": "python3 clean.py"}, huge],
        ]
    )
    assert time.monotonic() - started < 5
