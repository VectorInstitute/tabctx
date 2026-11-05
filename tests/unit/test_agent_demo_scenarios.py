"""The agent demo's scenarios (examples/agent_demo/scenario.py and
scenarios/): how one is picked, what run.sh gives each agent, and the
insurance scenario's synthetic data. No download, no agent, no model."""

import json
import os
import re
import stat
import subprocess
import sys
from pathlib import Path

import pytest

pd = pytest.importorskip("pandas")
np = pytest.importorskip("numpy")

DEMO = Path(__file__).resolve().parents[2] / "examples" / "agent_demo"
sys.path.insert(0, str(DEMO))
import scenario as scenarios  # noqa: E402
from scenarios.insurance import prepare_data as insurance_data  # noqa: E402
from scenarios.insurance import task as insurance  # noqa: E402

PAGE_STEPS = {"clean", "evaluate", "predict", "explain", "what_if", "report"}


def test_every_scenario_folder_is_listed():
    assert scenarios.names() == ["insurance", "readmission"]


def test_the_default_is_readmission_and_the_env_picks_another(monkeypatch):
    monkeypatch.delenv("DEMO_SCENARIO", raising=False)
    assert scenarios.load().name == "readmission"
    monkeypatch.setenv("DEMO_SCENARIO", "insurance")
    assert scenarios.load().task is insurance


def test_an_unknown_scenario_names_the_known_ones():
    with pytest.raises(SystemExit, match="use one of insurance, readmission"):
        scenarios.load("telecom")


@pytest.mark.parametrize("name", scenarios.names())
def test_the_prompt_names_the_files_and_the_steps_have_icons(name):
    scenario = scenarios.load(name)
    task = scenario.task
    prompt = scenario.prompt.read_text()
    assert task.HISTORY in prompt and task.CURRENT in prompt
    assert getattr(task, "RAW_TARGET", task.TARGET) in prompt
    assert {step[0] for step in task.LABELS["steps"]} <= PAGE_STEPS
    assert f"p_{task.TARGET}={task.POSITIVE}" == task.PROBA
    # Only a task that has the agent run code gets a shell, and says so.
    assert scenario.needs_shell == ("clean" in {s[0] for s in task.LABELS["steps"]})


@pytest.mark.parametrize("name", scenarios.names())
def test_run_sh_asks_the_scenario(name, tmp_path):
    env = {**os.environ, "DEMO_SCENARIO": name}
    env.pop("DEMO_WORKSPACE", None)
    task = scenarios.load(name).task

    def ask(what):
        out = subprocess.run(
            ["python3", "scenario.py", what],  # run.sh's python3: no pandas needed
            cwd=DEMO,
            env=env,
            capture_output=True,
            text=True,
            check=True,
        )
        return out.stdout.strip()

    assert ask("files").split() == [task.HISTORY, task.CURRENT]
    assert Path(ask("prompt")) == DEMO / "scenarios" / name / "PROMPT.md"
    assert Path(ask("workspace")) == DEMO / "workspace" / name
    assert ask("shell") == ("1" if getattr(task, "SHELL", False) else "0")


@pytest.mark.parametrize(
    ("name", "agents"),
    [("readmission", {"claude", "codex", "pi"}), ("insurance", {"claude"})],
)
def test_showcase_runs_match_the_task(name, agents):
    """The replayable runs were made with this task's data and scoring."""
    scenario = scenarios.load(name)
    assert {p.name for p in scenario.showcase.iterdir()} == agents
    for agent in agents:
        folder = scenario.showcase / agent
        stored = json.loads((folder / "score.json").read_text())
        assert {
            "roc_auc",
            "base_rate",
            "positive_in_top_10",
            "positive_in_top_100",
        } <= set(stored)
        assert scenario.task.CURRENT in (folder / "transcript.jsonl").read_text()


# ---- run.sh, with a stand-in agent that records how it was started ---------


def _executable(path: Path, body: str) -> Path:
    path.write_text("#!/bin/sh\n" + body)
    path.chmod(path.stat().st_mode | stat.S_IEXEC)
    return path


@pytest.fixture
def fake_agents(tmp_path):
    """A bin/ whose `claude` and `gemini` write their arguments (one per
    line) to args.txt in the run folder, and a stand-in tabctx-mcp."""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    record = 'for a in "$@"; do printf "%s\\n" "$a"; done > args.txt\n'
    _executable(bin_dir / "claude", record)
    _executable(bin_dir / "gemini", record)
    server = _executable(bin_dir / "tabctx-mcp", "exit 0\n")
    return bin_dir, server


def _run(name, agent, tmp_path, fake_agents):
    bin_dir, server = fake_agents
    task = scenarios.load(name).task
    workspace = tmp_path / "workspace"
    workspace.mkdir(exist_ok=True)
    for f in (task.HISTORY, task.CURRENT):
        (workspace / f).write_text("a\n1\n")
    run_dir = tmp_path / "run"
    env = {
        **os.environ,
        "PATH": f"{bin_dir}:{os.environ['PATH']}",
        "DEMO_SCENARIO": name,
        "DEMO_WORKSPACE": str(workspace),
        "RUN_DIR": str(run_dir),
        "TABCTX_MCP_BIN": str(server),
        "TABCTX_API_KEY": "vp_test",
    }
    subprocess.run(
        [str(DEMO / "run.sh"), agent], env=env, check=True, capture_output=True
    )
    return run_dir


def _arg(args: list[str], flag: str) -> str:
    return args[args.index(flag) + 1]


def test_claude_gets_a_sandboxed_shell_only_when_the_task_runs_code(
    tmp_path, fake_agents
):
    run_dir = _run("insurance", "claude", tmp_path, fake_agents)
    args = (run_dir / "args.txt").read_text().splitlines()
    assert _arg(args, "--tools").split(",")[-1] == "Bash"
    assert "Bash" in args[args.index("--allowedTools") :]
    sandbox = json.loads(_arg(args, "--settings"))["sandbox"]
    assert sandbox["enabled"] is True
    # Each run gets its own copy of the raw exports.
    assert (run_dir / insurance.HISTORY).is_file()

    run_dir = _run("readmission", "claude", tmp_path, fake_agents)
    args = (run_dir / "args.txt").read_text().splitlines()
    assert "Bash" not in _arg(args, "--tools") and "--settings" not in args


def test_gemini_keeps_its_shell_only_when_the_task_runs_code(tmp_path, fake_agents):
    run_dir = _run("insurance", "gemini", tmp_path, fake_agents)
    settings = json.loads((run_dir / ".gemini" / "settings.json").read_text())
    assert settings["excludeTools"] == []
    run_dir = _run("readmission", "gemini", tmp_path, fake_agents)
    settings = json.loads((run_dir / ".gemini" / "settings.json").read_text())
    assert settings["excludeTools"] == ["run_shell_command"]


# ---- the insurance data -----------------------------------------------------


@pytest.fixture(scope="module")
def small_book():
    """A small build: (history export, current export, current outcomes)."""
    old = insurance_data.HISTORY_ROWS, insurance_data.CURRENT_ROWS
    insurance_data.HISTORY_ROWS, insurance_data.CURRENT_ROWS = 1_500, 400
    try:
        return insurance_data.build(seed=3)
    finally:
        insurance_data.HISTORY_ROWS, insurance_data.CURRENT_ROWS = old


def test_the_exports_have_the_systems_headers(small_book):
    history, current, _ = small_book
    headers = list(insurance_data.HEADERS.values())
    assert sorted(history.columns) == sorted(headers)
    assert sorted(current.columns) == sorted(
        h for h in headers if h != insurance.RAW_TARGET
    )


def test_every_due_policy_is_in_the_current_export_and_none_in_the_history(
    small_book,
):
    history, current, answers = small_book
    ids = set(answers[insurance.ID])
    assert len(ids) == len(answers) == 400
    assert set(current[insurance.RAW_ID]) == ids
    assert ids.isdisjoint(history[insurance.RAW_ID])
    assert set(answers[insurance.TARGET]) == {"Yes", "No"}


def test_the_exports_carry_the_mess_the_agent_should_find(small_book):
    history, current, _ = small_book
    ids = history[insurance.RAW_ID]
    assert ids.duplicated().sum() >= 20 and current[insurance.RAW_ID].duplicated().any()
    assert ids.str.startswith("TEST-").sum() == 6
    assert ids.iloc[-1] == "TOTAL"  # a totals row at the bottom
    assert not current[insurance.RAW_ID].str.startswith(("TEST-", "TOTAL")).any()
    assert history["Prov"].nunique() > 9  # one province, several spellings
    assert history["Advisor?"].nunique() > 2
    assert {"$", "CAD"} <= {
        m for v in history["Annual Premium"] for m in re.findall(r"\$|CAD", str(v))
    }
    assert history["Premium Change"].astype(str).str.endswith("%").mean() > 0.3
    ages = pd.to_numeric(history["Age"], errors="coerce")
    assert (ages > 120).any() or (ages <= 0).any()


def _clean_like_an_agent(raw: pd.DataFrame) -> pd.DataFrame:
    """A reference cleaning: every messy cell back to its clean value."""
    inverse = {}
    for table, column in (
        (insurance_data.PROVINCES, "Prov"),
        (insurance_data.PRODUCTS, "Product Type"),
        (insurance_data.PAYMENT, "Pay Method"),
    ):
        inverse[column] = {s: name for name, v in table.items() for s in v[1]}
    out = pd.DataFrame({insurance.ID: raw[insurance.RAW_ID]})
    for column, mapping in inverse.items():
        out[column] = raw[column].map(mapping)
    out["premium"] = raw["Annual Premium"].map(insurance_data._amount)
    out["change"] = raw["Premium Change"].astype(str).str.rstrip("%").astype(float)
    out["renewal"] = raw["Renewal Date"].map(
        lambda d: pd.to_datetime(d, format="mixed")
    )
    return out


def test_the_mess_is_recoverable(small_book):
    """Nothing the export does loses information a careful cleaner needs:
    its spellings, amounts, percents and dates all map back."""
    history, _, _ = small_book
    rows = history[~history[insurance.RAW_ID].str.startswith(("TEST-", "TOTAL"))]
    clean = _clean_like_an_agent(rows)
    assert clean.notna().all().all()
    assert clean["Prov"].nunique() == len(insurance_data.PROVINCES)
    assert clean["premium"].min() > 0
    assert clean["renewal"].dt.year.eq(2025).all()


def test_the_levers_move_cancellation_the_way_the_story_says():
    book = insurance_data.book(
        20_000, insurance_data.HISTORY_RENEWALS, np.random.default_rng(0)
    )
    cancelled = book[insurance.TARGET] == "Yes"
    assert 0.1 < cancelled.mean() < 0.22

    def rate(mask):
        return cancelled[mask].mean()

    hike = book["premium_change_pct"] >= 9
    contacted = book["advisor_contacts_12m"] >= 1
    assert rate(hike) > rate(~hike) + 0.08
    assert rate(book["payment_method"] == "Cheque") > rate(
        book["payment_method"] == "Pre-authorized debit"
    )
    assert rate(contacted) < rate(~contacted)
    # A check-in softens a hike: the gap a hike opens is smaller for them.
    assert rate(hike & contacted) - rate(~hike & contacted) < rate(
        hike & ~contacted
    ) - rate(~hike & ~contacted)


def test_no_values_pile_up_on_a_bound():
    """Ages and tenures are redrawn, not clipped, into their range: a spike
    at the edge looks like a data-entry default to anyone cleaning it."""
    book = insurance_data.book(
        20_000, insurance_data.HISTORY_RENEWALS, np.random.default_rng(1)
    )
    ages = book["age"].value_counts()
    assert ages.index.min() >= 21 and ages.index.max() <= 84
    assert ages[21] < 2 * ages[22]
    years = (book["renewal_date"] - book["policy_start"]).dt.days / 365.25
    assert years.min() >= 0.59 and (years < 0.65).mean() < 0.01


def test_main_writes_the_workspace_and_keeps_the_answers_apart(tmp_path, monkeypatch):
    monkeypatch.delenv("DEMO_WORKSPACE", raising=False)
    monkeypatch.setattr(insurance_data, "HISTORY_ROWS", 300)
    monkeypatch.setattr(insurance_data, "CURRENT_ROWS", 80)
    from dataclasses import replace

    scenario = replace(scenarios.load("insurance"), root=tmp_path)
    scenario.prepare()
    assert sorted(p.name for p in scenario.workspace.iterdir()) == sorted(
        [insurance.HISTORY, insurance.CURRENT]
    )
    answers = pd.read_csv(scenario.answers / insurance.ANSWERS)
    assert len(answers) == 80
    history = pd.read_csv(scenario.workspace / insurance.HISTORY)
    assert 0.05 < insurance.positive_rate(history) < 0.3
