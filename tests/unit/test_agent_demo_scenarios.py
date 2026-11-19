"""The agent demo's scenarios (examples/agent_demo/scenario.py and
scenarios/): how one is picked, what run.sh gives each agent, scoring a
run, the page's view of a raw export, and the insurance scenario's
synthetic data. No download, no agent, no model."""

import importlib
import json
import os
import re
import shlex
import stat
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import pytest

pd = pytest.importorskip("pandas")
np = pytest.importorskip("numpy")

DEMO = Path(__file__).resolve().parents[2] / "examples" / "agent_demo"
sys.path[:0] = [str(DEMO), str(DEMO / "ui")]
import scenario as scenarios  # noqa: E402
import score as score_module  # noqa: E402
from scenarios.insurance import prepare_data as insurance_data  # noqa: E402
from scenarios.insurance import task as insurance  # noqa: E402

PAGE_STEPS = {"clean", "evaluate", "predict", "explain", "what_if", "report"}
INSURANCE = scenarios.load("insurance")


def _clean_env(**extra: str) -> dict:
    """A subprocess environment without this shell's demo settings
    (PROMPT_FILE, CLAUDE_MODEL, ISOLATE_HOME, DEMO_* ...)."""
    keep = {k: os.environ[k] for k in ("PATH", "HOME", "TMPDIR") if k in os.environ}
    return {**keep, **extra}


# ---- picking a scenario ------------------------------------------------------


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


def test_a_run_folder_names_its_scenario(tmp_path, monkeypatch):
    monkeypatch.delenv("DEMO_SCENARIO", raising=False)
    assert scenarios.of_run(tmp_path / "insurance" / "claude").name == "insurance"
    # A folder elsewhere (a hosted visitor's) falls back to DEMO_SCENARIO.
    assert scenarios.of_run(tmp_path / "visitors" / "claude").name == "readmission"


@pytest.mark.parametrize("name", scenarios.names())
def test_the_prompt_names_the_files_and_the_steps_have_icons(name):
    scenario = scenarios.load(name)
    task = scenario.task
    prompt = scenario.prompt.read_text()
    assert task.HISTORY in prompt and task.CURRENT in prompt
    assert scenario.raw_target in prompt
    assert set(scenario.steps) <= PAGE_STEPS
    assert f"p_{task.TARGET}={task.POSITIVE}" == task.PROBA


def test_only_a_task_with_a_clean_step_needs_a_shell():
    assert INSURANCE.needs_shell and INSURANCE.steps[0] == "clean"
    assert not scenarios.load("readmission").needs_shell


@pytest.mark.parametrize("name", scenarios.names())
def test_run_sh_gets_everything_in_one_call(name):
    out = subprocess.run(
        ["python3", "scenario.py", "env"],  # run.sh's python3: no pandas needed
        cwd=DEMO,
        env=_clean_env(DEMO_SCENARIO=name),
        capture_output=True,
        text=True,
        check=True,
    )
    values = dict(line.split("=", 1) for line in shlex.split(out.stdout))
    scenario = scenarios.load(name)
    assert values == {
        "SCENARIO_NAME": name,
        "SCENARIO_HISTORY": scenario.task.HISTORY,
        "SCENARIO_CURRENT": scenario.task.CURRENT,
        "SCENARIO_PROMPT": str(DEMO / "scenarios" / name / "PROMPT.md"),
        "SCENARIO_WORKSPACE": str(DEMO / "workspace" / name),
        "SCENARIO_ANSWERS": str(DEMO / "answers" / name),
        "SCENARIO_SHELL_TOOLS": "1" if scenario.needs_shell else "0",
    }


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


def _run(name, agent, tmp_path, fake_agents, **extra):
    bin_dir, server = fake_agents
    task = scenarios.load(name).task
    workspace = tmp_path / "workspace"
    workspace.mkdir(exist_ok=True)
    for f in (task.HISTORY, task.CURRENT):
        (workspace / f).write_text("a\n1\n")
    run_dir = tmp_path / "run"
    env = _clean_env(
        PATH=f"{bin_dir}:{os.environ['PATH']}",
        DEMO_SCENARIO=name,
        DEMO_WORKSPACE=str(workspace),
        RUN_DIR=str(run_dir),
        TABCTX_MCP_BIN=str(server),
        TABCTX_API_KEY="vp_test",
        **extra,
    )
    subprocess.run(
        [str(DEMO / "run.sh"), agent], env=env, check=True, capture_output=True
    )
    return run_dir


def _arg(args: list[str], flag: str) -> str:
    return args[args.index(flag) + 1]


def test_claude_gets_bash_only_inside_its_sandbox(tmp_path, fake_agents):
    run_dir = _run("insurance", "claude", tmp_path, fake_agents)
    args = (run_dir / "args.txt").read_text().splitlines()
    assert _arg(args, "--tools").split(",")[-1] == "Bash"
    # Pre-allowing Bash would also allow a retry outside the sandbox.
    allowed = args[args.index("--allowedTools") + 1 : args.index("--output-format")]
    assert "Bash" not in allowed and "mcp__tabctx" in allowed
    settings = json.loads(_arg(args, "--settings"))
    assert settings["sandbox"] == {
        "enabled": True,
        "autoAllowBashIfSandboxed": True,
        "allowUnsandboxedCommands": False,
    }
    assert settings["permissions"]["deny"] == [f"Read(/{INSURANCE.answers}/**)"]
    # Each run gets its own copy of the raw exports.
    assert (run_dir / insurance.HISTORY).is_file()

    run_dir = _run("readmission", "claude", tmp_path, fake_agents)
    args = (run_dir / "args.txt").read_text().splitlines()
    assert "Bash" not in _arg(args, "--tools") and "--settings" not in args


def test_hosted_claude_runs_bash_as_its_run_user_without_the_sandbox(
    tmp_path, fake_agents
):
    run_dir = _run("insurance", "claude", tmp_path, fake_agents, CLAUDE_SANDBOX="off")
    args = (run_dir / "args.txt").read_text().splitlines()
    allowed = args[args.index("--allowedTools") + 1 : args.index("--output-format")]
    assert "Bash" in allowed
    settings = json.loads(_arg(args, "--settings"))
    assert "sandbox" not in settings
    assert settings["permissions"]["deny"] == [f"Read(/{INSURANCE.answers}/**)"]


def test_the_page_says_when_a_hosted_agent_has_no_sandbox(
    monkeypatch, tmp_path, fresh_app
):
    monkeypatch.setenv("DEMO_SCENARIO", "insurance")
    monkeypatch.setenv("DEMO_RUNS", str(tmp_path))
    monkeypatch.setenv("CLAUDE_SANDBOX", "off")
    monkeypatch.setenv("CODEX_SANDBOX", "danger-full-access")
    agents = fresh_app().AGENTS
    assert agents["claude"]["tools"] == agents["codex"]["tools"] == "MCP · shell"


def test_gemini_keeps_its_shell_only_when_the_task_runs_code(tmp_path, fake_agents):
    run_dir = _run("insurance", "gemini", tmp_path, fake_agents)
    settings = json.loads((run_dir / ".gemini" / "settings.json").read_text())
    assert settings["excludeTools"] == []
    run_dir = _run("readmission", "gemini", tmp_path, fake_agents)
    settings = json.loads((run_dir / ".gemini" / "settings.json").read_text())
    assert settings["excludeTools"] == ["run_shell_command"]


# ---- scoring a run ------------------------------------------------------------


@pytest.fixture
def scored(tmp_path):
    """INSURANCE with tmp data: 4 due policies, 2 cancelled, and a run
    folder named for the scenario."""
    scenario = replace(INSURANCE, root=tmp_path)
    scenario.answers.mkdir(parents=True)
    ids = ["GB-1", "GB-2", "GB-3", "GB-4"]
    truth = pd.DataFrame(
        {insurance.ID: ids, insurance.TARGET: ["Yes", "No", "Yes", "No"]}
    )
    truth.to_csv(scenario.answers / insurance.ANSWERS, index=False)
    run = tmp_path / "runs" / "insurance" / "claude"
    run.mkdir(parents=True)
    return scenario, run, ids


def _predictions(path: Path, ids, proba, mtime: int) -> None:
    pd.DataFrame({insurance.ID: ids, insurance.PROBA: proba}).to_csv(path, index=False)
    os.utime(path, (mtime, mtime))


def test_score_passes_over_a_newer_partial_table(scored):
    scenario, run, ids = scored
    # Every policy, one listed twice (as in a raw export); then a newer
    # what-if table for one policy.
    _predictions(run / "due.csv", ids + ["GB-1"], [0.9, 0.1, 0.8, 0.2, 0.9], 1_000)
    _predictions(run / "what_if.csv", ["GB-1"], [0.3], 2_000)
    out = score_module.score(run, scenario)
    assert out["file"] == "due.csv" and out["rows"] == 4 and out["roc_auc"] == 1.0


def test_score_names_a_missing_id_column(scored):
    scenario, run, _ = scored
    pd.DataFrame({"Plan ID": ["GB-1"], insurance.PROBA: [0.5]}).to_csv(
        run / "p.csv", index=False
    )
    with pytest.raises(SystemExit, match="no 'plan_id' column"):
        score_module.score(run, scenario)


def test_score_reads_the_scenario_from_the_run_folder(scored, monkeypatch):
    scenario, run, ids = scored
    _predictions(run / "due.csv", ids, [0.9, 0.1, 0.8, 0.2], 1_000)
    monkeypatch.delenv("DEMO_SCENARIO", raising=False)  # readmission otherwise
    load = scenarios.load
    monkeypatch.setattr(
        scenarios, "load", lambda name=None: replace(load(name), root=scenario.root)
    )
    assert score_module.score(run)["rows"] == 4


# ---- the page, on a raw export ---------------------------------------------


@pytest.fixture(scope="module")
def small_book():
    """A small build: (history export, current export, current outcomes)."""
    return insurance_data.build(seed=3, history_rows=1_500, current_rows=400)


@pytest.fixture
def fresh_app(monkeypatch):
    """Import ui/app.py anew (it reads its scenario on import), and drop it
    again so the next importer gets its own."""
    for name in ("app", "runner"):
        monkeypatch.delitem(sys.modules, name, raising=False)
    yield lambda: importlib.import_module("app")
    for name in ("app", "runner"):
        sys.modules.pop(name, None)


def test_the_page_reads_a_raw_export_through_the_scenario(
    small_book, tmp_path, monkeypatch, fresh_app
):
    history, current, _ = small_book
    history.to_csv(tmp_path / insurance.HISTORY, index=False)
    current.to_csv(tmp_path / insurance.CURRENT, index=False)
    monkeypatch.setenv("DEMO_SCENARIO", "insurance")
    monkeypatch.setenv("DEMO_WORKSPACE", str(tmp_path))
    monkeypatch.setenv("DEMO_RUNS", str(tmp_path / "runs"))
    monkeypatch.delenv("DEMO_HOSTED", raising=False)
    monkeypatch.delenv("CLAUDE_SANDBOX", raising=False)
    app = fresh_app()
    data = app.dataset()
    assert data["history_rows"] == len(history)
    assert insurance.RAW_ID not in data["columns"]
    assert len(data["columns"]) == len(insurance_data.HEADERS) - 2  # no id, outcome
    real = history[history[insurance.RAW_ID].str.startswith("GB-")].drop_duplicates()
    expected = real[insurance.RAW_TARGET].isin(insurance.SPELLINGS["Yes"]).mean()
    assert data["positive_rate"] == round(expected, 3)
    assert data["raw_id"] == insurance.RAW_ID
    assert data["steps"][0][0] == "clean"
    assert app.AGENTS["claude"]["tools"] == "MCP · sandboxed shell"


# ---- the insurance data -----------------------------------------------------


def test_the_exports_have_the_systems_headers(small_book):
    history, current, _ = small_book
    headers = list(insurance_data.HEADERS.values())
    assert sorted(history.columns) == sorted(headers)
    assert sorted(current.columns) == sorted(
        h for h in headers if h != insurance.RAW_TARGET
    )


def test_every_due_plan_is_in_the_current_export_and_none_in_the_history(
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
    assert history["Industry"].nunique() > 7
    assert history["Online Admin?"].nunique() > 2
    marks = {
        m for v in history["Annual Premium"] for m in re.findall(r"\$|CAD", str(v))
    }
    assert {"$", "CAD"} <= marks
    assert history["Renewal Increase"].astype(str).str.endswith("%").mean() > 0.3
    # The loss ratio is a percent in some rows and a fraction in others.
    loss = history["Loss Ratio (12m)"].astype(str)
    assert (
        loss.str.endswith("%").mean() > 0.3 and loss.str.startswith("0.").mean() > 0.2
    )
    lives = pd.to_numeric(history["Lives Covered"], errors="coerce")
    assert (lives > 50_000).any() and (lives <= 0).any()
    statuses = set(history[insurance.RAW_TARGET]) - {""}
    assert statuses == {s for v in insurance.SPELLINGS.values() for s in v}


def _clean_like_an_agent(raw: pd.DataFrame) -> pd.DataFrame:
    """A reference cleaning: every messy cell back to its clean value."""
    inverse = {}
    for table, column in (
        (insurance_data.PROVINCES, "Prov"),
        (insurance_data.INDUSTRIES, "Industry"),
        (insurance_data.PLAN_TYPES, "Plan Type"),
        (insurance_data.CHANNELS, "Channel"),
        (insurance_data.BILLING, "Billing"),
    ):
        inverse[column] = {s: name for name, v in table.items() for s in v[1]}
    inverse["Funding"] = {
        s: name for name, v in insurance_data.FUNDING.items() for s in v[0]
    }
    out = pd.DataFrame({insurance.ID: raw[insurance.RAW_ID]})
    for column, mapping in inverse.items():
        out[column] = raw[column].map(mapping)
    out["premium"] = raw["Annual Premium"].map(insurance_data._amount)
    out["increase"] = raw["Renewal Increase"].astype(str).str.rstrip("%").astype(float)
    loss = raw["Loss Ratio (12m)"].astype(str)
    out["loss"] = np.where(
        loss.str.endswith("%"),
        loss.str.rstrip("%").astype(float),
        pd.to_numeric(loss, errors="coerce") * 100,
    )
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
    assert clean["loss"].between(40, 130).all()  # both units read as one
    assert clean["renewal"].dt.year.eq(2025).all()


def test_the_levers_move_leaving_the_way_the_story_says():
    book = insurance_data.book(
        20_000, insurance_data.HISTORY_RENEWALS, np.random.default_rng(0)
    )
    left = book[insurance.TARGET] == "Yes"
    assert 0.08 < left.mean() < 0.18

    def rate(mask):
        return left[mask].mean()

    increase = book["rate_increase_pct"] >= 9
    reviewed = book["account_reviews_12m"] >= 1
    assert rate(increase) > rate(~increase) + 0.08
    # A big increase follows a bad claims year.
    assert book.loc[increase, "loss_ratio_pct"].mean() > 90
    assert rate(book["service_issues_12m"] >= 1) > rate(book["service_issues_12m"] == 0)
    assert rate(book["lives_covered"] < 20) > rate(book["lives_covered"] >= 200)
    assert rate(book["funding"] == "ASO") < rate(book["funding"] == "Fully insured")
    assert rate(book["billing"] == "Cheque") > rate(
        book["billing"] == "Pre-authorized debit"
    )
    assert rate(reviewed) < rate(~reviewed)
    # A review softens an increase: the gap an increase opens is smaller.
    gap_reviewed = rate(increase & reviewed) - rate(~increase & reviewed)
    assert gap_reviewed < rate(increase & ~reviewed) - rate(~increase & ~reviewed)
