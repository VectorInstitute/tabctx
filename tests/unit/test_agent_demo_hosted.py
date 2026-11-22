"""The hosted agent demo (examples/agent_demo/ui): which models a key gets,
sessions, and the web app's sign-in, run and replay endpoints, with the
gateway and the agent processes faked."""

import importlib
import sys
import time
from pathlib import Path

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("pandas")
pytest.importorskip("sklearn")
from fastapi.testclient import TestClient

UI = Path(__file__).resolve().parents[2] / "examples" / "agent_demo" / "ui"
sys.path.insert(0, str(UI))
import gateway  # noqa: E402
import sessions as sessions_module  # noqa: E402

ALL = {
    "tabpfn-3.5",
    "tabicl-v2",
    "claude-opus-5",
    "claude-haiku-5-5",
    "claude-sonnet-5-5",
    "gpt-5",
}
KEY = "vp_abc123.secretpart"


# ---- models per key ------------------------------------------------------------


def test_each_agent_gets_the_best_model_the_key_can_use():
    env = gateway.agent_env(ALL)
    assert env["claude"] == {
        "CLAUDE_VIA_GATEWAY": "1",
        "CLAUDE_MODEL": "claude-opus-5",
        "ANTHROPIC_DEFAULT_HAIKU_MODEL": "claude-haiku-5-5",
        "ANTHROPIC_SMALL_FAST_MODEL": "claude-haiku-5-5",
    }
    assert env["codex"] == {"CODEX_MODEL": "gpt-5"}
    assert env["pi"] == {"PI_MODEL": "claude-sonnet-5-5"}  # no Qwen in ALL
    assert gateway.model_of(env["pi"]) == "claude-sonnet-5-5"


def test_pi_prefers_the_on_prem_qwen():
    env = gateway.agent_env(ALL | {"Qwen3_8-27B"})
    assert env["pi"] == {"PI_MODEL": "Qwen3_8-27B"}


def test_agents_the_key_cannot_run():
    env = gateway.agent_env({"tabpfn-3.5", "claude-sonnet-5"})
    assert env["codex"] is None
    assert env["claude"]["CLAUDE_MODEL"] == "claude-sonnet-5"
    assert env["claude"]["ANTHROPIC_SMALL_FAST_MODEL"] == "claude-sonnet-5"  # no haiku


@pytest.mark.parametrize("key", ["", "sk-ant-123", "vp_has space", "vp_" + "x" * 600])
def test_malformed_keys_are_refused_before_any_request(key):
    with pytest.raises(gateway.RefusedKey) as e:
        gateway.project_models("http://unused.invalid", key)
    assert e.value.status == 400


# ---- sessions -------------------------------------------------------------------


def test_sessions_end_and_expire_and_clean_up(tmp_path):
    ended = []
    store = sessions_module.SessionStore(tmp_path, idle_seconds=60, on_end=ended.append)
    a = store.create(KEY, {})
    assert store.get(a.id) is a and a.key_id == "vp_abc123" and KEY not in repr(a)
    store.runs_dir(a).mkdir(parents=True)
    store.end(a.id)
    assert store.get(a.id) is None and not store.runs_dir(a).exists()
    assert ended == [store.runs_dir(a)]

    b = store.create(KEY, {})
    b.seen = time.time() - 120
    assert store.get(b.id) is None  # idle too long
    assert store.get(None) is None and store.get("forged") is None


# ---- the web app, hosted ------------------------------------------------------------


ASK = {
    "question": "Which provinces lose the most plans?",
    "plan": [
        {"kind": "analyze", "text": "Count leavers by province"},
        {"kind": "report", "text": "Write the answer"},
    ],
}


@pytest.fixture
def hosted(tmp_path, monkeypatch):
    return _hosted(tmp_path, monkeypatch)


@pytest.fixture
def open_hosted(tmp_path, monkeypatch):
    """The hosted image's scenario, which takes open questions."""
    return _hosted(tmp_path, monkeypatch, scenario="insurance")


def _hosted(tmp_path, monkeypatch, scenario=None):
    monkeypatch.setenv("DEMO_HOSTED", "1")
    monkeypatch.setenv("DEMO_RUNS", str(tmp_path / "runs"))
    monkeypatch.delenv("TABCTX_API_KEY", raising=False)
    # The hosted image's scenario and data, whatever this shell exports.
    monkeypatch.delenv("DEMO_SCENARIO", raising=False)
    monkeypatch.delenv("DEMO_WORKSPACE", raising=False)
    if scenario:
        monkeypatch.setenv("DEMO_SCENARIO", scenario)
    for name in ("app", "runner"):
        sys.modules.pop(name, None)
    module = importlib.import_module("app")
    monkeypatch.setattr(
        module, "project_models", lambda base, key: ALL if key == KEY else set()
    )
    monkeypatch.setattr(module, "installed", lambda agent: True)
    started = []

    def fake_start(agent, folder, prompt, env, ask=None):
        started.append(
            {"agent": agent, "folder": folder, "prompt": prompt, "env": env, "ask": ask}
        )
        return module.Run(agent, folder)

    monkeypatch.setattr(module.runner, "start", fake_start)
    module._attempts.clear()
    client = TestClient(module.app, base_url="https://demo.test")
    return module, client, started


def test_signed_out_visitors_can_replay_but_not_run(hosted):
    module, client, started = hosted
    setup = client.get("/api/setup").json()
    assert (
        setup["hosted"] and setup["signed_in"] is None and not setup["prompt_editable"]
    )
    assert [a["id"] for a in setup["agents"]] == ["claude", "codex", "pi"]  # no Gemini
    assert not any(a["allowed"] for a in setup["agents"])
    assert {r["agent"] for r in setup["runs"]} >= {"claude", "codex", "pi"}  # showcase
    assert client.post("/api/runs", json={"agent": "claude"}).status_code == 401
    events = client.get("/api/runs/codex/events").text
    assert "event: event" in events and "event: outcome" in events
    assert not started


def test_sign_in_run_and_sign_out(hosted):
    module, client, started = hosted
    r = client.post("/api/login", json={"key": f"  {KEY} "})
    assert r.status_code == 200 and r.json()["key_id"] == "vp_abc123"
    cookie = r.headers["set-cookie"].lower()
    assert "httponly" in cookie and "samesite=strict" in cookie and "secure" in cookie
    assert KEY not in r.text

    setup = client.get("/api/setup").json()
    assert setup["signed_in"] == {"key_id": "vp_abc123"} and KEY not in str(setup)
    assert all(a["allowed"] for a in setup["agents"])

    r = client.post(
        "/api/runs", json={"agent": "claude", "prompt": "do something else"}
    )
    assert r.status_code == 200
    run = started[-1]
    assert run["prompt"] == module.PROMPT.read_text()  # the task is fixed
    assert run["ask"] is None
    assert run["env"]["TABCTX_API_KEY"] == KEY and run["env"]["VP_API_KEY"] == KEY
    assert (
        run["env"]["ISOLATE_HOME"] == "1"
        and run["env"]["CLAUDE_MODEL"] == "claude-opus-5"
    )
    assert run["folder"].parent.parent == Path(module.RUNS) / "visitors"

    assert client.post("/api/runs", json={"agent": "gemini"}).status_code == 403
    client.post("/api/logout", json={})
    assert client.get("/api/setup").json()["signed_in"] is None
    assert client.post("/api/runs", json={"agent": "claude"}).status_code == 401


@pytest.mark.parametrize(
    ("models", "status", "message"),
    [
        (set(), 403, "no access to the tabular models"),
        ({"tabpfn-3.5"}, 403, "no model the demo's agents can use"),
    ],
)
def test_keys_without_the_models(hosted, monkeypatch, models, status, message):
    module, client, _ = hosted
    monkeypatch.setattr(module, "project_models", lambda base, key: models)
    r = client.post("/api/login", json={"key": KEY})
    assert r.status_code == status and message in r.json()["detail"]


def test_a_refused_key_and_the_attempt_limit(hosted, monkeypatch):
    module, client, _ = hosted

    def refuse(base, key):
        raise gateway.RefusedKey(401, "the gateway did not accept this key")

    monkeypatch.setattr(module, "project_models", refuse)
    for _ in range(10):
        assert client.post("/api/login", json={"key": "vp_x.y"}).status_code == 401
    assert client.post("/api/login", json={"key": "vp_x.y"}).status_code == 429


def test_posts_must_be_json_and_pages_carry_security_headers(hosted):
    _, client, _ = hosted
    r = client.post("/api/login", data={"key": KEY})
    assert r.status_code == 415
    page = client.get("/")
    assert "frame-ancestors 'none'" in page.headers["content-security-policy"]
    assert page.headers["x-content-type-options"] == "nosniff"
    assert client.get("/docs").status_code == 404
    assert client.head("/").status_code == 200 and client.get("/health").json() == {
        "ok": True
    }


def test_stop_needs_sign_in_and_stops_only_the_viewers_run(hosted, monkeypatch):
    module, client, _ = hosted
    stopped = []
    monkeypatch.setattr(
        module.runner, "stop", lambda folder: stopped.append(folder) or True
    )
    assert client.post("/api/runs/claude/stop", json={}).status_code == 401
    client.post("/api/login", json={"key": KEY})
    assert client.post("/api/runs/nope/stop", json={}).status_code == 404
    r = client.post("/api/runs/claude/stop", json={})
    assert r.status_code == 200 and r.json() == {"stopped": True}
    assert stopped[-1].name == "claude"
    assert stopped[-1].parent.parent == Path(module.RUNS) / "visitors"


def test_runner_stop_ends_the_whole_process_group_and_frees_the_slot(tmp_path):
    import subprocess

    import runner as runner_module

    r = runner_module.Runner()
    folder = tmp_path / "claude"
    folder.mkdir()
    (folder / "transcript.jsonl").write_text("{}\n")
    run = runner_module.Run("claude", folder)
    # A shell with a child, like run.sh and its agent.
    run.process = subprocess.Popen(
        ["sh", "-c", "sleep 60 & wait"], start_new_session=True
    )
    r._runs[folder] = run
    assert r.running_in(tmp_path) is run
    t = time.time()
    assert r.stop(folder) is True
    assert time.time() - t < 10 and not run.running
    assert r.running_in(tmp_path) is None
    assert "stopped by the presenter" in run.log
    assert not folder.exists()  # the half-done run is gone; the showcase shows again
    assert r.stop(folder) is False  # nothing left to stop


def test_open_questions_plan_then_run(open_hosted, monkeypatch):
    module, client, started = open_hosted
    asked = []

    def draft(base, key, model, context, question, files):
        asked.append((key, model, question, [f.name for f in files]))
        return ASK["plan"]

    monkeypatch.setattr(module.planner, "draft", draft)
    q = {"question": ASK["question"]}
    assert client.post("/api/plan", json=q).status_code == 401  # signed out
    client.post("/api/login", json={"key": KEY})
    assert client.get("/api/setup").json()["open"]
    r = client.post(
        "/api/plan", json={"question": "  Which provinces\n lose the most plans? "}
    )
    assert r.status_code == 200 and r.json() == ASK
    assert asked[-1][:3] == (KEY, "claude-sonnet-5-5", ASK["question"])
    assert asked[-1][3] == [module.task.HISTORY, module.task.CURRENT]
    assert client.post("/api/plan", json={"question": " "}).status_code == 400
    assert client.post("/api/plan", json={"question": "x" * 601}).status_code == 400

    # A run needs a question and a plan with known kinds of steps.
    bad = [
        {"question": "", "plan": ASK["plan"]},
        {"question": "q", "plan": []},
        {"question": "q", "plan": [{"kind": "rm -rf", "text": "x"}]},
        {"question": "q", "plan": "1. do it"},
    ]
    for body in bad:
        r = client.post("/api/runs", json={"agent": "claude", **body})
        assert r.status_code in (400, 422), body  # 422: not even a list
    # Long steps are cut, extra steps dropped, unknown kinds skipped.
    long = {
        "question": "q",
        "plan": [{"kind": "analyze", "text": "y" * 500}] * 9
        + [{"kind": "nope", "text": "z"}],
    }
    assert client.post("/api/runs", json={"agent": "claude", **long}).status_code == 200
    plan = started[-1]["ask"]["plan"]
    assert len(plan) == 7 and all(len(s["text"]) == 140 for s in plan)

    r = client.post("/api/runs", json={"agent": "claude", "prompt": "rm", **ASK})
    run = started[-1]
    # The setting is fixed; the visitor brings only the question and plan.
    assert run["prompt"].startswith(module.CONTEXT.read_text().rstrip())
    assert "rm" not in run["prompt"].split("The question:")[1]
    assert ASK["question"] in run["prompt"]
    assert "1. Count leavers by province" in run["prompt"]
    assert run["ask"] == ASK


def test_a_fixed_brief_has_no_planning(hosted):
    module, client, _ = hosted
    client.post("/api/login", json={"key": KEY})
    assert not client.get("/api/setup").json()["open"]
    assert client.post("/api/plan", json={"question": "q"}).status_code == 404


def test_a_runs_question_and_plan_come_back_with_its_events(tmp_path):
    import runner as runner_module

    folder = tmp_path / "claude"
    folder.mkdir()
    run = runner_module.Run("claude", folder)
    assert run.ask is None
    (tmp_path / ".claude.ask.json").write_text('{"question": "q", "plan": []}')
    assert run.ask == {"question": "q", "plan": []}
    (folder / "ask.json").write_text('{"question": "showcase"}')  # a showcase run
    assert run.ask == {"question": "showcase"}


def test_the_planner_reads_a_fenced_reply_and_keeps_known_kinds():
    import planner

    reply = 'Here:\n```json\n{"steps": [{"kind": "clean", "text": " Fix  it "}, {"kind": "x", "text": "y"}]}\n```'
    assert planner.clean_plan(planner.parse(reply)) == [
        {"kind": "clean", "text": "Fix it"}
    ]
    assert planner.parse("no json") is None
    prompt = planner.prompt("CTX", "Q?", [{"kind": "clean", "text": "Fix it"}])
    assert prompt.startswith("CTX") and "Q?" in prompt and "1. Fix it" in prompt
