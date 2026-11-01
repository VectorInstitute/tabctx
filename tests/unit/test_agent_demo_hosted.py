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


@pytest.fixture
def hosted(tmp_path, monkeypatch):
    monkeypatch.setenv("DEMO_HOSTED", "1")
    monkeypatch.setenv("DEMO_RUNS", str(tmp_path / "runs"))
    monkeypatch.delenv("TABCTX_API_KEY", raising=False)
    for name in ("app", "runner"):
        sys.modules.pop(name, None)
    module = importlib.import_module("app")
    monkeypatch.setattr(
        module, "project_models", lambda base, key: ALL if key == KEY else set()
    )
    monkeypatch.setattr(module, "installed", lambda agent: True)
    started = []

    def fake_start(agent, folder, prompt, env):
        started.append({"agent": agent, "folder": folder, "prompt": prompt, "env": env})
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
    assert run["prompt"] == (module.DEMO / "PROMPT.md").read_text()  # the task is fixed
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
