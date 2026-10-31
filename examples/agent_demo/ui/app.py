"""The demo's web page: pick an agent, run it (or replay a finished run),
and watch its tabctx tool calls arrive live, then its report and its
score against what the customers actually did.

Locally (agents use your own logins and $TABCTX_API_KEY):

    export TABCTX_API_KEY=vp_...
    uv run --extra mcp --with fastapi --with uvicorn \\
        python examples/agent_demo/ui/app.py        # http://127.0.0.1:8765

Hosted (DEMO_HOSTED=1, see ../Dockerfile): anyone can replay the showcase
runs; to start a run a visitor signs in with their own gateway key, and
every agent's LLM calls go through the gateway with that key. The task is
fixed, each visitor's runs live in their own folder with their own HOME,
and the key stays in memory (sessions.py).
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import time
from collections import defaultdict, deque
from functools import cache
from pathlib import Path

import pandas as pd
import uvicorn
from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

sys.path.insert(0, str(Path(__file__).resolve().parent))
from gateway import (  # noqa: E402
    TABULAR,
    RefusedKey,
    agent_env,
    model_of,
    project_models,
)
from runner import (  # noqa: E402
    AGENTS,
    DEMO,
    RUNS,
    SHOWCASE,
    BusyError,
    Run,
    Runner,
    finished,
    installed,
    outcome,
)
from sessions import COOKIE, Session, SessionStore  # noqa: E402
from transcripts import PARSERS  # noqa: E402

HOSTED = os.environ.get("DEMO_HOSTED") == "1"
GATEWAY = os.environ.get("TABCTX_BASE_URL", "https://proxy.vectorinstitute.ai")
WORKSPACE = Path(os.environ.get("DEMO_WORKSPACE", DEMO / "workspace"))
STATIC = Path(__file__).resolve().parent / "static"
POLL_SECONDS = 0.25
HOSTED_AGENTS = ("claude", "codex", "pi")  # Gemini CLI can't use the gateway
LOGIN_ATTEMPTS = (10, 600)  # per client address: at most 10 per 10 minutes

app = FastAPI(
    title="tabctx agent demo", docs_url=None, redoc_url=None, openapi_url=None
)


def _uid_range(spec: str | None) -> range | None:
    """DEMO_RUN_UIDS="20000-20063": run each agent as its own user."""
    if not spec:
        return None
    low, high = (int(x) for x in spec.split("-"))
    return range(low, high + 1)


runner = Runner(
    max_running=int(os.environ.get("DEMO_MAX_RUNS", "3")),
    timeout_seconds=float(os.environ.get("DEMO_RUN_TIMEOUT", "900")),
    uids=_uid_range(os.environ.get("DEMO_RUN_UIDS")),
)
sessions = SessionStore(RUNS / "visitors", on_end=runner.stop_in)
_attempts: dict[str, deque] = defaultdict(deque)

CSP = "; ".join(
    [
        "default-src 'self'",
        "script-src 'self' https://cdn.jsdelivr.net",
        "style-src 'self' 'unsafe-inline' https://fonts.googleapis.com",
        "font-src https://fonts.gstatic.com",
        "img-src 'self' data:",
        "connect-src 'self'",
        "frame-ancestors 'none'",
        "base-uri 'none'",
        "form-action 'none'",
    ]
)


@app.middleware("http")
async def headers(request: Request, call_next):
    if request.method == "POST" and not request.headers.get(
        "content-type", ""
    ).startswith("application/json"):
        # A cross-site form can't send JSON without a CORS preflight.
        return JSONResponse({"detail": "send JSON"}, status_code=415)
    response = await call_next(request)
    response.headers["Content-Security-Policy"] = CSP
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["Referrer-Policy"] = "no-referrer"
    response.headers["Cache-Control"] = "no-store"
    return response


class StartRun(BaseModel):
    agent: str
    prompt: str | None = None


class Login(BaseModel):
    key: str


# ---- who is looking ----------------------------------------------------------


def viewer(request: Request) -> Session | None:
    return sessions.get(request.cookies.get(COOKIE)) if HOSTED else None


def my_runs(session: Session | None) -> Path | None:
    """Where this viewer's runs go: their own folder when hosted."""
    if not HOSTED:
        return RUNS
    return sessions.runs_dir(session) if session else None


def agents_for(session: Session | None) -> list[dict]:
    out = []
    for key, info in AGENTS.items():
        if HOSTED and key not in HOSTED_AGENTS:
            continue
        entry = {"id": key, **info, "available": installed(key)}
        if HOSTED:
            model = model_of(session.agents.get(key)) if session else None
            entry["llm"] = f"{model} via your key" if model else "via your gateway key"
            entry["allowed"] = bool(session and session.agents.get(key))
        else:
            entry["allowed"] = entry["available"]
        out.append(entry)
    return out


# ---- pages and setup ------------------------------------------------------------


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC / "index.html")


@app.get("/healthz")
def healthz() -> dict:
    return {"ok": True}


@app.get("/api/setup")
def setup(request: Request) -> dict:
    session = viewer(request)
    mine = my_runs(session)
    runs = {
        **(finished(SHOWCASE) if HOSTED else {}),
        **(finished(mine) if mine else {}),
    }
    live = runner.running_in(mine) if mine else None
    return {
        "hosted": HOSTED,
        "signed_in": {"key_id": session.key_id} if session else None,
        "agents": agents_for(session),
        "prompt": (DEMO / "PROMPT.md").read_text(),
        "prompt_editable": not HOSTED,
        "dataset": dataset(),
        "gateway": GATEWAY,
        "key_set": bool(session) if HOSTED else bool(os.environ.get("TABCTX_API_KEY")),
        "runs": [{"agent": a, "score": s} for a, s in runs.items()],
        "running": live.agent if live else None,
    }


@cache
def dataset() -> dict | None:
    try:
        history = pd.read_csv(WORKSPACE / "customers_history.csv")
        current = pd.read_csv(WORKSPACE / "customers_current.csv")
    except FileNotFoundError:
        return None
    return {
        "history_rows": len(history),
        "current_rows": len(current),
        "features": len(current.columns) - 1,  # without customer_id
        "churn_rate": round(float((history["Churn"] == "Yes").mean()), 3),
        "columns": [c for c in current.columns if c != "customer_id"],
    }


# ---- signing in (hosted) -----------------------------------------------------------


@app.post("/api/login")
def login(body: Login, request: Request, response: Response) -> dict:
    if not HOSTED:
        raise HTTPException(404, "sign-in is for the hosted demo")
    _limit(request)
    key = body.key.strip()
    try:
        models = project_models(GATEWAY, key)
    except RefusedKey as e:
        raise HTTPException(e.status, str(e)) from e
    if not any(m in models for m in TABULAR):
        raise HTTPException(
            403,
            "this key's project has no access to the tabular models; ask an admin to add it",
        )
    agents = agent_env(models)
    if not any(agents.values()):
        raise HTTPException(
            403, "this key's project has no model the demo's agents can use"
        )
    session = sessions.create(key, agents)
    response.set_cookie(
        COOKIE,
        session.id,
        httponly=True,
        secure=request.url.scheme == "https"
        or request.headers.get("x-forwarded-proto") == "https",
        samesite="strict",
        max_age=int(sessions.idle_seconds),
        path="/",
    )
    return {"key_id": session.key_id, "agents": agents_for(session)}


@app.post("/api/logout")
def logout(request: Request, response: Response) -> dict:
    sessions.end(request.cookies.get(COOKIE))
    response.delete_cookie(COOKIE, path="/")
    return {"ok": True}


def _limit(request: Request) -> None:
    count, window = LOGIN_ATTEMPTS
    # Cloud Run's front end appends the client's address last; earlier
    # entries come from the client and can be anything.
    who = request.headers.get(
        "x-forwarded-for", request.client.host if request.client else ""
    )
    who = who.split(",")[-1].strip()
    now = time.time()
    attempts = _attempts[who]
    while attempts and attempts[0] < now - window:
        attempts.popleft()
    if len(attempts) >= count:
        raise HTTPException(429, "too many sign-in attempts; wait a few minutes")
    attempts.append(now)


# ---- runs -----------------------------------------------------------------------------


@app.post("/api/runs")
def start(body: StartRun, request: Request) -> dict:
    if body.agent not in AGENTS:
        raise HTTPException(400, f"unknown agent {body.agent!r}")
    session = viewer(request)
    if HOSTED:
        if not session:
            raise HTTPException(401, "sign in with your gateway key to run an agent")
        env = session.agents.get(body.agent) if body.agent in HOSTED_AGENTS else None
        if not env:
            raise HTTPException(403, "your key's project can't use this agent's model")
        env = {
            **env,
            "TABCTX_API_KEY": session.key,
            "VP_API_KEY": session.key,
            "ISOLATE_HOME": "1",
        }
        prompt = (DEMO / "PROMPT.md").read_text()
    else:
        if not os.environ.get("TABCTX_API_KEY"):
            raise HTTPException(400, "set TABCTX_API_KEY before starting the UI")
        env, prompt = {}, body.prompt or (DEMO / "PROMPT.md").read_text()
    if not installed(body.agent):
        raise HTTPException(400, f"{AGENTS[body.agent]['label']} is not installed")
    folder = my_runs(session)
    assert folder is not None  # signed in when hosted, RUNS locally
    try:
        run = runner.start(body.agent, folder / body.agent, prompt, env)
    except BusyError as e:
        raise HTTPException(409, str(e)) from e
    return {"agent": run.agent, "started": run.started}


@app.get("/api/runs/{agent}/events")
async def events(
    agent: str, request: Request, source: str = "auto"
) -> StreamingResponse:
    """A run's events: the viewer's own run of agent, or (source=showcase,
    or when they have none) the showcase run."""
    if agent not in AGENTS:
        raise HTTPException(404, f"unknown agent {agent!r}")
    mine = my_runs(viewer(request))
    run = runner.get(agent, mine / agent) if mine else None
    if source == "showcase" or not (run and run.exists):
        run = Run(agent, SHOWCASE / agent) if HOSTED else None
    if not run or not run.exists:
        raise HTTPException(404, f"no run of {agent} yet")
    return StreamingResponse(_stream(run), media_type="text/event-stream")


async def _stream(run: Run):
    """Server-sent events: the transcript's events as they are written
    (the whole file at once for a finished run), then the outcome."""
    yield _sse(
        "status", {"agent": run.agent, "live": run.running, "started": run.started}
    )
    # run.sh deletes the previous run first: wait for this run's transcript.
    while run.running and not _fresh(run):
        await asyncio.sleep(POLL_SECONDS)
    parser = PARSERS[run.agent]()
    if run.transcript.is_file():
        with run.transcript.open() as f:
            pending = ""
            while True:
                chunk = f.readline()
                if chunk:
                    pending += chunk
                    if not pending.endswith("\n"):
                        continue  # the agent is mid-line
                    for event in parser.feed(pending):
                        yield _sse("event", event)
                    pending = ""
                elif run.running:
                    await asyncio.sleep(POLL_SECONDS)
                else:
                    for event in parser.feed(pending):
                        yield _sse("event", event)
                    break
    yield _sse("outcome", outcome(run))


def _fresh(run: Run) -> bool:
    return (
        run.transcript.is_file() and run.transcript.stat().st_mtime >= run.started - 1
    )


def _sse(kind: str, data: dict) -> str:
    return f"event: {kind}\ndata: {json.dumps(data, default=str)}\n\n"


app.mount("/static", StaticFiles(directory=STATIC), name="static")

if __name__ == "__main__":
    # Locally only on 127.0.0.1: the page starts agents on this machine.
    host = "0.0.0.0" if HOSTED else "127.0.0.1"
    uvicorn.run(
        app, host=host, port=int(os.environ.get("PORT", "8765")), proxy_headers=True
    )
