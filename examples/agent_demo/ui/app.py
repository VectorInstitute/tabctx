"""The demo's web page: pick an agent, edit the task, run it, and watch its
tabctx tool calls arrive live, then its report and its score against
what the customers actually did.

    export TABCTX_API_KEY=vp_...
    uv run --extra mcp --with fastapi --with uvicorn \\
        python examples/agent_demo/ui/app.py        # http://127.0.0.1:8765

Runs go through ../run.sh, as from the shell; past runs in ../runs/ can
be replayed without running anything. Serves on 127.0.0.1 only: it
starts agents on this machine.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
from functools import cache
from pathlib import Path

import pandas as pd
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

sys.path.insert(0, str(Path(__file__).resolve().parent))
from runner import (  # noqa: E402
    AGENTS,
    DEMO,
    Run,
    Runner,
    available,
    outcome,
    past_runs,
)
from transcripts import PARSERS  # noqa: E402

STATIC = Path(__file__).resolve().parent / "static"
POLL_SECONDS = 0.25

app = FastAPI(title="tabctx agent demo")
runner = Runner()


class StartRun(BaseModel):
    agent: str
    prompt: str


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC / "index.html")


@app.get("/api/setup")
def setup() -> dict:
    return {
        "agents": [
            {"id": key, **info, "available": available(key)}
            for key, info in AGENTS.items()
        ],
        "prompt": (DEMO / "PROMPT.md").read_text(),
        "dataset": dataset(),
        "gateway": os.environ.get(
            "TABCTX_BASE_URL", "https://proxy.vectorinstitute.ai"
        ),
        "key_set": bool(os.environ.get("TABCTX_API_KEY")),
        "runs": past_runs(),
        "running": runner.current.agent
        if runner.current and runner.current.running
        else None,
    }


@cache
def dataset() -> dict | None:
    workspace = DEMO / "workspace"
    try:
        history = pd.read_csv(workspace / "customers_history.csv")
        current = pd.read_csv(workspace / "customers_current.csv")
    except FileNotFoundError:
        return None
    return {
        "history_rows": len(history),
        "current_rows": len(current),
        "features": len(current.columns) - 1,  # without customer_id
        "churn_rate": round(float((history["Churn"] == "Yes").mean()), 3),
        "columns": [c for c in current.columns if c != "customer_id"],
    }


@app.post("/api/runs")
def start(body: StartRun) -> dict:
    try:
        run = runner.start(body.agent, body.prompt)
    except ValueError as e:
        raise HTTPException(400, str(e)) from e
    except RuntimeError as e:
        raise HTTPException(409, str(e)) from e
    return {"agent": run.agent, "started": run.started}


@app.get("/api/runs/{agent}/events")
async def events(agent: str) -> StreamingResponse:
    if agent not in AGENTS:
        raise HTTPException(404, f"unknown agent {agent!r}")
    run = runner.get(agent)
    if not run.running and not run.transcript.is_file():
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
    uvicorn.run(app, host="127.0.0.1", port=int(os.environ.get("PORT", "8765")))
