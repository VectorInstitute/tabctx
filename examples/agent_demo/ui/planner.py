"""Open questions: the agent's plan, drafted for the asker to approve, and
the prompt a run gets (the scenario's context, the question, the plan).

The plan comes from one quick model call through the gateway, on the
question and a peek at the files (their headers and a few rows), so the
asker sees it in seconds, before any agent starts. Each step has a kind,
which the page draws with that kind's picture, and a short sentence.
"""

from __future__ import annotations

import csv
import json
import re
import urllib.error
import urllib.request
from pathlib import Path

# What a step can be: the page has a picture for each.
KINDS = {
    "clean": "fix the raw exports",
    "evaluate": "check how well the outcome can be predicted",
    "predict": "score the rows that matter",
    "explain": "find what drives the outcome",
    "what_if": "test what could change it",
    "analyze": "count, sum or compare (plain analysis, no model)",
    "report": "write the answer",
}
MAX_QUESTION = 600
MAX_STEPS = 7
MAX_STEP = 140


class PlanError(Exception):
    """The plan could not be drafted; .status is the HTTP status to answer."""

    def __init__(self, status: int, message: str) -> None:
        super().__init__(message)
        self.status = status


def clean_question(text: str) -> str:
    q = " ".join(str(text or "").split())
    if not q:
        raise PlanError(400, "ask a question first")
    if len(q) > MAX_QUESTION:
        raise PlanError(400, f"keep the question under {MAX_QUESTION} characters")
    return q


def clean_plan(steps: object) -> list[dict[str, str]]:
    """The plan as the asker approved it: known kinds, short plain steps."""
    if not isinstance(steps, list) or not steps:
        raise PlanError(400, "the plan has no steps")
    out = []
    for step in steps[:MAX_STEPS]:
        if not isinstance(step, dict):
            continue
        kind = str(step.get("kind", "")).strip()
        text = " ".join(str(step.get("text", "")).split())[:MAX_STEP]
        if kind in KINDS and text:
            out.append({"kind": kind, "text": text})
    if not out:
        raise PlanError(400, "the plan has no steps")
    return out


def peek(files: list[Path], rows: int = 3) -> str:
    """Each file's header and first rows, as the planner sees them."""
    parts = []
    for path in files:
        with path.open(newline="") as f:
            reader = csv.reader(f)
            lines = [next(reader, []) for _ in range(rows + 1)]
        body = "\n".join(",".join(cell[:40] for cell in line) for line in lines)
        parts.append(f"{path.name}:\n{body}")
    return "\n\n".join(parts)


def _ask(context: str, question: str, sample: str) -> str:
    kinds = "\n".join(f"- {k}: {v}" for k, v in KINDS.items())
    return f"""{context}

Someone asked:
{question}

A peek at the files:
{sample}

Before doing any work, propose a plan for them to approve: 3 to 6 steps,
in the order you would do them, each one short plain sentence a business
reader understands (under 14 words, no jargon, no file names). Only steps
this question needs. Each step has one kind:
{kinds}

Answer with JSON only: {{"steps": [{{"kind": "...", "text": "..."}}]}}"""


def draft(
    base_url: str,
    key: str,
    model: str,
    context: str,
    question: str,
    files: list[Path],
    timeout: float = 60,
) -> list[dict[str, str]]:
    """The plan for question, from model through the gateway's Messages API."""
    body = {
        "model": model,
        "max_tokens": 2000,
        "messages": [{"role": "user", "content": _ask(context, question, peek(files))}],
    }
    req = urllib.request.Request(
        f"{base_url.rstrip('/')}/v1/messages",
        data=json.dumps(body).encode(),
        headers={
            "Authorization": f"Bearer {key}",
            "x-api-key": key,
            "anthropic-version": "2023-06-01",
            "Content-Type": "application/json",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            reply = json.load(resp)
    except urllib.error.HTTPError as e:
        if e.code == 402:
            raise PlanError(402, "this key's budget is used up") from e
        raise PlanError(502, f"the model answered {e.code}; try again") from e
    except (urllib.error.URLError, TimeoutError, ValueError) as e:
        raise PlanError(502, "could not reach the model; try again") from e
    text = "".join(
        b.get("text", "")
        for b in reply.get("content", [])
        if isinstance(b, dict) and b.get("type") == "text"
    )
    try:
        return clean_plan(parse(text))
    except PlanError as e:
        raise PlanError(502, "the model's plan could not be read; try again") from e


def parse(text: str) -> object:
    """The steps in the model's reply (JSON, possibly fenced or wrapped)."""
    match = re.search(r"\{.*\}", text, re.S)
    try:
        return json.loads(match.group(0))["steps"] if match else None
    except (ValueError, KeyError, TypeError):
        return None


def prompt(context: str, question: str, plan: list[dict[str, str]]) -> str:
    """What the agent is asked: the setting, the question, the approved plan."""
    steps = "\n".join(f"{i}. {s['text']}" for i, s in enumerate(plan, 1))
    return f"""{context.rstrip()}

The question:
{question}

The plan they approved:
{steps}

Follow the plan. If the data makes you change a step, say so in the report.
"""
