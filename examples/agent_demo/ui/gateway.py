"""What a visitor's gateway key may use, and which model each agent gets.

The hosted demo runs every agent through the Vector gateway with the
visitor's own key: its LLM calls are billed to that key's project, and the
tabular calls are free. A project sees only the models an admin granted
it, so each agent gets the best model the key can use, or is unavailable.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request

TABULAR = ("tabpfn-3.5", "tabicl-v2")

# Preference order per agent. Codex stays on gpt-5: for newer OpenAI models
# Codex adds a tool_search tool, which the gateway refuses (see run.sh).
_AGENT_MODELS = {
    "claude": ["claude-opus-5-5", "claude-opus-5", "claude-sonnet-5-5", "claude-sonnet-5"],
    "codex": ["gpt-5"],
    "pi": ["claude-sonnet-5-5", "claude-sonnet-5", "claude-opus-5"],
}
# Claude Code's model for small background tasks (titles, summaries).
_CLAUDE_SMALL = ["claude-haiku-5-5", "claude-haiku-4-5-20251001"]


class RefusedKey(Exception):
    """The key was refused; .status is the HTTP status to answer with."""

    def __init__(self, status: int, message: str) -> None:
        super().__init__(message)
        self.status = status


def project_models(base_url: str, key: str, timeout: float = 15) -> set[str]:
    """The model ids the key's project may use (GET /admin/me/models)."""
    if not key.startswith("vp_") or len(key) > 512 or any(c.isspace() for c in key):
        raise RefusedKey(400, "that doesn't look like a gateway key (vp_...)")
    req = urllib.request.Request(
        f"{base_url.rstrip('/')}/admin/me/models",
        headers={"Authorization": f"Bearer {key}", "Accept": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = json.load(resp)
    except urllib.error.HTTPError as e:
        if e.code in (401, 403):
            raise RefusedKey(401, "the gateway did not accept this key") from e
        raise RefusedKey(502, f"the gateway answered {e.code}; try again") from e
    except (urllib.error.URLError, TimeoutError, ValueError) as e:
        raise RefusedKey(502, "could not reach the gateway; try again") from e
    items = body if isinstance(body, list) else body.get("data") or body.get("models") or []
    return {
        str(m.get("id") or m.get("model_id") or m.get("name"))
        for m in items
        if isinstance(m, dict)
    }


def agent_env(models: set[str]) -> dict[str, dict[str, str] | None]:
    """Per agent, the environment run.sh needs to use the key's models
    through the gateway, or None if the key can't run that agent."""
    out: dict[str, dict[str, str] | None] = {}
    for agent, prefs in _AGENT_MODELS.items():
        model = next((m for m in prefs if m in models), None)
        if model is None:
            out[agent] = None
        elif agent == "claude":
            small = next((m for m in _CLAUDE_SMALL if m in models), model)
            out[agent] = {
                "CLAUDE_VIA_GATEWAY": "1",
                "CLAUDE_MODEL": model,
                "ANTHROPIC_DEFAULT_HAIKU_MODEL": small,
                "ANTHROPIC_SMALL_FAST_MODEL": small,
            }
        elif agent == "codex":
            out[agent] = {"CODEX_MODEL": model}
        else:
            out[agent] = {"PI_MODEL": model}
    return out


def model_of(env: dict[str, str] | None) -> str | None:
    if not env:
        return None
    return env.get("CLAUDE_MODEL") or env.get("CODEX_MODEL") or env.get("PI_MODEL")
