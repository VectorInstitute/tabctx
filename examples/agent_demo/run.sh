#!/usr/bin/env bash
# Run the churn demo with a coding agent that has the tabctx tools.
#
#   uv sync --extra mcp                                # once: installs tabctx-mcp
#   uv run --extra mcp python examples/agent_demo/prepare_data.py
#   export TABCTX_API_KEY=vp_...                       # your Vector gateway key
#   examples/agent_demo/run.sh claude|codex|gemini|pi
#   uv run --extra mcp python examples/agent_demo/score.py examples/agent_demo/runs/claude
#
# Each run works in its own copy of the workspace (runs/<agent>/) and
# leaves transcript.jsonl and report.md there. Claude Code, Codex and
# Gemini CLI get the tools over MCP. pi has no MCP, so it gets the same
# tools as the tabctx-mcp command plus the tabctx skill (SKILL.md).
#
# The agent's own LLM: Claude Code and Gemini CLI use your login. Codex
# and pi go through the same Vector gateway as the tabular models, with
# the same key (gpt-5 and claude-sonnet-5-5 by default; set CODEX_MODEL
# or PI_MODEL to change). CLAUDE_VIA_GATEWAY=1 sends Claude Code there too.
set -euo pipefail

HERE=$(cd "$(dirname "$0")" && pwd)
REPO=$(cd "$HERE/../.." && pwd)
AGENT=${1:?usage: run.sh claude|codex|gemini|pi}
case $AGENT in
claude | codex | gemini | pi) ;;
*) echo "unknown agent '$AGENT': use claude, codex, gemini or pi" >&2; exit 2 ;;
esac
: "${TABCTX_API_KEY:?set TABCTX_API_KEY to your Vector gateway key (vp_...)}"
export TABCTX_BASE_URL=${TABCTX_BASE_URL:-https://proxy.vectorinstitute.ai}
export VP_API_KEY=${VP_API_KEY:-$TABCTX_API_KEY}
GATEWAY=https://proxy.vectorinstitute.ai

SERVER=$REPO/.venv/bin/tabctx-mcp
[[ -x $SERVER ]] || { echo "no $SERVER: run 'uv sync --extra mcp' in $REPO" >&2; exit 1; }
[[ -f $HERE/workspace/customers_history.csv ]] || {
  echo "no demo data: run prepare_data.py first" >&2; exit 1; }

RUN=$HERE/runs/$AGENT
rm -rf "$RUN"
mkdir -p "$RUN"
cp "$HERE"/workspace/customers_history.csv "$HERE"/workspace/customers_current.csv "$RUN"/
export TABCTX_MCP_HOME=$RUN/.tabctx-mcp
# python with pandas, and the tabctx-mcp command, for the agent's shell.
export PATH="$REPO/.venv/bin:$PATH"
PROMPT=$(cat "${PROMPT_FILE:-$HERE/PROMPT.md}")  # the UI passes an edited prompt
cd "$RUN"

case $AGENT in
claude)
  if [[ ${CLAUDE_VIA_GATEWAY:-0} == 1 ]]; then
    export ANTHROPIC_BASE_URL=$GATEWAY ANTHROPIC_AUTH_TOKEN=$VP_API_KEY
  fi
  # The server inherits this environment (the key never goes on a command line).
  config=$(printf '{"mcpServers": {"tabctx": {"command": "%s"}}}' "$SERVER")
  claude -p "$PROMPT" --mcp-config "$config" --strict-mcp-config \
    --tools Read,Write,Edit,Glob,Grep --allowedTools mcp__tabctx Read Write Edit Glob Grep \
    --output-format stream-json --verbose >transcript.jsonl
  ;;
codex)
  # gpt-5, not a newer model: for gpt-5.4 and up Codex adds its
  # tool_search tool, which the gateway refuses (it can't bill it).
  codex exec --skip-git-repo-check --sandbox workspace-write --json \
    -c model_provider=vector -m "${CODEX_MODEL:-gpt-5}" \
    -c 'model_providers.vector.name="Vector AI Gateway"' \
    -c "model_providers.vector.base_url=\"$GATEWAY/v1\"" \
    -c 'model_providers.vector.env_key="VP_API_KEY"' \
    -c 'model_providers.vector.wire_api="responses"' \
    -c "mcp_servers.tabctx.command=\"$SERVER\"" \
    -c 'mcp_servers.tabctx.env_vars=["TABCTX_API_KEY", "TABCTX_BASE_URL", "TABCTX_MCP_HOME"]' \
    -c 'mcp_servers.tabctx.tool_timeout_sec=300' \
    -c 'mcp_servers.tabctx.default_tools_approval_mode="approve"' \
    "$PROMPT" </dev/null >transcript.jsonl
  ;;
gemini)
  mkdir -p .gemini
  # Gemini CLI expands $VARS in settings.json: no key is written to disk.
  cat >.gemini/settings.json <<EOF
{
  "mcpServers": {
    "tabctx": {
      "command": "$SERVER",
      "env": {
        "TABCTX_API_KEY": "\$TABCTX_API_KEY",
        "TABCTX_BASE_URL": "\$TABCTX_BASE_URL",
        "TABCTX_MCP_HOME": "\$TABCTX_MCP_HOME"
      },
      "timeout": 300000,
      "trust": true
    }
  },
  "excludeTools": ["run_shell_command"]
}
EOF
  # No shell; file edits and the (trusted) tabctx tools run unprompted.
  gemini -p "$PROMPT" --approval-mode auto_edit -o stream-json >transcript.jsonl
  ;;
pi)
  export PI_CODING_AGENT_DIR=$RUN/.pi-agent
  mkdir -p "$PI_CODING_AGENT_DIR" .pi/extensions
  # pi has no MCP: the extension registers the same tools as pi tools.
  tabctx-mcp pi-extension >.pi/extensions/tabctx.ts
  tools=read,write,edit,ls,find,grep
  for t in $(tabctx-mcp tools | python -c 'import json,sys; print(" ".join(t["name"] for t in json.load(sys.stdin)["tools"]))'); do
    tools+=,tabctx_$t
  done
  cat >"$PI_CODING_AGENT_DIR/models.json" <<EOF
{
  "providers": {
    "vector": {
      "baseUrl": "$GATEWAY/v1",
      "api": "openai-completions",
      "apiKey": "VP_API_KEY",
      "models": [{"id": "${PI_MODEL:-claude-sonnet-5-5}", "contextWindow": 200000, "maxTokens": 16000}]
    }
  }
}
EOF
  PI=${PI_BIN:-pi}
  # No bash: file tools and the tabctx tools only.
  "$PI" -p --provider vector --model "${PI_MODEL:-claude-sonnet-5-5}" \
    --tools "$tools" --mode json --no-session "$PROMPT" </dev/null >transcript.jsonl
  ;;
esac

echo "done: $RUN"
ls "$RUN"
