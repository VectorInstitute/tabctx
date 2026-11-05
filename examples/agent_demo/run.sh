#!/usr/bin/env bash
# Run a demo scenario with a coding agent that has the tabctx tools.
#
#   uv sync --extra mcp                                # once: installs tabctx-mcp
#   export DEMO_SCENARIO=insurance                     # default: readmission
#   uv run --extra mcp python examples/agent_demo/prepare_data.py
#   export TABCTX_API_KEY=vp_...                       # your Vector gateway key
#   examples/agent_demo/run.sh claude|codex|gemini|pi
#   uv run --extra mcp python examples/agent_demo/score.py examples/agent_demo/runs/insurance/claude
#
# Each run works in its own copy of the workspace (runs/<scenario>/<agent>/)
# and leaves transcript.jsonl and report.md there. Codex always has its
# sandboxed shell. Claude Code, Gemini CLI and pi get file tools and the
# tabctx tools only, unless the task has the agent run code (a "clean"
# step): then Claude Code gets Bash in its sandbox (no reads of answers/,
# no writes outside the run folder), and Gemini CLI and pi get their
# shells without a sandbox. Claude Code, Codex and
# Gemini CLI get the tools over MCP. pi has no MCP, so it gets the same
# tools as the tabctx-mcp command plus the tabctx skill (SKILL.md).
#
# The agent's own LLM: Claude Code and Gemini CLI use your login. Codex
# and pi go through the same Vector gateway as the tabular models, with
# the same key (gpt-5 and the on-prem Qwen3_8-27B by default; set CODEX_MODEL
# or PI_MODEL to change). CLAUDE_VIA_GATEWAY=1 sends Claude Code there too
# (CLAUDE_MODEL picks its model).
#
# The hosted demo (ui/, Dockerfile) sets: TABCTX_MCP_BIN (the tabctx-mcp
# command), RUN_DIR (this run's folder),
# ISOLATE_HOME=1 (the agent gets HOME inside RUN_DIR, so no state is shared
# between visitors) and CODEX_SANDBOX.
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
GATEWAY=${TABCTX_BASE_URL%/}

SERVER=${TABCTX_MCP_BIN:-$REPO/.venv/bin/tabctx-mcp}
[[ -x $SERVER ]] || { echo "no $SERVER: run 'uv sync --extra mcp' in $REPO" >&2; exit 1; }
export TABCTX_MCP_BIN=$SERVER
# SCENARIO_NAME, _HISTORY, _CURRENT, _PROMPT, _WORKSPACE, _ANSWERS and
# _SHELL_TOOLS (1: the task has the agent run code).
env_lines=$(python3 "$HERE/scenario.py" env) || exit 1
eval "$env_lines"
WORKSPACE=$SCENARIO_WORKSPACE
FILES=("$SCENARIO_HISTORY" "$SCENARIO_CURRENT")
SHELL_TOOLS=$SCENARIO_SHELL_TOOLS
[[ -f $WORKSPACE/${FILES[0]} ]] || {
  echo "no demo data in $WORKSPACE: run prepare_data.py first" >&2; exit 1; }

RUN=${RUN_DIR:-$HERE/runs/$SCENARIO_NAME/$AGENT}
rm -rf "$RUN"
mkdir -p "$RUN"
for f in "${FILES[@]}"; do cp "$WORKSPACE/$f" "$RUN"/; done
export TABCTX_MCP_HOME=$RUN/.tabctx-mcp
if [[ ${ISOLATE_HOME:-0} == 1 ]]; then
  export HOME=$RUN/.home XDG_CONFIG_HOME=$RUN/.home/.config
  mkdir -p "$XDG_CONFIG_HOME"
fi
# python with pandas, and the tabctx-mcp command, for the agent's shell.
export PATH="$(dirname "$SERVER"):$PATH"
PROMPT=$(cat "${PROMPT_FILE:-$SCENARIO_PROMPT}")  # the UI passes an edited prompt
cd "$RUN"

case $AGENT in
claude)
  if [[ ${CLAUDE_VIA_GATEWAY:-0} == 1 ]]; then
    export ANTHROPIC_BASE_URL=$GATEWAY ANTHROPIC_AUTH_TOKEN=$VP_API_KEY
  fi
  # The server inherits this environment (the key never goes on a command line).
  config=$(printf '{"mcpServers": {"tabctx": {"command": "%s"}}}' "$SERVER")
  model=()
  [[ -n ${CLAUDE_MODEL:-} ]] && model=(--model "$CLAUDE_MODEL")
  allowed=Read,Write,Edit,Glob,Grep tools=$allowed settings=()
  if [[ $SHELL_TOOLS == 1 ]]; then
    # Bash only in Claude Code's sandbox, never outside it: writes stay in
    # this run's folder, and the answers can't be read. Sandboxed commands
    # are allowed by the sandbox setting, so Bash is not in --allowedTools
    # (which would also allow an unsandboxed retry).
    tools+=,Bash
    settings=(--settings "$(printf '{"sandbox": {"enabled": true, "autoAllowBashIfSandboxed": true, "allowUnsandboxedCommands": false}, "permissions": {"deny": ["Read(/%s/**)"]}}' "$SCENARIO_ANSWERS")")
  fi
  # ${a[@]+...}: an empty array is "unbound" to macOS's bash 3.2 under set -u.
  claude -p "$PROMPT" ${model[@]+"${model[@]}"} ${settings[@]+"${settings[@]}"} --mcp-config "$config" --strict-mcp-config \
    --tools "$tools" --allowedTools mcp__tabctx ${allowed//,/ } \
    --output-format stream-json --verbose >transcript.jsonl
  ;;
codex)
  # gpt-5, not a newer model: for gpt-5.4 and up Codex adds its
  # tool_search tool, which the gateway refuses (it can't bill it).
  codex exec --skip-git-repo-check --sandbox "${CODEX_SANDBOX:-workspace-write}" --json \
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
  exclude='["run_shell_command"]'
  [[ $SHELL_TOOLS == 1 ]] && exclude='[]'
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
  "excludeTools": $exclude
}
EOF
  # File edits and the (trusted) tabctx tools run unprompted; a shell, when
  # the task needs one, too.
  approval=auto_edit
  [[ $SHELL_TOOLS == 1 ]] && approval=yolo
  gemini -p "$PROMPT" --approval-mode "$approval" -o stream-json >transcript.jsonl
  ;;
pi)
  export PI_CODING_AGENT_DIR=$RUN/.pi-agent
  mkdir -p "$PI_CODING_AGENT_DIR" .pi/extensions
  # pi has no MCP: the extension registers the same tools as pi tools.
  tabctx-mcp pi-extension >.pi/extensions/tabctx.ts
  tools=read,write,edit,ls,find,grep
  [[ $SHELL_TOOLS == 1 ]] && tools+=,bash
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
      "models": [{"id": "${PI_MODEL:-Qwen3_8-27B}", "contextWindow": 262144, "maxTokens": 16000}]
    }
  }
}
EOF
  PI=${PI_BIN:-pi}
  # File tools and the tabctx tools (and bash, when the task needs a shell).
  "$PI" -p --provider vector --model "${PI_MODEL:-Qwen3_8-27B}" \
    --tools "$tools" --mode json --no-session "$PROMPT" </dev/null >transcript.jsonl
  ;;
esac

echo "done: $RUN"
ls "$RUN"
