# The hosted tabctx agent demo: the web page, the three agents (Claude Code,
# Codex, pi) and tabctx-mcp in one image. Visitors sign in with their own
# Vector gateway key; nothing secret is baked in.
#
#   docker build -f examples/agent_demo/Dockerfile -t tabctx-agent-demo .
#   docker run -p 8080:8080 tabctx-agent-demo        # http://localhost:8080
#
# Build from the repository root (it installs this checkout of tabctx).

FROM node:22-bookworm-slim AS node

FROM python:3.12-slim-bookworm

# Node for the agent CLIs (copied, not apt-installed: one exact version).
COPY --from=node /usr/local/bin/node /usr/local/bin/node
COPY --from=node /usr/local/lib/node_modules /usr/local/lib/node_modules
RUN ln -s ../lib/node_modules/npm/bin/npm-cli.js /usr/local/bin/npm \
 && ln -s ../lib/node_modules/npm/bin/npx-cli.js /usr/local/bin/npx

# The agents, at the versions the demo was tested with.
RUN npm install -g --no-fund --no-audit \
      @anthropic-ai/claude-code@2.1.316 \
      @openai/codex@0.137.0 \
      @mariozechner/pi-coding-agent@0.73.1 \
 && npm cache clean --force

# ripgrep for the agents' search tools.
RUN apt-get update \
 && apt-get install -y --no-install-recommends ripgrep ca-certificates \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY pyproject.toml README.md LICENSE.md ./
COPY src ./src
RUN pip install --no-cache-dir ".[mcp]" "fastapi>=0.115" "uvicorn>=0.30"

COPY examples/agent_demo ./examples/agent_demo
# The demo data (public, from OpenML), prepared at build time. The answers
# stay in the image for scoring; agents only ever get the workspace copy.
RUN python examples/agent_demo/prepare_data.py \
 && rm -rf examples/agent_demo/runs \
 && chmod -R go-rwx examples/agent_demo/answers

# The server runs as root only to start each agent run as a user of its
# own (DEMO_RUN_UIDS; see ui/runner.py): runs can't read each other's
# files, keys (process environments) or the answers above. Codex's own
# sandbox (bubblewrap) can't work inside a container, so Codex runs
# unsandboxed in that per-run user instead.
RUN mkdir -p /tmp/runs/visitors && chmod 711 /tmp/runs /tmp/runs/visitors

ENV DEMO_HOSTED=1 \
    DEMO_RUNS=/tmp/runs \
    DEMO_RUN_UIDS=20000-20063 \
    CODEX_SANDBOX=danger-full-access \
    DEMO_WORKSPACE=/app/examples/agent_demo/workspace \
    TABCTX_BASE_URL=https://proxy.vectorinstitute.ai \
    TABCTX_MCP_BIN=/usr/local/bin/tabctx-mcp \
    DISABLE_AUTOUPDATER=1 \
    CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC=1 \
    PI_OFFLINE=1 \
    PORT=8080

EXPOSE 8080
CMD ["python", "examples/agent_demo/ui/app.py"]
