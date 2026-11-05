"""Build a scenario's data: the agent's workspace and the withheld answers
(see scenario.py). The scenario is the argument, else DEMO_SCENARIO, else
readmission.

    uv run --extra mcp python examples/agent_demo/prepare_data.py [insurance]
"""

from __future__ import annotations

import sys

import scenario

if __name__ == "__main__":
    scenario.load(sys.argv[1] if sys.argv[1:] else None).prepare()
