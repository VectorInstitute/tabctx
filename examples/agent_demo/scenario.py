"""The demo's scenarios: each is one question about one table, in
scenarios/<name>/:

- task.py          the files, the id and outcome columns, the page's labels
- PROMPT.md        the agent's task
- prepare_data.py  builds the data: main(scenario) writes the workspace and
                   the withheld answers

DEMO_SCENARIO picks one (default: readmission). run.sh, score.py,
prepare_data.py, make_showcase.py and the web page all go through load(),
so a new scenario is a new folder and nothing else. Its data lives in
workspace/<name>/ (what the agent gets) and answers/<name>/ (what it never
sees); its runs in runs/<name>/<agent>/ and its replayable runs in
showcase/<name>/<agent>/.

For run.sh, which has no pandas (this file and task.py import nothing
outside the standard library):

    python3 scenario.py files       # the history and current file names
    python3 scenario.py prompt      # PROMPT.md's path
    python3 scenario.py workspace   # the data's folder
    python3 scenario.py shell       # 1 if the task needs a shell, else 0
"""

from __future__ import annotations

import importlib
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType

HERE = Path(__file__).resolve().parent
SCENARIOS = HERE / "scenarios"
DEFAULT = "readmission"

if str(HERE) not in sys.path:  # scenarios/ is a package under this folder
    sys.path.insert(0, str(HERE))


@dataclass(frozen=True)
class Scenario:
    name: str
    task: ModuleType
    root: Path = HERE  # where the data and runs live (tests use a tmp dir)

    @property
    def folder(self) -> Path:
        return SCENARIOS / self.name

    @property
    def prompt(self) -> Path:
        return self.folder / "PROMPT.md"

    @property
    def workspace(self) -> Path:
        # The hosted image may keep the data elsewhere.
        override = os.environ.get("DEMO_WORKSPACE")
        return Path(override) if override else self.root / "workspace" / self.name

    @property
    def answers(self) -> Path:
        return self.root / "answers" / self.name

    @property
    def runs(self) -> Path:
        return self.root / "runs" / self.name

    @property
    def showcase(self) -> Path:
        return self.root / "showcase" / self.name

    @property
    def needs_shell(self) -> bool:
        """Whether the task has the agent run code (e.g. to clean a table),
        so run.sh gives each agent its shell."""
        return bool(getattr(self.task, "SHELL", False))

    def prepare(self) -> None:
        module = importlib.import_module(f"scenarios.{self.name}.prepare_data")
        module.main(self)


def names() -> list[str]:
    return sorted(p.parent.name for p in SCENARIOS.glob("*/task.py"))


def load(name: str | None = None) -> Scenario:
    name = name or os.environ.get("DEMO_SCENARIO") or DEFAULT
    if name not in names():
        raise SystemExit(f"unknown scenario {name!r}: use one of {', '.join(names())}")
    return Scenario(name, importlib.import_module(f"scenarios.{name}.task"))


def _cli(args: list[str]) -> str:
    scenario = load()
    answers = {
        "files": lambda: f"{scenario.task.HISTORY} {scenario.task.CURRENT}",
        "prompt": lambda: str(scenario.prompt),
        "workspace": lambda: str(scenario.workspace),
        "shell": lambda: "1" if scenario.needs_shell else "0",
    }
    if len(args) != 1 or args[0] not in answers:
        raise SystemExit(f"usage: scenario.py {'|'.join(answers)}")
    return answers[args[0]]()


if __name__ == "__main__":
    print(_cli(sys.argv[1:]))
