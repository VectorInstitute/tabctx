"""The demo's scenarios: each is one question about one table, in
scenarios/<name>/:

- task.py          the files, the id and outcome columns, the page's labels
                   (its steps); optionally the raw files' own names for the
                   id and outcome, and how to read the outcome's rate there
- PROMPT.md        the agent's task
- prepare_data.py  builds the data: main(scenario) writes the workspace and
                   the withheld answers

DEMO_SCENARIO picks one (default: readmission). run.sh, score.py,
prepare_data.py, make_showcase.py and the web page all go through load(),
and only through Scenario, which holds every default, so a new scenario is
a new folder and nothing else. Its data lives in workspace/<name>/ (what
the agent gets) and answers/<name>/ (what it never sees); its runs in
runs/<name>/<agent>/ and its replayable runs in showcase/<name>/<agent>/.

For run.sh, which has no pandas (this file and task.py import nothing
outside the standard library), one call prints everything it needs as
shell assignments:

    eval "$(python3 scenario.py env)"   # NAME HISTORY CURRENT PROMPT ...
"""

from __future__ import annotations

import importlib
import os
import shlex
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
        # DEMO_WORKSPACE replaces the data folder, whatever the scenario
        # (for a copy of the data kept elsewhere).
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
    def steps(self) -> list[str]:
        """The task's steps, as the page's icon names."""
        return [step[0] for step in self.task.LABELS["steps"]]

    @property
    def needs_shell(self) -> bool:
        """A task with a "clean" step has the agent write and run code, so
        run.sh gives each agent its shell."""
        return "clean" in self.steps

    @property
    def raw_id(self) -> str:
        """The id column's name in the files the agent gets (raw exports
        name it their own way)."""
        return getattr(self.task, "RAW_ID", self.task.ID)

    @property
    def raw_target(self) -> str:
        return getattr(self.task, "RAW_TARGET", self.task.TARGET)

    def positive_rate(self, history) -> float:
        """The outcome's share in the history file the agent gets (a pandas
        DataFrame); a raw export's task says how to read it."""
        if hasattr(self.task, "positive_rate"):
            return float(self.task.positive_rate(history))
        return float((history[self.task.TARGET] == self.task.POSITIVE).mean())

    def prepare(self) -> None:
        module = importlib.import_module(f"scenarios.{self.name}.prepare_data")
        module.main(self)

    def shell_env(self) -> str:
        """What run.sh needs, as shell assignments."""
        values = {
            "NAME": self.name,
            "HISTORY": self.task.HISTORY,
            "CURRENT": self.task.CURRENT,
            "PROMPT": self.prompt,
            "WORKSPACE": self.workspace,
            "ANSWERS": self.answers,
            "SHELL_TOOLS": int(self.needs_shell),
        }
        return "\n".join(
            f"SCENARIO_{k}={shlex.quote(str(v))}" for k, v in values.items()
        )


def names() -> list[str]:
    return sorted(p.parent.name for p in SCENARIOS.glob("*/task.py"))


def load(name: str | None = None) -> Scenario:
    name = name or os.environ.get("DEMO_SCENARIO") or DEFAULT
    if name not in names():
        raise SystemExit(f"unknown scenario {name!r}: use one of {', '.join(names())}")
    return Scenario(name, importlib.import_module(f"scenarios.{name}.task"))


def of_run(run_dir: Path) -> Scenario:
    """The scenario a run folder (runs/<scenario>/<agent>) belongs to, else
    the one DEMO_SCENARIO picks."""
    parent = Path(run_dir).resolve().parent.name
    return load(parent if parent in names() else None)


if __name__ == "__main__":
    if sys.argv[1:] != ["env"]:
        raise SystemExit("usage: scenario.py env")
    print(load().shell_env())
