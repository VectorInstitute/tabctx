"""Fitted models on disk, by name.

Each model is two files in the store's directory: <name>.pkl, the fitted
scikit-learn estimator (it keeps its training table, so it can fit the
server again if the server dropped the context), and <name>.json, what an
agent needs to know about it (listed without unpickling anything).

The pickles are trusted: the store only loads files it wrote. The API key
is never in them (TabctxClient leaves it out when pickled).
"""

from __future__ import annotations

import json
import os
import pickle
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

HOME_ENV = "TABCTX_MCP_HOME"
DEFAULT_HOME = ".tabctx-mcp"

_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,99}$")


@dataclass
class ModelInfo:
    name: str
    model: str
    task: str
    target: str
    features: list[str]
    train_file: str
    train_rows: int
    numeric_features: list[str] = field(default_factory=list)
    classes: list[str] | None = None
    dropped: list[str] = field(default_factory=list)
    created: str = ""

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


class ModelStore:
    def __init__(self, root: str | Path | None = None) -> None:
        """root: the directory to keep models in; default $TABCTX_MCP_HOME,
        else .tabctx-mcp in the working directory."""
        self.root = Path(root or os.environ.get(HOME_ENV) or DEFAULT_HOME)

    def save(self, info: ModelInfo, estimator: Any) -> None:
        check_name(info.name)
        self.root.mkdir(parents=True, exist_ok=True)
        # The pickle first: a listed model always loads.
        _write(self.root / f"{info.name}.pkl", pickle.dumps(estimator))
        _write(
            self.root / f"{info.name}.json",
            json.dumps(info.as_dict(), indent=2).encode(),
        )

    def info(self, name: str) -> ModelInfo:
        check_name(name)
        path = self.root / f"{name}.json"
        if not path.is_file():
            known = [i.name for i in self.list()]
            raise ValueError(f"no model named {name!r}; fitted models: {known}")
        return ModelInfo(**json.loads(path.read_text()))

    def load(self, name: str) -> tuple[ModelInfo, Any]:
        info = self.info(name)
        return info, pickle.loads((self.root / f"{name}.pkl").read_bytes())

    def list(self) -> list[ModelInfo]:
        if not self.root.is_dir():
            return []
        return [
            ModelInfo(**json.loads(p.read_text()))
            for p in sorted(self.root.glob("*.json"))
        ]


def check_name(name: str) -> None:
    if not _NAME.match(name):
        raise ValueError(
            f"model name {name!r}: use letters, digits, '.', '_' or '-' "
            "(at most 100, starting with a letter or digit)"
        )


def _write(path: Path, data: bytes) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_bytes(data)
    tmp.replace(path)
