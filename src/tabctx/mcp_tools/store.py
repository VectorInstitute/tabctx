"""Fitted models on disk, by name.

Each model is two files in the store's directory: <name>.pkl, the fitted
scikit-learn estimator (it keeps its training table, so it can fit the
server again if the server dropped the context), and <name>.json, what an
agent needs to know about it (listed without unpickling anything).

Unpickling runs code, and the store lives in the working directory, where
a cloned repository could plant files. So every pickle is signed: the
JSON holds an HMAC of the pickle under a key kept in the user's own
config directory (created on first use, readable by the user only).
load() checks the signature before unpickling, and list() leaves out
models it cannot verify. The API key is never in the pickles
(TabctxClient leaves it out when pickled).
"""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import pickle
import re
import secrets
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

HOME_ENV = "TABCTX_MCP_HOME"
KEY_ENV = "TABCTX_MCP_KEY_FILE"
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

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ModelInfo:
        """Fields this version doesn't know (from a newer one) are ignored."""
        known = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in data.items() if k in known})


class ModelStore:
    def __init__(
        self, root: str | Path | None = None, key_file: str | Path | None = None
    ) -> None:
        """root: the directory to keep models in; default $TABCTX_MCP_HOME,
        else .tabctx-mcp in the working directory. key_file: the signing
        key; default $TABCTX_MCP_KEY_FILE, else
        ~/.config/tabctx-mcp/signing.key."""
        self.root = Path(root or os.environ.get(HOME_ENV) or DEFAULT_HOME)
        self.key_file = Path(
            key_file
            or os.environ.get(KEY_ENV)
            or Path.home() / ".config" / "tabctx-mcp" / "signing.key"
        )

    def save(self, info: ModelInfo, estimator: Any) -> None:
        check_name(info.name)
        self.root.mkdir(parents=True, exist_ok=True)
        data = pickle.dumps(estimator)
        record = {**info.as_dict(), "signature": self._sign(data)}
        # The pickle first: a listed model always loads.
        _write(self.root / f"{info.name}.pkl", data)
        _write(self.root / f"{info.name}.json", json.dumps(record, indent=2).encode())

    def exists(self, name: str) -> bool:
        check_name(name)
        return (self.root / f"{name}.json").is_file()

    def info(self, name: str) -> ModelInfo:
        return ModelInfo.from_dict(self._record(name))

    def load(self, name: str) -> tuple[ModelInfo, Any]:
        record = self._record(name)
        data = (self.root / f"{name}.pkl").read_bytes()
        if not hmac.compare_digest(str(record.get("signature", "")), self._sign(data)):
            raise ValueError(
                f"model {name!r} in {self.root} was not saved by your tabctx-mcp "
                "(its signature does not match), so it is not loaded; fit it again"
            )
        return ModelInfo.from_dict(record), pickle.loads(data)

    def list(self) -> list[ModelInfo]:
        """The models this user's tabctx-mcp saved (others are left out)."""
        if not self.root.is_dir():
            return []
        out = []
        for meta in sorted(self.root.glob("*.json")):
            try:
                record = json.loads(meta.read_text())
                data = (self.root / f"{meta.stem}.pkl").read_bytes()
            except (OSError, ValueError):
                continue
            if hmac.compare_digest(str(record.get("signature", "")), self._sign(data)):
                out.append(ModelInfo.from_dict(record))
        return out

    def _record(self, name: str) -> dict[str, Any]:
        check_name(name)
        path = self.root / f"{name}.json"
        if not path.is_file():
            known = [i.name for i in self.list()]
            raise ValueError(f"no model named {name!r}; fitted models: {known}")
        return json.loads(path.read_text())

    def _sign(self, data: bytes) -> str:
        return hmac.new(self._key(), data, hashlib.sha256).hexdigest()

    def _key(self) -> bytes:
        try:
            return bytes.fromhex(self.key_file.read_text().strip())
        except FileNotFoundError:
            pass
        self.key_file.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        key = secrets.token_bytes(32)
        try:
            # O_EXCL: two servers starting at once agree on one key.
            fd = os.open(self.key_file, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        except FileExistsError:
            return bytes.fromhex(self.key_file.read_text().strip())
        with os.fdopen(fd, "w") as f:
            f.write(key.hex())
        return key


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
