"""The serving app imports names from tabctx modules; check they exist
without importing the app itself (which needs Ray, absent from the unit
environment -- a missing export once passed every unit test and only
failed when Ray Serve started the replica)."""

import ast
import importlib
from pathlib import Path

import pytest

SERVE = Path(__file__).resolve().parents[2] / "src" / "tabctx" / "serve"


def _tabctx_imports(path: Path):
    for node in ast.walk(ast.parse(path.read_text())):
        if isinstance(node, ast.ImportFrom) and (node.module or "").startswith(
            "tabctx"
        ):
            for alias in node.names:
                yield node.module, alias.name


@pytest.mark.parametrize("path", sorted(SERVE.glob("*.py")), ids=lambda p: p.name)
def test_every_tabctx_name_the_serve_layer_imports_exists(path):
    missing = []
    for module, name in _tabctx_imports(path):
        try:
            mod = importlib.import_module(module)
        except ImportError as e:  # an optional dependency of that module
            if "tabctx" in str(e):
                missing.append(f"{module} ({e})")
            continue
        if not hasattr(mod, name):
            try:
                importlib.import_module(f"{module}.{name}")
            except ImportError:
                missing.append(f"{module}.{name}")
    assert not missing, missing
