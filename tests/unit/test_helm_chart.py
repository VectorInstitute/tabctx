"""The Helm chart released with tabctx deploys the matching image: its
version and appVersion move with the package's (a release PR bumps all
three)."""

import re
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def _chart_field(name: str) -> str:
    text = (ROOT / "deploy/helm/tabctx/Chart.yaml").read_text()
    match = re.search(rf"^{name}:\s*\"?([^\"\s]+)\"?\s*$", text, re.M)
    assert match, f"Chart.yaml has no {name}"
    return match.group(1)


def test_chart_versions_match_the_package():
    version = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]["version"]
    assert _chart_field("version") == version
    assert _chart_field("appVersion") == version
