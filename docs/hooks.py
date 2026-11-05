"""MkDocs build hooks for the tabctx docs site.

on_post_build writes the service's OpenAPI spec next to the HTTP API
reference page, generated from the routes themselves (with the fake
backend, so no GPU or model weights are needed), so the reference can't
drift from the code.
"""

from __future__ import annotations

import json
import os
from pathlib import Path


def on_post_build(config, **kwargs) -> None:
    os.environ.setdefault("TABCTX_BACKEND", "fake")
    from tabctx.serve.standalone import make_app

    spec = make_app().openapi()
    out = Path(config["site_dir"]) / "reference" / "openapi.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(spec, indent=1))
