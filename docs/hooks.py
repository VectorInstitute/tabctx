"""MkDocs build hooks for the tabctx docs site.

on_post_build writes the service's OpenAPI spec next to the HTTP API
reference page, generated from the routes themselves, so the reference
can't drift from the code. No service is built: nothing reads the
environment, loads a model or touches a GPU.
"""

from __future__ import annotations

import json
from pathlib import Path


def on_post_build(config, **_) -> None:
    from tabctx.serve.standalone import openapi_spec

    out = Path(config["site_dir"]) / "reference" / "openapi.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(openapi_spec(), indent=1))
