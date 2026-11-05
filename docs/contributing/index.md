# Contributing

Issues and pull requests are welcome. Good places to start are issues
labelled
[`good first issue`](https://github.com/VectorInstitute/tabctx/labels/good%20first%20issue)
and [`help wanted`](https://github.com/VectorInstitute/tabctx/labels/help%20wanted),
and [adding a model](adding-a-model.md).

## Set up

```bash
git clone https://github.com/VectorInstitute/tabctx && cd tabctx
uv sync --dev --extra mcp
uv run pre-commit install
```

## Test

No GPU is needed: the unit tests run the whole engine and HTTP service
against a fake model.

```bash
uv run pytest tests/unit -n auto    # in parallel; about 10 s
uv run pytest tests/integration     # boots a local Ray Serve cluster
```

Tests should be fast, deterministic and independent: no fixed ports, no
shared state, and threads synchronized with events rather than sleeps.

## Docs

```bash
uv sync --group docs
uv run mkdocs serve                 # http://127.0.0.1:8000
```

## Pull requests

- One change per pull request, with tests for the behaviour it adds or
  fixes and a line in `CHANGELOG.md` under *Unreleased*.
- `uv run pre-commit run --all-files` passes (ruff lint and format).
