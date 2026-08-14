# Contributing

DuckPD uses Python 3.11+, `uv`, pytest, Ruff, and Pyright.

## Setup

```bash
uv sync --group dev
uv run pre-commit install
```

## Checks

Run focused tests while developing, then run the complete gate before opening a
pull request:

```bash
uv run pytest
uv run ruff check .
uv run pyright
uv build
```

Every public operation must remain lazy until an explicit execution boundary,
update frame metadata, reject unsupported arguments before execution, and have
differential tests against the supported pandas behavior.

Do not add an automatic pandas fallback. Unsupported behavior should raise a
specific exception.