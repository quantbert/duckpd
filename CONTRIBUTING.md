# Contributing

DuckPD uses Python 3.11+, `uv`, pytest, Ruff, and Pyright.

## Setup

```bash
uv sync --frozen --group dev
uv run pre-commit install
```

## Checks

Run focused tests while developing, then run the complete gate before opening a
pull request:

```bash
make check
make build
```

The equivalent commands, useful on systems without GNU Make, are:

```bash
uv run pytest
uv run ruff check .
uv run ruff format --check .
uv run pyright
uv build
```

Every public operation must remain lazy until an explicit execution boundary,
update frame metadata, reject unsupported arguments before execution, and have
differential tests against the supported pandas behavior.

Do not add an automatic pandas fallback. Unsupported behavior should raise a
specific exception.

### Error Messages and Diagnostics

Error messages must be actionable, precise, and user-facing:
- State what condition or contract failed.
- Explain how the user can resolve the issue (e.g. declare `order_by=` when
	creating a SQL/table source or sort using `.sort_values(...)`).
- Distinguish API call-site parameters from pipeline configuration (avoid wording that implies an operation accepts an unsupported parameter).