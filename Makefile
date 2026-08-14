.DEFAULT_GOAL := help

.PHONY: help install test lint format format-check typecheck check build demos-smoke release-check clean publish

help: ## Show available targets
	@grep -E '^[a-z-]+:.*##' $(MAKEFILE_LIST) | awk -F ':.*## ' '{printf "  %-20s %s\n", $$1, $$2}'

install: ## Sync the locked development environment
	uv sync --frozen --group dev

test: ## Run tests with the configured coverage requirements
	uv run pytest

lint: ## Run Ruff lint checks
	uv run ruff check .

format: ## Format the repository with Ruff
	uv run ruff format .

format-check: ## Verify Ruff formatting without changing files
	uv run ruff format --check .

typecheck: ## Run strict Pyright checks
	uv run pyright

check: lint format-check typecheck test ## Run the complete local quality gate

build: ## Build the wheel and source distribution
	rm -rf dist
	uv build

demos-smoke: ## Run the inexpensive executable demos
	uv run python demo/basic_pipeline.py
	uv run python demo/parquet_pipeline.py
	uv run python demo/reduction_pipeline.py
	uv run python demo/generate_market_data.py smoke

release-check: check build ## Validate source and build release artifacts

publish: test ## Run tests, bump patch version, build, and publish to PyPI
	uv version --bump patch
	rm -rf dist
	uv build
	uv publish

clean: ## Remove caches, coverage data, and build artifacts
	rm -rf .pytest_cache .ruff_cache .hypothesis htmlcov build dist
	rm -f .coverage coverage.xml
	find . -type d \( -path './.git' -o -path './.venv' \) -prune -o -type d -name __pycache__ -exec rm -rf {} +
