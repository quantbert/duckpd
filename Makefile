.DEFAULT_GOAL := help

.PHONY: help install test remote-db-setup remote-db-test remote-db-cleanup lint format format-check typecheck compatibility-check check build package-smoke demos-smoke benchmark benchmark-all benchmark-tracks optimizer-gate bump release-check clean publish

help: ## Show available targets
	@grep -E '^[a-z-]+:.*##' $(MAKEFILE_LIST) | awk -F ':.*## ' '{printf "  %-20s %s\n", $$1, $$2}'

install: ## Sync the locked development environment
	uv sync --frozen --group dev

test: ## Run tests with the configured coverage requirements
	uv run pytest

REMOTE_POSTGRES_CONTAINER ?= duckpd-postgres
REMOTE_MYSQL_CONTAINER ?= duckpd-mysql
REMOTE_DB_NAME ?= duckpd
REMOTE_DB_USER ?= duckpd
REMOTE_DB_PASSWORD ?= duckpd_secret
REMOTE_MYSQL_ROOT_PASSWORD ?= root_secret

remote-db-setup: ## Start disposable PostgreSQL and MySQL test containers
	@command -v docker >/dev/null || { echo "docker is required"; exit 1; }
	@docker rm -f $(REMOTE_POSTGRES_CONTAINER) $(REMOTE_MYSQL_CONTAINER) >/dev/null 2>&1 || true
	docker run --rm -d --name $(REMOTE_POSTGRES_CONTAINER) \
		-e POSTGRES_DB=$(REMOTE_DB_NAME) \
		-e POSTGRES_USER=$(REMOTE_DB_USER) \
		-e POSTGRES_PASSWORD=$(REMOTE_DB_PASSWORD) \
		-p 5432:5432 postgres:17-alpine
	docker run --rm -d --name $(REMOTE_MYSQL_CONTAINER) \
		-e MYSQL_DATABASE=$(REMOTE_DB_NAME) \
		-e MYSQL_USER=$(REMOTE_DB_USER) \
		-e MYSQL_PASSWORD=$(REMOTE_DB_PASSWORD) \
		-e MYSQL_ROOT_PASSWORD=$(REMOTE_MYSQL_ROOT_PASSWORD) \
		-p 3306:3306 mysql:8.4
	@echo "Waiting for PostgreSQL..."
	@attempt=0; \
	until docker exec $(REMOTE_POSTGRES_CONTAINER) pg_isready -U $(REMOTE_DB_USER) -d $(REMOTE_DB_NAME) >/dev/null 2>&1; do \
		attempt=$$((attempt + 1)); \
		if [ $$attempt -ge 90 ]; then \
			echo "PostgreSQL did not become ready within 90 seconds."; \
			docker logs $(REMOTE_POSTGRES_CONTAINER); \
			exit 1; \
		fi; \
		sleep 1; \
	done
	@echo "Waiting for MySQL..."
	@attempt=0; \
	until docker exec $(REMOTE_MYSQL_CONTAINER) mysqladmin ping -h localhost -u root -p$(REMOTE_MYSQL_ROOT_PASSWORD) --silent >/dev/null 2>&1; do \
		attempt=$$((attempt + 1)); \
		if [ $$attempt -ge 120 ]; then \
			echo "MySQL did not become ready within 120 seconds."; \
			docker logs $(REMOTE_MYSQL_CONTAINER); \
			exit 1; \
		fi; \
		sleep 1; \
	done
	@echo "PostgreSQL and MySQL are ready."

remote-db-test: remote-db-setup ## Start databases and run live attachment tests
	DUCKPD_TEST_POSTGRES_HOST=127.0.0.1 \
	DUCKPD_TEST_POSTGRES_PORT=5432 \
	DUCKPD_TEST_POSTGRES_DATABASE=$(REMOTE_DB_NAME) \
	DUCKPD_TEST_POSTGRES_USER=$(REMOTE_DB_USER) \
	DUCKPD_TEST_POSTGRES_PASSWORD=$(REMOTE_DB_PASSWORD) \
	DUCKPD_TEST_MYSQL_HOST=127.0.0.1 \
	DUCKPD_TEST_MYSQL_PORT=3306 \
	DUCKPD_TEST_MYSQL_DATABASE=$(REMOTE_DB_NAME) \
	DUCKPD_TEST_MYSQL_USER=$(REMOTE_DB_USER) \
	DUCKPD_TEST_MYSQL_PASSWORD=$(REMOTE_DB_PASSWORD) \
	uv run pytest -o addopts='--strict-config --strict-markers' tests/test_remote_attachments.py

remote-db-cleanup: ## Stop and remove disposable database containers
	@docker rm -f $(REMOTE_POSTGRES_CONTAINER) $(REMOTE_MYSQL_CONTAINER) >/dev/null 2>&1 || true
	@echo "Disposable PostgreSQL and MySQL containers removed."

lint: ## Run Ruff lint checks
	uv run ruff check .

format: ## Format the repository with Ruff
	uv run ruff format .

format-check: ## Verify Ruff formatting without changing files
	uv run ruff format --check .

typecheck: ## Run strict Pyright checks
	uv run pyright

compatibility-check: ## Verify generated compatibility documentation
	uv run python scripts/generate_compatibility.py --check

check: lint format-check typecheck compatibility-check test ## Run the complete local quality gate

build: ## Build the wheel and source distribution
	rm -rf dist
	uv build

PYTHONS ?= 3.11 3.12 3.13 3.14

package-smoke: build ## Inspect artifacts and clean-install on supported Python versions
	uv run python scripts/package_smoke.py dist --artifacts-only
	@for python in $(PYTHONS); do \
		uv run python scripts/package_smoke.py dist --python $$python || exit 1; \
	done

demos-smoke: ## Run the inexpensive executable demos
	uv run python demo/basic_pipeline.py
	uv run python demo/parquet_pipeline.py
	uv run python demo/reduction_pipeline.py
	uv run python demo/generate_market_data.py smoke

SIZES ?= 5mb 50mb 500m
REPETITIONS ?= 3
THREADS ?= 4
REPORT ?= benchmark/REPORT.md

benchmark: ## Run benchmarks across file sizes and generate Markdown report
	uv run python -m benchmark --sizes $(SIZES) --repetitions $(REPETITIONS) --threads $(THREADS) --report $(REPORT)

benchmark-all: ## Run benchmarks across all preset sizes including 5GB and 50GB
	uv run python -m benchmark --sizes 5mb 50mb 500m 5g 50g --repetitions $(REPETITIONS) --threads $(THREADS) --report $(REPORT)

benchmark-tracks: ## Run validated cold/warm tracks and evidence scorecard
	uv run python -m benchmark.tracks --rows 100000 --output benchmark/TRACKS.json --scorecard-output benchmark/SCORECARD.json

optimizer-gate: ## Verify optimizer correctness and regression threshold
	uv run python scripts/benchmark_optimizer.py --rows 250000 --iterations 7

# Support both `make bump patch`, `make bump minor`, `make bump 0.1.4` and `make bump PART=0.1.4`
ifeq (bump,$(firstword $(MAKECMDGOALS)))
  BUMP_ARGS := $(wordlist 2,$(words $(MAKECMDGOALS)),$(MAKECMDGOALS))
  ifneq ($(BUMP_ARGS),)
    PART ?= $(firstword $(BUMP_ARGS))
    $(eval $(BUMP_ARGS):;@:)
  endif
endif
PART ?= patch

bump: ## Bump version, update lockfile, and verify changelog (e.g. make bump [patch|minor|major|0.1.4])
	uv run python scripts/bump_version.py $(PART)

release-check: check package-smoke ## Validate source, metadata, and installed artifacts
	uv run python scripts/verify_release.py

publish: release-check ## Publish the already-versioned immutable release
	uv publish

clean: ## Remove caches, coverage data, and build artifacts
	rm -rf .pytest_cache .ruff_cache .hypothesis htmlcov build dist
	rm -f .coverage coverage.xml
	find . -type d \( -path './.git' -o -path './.venv' \) -prune -o -type d -name __pycache__ -exec rm -rf {} +
