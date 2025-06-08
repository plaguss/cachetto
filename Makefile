sources = src/dfcache tests

.PHONY: .uv
.uv: ## Check that uv is installed
	@uv --version || echo 'Please install uv: https://docs.astral.sh/uv/getting-started/installation/'

.PHONY: .pre-commit
.pre-commit: ## Check that pre-commit is installed
	@pre-commit -V || echo 'Please install pre-commit: https://pre-commit.com/'

.PHONY: install
install: .uv .pre-commit ## Install the package, dependencies, and pre-commit for local development
	uv sync --group dev

.PHONY: format
format:
	uv run ruff --version
	uv run ruff check --fix $(sources)
	uv run ruff format $(sources)

.PHONY: lint
lint:
	uv run ruff --version
	uv run ruff check $(sources)
	uv run ruff format --check $(sources)

.PHONY: unit-tests
unit-tests:
	uv run pytest tests/unit

.PHONY: cov-tests
cov-tests:
	uv run pytest tests/unit --cov=dfcache --cov-report=html
