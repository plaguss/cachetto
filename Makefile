sources = src/cachetto tests

.PHONY: .uv
.uv: ## Check that uv is installed
	@uv --version || echo 'Please install uv: https://docs.astral.sh/uv/getting-started/installation/'

.PHONY: .pre-commit
.pre-commit: ## Check that pre-commit is installed
	@pre-commit -V || echo 'Please install pre-commit: https://pre-commit.com/'

.PHONY: install
install: .uv .pre-commit ## Install the package, dependencies, and pre-commit for local development
	uv sync --group dev

.PHONY: install-all-python
install-all-python: ## Install and synchronize an interpreter for every python version
	UV_PROJECT_ENVIRONMENT=.venv310 uv sync --python 3.10 --group dev
	UV_PROJECT_ENVIRONMENT=.venv311 uv sync --python 3.11 --group dev
	UV_PROJECT_ENVIRONMENT=.venv312 uv sync --python 3.12 --group dev
	UV_PROJECT_ENVIRONMENT=.venv313 uv sync --python 3.13 --group dev

.PHONY: test-all-python
test-all-python: install-all-python ## Run tests on Python 3.9 to 3.13
	UV_PROJECT_ENVIRONMENT=.venv310 uv run --python 3.10 coverage run -m pytest tests/unit
	UV_PROJECT_ENVIRONMENT=.venv311 uv run --python 3.11 coverage run -m pytest tests/unit
	UV_PROJECT_ENVIRONMENT=.venv312 uv run --python 3.12 coverage run -m pytest tests/unit
	UV_PROJECT_ENVIRONMENT=.venv313 uv run --python 3.13 coverage run -m pytest tests/unit
	@uv run coverage combine
	@uv run coverage report

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

.PHONY: typecheck-mypy
typecheck-mypy:
	uv run mypy src/cachetto

.PHONY: unit-tests
unit-tests:
	uv run pytest tests/unit

.PHONY: cov-tests
cov-tests:
	uv run pytest tests/unit --cov=cachetto --cov-report=html

.PHONY: release
release:
	uv build
	uv publish --token "${PYPI_TOKEN}"
	uv run --with cachetto --no-project -- python -c "from cachetto import cached"
