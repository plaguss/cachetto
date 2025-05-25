sources = src/dfcache tests

.PHONY: format
format:
	ruff --version
	ruff check --fix $(sources)
	ruff format $(sources)

.PHONY: lint
lint:
	ruff --version
	ruff check $(sources)
	ruff format --check $(sources)

.PHONY: unit-tests
unit-tests:
	pytest tests/unit

.PHONY: cov-tests
cov-tests:
	pytest tests/unit --cov=dfcache --cov-report=html
