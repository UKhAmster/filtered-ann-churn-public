.PHONY: test lint fmt
PYTHON ?= .venv/bin/python
RUFF ?= .venv/bin/ruff

test:
	$(PYTHON) -m pytest -v

lint:
	$(RUFF) check src tests

fmt:
	$(RUFF) format src tests
