# PY picks the interpreter: `make qc PY=/path/to/env/bin/python`, or set PY in the environment.
PY ?= python

.PHONY: qc lint test fmt

# What CI runs and what the Stop hook runs. The layering check joins it when the modules move
# (docs/ARCHITECTURE.md).
qc: lint test

lint:
	$(PY) -m ruff check caudal tests scripts

test:
	$(PY) -m pytest tests/

fmt:
	$(PY) -m ruff check --fix caudal tests scripts
