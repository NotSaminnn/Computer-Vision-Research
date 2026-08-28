# Intervene3D -- developer entry points.
# Every target here is also documented in README.md.
PY ?= .venv/bin/python
SEED ?= 42

.PHONY: help setup install smoke test unit integration data experiment aggregate figures validate-datasets clean-pyc lint

help:
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-22s\033[0m %s\n", $$1, $$2}'

setup:  ## Create the environment and verify the installation
	bash scripts/setup_environment.sh

install:  ## Install the package in editable mode into the active interpreter
	$(PY) -m pip install -e ".[dev]"

data:  ## Generate the tiny synthetic benchmark used by the smoke test
	$(PY) scripts/generate_synthetic_data.py --config configs/synthetic/smoke.yaml

smoke:  ## Run the mandatory end-to-end smoke test
	bash scripts/run_smoke_test.sh

test:  ## Run the full test suite
	$(PY) -m pytest tests -q

unit:  ## Run unit tests only
	$(PY) -m pytest tests/unit -q

integration:  ## Run integration tests only
	$(PY) -m pytest tests/integration -q

experiment:  ## Run the Phase 1 problem-existence experiment (SEED=42 by default)
	$(PY) scripts/run_experiment.py --config configs/experiments/phase1_problem_existence.yaml --seed $(SEED)

aggregate:  ## Aggregate every run of the Phase 1 experiment across seeds
	$(PY) scripts/aggregate_results.py --experiment phase1_problem_existence

figures:  ## Regenerate every figure from saved result files
	$(PY) scripts/generate_all_figures.py

validate-datasets:  ## Validate all registered external datasets
	$(PY) scripts/validate_datasets.py --all

lint:  ## Lint with ruff (if installed)
	$(PY) -m ruff check src scripts tests || true

clean-pyc:
	find . -name '__pycache__' -type d -prune -exec rm -rf {} + 2>/dev/null || true
