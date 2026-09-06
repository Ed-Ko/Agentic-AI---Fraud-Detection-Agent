.PHONY: check build test all

PYTHON ?= python3
DATA_DIR ?= data/raw
RESULTS_FILE ?= results/metrics/baseline_results.json

check:
	$(PYTHON) scripts/check_environment.py --data-dir $(DATA_DIR)

build:
	$(PYTHON) scripts/build_models.py --data-dir $(DATA_DIR) --output $(RESULTS_FILE)

test:
	$(PYTHON) scripts/run_tests.py

all: check build test
