# Persona Capacity Simulator — Make targets
#
# Override on the command line, e.g.:
#   make run-cohort ENGINE=vllm MODEL=Qwen/Qwen2.5-7B-Instruct COHORT=chat_heavy

ENGINE  ?= vllm
MODEL   ?= Qwen/Qwen2.5-7B-Instruct
COHORT  ?= chat_heavy
CONFIG  ?= config/default.yaml
RUN_DIR ?= runs
PY      ?= python

.DEFAULT_GOAL := help

.PHONY: help
help:
	@echo "Persona Capacity Simulator — targets:"
	@echo ""
	@echo "  make setup                   Install package and dependencies"
	@echo "  make launch-engine           Launch engine only (manual testing)"
	@echo "  make run-cohort              Run a single cohort"
	@echo "  make run-sweep               Run all cohorts back-to-back"
	@echo "  make dashboard               Live progress view of active run"
	@echo "  make export                  Export buyer-page JSON from runs/"
	@echo "  make web                     Static-serve the reference buyer page"
	@echo "  make analyze-prefix-cache    Prefix-cache hit-rate analysis on the latest .db"
	@echo "  make test                    Run pytest"
	@echo "  make clean                   Remove caches"
	@echo "  make clean-runs              Remove all run databases"
	@echo ""
	@echo "Variables: ENGINE=$(ENGINE) MODEL=$(MODEL) COHORT=$(COHORT)"

.PHONY: setup
setup:
	$(PY) -m pip install --upgrade pip
	$(PY) -m pip install -e .

.PHONY: launch-engine
launch-engine:
	$(PY) -m simulator.cli launch-engine \
		--engine $(ENGINE) \
		--model $(MODEL) \
		--config $(CONFIG)

.PHONY: run-cohort
run-cohort:
	$(PY) -m simulator.cli run \
		--engine $(ENGINE) \
		--model $(MODEL) \
		--cohort $(COHORT) \
		--config $(CONFIG)

.PHONY: run-sweep
run-sweep:
	$(PY) -m simulator.cli sweep \
		--engine $(ENGINE) \
		--model $(MODEL) \
		--config $(CONFIG)

.PHONY: dashboard
dashboard:
	$(PY) -m simulator.cli dashboard --run-dir $(RUN_DIR)

.PHONY: export
export:
	$(PY) -m simulator.cli export \
		--input-dir $(RUN_DIR) \
		--output buyer_page_data.json

# Static-serve the reference buyer page on http://localhost:8765.
# Copies / symlinks buyer_page_data.json next to web/index.html so the
# default fetch path works without query params.
.PHONY: web
web: export
	@cp -f buyer_page_data.json web/buyer_page_data.json
	@echo "Reference buyer page at http://localhost:8765/"
	@cd web && $(PY) -m http.server 8765

# Prefix-cache analysis on a single .db (latest in $(RUN_DIR) by default).
.PHONY: analyze-prefix-cache
analyze-prefix-cache:
	@DB=$$(ls -t $(RUN_DIR)/*.db 2>/dev/null | head -n 1); \
	if [ -z "$$DB" ]; then echo "No .db in $(RUN_DIR)"; exit 1; fi; \
	$(PY) -m simulator.cli analyze-prefix-cache "$$DB"

.PHONY: test
test:
	$(PY) -m pytest tests/ -q

.PHONY: clean
clean:
	rm -rf __pycache__ .pytest_cache .ruff_cache
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name '*.pyc' -delete

.PHONY: clean-runs
clean-runs:
	rm -rf $(RUN_DIR)/*.db $(RUN_DIR)/*.db-journal buyer_page_data.json
