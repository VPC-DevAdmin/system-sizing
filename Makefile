# Persona Capacity Simulator — Make targets
#
# Headline workflow:
#   make ready CONFIG=config/r7735_sglang_qwen3_30b_a3b.yaml
#   make run-cohort CONFIG=config/r7735_sglang_qwen3_30b_a3b.yaml \
#                   COHORT=chat_heavy
#   make dashboard
#   make export
#   make web

# Engine + model are read from CONFIG by default. Set ENGINE=... or
# MODEL=... on the command line ONLY when you want to override what the
# YAML says (rare). The previous defaults silently overrode the YAML.
ENGINE  ?=
MODEL   ?=
COHORT  ?= chat_heavy
# For run-sweep: 'all' | 'singles' | 'mixes' | a,b,c list of cohort ids.
COHORTS ?= all
CONFIG  ?= config/default.yaml
RUN_DIR ?= runs

# Project-local venv. ``make ready`` creates it if missing; every other
# target uses it via $(PY) when present, falling back to system python
# otherwise. Override with ``PY=/path/to/python`` on the command line if
# you'd rather use a venv elsewhere.
VENV    ?= .venv
PY      ?= $(if $(wildcard $(VENV)/bin/python),$(VENV)/bin/python,python3)

# SGLang Docker pipeline knobs (used by the hidden sub-targets).
SGLANG_REPO         ?= https://github.com/sgl-project/sglang.git
SGLANG_SRC          ?= /tmp/sglang
SGLANG_BASE_IMAGE   ?= sglang-cpu:xeon
SGLANG_FIXED_IMAGE  ?= sglang-cpu:xeon-fixed
SGLANG_DOCKERFILE   ?= $(SGLANG_SRC)/docker/xeon.Dockerfile

# Model staging (used by the hidden download-model sub-target).
MODELS_DIR          ?= /data/ml/models
HF_CACHE_DIR        ?= /data/ml/huggingface
LOCAL_MODEL_DIR     ?= $(notdir $(MODEL))

.DEFAULT_GOAL := help

.PHONY: help
help:
	@echo "Persona Capacity Simulator"
	@echo ""
	@echo "Setup:"
	@echo "  make ready CONFIG=...        Install deps, build engine image, download model, preflight"
	@echo ""
	@echo "Run:"
	@echo "  make run-cohort CONFIG=... COHORT=...   Run a single cohort"
	@echo "  make run-sweep  CONFIG=... [COHORTS=]   Sweep cohorts. COHORTS=all|singles|mixes|a,b,c"
	@echo "  make list-cohorts                       List available cohorts grouped by category"
	@echo "  make dashboard                          Live progress view of latest run"
	@echo ""
	@echo "After runs:"
	@echo "  make export                             Build buyer_page_data.json"
	@echo "  make web                                Static-serve the buyer page (http://localhost:8765)"
	@echo "  make analyze-prefix-cache               Prefix-cache hit-rate report"
	@echo ""
	@echo "Diagnostics:"
	@echo "  make preflight CONFIG=...               Hardware-only check (no install / build)"
	@echo "  make launch-engine CONFIG=...           Manually launch the engine (no cohort)"
	@echo "  make sglang-shell                       Interactive bash inside sglang-cpu:xeon-fixed"
	@echo "  make test                               Run pytest"
	@echo "  make clean / clean-runs / clean-venv    Tidy"
	@echo ""
	@echo "Variables: CONFIG=$(CONFIG)  COHORT=$(COHORT)  MODEL=$(MODEL)"

# ── Headline target ───────────────────────────────────────────────────
# `ready` is idempotent. First run: creates a project-local venv if
# missing, installs deps, builds the docker image (SGLang only),
# downloads the model, validates hardware. Re-runs skip already-done
# steps (the venv check, image inspect, and model dir check are all
# no-ops when state is already good).
.PHONY: ready
ready: $(VENV)/bin/python
	$(VENV)/bin/python -m pip install --upgrade pip
	$(VENV)/bin/python -m pip install -e .
	@if [ -f "$(CONFIG)" ]; then \
		$(VENV)/bin/python -m simulator.cli ready --config "$(CONFIG)"; \
	else \
		echo "WARN: $(CONFIG) not found; skipped engine/model prep"; \
	fi
	@echo ""
	@echo "==> $(VENV) ready. To work interactively: source $(VENV)/bin/activate"

# Create the venv if missing. Uses python3 (system) for the bootstrap;
# every other invocation uses the venv's python via $(PY).
$(VENV)/bin/python:
	@echo "==> Creating venv at $(VENV)"
	python3 -m venv "$(VENV)"

# Back-compat alias.
.PHONY: setup
setup: ready

# ── Run targets ───────────────────────────────────────────────────────

.PHONY: run-cohort
run-cohort:
	$(PY) -m simulator.cli run \
		--cohort $(COHORT) \
		--config $(CONFIG) \
		$(if $(ENGINE),--engine $(ENGINE)) \
		$(if $(MODEL),--model $(MODEL))

.PHONY: run-sweep
run-sweep:
	$(PY) -m simulator.cli sweep \
		--config $(CONFIG) \
		--cohorts $(COHORTS) \
		$(if $(ENGINE),--engine $(ENGINE)) \
		$(if $(MODEL),--model $(MODEL))

.PHONY: dashboard
dashboard:
	$(PY) -m simulator.cli dashboard --run-dir $(RUN_DIR)

.PHONY: list-cohorts
list-cohorts:
	$(PY) -m simulator.cli list-cohorts

# ── Output / analysis ─────────────────────────────────────────────────

.PHONY: export
export:
	$(PY) -m simulator.cli export \
		--input-dir $(RUN_DIR) \
		--output buyer_page_data.json

.PHONY: web
web: export
	@cp -f buyer_page_data.json web/buyer_page_data.json
	@echo "Reference buyer page at http://localhost:8765/"
	@cd web && $(PY) -m http.server 8765

.PHONY: analyze-prefix-cache
analyze-prefix-cache:
	@DB=$$(ls -t $(RUN_DIR)/*.db 2>/dev/null | head -n 1); \
	if [ -z "$$DB" ]; then echo "No .db in $(RUN_DIR)"; exit 1; fi; \
	$(PY) -m simulator.cli analyze-prefix-cache "$$DB"

# ── Diagnostics ───────────────────────────────────────────────────────

.PHONY: preflight
preflight:
	$(PY) -m simulator.cli preflight --config $(CONFIG)

.PHONY: launch-engine
launch-engine:
	$(PY) -m simulator.cli launch-engine \
		--config $(CONFIG) \
		$(if $(ENGINE),--engine $(ENGINE)) \
		$(if $(MODEL),--model $(MODEL))

.PHONY: sglang-shell
sglang-shell:
	docker run --rm -it \
		-v "$(MODELS_DIR):/models" \
		-v "$(HF_CACHE_DIR):/root/.cache/huggingface" \
		--entrypoint /bin/bash \
		"$(SGLANG_FIXED_IMAGE)"

.PHONY: test
test:
	$(PY) -m pytest tests/ -q

.PHONY: clean
clean:
	rm -rf __pycache__ .pytest_cache .ruff_cache
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name '*.pyc' -delete

.PHONY: clean-venv
clean-venv:
	rm -rf $(VENV)

.PHONY: clean-runs
clean-runs:
	rm -rf $(RUN_DIR)/*.db $(RUN_DIR)/*.db-journal buyer_page_data.json

# ── Hidden sub-targets ────────────────────────────────────────────────
# These power the SGLang docker pipeline. ``make ready`` calls into the
# Python equivalents (``simulator ready``); the make-side targets remain
# as escape hatches for when the python orchestration mis-detects state.

.PHONY: sglang-clone
sglang-clone:
	@if [ -d "$(SGLANG_SRC)/.git" ]; then \
		echo "==> Updating $(SGLANG_SRC)"; \
		cd "$(SGLANG_SRC)" && git fetch --depth 1 origin && git reset --hard origin/HEAD; \
	else \
		echo "==> Cloning $(SGLANG_REPO) -> $(SGLANG_SRC)"; \
		git clone --depth 1 "$(SGLANG_REPO)" "$(SGLANG_SRC)"; \
	fi
	@test -f "$(SGLANG_DOCKERFILE)" || \
		(echo "ERROR: $(SGLANG_DOCKERFILE) not found." && exit 1)

.PHONY: sglang-base
sglang-base: sglang-clone
	docker build -f "$(SGLANG_DOCKERFILE)" -t "$(SGLANG_BASE_IMAGE)" "$(SGLANG_SRC)"

.PHONY: sglang-fixed
sglang-fixed: sglang-base Dockerfile.xeon-fixed
	docker build -f Dockerfile.xeon-fixed -t "$(SGLANG_FIXED_IMAGE)" .

.PHONY: sglang-verify
sglang-verify:
	docker run --rm "$(SGLANG_FIXED_IMAGE)" /opt/.venv/bin/python -c \
		"import sglang, sgl_kernel, sentencepiece, tiktoken; \
		from google import protobuf; \
		print('OK', sglang.__version__)"

.PHONY: sglang-build
sglang-build: sglang-fixed sglang-verify

.PHONY: models-dirs
models-dirs:
	@if [ ! -d "$(MODELS_DIR)" ] || [ ! -w "$(MODELS_DIR)" ]; then \
		sudo mkdir -p "$(MODELS_DIR)" "$(HF_CACHE_DIR)"; \
		sudo chown -R $$USER:$$USER "$(MODELS_DIR)" "$(HF_CACHE_DIR)"; \
	fi

.PHONY: download-model
download-model: models-dirs
	HF_HUB_ENABLE_HF_TRANSFER=1 hf download "$(MODEL)" \
		--local-dir "$(MODELS_DIR)/$(LOCAL_MODEL_DIR)"
