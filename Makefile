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
PERSONA ?= quick_lookup
# What to sweep: 'all' | 'personas' | 'cohorts' | a,b,c list of ids.
SWEEP_TYPE ?= all
CONFIG  ?= config/default.yaml
RUN_DIR ?= runs
# RUN_NEW=true cuts a fresh runs/run_NN+1/ instead of resuming the
# latest run_NN. Resume-by-default lets ``make run-sweep`` pick up
# where an interrupted earlier invocation left off.
RUN_NEW ?=

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
	@echo "  make run-persona CONFIG=... PERSONA=... Run one persona (a single user archetype)"
	@echo "  make run-cohort  CONFIG=... COHORT=...  Run one cohort (a team mix of personas)"
	@echo "  make run-sweep   CONFIG=... [SWEEP_TYPE=]  Sweep multiple workloads."
	@echo "                                          SWEEP_TYPE=all|personas|cohorts|a,b,c"
	@echo "                                          Resumes latest run_NN by default;"
	@echo "                                          set RUN_NEW=true to cut a fresh run dir."
	@echo "  make run-*-bg                           Same as above but nohup'd; survives SSH disconnects"
	@echo "  make tail-log / make stop-bg            Tail latest -bg log / stop a backgrounded run + engine"
	@echo "  make list-runs                          List run_NN/ directories under $(RUN_DIR)"
	@echo "  make list-personas                      List available user archetypes"
	@echo "  make list-cohorts                       List available team mixes"
	@echo "  make dashboard                          Live progress view of latest run_NN"
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

.PHONY: run-persona
run-persona:
	$(PY) -m simulator.cli run-persona \
		--persona $(PERSONA) \
		--config $(CONFIG) \
		$(if $(ENGINE),--engine $(ENGINE)) \
		$(if $(MODEL),--model $(MODEL))

.PHONY: run-sweep
run-sweep:
	$(PY) -m simulator.cli sweep \
		--config $(CONFIG) \
		--type $(SWEEP_TYPE) \
		$(if $(RUN_NEW),--new-run) \
		$(if $(ENGINE),--engine $(ENGINE)) \
		$(if $(MODEL),--model $(MODEL))

.PHONY: dashboard
dashboard:
	$(PY) -m simulator.cli dashboard --run-dir $(RUN_DIR)

# Long-running variants — nohup + dated log file. Survive SSH
# disconnects; check status from another login via ``make dashboard``.
# For -bg targets we resolve the current run_NN/ up front (creating it
# if needed) and drop the log inside, so every artifact for a run lives
# in one directory. RUN_NEW=true on run-sweep-bg cuts a fresh run_NN+1.
RESOLVE_RUN_DIR = $(PY) -m simulator.cli current-run-dir --base $(RUN_DIR) $(if $(RUN_NEW),--new-run)

.PHONY: run-cohort-bg
run-cohort-bg:
	@RD="$$($(RESOLVE_RUN_DIR))" ; \
	LOG="$$RD/cohort_$$(date +%Y%m%dT%H%M%S).log" ; \
	nohup $(PY) -m simulator.cli run \
		--cohort $(COHORT) \
		--config $(CONFIG) \
		$(if $(ENGINE),--engine $(ENGINE)) \
		$(if $(MODEL),--model $(MODEL)) \
		>"$$LOG" 2>&1 & \
	PID=$$! ; \
	echo "Cohort run started in background (PID $$PID)" ; \
	echo "  run dir: $$RD" ; \
	echo "  log: $$LOG" ; \
	echo "  tail: tail -f $$LOG" ; \
	echo "  dashboard: make dashboard" ; \
	echo "  stop: make stop-bg  (or: kill $$PID)"

.PHONY: run-persona-bg
run-persona-bg:
	@RD="$$($(RESOLVE_RUN_DIR))" ; \
	LOG="$$RD/persona_$(PERSONA)_$$(date +%Y%m%dT%H%M%S).log" ; \
	nohup $(PY) -m simulator.cli run-persona \
		--persona $(PERSONA) \
		--config $(CONFIG) \
		$(if $(ENGINE),--engine $(ENGINE)) \
		$(if $(MODEL),--model $(MODEL)) \
		>"$$LOG" 2>&1 & \
	PID=$$! ; \
	echo "Persona run started in background (PID $$PID)" ; \
	echo "  run dir: $$RD" ; \
	echo "  log: $$LOG" ; \
	echo "  tail: tail -f $$LOG" ; \
	echo "  dashboard: make dashboard" ; \
	echo "  stop: make stop-bg  (or: kill $$PID)"

.PHONY: run-sweep-bg
run-sweep-bg:
	@RD="$$($(RESOLVE_RUN_DIR))" ; \
	LOG="$$RD/sweep_$$(date +%Y%m%dT%H%M%S).log" ; \
	nohup $(PY) -m simulator.cli sweep \
		--config $(CONFIG) \
		--type $(SWEEP_TYPE) \
		$(if $(RUN_NEW),--new-run) \
		$(if $(ENGINE),--engine $(ENGINE)) \
		$(if $(MODEL),--model $(MODEL)) \
		>"$$LOG" 2>&1 & \
	PID=$$! ; \
	echo "Sweep started in background (PID $$PID)" ; \
	echo "  run dir: $$RD$(if $(RUN_NEW), (fresh — RUN_NEW=true))" ; \
	echo "  log: $$LOG" ; \
	echo "  tail: tail -f $$LOG" ; \
	echo "  dashboard: make dashboard" ; \
	echo "  stop: make stop-bg  (or: kill $$PID)"

# Tail the most-recent background log file in the latest run_NN/.
# Useful when reconnecting to a host where you started a -bg run earlier.
.PHONY: tail-log
tail-log:
	@RD=$$(ls -d $(RUN_DIR)/run_* 2>/dev/null | sort | tail -n 1) ; \
	if [ -z "$$RD" ]; then echo "No run_NN/ dirs in $(RUN_DIR)"; exit 1; fi ; \
	LOG=$$(ls -t "$$RD"/sweep_*.log "$$RD"/cohort_*.log "$$RD"/persona_*.log 2>/dev/null | head -n 1) ; \
	if [ -z "$$LOG" ]; then echo "No background-run logs in $$RD"; exit 1; fi ; \
	echo "Tailing $$LOG" ; \
	tail -f "$$LOG"

# Kill any running simulator background process and its engine
# containers. Confirms the kill before doing anything destructive.
.PHONY: stop-bg
stop-bg:
	@PIDS=$$(pgrep -f "simulator.cli (run|run-persona|sweep)" 2>/dev/null) ; \
	if [ -n "$$PIDS" ]; then \
		echo "Killing simulator processes: $$PIDS" ; \
		kill $$PIDS 2>/dev/null || true ; \
		sleep 2 ; \
		PIDS=$$(pgrep -f "simulator.cli (run|run-persona|sweep)" 2>/dev/null) ; \
		if [ -n "$$PIDS" ]; then \
			echo "Forcing kill: $$PIDS" ; kill -9 $$PIDS 2>/dev/null || true ; \
		fi ; \
	else \
		echo "No simulator processes running." ; \
	fi
	@CIDS=$$(docker ps -q --filter name=vllm-r --filter name=sglang- 2>/dev/null) ; \
	if [ -n "$$CIDS" ]; then \
		echo "Stopping engine containers..." ; \
		docker stop -t 30 $$CIDS >/dev/null ; \
		echo "Stopped: $$CIDS" ; \
	else \
		echo "No engine containers running." ; \
	fi

.PHONY: list-cohorts
list-cohorts:
	$(PY) -m simulator.cli list-cohorts

.PHONY: list-personas
list-personas:
	$(PY) -m simulator.cli list-personas

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
	@RD=$$(ls -d $(RUN_DIR)/run_* 2>/dev/null | sort | tail -n 1) ; \
	[ -n "$$RD" ] || RD="$(RUN_DIR)" ; \
	DB=$$(ls -t "$$RD"/*.db 2>/dev/null | head -n 1); \
	if [ -z "$$DB" ]; then echo "No .db in $$RD"; exit 1; fi; \
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
	rm -rf $(RUN_DIR)/run_* $(RUN_DIR)/*.db $(RUN_DIR)/*.db-journal buyer_page_data.json

.PHONY: list-runs
list-runs:
	@if [ ! -d "$(RUN_DIR)" ]; then echo "$(RUN_DIR) does not exist"; exit 0; fi ; \
	for d in $$(ls -d $(RUN_DIR)/run_* 2>/dev/null | sort); do \
		mtime=$$(stat -f '%Sm' -t '%Y-%m-%d %H:%M' "$$d" 2>/dev/null || stat -c '%y' "$$d" | cut -d. -f1) ; \
		db="$$d/run.db" ; \
		if [ -f "$$db" ]; then \
			cohorts=$$(sqlite3 "$$db" "SELECT COUNT(*) FROM cohort_run" 2>/dev/null || echo "?") ; \
			ok=$$(sqlite3 "$$db" "SELECT COUNT(*) FROM cohort_run WHERE final_status='ok'" 2>/dev/null || echo "?") ; \
			printf "  %-12s  %s  (%s cohort_run rows, %s ok)\n" "$$(basename $$d)" "$$mtime" "$$cohorts" "$$ok" ; \
		else \
			printf "  %-12s  %s  (no run.db yet)\n" "$$(basename $$d)" "$$mtime" ; \
		fi ; \
	done

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
