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
# Stepper mode. Default is the fixed-grid sweep (powers of 2 from 4
# to 256 with early-stop one step past first failure) — gives uniform
# x-axis density for downstream capacity curves. Set ADAPTIVE=true to
# opt into the two-knee adaptive stepper (Wilson-CI bisection) when
# you care more about precise knee placement than uniform sampling.
ADAPTIVE ?=
# POOL_SIZES=8,16,32,... overrides the default powers-of-2 grid in
# fixed-grid mode. Mutually exclusive with ADAPTIVE=true. Empty =
# use the default grid.
POOL_SIZES ?=
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
	@echo "  make run-sweep   CONFIG=... [SWEEP_TYPE=] [POOL_SIZES=] [ADAPTIVE=true]"
	@echo "                                          Sweep multiple workloads."
	@echo "                                          SWEEP_TYPE=all|personas|cohorts|a,b,c"
	@echo "                                          Default: fixed-grid sweep (4,8,16,32,64,128,256)"
	@echo "                                          early-stops one step past first failure."
	@echo "                                          POOL_SIZES=... overrides the default grid."
	@echo "                                          ADAPTIVE=true → two-knee adaptive stepper."
	@echo "                                          Always nohup'd + auto-tailed; survives SSH disconnect."
	@echo "                                          Ctrl-C exits the tail (sweep keeps running)."
	@echo "                                          Resumes latest run_NN by default;"
	@echo "                                          set RUN_NEW=true to cut a fresh run dir."
	@echo "  make run-cohort-bg / run-persona-bg     Single-workload nohup variants (-bg)"
	@echo "  make tail-log / make stop-bg            Reattach to the latest run / stop a running sweep + engine"
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
	@echo "Tuning:"
	@echo "  make optimize-engine [ONLY=...] [RUN_NEW=true]"
	@echo "                                          A/B vLLM launch shapes, pick the best for this host."
	@echo "                                          Always nohup'd + auto-tailed; resumes existing"
	@echo "                                          runs/engine_optimizer/run.json by default."
	@echo "  make optimize-dashboard                 Read-only dashboard against the running optimizer"
	@echo "                                          (use from a second SSH session)."
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
		$(if $(MODEL),--model $(MODEL)) \
		$(if $(filter true,$(ADAPTIVE)),--adaptive) \
		$(if $(POOL_SIZES),--pool-sizes $(POOL_SIZES))

.PHONY: run-persona
run-persona:
	$(PY) -m simulator.cli run-persona \
		--persona $(PERSONA) \
		--config $(CONFIG) \
		$(if $(ENGINE),--engine $(ENGINE)) \
		$(if $(MODEL),--model $(MODEL)) \
		$(if $(filter true,$(ADAPTIVE)),--adaptive) \
		$(if $(POOL_SIZES),--pool-sizes $(POOL_SIZES))

# Sweeps are always nohup'd + log-teed + auto-tailed. SSH disconnect
# leaves the simulator (and its docker engine containers) running;
# Ctrl-C exits the tail without killing the sweep. Reattach later
# with ``make tail-log``; stop the run with ``make stop-bg``.
#
# Why backgrounded with ``&`` works as "Ctrl-C safe": in non-interactive
# shells (which is what make uses), backgrounded processes get SIGINT
# explicitly ignored. nohup adds SIGHUP-ignore on top, so terminal
# disconnect can't kill it either. ``</dev/null`` belt-and-braces so
# the engine subprocess can never block waiting on stdin.
#
# IMPORTANT: do NOT pass --new-run to the Python sweep here.
# RESOLVE_RUN_DIR (above) already creates the fresh run_NN+1 when
# RUN_NEW=true; the Python sweep's resolve_run_dir() with new=False
# picks up the freshly-created (latest) dir via latest_run_dir().
# Passing --new-run a second time would create yet another run dir,
# splitting the sweep's DBs and the wrapper's log file across two
# directories. Resume-skip is harmless on a fresh dir (find_completed_runs
# returns empty) so RUN_NEW=true semantics are preserved.
# Depend on ``stop-bg`` so a stale sweep + engine container from a
# previous invocation are cleaned up before we launch a new one.
# Symptom this guards against: Ctrl-C on the tail leaves the
# nohup-detached sweep alive; a follow-up ``make run-sweep`` then
# silently shares the still-listening engine container with the
# orphan sweep, halving each one's effective throughput and producing
# garbage data. ``stop-bg`` is a no-op when nothing is running.
.PHONY: run-sweep
run-sweep: stop-bg
	@RD="$$($(RESOLVE_RUN_DIR))" ; \
	LOG="$$RD/sweep_$$(date +%Y%m%dT%H%M%S).log" ; \
	nohup $(PY) -m simulator.cli sweep \
		--config $(CONFIG) \
		--type $(SWEEP_TYPE) \
		$(if $(ENGINE),--engine $(ENGINE)) \
		$(if $(MODEL),--model $(MODEL)) \
		$(if $(filter true,$(ADAPTIVE)),--adaptive) \
		$(if $(POOL_SIZES),--pool-sizes $(POOL_SIZES)) \
		>"$$LOG" 2>&1 </dev/null & \
	PID=$$! ; \
	echo "" ; \
	echo "Sweep started in background (PID $$PID) — survives SSH disconnect" ; \
	echo "  run dir: $$RD$(if $(RUN_NEW), (fresh — RUN_NEW=true))" ; \
	echo "  log:     $$LOG" ; \
	echo "" ; \
	echo "Following live (Ctrl-C exits the tail; the sweep keeps running)." ; \
	echo "  reattach later: make tail-log" ; \
	echo "  stop the sweep: make stop-bg" ; \
	echo "" ; \
	for i in 1 2 3 4 5 ; do [ -f "$$LOG" ] && break ; sleep 0.2 ; done ; \
	tail -f "$$LOG" & \
	TAIL=$$! ; \
	trap 'kill $$TAIL 2>/dev/null ; exit 0' INT TERM ; \
	while kill -0 $$PID 2>/dev/null ; do sleep 2 ; done ; \
	wait $$PID 2>/dev/null ; SWEEP_EXIT=$$? ; \
	sleep 1 ; \
	kill $$TAIL 2>/dev/null ; \
	echo "" ; \
	echo "=== Sweep finished (PID $$PID, exit $$SWEEP_EXIT) — see $$LOG ===" ; \
	exit $$SWEEP_EXIT

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

# Audit a finished run for curve-quality anomalies (no-marginal-band,
# no-fail-observed, single-point-rescue, boundary-status). Writes a
# JSON plan to <run-dir>/audit_report.json that ``make spot-check``
# reads to re-measure the specific points needed to shore up the data.
.PHONY: audit
audit:
	@RD=$$(ls -d $(RUN_DIR)/run_* 2>/dev/null | sort | tail -n 1) ; \
	[ -n "$$RD" ] || { echo "No run_NN/ in $(RUN_DIR)"; exit 1; } ; \
	$(PY) scripts/audit_run.py "$$RD"

# Re-measure the (cohort, pool_size) points an ``audit`` flagged.
# Each point is APPENDED to its existing cohort_run row, so re-running
# ``make export`` afterward picks up the enriched curve. Depends on
# ``stop-bg`` to ensure no other engine/sweep is consuming the host.
.PHONY: spot-check
spot-check: stop-bg
	@RD=$$(ls -d $(RUN_DIR)/run_* 2>/dev/null | sort | tail -n 1) ; \
	[ -n "$$RD" ] || { echo "No run_NN/ in $(RUN_DIR)"; exit 1; } ; \
	PLAN="$$RD/audit_report.json" ; \
	[ -f "$$PLAN" ] || { echo "No $$PLAN — run ``make audit`` first"; exit 1; } ; \
	LOG="$$RD/spot_check_$$(date +%Y%m%dT%H%M%S).log" ; \
	nohup $(PY) -m simulator.cli spot-check \
		--plan "$$PLAN" \
		--run-dir "$$RD" \
		>"$$LOG" 2>&1 </dev/null & \
	PID=$$! ; \
	echo "" ; \
	echo "Spot-check started in background (PID $$PID)" ; \
	echo "  run dir: $$RD" ; \
	echo "  log:     $$LOG" ; \
	echo "  tail:    make tail-log" ; \
	echo "  stop:    make stop-bg" ; \
	echo "" ; \
	for i in 1 2 3 4 5 ; do [ -f "$$LOG" ] && break ; sleep 0.2 ; done ; \
	tail -f "$$LOG" & \
	TAIL=$$! ; \
	trap 'kill $$TAIL 2>/dev/null ; exit 0' INT TERM ; \
	while kill -0 $$PID 2>/dev/null ; do sleep 2 ; done ; \
	wait $$PID 2>/dev/null ; SC_EXIT=$$? ; \
	sleep 1 ; \
	kill $$TAIL 2>/dev/null ; \
	echo "" ; \
	echo "=== Spot-check finished (PID $$PID, exit $$SC_EXIT) — see $$LOG ===" ; \
	exit $$SC_EXIT

.PHONY: list-cohorts
list-cohorts:
	$(PY) -m simulator.cli list-cohorts

.PHONY: list-personas
list-personas:
	$(PY) -m simulator.cli list-personas

# ── Output / analysis ─────────────────────────────────────────────────

# Build buyer_page_data.json from runs/run_NN/run.db.
#
# Pass SLIM=true to produce the summary-only buyer_page_data_slim.json
# (~99% smaller — drops per-step telemetry samples + turn events +
# the cohort-level 1 Hz heartbeat). Useful for buyer-facing summary
# distribution; the full export stays available for diagnostics.
.PHONY: export
export:
	$(PY) -m simulator.cli export --input-dir $(RUN_DIR) \
		$(if $(SLIM),--slim)

.PHONY: web
web: export
	@RD=$$(ls -d $(RUN_DIR)/run_* 2>/dev/null | sort | tail -n 1) ; \
	[ -n "$$RD" ] || RD="$(RUN_DIR)" ; \
	JSON="$$RD/buyer_page_data.json" ; \
	if [ ! -f "$$JSON" ]; then echo "No $$JSON yet — did make export succeed?" ; exit 1 ; fi ; \
	cp -f "$$JSON" web/buyer_page_data.json ; \
	echo "Reference buyer page at http://localhost:8765/  (source: $$JSON)"
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

# Engine A/B harness: iterates docker launch shapes against the
# vllm-openai-cpu image, measures TTFT/TPOT/throughput per config,
# writes runs/engine_optimizer/run.json. Always nohup'd + auto-tailed
# (60-90 min total runtime; SSH disconnect must not kill it). Resumes
# the existing JSON by default — to start fresh, RUN_NEW=true.
#
# Knobs:
#   ONLY=baseline,kv_xl   restrict to a subset of configs
#   RUN_NEW=true          wipe runs/engine_optimizer/run.json first
#   LIST=1                print the registered configs and exit
#
# Same Ctrl-C semantics as run-sweep: tail exits, optimizer keeps
# running. Reattach with ``tail -f runs/engine_optimizer/optimizer_*.log``.
.PHONY: optimize-engine
optimize-engine:
	@if [ -n "$(LIST)" ]; then \
		$(PY) scripts/engine_optimizer.py --list \
			$(if $(PROFILE),--profile $(PROFILE)) ; \
		exit 0 ; \
	fi ; \
	OUT_DIR="runs/engine_optimizer" ; \
	OUT="$$OUT_DIR/run.json" ; \
	LOG="$$OUT_DIR/optimizer_$$(date +%Y%m%dT%H%M%S).log" ; \
	mkdir -p "$$OUT_DIR" ; \
	nohup $(PY) scripts/engine_optimizer.py \
		--out "$$OUT" \
		$(if $(RUN_NEW),--new-run) \
		$(if $(ONLY),--only $$(echo $(ONLY) | tr ',' ' ')) \
		$(if $(PROFILE),--profile $(PROFILE)) \
		>"$$LOG" 2>&1 </dev/null & \
	PID=$$! ; \
	echo "" ; \
	echo "Engine optimizer started in background (PID $$PID) — survives SSH disconnect" ; \
	echo "  json:    $$OUT$(if $(RUN_NEW), (fresh — RUN_NEW=true))" ; \
	echo "  log:     $$LOG" ; \
	echo "" ; \
	echo "Following live (Ctrl-C exits the tail; the optimizer keeps running)." ; \
	echo "  reattach later: tail -f $$LOG" ; \
	echo "  stop optimizer: pkill -f scripts/engine_optimizer.py" ; \
	echo "" ; \
	for i in 1 2 3 4 5 ; do [ -f "$$LOG" ] && break ; sleep 0.2 ; done ; \
	tail -f "$$LOG" & \
	TAIL=$$! ; \
	trap 'kill $$TAIL 2>/dev/null ; exit 0' INT TERM ; \
	while kill -0 $$PID 2>/dev/null ; do sleep 2 ; done ; \
	wait $$PID 2>/dev/null ; OPT_EXIT=$$? ; \
	sleep 1 ; \
	kill $$TAIL 2>/dev/null ; \
	echo "" ; \
	echo "=== Optimizer finished (PID $$PID, exit $$OPT_EXIT) — see $$OUT ===" ; \
	exit $$OPT_EXIT

# Read-only dashboard against a running (or completed) optimizer.
# Polls runs/engine_optimizer/run.json + the latest optimizer_*.log so
# you can watch progress from a second SSH session without touching
# the backgrounded optimizer process. Ctrl-C exits the dashboard.
.PHONY: optimize-dashboard
optimize-dashboard:
	$(PY) scripts/engine_optimizer.py --watch \
		--out runs/engine_optimizer/run.json

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
