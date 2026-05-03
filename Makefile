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

# SGLang build/download knobs — override when needed.
SGLANG_REPO         ?= https://github.com/sgl-project/sglang.git
SGLANG_SRC          ?= /tmp/sglang
SGLANG_BASE_IMAGE   ?= sglang-cpu:xeon
SGLANG_FIXED_IMAGE  ?= sglang-cpu:xeon-fixed
SGLANG_DOCKERFILE   ?= $(SGLANG_SRC)/docker/xeon.Dockerfile

MODELS_DIR          ?= /data/ml/models
HF_CACHE_DIR        ?= /data/ml/huggingface
# When MODEL is a HF id like "Qwen/Qwen3-30B-A3B-Instruct-2507", the
# local dir is just the basename. Override LOCAL_MODEL_DIR if you want
# a different folder name.
LOCAL_MODEL_DIR     ?= $(notdir $(MODEL))

.DEFAULT_GOAL := help

.PHONY: help
help:
	@echo "Persona Capacity Simulator — targets:"
	@echo ""
	@echo "  make setup                   Install package and dependencies"
	@echo "  make preflight               Validate host satisfies CONFIG's hardware_requirements"
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
	@echo "SGLang Docker pipeline:"
	@echo "  make sglang-setup            Clone source, build both images, verify"
	@echo "  make sglang-clone            Clone / update SGLang source ($(SGLANG_SRC))"
	@echo "  make sglang-base             Build $(SGLANG_BASE_IMAGE) (~15-20 min first run)"
	@echo "  make sglang-fixed            Build $(SGLANG_FIXED_IMAGE) (depends on -base)"
	@echo "  make sglang-verify           Run import smoke test inside the fixed image"
	@echo "  make sglang-shell            Interactive shell inside the fixed image"
	@echo ""
	@echo "Model download:"
	@echo "  make download-model MODEL=...  hf download to $(MODELS_DIR)/<basename>"
	@echo "  make models-dirs               Create $(MODELS_DIR) + $(HF_CACHE_DIR) (sudo)"
	@echo ""
	@echo "Variables: ENGINE=$(ENGINE) MODEL=$(MODEL) COHORT=$(COHORT)"

.PHONY: setup
setup:
	$(PY) -m pip install --upgrade pip
	$(PY) -m pip install -e .

.PHONY: preflight
preflight:
	$(PY) -m simulator.cli preflight --config $(CONFIG)

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

# ── SGLang Docker pipeline ────────────────────────────────────────────

# Clone or update the upstream SGLang repo (needed for the CPU Dockerfile).
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
		(echo "ERROR: $(SGLANG_DOCKERFILE) not found. Inspect $(SGLANG_SRC)/docker/" && exit 1)

.PHONY: sglang-base
sglang-base: sglang-clone
	@echo "==> Building $(SGLANG_BASE_IMAGE) from $(SGLANG_DOCKERFILE)"
	docker build -f "$(SGLANG_DOCKERFILE)" -t "$(SGLANG_BASE_IMAGE)" "$(SGLANG_SRC)"

# Layer the tokenizer-deps fix on top.
.PHONY: sglang-fixed
sglang-fixed: sglang-base Dockerfile.xeon-fixed
	@echo "==> Building $(SGLANG_FIXED_IMAGE)"
	docker build -f Dockerfile.xeon-fixed -t "$(SGLANG_FIXED_IMAGE)" .

.PHONY: sglang-verify
sglang-verify:
	@echo "==> Verifying $(SGLANG_FIXED_IMAGE)"
	docker run --rm "$(SGLANG_FIXED_IMAGE)" /opt/.venv/bin/python -c \
		"import sglang, sgl_kernel, sentencepiece, tiktoken; \
		from google import protobuf; \
		print('OK sglang', sglang.__version__, '| sentencepiece', sentencepiece.__version__, '| tiktoken', tiktoken.__version__, '| protobuf', protobuf.__version__)"

.PHONY: sglang-build
sglang-build: sglang-fixed sglang-verify

# Drop into an interactive shell in the fixed image with the model dir
# mounted — useful for debugging tokenizer / config issues.
.PHONY: sglang-shell
sglang-shell:
	docker run --rm -it \
		-v "$(MODELS_DIR):/models" \
		-v "$(HF_CACHE_DIR):/root/.cache/huggingface" \
		--entrypoint /bin/bash \
		"$(SGLANG_FIXED_IMAGE)"

# ── Model staging ─────────────────────────────────────────────────────

.PHONY: models-dirs
models-dirs:
	@if [ ! -d "$(MODELS_DIR)" ] || [ ! -w "$(MODELS_DIR)" ]; then \
		echo "==> Creating $(MODELS_DIR) and $(HF_CACHE_DIR) (sudo)"; \
		sudo mkdir -p "$(MODELS_DIR)" "$(HF_CACHE_DIR)"; \
		sudo chown -R $$USER:$$USER "$(MODELS_DIR)" "$(HF_CACHE_DIR)"; \
	fi

.PHONY: download-model
download-model: models-dirs
	@command -v hf >/dev/null 2>&1 || \
		(echo "ERROR: hf CLI not found. pip install huggingface_hub[cli,hf_transfer]" && exit 1)
	@echo "==> Downloading $(MODEL) -> $(MODELS_DIR)/$(LOCAL_MODEL_DIR)"
	HF_HUB_ENABLE_HF_TRANSFER=1 hf download "$(MODEL)" \
		--local-dir "$(MODELS_DIR)/$(LOCAL_MODEL_DIR)"
	@echo "==> Sanity check"
	@ls "$(MODELS_DIR)/$(LOCAL_MODEL_DIR)/"*.safetensors 2>/dev/null | wc -l \
		| awk '{ print "  safetensors shards: " $$1 }'
	@for f in tokenizer.json tokenizer_config.json config.json; do \
		test -f "$(MODELS_DIR)/$(LOCAL_MODEL_DIR)/$$f" \
			&& echo "  $$f: present" \
			|| echo "  $$f: MISSING (re-run with --include 'tokenizer*' '*.json' if so)"; \
	done

# Composite: clone, build, verify, download. Default model is the one
# r7735_sglang_qwen3_30b_a3b.yaml expects.
.PHONY: sglang-setup
sglang-setup: MODEL ?= Qwen/Qwen3-30B-A3B-Instruct-2507
sglang-setup: sglang-build download-model
	@echo ""
	@echo "==> SGLang setup complete."
	@echo "    Image: $(SGLANG_FIXED_IMAGE)"
	@echo "    Model: $(MODELS_DIR)/$(LOCAL_MODEL_DIR)"
	@echo ""
	@echo "Next:"
	@echo "  make launch-engine ENGINE=sglang \\"
	@echo "                     MODEL=$(MODEL) \\"
	@echo "                     CONFIG=config/r7735_sglang_qwen3_30b_a3b.yaml"
