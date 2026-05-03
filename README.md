# Persona Capacity Simulator

Drives a single, long-running LLM inference engine (vLLM or SGLang) with realistic persona-based workloads to find the per-cohort capacity knee, with telemetry-attributed bottleneck analysis.

## Quick start

```bash
make setup
make run-cohort  ENGINE=vllm  MODEL=Qwen/Qwen2.5-7B-Instruct  COHORT=chat_heavy
make dashboard                                      # in another terminal
make run-sweep   ENGINE=vllm  MODEL=Qwen/Qwen2.5-7B-Instruct
make export                                         # buyer_page_data.json
```

## Make targets

| Target | What it does |
|---|---|
| `make setup` | Install package + deps |
| `make launch-engine` | Launch engine subprocess only (manual testing) |
| `make run-cohort` | Run a single cohort end-to-end |
| `make run-sweep` | Run all cohorts back-to-back against one engine |
| `make dashboard` | Live `rich`-based progress view of the latest run |
| `make export` | Build `buyer_page_data.json` from `runs/*.db` |
| `make web` | Serve the reference buyer page on `http://localhost:8765` |
| `make analyze-prefix-cache` | Prefix-cache hit-rate report on the latest `.db` |
| `make test` | Run pytest |
| `make clean` / `clean-runs` | Tidy caches / wipe run databases |

Variables: `ENGINE` (vllm|sglang), `MODEL`, `COHORT`, `CONFIG`, `RUN_DIR`.

## Layout

```
simulator/
  cli.py              # typer entry point used by Make targets
  runner.py           # cohort run orchestration
  config.py           # YAML + CLI config
  personas.py         # 6 personas, 5 cohorts
  distributions.py    # LogNormal / Discrete samplers
  tokenizer_corpus.py # filler text targeted at token counts
  engines/
    base.py           # subprocess + Prometheus metric parser
    vllm.py           # Xeon CPU tuning recipe
    sglang.py         # CPU launcher
  virtual_user.py     # one async task per simulated user
  pool_manager.py     # spawn/replace virtual users at target pool size
  adaptive.py         # next-pool-size selection logic
  measurement.py      # ramp -> stabilize -> measure
  telemetry.py        # snapshots, perf-stat PMU, engine metrics
  database.py         # SQLite schema + capture
  dashboard.py        # rich live view
  export.py           # buyer-page JSON
  prefix_cache.py     # post-hoc prefix-cache hit-rate analysis
  bandwidth.py        # IMC-uncore memory bandwidth (per-controller on GNR)
  perf_collector.py   # PMU events with AMX raw fallback for GNR
  power_probe.py      # RAPL package power
  frequency.py        # bound-CPU effective frequency, three-tier read
  amx_utilization.py  # oneDNN verbose log -> AMX dispatch fraction
  cpu_binding.py      # parse VLLM_CPU_OMP_THREADS_BIND
web/
  index.html          # self-contained Chart.js renderer for buyer_page_data.json
tests/                # pytest suites
config/default.yaml
Makefile
```

Each cohort run produces one SQLite file in `runs/`.

## SGLang on CPU (Docker required)

SGLang's mainline pip wheel ships GPU-only `sgl_kernel` binaries — importing the package on a CPU-only host fails before the launcher sees its arguments. The working CPU path is the upstream `sglang-cpu` Docker image, layered with `sentencepiece` / `tiktoken` / `protobuf` so modern HF tokenizers (Qwen3, GLM, Mistral, Llama 3) load.

There's no published Docker Hub tag for the CPU build; build from SGLang source.

### One-time: build the images

```bash
# Base — ~15-20 min. Uses uv as package manager; venv lands at /opt/.venv.
git clone --depth 1 https://github.com/sgl-project/sglang.git /tmp/sglang
docker build -f /tmp/sglang/docker/xeon.Dockerfile -t sglang-cpu:xeon /tmp/sglang

# Layered fix — ~1-2 min. Bootstraps pip via ensurepip, installs the
# three tokenizer deps the base image lacks.
docker build -f Dockerfile.xeon-fixed -t sglang-cpu:xeon-fixed .

# Verify
docker run --rm sglang-cpu:xeon-fixed /opt/.venv/bin/python -c \
  "import sglang, sgl_kernel, sentencepiece, tiktoken; \
   from google import protobuf; \
   print('OK', sglang.__version__)"
```

### One-time: stage the model

```bash
sudo mkdir -p /data/ml/models /data/ml/huggingface
sudo chown -R $USER:$USER /data/ml

cd /data/ml/models
hf download Qwen/Qwen3-30B-A3B-Instruct-2507 \
    --local-dir Qwen3-30B-A3B-Instruct-2507
```

The `-2507` suffix is part of the actual published HF repo name, not a separate version tag.

For long downloads, run inside `tmux` so an SSH disconnect doesn't kill the transfer. If only the safetensors finish but tokenizer files are missing, re-run the same command with `--include "tokenizer*" "*.json"`. Note: `protobuf` installs as `protobuf` but imports as `google.protobuf` — `import protobuf` will fail even though the install is fine.

### Run

```bash
make launch-engine ENGINE=sglang \
                   MODEL=Qwen/Qwen3-30B-A3B-Instruct-2507 \
                   CONFIG=config/r7735_sglang_qwen3_30b_a3b.yaml
```

The container streams its stdout/stderr to `runs/engine_sglang_*.log`. Wait for the simulator's `SGLang ready after Xs` message — this only fires once `/v1/models` returns 200, which means the model is fully loaded. Expected times:

| Variant | Resident | Cold load |
|---|---|---|
| BF16 (`Qwen3-30B-A3B-Instruct-2507`) | ~58 GB | ~2-3 min |
| FP8 (`Qwen3-30B-A3B-Instruct-2507-FP8`) | ~32 GB | ~1-2 min |

### Known-good parallelism shapes

**TP=1** baseline and **TP=4** only. DP/EP combos are upstream-broken on CPU. The simulator's `derive_sglang_thread_binding` validates `tensor_parallel_size` against `engine.cpu_bind` at launch — misconfigurations surface immediately instead of 20 minutes into a model load.
