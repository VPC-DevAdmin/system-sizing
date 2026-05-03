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

SGLang's mainline pip wheel ships GPU-only `sgl_kernel` binaries; the working CPU path is the upstream `sglang-cpu` Docker image. Build the local layer once:

```bash
docker pull lmsysorg/sglang:latest-cpu
docker tag  lmsysorg/sglang:latest-cpu sglang-cpu:xeon
docker build -f Dockerfile.xeon-fixed -t sglang-cpu:xeon-fixed .
```

The `-fixed` layer adds `sentencepiece`, `tiktoken`, and `protobuf` — modern HF tokenizers (Qwen3, GLM, Mistral, Llama 3) won't load without them.

Stage the model on the host at `/data/ml/models/<model-dir>` and reference it as `model_local_path: /models/<model-dir>` in the engine config. Hugging Face cache mounts at `/data/ml/huggingface`.

Known-good parallelism shapes on SGLang CPU: **TP=1 baseline** and **TP=4**. Anything else (DP/EP combos) is upstream-broken at time of writing.

The simulator derives `SGLANG_CPU_OMP_THREADS_BIND` from `engine.cpu_bind` and validates it aligns with `tensor_parallel_size`; misconfigurations surface at launch instead of 20 minutes into a model load.
