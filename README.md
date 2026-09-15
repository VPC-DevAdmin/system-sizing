# capsim — Persona Capacity Simulator

Drives a single, long-running LLM inference engine (vLLM or SGLang) with realistic persona-based workloads to find the per-cohort capacity knee, with telemetry-attributed bottleneck analysis. Roadmap to a general AI sizing & capacity engine with its own UI: [docs/roadmap.md](docs/roadmap.md).

## Landing on a fresh host

Three commands from bare box to verified pipeline — prerequisites are just Python 3.10+, Docker, and git:

```bash
git clone <repo> && cd system-sizing && ./install.sh   # installs uv + capsim (isolated, no sudo)
capsim doctor                                          # validate host: CPU/GPU/docker/disk/telemetry perms/HF
capsim smoke --profile <recommended>                   # ~10 min end-to-end proof with a tiny model
```

`capsim doctor` prints a pass/warn/fail table (and `doctor.json` for scripting), including GPU stack checks (nvidia-smi + container toolkit) on Xeon+NVIDIA boxes, and recommends candidate **profiles** for the detected hardware. `capsim smoke` launches the real engine with a ~1.5 GB stand-in model, runs 2 virtual users through a short measured window, exports, and validates the export against the schema contract — if smoke passes, the whole pipeline works on this host. Then commit to the full model:

```bash
capsim ready --profile <recommended>                   # full model download + image build/pull
```

`capsim` and `python -m simulator.cli` are the same CLI (the `simulator` entry point remains as a deprecated alias); the Make targets below wrap it.

## Targets and profiles

The same persona/knee methodology runs against three target kinds, selected by `engine.type` in the config:

- **CPU local-docker** (`vllm`, `sglang`, `vllm_dual_socket`) — the original path, full host telemetry (PMU, IMC bandwidth, RAPL power, frequency, AMX).
- **GPU local-docker** (`vllm_cuda`) — upstream `vllm/vllm-openai` CUDA image with `--gpus`, for Xeon+NVIDIA hosts. The GPU collector (NVML, nvidia-smi fallback) samples SM util / VRAM / power / clocks / throttle state at 1 Hz alongside the host CPU collectors, and bottleneck attribution gains `gpu_compute` and `gpu_throttled` labels.
- **Remote endpoint** (`remote`) — benchmark an OpenAI-compatible endpoint you don't own; host telemetry is skipped (it would measure the client box, and the export records that), the endpoint's `/metrics` is scraped when configured. See the [remote-endpoint template](config/profiles/remote-endpoint.yaml).

There's also a **mock target** (`--profile mock`) — a real in-process OpenAI-compatible SSE server with a synthetic, tunable capacity knee. The whole pipeline (virtual users, measurement, telemetry, DB, export) runs on a laptop with zero hardware; it powers CI integration tests and UI development.

A **profile** is a curated named config for one host class. `capsim list-profiles` shows what's available; every command accepts `--profile <name>` in place of `--config <path>`. Curated profiles live in [config/profiles/](config/profiles/) (e.g. `xeon-gpu-qwen3-30b`); the pre-existing `config/*.yaml` files are addressable by stem too. The export's per-cohort `collectors` block records which telemetry sources actually ran, so downstream consumers know what evidence backs each bottleneck claim.

## Quick start

```bash
# Two commands to first measurement.
make ready CONFIG=config/r7735_sglang_qwen3_30b_a3b.yaml
make run-cohort CONFIG=config/r7735_sglang_qwen3_30b_a3b.yaml \
                COHORT=chat_heavy
# In a second terminal:
make dashboard

# After the run finishes:
make export
capsim serve                                        # web UI: http://localhost:8321
```

## Make targets

The headline workflow is `ready` → `run-cohort` → `dashboard` → `export`. Everything else is either a downstream analysis step or a diagnostic.

| Target | What it does |
|---|---|
| `make ready CONFIG=...` | Idempotent: pip install, build engine docker image (SGLang only) if missing, download model if missing, validate hardware. |
| `make doctor` | Full host validation — CPU/NUMA/docker/GPU stack/disk/telemetry permissions/HF reachability as a pass/warn/fail table plus `doctor.json`. Run first on any fresh box. |
| `make smoke CONFIG=...` | End-to-end pipeline proof: real engine + tiny model (~1.5 GB), 2 virtual users, short measured window, export validated against the schema contract. ~10 min on a fresh box. |
| `capsim serve` | Control-plane service + web UI on `127.0.0.1:8321`: run lifecycle over HTTP (`/api/runs`, `/api/status`, `/api/export`, `/api/doctor`, catalogs) and live telemetry over WebSocket (`/ws/telemetry` — run/snapshot/turn/step events). One run at a time, in-process; `run.db` stays the source of truth. The UI (no build step, Chart.js vendored so offline benchmark boxes work) has three views: **Run control** (profile + workload pickers, doctor, run history), **Live telemetry** (pool/in-flight, rolling TTFT/TPOT percentiles, KV/CPU/GPU, step progress), and **Results** (landing zones, knee chart with CI band, bottleneck evidence, step drill-down, cross-run comparison, export download). |
| `make run-persona CONFIG=... PERSONA=...` | Run one **persona** (a single user archetype) end-to-end. |
| `make run-cohort CONFIG=... COHORT=...` | Run one **cohort** (a team mix of personas) end-to-end. |
| `make run-sweep CONFIG=... [SWEEP_TYPE=...] [RUN_NEW=true]` | Sweep multiple workloads. **Always nohup'd, log-teed, and auto-tailed** — the sweep + its engine containers survive SSH disconnect; Ctrl-C only exits the tail. The terminal prints the run dir + log path on launch and starts following the log live; reattach later with `make tail-log`, stop with `make stop-bg`. `SWEEP_TYPE` accepts `all` (default — every persona + every cohort), `personas`, `cohorts`, or a comma-separated list of persona/cohort ids. **Resumes the latest `runs/run_NN/` by default** — workloads with `final_status='ok'` are skipped, so an interrupted sweep auto-continues. Pass `RUN_NEW=true` to cut a fresh `run_NN+1` dir (use this when config or hardware has changed and the prior run's data should NOT be merged with the new one). |
| `make run-cohort-bg ...` / `make run-persona-bg ...` | Background single-workload variants for cohorts and personas. Same nohup pattern; non-blocking (no auto-tail). |
| `make tail-log` | Tail the most-recent background-run log (auto-picks the latest). |
| `make stop-bg` | Kill any running simulator background process and its engine containers. |
| `make list-personas` | Show available user archetypes (each with its SLA floors). |
| `make list-cohorts` | Show available team mixes with persona weights. |
| `make list-runs` | List `run_NN/` directories under `runs/` with their DB counts. |
| `make dashboard` | Live `rich`-based progress view of the latest run. |
| `make export [SLIM=true]` | Build `buyer_page_data.json` from `runs/run_NN/run.db`, **landing the JSON inside the same `runs/run_NN/` directory** so all per-run artifacts (DB, engine logs, perf CSVs, exported JSON) stay grouped. Includes per-step rollups (`curve[]`), per-step time series (`curve[i].telemetry_samples`, `curve[i].turns`), and the 1 Hz whole-run heartbeat (`cohort.snapshots`) so a downstream website can drill from the knee chart into the underlying turn-by-turn data without a second round-trip. **`SLIM=true`** produces `buyer_page_data_slim.json` instead — same headline summary, capacity landing zones, per-step rollup, and bottleneck attribution, but without the per-step time-series or whole-run heartbeat. ~99% smaller (35 MB → 100-200 KB) — use for buyer-facing summary distribution. Override the destination with `--output <path>` if needed. |
| `make analyze-prefix-cache` | Prefix-cache hit-rate report on the latest `.db`. |
| `make optimize-engine [ONLY=...] [RUN_NEW=true]` | Iterate a registry of vLLM-CPU launch shapes (dual-replica, chunked prefill, larger KV pool, single-replica, TP=2, …), measure TTFT / TPOT / throughput across representative input/output/concurrency cells (incl. the long-context pain point at c=8 / c=16). **Always nohup'd + auto-tailed** — survives SSH disconnect; Ctrl-C exits the tail without killing the run. Resumes the existing `runs/engine_optimizer/run.json` by default (skips configs already marked `ok` / `launch_failed`), persists after every cell so a crash mid-config loses at most one cell. Pass `RUN_NEW=true` to wipe and start fresh. `ONLY=baseline,kv_xl` runs a subset; `LIST=1` prints the registered configs. Use this on a new host to find the best engine config before kicking off a full sweep. |
| `make optimize-dashboard` | Read-only live dashboard against the running optimizer. Polls `runs/engine_optimizer/run.json` plus the latest `optimizer_*.log` in the same dir, renders the same layout as the foreground run (current config, phase, log tail, results-so-far). Use from a second SSH session to watch a backgrounded `make optimize-engine` without touching it. |
| `make preflight CONFIG=...` | Hardware-only check (no install / build). |
| `make launch-engine CONFIG=...` | Manually launch the engine without running a cohort (curl-poking). |
| `make sglang-shell` | Interactive `bash` inside `sglang-cpu:xeon-fixed` with the model dir mounted. |
| `make test` | Run pytest. |
| `make clean` / `clean-runs` | Tidy. |

Variables: `CONFIG` (path to yaml), `COHORT` (cohort id), `ENGINE`, `MODEL`, `RUN_DIR`.

Power-user / debugging escape hatches: `make sglang-clone`, `make sglang-base`, `make sglang-fixed`, `make sglang-verify`, `make download-model MODEL=...`. These are what `make ready` invokes internally; call them by hand if the orchestration mis-detects state.

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
tests/                # pytest suites
config/default.yaml
Makefile
```

### Run directory layout

```
runs/
  run_01/
    run.db                              # one DB per run_NN
    engine_sglang_1714851402.log
    perf_m0_8.csv
    sweep_20260504T231642.log
  run_02/
    ...
```

Every invocation writes into a numbered `run_NN/` subdirectory and
all cohort/persona invocations against that run share a single
`run.db` — the schema keys every row by `cohort_run_id` so cohorts
don't interfere. Schema reference: [docs/database_schema.md](docs/database_schema.md). Default behaviour is **resume**: the latest
`run_NN/` is reused, and `make run-sweep` skips personas/cohorts that
already have `final_status='ok'` inside it. Pass `RUN_NEW=true` to
cut a fresh `run_NN+1/`. `make dashboard`, `make export`, and
`make analyze-prefix-cache` all read from the latest `run_NN/`.

## vLLM dual-socket (AMD EPYC)

The `vllm_dual_socket` engine type runs two vLLM-CPU containers — one pinned to each NUMA node. The simulator load-balances virtual users across replicas with **sticky assignment**: each user is assigned to the least-loaded replica on first request and stays there for their full lifetime, so multi-turn conversations preserve prefix-cache locality on one backend. Scales nearly linearly across sockets (~2× single-socket) versus ~1.4× for TP=2 across sockets where Gloo all-reduce becomes the bottleneck.

```bash
make ready CONFIG=config/r7735_vllm_dual_socket_qwen3_30b_a3b.yaml
make run-cohort CONFIG=config/r7735_vllm_dual_socket_qwen3_30b_a3b.yaml \
                COHORT=chat_heavy
```

`make ready` will `docker pull` the upstream `vllm/vllm-openai-cpu:latest-x86_64` image automatically. Two containers come up per launch: `vllm-r0-*` (NUMA 0) and `vllm-r1-*` (NUMA 1). The simulator owns the per-user routing — no proxy in the request path.

**Image choice matters.** Use `vllm/vllm-openai-cpu`, not the SGLang `xeon-fixed` image — the latter's torch wheel lacks AVX-512 BF16 dispatch and runs at ~3% of theoretical compute on AMD. The SGLang Docker pipeline targets Intel only.

**Stickiness verification.** At end of run the simulator logs the per-replica user-assignment count (`replica[0]=N, replica[1]=M  (X users total)`) — should always show ±1 balance — and the aggregated prefix-cache hit/query counts pulled from each replica's `/metrics`. A persona on its 5th turn should be hitting the prefix cache; if `prefix_cache_hit_rate` is near 0% the sticky assignment broke or KV was evicted under pressure.

**Pin to physical cores only.** The R7735 config's `cpuset_cpus: "0-31"` / `"32-63"` excludes SMT siblings (CPUs 64-127). Using SMT siblings hurts BF16 matmul because the AVX-512 SIMD unit is shared between siblings.

**Smoke-check the wheel before any cohort run.** The `vllm-openai-cpu` image's torch wheel needs the AVX-512 BF16 dispatch path. A single-trial matmul of `(2048×2048) × (2048×2048)` should hit **~7000+ GFLOPS** on 32 EPYC 9374F cores — but **only after warmup**. The first trial is consistently 50% of steady-state because Linux schedules threads onto cold cores and the AVX-512 BF16 vector unit takes a few iterations to stabilise. Run multiple trials and look at the median:

```bash
docker run --rm --cpuset-cpus 32-63 --cpuset-mems 1 \
    -e OMP_NUM_THREADS=32 \
    --entrypoint python vllm/vllm-openai-cpu:latest-x86_64 \
    -c "
import torch, time, statistics
torch.set_num_threads(32)
a = torch.randn(2048, 2048, dtype=torch.bfloat16)
b = torch.randn(2048, 2048, dtype=torch.bfloat16)
for _ in range(20): torch.matmul(a, b)        # warmup
trials = []
for _ in range(5):
    s = time.time()
    for _ in range(30): torch.matmul(a, b)
    trials.append(30 * 2 * 2048**3 / (time.time() - s) / 1e9)
print(f'Median: {statistics.median(trials):.1f} GFLOPS')"
```

The diagnostic value is the **median across trials**, not the first number. ~7900 is healthy. ~225 means the torch wheel is broken on this host — file a build issue and stop. ~4000 with a single trial usually means you didn't warm up; rerun with the loop above.

## SGLang on CPU (Docker required, Intel only)

SGLang's mainline pip wheel ships GPU-only `sgl_kernel` binaries — importing the package on a CPU-only host fails before the launcher sees its arguments. The working CPU path is the upstream `sglang-cpu` Docker image, layered with `sentencepiece` / `tiktoken` / `protobuf` so modern HF tokenizers (Qwen3, GLM, Mistral, Llama 3) load.

There's no published Docker Hub tag for the CPU build; build from SGLang source.

### One-shot setup

```bash
make ready CONFIG=config/r7735_sglang_qwen3_30b_a3b.yaml
```

That handles everything: pip install, clone SGLang source if missing, build `sglang-cpu:xeon` (~15-20 min first run only), build the layered `sglang-cpu:xeon-fixed`, download the model with `HF_HUB_ENABLE_HF_TRANSFER=1`, and run the hardware preflight. Re-running is idempotent — already-built images and already-downloaded models are detected and skipped.

For an SSH-resilient first run on a fresh box, wrap in `tmux`.

The `-2507` suffix is part of the actual published HF repo name, not a separate version tag. Two variants exist: `-Instruct-2507` (BF16, portable) and `-Instruct-2507-FP8` (Intel-only — SGLang's CPU FP8 path requires AMX). If a download finishes with safetensors but missing tokenizer files, re-run `hf download $MODEL --local-dir ... --include 'tokenizer*' '*.json'`. Note: `protobuf` installs as `protobuf` but imports as `google.protobuf` — `import protobuf` will fail even though the install is fine.

### Run

```bash
# BF16 baseline
make launch-engine ENGINE=sglang \
                   MODEL=Qwen/Qwen3-30B-A3B-Instruct-2507 \
                   CONFIG=config/r7735_sglang_qwen3_30b_a3b.yaml

# FP8 on Intel Xeon (AMX) — preflight blocks AMD hosts
make launch-engine ENGINE=sglang \
                   MODEL=Qwen/Qwen3-30B-A3B-Instruct-2507-FP8 \
                   CONFIG=config/xeon_sglang_qwen3_30b_a3b_fp8.yaml
```

The container streams its stdout/stderr to `runs/engine_sglang_*.log`. Wait for the simulator's `SGLang ready after Xs` message — this only fires once `/v1/models` returns 200, which means the model is fully loaded. Expected times:

| Variant | Resident | Cold load |
|---|---|---|
| BF16 (`Qwen3-30B-A3B-Instruct-2507`) | ~58 GB | ~2-3 min |
| FP8 (`Qwen3-30B-A3B-Instruct-2507-FP8`) | ~32 GB | ~1-2 min |

**Important:** SGLang's CPU FP8 path is gated on Intel AMX (`Fp8LinearMethod on CPU requires that CPU has AMX support`). On AMD CPUs (R7735), use the BF16 variant. The FP8 config is preserved for the Xeon comparison run only.

### Known-good parallelism shapes

**TP=1** baseline and **TP=4** only. DP/EP combos are upstream-broken on CPU. The simulator's `derive_sglang_thread_binding` validates `tensor_parallel_size` against `engine.cpu_bind` at launch — misconfigurations surface immediately instead of 20 minutes into a model load.
