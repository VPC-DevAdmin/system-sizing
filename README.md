# capsim — AI sizing and capacity engine

Finds how much LLM traffic one box can carry, and says why it stops there. capsim launches an inference engine (vLLM, SGLang, TensorRT-LLM, KTransformers — CPU or GPU — or an endpoint you don't own), drives it with persona-based multi-turn sessions, and reports capacity with telemetry-attributed bottleneck evidence and a versioned JSON export.

capsim finds capacity two ways:

- **Open-loop (primary).** Sessions arrive as a Poisson process at a controlled rate λ; within a session the user behaves closed-loop (turn N+1 waits for turn N plus think time). Capacity is the arrival rate at which the engine's waiting queue turns divergent — below it the queue is stationary, above it the backlog grows without bound. Two knees come out: **λ_max** (highest stable rate) and **λ_sla** (highest stable rate whose steady-state turns also pass the persona SLAs). Concurrent-session capacity is derived from λ_sla and the measured mean session length, never assumed. This is the default for `capsim run`, `run-persona` and every run started from the UI.
- **Closed-loop pool ramp.** A fixed pool of virtual users stepped up a grid (4, 8, … 256) until the SLA pass rate drops; a Wilson-CI two-knee stepper is opt-in. A closed pool throttles its own offered load and never exhibits the collapse that defines capacity, so this path remains for sweeps, spot-checks and A/B comparisons against older runs.

The methodology is spelled out in [docs/algorithm.md](docs/algorithm.md). Roadmap and what shipped per phase: [docs/roadmap.md](docs/roadmap.md). Open findings from the 2026-09-19 review: [docs/improvement_plan.md](docs/improvement_plan.md).

## Landing on a fresh host

Three commands from bare box to verified pipeline — prerequisites are just Python 3.10+, Docker, and git ([docs/deploy.md](docs/deploy.md) has the per-host-class detail, including the sudo-only driver stage):

```bash
git clone <repo> && cd system-sizing && ./install.sh   # installs uv + capsim (isolated, no sudo)
capsim doctor                                          # validate host: CPU/GPU/docker/disk/telemetry perms/HF
capsim smoke --profile <recommended>                   # ~10 min end-to-end proof with a tiny model
```

`capsim doctor` prints a pass/warn/fail table (and writes `runs/doctor.json` for scripting), including GPU stack checks (nvidia-smi + container toolkit) on Xeon+NVIDIA boxes, and recommends candidate **profiles** for the detected hardware. `capsim smoke` launches the real engine with a ~1.5 GB stand-in model, runs 2 virtual users through a short measured window, exports, and validates the export against the schema contract — if smoke passes, the whole pipeline works on this host. Then commit to the full model:

```bash
capsim ready --profile <recommended>                   # full model download + image build/pull
capsim serve                                           # web UI: http://localhost:8321
```

`capsim` and `python -m simulator.cli` are the same CLI (the `simulator` entry point remains as a deprecated alias); the Make targets below wrap it.

## Targets and profiles

The same persona/knee methodology runs against every engine type registered in [simulator/engines/\_\_init\_\_.py](simulator/engines/__init__.py), selected by `engine.type` in the config:

| `engine.type` | What it launches |
|---|---|
| `vllm` | vLLM-CPU in Docker with the Xeon CPU tuning recipe (thread binding, KV pool sizing). |
| `vllm_cuda` | vLLM CUDA in Docker — upstream `vllm/vllm-openai` image with `--gpus`, for Xeon+NVIDIA hosts. |
| `vllm_cuda_multi` | Multi-replica CUDA vLLM — the whole box as N independent replicas. |
| `sglang` | SGLang-CPU in Docker (Intel only; image built from source, see below). |
| `sglang_cuda` | SGLang on GPUs — the whole box as N independent replicas. |
| `trtllm` | TensorRT-LLM — `trtllm-serve` as a whole-box, N-replica target, measured on the same footing as vLLM. |
| `ktransformers` | KTransformers — MoE experts on the CPU, attention on the GPU. |
| `vllm_dual_socket` | vLLM-CPU dual-replica engine for dual-socket NUMA boxes (one container per socket, sticky user routing). |
| `remote` | An OpenAI-compatible endpoint you don't own; host telemetry is skipped and recorded as such, the endpoint's `/metrics` is scraped when configured. See the [remote-endpoint template](config/profiles/remote-endpoint.yaml). |
| `mock` | A real in-process OpenAI-compatible SSE server with a synthetic, tunable capacity knee — the whole pipeline on a laptop with zero hardware. Powers CI and UI development (`--profile mock`). |

Local CPU engines get the full host telemetry set (PMU, IMC bandwidth, RAPL power, frequency, AMX); GPU engines add the GPU collector (NVML, nvidia-smi fallback: SM util / VRAM / power / clocks / throttle state at 1 Hz) and bottleneck attribution gains `gpu_compute` and `gpu_throttled` labels. Every engine reads one memory knob (`engines/vram.py`) and one launch-lever vocabulary (`engines/knobs.py`), translated per engine, so a profile or an Optimize search means the same thing whichever server is under test.

A **profile** is a curated named config for one host class. `capsim list-profiles` shows what's available; every command accepts `--profile <name>` in place of `--config <path>`. Curated profiles live in [config/profiles/](config/profiles/) (`xeon-gpu-qwen3-30b`, `mock`, and the `remote-endpoint` template); the CPU configs at `config/xeon_*.yaml` and `config/r7735_*.yaml` are addressable by stem too (`--profile r7735_vllm_dual_socket_qwen3_30b_a3b`). The export's per-cohort `collectors` block records which telemetry sources actually ran, so downstream consumers know what evidence backs each bottleneck claim.

## Quick start

```bash
# Zero hardware: the whole pipeline against the mock engine.
capsim run --cohort chat_heavy --profile mock

# A real box, headless. Two commands to first measurement.
make ready CONFIG=config/r7735_vllm_dual_socket_qwen3_30b_a3b.yaml
make run-cohort CONFIG=config/r7735_vllm_dual_socket_qwen3_30b_a3b.yaml \
                COHORT=chat_heavy
# In a second terminal:
make dashboard

# After the run finishes:
make export
capsim serve                                        # web UI: http://localhost:8321
```

`capsim run` and `capsim run-persona` take `--mode open|closed` (default `open`); the same knob is `simulation.mode` in YAML. Sweeps (`make run-sweep`, `capsim sweep`) stay closed-loop.

## The web UI

`capsim serve` starts the control-plane service and the UI on `127.0.0.1:8321`: run lifecycle over HTTP (`/api/runs`, `/api/status`, `/api/export`, `/api/doctor`, catalogs, the optimizer and roofline endpoints) and live telemetry over WebSocket (`/ws/telemetry` — run/snapshot/telemetry/turn/step events). One run at a time, in-process; `run.db` stays the source of truth. The service has no auth: binding `--host` to anything other than loopback requires `--insecure`. Over SSH, tunnel it (`ssh -L 8321:localhost:8321 <host>`).

The UI has no build step (Chart.js is vendored so offline benchmark boxes work) and six tabs, in the order a new box goes through them:

| Tab | What it is for |
|---|---|
| **Prepare** | The landing checklist. *Validate the host* (runs doctor, ends with a profile recommendation), *Choose where models live* (pick the disk and directory weights download to — doctor, downloads, engines and the optimizer all follow it), *Stage model weights* (the catalog, what is cached, add or discover Hugging Face models), *Stage engine runtimes* (which server images are on the box — an engine only appears downstream once its image is here), then *Find the best launch shape* → Open Optimize. |
| **Optimize** | *Build the test arena* — every launch shape this installation can run (detected GPUs and PCIe domains, optionally hinted by `config/arena.yaml`, copied from [config/arena.example.yaml](config/arena.example.yaml); the tab warns when the file claims more GPUs than are detected) — then *Guided search* over it and *Results* with the mean-rank composite that names the winner. The winner is promoted into the launch the Workload tab benchmarks with. |
| **Roofline** | The *Roofline autopilot*: "what is the most this box can do?" across models × engines × shapes. Stages what it needs, searches, confirms the best cell, and reports a model × engine matrix plus every measured cell. Resumable. |
| **Workload** | *Start a capacity benchmark*: model, CPU-only vs CPU+GPU, workload (a persona or cohort, or a headline saturation benchmark that steps fixed concurrency the way vendor numbers are made), and the advanced engine settings (replicas, TP, placement, memory fraction, batch width, KV precision, expert parallel, trust-remote-code). Live telemetry lives here while a run is in flight: sessions / in-flight / queue, rolling TTFT and TPOT percentiles, engine and host utilization, prefill vs decode token rates, session states, system detail, and completed steps. |
| **Results** | The run list (newest first; click one, check several to compare). One run renders as a narrative capacity report — headline verdict, user experience (SLA violations and latency vs load), GPUs, CPU and host memory, power and efficiency — with step drill-down. *Comparison* overlays the checked runs on shared axes. *Download JSON* fetches the versioned export. |
| **Edit workloads** | The *Workload designer*: personas (how much a user asks, gets back, reads and thinks; SLA floors) and cohorts (the persona mix a deployment sees), edited graphically and saved through validating endpoints. The next run picks edits up. |

## CLI

Every subcommand, from `capsim --help`:

| Command | What it does |
|---|---|
| `capsim run --cohort <id>` | Run a single cohort (team mix) end-to-end. `--mode open\|closed`, default open. |
| `capsim run-persona --persona <id>` | Run a single persona end-to-end (one user archetype, no team mix). `--mode open\|closed`, default open. |
| `capsim sweep` | Run multiple personas + cohorts back-to-back against one engine (closed-loop; resumes the latest `run_NN/`, `--new-run` cuts a fresh one). |
| `capsim spot-check --plan … --run-dir …` | Re-measure specific (cohort, pool_size) points from an audit plan, appending them to the existing cohort_run rows. |
| `capsim launch-engine` | Launch the engine without running simulations (manual testing). |
| `capsim dashboard` | Live progress view of the most recent run. |
| `capsim export` | Export simplified JSON for the buyer-facing webpage. |
| `capsim list-profiles` | List available hardware profiles (curated `config/profiles/` plus plain `config/` stems). |
| `capsim list-cohorts` | List available team-mix cohorts. |
| `capsim list-personas` | List available user-archetype personas. |
| `capsim ready` | Prepare engine + model + host for a config. |
| `capsim serve` | Run the control-plane service: run lifecycle over HTTP + live telemetry over WebSocket, and the web UI. |
| `capsim doctor` | Full host validation for a fresh benchmark box; writes `runs/doctor.json` (`--output` to move it). |
| `capsim smoke` | End-to-end micro-benchmark: real engine, tiny model, 2 virtual users, export validated against the schema. |
| `capsim preflight --config …` | Validate the host satisfies a config's `hardware_requirements`. |
| `capsim current-run-dir` | Print the `run_NN` directory the next invocation would use. |
| `capsim analyze-prefix-cache <db>` | Compute prefix-cache hit rate from captured turn events. |

Every run/sweep/ready/smoke command accepts `--profile <name>` or `--config <path>`, and `--engine` / `--model` overrides of what the YAML says.

## Make targets

The headline workflow is `ready` → `run-cohort` → `dashboard` → `export`. Everything else is either a downstream analysis step or a diagnostic. `make` with no target (or `make help`) prints the short form of this table.

| Target | What it does |
|---|---|
| `make ready CONFIG=...` | Idempotent: create the project venv if missing, `pip install -e .`, then `capsim ready` — build the engine docker image (SGLang only) if missing, download the model if missing, validate hardware. |
| `make setup` | Back-compat alias for `ready`. |
| `make doctor` | Full host validation — CPU/NUMA/docker/GPU stack/disk/telemetry permissions/HF reachability as a pass/warn/fail table plus `runs/doctor.json`. Run first on any fresh box. |
| `make smoke CONFIG=...` | End-to-end pipeline proof: real engine + tiny model (~1.5 GB), 2 virtual users, short measured window, export validated against the schema contract. ~10 min on a fresh box. |
| `make run-persona CONFIG=... PERSONA=...` | Run one **persona** (a single user archetype) end-to-end. |
| `make run-cohort CONFIG=... COHORT=...` | Run one **cohort** (a team mix of personas) end-to-end. |
| `make run-sweep CONFIG=... [SWEEP_TYPE=...] [RUN_NEW=true]` | Sweep multiple workloads (closed-loop). **Always nohup'd, log-teed, and auto-tailed** — the sweep + its engine containers survive SSH disconnect; Ctrl-C only exits the tail. Prints the run dir + log path on launch and follows the log live; reattach with `make tail-log`, stop with `make stop-bg`. `SWEEP_TYPE` accepts `all` (default — every persona + every cohort), `personas`, `cohorts`, or a comma-separated list of ids. **Resumes the latest `runs/run_NN/` by default** — workloads with `final_status='ok'` are skipped. `RUN_NEW=true` cuts a fresh `run_NN+1` (use it when config or hardware changed). |
| `make run-cohort-bg ...` / `make run-persona-bg ...` | Background single-workload variants. Same nohup pattern; non-blocking (no auto-tail). |
| `make tail-log` | Tail the most-recent background-run log (auto-picks the latest). |
| `make stop-bg` | Kill any running simulator background process and its engine containers. |
| `make audit` | Audit the latest run for curve-quality anomalies (no marginal band, no fail observed, single-point rescue, boundary status) with `scripts/audit_run.py`; writes `runs/run_NN/audit_report.json`. |
| `make spot-check` | Re-measure the (cohort, pool_size) points `audit` flagged, appending them to the existing cohort_run rows so a re-run `make export` picks up the enriched curve. nohup'd + auto-tailed; stops any other engine first. |
| `make list-personas` | Show available user archetypes (each with its SLA floors). |
| `make list-cohorts` | Show available team mixes with persona weights. |
| `make list-runs` | List `run_NN/` directories under `runs/` with their DB counts. |
| `make dashboard` | Live `rich`-based progress view of the latest run. |
| `make export [SLIM=true]` | Build `buyer_page_data.json` from `runs/run_NN/run.db`, **landing the JSON inside the same `runs/run_NN/` directory** so all per-run artifacts (DB, engine logs, perf CSVs, exported JSON) stay grouped. Includes per-step rollups (`curve[]`), per-step time series (`curve[i].telemetry_samples`, `curve[i].turns`), and the 1 Hz whole-run heartbeat (`cohort.snapshots`). **`SLIM=true`** produces `buyer_page_data_slim.json` — same headline, landing zones, per-step rollup and bottleneck attribution, without the time series (~99% smaller). |
| `make analyze-prefix-cache` | Prefix-cache hit-rate report on the latest `.db`. |
| `make optimize-engine [PROFILE=...] [ONLY=...] [RUN_NEW=true]` | Iterate a registry of engine launch shapes, measure TTFT / TPOT / throughput across representative input/output/concurrency cells. **Always nohup'd + auto-tailed**. Resumes `runs/engine_optimizer/run.json` by default (skips configs already `ok` / `launch_failed`), persists after every cell. `ONLY=baseline,kv_xl` runs a subset; `LIST=1` prints the registered configs; `PROFILE=` selects the registry — CPU profiles (`amd_*`, `intel_*`) sweep cpuset/NUMA/OMP shapes, `nvidia_qwen3` sweeps CUDA-vLLM shapes. The UI's **Optimize** tab is the interactive front end to the same driver. |
| `make optimize-search [SPACE=config/search/<space>.yaml]` | Guided coarse-to-fine search over a full parameter space (model variants/precision, engine, TP, DP, GPU placement across PCIe domains, memory fraction, batch width, chunked prefill): a coverage sample, an SLA-aware objective (throughput with p95-latency caps), then one-dimension neighborhood refinement around the leaders until the budget is spent. Deterministic under a seed, deduped, resumable; the pure search logic is unit-tested in [simulator/search.py](simulator/search.py). Same as the Optimize tab's *Guided search*. |
| `make optimize-dashboard` | Read-only live dashboard against the running optimizer (polls `runs/engine_optimizer/run.json` + the latest `optimizer_*.log`). Use from a second SSH session. |
| `make preflight CONFIG=...` | Hardware-only check (no install / build). |
| `make launch-engine CONFIG=...` | Manually launch the engine without running a cohort (curl-poking). |
| `make sglang-shell` | Interactive `bash` inside `sglang-cpu:xeon-fixed` with the model dir mounted. |
| `make sglang-build` | Build the SGLang CPU image pipeline end-to-end: clone source, build `sglang-cpu:xeon`, layer `sglang-cpu:xeon-fixed`, verify the imports. |
| `make models-dirs` | Create `MODELS_DIR` and `HF_CACHE_DIR` (sudo, once) and chown them to the current user. |
| `make download-model MODEL=...` | `hf download` the model into `MODELS_DIR` with `HF_HUB_ENABLE_HF_TRANSFER=1`. |
| `make test` | Run pytest. |
| `make clean` / `clean-runs` / `clean-venv` | Remove caches / `runs/run_*` and stray DBs / the project venv. |

Variables (Makefile lines 14–46): `ENGINE`, `MODEL` (override what the YAML says — rare), `COHORT`, `PERSONA`, `SWEEP_TYPE`, `ADAPTIVE` (opt into the two-knee stepper in closed-loop runs), `POOL_SIZES` (override the fixed grid), `CONFIG`, `RUN_DIR`, `RUN_NEW`, `VENV`, `PY`, `SGLANG_REPO`, `SGLANG_SRC`, `SGLANG_BASE_IMAGE`, `SGLANG_FIXED_IMAGE`, `SGLANG_DOCKERFILE`, `MODELS_DIR`, `HF_CACHE_DIR`, `LOCAL_MODEL_DIR`. The optimizer targets add `PROFILE`, `ONLY`, `LIST`, `SPACE`.

Power-user / debugging escape hatches: `make sglang-clone`, `make sglang-base`, `make sglang-fixed`, `make sglang-verify`. These are what `make sglang-build` (and `make ready`, via the Python equivalents) invoke internally; call them by hand if the orchestration mis-detects state.

## Layout

```
simulator/
  # core loop — how a measurement is made
  open_loop.py        # open-loop cohort orchestration: arrival-rate capacity search (primary)
  rate_search.py      # arrival-rate search: λ_max and λ_sla, doubling then log-space bisection
  stability.py        # queue-stability statistics: Mann-Kendall × Theil-Sen verdict per window
  arrivals.py         # open-loop session arrival generation (Poisson, cancel-newest on revert)
  loadgen_worker.py   # load-generator worker process (sharded Poisson, tardiness signal)
  virtual_user.py     # virtual user runtime: one async task per simulated user
  streaming.py        # tiered-timeout SSE stream consumer
  pool_manager.py     # closed-loop pool manager: keeps the active user set at target size
  measurement.py      # closed-loop measurement controller: ramp → stabilize → measure
  adaptive.py         # two-knee adaptive stepper (closed-loop, opt-in)
  timeline.py         # per-measurement phase-distribution timeline
  runner.py           # top-level closed-loop cohort-run orchestration
  # control plane
  service/            # control-plane HTTP service + UI host (FastAPI, /api + /ws/telemetry)
    app.py            #   create_app / serve: state on app.state, runs-dir lock, routers, UI mount
    state.py          #   ActiveRun, Paths, the runs-dir lock
    schemas.py        #   request bodies
    runs.py           #   run lifecycle, run list, exports, live backfill
    catalog.py        #   profiles, personas/cohorts + editor, headline shapes, hardware
    prepare.py        #   storage, model staging, engine runtime pulls
    optimizer.py      #   engine optimizer + arena, history, promote
    roofline.py       #   roofline state + candidates
    telemetry.py      #   /ws/telemetry
  bus.py              # in-process telemetry event bus
  runs.py             # run-directory layout helpers (run_NN, resume-by-default)
  cli.py              # `capsim` command-line interface (typer)
  config.py           # configuration loading and validation (YAML + profiles + overrides)
  # optimize — finding the launch shape before measuring with it
  search.py           # guided launch-shape search: coarse-to-fine over a parameter space
  arena.py            # the test arena: every launch shape this installation can run
  roofline.py         # roofline autopilot: what is the most this hardware can do?
  headline_sweep.py   # headline (saturation) sweep — the marketing number, measured honestly
  headline_search.py  # headline shape search: which (input, output) shape jointly maximises
  headline_optimize.py# joint engine + shape search for the headline number
  headline_shapes.py  # per-model-family store of optimal headline shapes
  promote.py          # promote an optimizer winner into a benchmark profile
  engine_runtimes.py  # which engine runtimes this host can actually launch
  engine_notes.py     # measured tuning levers, and what they actually did on this box
  model_catalog.py    # model catalog: models as data, additions as one line of YAML
  models.py           # model weight staging: HF cache inspection + download commands
  discovery.py        # live model discovery: the Hub, filtered through this box
  engines/
    __init__.py       # engine registry (make_engine)
    base.py           # engine abstraction: subprocess + Prometheus metric parser
    docker_replica.py # shared machinery for whole-box, N-replica Docker engines
    knobs.py          # one vocabulary of launch knobs, two engine dialects
    vram.py           # one memory knob, translated per engine
    vllm.py           # vLLM-CPU with the Xeon tuning recipe
    vllm_cuda.py      # vLLM CUDA (Docker)
    vllm_cuda_multi.py# multi-replica CUDA vLLM
    vllm_dual_socket.py # vLLM-CPU, one replica per NUMA node
    sglang.py         # SGLang-CPU (Docker)
    sglang_cuda.py    # SGLang on GPUs, N replicas
    trtllm.py         # TensorRT-LLM (trtllm-serve), N replicas
    ktransformers.py  # KTransformers: MoE experts on CPU, attention on GPU
    remote.py         # remote OpenAI-compatible endpoint
    mock.py           # in-process mock engine with a synthetic knee
  # telemetry
  collectors/
    __init__.py       # telemetry collector plugins (registry)
    gpu.py            # NVIDIA GPU collector (NVML, nvidia-smi fallback)
    host.py           # host-detail collector: CPUs, memory, disks, NICs
  telemetry.py        # telemetry orchestration: snapshots, PMU, engine metrics
  perf_collector.py   # perf-stat PMU counters with GNR-aware AMX probe
  bandwidth.py        # memory bandwidth via Intel IMC uncore events
  power_probe.py      # package power via Intel RAPL
  frequency.py        # effective CPU frequency over the engine's bound CPU set
  amx_utilization.py  # oneDNN verbose log → AMX dispatch fraction
  # data
  database.py         # SQLite schema (PRAGMA user_version = 7) + capture helpers
  export.py           # the buyer-facing JSON document (validated against export_schema/)
  export_schema/      # buyer_page_data.schema.json — the export contract
  prefix_cache.py     # prefix-cache hit analysis from captured turn events
  persona_loader.py   # persona/cohort YAML loading and serialization
  personas.py         # persona and cohort dataclasses + the loaded catalog
  personas_data/      # default.yaml — 8 personas, 5 cohorts
  models_data/        # model catalog YAML
  distributions.py    # LogNormal / Discrete / Constant samplers
  tokenizer_corpus.py # filler text targeted at token counts
  # host
  doctor.py           # host validation for the one-command landing flow
  preflight.py        # hardware-compatibility preflight against a config
  cpu_binding.py      # expand a vLLM-style thread-binding string into a CPU id set
  dashboard.py        # rich live progress view
  ui/                 # index.html, app.js, app.css, vendored Chart.js — no build step
scripts/
  audit_run.py        # curve-quality audit → audit_report.json (make audit)
  engine_optimizer.py # the optimizer / guided-search driver (make optimize-*)
  gen_schema_doc.py   # regenerates docs/database_schema.md from the DDL
tests/                # pytest suites (unit, contract, mock-engine integration)
config/
  default.yaml        # base config every profile overlays
  profiles/           # curated profiles: xeon-gpu-qwen3-30b, mock, remote-endpoint (template)
  search/             # guided-search spaces (xe7740-qwen3, xe7740-multimodel)
  arena.example.yaml  # copy to config/arena.yaml to hint PCIe/NUMA GPU groups
  xeon_*.yaml, r7735_*.yaml  # CPU host configs, addressable by stem with --profile
docs/                 # algorithm, metrics, database_schema, deploy, roadmap, improvement_plan
Makefile
```

### Documentation

| Doc | What it covers |
|---|---|
| [docs/algorithm.md](docs/algorithm.md) | The methodology: open-loop arrival-rate search, the closed-loop ramp, distributions, personas, virtual-user lifecycle, measurement windows, steppers, Wilson CI, reasoning-model handling, phase timeline. |
| [docs/metrics.md](docs/metrics.md) | Every telemetry metric, its source, and what the stack cannot honestly measure. |
| [docs/database_schema.md](docs/database_schema.md) | `run.db` tables and columns at the current schema version — generated from the DDL. |
| [docs/deploy.md](docs/deploy.md) | Landing flow per host class, from bare OS to first benchmark; driver field notes. |
| [docs/roadmap.md](docs/roadmap.md) | Phases 0–5: what was planned, what shipped, and the open items. |
| [docs/improvement_plan.md](docs/improvement_plan.md) | The 2026-09-19 review: accuracy, completeness, UX and robustness findings, with the ordered fix plan. |

### Run directory layout

```
runs/
  doctor.json                             # latest `capsim doctor` report
  run_01/
    run.db                                # one DB per run_NN
    engine_sglang_1714851402.log
    perf_m0_8.csv
    sweep_20260504T231642.log
    audit_report.json                     # from `make audit`, read by `make spot-check`
    buyer_page_data.json                  # from `make export`
  run_02/
    ...
  engine_optimizer/
    run.json, search.json, optimizer_*.log
  smoke/                                  # isolated `capsim smoke` run
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
make ready CONFIG=config/xeon_sglang_qwen3_30b_a3b_fp8.yaml
```

That handles everything: pip install, clone SGLang source if missing, build `sglang-cpu:xeon` (~15-20 min first run only), build the layered `sglang-cpu:xeon-fixed`, download the model with `HF_HUB_ENABLE_HF_TRANSFER=1`, and run the hardware preflight. Re-running is idempotent — already-built images and already-downloaded models are detected and skipped. `make sglang-build` runs just the image pipeline.

For an SSH-resilient first run on a fresh box, wrap in `tmux`.

The `-2507` suffix is part of the actual published HF repo name, not a separate version tag. Two variants exist: `-Instruct-2507` (BF16, portable) and `-Instruct-2507-FP8` (Intel-only — SGLang's CPU FP8 path requires AMX). If a download finishes with safetensors but missing tokenizer files, re-run `hf download $MODEL --local-dir ... --include 'tokenizer*' '*.json'`. Note: `protobuf` installs as `protobuf` but imports as `google.protobuf` — `import protobuf` will fail even though the install is fine.

### Run

```bash
# FP8 on Intel Xeon (AMX) — preflight blocks AMD hosts
make launch-engine CONFIG=config/xeon_sglang_qwen3_30b_a3b_fp8.yaml

# BF16 variant of the same config (portable; override the model id)
make launch-engine CONFIG=config/xeon_sglang_qwen3_30b_a3b_fp8.yaml \
                   MODEL=Qwen/Qwen3-30B-A3B-Instruct-2507
```

The container streams its stdout/stderr to `runs/engine_sglang_*.log`. Wait for the simulator's `SGLang ready after Xs` message — this only fires once `/v1/models` returns 200, which means the model is fully loaded. Expected times:

| Variant | Resident | Cold load |
|---|---|---|
| BF16 (`Qwen3-30B-A3B-Instruct-2507`) | ~58 GB | ~2-3 min |
| FP8 (`Qwen3-30B-A3B-Instruct-2507-FP8`) | ~32 GB | ~1-2 min |

**Important:** SGLang's CPU FP8 path is gated on Intel AMX (`Fp8LinearMethod on CPU requires that CPU has AMX support`). On AMD CPUs (R7735), use the BF16 variant. The FP8 config is preserved for the Xeon comparison run only.

### Known-good parallelism shapes

**TP=1** baseline and **TP=4** only. DP/EP combos are upstream-broken on CPU. The simulator's `derive_sglang_thread_binding` validates `tensor_parallel_size` against `engine.cpu_bind` at launch — misconfigurations surface immediately instead of 20 minutes into a model load.
