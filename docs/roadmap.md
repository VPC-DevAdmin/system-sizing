# Roadmap: AI Sizing & Capacity Engine

Target end state: a packaged sizing/capacity engine ("capsim") that lands on a
new host with one command, validates itself, runs benchmarks through a web UI
with live telemetry graphs, and emits versioned JSON for downstream consumers.
Supported host classes at v1: **Xeon CPU-only** and **Xeon + NVIDIA GPU**.

Phases are ordered so that each one ships something usable on real hardware;
the UI comes after the service layer it depends on. Phases 0 and 1 are the
critical path for the "land on a new box" requirement.

---

## Phase 0 — Contract, packaging, and the landing story ✅ (2026-09-15)

Goal: `capsim` installs in one command on a fresh host, proves the host is
usable in minutes, and its JSON output is a versioned contract.

Shipped with two deliberate deviations from the plan below: the export
schema lives at `simulator/export_schema/buyer_page_data.schema.json`
(packaged, so an installed capsim can self-validate) rather than
`docs/`; and 0.2's "migrate-on-open" applies to the write path only —
the export/dashboard read path deliberately opens run.db read-only and
keeps its presence-filtering, which is the right behavior for reading
someone else's legacy run artifacts.

### 0.1 Versioned export contract
- Add `schema_version` (semver) to `buyer_page_data*.json`.
- Write a JSON Schema for both full and slim exports (`docs/export_schema/`).
- Pytest that validates every export produced by the test suite against the
  schema; CI-gate it. `export.py` changes then require a deliberate schema bump.
- Export gains a `collectors` block recording which telemetry sources actually
  ran (pmu, rapl, imc_bandwidth, nvml, engine_metrics, ...) so downstream
  consumers and bottleneck attribution know what evidence exists.

### 0.2 DB migrations
- Schema version stamped in `run.db` + ordered migration list in
  `database.py`. Shipped as SQLite's `PRAGMA user_version` (currently 7)
  plus a contiguous `MIGRATIONS` list — there is no `schema_version`
  table. The column reference is generated from the DDL:
  [database_schema.md](database_schema.md).
- Replace the "tolerate missing columns" defensive reads (a59c0c2) with
  migrate-on-open.

### 0.3 Package + CLI rename
- Rename entry point to `capsim` with subcommands: `capsim ready`, `capsim run`,
  `capsim sweep`, `capsim export`, `capsim doctor`, `capsim smoke`,
  (later `capsim serve`). Keep `simulator` as an alias for one release.
- Build wheel/sdist in CI; version from one place.
- Makefile becomes thin wrappers over `capsim` (keep — it's the muscle-memory
  interface on benchmark boxes).

### 0.4 One-command landing (the deploy story)
Fresh-host flow, both hardware classes:

```bash
# On the target host (only prerequisites: python3.10+, docker, git or curl):
curl -fsSL <repo>/install.sh | bash        # or: git clone && ./install.sh
capsim doctor                              # validate host, pick/confirm profile
capsim smoke                               # end-to-end micro-benchmark, ~10 min
capsim ready --profile xeon-gpu-qwen3-30b  # full model download + image pull
```

- `install.sh`: installs `uv` if missing, then `uv tool install` capsim from
  the repo (or a release wheel). No system python pollution, idempotent.
- **`capsim doctor`** — extends `preflight.py` from "CPU flags check" to a full
  host report with pass/warn/fail per item:
  - CPU: vendor, AMX/AVX-512 flags, sockets, NUMA topology, physical cores
  - GPU: `nvidia-smi` present, driver + CUDA version, VRAM per device,
    nvidia-container-toolkit works (`docker run --gpus all` probe)
  - Docker: daemon reachable, disk space on image + model volumes
  - Perms: `perf_event_paranoid`, RAPL readability, msr access (warn-only —
    telemetry degrades gracefully)
  - Network: HF reachable, token present if model is gated
  - Output: human table + `doctor.json`; exit code fit for scripting.
  - Ends by recommending a hardware **profile** (see 1.3) for this host.
- **`capsim smoke`** — proves the entire pipeline before any 30B download:
  - Downloads a tiny model (e.g. Qwen3-0.6B, ~1.5 GB), launches the profile's
    engine, runs 2 virtual users for ~60 s of measured time, exports, and
    validates the export against the JSON Schema.
  - Pass/fail summary: engine launched, streaming worked, telemetry collectors
    sampled, DB written, export valid. This is the "validate and test" gate.

### 0.5 Housekeeping
- Commit `site/` and `artifacts/` (large JSONs under git-lfs or a `data/`
  convention) — nothing load-bearing stays untracked.

---

## Phase 1 — Target generalization + Xeon/NVIDIA GPU support ✅ (2026-09-15)

Goal: the same persona/knee methodology runs against CPU engines, local GPU
engines, and (cheaply, as a byproduct) remote OpenAI-compatible endpoints.

Shipped as planned, with notes: the target concept is expressed through
`engine.type` (`vllm_cuda` for GPU local-docker, `remote` for endpoint-only)
rather than a separate config axis — one selector, no overlap. The collector
"plugin interface" is the new `simulator/collectors/` package (GpuCollector
first); the pre-existing CPU collectors already behave like plugins and were
left in place rather than mechanically wrapped. VRAM-bound attribution rides
the existing kv_cache check (vLLM preallocates VRAM to gpu-memory-utilization,
so raw used/total carries no signal); `gpu_compute` and `gpu_throttled` are
new labels. Profiles resolve from `config/profiles/` with plain `config/`
stems accepted, so the legacy configs are profiles already. The full-curve
deliverable check (install → doctor → smoke → ready → sweep on a real
Xeon+GPU box) still needs a run on actual hardware.

### 1.1 Target abstraction
- Config gains a `target` concept:
  - `local-docker` (today's behavior: capsim owns the engine container)
  - `remote-endpoint` (URL only; no host telemetry, client-side latencies +
    engine `/metrics` if exposed)
- Engine registry stays; add `vllm_cuda` launcher (upstream `vllm/vllm-openai`
  image, `--gpus`, tensor-parallel across GPUs, VRAM-aware `gpu-memory-utilization`).

### 1.2 Collector plugin interface
- `Collector` protocol: `name`, `is_available() -> bool | reason`,
  `start()/sample()/stop()`. Registry assembles the active set per target.
- Wrap existing collectors (perf/PMU, IMC bandwidth, RAPL, frequency, AMX) as
  plugins — Linux-CPU-only, auto-skipped elsewhere.
- New GPU collectors (NVML via `pynvml`, fallback `nvidia-smi --query-gpu`):
  SM utilization, VRAM used/total, power draw, clocks, PCIe throughput;
  per-device, 1 Hz, same cadence as existing telemetry.
- DB: collector samples keyed by collector name (schema migration, see 0.2);
  export `collectors` block from 0.1 wired here.

### 1.3 Hardware profiles
- `config/profiles/`: named, curated combinations of engine + model + binding +
  requirements. Shipped: `xeon-gpu-qwen3-30b`, `mock`, and the
  `remote-endpoint` template. No curated CPU-only profile exists; the
  CPU configs (`config/xeon_*.yaml`, `config/r7735_*.yaml`) are
  addressable by stem with `--profile`, which is how "existing yaml
  configs become profiles" was delivered. `capsim doctor` maps a host
  to candidate profiles; `capsim ready --profile X` does the rest.

### 1.4 Bottleneck attribution for GPU
- Extend the attribution logic: compute-bound (SM util high), VRAM-bound
  (KV cache pressure / preemptions from vLLM metrics), power/clock-throttled.
  CPU heuristics stay as-is.

Deliverable check: on a Xeon+A100/L40S box, `install.sh → doctor → smoke →
ready → sweep` produces a knee curve with GPU-attributed bottleneck evidence.

---

## Phase 2 — Control-plane service + live telemetry bus ✅ (2026-09-15)

Goal: everything the Makefile/CLI can do is callable over HTTP, and telemetry
streams to subscribers in real time.

Shipped with two notes. (1) The rich-TUI dashboard was NOT rewritten as a bus
subscriber: it runs in a separate process over SSH and polls the DB, which the
in-process bus can't serve — it stays as-is, and the live view moves to the
browser (a bus/WebSocket client) in Phase 3. (2) 2.4's measurement-math
coverage largely predated this phase (steppers, Wilson CI, and timeline each
had extensive fixture tests); the piece that was genuinely missing — a
mock-engine end-to-end integration test through the real HTTP/SSE/client/
measurement/DB/export path — now runs in ~5 s in CI, plus the same flow driven
through the service API with events observed on the WebSocket. Service binds
127.0.0.1:8321 by default, no auth by design.

### 2.1 Service
- FastAPI app: `capsim serve` (localhost by default, single-operator, no auth).
- Endpoints: list/create/resume/stop runs; run status; list personas/cohorts/
  profiles; doctor report; trigger export; fetch export JSON.
- Runner executes in-process (async already) with the service owning lifecycle;
  `run.db` remains the source of truth.

### 2.2 Telemetry bus
- In-process pub/sub: every 1 Hz telemetry sample and every turn-completion
  event goes to DB **and** to subscribers.
- WebSocket endpoint streams: engine metrics, collector samples, pool phase
  distribution, rolling TTFT/TPOT percentiles, SLA pass-rate per step.
- `dashboard.py` (rich TUI) is rewritten as a bus subscriber — one data path,
  two views.

### 2.3 Mock engine
- `engine: mock` replays canned SSE streams with configurable latency
  distributions. Powers: UI development without hardware, integration tests in
  CI, and deterministic fixtures for measurement-math unit tests.

### 2.4 Measurement-math tests
- Fixture-based unit tests (synthetic run.dbs / mock-engine runs) for knee
  detection, Wilson-CI stepping, and phase-timeline attribution — the area the
  last three fix commits touched.

---

## Phase 3 — The UI ✅ (2026-09-15)

Goal: run the benchmark end-to-end from a browser with ongoing telemetry graphs.

Shipped as planned and verified in a live browser against a mock run: run
control (profile + workload pickers, start/stop, doctor, run history), live
telemetry (pool/in-flight, rolling TTFT/TPOT percentiles from turn events,
KV/CPU/GPU chart, per-step progress bar, completed-steps table), and results
(landing zones, knee chart with Wilson CI band + zone markers, latency vs
pool, bottleneck evidence + collector statuses, click-through step detail,
cross-run comparison overlay with lazy per-run export loading, export JSON
download). Two deviations: Chart.js is VENDORED into the package rather than
CDN-loaded — benchmark boxes are often offline (the wheel ships the UI as
package data at simulator/ui/); and uvicorn moved to uvicorn[standard]
because plain uvicorn has no WebSocket protocol (caught live — TestClient's
in-process ASGI masks it). web/index.html retirement stays in Phase 4.

- **Stack decision: stay no-build.** Chart.js + vanilla ES modules served by
  `capsim serve` (same origin as the API/WebSocket). The existing `web/` and
  `site/assets/app.js` code carries over; no toolchain to land on benchmark
  boxes. Revisit only if the UI outgrows it.
- Three views (as shipped in Phase 3; Phase 5 grew the UI to six tabs —
  Prepare, Optimize, Roofline, Workload, Results, Edit workloads):
  1. **Run control** — pick profile + persona/cohort/sweep, see doctor status,
     launch, stop/resume; run list with per-run status.
  2. **Live telemetry** — streaming charts during a run: in-flight vs pool
     size, TTFT/TPOT percentiles, tokens/s, phase distribution stack,
     CPU util + memory bandwidth + power (CPU hosts) or SM util + VRAM +
     power (GPU hosts), current step + SLA pass-rate with CI.
  3. **Results** — knee curve with landing zones, bottleneck evidence panel,
     drill-down to turns (port from `web/index.html`), **run-comparison
     overlay** (two+ runs' knee curves on one chart — Intel vs GPU, config A
     vs B).
- Export button in Results → downloads the versioned slim/full JSON.
- `site/` (Dell narrative pages) is explicitly downstream: consumes exports,
  no longer part of the tool.

---

## Phase 4 — Personas as data + productization polish ✅ (2026-09-15)

Shipped: the persona/cohort catalog is YAML (packaged canonical file
verified equivalent to the old Python literals; config/personas/*.yaml
overlays; in-place registry reload), edited from a new Personas tab (now
**Edit workloads**) in the UI through validating GET/PUT endpoints — a bad save restores the
previous file and never wedges the registry. CI (GitHub Actions) runs
ruff + the full suite (contract tests + mock-engine integration) on
3.11/3.12 plus a wheel build that asserts the UI/schema/personas
package data made it in; an explicit ruff policy landed with the
codebase brought to clean. docs/deploy.md carries the landing flow per
host class. web/index.html and `make web` are retired — the capsim UI
supersedes them (site/ remains as the downstream Dell narrative).
Nightly smoke on a real runner remains open until a benchmark box is
attached to CI.

**Project status at Phase 4:** the four planned phases complete. What
shipped after them, unplanned, is Phase 5 below; the open items are
listed after it.

- Personas/cohorts move from `personas.py` to YAML (`config/personas/`),
  validated on load; UI gains a persona/cohort editor (writes YAML through the
  service). SLA floors live with the persona.
- CI: ruff + pytest + export-schema validation + mock-engine integration run +
  wheel build on every push; optional nightly smoke on a real runner.
- Docs: update README around `capsim`, add `docs/deploy.md` (the Phase 0 flow),
  keep `docs/algorithm.md` authoritative for methodology.
- Retire what's superseded: `web/index.html` (absorbed into UI), any Makefile
  targets fully replaced by `capsim`.

---

## Phase 5 — open-loop methodology, roofline, arena, headline (2026-09-16 → 2026-09-19)

Not in the original plan. Once the tool ran end to end on the XE7740
(8× RTX PRO 6000 Blackwell), the closed-loop ramp reached 2,048 users
with zero SLA violations — a fixed pool throttles its own offered load
and never collapses — so the methodology had to change, and the
optimizer had to cover more than one engine. Everything below landed
between 2026-09-16 and 2026-09-19; commit hashes are the trail.

### 5.1 Open-loop arrival-rate capacity (`de74988`)
- `simulator/open_loop.py`, `rate_search.py`, `stability.py`,
  `arrivals.py`, `loadgen_worker.py`: sessions arrive as a Poisson
  process at rate λ; capacity is the rate at which the engine's queue
  turns divergent (Mann-Kendall × Theil-Sen verdict per window). Two
  knees, λ_max and λ_sla; concurrency is derived from λ_sla × mean
  session length. Load generation shards across worker subprocesses
  and reports `client_limited` when the generator, not the engine,
  gives out. `docs/algorithm.md` §0 is the reference.
- Schema v7 carries the open-loop columns; the read-only export path
  tolerates pre-v7 DBs (`1414934`).
- The UI runs open-loop only; runs are deletable and orphaned runs no
  longer read "running" (`5d09e8e`). Results became a run list with
  click-to-open and check-to-compare (`7a0eb33`), rendering a narrative
  report — a headline that answers, sections that explain (`d41a1ea`,
  `40f2fb2`, `53bef2e`).

### 5.2 Workloads as a designer, headline (saturation) benchmarks
- Graphical persona/cohort designer replaces the YAML editor; workloads
  carry human names; client-limited windows are reported honestly
  (`b44dcda`, `ea6e122`). The tab is **Edit workloads**.
- Headline stress generators — the vendor-convention number, measured
  honestly: fixed concurrency stepped up a ladder, steady-state guard
  before measuring, engine counters as the source, pinned 128/128
  prompt/output shapes (`5c7dac6`, `7b3b54a`, `06f2d95`, `cd94b4d`,
  `95f3626`, `d5bfb4a`). A shape search finds the joint concurrency ×
  throughput optimum per model family and saves it (`7d187de`,
  `fa80efd`, `5b920a8`); one button searches engine and shape together
  (`efb6f3a`). Live saturation curves and a headline view that survives
  the end of the run (`183ccb3`, `6266710`, `d336358`).

### 5.3 Engines beyond vLLM
- TensorRT-LLM as a first-class target (`012e283`), SGLang on GPUs
  (`1535316`), KTransformers with MoE experts on CPU and attention on
  GPU (`c31b592`), multi-replica CUDA vLLM. One memory knob translated
  per engine and then checked (`381ef52`); nvfp4 KV cache as a third
  precision (`5687681`); opt-in trust-remote-code recorded on the run
  (`89ea5da`); real preflight, a runaway-error fuse and a
  stale-container sweep (`8cdbda4`); the engine's own error surfaces in
  a startup failure (`99d0b02`, `c5a9838`).
- Field notes recorded in `engine_notes.py` and the commit trail:
  stock `trtllm-serve` is not TRT-LLM's fast path (`dfb02dc`); CUDA
  graphs and chunked prefill are charged against the KV pool
  (`d2779be`, `526942c`); MoE is unserviceable on SM120 and CUTLASS does
  not avoid DeepGEMM (`b2aed30`, `7a86433`); iteration stats must be on
  or nothing is measured (`1a8c66c`); SGLang needs a port range per
  launch, not per replica (`4ca8829`, `9218284`, `ae9cf88`).

### 5.4 Prepare, arena, guided search, roofline
- **Prepare** tab: doctor, storage location, stage model weights, stage
  engine runtimes — an engine appears downstream only once its image is
  on the box (`394fba6`, `905fb6b`). Model-first benchmark form where
  the optimized launch is the default (`b56f8e2`, `2907d98`).
- **Optimize**: the engine is a search dimension, not an assumption
  (`3b44650`); a search aims its own grid (`93b6da4`); arena cards carry
  the day's tuning evidence (`5afed69`, `1d7fbc3`); the optimizer ladder
  climbs to each candidate's own batch capacity with a live heartbeat
  through engine launch (`15d5d8d`, `b51ed2a`); it measures what it
  claims to measure (`66c7e36`), and levers now reach the benchmark and
  roofline requests (`11f5221`).
- **Roofline** autopilot — what is the most this box can do? — across
  models × engines × shapes, with a measured KV cost outranking a
  guessed one, its custom block as a template, liveness from the run
  registry, and no retries of cells that cannot succeed (`d7cf30c`,
  `92a536e`, `806d2f3`, `f2871f6`, `2ae931f`). Benchmark split into
  Roofline and Workload tabs (`ee3dc82`).
- Service: run starts serialized (`e3aa58a`); UI assets
  must-revalidate so a deploy is never invisible (`e30857e`).

### 5.5 Review and the honest baseline (2026-09-19)
- A full review of accuracy, completeness, UX and robustness at
  `2ae931f`: [improvement_plan.md](improvement_plan.md) (`f7f66fa`).
- Step 0 of that plan: lint green, `doctor.json` moved into `runs/`,
  dead config paths fixed, version synced, `superseded` added to the
  export schema (`33b33e1`); `database_schema.md` generated from the
  DDL and tested for staleness (`cb07424`).

## Open items

The authoritative list is [improvement_plan.md](improvement_plan.md);
it cites the line every finding was verified at and orders the fixes
into steps that each leave `main` green. In outline:

- **Accuracy (A):** the open-loop revert, worker-death detection,
  warmup vs session settling time, autocorrelated stability samples,
  and the optimizer/benchmark launch-path drift.
- **Completeness (B):** CLI/YAML reach for open-loop (`--mode`,
  `simulation.mode`), docs parity — this document, the README and
  `deploy.md` are Step 4 of the plan.
- **UX (C) and robustness (D):** dead controls, empty states,
  secrets in logs, the non-loopback bind, run-lifecycle gaps,
  event-loop stalls, repo hygiene.
- **Real hardware:** the acceptance run on the XE7740 — install →
  doctor → smoke → ready → an open-loop run, a scaled-out worker
  window, a TRT-LLM lever sweep, a roofline restart with a changed
  prompt length — exercises everything the mock cannot.

---

## Validation & test tiers (applies across phases)

| Tier | What | Where |
|---|---|---|
| Unit | distributions, stepping/CI math, timeline attribution, export shaping | CI, no hardware |
| Contract | every export validates against JSON Schema | CI |
| Integration | mock engine end-to-end: run → db → export | CI |
| Smoke | `capsim smoke` — tiny model, real engine, real telemetry, ~10 min | every new host, post-install |
| Full | profile sweep with knee + attribution | benchmark hosts |

## Sequencing rationale

- Phase 0 first because the landing/validation story is needed *now* for new
  Xeon and GPU boxes, and the schema contract must exist before more consumers
  (UI, downstream apps) latch onto the export.
- Phase 1 before the service/UI: GPU support only touches engines/collectors/
  config, and every later phase benefits from targets + profiles existing.
- Phase 2 before 3: the UI is a client of the service and bus; the mock engine
  from 2.3 is what makes UI development fast off-hardware.
- Phase 4 last: data-driven personas and polish are valuable but nothing else
  blocks on them.
