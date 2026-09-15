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
- `schema_version` table in `run.db` + ordered migration list in `database.py`.
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
  requirements, e.g. `xeon-cpu-qwen3-30b`, `xeon-gpu-qwen3-30b`,
  `xeon-gpu-gptoss-120b`. `capsim doctor` maps a host to candidate profiles;
  `capsim ready --profile X` does the rest. Existing yaml configs become
  profiles.

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
- Three views:
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

## Phase 4 — Personas as data + productization polish

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
