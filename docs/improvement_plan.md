# capsim improvement plan

Review date: 2026-09-19, at commit `2ae931f`. Scope: accuracy of the
measurement and its claims, completeness of docs and surfaces, user
experience of the CLI and web UI, and overall functionality. Findings
were produced by reading the code, running the suite, driving the UI
against the mock profile in a browser, and reproducing service
behaviour against a spare instance. Every item below cites the line
it was verified at; nothing is speculative.

## Where the project stands

| Check | Result |
|---|---|
| `pytest tests/` | 522 passed, 1 skipped, 50 s |
| `ruff check simulator tests scripts` | 19 errors (17 auto-fixable) — CI's lint step is red on `main` |
| `capsim doctor` on a Mac | usable; degrades cleanly to skips |
| Mock open-loop run from the UI | starts, streams live telemetry, stops cleanly, lands in Results |
| Tracked repo size | 44 MB, of which 41 MB is two JSON artifacts |

The tool works end to end. The problems are concentrated in four
places: the open-loop methodology does not yet do what the docs say
it does, the docs describe a smaller product than the one shipped,
the arena/optimizer path has drifted from the benchmark path, and the
UI has a handful of dead or misdirected controls.

---

## A. Accuracy — the measurement and its claims

These change the numbers the tool reports, so they come first.

### A1. Open-loop revert does not abort in-flight requests (high)
`docs/algorithm.md` §0 says an overshoot is reverted by cancelling the
excess sessions "inside the engine, so the queue falls back to the
stable density". `arrivals.py:150-167` only sets a cancel event, and
`virtual_user.py:295-302,515` checks it between turns; `streaming.py`
has no cancel path. Excess requests stay queued until they finish or
time out, the revert waits up to `open_loop_drain_timeout_s` for a
naturally draining queue, and the next bisection window starts from a
backlog.
**Fix:** propagate cancellation into the streaming consumer (cancel the
HTTP task, not just the loop) and assert in a mock test that queue
depth drops within a few seconds of `trim_active`.

### A2. A dead load-generator worker is never noticed (high)
`open_loop.py:244-262`: `_read_loop` returns silently on EOF, the
return code is never checked, and the last stats stay frozen. Offered
load silently becomes λ·(k−1)/k while `arrival_rate_per_min` records
λ, so a window can read "stable" at a rate the engine never saw.
`loadgen_worker.py:183-202` also swallows task exceptions until stop,
so a worker can keep generating at a stale rate and ignore commands.
**Fix:** race `proc.wait()` with the read loop, mark the window
`client_limited`/`error` on worker death, and use
`asyncio.wait(FIRST_EXCEPTION)` in the worker.

### A3. `superseded` breaks the export contract (medium-high)
`open_loop.py:933` writes `stability="superseded"` when workers scale
out; the schema enum at `export_schema/buyer_page_data.schema.json:460`
allows only `stable | divergent | client_limited | null`. Any run that
scaled out fails `validate_export`, which is the `capsim smoke` gate.
**Fix:** add the value to the schema (bump `EXPORT_SCHEMA_VERSION`) and
add a contract test that exercises a scaled-out mock run.

### A4. Warmup is capped below the session settling time (medium-high)
`open_loop.py:589-595` caps warmup at 300 s, but the population
settles over one mean session length (1000–2000 s for `code_assist` /
`document_qa`). The window then measures the settling ramp, which the
Mann-Kendall test reads as divergent.
**Fix:** either warm up for ≥ one mean session duration (with a cap
documented as a known limit), or detect settling (population within
±5 % of λ·W for N seconds) before opening the window.

### A5. Stability test on autocorrelated samples (medium)
`stability.py:98-132` is formula-correct but assumes independence; on
stationary AR(1) queue series with ρ≈0.9 roughly 5 % of stable windows
read divergent and ~25 % go inconclusive. **Fix:** pre-aggregate to
5–10 s means or apply the Hamed–Rao effective-n correction; keep the
existing unit tests and add a stationary-AR(1) false-positive test.

### A6. Smaller measurement biases (medium)
- Marginal windows count as SLA fail (`open_loop.py:820`,
  `rate_search.py:196`), so λ_sla is biased low and the "< 5 %
  violation" wording in `rate_search.py:9` and the docs is wrong.
- One failed metrics scrape re-arms the completions-only fuse
  (`open_loop.py:615,628-632`) — the exact false-abort the docstring
  says was fixed. Keep the last good token baseline.
- `mean_session_duration_s` includes trimmed/cancelled/errored
  sessions (`arrivals.py:269-273`) and feeds the Little's-law
  concurrency headline (`export.py:791-795`). Exclude non-natural ends.
- Practical-significance floor uses client in-flight including queued
  requests instead of the engine's running batch
  (`open_loop.py:727-743`); `num_running` is scraped but unused.
- Windows censor the slow tail and pre-first-token failures carry
  `tpot_ms=0` (`open_loop.py:700`, `measurement.py:312`,
  `virtual_user.py:405-410`), making knee-side latency optimistic.
- Open-loop landing zones and bottleneck sort by `target_pool_size`,
  which for open-loop is `round(active_sessions_mean)` — non-monotone
  in λ (`export.py:989-994,1149-1160`). A divergent window can vanish
  and the export says `none_observed`.
- `adaptive.py:274-282`: the 0.50 stop branch is unreachable behind the
  0.30 fail check; docs say doubling stops at 0.50.

### A7. Optimizer path has drifted from the benchmark path (high)
- `scripts/engine_optimizer.py` never applies the TensorRT-LLM levers
  that `search.py` enumerates as dimensions (zero references in the
  script). An arena search spends up to 32 evaluations per shape on
  identical launches. Commit `11f5221` fixed the benchmark/roofline
  path only.
- `engine_optimizer.py:1310-1315,1360` passes vLLM's
  `gpu_memory_utilization` untranslated as TRT-LLM
  `free_gpu_memory_fraction` and SGLang `--mem-fraction-static`; the
  `vram.to_engine_fraction` translation exists precisely for this.
  `arena.FIXED_GMU=0.92` is shown in the UI but never written into the
  space doc.
- `search.py:329-345` `normalize()` does not collapse engine-foreign
  levers, so the coverage sampler chases duplicate keys.
**Fix:** route the optimizer's launch through the same
`_build_custom_config` / engine classes the service uses, then delete
the third copy of mount/GPU/token argv construction.

### A8. Roofline resume key is too coarse (medium-high)
`roofline.py:264-266` keys cells on (model, engine, max_num_seqs,
output_tokens); input tokens, memory fraction, KV dtype and levers are
not in the key, the UI never sends `resume=false`, and `new_run` is
ignored. Changing prompt length and pressing Start reuses old cells.
Also `roofline.py:226-261` writes off cells permanently on two startup
timeouts, and the error signature carries the replica index so the
same failure from `replica 3` is "different".

---

## B. Completeness — docs, CLI and surface parity

### B1. README first-run commands fail (high)
`README.md:39,40,189,204` and `Makefile:4-5` reference
`config/r7735_sglang_qwen3_30b_a3b.yaml`, which does not exist. Every
copy-pasted quick-start fails. The `install.sh` "next steps" say
`--config` while the README says `--profile`.

### B2. README describes a smaller product (high)
- UI: README says three views (Run control, Live telemetry, Results);
  `ui/index.html:17-23` has six (Prepare, Optimize, Roofline,
  Workload, Results, Edit workloads).
- Methodology: README never mentions open-loop, yet it is the default
  for every UI run (`service.py:105`) and `docs/algorithm.md` calls it
  primary.
- Engines: `trtllm`, `sglang_cuda`, `ktransformers`, `vllm_cuda_multi`
  are registered (`engines/__init__.py`) and undocumented.
- Layout: 24 of 66 modules listed; the whole open-loop stack, the
  service, roofline/arena/headline, and `collectors/` are absent.
- Make targets `audit`, `spot-check` are undocumented.

### B3. Open-loop is unreachable from the CLI and YAML (high)
`capsim run` / `run-persona` / `sweep` are hard-wired to the closed
loop (`cli.py:146,183`); `mode` exists only as an HTTP field. The docs
present `mode: "closed"` as a YAML knob, but `config._merge_dataclass`
(`config.py:429-437`) silently drops unknown keys — which also hides
three stale `stabilization_*` keys in `config/default.yaml:37-39`.
**Fix:** add `--mode open|closed` to the CLI and a `simulation.mode`
field; make `_merge_dataclass` warn (or fail) on unknown keys.

### B4. Schema doc is six versions behind (high)
`docs/database_schema.md` documents the v0 columns; `database.py` is at
`SCHEMA_VERSION = 7`. Roughly 50 columns (all open-loop fields, GPU
aggregates, `collectors_json`, `mode`) are missing, there is no
versioning section, `final_status` omits `cancelled`/`error`, and line
225 contradicts lines 107-111.

### B5. Other doc drift (medium)
- `docs/algorithm.md:443`: `max_pool_size` default 1024 vs code 65536;
  none of the 12 `open_loop_*` knobs are documented; personas listed
  as 6 (catalog has 8) and located in `personas.py` (now YAML).
- `docs/roadmap.md`: claims a `schema_version` table (it is
  `PRAGMA user_version`), names a `xeon-cpu-qwen3-30b` profile that
  does not exist, says "all phases complete" while roofline, arena,
  headline and open-loop belong to no phase. `docs/deploy.md` presents
  Xeon CPU-only as first-class but no curated CPU profile exists.
- `simulator/__init__.py` says `0.1.0`; `pyproject.toml` says `0.2.0`.
- `engine_notes.py:157-166` still says MoE needs CUTLASS named
  explicitly; commit `7a86433` corrected the lever but not the note.
- `docs/metrics.md` is linked from nowhere.

### B6. Host-specific config committed as the default (medium)
`config/arena.yaml` pins the XE7740's `device_groups`, and
`arena.hardware()` (`arena.py:91-112`) lets config win over detection.
On any other host the Optimize tab announces "8 GPUs · ? GB each · 2
domains". **Fix:** ship it as `arena.example.yaml`, and warn when
detection finds fewer GPUs than config claims.

---

## C. User experience

Observed in a browser against the mock profile, plus the CLI.

### C1. Results view mislabels a cancelled open-loop run (medium)
`app.js:1623` decides open-loop only if some point carries an arrival
rate; a cancelled run with one empty window falls through to the pool
ramp template and renders "Stable at every load tested, up to 0 users
— no collapse point observed". The export already carries
`methodology: "open_loop"`; use it, and add a "no measured windows"
empty state.

### C2. Dead and misdirected controls (medium)
- `app.js:433-437` injects a second `#goto-optimize`; the Workload
  tab's "run the optimizer →" link does nothing and stacks listeners
  on Prepare's button.
- Model download failures from Prepare are reported into `#opt-msg` on
  the Optimize tab (`app.js:3413`).
- `Editor.refreshLists`/`open` are unguarded (`app.js:3477,3499`); a
  failed GET leaves a blank editor.
- `pollStatus` swallows errors (`app.js:807`); when the server dies the
  pill stays "run active" forever. Add a "disconnected" state.
- `Optimizer.loadArena` caches forever (`app.js:2239`); models added in
  Prepare never appear until reload.
- `#hl-optimize` gets a new listener every 2 s (`app.js:713`).

### C3. Defaults and empty states (medium)
- Workload tab defaults to the first catalog entry
  (MiniMax-M2.7, 230 GB, not downloaded) and "CPU + GPU" on a host
  with no GPU. Default to a cached model, or the mock when nothing is
  staged, and pick the device mode from doctor.
- Idle Workload view shows five empty 0–1 charts and "waiting for
  telemetry…"; during warmup the TTFT/TPOT charts stay empty even with
  89 completed turns. Show a "no run" state, and plot warmup turns
  dimmed.
- Prepare never auto-runs doctor; the models table has no empty state;
  "gmu · mns · mbt" shorthand and the `wipefs`/`mkfs` block need
  labels or removal.
- Tab state is not in the URL; reload always lands on Prepare.
- Nav overflows at phone width (benchmark-box tool, low priority).

### C4. CLI (low-medium)
- `capsim doctor` writes `doctor.json` into the cwd and it is not
  gitignored; put it in the runs dir or add `--output`.
- `capsim --help` still says "Persona Capacity Simulator"; commands do
  not mention `serve` as the primary entry point or the open-loop
  default.
- The mock profile's "completes in minutes" comment is true only for
  closed loop; an open-loop mock run uses the 90 s warmup and 120 s
  windows from `config.py` defaults. Add `open_loop_*` overrides to
  `config/profiles/mock.yaml`.

### C5. Accessibility (low)
Tab buttons lack `role="tab"`/`aria-selected`; run rows, filesystem
chips and persona list items are clickable divs with no keyboard
path; hint text runs to ~10 px.

---

## D. Functionality and robustness

### D1. Secrets in logs (high)
`docker_replica.py:189-191,261` and `vllm_cuda.py:67` log the full
`docker run` argv including `-e HF_TOKEN=…`; `engine_optimizer.py:1459`
persists it into `run.json` as `failure_reason`. Redact env values
when logging argv.

### D2. Service on `--host 0.0.0.0` (critical if ever used)
Bind is 127.0.0.1 by default and that is documented, but the API
accepts: arbitrary container image, args and host mount via the
`custom` levers (`service.py:273-277,311`), any filesystem path as
`config`/`space` (`357-361`, `2115-2119`), and any directory as the RW
engine mount via `POST /api/storage` (`1378-1385`). Either refuse
`--host` other than loopback without an explicit `--insecure` flag, or
whitelist image names and confine paths to the repo/runs dirs.

### D3. Run lifecycle gaps (medium)
- `optimizer_start` is outside `start_lock` and awaits before `Popen`
  (`service.py:2071-2148`), so a run and an optimizer can still start
  together.
- Stop during engine launch orphans containers: the launch thread
  keeps running after `to_thread` is cancelled and `shutdown()` only
  sees replicas already appended (`open_loop.py:1193`,
  `runner.py:675,704`, `docker_replica.py:238`).
- `base.py:101-122` never terminates the CPU `vllm serve` process on a
  health-check failure; there is no host-process sweep.
- The stale-container sweep filters `name=vllm-` unanchored
  (`docker_replica.py:70-75`) and will remove a user's `my-vllm-dev`.
- A second `capsim serve` on the same runs dir stamps the live run
  `interrupted` (`service.py:512-534`).
- `sglang_cuda.py:59-76` picks a port window by `sha1(run_id) % 64`;
  consecutive launches collide 1/64 of the time, ~50 % per 48-cell
  roofline. Use a monotonic counter.

### D4. Event-loop stalls (medium)
`/api/runs` does synchronous sqlite per run dir (`service.py:917`);
`/api/models` reads whole download logs on every 2.5 s poll
(`1402`); export/search JSON is parsed on the loop (`2308,2280`). Move
to `to_thread` and tail-read logs.

### D5. Repo hygiene (low)
Two 20 MB JSON files in `artifacts/` (41 MB of a 44 MB tree); the
roadmap's git-lfs note was never acted on. `scripts/forklift_run.py`
is reachable only from its tests. `.venv` is Python 3.14 while CI
tests 3.11/3.12.

---

## E. Code health

- `service.py` is a 1,900-line closure; split into routers
  (`schemas`, `state`, `runs`, `catalog`, `prepare`, `optimizer`,
  `roofline`, `telemetry`, `app`). Settings on `app.state` so helpers
  stop capturing lexical variables.
- `ui/app.js` (5,000 lines) already loads as a module; split into
  `lib/{api,theme,tabs,events}.js` plus one file per tab, with a single
  `makeChart` factory replacing the ten `new Chart(` sites. Stays
  no-build.
- Duplicated logic: `vllm_cuda.py` re-implements what
  `docker_replica.py` centralises; `vllm_dual_socket.py` duplicates the
  log streamer and metric aggregation; the closed-loop and open-loop
  paths each carry their own turn aggregation, row mapping and
  finalisation (`measurement.py` vs `open_loop.py`, `runner.py` vs
  `open_loop.run_cohort_open_loop`).
- `open_loop._measure_window` is 320 lines; `runner.run_cohort` is
  425.

---

## Plan

Ordered so that each step leaves `main` green and each ships something
a user notices. Effort is a rough size, not a promise.

### Step 0 — Make CI honest (half a day)
1. `ruff check --fix`, fix the two manual items (`E741`, `B905`).
2. Add `doctor.json` to `.gitignore`; move the write into the runs dir.
3. Fix the four dead config paths in README and the Makefile header;
   align `install.sh` next-steps with `--profile`.
4. Sync `simulator.__version__` from `pyproject.toml` (or delete it).
5. Add `superseded` to the export schema enum, bump the schema
   version, add the scaled-out contract test (A3).
**Done when:** CI lint and tests are green on `main`, and a fresh clone
can paste the quick start.

### Step 1 — Stop leaking and stop lying (2–3 days)
1. Redact env values from logged argv and persisted failure reasons
   (D1). Test: a launch failure with `HF_TOKEN` set leaves no token in
   the log or `run.json`.
2. Results view keys on `methodology` and shows an empty state for
   runs with no measured windows (C1).
3. Fix the dead `#goto-optimize`, the misrouted download error, the
   unguarded editor fetches, the arena cache, and the listener leak
   (C2). Add a "disconnected" pill state.
4. Ship `config/arena.yaml` as an example; warn when config claims more
   GPUs than detected (B6).
5. Anchor the container sweep filter; add the host-process cleanup for
   CPU `vllm` (D3).
**Done when:** the mock walkthrough in a browser has no dead control
and no misleading verdict.

### Step 2 — Make the open-loop methodology do what the doc says (1–2 weeks)
In priority order, each with a mock-engine test:
1. Cancel in-flight requests on trim/drain (A1).
2. Detect worker death and surface it as `client_limited`/`error` (A2).
3. Warmup tied to session length or a settling detector (A4).
4. Keep the last good token baseline across failed scrapes (A6).
5. Treat `marginal` consistently and correct the "< 5 %" wording, or
   change the gate to match the wording (A6).
6. Exclude non-natural session ends from `mean_session_duration_s`;
   sort open-loop landing zones by arrival rate, not pseudo pool size
   (A6).
7. Aggregate queue samples before Mann-Kendall or apply the
   effective-n correction (A5); add the stationary-AR(1) test.
8. `--mode open|closed` on the CLI and `simulation.mode` in YAML;
   `_merge_dataclass` warns on unknown keys; remove the stale
   `stabilization_*` keys (B3).
**Done when:** `docs/algorithm.md` §0 is true sentence by sentence, and
a mock run with an injected worker death reports it.

### Step 3 — One launch path for benchmark, roofline and arena (1 week)
1. Route `scripts/engine_optimizer.py` through the service's
   `_build_custom_config` and the engine classes; delete its private
   argv builder. Levers and `to_engine_fraction` then apply
   everywhere (A7).
2. `search.normalize` collapses engine-foreign levers.
3. Roofline: include input tokens, memory fraction, KV dtype and
   levers in the cell key; honour `new_run`; do not write off cells on
   startup timeouts; strip the replica index from error signatures
   (A8).
4. Monotonic port windows for SGLang; `optimizer_start` under the start
   lock; cancel-safe engine launch that tracks replicas as they are
   created (D3).
5. `KTransformers` roofline defaults: one replica, no fp8 KV, consult
   `unsupported()` in the arena driver.
**Done when:** a TRT-LLM arena search with two lever values produces
two different container commands, and a roofline restarted with a new
prompt length measures new cells.

### Step 4 — Documentation catch-up (2–3 days)
1. README: six tabs, open-loop as the default methodology, all ten
   engine types, full module layout, `audit`/`spot-check`, links to
   every doc under `docs/`.
2. `docs/database_schema.md` regenerated from `database.py` at v7,
   with a versioning section (consider a small script that emits the
   table so it cannot drift again).
3. `docs/algorithm.md`: open-loop knobs table, corrected defaults,
   persona list from the YAML, the 0.30 stop threshold.
4. `docs/roadmap.md`: add a Phase 5 that names roofline, arena,
   headline and open-loop, and list the open items from this plan.
5. Either add a curated CPU profile or drop "Xeon CPU-only" from the
   first-class list in `deploy.md`.

### Step 5 — Structure and UX polish (ongoing, 1–2 weeks)
1. Split `service.py` into routers and `app.js` into modules (E).
2. Move blocking IO off the event loop (D4).
3. Workload tab defaults from doctor and the model cache; idle and
   warmup states for the live charts; tab in the URL hash; doctor
   auto-run on first visit (C3).
4. Hard-refuse non-loopback bind without `--insecure`, or confine
   image names and paths (D2).
5. Accessibility pass on tabs and list rows (C5).
6. Move the two 20 MB artifacts out of the tree or into git-lfs; delete
   or wire `forklift_run.py` (D5).

### What to run on real hardware after Step 3
The roadmap's open item still stands: the full install → doctor →
smoke → ready → sweep on the XE7740, now with an open-loop run, a
scaled-out worker window, a TRT-LLM lever sweep, and a roofline
restart with a changed prompt length. Those four exercise every fix
above that the mock cannot.

---

## Status — 2026-09-20

Implemented on `main` (commits `33b33e1` … this one), with the full suite
at 608 passed:

- **Step 0** — lint clean, `runs/doctor.json`, real config paths, one
  version, `superseded` in the export contract.
- **Step 1** — secrets redacted from logs and failure records; Results
  keys on `methodology` with an empty state; dead and misrouted
  controls fixed; `config/arena.yaml` is an example with a
  detection-mismatch warning; anchored container sweeps; CPU engine
  cleanup on health failure.
- **Step 2** — in-flight cancel on trim/drain; worker-death detection;
  token baseline across failed scrapes; settling detector before a
  window; 5 s binning plus AR(1) effective-n in the stability test;
  one SLA gate (Wilson upper < 5 %); natural-end session durations;
  engine `num_running` as the significance basis; open-loop landing
  zones ordered by arrival rate; TPOT excludes pre-first-token
  failures; `--mode open|closed` on the CLI and `simulation.mode` in
  YAML with unknown-key warnings.
- **Step 3** — the arena driver launches through the engine classes
  (levers and the memory-fraction translation apply); `normalize`
  collapses engine-foreign levers; roofline cells keyed on the full
  launch shape, `new_run` honoured, transient failures never written
  off; monotonic SGLang port windows; optimizer start under the lock;
  cancel-safe launches; KTransformers roofline defaults; doctor fails
  on a broken GPU stack.
- **Step 4** — README, deploy, roadmap rewritten; the schema doc is
  generated from the DDL and tested for drift; `docs/algorithm.md`
  matches the code it describes.
- **Step 5** — `simulator/service/` router package (identical OpenAPI
  document); `simulator/ui/` ES modules with one chart factory;
  blocking IO off the event loop; `--insecure` bind guard;
  accessibility pass; defaults from detected hardware.

Found and fixed while deploying to the XE7740: the roofline endpoint
required an engine template the UI never sent, and engines called the
Hugging Face hub at startup even with staged weights (fatal on a box
without outbound DNS) — containers now run with `HF_HUB_OFFLINE=1`
when the snapshot is complete.

Still open: the real-hardware acceptance runs listed above (the
roofline started 2026-09-20 exercises the offline launch, the
transient-failure rule and the KTransformers defaults); the registry
mode of `scripts/engine_optimizer.py` still uses its own vLLM argv
builder for CPU cpuset/NUMA shapes.
