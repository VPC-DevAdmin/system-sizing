# Test backlog: runs that still need to be run or re-run

Tests we owe, and why. Keep entries until the run is done and its
results are handed over; then move them to **Done** with the date and
where the results live. Newest decisions first.

---

## RE-RUN REQUIRED

### 1. XE7740 GPU persona runs for the AI capacity planner — full re-run

**Status:** NOT STARTED. Every existing file is superseded.

**Why a re-run:** runs made before commit `ff08430` (2026-09-26)
understate capacity by up to 2x. In the open-loop search, the
narrowing steps between the last passing rate and the first failing
one started on the backlog the failing step left behind:

- the rig fell back only after *divergent* steps, not after steps
  where every request timed out (the queue looked stable because
  timeouts drained it);
- when it did fall back, it waited for the engine's *waiting* queue
  but not its *running* requests (code_assist at 8 GPUs kept
  3,000-4,300 requests running at 250-275 s TTFT through every
  narrowing step).

So nearly every persona reported its last power-of-two passing rate as
capacity. It shows up as capacity that doesn't scale with GPU count
(code_assist read lower at 8 GPUs than at 4). The 1-GPU Qwen3-30B run
also aborted after its first persona: the error fuse ended the whole
sweep on that backlog.

Fixed in `1fd3680` (a hopeless step is a ceiling; a fuse abort ends
one persona, not the sweep) and `ff08430` (the rig falls back after
hopeless steps and waits for running load to recover). Validation:
conversational, Qwen3-30B, 2 GPUs, run `runs_ai/run_09` on the box
(started 2026-09-26). Check its narrowing steps pass before launching
the full re-run.

**What to run:** both models, each at 1, 2, 4 and 8 GPUs, all six
planner personas (`quick_lookup, conversational, writer, document_qa,
code_assist, long_form_generator`).

| Setting | Value |
|---|---|
| Models | `Qwen/Qwen3-30B-A3B-Instruct-2507-FP8` first (the only model the planner imports), then `nvidia/Qwen3.6-35B-A3B-NVFP4` |
| Engine | vLLM, TP 1, one replica per GPU, `spread` placement |
| Context | `max_model_len` 32768. 8K rejected 24% of document_qa, 12% of code_assist and 1.4% of long_form_generator prompts. The R470 reference served them at the engine's default context. |
| Memory | `gpu_memory_utilization` 0.90 |
| Method | Open-loop (`--mode open`). Generator starts at 4 workers per GPU, capped at 8 per GPU (`simulator/ai_run.py`). |
| Host identity | `meta.system`: memoryGb **2048** (installed), Intel:6787P, 2 sockets, 8x RTX PRO 6000 96 GB |

**How:** `simulator/ai_run.py` writes the configs, adds `meta.system`
and `meta.personas`, writes the sidecar, and checks the importer's
rules. The box driver pattern is `/data/capsim/run_ai_queue.sh`
(sweep, then `capsim export --slim`, then `ai_run finalize`). Output
goes to `/data/capsim/exports/ai_runs/<model>_<n>gpu.json` plus
`.system.json`.

**Estimated time:** about 10-12 h per GPU count at 8 GPUs, less at
fewer. Roughly 2 days for the Qwen3-30B set, about 4 days for both.

**Superseded files, keep for reference only (do not import):**
`/data/capsim/exports/ai_runs/qwen3-30b-a3b-instruct-2507-fp8_{8,4,2}gpu.json`,
`qwen3.6-35b-a3b-nvfp4_8gpu.json`, `superseded_8k_qwen3-30b_8gpu.json`,
`superseded_aborted_qwen3-30b_1gpu.json`. Copies of the first four are
in `exports/ai_runs/` locally. Rename them `superseded_*` when the
re-run starts.

**Done when:** four Qwen3-30B files validate as `importable` with no
`floor:` lines, capacity rises roughly with GPU count, and the files
are handed over for `import-ai-run.mjs`.

---

### 2. R470 CPU reference run — re-run on the fixed rig

**Status:** NOT STARTED. Needs the R470 (a separate box).

**Why a re-run:** the planner's CPU capacity comes from the R470
six-persona run on the same rig and search code, so it very likely has
the same backlog bug (see item 1) and understates CPU capacity by up
to 2x. CPU and GPU plans only compare fairly if both are measured with
the fix (`ff08430` or later).

**What to run:** the same six-persona sweep as the original R470
reference, on the same model (`Qwen/Qwen3-30B-A3B-Instruct-2507-FP8`).
Match its original context setting. The export in
`artifacts/buyer_page_data.json` (2026-05, SGLang) records no
`max_model_len`, so it ran at the engine default. Open-loop, the same
personas and SLAs.

**Done when:** the new R470 file imports and replaces the old CPU
reference in the planner.

---

## OUTSTANDING (not re-runs, still owed)

- **Why the first 8K run's quick_lookup read ~60% low** (8-GPU,
  2026-09-23: 1,680 vs 3,810 sessions). The likely cause is the same
  backlog bug plus the old one-worker-at-a-time generator. Confirm from
  `runs_ai/run_01` step data; no GPU time needed.
- **KTransformers placement test, not run** (`simulator/kt_placement.py`,
  plan `/data/capsim/kt_placement_full.json`; stopped 2026-09-23 when the
  roofline resumed): calibrated placement on the code-heavy and
  Chinese-heavy mixes, TP4 x DP1 with calibrated placement, and SGLang
  TP8 NVFP4 on the same prompts. Results so far: uniform vs calibrated
  1.38x at 64 and 128 streams
  (`runs/kt_placement/kt_placement.json`).
- **KTransformers capacity tests from the expert-residency review**
  (phase 3): calibrated placement at TP8 spanning both banks (~370
  experts/layer on GPU) vs DP2; the GPU prefill threshold; the fork's
  dynamic expert update; domain-routed bank specialisation.
- **Phase 2b / 2c from the review:** single-stream per-request routing
  traces (about 40 min GPU), and measuring the cost of each GPU-to-GPU
  and memory-to-GPU path (about 20 min GPU).
- **llama.cpp throughput is wave-quantized.** It has no live
  throughput gauge, so its rates count finishing requests in waves.
  The export flags those rates and gives the timing-implied rate beside
  them. A real fix needs a per-token signal from llama.cpp.

---

## Done

- 2026-09-24: XE7740 roofline giants pass finished; gpt-oss-20b
  re-confirmed at 105,890 gen tok/s after the stop-rule fix (`e67fc1f`).
  Sizing export in the UI (Roofline tab → Sizing export).
