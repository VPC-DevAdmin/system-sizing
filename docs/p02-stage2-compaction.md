# P02 stage 2: unbounded sessions through Xeon context compaction

Stage 1 (docs/p02-methodology.md) keeps sessions warm: the Xeon DRAM tier
restores evicted KV instead of the GPUs rebuilding it. Stage 2 keeps sessions
small: a summarizer on the Xeon rewrites the trailing turn history when a
session crosses a context threshold, so live context oscillates between a
target and a trigger instead of growing until the engine limit. The raw history
stays resident in host DRAM as text; it is never deleted and never re-summarized
from a prior summary.

Terminology: this is context compaction (the prompt text is rewritten and the
KV for the new summary is prefilled once). It is not KV compaction (shrinking
the cache in latent space without rewriting text), which has no production path
in vLLM or Dynamo today and is recorded as a stage 3 candidate at the end.

## Claims this stage is designed to support

1. X% more knowledge workers per server with unbounded session length, at the
   same quality and response SLO as stage 1. Expected around +33% at a 96K
   trigger and 48K target, because density follows mean live context across
   the population, not peak.
2. Sessions run indefinitely inside a fixed context budget: the sawtooth plot,
   several cycles per worker per window, no context-limit failures.
3. Session maintenance costs the GPUs nothing: native GPU prefill and decode
   counters attribute zero tokens to compaction; Xeon utilization and compaction
   lag show the host carrying it.
4. Recall is bounded and does not compound across compaction cycles: per-cycle
   recall on three probe types stays flat, measured before and after every
   compaction.

Turns per kWh remains owed until whole-server power exists (same gate as P02).
No Xeon-versus-other-CPU claim. No general writing or coding quality claim.

## Arms

| Arm | Live context | Compaction | Purpose |
|---|---|---|---|
| A, stage 1 baseline | steady ~96K mature history, slow growth | none | the P02 CPU-tier treatment as it stands |
| B, stage 2 | 48K to 96K sawtooth | Xeon summarizer, prefix-aligned, ledger | the headline |
| C, naive control (one cell) | 48K to 96K sawtooth | summary of previous summary plus new turns | shows the compounding loss the design avoids |

Both A and B run on the same Dynamo deployment, GPU KV allocation, CPU tier,
model, revision, precision, fixture, population ladder and think time. The only
difference is whether the client compacts. Arm C is a single cell at the
baseline population, not a ladder.

A grows about 1K tokens per turn from a 96K start; B grows about 7K per turn
from a 48K start. State this plainly in the report as "mature uncompacted
session versus compacted session". A same-growth variant where A starts a new
chat at 96K is a continuity claim, not a density claim, and is not run here.

## Fixture and session script

Model: GPT-OSS-120B, FP8 KV, the stage 1 headline model. Fixture: the padded
16-record reference document that passed the 96K diagnostic (header, padding,
records, question). The record block is the immutable reference prefix.

Each turn adds a work item of roughly 6K input tokens plus the question, and the
model answers in roughly 1K tokens. A work item is neutral padding plus a small
amount of session state that the probes below depend on (which records were
requested this turn, one field whose value is updated this turn). Starting at
48K the session crosses 96K after about seven turns. With the current 60 s mean
think time that is one compaction per worker every seven to eight minutes and
two to three complete cycles in a 1500 s window.

Spec changes: raise `session_new_input_tokens` above its current 2048 ceiling,
add `session_turns` large enough for three cycles (about 24), and add a
`session_compaction` block (trigger tokens, target tokens, mode none|xeon|naive,
summarizer endpoint, summary budget, ledger on/off).

## Compaction mechanism

Live prompt layout, in this order, always:

1. Reference prefix (records), byte-identical for the life of the session. Its
   KV stays cached in GPU or the CPU tier.
2. One summary per completed cycle, each written once and never rewritten.
3. Structured ledger: a small JSON block the summarizer must emit alongside the
   prose. For this fixture it holds the running list of record IDs requested so
   far and the current value of each updated field.
4. Recent raw turns since the last compaction.

Trigger: when a turn's prompt crosses 96K tokens, the client sends that cycle's
raw turns (about 48K) to the summarizer and continues the session without
waiting. Target: the compacted prompt is prefix plus summaries plus ledger plus
the most recent turn, about 48K. Budget: summary plus ledger at most 4K tokens.

The summary is generated from raw turns only. The input to the summarizer never
contains a previous summary. Arm C is the deliberate violation of this rule.

Asynchronous with lag: if the summary has not landed by the next turn, the
session keeps growing and the turn is tagged `compaction_pending`. Lag per
compaction and the fraction of pending turns are reported. Sessions that reach
the engine limit while pending are counted as failures, not hidden.

Prefix alignment is a hard requirement because CacheBlend non-prefix reuse is
broken with the vLLM V1 connector. After compaction the next turn prefills only
the new summary and ledger (about 4K) on top of the cached prefix; the pre- and
post-compaction prefill token counts are both recorded.

Raw archive: every turn's messages are kept in host memory keyed by session and
turn, and written to the campaign directory. Retrieval on demand from the
archive is not part of stage 2; it is the first thing to add if abstention or
session recall fails the gate.

Summarizer: a small instruct model on the two Xeon 6787P sockets, candidates
Qwen3-4B or Qwen3-8B, BF16 or INT8 on AMX, served by the vLLM CPU image already
wrapped by `simulator/engines/base.py` or by llama.cpp. One instance per socket,
pinned to that socket's cores and memory. The summarizer prompt is fixed and
recorded; it asks for prose plus the ledger JSON and names the fields the ledger
must carry.

## Execution

Phase 0, summarizer calibration (about one hour, no GPUs). Measure prefill and
decode rate per socket for 48K-token inputs and 4K outputs at concurrency 1, 4,
8 and 16. Compute compactions per second the host sustains. Required capacity at
96 workers is about 0.2 per second; at the expected 60 to 180 s per compaction
that is 12 to 36 in flight. Pick model size, precision and summary budget from
this. If one socket cannot carry the baseline population at under half
utilization, reduce the summarizer or chunk the input before any GPU time is
spent.

Phase 1, loop demonstration (one cell). Arm B at the stage 1 baseline
population, one 1500 s window. Deliverable: the sawtooth plot of live tokens
per turn for every worker with compaction events marked, zero context-limit
failures, compaction lag distribution, and the per-cycle recall table below.
This phase gates everything after it.

Phase 2, density ladder. Arms A and B over the stage 1 population ladder with
the stage 1 SLO grid, same arrival discipline (one outstanding turn per worker,
exponential think time), same overload rules (two overload points required,
final-window deadline failures at or above 5% is overload). Rank by workers
supported at the quality and SLO gate, not by peak throughput.

Phase 3, optional. Arm C at the baseline population for the recall contrast.
One deeper compaction cell (96K to 32K, or 64K to 32K) only if the density
curve is worth showing. Confirmation per stage 1 rules: five paired blocks,
600 s windows, frozen populations.

Order on xe7740: phase 0 can run while the current ladder-mid and ladder-control
campaigns finish, since it uses no GPUs. Never restart Dynamo during a run.

## Recall accuracy protocol

The protocol is the standard one from LongMemEval and the KV-compression
literature: a fixed probe set scored against the uncompacted ground truth,
reported per probe type and per compaction cycle, before and after each
compaction. No external benchmark harness; the existing exact-match grader is
extended with three probe types that rotate through the turn stream.

| Probe | Lives in | Expected | What it catches |
|---|---|---|---|
| Reference recall | prefix | 100% | the prefix was touched or evicted |
| Session recall and knowledge update | compacted history | flat across cycles | summarizer or ledger dropped or stale facts |
| Abstention | nowhere | "not discussed" | confabulation from a summary |

Reference recall is the current record-extraction question. Session recall asks
which record IDs were requested at a named earlier turn; knowledge update asks
for the current value of a field that was changed mid-session, scored against
the latest value. Abstention asks about a record ID or decision that never
occurred; a non-abstaining answer is scored wrong.

Pairing: the turn immediately before a compaction and the turn immediately after
it carry the same session-recall probe, so every compaction event produces a
before and after score. Cycle 0 is the uncompacted session and is the reference.

Reported as recall per probe type at cycle 0, 1, 2, 3 for arms B and C, with the
same table for arm A at matched turn counts. The quality gate for density
claims is the stage 1 gate (95% correct and within SLO) applied to all probe
types together. Arm C is expected to fail it; that is the result.

Scope statement for the report: the probes test recall of session state in a
synthetic extraction workload with a fixed summarizer prompt. They do not
establish general summarization quality, and the ledger is a fixture-specific
schema. Loss is bounded, not zero; the claim is that it does not compound.

## Metrics and evidence

Per turn: live prompt tokens, new input tokens, output tokens, prefill tokens
charged by the engine, TTFT, completion latency, probe type, correct, cycle
index, compaction_pending flag.

Per compaction: session, trigger tokens, raw tokens sent, summary tokens,
ledger bytes, summarizer wall time, lag to landing, Xeon socket.

Per window: workers at SLO, quality fraction, native prefix-cache hits split
GPU and CPU (same counters as stage 1), engine KV usage gauge (should average
lower in B), Xeon utilization per socket, GPU utilization, compactions per
minute, pending-turn fraction, context-limit failures.

Evidence: campaign manifest, summarizer image and model fingerprints, the fixed
summarizer prompt, the raw archive, every summary and ledger, every probe with
expected and actual answers, engine counters before and after each window.

## Gates that block a claim

- Phase 0 shows the host cannot sustain the baseline population's compaction
  rate: no density ladder until the summarizer is resized.
- Any context-limit failure in arm B: claim 2 fails.
- Reference recall below 100% in arm B: the prefix was disturbed; stop.
- Session recall or knowledge update falling across cycles in arm B beyond the
  stage 1 quality gate: claim 4 fails; add archive retrieval and re-run phase 1.
- Prefix-cache hit fraction after compaction below the stage 1 level: the layout
  is not prefix-aligned; stop.
- Pending-turn fraction above 10%: the loop is not keeping up; the density
  result is reported with that caveat or the summarizer is resized.
- Single blocks never produce uplift percentages; same rules as stage 1.

## Harness changes

1. `knowledge_workers.py`: work-item turns, three probe types, cycle index,
   before and after probe pairing, compaction call-out instead of the
   context-limit ValueError when compaction is enabled, raw archive writes.
2. New `session_compaction.py`: summarizer client, prefix-aligned layout,
   ledger parse and validation, async compaction with lag accounting, naive mode
   for arm C.
3. `spec.py`: `session_compaction` block, higher `session_new_input_tokens`
   ceiling, `session_turns` to 32.
4. Summarizer launch: reuse the CPU vLLM or llama.cpp engine wrappers with
   socket pinning; record fingerprints in the campaign manifest.
5. Report: sawtooth plot, per-cycle recall table, compaction lag, density ladder
   alongside the stage 1 report.

## Stage 3 candidates, not promised

KV compaction in latent space (attention matching, Still) on KV already resident
in the Xeon tier; archive retrieval into the live prompt; model-side sparse
attention (DeepSeek DSA class) as a headline-model change. Each needs engine
support or a model change that stage 2 does not depend on.
