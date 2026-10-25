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

## The claims

Three density numbers, all in the stage 1 headline shape, more long-session
knowledge workers per server at the same quality and response SLO:

1. Xeon compaction alone: arm C over arm A.
2. Xeon warm cache alone: arm B over arm A. This is the stage 1 result.
3. Both together: arm D over arm A, with D over B as the stage 2 increment.

Density follows mean live context across the population, not peak. All four
arms run the same growing script; a no-compaction session reaches the engine
limit and starts a new session, so it oscillates between 48K and about 127K,
mean roughly 84K. The compaction band sets the expected gain:

| Compaction band | Mean live context | Expected gain over new-session arms |
|---|---|---|
| 96K to 48K | ~72K | ~1.17x |
| 96K to 32K | ~64K | ~1.31x |
| 64K to 32K | ~48K | ~1.75x |

Default band: 96K trigger, 32K target. Compaction alone (C) is expected to
gain more than the increment D over B, because without the Xeon tier every
evicted session is rebuilt by full GPU prefill and compaction shrinks that
rebuild as well.

Second claim, session memory survives: at the end of the run, recall of
session facts is X% after N compactions with Xeon compaction, versus Y% after
M context resets without it, and the curve over the run shows why. The
reported figures are the per-cycle curve and the end state, which are what a
user experiences; a window average blends early intact history with late lost
history and is not reported as a headline. The
same probes run in all four arms on the same timeline as the context plot, and
they sample the worker's entire history, not the live window. In A and B,
recall of everything before a reset is zero for the rest of the session, so
whole-history recall steps down at each reset and keeps falling as more of the
history sits behind one; reference recall stays at 100% because the prefix is
kept, which shows it is the history that was lost, not the document. In C and
D, whole-history recall stays flat across compactions. The claim is bounded,
non-compounding loss, never zero loss.

The summary prefill after each compaction is a cost C and D pay and the ladder
includes it. Compaction wins only if the KV it frees is worth more than the
prefill it adds. That is the question under test, not a confound.

Supporting lines under the headlines:

1. Sessions never hit the context wall: the sawtooth plot, several cycles per
   worker per window, zero context-limit failures in C and D.
2. Why density improves: lower engine KV occupancy at the same population, the
   small summary prefill per cycle shown beside it, and compaction running on
   the Xeon with no GPU tokens except that prefill.

For C and D, recall per cycle is also a pass/fail gate on the density claim,
the same way quality is a gate in stage 1.

Turns per kWh remains owed until whole-server power exists (same gate as P02).
No Xeon-versus-other-CPU claim. No general writing or coding quality claim.

## Arms

| Arm | Xeon KV tier | Xeon compaction | Live context | At the limit |
|---|---|---|---|---|
| A | no | no | 48K to ~127K | new session, history dropped |
| B | yes | no | 48K to ~127K | new session, history dropped |
| C | no | yes | 32K to 96K sawtooth | never reached |
| D | yes | yes | 32K to 96K sawtooth | never reached |

All four arms are new runs on the same Dynamo deployment, GPU KV allocation,
model, revision, precision, reference prefix, growing turn script, population
ladder and think time. Stage 1 held a flat 96K history, so its A and B results
are not reused here. A and B use GPU-only prefix caching versus the CPU tier
exactly as stage 1 defines them; C and D add the client-side compaction loop.

No-compaction policy at the limit: when the next turn would exceed the engine
limit, the worker keeps the reference prefix, drops the history and continues
as a new session. This is what happens today without compaction and keeps all
four arms on identical turns. Letting A and B fail at the limit is not used; it
would widen the gap with a strawman.

In D, after each compaction the next request restores the prefix and earlier
summaries from CPU KV exactly as B does, then prefills the new summary and
ledger. In C, an evicted session is rebuilt by full GPU prefill of the
compacted prompt, which is shorter than A's rebuild. Compaction trades a small
prefill per cycle for lower KV residency, smaller rebuilds, less tier traffic
and cheaper decode at shorter context. The ladder nets those out; the per-turn
cost accounting in the report explains the result.

| Per compaction, per worker | Prefix-aligned | Whole prompt rewritten |
|---|---|---|
| GPU prefill tokens | ~4K | ~48K |
| GPU time at ~15.7K tok/s | ~0.25 s | ~3 s |
| Share of one GPU's prefill at 96 workers, 0.2 compactions/s | ~5% | ~60% |

The right-hand column is why prefix alignment is a hard requirement.

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
two to three complete cycles in a 1500 s window. A no-compaction session
reaches the engine limit after about eleven turns and resets, so A and B show
one to two resets per worker per window.

Spec changes: raise `session_new_input_tokens` above its current 2048 ceiling,
add `session_turns` large enough for three cycles (about 24), add a
`session_compaction` block (trigger tokens, target tokens, mode none|xeon,
summarizer endpoint, summary budget, ledger on/off), and a
`session_limit_policy` of new_session for the no-compaction arms.

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
raw turns (about 64K) to the summarizer and continues the session without
waiting. Target: the compacted prompt is prefix plus summaries plus ledger plus
the most recent turn, about 32K. Budget: summary plus ledger at most 4K tokens.

The summary is generated from raw turns only. The input to the summarizer never
contains a previous summary.

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

Phase 1, loop demonstration (one cell). Arm D at the stage 1 baseline
population, one 1500 s window. Deliverable: the sawtooth plot of live tokens
per turn for every worker with compaction events marked, zero context-limit
failures, compaction lag distribution, and the per-cycle recall table below.
This phase gates everything after it.

Phase 2, density ladder. All four arms over the stage 1 population ladder with
the stage 1 SLO grid, same arrival discipline (one outstanding turn per worker,
exponential think time), same overload rules (two overload points required,
final-window deadline failures at or above 5% is overload). The result per arm
is the largest population supported at the quality and SLO gate. The headlines
are the ratios C/A, B/A, D/A and the increment D/B. Per correct turn the report also records GPU prefill tokens
charged by the engine, decode tokens weighted by live context, mean engine
KV-usage gauge, restore and offload blocks, and summary prefill tokens per
cycle, as the explanation under the headline.

Phase 3, confirmation and optional depth. Confirmation per stage 1 rules at the
frozen populations: five paired blocks, randomized arm order, 600 s windows.
One alternate band cell (96K to 48K, or 64K to 32K) only if the curve is
worth showing.

Order on xe7740: phase 0 can run while the current ladder-mid and ladder-control
campaigns finish, since it uses no GPUs. Never restart Dynamo during a run.

## Recall accuracy protocol

The protocol is the standard one from LongMemEval and the KV-compression
literature: a fixed probe set scored against the uncompacted ground truth,
reported per probe type and per compaction cycle, before and after each
compaction. No external benchmark harness; the existing exact-match grader is
extended with three probe types that rotate through the turn stream. The
probes are identical in all four arms; in A and B the event that pairs the
before and after scores is the context reset instead of a compaction.

| Probe | Lives in | Expected | What it catches |
|---|---|---|---|
| Reference recall | prefix | 100% | the prefix was touched or evicted |
| Session recall and knowledge update | compacted history | flat across cycles | summarizer or ledger dropped or stale facts |
| Abstention | nowhere | "not discussed" | confabulation from a summary |

Reference recall is the current record-extraction question. Session recall asks
which record IDs were requested at an earlier turn; knowledge update asks for
the current value of a field that was changed mid-session, scored against the
latest value. Abstention asks about a record ID or decision that never
occurred; a non-abstaining answer is scored wrong.

Ground truth follows the virtual user, not the live window. The harness keeps a
fact log per worker: every record request and every field update, with the turn
it was introduced. Session-recall and knowledge-update probes draw their target
turn uniformly over the worker's whole history since the session began,
including turns that are behind a reset in A and B or behind a compaction in C
and D. A probe about a fact the worker has lost is scored wrong, never skipped.
This is what makes the metric honest: after a reset, A and B cannot recover the
facts behind it no matter how much new history accumulates, and the score has
to show that. Probing only recent turns would hide it.

Three metrics from the same probes. Whole-history recall counts only session
recall and knowledge update, the probes whose answer is a session fact; it
declines in A and B as the share of facts behind a reset grows and holds in C
and D at the summarizer and ledger retention rate. Abstention is reported on
its own line: it is a confabulation check, and after a reset "nothing was set"
is the correct answer, so counting it as recall would flatter an arm that has
just lost its history. Intact versus behind a reset: recall of facts newer
than the worker's last reset against facts older than it, which is the
cleanest statement of what a reset costs (measured in arm B at 56 workers:
0.77 intact, 0.00 behind). Recall by fact age separates "lost at the reset"
from "lost in the summary".

A window average understates the reset cost when most of the window is
pre-reset: from a 24K start a session takes about 50 minutes to reach the
engine limit, so a 60-minute window is four fifths intact history. Report the
split, not the average.

Pairing: the turn immediately before a compaction or reset and the turn
immediately after it carry probes about the same earlier facts, so every event
produces a before and after score. Cycle 0 is the uncompacted session and is
the reference.

Benchmark alignment: LongMemEval and LoCoMo score recall over the whole
history but ask at the end of the conversation; the streaming memory
benchmarks (StreamMemBench, incremental multi-turn evaluation) interleave
probes as history grows, which is the shape used here. RULER and needle tests
are single-prompt and do not apply.

Reported two ways. First, a recall timeline per arm: whole-history recall per
turn plotted under the live-context plot, so the resets in A and B and the
compactions in C and D line up with their recall effect, and the staircase
decline in A and B is visible against the flat line in C and D. Second, a
table of recall per probe type and by fact age at cycle 0, 1, 2, 3 for C and
D, and before and after each reset for A and B. The quality gate for the density claim in C and D is
the stage 1 gate (95% correct and within SLO) applied to all probe types
together. A and B are not gated on session recall, since losing it is the
behavior under comparison; they are gated on reference recall and SLO as in
stage 1.

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

Per window: quality fraction, fraction within SLO, native prefix-cache hits
split GPU and CPU (same counters as stage 1), restore and offload blocks, engine
KV usage gauge (should average lower in B), GPU utilization, Xeon utilization
per socket, compactions per minute, summary prefill tokens, pending-turn
fraction, context-limit failures. All per-turn costs are divided by correct
turns in the window.

Evidence: campaign manifest, summarizer image and model fingerprints, the fixed
summarizer prompt, the raw archive, every summary and ledger, every probe with
expected and actual answers, engine counters before and after each window.

## Gates that block a claim

- Phase 0 shows the host cannot sustain the baseline population's compaction
  rate: no density ladder until the summarizer is resized.
- Any context-limit failure in arm C or D: the context-wall line fails.
- Reference recall below 100% in arm C or D: the prefix was disturbed; stop.
- Session recall or knowledge update falling across cycles in C or D beyond the
  stage 1 quality gate: the recall gate fails; add archive retrieval and re-run
  phase 1.
- Prefix-cache hit fraction after compaction below the stage 1 level: the layout
  is not prefix-aligned; stop.
- Pending-turn fraction above 10%: the loop is not keeping up; the density
  result is reported with that caveat or the summarizer is resized.
- Summary prefill per cycle above the prefix-aligned expectation (about 4K
  tokens): the layout is leaking history into the rebuild; stop.
- Single blocks never produce uplift percentages; same rules as stage 1.

## Running it

Phase 0, summarizer calibration (no GPUs). Start the summarizer on one Xeon
socket yourself (vLLM CPU image or llama.cpp, pinned with numactl or cpuset),
then copy `docs/examples/p02-stage2-calibration.json`, set the endpoint and
model, and run:

```
python -m simulator.infrastructure.compaction_calibration plan.json /path/to/summarizer/tokenizer out/
```

It writes `calibration.json` and `calibration.md` with compactions per second
at each concurrency and a pass/fail against twice the required rate.

Phases 1 to 3 use the P02 capacity campaign with the stage 2 settings in
`docs/examples/p02-stage2-settings.json`. The schedule multiplies the cache
arms by `session_compaction_arms`, so one campaign holds A, B, C and D at each
population. Phase 1 is that file with one population and `["xeon"]` only.

## Harness changes (implemented)

1. `knowledge_workers.py`: work-item turns (`session_growth_tokens`), a
   per-worker fact log as ground truth, probes sampled uniformly over the whole
   history (`session_probe_mix`), cycle index and before/after event edges,
   new-session reset at the limit (`session_limit_policy`), the async
   compaction loop with lag and pending-turn accounting, raw archive and
   fact-log evidence, recall and compaction metrics, the four-arm comparison
   with C/A, B/A, D/A and D/B ratios, and the live-context and recall
   timeline charts in the report.
2. `session_compaction.py`: fixed summarizer prompt, summary and ledger
   parsing, ledger merge, prefix-aligned header, restore of live messages.
3. `spec.py`: the `session_compaction_*`, `session_growth_tokens`,
   `session_probe_mix` and `session_limit_policy` fields with validation.
4. `driver.py`: per-probe answer schemas. `campaign.py`: compaction arms in
   the schedule. `session_capacity_worker.py`: cell directories and pass-through.
5. `compaction_calibration.py`: the phase 0 runner.

Not implemented: launching the summarizer from the harness. It is an operator
prerequisite recorded by endpoint and model name, like the correctness runner.
Summary prefill tokens after compaction are recorded per compaction
(`post_compaction_tokens`); attributing them inside engine counters remains
the stage 1 attribution gap.

## Agreed methodology (2026-10-06)

Two experiments on one harness, both fixed work rather than fixed time: every
worker delivers the same number of turns and the same tokens of new material
in every arm, and the time it takes is part of the result.

Compaction is blocking. When a turn's prompt crosses the trigger, the raw
turns since the last compaction go to the Xeon summarizer and the worker's
next turn waits for the summary. The wait is the user-felt cost of compaction
and is measured per compaction: pause per compaction, waiting turns as a
fraction of all turns, total added minutes per worker, and the pause curve
against population, which is the Xeon's capacity stated directly. The per-turn
response SLO is evaluated on model response time and excludes the compaction
wait; the two are reported side by side, never combined.

No-compaction arms reset at a context budget of 96K, the stage 1 validated
context and the most generous budget a deployment would run; the engine limit
is not used because extraction accuracy in this fixture falls from 0.97 to
0.88 above about 80K. A reset keeps the 8K reference prefix and drops the
history. Compaction arms never reset.

Recall is reported as intact history versus behind a reset, with abstention
on its own line and excluded from whole-history recall, and as recall per
cycle index. Session notes set a working allocation, a session fact that is
not a reference field, so reference probes never conflict with notes.

One combined experiment: four arms (A GPU-only, B Xeon KV tier, C Xeon
compaction, D both) at 56 and 96 workers, 8K prefix, 9K tokens of new
material per turn at 75 s mean think time, 48 turns of fixed work per worker.
No-compaction arms reset at the 96K budget about five times per worker;
compaction arms compact at 64K about ten times per worker. Random-phase
start: each worker begins at a uniformly random point of its cycle via a
synthetic prior history of fabricated correct answers, so triggers do not
reach the summarizer in lockstep and the run represents a steady population;
cycles are counted per worker, so the per-cycle recall staircase is
unaffected. At 56 workers the summarizer is loaded; at 96 it is several times
over capacity and the pauses grow through the cell, which is the Xeon's limit
stated directly. Deliverables from the one evidence set: whole-history recall
per cycle (a staircase toward zero without compaction, flat with it), the
compaction pause distribution and its curve against population, and density
at the task gate (reference-probe accuracy of at least 0.95 and the response
SLO, which excludes the compaction wait). Compaction arms take longer to
deliver the same work; that added time is reported, not hidden.

Out of scope, recorded as a future optimization: asynchronous compaction that
overlaps the summary with think time and only waits when a session would
cross its budget. It removes the pause in normal operation at the cost of a
small re-prefill of the in-flight turns; it was prototyped during execution
and set aside so the measured cost stays explicit.

## Execution log (XE7740, 2026-10-05 to 2026-10-06)

Phase 0 established that the summarizer is bound by raw token throughput, not
by model size. Qwen3-30B-A3B (MoE, 3B active) and dense Qwen3-4B both decode
a 64K-context summary slowly on one socket because every step re-reads the
KV cache of the whole transcript; guided JSON output crashes the CPU vLLM
engine, so the ledger is a line format parsed by the harness. The working
configuration is chunked compaction (chunks of a few turns, in parallel across
one vLLM CPU instance per socket, Qwen3-4B-Instruct-2507 BF16), which
calibrated at 0.164 compactions/s for 64K inputs. That calibration was
optimistic: its concurrent calls shared identical text and hit the prefix
cache; the calibration runner now uses distinct transcripts per call.

Phase 1 (campaign d8acd3fe, arm D, 56 workers, 48K prefix, 6K per turn, 75 s
think, 96K trigger) showed the loop: 101 compactions all landed, two to three
cycles per worker. It also exposed three corrections now in the plan:

- Token rate is the constraint. Under campaign load the two sockets absorbed
  about 2.8K raw tokens/s; 56 workers at 6K tokens per 75 s need 4.5K, so lag
  grew past six minutes, half the turns were sent while a compaction was
  pending, and sessions that outran their compaction reset at the engine
  limit. The ladder uses 3K per turn at 90 s think (1.9K tokens/s at 56
  workers, 3.2K at 96, which is the expected Xeon limit at 96).
- The reference prefix was 48K, so a 32K target was unreachable (post
  compaction prompt 61K). The ladder uses a 24K prefix, a 64K trigger and lands
  near 34K.
- Session notes updated a reference field (quantity), so reference probes
  disagreed with the notes: 45 of 58 failed reference answers used the noted
  value. Notes now set a separate working allocation.

Windows are 3600 s so no-compaction sessions reach the engine limit and
compaction sessions complete at least two cycles from a cold 24K start.

Compacted-prompt layout, revised after the 56-worker cells of the combined run
(2026-10-06): reference accuracy in the compaction arms fell with the number
of prose summaries accumulated after the reference records (0.94 to 0.97 with
up to three summaries, 0.83 to 0.91 with seven to nine), and the same
accumulation is the likely cause of the slow recall drift across cycles. The
header is now prefix, then the ledger, then at most two sentences of prose per
cycle. The ledger carries the facts the probes need; the prose is narrative.
The run with the Qwen3-30B-A3B FP8 summarizer on SGLang CPU uses this layout;
the 4B run keeps the original layout as recorded, and the two are compared on
that basis.

### Combined run with the Qwen3-4B summarizer (campaign c38a4714, 2026-10-06/07)

Eight cells, fixed work of 48 turns at 9K tokens per turn, 8K prefix, 75 s think,
resets at 96K, blocking compaction at 64K, random-phase start. Reanalyzed with
the curve and end-state metrics; outputs under `analyses/stage2/`.

| Arm, workers | Task accuracy | Task SLO 30 s | TTFT p95 | End-state recall | Recall curve | Resets or compactions per worker | Pause p50 | Added min per worker | Wall |
|---|---|---|---|---|---|---|---|---|---|
| A GPU-only, 56 | 0.971 | 0.97 | 5.3 s | 0.05 | 0.88 to 0.05 | 5 resets | none | 0 | 87 min |
| B KV tier, 56 | 0.965 | 0.97 | 5.2 s | 0.05 | 0.86 to 0.05 | 5 resets | none | 0 | 86 min |
| C compaction, 56 | 0.945 | 0.95 | 4.3 s | 0.76 | 0.92 to 0.70 | 9 compactions | 201 s | 25.6 | 109 min |
| D both, 56 | 0.921 | 0.92 | 4.1 s | 0.77 | 0.92 to 0.72 | 9 compactions | 190 s | 25.6 | 110 min |
| A GPU-only, 96 | 0.974 | 0.86 | 21.0 s | 0.08 | 0.87 to 0.06 | 5 resets | none | 0 | 92 min |
| B KV tier, 96 | 0.975 | 0.96 | 7.1 s | 0.08 | 0.89 to 0.06 | 5 resets | none | 0 | 88 min |
| C compaction, 96 | 0.932 | 0.93 | 4.5 s | 0.76 | 0.96 to 0.73 | 9 compactions | 531 s | 77.5 | 159 min |
| D both, 96 | 0.938 | 0.94 | 4.2 s | 0.78 | 0.91 to 0.78 | 9 compactions | 543 s | 78.8 | 159 min |

Readings. Recall: end state 0.76 to 0.78 with compaction against 0.05 to 0.08
without, at both populations; the reset arms step down at every reset and the
compaction arms drift slowly from the low 0.80s to the low 0.70s. Density at
the 30 s SLO with the 0.95 task gate: A passes only at 56 (TTFT p95 21 s at
96, 37% of prefix tokens rebuilt on the GPU), B passes at 96, so the stage 1
warm-cache result reproduces at 1.71x. C and D fail the task gate at both
populations by 0.5 to 3 points, caused by the accumulated prose summaries, and
not by latency: their TTFT and response latency are the best of all arms at
96. The compaction pause is the measured Xeon limit: a median of about 195 s
at 56 workers and about 540 s at 96, where the summarizer runs roughly four
times over its throughput, adding 26 and 78 minutes per worker to the same 48
turns of work. The 30B-A3B run with the ledger-first layout targets the task
gate directly.

### Summarizer calibration, 4B versus 30B-A3B (2026-10-07)

Corrected runner (distinct transcripts per call), 18K raw inputs in two-turn
chunks, 1024-token budget, two sockets, both models on the vLLM CPU image with
threads bound to the 86 physical cores per socket.

| Model | Compactions/s at concurrency 32 | Wall p50 at concurrency 8 | Completion tokens | Headroom at 56 workers |
|---|---|---|---|---|
| Qwen3-4B-Instruct-2507 BF16 | 0.415 | 22 s | ~200 | 2.78x |
| Qwen3-30B-A3B-Instruct-2507 BF16 | 0.258 | 38 s | ~220 | 1.73x |

The MoE's three billion active parameters do not make it faster on this
backend: decode re-reads the full expert weights and the vLLM CPU MoE path is
the slower of the two. The SGLang CPU FP8 path with the AMX attention backend
is the recorded fast path for this model (R470: about 284 tokens/s peak decode
on 64 cores); its image build on the XE7740 is waiting on a Docker Hub pull
rate limit. Plan: the 30B run at 56 workers proceeds on vLLM as a model
comparison on the same backend; the 96-worker cells are reserved for SGLang.

### 30B-A3B on vLLM CPU, arms C and D at 56 (campaign d4aa8dfc, 2026-10-07)

Same design as the 4B combined run, ledger-first header with two-sentence
summaries, Qwen3-30B-A3B-Instruct-2507 BF16 on the vLLM CPU image.

| Arm | Task accuracy | End-state recall | Recall curve | Compactions | Pause p50 / p95 | Resets |
|---|---|---|---|---|---|---|
| C compaction | 0.956 | 0.67 | 0.92 to 0.64 | 491 of 500 | 407 s / 710 s | 2 |
| D both | 0.961 | 0.69 | 0.88 to 0.76 | 495 of 496 | 426 s / 823 s | 0 |

Both arms pass the 0.95 task gate, which the 4B arms on the old layout did
not (0.921 and 0.945). End-state recall is lower than the 4B's 0.76 to 0.77,
but model and layout changed together; the 4B control on the new layout
separates them. Nine compactions failed in C: six where the 30B's output lacked
the summary or ledger block and three summarizer read timeouts; each failure
leaves the session growing to the 96K budget, hence the two resets. The pause
is about twice the 4B's, as the calibration predicted for this backend.

### 4B on the new layout, arm D at 56 (campaign 573ada2e, control, 2026-10-07)

Same design and ledger-first layout as the 30B run, with the Qwen3-4B
summarizer back in place, so that the model and the layout are separated.

| Arm | Task accuracy | End-state recall | Recall curve | Compactions | Pause p50 / p95 | Resets |
|---|---|---|---|---|---|---|
| D both | 0.952 | 0.75 | 0.92 to 0.85 (0.73 low) | 491 of 491 | 256 s / 567 s | 0 |

The layout change, not the model, is what moved the task gate: the 4B passes
0.95 on the new layout and keeps its end-state recall at 0.75, while the
30B-A3B on the same layout landed at 0.69. The smaller dense model is the
better extractor on this backend as well as the faster one. One measurement
caveat: these summarizers were relaunched with the 30B run's engine flags
(block size 32, 32K model length) instead of the first run's (block size 128,
64K), which cost about 10 percent summarizer throughput (3460 versus 3800
tokens per second) and a deeper queue, so the pause here is 256 s against 190 s
yesterday for identical work. Recall and task accuracy do not depend on those
flags. Arm C of this control is running on the same summarizers.

### Summarizer load imbalance found and fixed (2026-10-07)

The engine's own request timers on the two summarizer sockets showed an uneven
split during the 4B control run: one socket had served about twice the requests
of the other, and its requests waited far longer for a batch slot.

| Socket endpoint | Requests | Queue wait mean | Prefill mean | Decode mean | End to end mean |
|---|---|---|---|---|---|
| socket 1, port 18000 | 1462 | 163 s | 8 s | 164 s | 335 s |
| socket 0, port 18001 | 746 | 13 s | 11 s | 100 s | 125 s |

Cause: the replay called the compaction without its shared round-robin
counter, so every three-chunk compaction sent chunks to sockets in the order
1, 0, 1 and socket 1 carried two thirds of all summarizer work. A compaction
lands only when its slowest chunk does, so every pause in every compaction run
so far (4B, 30B-A3B, control) was set by the overloaded socket. The comparisons
between those runs stay valid because they share the defect; their absolute
pause times are overstated. The calibration runner had the same defect.

Fix: chunks now take the next endpoint from one counter shared across all
compactions, so the sockets split the chunks evenly. Each compaction cell also
records the engine's per-request queue, prefill, decode and end-to-end means
for the cell in summarizer-timers.json, which separates waiting for a slot from
executing. The SGLang run is the first to use the fix; SGLang does not expose
the same timers, so that file records null for it and the client-side chunk
walls remain the measure there.

### Best-case summarizer sweep (2026-10-07)

Attribution between model, layout and backend is no longer a goal; the
remaining runs aim at the best Xeon compaction available. The arm C cell of
the 4B control was cancelled for that reason. Changes for the best case:

- Chunks split evenly across the two sockets (the fix above).
- OMP threads on 80 of the 86 physical cores per socket, so that the GPU
  engines' host threads and the harness do not preempt an OMP barrier.
- vLLM block size 128, which was about 10 percent faster than 32.
- SGLang KV pool raised from 131K to 655K tokens. The R470 recipe's 131K pool
  holds only about eight 16K summarizer requests at once, which would have
  been a queue of its own.
- Calibration at the real compaction shape: 48K raw tokens in three 16K chunks,
  8 and 16 compactions in flight (24 and 48 chunks, the depth the 56-worker
  cells reached).

Configurations swept: vLLM 4B BF16 with default and 8K batched tokens, SGLang
4B BF16 with 4K and 8K prefill chunks, SGLang 30B-A3B FP8. The winner is the
most compactions per second at 16 in flight; a 4B configuration wins ties
within 15 percent because the 4B extracted better. One-turn chunks are then
tried on the winner and kept if more than 5 percent faster. The winner runs
arms C and D at 56 and 96 workers (plan exp-best), with the engine's queue
versus execution split recorded per cell.

Sweep results (idle GPUs; 48K raw tokens per compaction, 16 compactions in
flight, all calls succeeded):

| Setup | Compactions/s | Wall p50 | Summarizer tokens/s |
|---|---|---|---|
| vLLM 4B, 8K batched tokens, one-turn chunks (winner) | 0.363 | 29 s | about 18,500 |
| vLLM 4B, 8K batched tokens, two-turn chunks | 0.257 | 61 s | about 13,000 |
| vLLM 4B, default batching, two-turn chunks | 0.202 | 76 s | about 10,300 |
| SGLang 30B-A3B FP8 AMX | 0.137 | 110 s | about 7,000 |
| SGLang 4B BF16 AMX, 4K prefill chunks | 0.128 | 119 s | about 6,500 |
| SGLang 4B BF16 AMX, 8K prefill chunks | 0.118 | 128 s | about 6,000 |

vLLM beat SGLang at this shape. Larger prefill batches (8K) and one-turn
chunks (8K prompts instead of 16K, so less attention work per token and more
sequences to batch) each added about 30 to 40 percent.

One-turn chunks failed in the campaign. Attempt 1 of exp-best (launched 15:43
UTC) cut the pause p50 to 31 s, but 59 of the first 167 compactions failed and
nine sessions reset: given a single turn with no session note, the 4B invented
allocation lines from the reference records (city names, or one value repeated
until the budget ran out). Two-turn chunks had no such failures in 504
compactions. Attempt 2 runs the same engine with two-turn chunks (0.257
compactions/s, 61 s wall at idle), arms C and D at 56 and 96. These are idle-GPU
numbers; under campaign load the GPU engines share the cores, and the cell's
engine timers will show the real queue and execution split.

Under campaign load (attempt 2, arm D at 56, first 30 minutes): 190 of 190
compactions landed, reference accuracy 0.951, pause p50 163 s against 256 s
for the same work before the fixes. The engine timers over a five-minute
window split one chunk's 100 s as 9 to 15 s waiting for a batch slot, 6 s
prefill, and 80 s in the decode phase. Each chunk writes only about 115
tokens; the decode phase is long because every engine step also carries an
8K-token prefill slice of another request. The summarizer is therefore bound
by prompt-token throughput, about 7K tokens per second across both sockets
under load with a 55 percent prefix-cache hit rate, against about 13K at idle
in the sweep. The sockets are evenly loaded (63 and 61 requests in the window)
and the cores hold about 2.35 GHz, so contention is not the cause. The idle
calibration overstated capacity, most likely through more prefix sharing
between its synthetic transcripts. The remaining lever is fewer prompt tokens
per compaction: one-turn chunks were about 40 percent faster at idle, but they
need a prompt change, never writing an allocation line for a turn without a
session note, and a validation before use.

### Best case with two-turn chunks, arm D at 56 (campaign 3d8803d2, 2026-10-07)

| Metric | 4B first run (c38a4714) | 4B control (573ada2e) | Best case (3d8803d2) |
|---|---|---|---|
| Task accuracy | 0.921 | 0.952 | 0.964 |
| End-state recall | 0.77 | 0.75 | 0.75 |
| Compactions landed | 504 of 504 | 491 of 491 | 492 of 492 |
| Pause p50 / p95 | 190 s / 330 s | 256 s / 567 s | 105 s / 216 s |
| Added wait per worker | 25.6 min | 38.0 min | 10.1 min |
| Time to finish the fixed work | 110 min | 118 min | 96 min |

Engine split per chunk over the cell: queue 12 to 13 s, prefill 7 s, decode
phase 80 s, end to end 100 to 104 s, 740 requests per socket. The 96-worker
cells were held back at the user's request; the campaign was cancelled after
this cell to test one-turn chunks with tightened allocation instructions.

### One-turn chunks with tightened instructions (pre-check and run, 2026-10-07)

Changes: allocation lines only from sentences that begin "Session note", with
the note's integer value, never from record fields, each line once; the parser
drops a non-numeric allocation line (counted) instead of failing the
compaction; the calibration scores each ledger against the true facts.

| Pre-check, same transcripts, 16 in flight | One-turn chunks | Two-turn chunks |
|---|---|---|
| Compactions/s | 0.299 | 0.113 |
| Wall p50 | 42 s | 138 s |
| Ledger facts exactly right | 1.000 | 0.937 |
| Extra entries | 68 | 55 |
| Failures, dropped lines | 0, 0 | 0, 0 |

All extra entries in the one-turn run are "requested" lines for turns that
only asked a question, filled with the record of that turn's session note.
Recall probes ask only about turns that had a request, so such a line is never
scored and cannot displace a true fact. The two-turn run had the same kind plus
entries outside its turns and allocations without a note. The automatic gate
counted all extras and rejected the one-turn setup; it was launched by hand on
that reading as exp-best-1t, arms C and D at 56 workers.

### Result: one-turn chunks, arm D at 56 (campaign 73d74d9a, 2026-10-07)

The minimum test agreed with the user: arm D at 56 only. Arms A and B do not
use the summarizer, so their results from c38a4714 stand; arm C is inferred
from D (in every earlier run the two were within about 5 percent on pause and
0.02 on recall). The campaign was cancelled before arm C.

| Arm D at 56 workers | First 4B run | Two-turn best case | One-turn best case |
|---|---|---|---|
| Task accuracy | 0.921 | 0.964 | 0.955 |
| End-state recall | 0.77 | 0.75 | 0.83 |
| Recall curve, cycle 0 to 9 | 0.92 to 0.72 | 0.92 to 0.73 | 0.92 to 0.89 |
| Compactions landed | 504 of 504 | 492 of 492 | 502 of 502 |
| Compaction time p50 / p95 | 241 s / 358 s | 108 s / 244 s | 16 s / 34 s |
| Turn wait p50 / p95 | 190 s / 330 s | 105 s / 216 s | 8 s / 28 s |
| Total wait per worker | 25.6 min | 10.1 min | 0.3 min |
| Time to finish the fixed work | 110 min | 96 min | 87 min |
| Abstention (not in recall) | not recorded here | 0.906 | 0.851 |

Arm A finished the same work in 87 minutes, so compaction no longer adds
measurable time. Most compactions now land inside the user's think time, and
the next turn rarely waits. Engine split per chunk: queue under 0.1 s, prefill
about 2 s, decode phase 11 to 13 s, 14 to 16 s end to end, 1317 requests per
socket. Recall rose because the one-turn ledgers are more exact (1.000 against
0.937 in the pre-check). Abstention fell from 0.91 to 0.85 and is worth a look
before publication; it is reported apart from recall.

Decision (user, 2026-10-07): arm C is reported with arm D's one-turn numbers,
noted as inferred. Basis: in three runs C and D were within 0.02 on recall and
about 5 percent on pause, with the order flipping between runs; with compaction
the CPU tier supplied only about 2 percent of cached tokens in D (2.6M against
91M from the GPU), so D behaves as compaction alone at these populations. A
measured C, and a distinct D-over-C claim, are deferred to a later run that
raises the worker count until compacted contexts overflow GPU KV memory and
the CPU tier carries real load.

## Stage 3 candidates, not promised

- Push the worker count past GPU KV capacity with compaction on, so the CPU
  tier carries real load; measure arm C there and make the D-over-C claim.

KV compaction in latent space (attention matching, Still) on KV already resident
in the Xeon tier; archive retrieval into the live prompt; model-side sparse
attention (DeepSeek DSA class) as a headline-model change. Each needs engine
support or a model change that stage 2 does not depend on.
