# Simulator Algorithm

End-to-end description of how the persona-capacity simulator drives load, takes measurements, and decides what to measure next. This document is the source-of-truth reference for the algorithm; the code is in `simulator/`.

## 1. The closed-loop model

The simulator is a **closed-loop** load generator: each virtual user is an independent async task that follows a strict sequence — submit a request, stream the response, read+think, submit the next request. The next request can't start until the previous one finishes, so the system's throughput is what governs request rate, not a fixed RPS dial.

Closed-loop is the right model for chat-style workloads:

- Real human users don't fire requests at a constant RPS; they wait for the response and then think before composing the next message.
- When the backend slows down, real users automatically slow their request rate — they can't compose the next message until they've read the previous response. The closed-loop model captures this naturally; an open-loop RPS generator would keep firing requests into a saturated engine and produce nonsense queue depths.
- This means "pool_size" in the simulator means "concurrent active sessions," not "concurrent in-flight requests." At any moment most users are reading/thinking and only a fraction have a request in flight.

The key consequence: **the simulator's pool_size and the engine's in-flight count are different numbers**. A pool of 64 users with a duty cycle of ~15% (typical for chat) means the engine sees ~8-12 concurrent requests on average, with bursts higher.

## 2. Distributions

All variable persona parameters use one of three samplers, defined in [`simulator/distributions.py`](../simulator/distributions.py).

### LogNormal

Parameterized as `LogNormal.from_median(median, sigma)`:

- `median` — the median of the resulting distribution (= `exp(mu)`)
- `sigma` — the dispersion (standard deviation of the underlying normal)

Useful sigma reference points:

| sigma | P10 | median | P90 | spread |
|---|---|---|---|---|
| 0.35 | 0.64× | 1× | 1.57× | tight (token counts that don't vary much) |
| 0.4 | 0.60× | 1× | 1.67× | moderate |
| 0.5 | 0.53× | 1× | 1.90× | typical for token counts and read times |
| 0.6 | 0.46× | 1× | 2.16× | wide |
| 0.7 | 0.41× | 1× | 2.46× | heavy-tailed (active think times) |
| 1.0 | 0.28× | 1× | 3.60× | very heavy-tailed |

LogNormal is used for token counts, read/think times, and inter-session gaps. It's the right shape for things that are bounded below by zero, have a typical "central" value, and occasionally go large.

### Discrete

A weighted choice over named integer values, used for turns-per-session counts:

```python
turns_per_session = Discrete({1: 0.7, 2: 0.2, 3: 0.07, 4: 0.03})
```

Roughly 70% of sessions get 1 turn, 20% get 2, etc.

### Constant

A degenerate sampler for parameters that don't vary. Rarely used.

## 3. Personas

A persona is one user archetype: typical token counts, think time, SLA thresholds. Defined in [`simulator/personas.py`](../simulator/personas.py).

### Persona parameters at a glance

| persona | input median (σ) | output median (σ) | turns/session | read+think median (σ each) | ttft target/fail | tpot target/fail |
|---|---|---|---|---|---|---|
| quick_lookup | 350 (0.35) | 60 (0.4) | {1:0.7, 2:0.2, 3:0.07, 4:0.03} | 8s (0.5) + 10s (0.6) | 5s / 15s | 150ms / 225ms |
| conversational | 180 (0.5) | 220 (0.45) | {3:0.25, 5:0.35, 8:0.25, 12:0.15} | 28s (0.5) + 25s (0.6) | 9s / 30s | 150ms / 225ms |
| writer | 500 (0.5) | 350 (0.4) | {1:0.5, 2:0.3, 3:0.15, 5:0.05} | 44s (0.5) + 70s (0.7) | 15s / 45s | 180ms / 270ms |
| document_qa | 3500 (0.5) | 280 (0.4) | {1:0.4, 2:0.3, 4:0.2, 7:0.1} | 35s (0.5) + 115s (0.6) | 30s / 90s | 200ms / 300ms |
| code_assist | 1200 (0.6) | 450 (0.5) | {2:0.3, 4:0.35, 8:0.25, 15:0.1} | 56s (0.5) + 180s (0.7) | 20s / 60s | 220ms / 330ms |
| long_form_generator | 200 (0.5) | 3000 (0.4) | {2:0.75, 3:0.25} | 90s (0.5) + 120s (0.6) | 15s / 45s | 180ms / 270ms |

Notes on σ choices:

- **Token counts** use σ in the 0.35-0.6 range — wide enough to reflect that real prompts and responses vary, narrow enough that the median is meaningful.
- **Read time** uses σ ≈ 0.5 across all personas — moderate variance.
- **Active think time** uses σ ≈ 0.6-0.7 — heavier-tailed because deliberation time genuinely has long tails (some prompts are easy, others require thought).

### SLA thresholds

Each persona has two thresholds per latency metric:

- **Target** (the lower one) — "premium experience." User would describe the service as fast.
- **Failure** (the higher one) — "hard SLA boundary." Past this, the user would abandon or escalate.

The simulator computes two parallel pass/fail readings per measurement, one against each threshold. The failure threshold is what gates the headline `capacity_pool_size`; the target threshold gates `target_capacity_pool_size`. The two numbers let downstream consumers report both "supports N users at SLA" and "supports M users at premium quality."

### Cohorts

A cohort is a weighted mix of personas — a "team type." Defined in `COHORTS` in `simulator/personas.py`:

| cohort | mix |
|---|---|
| chat_heavy | 60% quick_lookup, 30% conversational, 10% writer |
| general_knowledge | 30% quick_lookup, 30% conversational, 25% writer, 10% long_form, 5% doc_qa |
| writer_dominant | 40% writer, 30% long_form, 20% conversational, 10% quick_lookup |
| software_engineering | 50% code_assist, 15% long_form, 15% quick_lookup, 15% conversational, 5% doc_qa |
| analyst_team | 70% doc_qa, 20% writer, 10% long_form |

When a virtual user is spawned in a cohort run, its persona is sampled from the cohort's weights. Each user holds a single persona for its whole life.

## 4. Virtual user lifecycle (per-session respawn)

One virtual user = one asyncio task = one chat session. The simulator uses **per-session respawn**: a user runs exactly one session then terminates, and the pool manager spawns a replacement. This makes `pool_size` mean "active concurrent sessions" rather than "active user identities."

Implementation: [`simulator/virtual_user.py`](../simulator/virtual_user.py).

The lifecycle:

```
spawn
  └─ initial phase offset (random 0–1× sample of read+think)
  └─ session start (turn 0)
        └─ generate input prompt (~target tokens of random words from corpus)
        └─ submit chat completion request (streaming)
        └─ consume stream (tiered timeouts; see §5)
        └─ record TurnEvent
        └─ append to history (input + output text, byte-identical for prefix-cache reuse)
        └─ if more turns: sleep for read_time + active_think (independent LogNormal samples)
        └─ session continues with growing history
  └─ session end (after N turns where N is sampled from turns_per_session)
  └─ user terminates; pool manager spawns replacement
```

### Initial phase offset

When a virtual user is spawned, it sleeps for `uniform(0, 1) × (read_time + active_think sample)` before issuing its first request. This staggers fresh users across the request/think cycle so a synchronous ramp burst doesn't produce a synchronized query storm at t=0.

Without this offset, all N users spawned in the same ramp burst hit `/chat` on the same event-loop tick and stay synchronized for many cycles. The phase offset breaks symmetry from the moment users start.

Controlled by `initial_phase_offset_enabled` in `SimulationConfig` (default `True`).

### Multi-turn within a session

Within a session, the user holds a growing chat history:

- Turn 0: prompt = `[{role: user, content: <random text>}]`
- Turn 1: prompt = `[turn 0 user, turn 0 assistant, turn 1 user]`
- Turn N: prompt includes all prior turns

Between turns, the user sleeps for `read_time_seconds + active_think_seconds` (independent samples each turn). The two are sampled separately and summed:

- `read_time_seconds` represents residual reading after the stream ends — the part the user hasn't caught up to yet. Sized as `output_tokens × max(0, 200ms - typical_TPOT) / 1000` assuming ~5 tok/s reading speed (~230 wpm).
- `active_think_seconds` represents deliberation and composition. Larger for code/document work, smaller for quick lookups.

This separation makes the buyer page able to report the distinction; the runtime just sums them.

### Session termination and respawn

When the last turn completes, the user task returns. The pool manager's reaper loop (running every 0.5s) detects the terminated task and spawns a replacement — a fresh virtual user with newly-sampled persona (from cohort weights) and a fresh user_id. This keeps the active pool at the target size while the run progresses.

Implementation note: a user's `sessions_before_leaving` and `inter_session_gap_seconds` parameters are deprecated under per-session-respawn. They remain on the dataclass for back-compat but no longer drive runtime behavior. Effective `sessions_target = 1` for every spawn.

## 5. Turn execution

The per-turn loop is at the heart of the load generator. Steps:

### 5.1 Input generation

`corpus.make_text(input_tokens_target, rng)` generates a random word-sequence of approximately the target token count from a pre-tokenized word corpus. The text varies across turns (no two prompts are byte-identical), which would defeat prefix caching across users. Within a session, history is reused byte-identically so the engine's prefix cache hits cleanly turn-to-turn.

### 5.2 Request submission

The request is an OpenAI-compatible streaming chat completion:

```python
client.chat.completions.create(
    model=model_id,
    messages=[<history>, {"role": "user", "content": query_text}],
    max_tokens=output_tokens_target,
    stream=True,
    temperature=0.7,
    extra_body=extra_body,  # reasoning_effort if applicable
)
```

`max_tokens` is the persona's sampled output target, bumped by `REASONING_OVERHEAD_TOKENS` if the engine is a reasoning model (see §10).

### 5.3 Streaming consumption with tiered abort

The simulator consumes the stream chunk by chunk, applying three layered timeouts ([`simulator/streaming.py`](../simulator/streaming.py)):

- **Tier 1 (pre-TTFT):** if no first token arrives within `pre_ttft_timeout_s = ttft_failure_seconds × 5`, abort. Catches admission deadlock — request enqueued but never scheduled.
- **Tier 2 (inter-token):** once the stream starts, if no new token arrives for `inter_token_timeout_s = tpot_failure_ms × 20 / 1000`, abort. Catches mid-stream worker hangs.
- **Tier 3 (hard ceiling):** total wall time can't exceed `hard_timeout_s` (default 900s, capped further by config `request_timeout_s`). Catches slow-but-progressing requests that are functional but pathologically backlogged.

A request that's aborted at any tier produces a synthetic `TurnEvent` carrying whatever partial timing was captured (TTFT if reached, partial token count, etc.) plus an `error` field categorizing the failure mode. Synthetic events are counted as SLA violations regardless of timing.

### 5.4 TurnEvent recording

For each turn (successful or failed), the simulator records a `TurnEvent` with:

- `submitted_at_ms`, `completed_at_ms` — wall-clock bounds
- `ttft_ms` — time from submit to first user-visible token (reasoning or content)
- `ttfct_ms` — time from submit to first content token (= ttft for non-reasoning models)
- `tpot_ms` — `(e2e - ttft) / max(1, total_visible_tokens - 1)` where `total_visible_tokens = output_tokens + reasoning_tokens`
- `input_tokens`, `history_tokens`, `output_tokens`, `reasoning_tokens`
- `in_flight_at_submit` (and avg/peak during) for queue-depth diagnosis
- `error` (null for successful turns)

These rows go to the `turn_events` SQLite table and become the basis for all downstream aggregation: percentiles, throughput, timeline reconstruction.

### 5.5 SLA flags

At record time, each turn is classified against both threshold tiers:

- `sla_ttft_violation = ttft > persona.ttft_failure_seconds`
- `sla_tpot_violation = tpot > persona.tpot_failure_ms`
- `ttft_target_miss = ttft > persona.ttft_target_seconds`
- `tpot_target_miss = tpot > persona.tpot_target_ms`

A failed (timeout/error) turn is forced to `violation = target_miss = 1` regardless of timing — a failed request is a violation in both axes by definition.

## 6. Pool manager

[`simulator/pool_manager.py`](../simulator/pool_manager.py) maintains the target active-user count.

### 6.1 Ramp

When a measurement step targets `n` users, the pool ramps up by spawning at most one new user per `ramp_spawn_interval_s` seconds (default 1.0s). This avoids the synchronized burst that an instantaneous N-user spawn would produce.

Combined with the per-user initial phase offset (§4), the ramp produces a pool of users distributed across the request/think cycle from the moment ramping completes.

### 6.2 Sticky load balancing across replicas

If the engine has multiple replicas (e.g. dual-socket vLLM with two endpoints), each virtual user is **stickily assigned** to one replica for its lifetime. The assignment uses a simple "least-assigned-count" rule on first request: pick the replica with the fewest currently-assigned users.

This keeps a user's multi-turn conversation pinned to one backend so prefix caching works. Rotating users across replicas turn-to-turn would defeat the prefix cache entirely.

A balance audit logged at run end shows `±1` user-assignment distribution across replicas under normal operation.

### 6.3 Per-user RNG streams

Each virtual user gets its own `random.Random` seeded deterministically from the pool's RNG seed (default `0xC0FFEE`). This makes runs reproducible: same seed → same persona assignments, same prompt token counts, same think times.

### 6.4 Reaper loop

A background task wakes every 0.5s and:

1. Detects terminated user tasks (completed, errored, or cancelled)
2. Removes them from the active pool
3. Spawns replacements until the active count equals the target

This is how per-session-respawn maintains pool size as sessions end.

## 7. Measurement step

One measurement step = one (cohort, pool_size) pair. The step has two phases: **warmup** and **measurement**.

Implementation: [`simulator/measurement.py`](../simulator/measurement.py).

### 7.1 Warmup with throughput convergence

After ramping the pool, the simulator doesn't start "measuring" until it's confident the system has stabilized.

Closed-loop workloads have intrinsically bursty `in_flight` counts (each user oscillates between request and think phases), so the average never converges in the noise-floor sense. Instead, the simulator monitors **completions per second** in rolling windows:

- Sample `state.completed` once per second
- Compare completions in the most-recent `convergence_window_s` (default 60s) against completions in the prior `convergence_window_s`
- Declare converged when `relative_change < convergence_threshold` (default 0.20 = 20%)
- Both windows must have at least `min_completions_per_window` (default 5) to be compared — avoids declaring convergence on noise when the engine is so slow only a handful of requests have finished

Warmup is bounded:

- `warmup_min_duration_s = 30` — don't even check convergence below this floor
- `warmup_max_duration_s = 300` — hard ceiling; proceed to measurement either way after this

Better to proceed with warmup-tail noise than have no data.

### 7.2 Measurement window

Once convergence (or warmup timeout) is reached, the simulator opens a measurement window and collects turn events for up to `measurement_timeout_s` (default 300s) or until `target_samples_per_step` (default 500) events accumulate, whichever comes first.

During the window, the simulator also samples engine-side telemetry every second (KV cache used, queue depth, prefix-cache hit count, CPU utilization, memory). These rows go to `measurement_telemetry`.

### 7.3 Per-step aggregation

When the window closes, all turn events in the window are aggregated into one `cohort_measurements` row:

- **Counts:** `sample_size` (number of turn events), failure-mode breakdowns
- **Violation rates:** `ttft_violation_rate`, `tpot_violation_rate`, `combined_violation_rate` (the headline) and the corresponding `*_target_miss_rate` against the looser target threshold
- **Percentiles:** `ttft_p50_ms`, `ttft_p95_ms`, `tpot_p50_ms`, `tpot_p95_ms`, plus `ttfct_p50/p95` for reasoning models
- **Wilson CI** on the combined violation rate (95% bounds)
- **Pass/marginal/fail status** computed via Wilson upper/lower bounds (§9)
- **Token throughput aggregates:** `prompt_tok` (input + history), `content_tok` (output), `reasoning_tok` summed across all turns in the window; divided by `measurement_duration_s` to produce per-second rates

## 8. Stepper modes

The stepper picks which `pool_size` to measure next. Two modes are available; **fixed-grid is the default** (opt into adaptive with `--adaptive`).

### 8.1 Fixed-grid stepper (default)

[`FixedGridStepper`](../simulator/adaptive.py) walks a pre-defined grid in order with **early-stop one step past the first observed failure**.

Default grid: `[4, 8, 16, 32, 64, 128, 256]` (powers of 2). Override via `--pool-sizes`.

Algorithm:

1. Yield grid entries in order via `next_pool_size()`
2. After each step's `record(StepResult)`, check `violation_rate >= stop_violation_threshold` (default 0.5)
3. On the first failure, truncate the remaining grid to keep exactly one more grid point (which doubles the failure pool by construction for powers-of-2) then stop
4. Subsequent failures don't re-truncate (the post-failure confirmatory step is preserved)

Rationale: capacity-curve generation wants uniform x-axis density across powers of 2, not knee localization. The early-stop bounds wall-clock without losing the curve shape past the knee — measuring far past the cliff is duplicative.

### 8.2 Adaptive two-knee stepper (opt-in)

[`TwoKneeStepper`](../simulator/adaptive.py) — five-phase Wilson-CI-aware bisection that locates two knees plus infill points. Use when you care about precise knee placement rather than uniform sampling.

- **Phase 1 (DOUBLING):** start at `initial_pool_size`, double until `violation_rate ≥ stop_violation_threshold` OR `max_pool_size`
- **Phase 1b (DOWNWARD_SEARCH):** if the initial pool already fails, halve down looking for an acceptable zone
- **Phase 2 (BISECT_FAIL):** bisect between the largest passing pool and smallest failing pool until gap ≤ `bisect_resolution` (4). Locates `fail_pool_size` (knee 2)
- **Phase 3 (BISECT_TARGET):** bisect between the largest target-passing pool and the smallest target-missing pool. Locates `soft_capacity_pool_size` (knee 1)
- **Phase 4 (INFILL):** add midpoint measurements between fast_max → knee 1 and knee 1 → knee 2 for curve density. Skipped if existing measurements are already within ±10% of the target

Wilson-CI gating: phase transitions use the upper/lower CI bound on the violation rate (not the point estimate) so a noisy small-sample 20% measurement doesn't get treated the same as a clean 20% from n=100.

### 8.3 Spot-check override

A third path exists for audit-driven re-measurement of specific pool sizes: `pool_size_override`. Used by `run_spot_check`. This bypasses both steppers and measures each listed pool size with no early-stop semantics — appropriate for an audit but not for sweep measurement.

## 9. Wilson CI status classification

Each measurement step gets a `pass | marginal | fail` label, computed via the **Wilson 95% confidence interval** on the violation rate ([`measurement.py:_classify_status`](../simulator/measurement.py)):

- `pass` — Wilson upper bound on violation_rate is **strictly below 5%**
- `fail` — Wilson lower bound on violation_rate is **at or above 30%**
- `marginal` — anything between

Why upper/lower bounds instead of point estimates: a measurement of 0 violations out of 38 turns has a Wilson upper bound around 9%, meaning we can't statistically rule out a true rate up to ~9%. Calling that "pass" would be over-confident on a small sample. With n=74+ at zero observed violations, the upper bound drops below 5% and the step gets promoted.

The same Wilson logic gates adaptive-stepper phase transitions. **In fixed-grid mode the label is cosmetic** — the stepper advances regardless of label, only early-stopping on point-estimate `violation_rate ≥ 0.5`.

## 10. Reasoning models (GPT-OSS, etc.)

When `engine.reasoning: true` is set in the config, the simulator handles models that stream chain-of-thought before the answer:

### 10.1 Token-budget bump

`REASONING_OVERHEAD_TOKENS` adds headroom to `max_tokens` so the reasoning phase doesn't eat the entire budget and leave no room for content:

| effort | overhead tokens |
|---|---|
| minimal | 100 |
| low | 250 |
| medium | 600 |
| high | 1200 |

Calibrated from observed GPT-OSS-20B reasoning-phase lengths. Without this bump, short-output personas (quick_lookup with 60-token median answer) truncate during reasoning and never emit content, producing `no_content_tokens` errors on every turn.

### 10.2 TTFT vs TTFCT split

- `ttft_ms` — time to first user-visible token (reasoning OR content)
- `ttfct_ms` — time to first content/answer token

For non-reasoning models, `ttft_ms == ttfct_ms` trivially. For reasoning models, the gap quantifies "user sees model thinking" responsiveness vs "model starts answering."

### 10.3 TPOT over visible tokens

TPOT is computed across `output_tokens + reasoning_tokens` so the per-token decode rate is honest for reasoning models — the engine is genuinely doing decode work during reasoning, even though the content phase hasn't started.

## 11. Phase-distribution timeline

[`simulator/timeline.py`](../simulator/timeline.py) reconstructs second-by-second user counts in each lifecycle phase from `turn_events` + `virtual_users`. Used for query-storm / oscillation diagnosis.

### 11.1 Phases

- **prefill** — request submitted, awaiting first token (`[submit, submit + ttft)`)
- **decode** — streaming tokens (`[submit + ttft, complete)`)
- **think** — sleeping until next request. Covers four physically-identical cases (all a user waiting to fire):
  1. Between-turn read+active_think gap inside a session.
  2. Pre-first-turn initial phase offset window (`[spawn, first_submit)` clamped to window bounds).
  3. **Trailing think** — from a user's last completed turn to either their termination or window-end, whichever is earlier. Without this the chart drained toward window-end as users fell off after their last completion.
  4. **Alive-but-turnless** — users spawned during the window whose first turn doesn't complete before window-end. At high concurrency on slow cohorts (long-prompt AMD, etc.) this can be a substantial fraction of the pool. They sit in pre-first-turn think for their entire alive-in-window interval. The timeline enumerates `virtual_users` (alive predicate: spawn ≤ window_end AND (terminate IS NULL OR terminate ≥ window_start)) so these users are visible even though they have zero rows in `turn_events`.

At any timestamp, `prefill + decode + think` should equal the active pool size (within measurement-window edge effects: a user whose warmup turns were drained before measurement starts appears in `think` from window-start until their first measurement-window submit).

### 11.2 Sweep-line algorithm

Each phase interval emits two events: `(timestamp, phase_index, +1)` at start, `(timestamp, phase_index, -1)` at end. Events are sorted (ties: close-before-open to avoid double-counting boundary instants) and replayed in order, snapshotting counters at each `resolution_ms` tick. O(events + samples).

### 11.3 Export shape

One `timeline` block per curve step:

```json
"timeline": {
  "resolution_ms": 1000,
  "schema": ["t_offset_s", "prefill", "decode", "think"],
  "rows": [[0, 0, 0, 8], [1, 2, 0, 6], ...]
}
```

Compact array-of-arrays. ~6 KB per step, ~450 KB for a full sweep. Replaces the prior cohort-level `snapshots` field which only carried aggregate in_flight (no phase split).

### 11.4 Storm detection

Storms manifest as periodic spikes in `prefill` count coincident with drops in `think`. The aggregate p95 TPOT of a step gets dragged up by the latency degradation during storms, even when the median (typical) experience is healthy. See `simulator/audit.py` for a rolling-window p95 analysis that distinguishes "steady degradation" from "occasional spike."

## 12. SLA framework

Two parallel readings of every measurement:

### 12.1 Failure-bound (hard SLA, headline capacity)

- Counts turns where TTFT > `ttft_failure_seconds` OR TPOT > `tpot_failure_ms` as violations
- The combined `violation_rate` gates `capacity_status` (pass/marginal/fail per §9)
- `capacity_pool_size` = last pool with `capacity_status='pass'` before the knee

### 12.2 Target-bound (premium quality)

- Same logic against the looser `target_*` thresholds
- Produces `target_status` and `target_capacity_pool_size`

A user "noticed it was slow but didn't abandon" is a target_miss without being a violation. The export surfaces both numbers so the buyer page can report "supports N users at SLA, M users at premium quality."

## 13. Reproducibility

All random sampling is seeded. Two runs with the same:

- Pool RNG seed (`0xC0FFEE`)
- Persona definitions
- Cohort weights
- Tokenizer corpus

…will produce byte-identical sequences of persona assignments, prompt token counts, and think-time samples. Wall-clock outcomes still vary because they depend on engine response time, but the **load profile** is deterministic.

## Quick reference — configuration knobs

In `SimulationConfig` (loaded from YAML, `simulator/config.py`):

| field | default | meaning |
|---|---|---|
| `initial_pool_size` | 4 | starting pool for adaptive stepper |
| `max_pool_size` | 1024 | adaptive stepper ceiling |
| `target_samples_per_step` | 500 | measurement window sample cap |
| `measurement_timeout_s` | 300 | measurement window time cap |
| `ramp_spawn_interval_s` | 1.0 | seconds between consecutive user spawns |
| `initial_phase_offset_enabled` | True | random per-user offset at spawn |
| `warmup_min_duration_s` | 30 | minimum warmup before convergence check |
| `warmup_max_duration_s` | 300 | hard warmup ceiling |
| `convergence_window_s` | 60 | throughput-comparison window |
| `convergence_threshold` | 0.20 | relative-change threshold for "converged" |
| `convergence_min_completions_per_window` | 5 | sample floor for valid convergence comparison |
| `stop_violation_threshold` | 0.5 | violation rate that triggers stepper stop / fixed-grid early-stop |

## Glossary

- **Cohort** — a weighted mix of personas representing a team type
- **Persona** — one user archetype with token-count, think-time, and SLA distributions
- **Pool** — the set of active virtual users at a given time
- **Pool size** — target concurrent active sessions (post-respawn semantics)
- **Step** — one (cohort, pool_size) measurement; produces one `cohort_measurements` row
- **Run** — one engine launch driving N steps across one or more cohorts
- **Sweep** — one or more cohorts run sequentially against the same engine launch
- **TTFT** — time to first token (user-visible — reasoning or content)
- **TTFCT** — time to first content token (= TTFT for non-reasoning models)
- **TPOT** — time per output token (`(e2e - ttft) / (visible_tokens - 1)`)
- **Violation** — TTFT or TPOT past the persona's `*_failure_*` threshold
- **Target miss** — past the persona's `*_target_*` threshold (looser)
- **Wilson CI** — confidence interval on a binomial proportion; used for status classification
- **Capacity** — last pool size where the measurement step passed the SLA gate
