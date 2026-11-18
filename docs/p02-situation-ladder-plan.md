# P02: situation ladders for the Xeon KV tier

Status: all phases measured 16 to 18 November 2026; see "Phase 1 results", "Repeats, phase 2 and phase 3 results" and "Scored against the 60-second SLA".

## Purpose

Measure how many users one XE7740 serves with and without the Xeon KV tier, in
situations a reader recognises, and produce one data set an interactive tool can
read. The tool lets a reader pick a situation and see what the tier changes.

The 2 October results (campaigns 2ce1867b and e78484ae) stay valid for first-token
wait and rebuilt tokens at 80 and 96 users. They do not measure users per server:
population was fixed and the request rate was raised until each user asked every
20 seconds. This plan makes population the variable.

## Ground rules

1. **The baseline gets its best case.** Every ladder starts at a population that
   fits in GPU memory, where GPU-only should work well.
2. **Users pace themselves.** One outstanding question per user, then an
   exponential think time. No open-loop arrival rate and no round-robin order.
3. **Every rung is recorded.** Wait targets are applied afterwards, so the tool
   can offer 2, 5, 10 and 30 second targets from the same runs.
4. **The limits are part of the result.** Each situation includes rungs where the
   tier does not help or stops helping.
5. **Nothing is extrapolated.** The tool shows measured rungs, labels
   interpolation between rungs, and shows "not measured" beyond the last rung.

## Fixed configuration

Same as the white paper, so old and new results line up.

| Item | Value |
|---|---|
| Server | XE7740, 2 × Xeon 6787P, 2 TB, 8 × RTX PRO 6000 Blackwell 96 GB |
| Model | GPT-OSS-120B, MXFP4 weights, FP8 KV, 131,072-token limit |
| Serving | One vLLM replica per GPU under Dynamo, pinned image, fixed user-to-replica affinity |
| GPU-only arm | Prefix caching on, automatic GPU KV sizing |
| Tier arm | Same, plus SimpleCPUOffloadConnector, NUMA-local pinned host cache |
| Fixture | 16 records placed after neutral padding, two-record exact-match questions, 512 new tokens |
| Default wait target | First token within 5 s for 95% of questions, at least 95% correct |

Host cache is 192 GiB per replica (768 GiB per socket, 1.5 TB in total) for every
tier cell, including small populations, so one configuration covers all rungs.

## Sizing behind the rungs

These are the measured numbers the rungs are built around.

| Quantity | 96K | 48K | 24K |
|---|---|---|---|
| Contexts that fit on 8 GPUs (897K tokens per GPU) | about 74 | about 149 | about 299 |
| Contexts that fit in 192 GiB of host cache per replica, 8 replicas | about 610 | about 1,220 | about 2,440 |
| Cold rebuild, one request | 18.3 s | about 4.5 s (between 2.6 s at 32K and 6.5 s at 64K) | about 1.7 s |

Compute ceiling with the tier at 96K: 2.4 to 4.0 questions a second inside 10
seconds (measured at 96 and 80 users). At a five-minute think time that is 720 to
1,200 users, so at 96K host memory (about 680 users with the GPU share) should
bind first. At a one-minute think time compute should bind first, near 144 to 240
users. Both are expectations to be tested.

GPU-only rebuild ceiling at 96K: 8 GPUs ÷ 18.3 s is about 0.44 questions a second.
At a five-minute think time that is reached near 130 users.

## Situations

Each situation is one view in the tool. S1 and S2 are the minimum for a headline.

### S1. Long documents, occasional questions (headline ladder)

Analysts or reviewers each keep a long document open and ask a question every few
minutes. The GPUs are mostly idle, but their memory is full.

- Context 96K, think time 300 s mean, 30-minute windows (about six questions per user).
- Users: 56, 64, 72, 80, 96, 144, 224, 336, 480, 640.
- Tier arm: every rung. GPU-only arm: 56 through 224, then upward only until the
  failure is in the data (added 16 November, for the replay demo's GPU-only side).
  Stop rule: run the next GPU-only rung only if the one below still answered most
  questions inside the 300 s deadline. Eight GPUs rebuild about 0.44 conversations
  a second, so the expected wait is about 210 s at 224 users and 460 s at 336.
  Expected outcome: one extra cell at 336 (about 50 minutes), none at 480. Above
  the first rung that mostly times out, the demo shows "failed at N users".
- The 56 to 96 rungs are also the **team-growth cliff** view: what happens when a
  server that fits 72 users is given 80.
- Expected: GPU-only holds to about 72 and degrades to roughly 20 s median by 96.
  The tier holds to about 640 or until host memory fills. The 640 rung is chosen
  to sit just past the host cache, so the tier's own ceiling is on the chart.

### S2. Everyone comes back at once

After a meeting or lunch, a share of idle users return inside one minute.

- Context 96K, 224 users (three times GPU memory), all primed, then 10 minutes quiet.
- Three storms in one launch per arm, using disjoint users: 10%, 25% and 50% of
  users each ask one question inside 60 seconds, with 10 minutes between storms.
- Reported: wait for the first word (median, 95th percentile) and time until the
  last returning user is answered.
- Expected: GPU-only needs about 56 rebuilds at 25%, around two minutes to clear.
  The tier should clear in seconds.

### S3. Shorter documents, more people

The same question pattern with smaller contexts, which is the tool's document
length control.

- 48K, think 300 s. Users: 128, 160, 224, 336, 640, 960, 1,280. Tier arm every
  rung, GPU-only 128 through 336.
- 24K, think 300 s. Two pairs only: 224 users (fits on the GPUs, expect no
  difference) and 640 users (rebuild is under 2 s, expect a small difference).
- The 24K pairs are the "where it does not help" boundary.

### S4. Busier users

Shorter think times, which is the tool's activity control. This shows where GPU
compute takes over from memory as the limit.

- 96K, think 120 s. Users: 96, 144, 224, 336, 480.
- 96K, think 60 s. Users: 96, 144, 224, 336.
- Tier arm every rung. GPU-only at 96 and 144 only.
- Expected: the tier's advantage narrows to roughly 2 to 3 times at 60 s.

### S5. Agents waiting on tools (second round)

A coding or research agent whose context grows each turn and pauses while a tool
runs. This reuses the stage 2 growing-session harness, arms A and B only.

- 8K prefix, 9K tokens per turn, 75 s think, 48 turns of fixed work, reset at 96K.
  Same shape as campaign c38a4714, so its 56 and 96 user cells are reused.
- New rungs: tier at 144, 224, 336; GPU-only at 144, 224.
- Each cell is about 95 minutes, so this is the most expensive situation.

## Run order and time

| Phase | Content | Cells | Box time |
|---|---|---|---|
| 0 | Smoke: 192 GiB tier launches on all replicas; 72 users GPU-only at 96K passes; closed-loop pacing matches the old 96-user result | 3 | 2 h |
| 1 | S1 and S2 | 19 | 13 h |
| 2 | S3 and S4 | 28 | 15 h |
| 3 | S5 | 5 | 8 h |

Phase 1 alone replaces the arithmetic user count with a measured one. Phases 2
and 3 exist to give the tool its controls and can wait for a decision after
phase 1.

A cell is about 8 minutes of launch, priming (users × rebuild time ÷ 8), and the
window. Results go to `/data/system-sizing/runs` (231 GB free); `/data2` has 43 GB
free and should only hold the plan files.

## Harness support

Implemented 16 November 2026 and covered by `tests/test_infrastructure_session_paced.py`.
None of it has run against a GPU yet; phase 0 is the first real check.

| Need | Where | Setting |
|---|---|---|
| Self-paced users on the padded document fixture | `session_paced.py` | `session_pacing: "think"`, `session_think_s` |
| Return surge | `session_paced.py` | `session_pacing: "surge"`, `session_surge_shares`, `session_surge_quiet_s`, `session_surge_window_s` |
| One arm only on the high rungs | `campaign.py` | `session_cache_arms: ["treatment"]` |
| Wall power per cell | `wall_power.py` | `session_power_env` (iDRAC env file on the box) |
| Host cache occupancy and evictions | `native_cpu_connector.py`, `session_native.py` | automatic in the tier arm |
| Ten-second timeline and flat export | `session_cells.py` | automatic; writes `cells.json` and `timeline.json` |

Think cells score the final half of the window by arrival, as the 2 October
ladder did. Timed-out questions stay in every fraction and are counted at the
request deadline in the percentiles.

Plan files are in `docs/examples/p02-situations/`:

| File | Cells | Use |
|---|---|---|
| `phase0-smoke.json` | 72 and 96 users, both arms | Phase 0; also the first two S1 rungs |
| `s1-ladder-both-arms.json` | 56, 64, 80, 144, 224 users, both arms | S1 |
| `s1-ladder-tier-only.json` | 336, 480, 640 users, tier only | S1 |
| `s1-gpu-only-336.json` | 336 users, GPU-only | S1 failure step, subject to the stop rule |
| `s2-return-surge.json` | 224 users, both arms | S2 |

Launch each with `session_deferred <plan.json> --launch` from a side-loaded tree,
as for the earlier ladders. A finished campaign can be re-exported with
`python -m simulator.infrastructure.session_cells <campaign dir>`.

Phase 0 passes when: all eight replicas start with the 192 GiB cache, priming
passes the quality gate, GPU-only at 72 users answers inside the 5 s target, the
96-user cells land near the 2 October results (about 20 s GPU-only, under 1 s
with the tier), and `cells.json` has power and host cache fields filled.

## Phase 1 results

Measured 16 to 17 November 2026 on the XE7740 from `/data2/p02-20261116b`. One
30-minute run per cell, scored on its final 15 minutes; the 72-user tier cell was
run twice and agreed to 0.01 s. Campaigns are under
`/data/system-sizing/runs/infrastructure/`, each with a `cells.json`.

| Campaign | Content |
|---|---|
| `01029e4365cd40ecaada065f4d987f46` | 72 and 96 users, both arms |
| `f9496d83edc54edebb37749817b94cd9` | 56, 64, 80, 144, 224 users, both arms |
| `03461be322104378bb967d4260522af3` | 336, 480, 640 users, tier only |
| `16dd977711044ae786a889391401f6da` | 336 users, GPU-only |
| `2116a0943e0d4ec3a91304ba501835a9` | S2 return surge, 224 users, both arms |

### S1: 96K documents, five-minute think time

Share of questions with a first token inside 5 s, median first-token wait, wall
power and energy per correct answer.

| Users | GPU-only within 5 s | GPU-only median | GPU-only W | GPU-only Wh | Tier within 5 s | Tier median | Tier W | Tier Wh | Host cache filled |
|---|---|---|---|---|---|---|---|---|---|
| 56 | 98.4% | 0.33 s | 1,882 | 2.50 | 98.4% | 0.33 s | 1,864 | 2.48 | 12% |
| 64 | 87.3% | 0.33 s | 2,044 | 2.52 | 99.5% | 0.33 s | 1,863 | 2.20 | 14% |
| 72 | 65% | 0.34 s | 2,441 | 2.83 | 100% | 0.33 s | 1,871 | 2.01 | 16% |
| 80 | 49.8% | 3.3 s | 3,009 | 3.31 | 99.2% | 0.33 s | 1,863 | 1.83 | 18% |
| 96 | 29% | 18.3 s | 4,270 | 3.98 | 99% | 0.33 s | 1,920 | 1.58 | 21% |
| 144 | 6% | 32.3 s | 5,383 | 4.07 | 98.7% | 0.35 s | 2,045 | 1.09 | 32% |
| 224 | 0% | 81 s | 6,113 | 4.55 | 99.3% | 0.60 s | 2,230 | 0.78 | 49% |
| 336 | 0% (98.8% timed out) | 300 s | 6,011 | n/a | 99.3% | 0.61 s | 2,530 | 0.60 | 74% |
| 480 | not run | | | | 99.2% | 0.62 s | 2,807 | 0.47 | 100%, evicting |
| 640 | not run | | | | 85.8% | 0.65 s | 3,555 | 0.48 | 100%, evicting |

At the default target (95% inside 5 s, 95% correct) GPU-only serves 56 users and
the tier serves 480, with 192 GiB of host cache per replica. GPU-only misses at
64 by a margin (87%), then degrades steadily; it is out of GPU compute at 224
(99% busy) and times out at 336. The tier's limit is host memory: the cache is
full at 480 and the 640 rung has a 15 s 95th-percentile wait.

GPU-only held fewer users than the memory arithmetic suggested (56, not 72):
nine 96K documents per GPU leave no room for question and answer tokens.

The rebuilt-token counter is unreliable in cells with a deep queue (the engine
recounts waiting requests), so it is left out of this table.

### S2: return surge, 224 users

Users returning inside 60 seconds after ten idle minutes. Time to clear is from
the first return to the last answer, so 60 s is the floor.

| Share returning | Users | GPU-only median / 95th pct first token | GPU-only time to clear | Tier median / 95th pct first token | Tier time to clear |
|---|---|---|---|---|---|
| 10% | 22 | 22 s / 48 s | 109 s | 0.62 s / 0.72 s | 64 s |
| 25% | 56 | 48 s / 116 s | 191 s | 0.61 s / 0.88 s | 64 s |
| 50% | 112 | 110 s / 214 s | 314 s | 0.62 s / 0.98 s | 63 s |

No GPU-only surge question had a first token inside 10 s; every tier surge
question that was answered correctly had one inside 2 s.

### Harness changes made during the runs

- The host-cache hook now counts `cached_block_hash_to_block`, the map the
  pinned vLLM host tier fills.
- Container removal is verified for up to ten minutes; a replica with 192 GiB
  pinned takes longer than the earlier 30 s to release it.

## Repeats, phase 2 and phase 3 results

Measured 17 to 18 November 2026, same fixed configuration, one arm per
campaign. "Inside 5 s" is the share of questions with a first token inside
5 s in the scored window; power is server wall power.

### Repeats (96K, 5-minute think)

| Cell | Inside 5 s, first run | Repeat | Median first token | Wall power |
|---|---|---|---|---|
| GPU-only 56 | 98.4% | 98.4% | 0.32 s | 1,819 W |
| GPU-only 64 | 87.3% | 83.2% | 0.33 s | 2,068 W |
| Host tier 480 | 99.2% | 99.1% | 0.62 s | 2,770 W |

### S3: shorter documents (5-minute think)

| Context | Cell | Inside 5 s | Median / p95 first token | Wall power | Wh per correct answer |
|---|---|---|---|---|---|
| 48K | GPU-only 112 | 98.3% | 0.18 / 0.19 s | 1,911 W | 1.37 |
| 48K | GPU-only 224 | 31.7% | 5.5 / 12.9 s | 3,392 W | 1.21 |
| 48K | GPU-only 336 | 12.0% | 7.5 / 22.3 s | 4,976 W | 1.30 |
| 48K | Host tier 224 | 99.7% | 0.18 / 0.33 s | 2,128 W | 0.73 |
| 48K | Host tier 640 | 99.7% | 0.33 / 0.45 s | 2,811 W | 0.35 |
| 48K | Host tier 960 | 99.5% | 0.34 / 0.52 s | 3,044 W | 0.26 |
| 48K | Host tier 1,280 | 96.6% | 0.36 / 2.98 s | 3,307 W | 0.21 |
| 24K | GPU-only 640 | 93.0% | 1.9 / 5.2 s | 3,735 W | 0.48 |
| 24K | Host tier 640 | 99.1% | 0.18 / 0.22 s | 2,783 W | 0.35 |

At 48K the tier still passes at 1,280 users with the host cache full
(4.3 million evictions), so its limit is at or just above 1,280; GPU-only
holds 112 (the lowest rung run) and fails at 224. At 24K GPU-only nearly
keeps up at 640 users: the benefit there is response time and about 950 W,
not a multiple of users. The tier cache is 38% full at that point.

### S4: busier users (96K)

| Think time | Cell | Inside 5 s | Median / p95 first token | Wall power | Wh per correct answer |
|---|---|---|---|---|---|
| 60 s | GPU-only 56 | 99.3% | 0.33 / 0.35 s | 2,392 W | 0.70 |
| 60 s | GPU-only 96 | 0% | 60 / 120 s | 5,799 W | 4.06 |
| 60 s | Host tier 144 | 99.4% | 0.60 / 0.95 s | 3,032 W | 0.35 |
| 60 s | Host tier 224 | 99.6% | 0.66 / 1.21 s | 3,413 W | 0.27 |
| 60 s | Host tier 336 | 88.6% | 1.21 / 6.5 s | 3,665 W | 0.22 |
| 120 s | GPU-only 56 | 97.8% | 0.33 / 0.35 s | 2,110 W | 1.19 |
| 120 s | Host tier 224 | 99.6% | 0.61 / 0.93 s | 2,970 W | 0.43 |
| 120 s | Host tier 336 | 99.3% | 0.64 / 1.09 s | 3,247 W | 0.32 |
| 120 s | Host tier 480 | 98.2% | 0.76 / 2.36 s | 3,563 W | 0.26 |

Think time does not move the GPU-only limit: it is set by how many
conversations fit in GPU memory. At 120 s the tier holds 480 users, close to
its limit (cache full, 540K evictions). At 60 s the tier passes at 224 with
GPUs 83% busy and misses the target at 336 (88.6%, every question still
answered, host cache 75% full): the limit there is GPU time, not memory.

### S5: agents waiting on tools

Knowledge-worker sessions (8K prefix, 9K per turn, 75 s tool wait, 48 turns,
reset at 96K), no compaction. This harness has no cells export or wall power.

| Cell | Median / p95 first token | Median / p95 turn | Turns per second | Time to finish all sessions | Prefix tokens the GPU reprocessed |
|---|---|---|---|---|---|
| GPU-only 144 | 19.1 / 54.3 s | 63.7 / 155 s | 0.79 | 8,526 s | 94% |
| Host tier 144 | 2.9 / 9.4 s | 8.8 / 27.7 s | 1.07 | 6,305 s | 20% |
| Host tier 224 | 4.3 / 20.5 s | 19.2 / 69.4 s | 1.57 | 6,723 s | 70% |

At 144 agents the tier cuts the median turn from 64 s to 9 s and finishes the
same work 26% sooner. At 224 the tier is past its comfortable point. Answer
quality is 59% in every cell, the known GPT-OSS weakness at this context
length, so these runs support the latency and work figures only.

### GPU-only with a 20-minute deadline (96K, 5-minute think)

Run for the replay demo, so every rebuild finishes and is answered.

| Cell | Questions answered | Median / p95 first token | p95 full answer | Wall power | Wh per correct answer |
|---|---|---|---|---|---|
| GPU-only 224 | 776 of 776 | 91 / 155 s | 321 s | 5,792 W | 3.82 |
| GPU-only 336 | 896 of 896 | 323 / 419 s | 593 s | 6,636 W | 4.47 |

With the 300 s deadline the 336-user cell answered 118 of 1,052: demand is
above the rebuild rate, the engine serves the oldest question first, and that
question gives up mid-rebuild. Timeouts are cancelled on the server (running
plus waiting matches the client's outstanding count on every GPU), so that
collapse is real, not a runner fault. With nobody giving up, the same load
is served at about 24 first tokens a minute and a five-minute median wait.

### Harness fault found and fixed

Three tier campaigns stopped mid-cell at the highest question rates
(60 s think at 224 and at 336 users, 48K at 1,280 users) with a missing
temporary file. Cause: the telemetry disk-budget check lists the run
directory and then stats each file, and the engine's evidence file is
written and renamed between the two. `simulator/benchmarks/worker.py` now
skips files that vanish. The 224 and 1,280 cells passed on rerun, and 60 s think
at 336 completed from a tree with the fix (`/data2/p02-20261116d`).

## Scored against the 60-second SLA

The service target is a first token inside 60 s for 95% of questions (decided
18 November 2026). The runs kept a 300 s give-up deadline so that slow
requests execute and are recorded instead of timing out; only GPU-only at 224
(40 gave up) and 336 (488 of 494) had timeouts in the scored window, and the
20-minute-deadline runs cover those two cells. Every cell recorded the 60 s
score, so this section rescores the same runs. The 5 s figures above stay as
the stricter view.

Share of questions inside 60 s, first token / complete answer:

| Situation | Users | GPU-only | Host tier |
|---|---|---|---|
| 96K, 5-min think | 56 | 98.9% / 98.9% | 98.4% / 98.4% |
| | 64 | 99.0% / 99.0% (repeat 98.0% / 97.0%) | 99.5% / 99.5% |
| | 72 | 99.5% / 99.5% | 100% / 100% |
| | 80 | 98.7% / 93.4% | 99.2% / 99.2% |
| | 96 | 95.6% / 77.0% | 99.0% / 99.0% |
| | 144 | 86.8% / 33.3% | 98.7% / 98.7% |
| | 224 | 33.0% / 4.3% (20-min deadline: 25.9% / 4.2%) | 99.3% / 99.3% |
| | 336 | 0% / 0% | 99.3% / 99.3% |
| | 480 | not run | 99.5% / 99.5% (repeat 99.3%) |
| | 640 | not run | 99.7% / 99.5% |
| 96K, 60 s think | 56 | 99.3% / 99.3% | not run |
| | 96 | 49.6% / 5.9% | not run |
| | 144 | not run | 99.4% / 99.4% |
| | 224 | not run | 99.6% / 99.6% |
| | 336 | not run | 99.4% / 99.4% |
| 96K, 120 s think | 56 | 99.4% / 99.4% | not run |
| | 224 to 480 | not run | 99.3% to 99.6% |
| 48K, 5-min think | 112 | 99.4% / 99.4% | not run |
| | 224 | 99.7% / 99.3% | 99.7% / 99.7% |
| | 336 | 99.7% / 87.0% | not run |
| | 640 to 1,280 | not run | 99.7% / 99.7% |
| 24K, 5-min think | 640 | 99.2% / 99.2% | 99.1% / 99.1% |

Users served at the SLA (highest passing cell):

| Situation | GPU-only | Host tier | Ratio |
|---|---|---|---|
| 96K, 5-min think | 96 (fails at 144) | 640, highest run | at least 6.7x |
| 96K, 60 s think | 56 (fails at 96) | 336, highest run | at least 6x |
| 96K, 120 s think | 56, only cell run | 480, highest run | not bounded |
| 48K, 5-min think | 336, highest run | 1,280, highest run | not separated |
| 24K, 640 users | passes | passes | none |

- The tier's ceiling was not reached at 60 s in any situation, so each ratio
  is a lower bound. The tier's misses are wrong answers, not slow ones.
- GPU-only at 96 users passes on the first token only: 77% of complete
  answers finish inside 60 s. On a complete-answer SLA its 96K limit is 72.
- At 48K a 60 s first-token SLA does not separate the arms at the populations
  run; the gap there shows in the complete answer (87.0% at 336) and in the
  5 s view (112 against 960 or more).

## Data the tool reads

One record per cell.

| Group | Fields |
|---|---|
| Identity | situation, arm, users, context tokens, think time, storm share, host cache GiB, campaign id |
| Wait | first-token median and 95th percentile; full-answer median and 95th percentile; share of questions inside 2, 5, 10, 30 and 60 s |
| Work | questions answered per minute, share correct, share timed out |
| Cache | tokens per question from GPU cache, from host cache, rebuilt; host cache used; host evictions |
| Machine | GPU busy %, host CPU busy %, wall power, watt-hours per correct answer |
| Timeline | the ten-second series from item 5 above |

The tool derives "users served at the chosen target" as the largest rung that
passes with every smaller rung also passing. Controls map to situations: users
(S1), return surge (S2), document length (S3), activity (S4), agent sessions (S5).

## What is not in this plan

- Other models, tensor-parallel layouts and the R7725. One model keeps the
  comparison simple; the per-context memory arithmetic lets a reader rescale.
- Repeat blocks. The 80 and 96 user campaigns already agree closely. Repeat only
  the rungs either side of each arm's limit if phase 1 looks noisy.
- Context compaction arms. That is the stage 2 paper.

## Roadmap after the ladders

Decided 16 November 2026. None of this is implemented or run.

1. **Replay export for the interactive demo.** A compact per-cell file with each
   question labelled GPU hit, host reload or rebuild. The tier arm's per-request
   record does not separate GPU hits from host reloads, so the label comes from
   matching each replica's reload counter to its questions. The demo may
   interpolate between measured rungs with a footnote.
2. **Other model sizes, one replica per GPU.** A short calibration per model
   (cache bytes per token, weights footprint, rebuild time at four lengths,
   reload time, answer speed) plus two validation rungs in both arms. Small end:
   GPT-OSS-20B. Large end: see the candidates below.
3. **Capped GPU cache on GPT-OSS-120B.** `session_gpu_gb` at 8 and 16 GB, 96
   users, both arms. This varies only the memory left for context.
4. **Multi-GPU replicas (tensor parallel above 1). Roadmap only.** The pinned
   vLLM 0.26.0 connector divides `cpu_bytes_to_use` by world size and accepts
   `cpu_bytes_to_use_per_rank`, so the code path exists. Untested here. Needs the
   launcher's fixed `--tensor-parallel-size 1` lifted for P02 and per-rank socket
   pinning, then a four-GPU smoke test.

Large-end candidates checked on 16 November. Weights size does not predict how
few conversations fit, because recent MoE models keep little cache per token.

| Model | Weights | Cache per token | Note |
|---|---|---|---|
| Nemotron 3 Super 120B NVFP4 (March 2026) | 75 GB, staged | Small: 8 attention layers of 88, the rest Mamba and MoE | Runs on stock vLLM. Connector with Mamba state is untested |
| Qwen3.8-Flash-Next FP4-Local (August 2026, quant 5 October) | 96 GiB, not staged | About 12 KB (12 full-attention layers of 48, 2 KV heads, 256 head dimension); model card states 850K tokens of FP8 cache on one RTX PRO 6000 | Needs a community vLLM container. From the checkpoint headers: 68.2 GiB on the GPU, 26.9 GiB embedding table in host memory per replica. Not yet loaded on the server |
| Gemma 4 31B NVFP4 | 30 GB, staged | Large: dense, 16 KV heads, 256 head dimension | Does not fill the GPU with weights; sliding-window layers |
