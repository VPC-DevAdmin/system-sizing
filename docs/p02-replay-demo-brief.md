# P02 replay demo: brief and data hand-off

Written 17 Nov 2026 (UTC) for a session that will build the interactive replay demo.
Everything here is unpublished, preliminary data. Do not publish, push or commit without asking.

## What to build

A single-page interactive demo for the Dell and Intel Xeon KV cache offload story.

- **Two sliders:** users per system, and context per user. Short tooltips say what is typical.
- **Two animations side by side:** the same users on GPU memory only, and on GPU memory backed by host memory. Each shows the 8 GPUs, the conversations held on each, and every question as it arrives: answered from GPU memory, reloaded from host memory, or rebuilt by the GPU.
- **Hero numbers under each side:** wait for the first word (typical and slow case), share of questions with a first word inside the 60-second SLA (5 seconds as the stricter view), wall power, energy per correct answer.
- **Measured points are replayed, not simulated.** The users slider snaps to the measured steps and plays the recorded requests. Between steps the demo may estimate, with a footnote that says so (the user approved this on 16 Nov).
- **Show the limits.** At 56 users both sides are identical. Above the point where GPU-only stops answering inside the deadline, show "failed at N users", not an extrapolated number.

The user's words: "a simple and compelling replay demo where a user has a slider for users per system and context per user with a few tooltips on what is typical and we show side by side animations of the sessions either on GPU and rebuilding or GPU with CPU backed cache, along with the hero numbers."

## The test in one paragraph

Dell PowerEdge XE7740, 2 Intel Xeon 6787P, 2 TB memory, 8 NVIDIA RTX PRO 6000 Blackwell (96 GB each). GPT-OSS-120B, one vLLM replica per GPU. Each user holds one 96,000-token conversation and asks a question about it on average every 5 minutes. Each GPU has room for about 9 such conversations (about 74 on the box). In the host-memory arm each replica also gets 192 GiB of host memory for conversations that no longer fit, about 76 more per GPU. A conversation that is not in GPU memory is either reloaded from host memory (about 0.3 s extra) or rebuilt by the GPU from the text (about 18 s at 96,000 tokens, and longer when rebuilds queue). Each step runs 30 minutes and is scored on the final 15. Users are pinned to a replica: `replica = session % 8`.

Background, in order of usefulness:
- `docs/p02-situation-ladder-plan.md` (test plan, sizing arithmetic, roadmap)
- White paper v2 build script, which holds the implementation wording already checked against the code: `<DELL>/2026 Q4 Deliverables/Xeon KV Cache Offload/source/kv4-set/wp2.py` (imports `wp.py` beside it)

`<DELL>` = `/Users/chrisbranch/Library/CloudStorage/OneDrive-VectorPathConsulting/Vector Path Consulting - Documents/General/Dell`

## The data

`<DELL>/2026 Q4 Deliverables/Xeon KV Cache Offload/source/replay-data/` (43 steps)

- `index.json`: one summary row per step (the table below, plus more fields).
- `gpu_only-<users>.json` and `xeon_tier-<users>.json`: one file per step, about 0.6 MB each.

Each step file:

| Key | Content |
|---|---|
| `summary` | The scored figures for the step (final 15 minutes) |
| `window` | 1,800 s long, scored from 900 s, 300 s deadline. Time zero is the start of measurement |
| `requests` | One row per question over the full 30 minutes: session, turn, replica, arrival_s, first_token_s, answer_s, finish_s, correct, completed, cached_tokens, prompt_tokens, **source** |
| `priming` | The first load of every conversation, before time zero (negative times) |
| `power` | Wall power, about one sample per second: t_s, watts |
| `gpus` | Per second: host CPU busy %, busy % for each of the 8 GPUs, watts for each GPU |
| `replicas` | Per replica, about every 1.6 s: requests running and waiting, GPU cache used, host cache used, cumulative token counters |
| `timeline` | Ten-second buckets: asked, answered, waiting, median wait, tokens by source, host cache fill, GPU busy, wall power |
| `labels` | How the `source` label was derived and how well it checks out |
| `columns` | Column names for the array-shaped series |

`source` is `gpu` (still in GPU memory), `host` (reloaded from host memory) or `rebuilt` (GPU reprocessed it). The raw evidence does not carry this label. The export derives it:
- GPU-only: from the server's cached-token count for the request.
- Host-memory arm: from the replica's host-reload counter across the request's wait. Where another request overlapped on the same replica, from the first-token time (below about 0.47 s is a GPU hit). On every request where the counter gave a clean answer, the time rule agreed (285 to 623 requests per step, 100%).

Regenerate or extend (the script skips steps that have not finished):

```bash
ssh xe7740 'cd /data/capsim/exports && python3 p02_demo_export.py p02-demo-20261117 01029e4365cd40ecaada065f4d987f46 f9496d83edc54edebb37749817b94cd9 03461be322104378bb967d4260522af3 16dd977711044ae786a889391401f6da 2116a0943e0d4ec3a91304ba501835a9 7c4388996f0f4dc9b36e3f658a69bf8f:-repeat 44c6fadc98a0469dabc5cf102382cb60:-repeat a6984db3bfef4f648478026f63225136 3aaa3e1120064c7e862cb1503cd235c1 3c481162e44e42b0a4958d668f7f9b60 86869757df63462c8af1252038559e65 dcbece70dde642cc8e9e6910b14bccd3 d43d179b4ba0482c98e656033e5c5771 04b4536b9eaf4bfbafa14ee0aad42298 eb3e0f222e5845ab9291927f3aa0ed5e 8607b70e85114e919067730b90f32ee8 5173bf96142047a0b9e0e2c8f062ca4a:-long 2c684436708249d09531b0b3c83a5381 f19e3db7776d4852b7aae2dcc3a17c67'
```

File names: `<arm>-<users>.json` is the 96K, 5-minute think step. Other steps add `-48k` (context), `-think60` (think time), `-think120`, `-24k`, `-surge` (return surge), `-repeat` (second run of the same step) or `-long` (GPU-only with a 20-minute deadline, so every question is answered).

For a loopable GPU-only overload clip use `gpu_only-336-long.json` anywhere from 900 s to 1,800 s: about 78 sessions are on GPUs and 110 to 140 queued throughout, and every 30 seconds brings about 12 new questions, 12 first words and 12 finished answers. `gpu_only-224-long.json` is the milder version (about 70 on GPUs, 20 to 35 queued). Each file's `window` gives its own length, scored-from time and deadline: think-time steps other than 5 minutes use shorter windows, and the surge files score the whole window.

Each request row ends with `output_tokens` and `answer_tokens_per_s` (output tokens over the time from first token to the end of the answer). A free GPU answers at about 112 tokens/s. Under GPU-only overload a session that is "answering" gets one token per engine step, and each step also carries a slice of another user's rebuild: the median rebuilt session at 144 users answers at 6.9 tokens/s and takes 45 s to deliver about 400 tokens. Label a session by this speed, not by phase. The speed is an average over the answer; per-token timing was not recorded.

The last column, `gpu_start_s`, is an estimate of when the request got onto its GPU (null if it never did). The engine serves each GPU's requests oldest first and reports how many it is working on, so at each 2-second sample the N oldest outstanding requests on a GPU are taken as the N it reports running. Show a session as "Queued" from `arrival_s` to `gpu_start_s`, as "Rebuilding" from `gpu_start_s` to the first token, then answering. A request with `completed` 0 gave up at the 300-second deadline (`finish_s`); if it has a `gpu_start_s`, the GPU was rebuilding its context when the user gave up, and that work was thrown away. Footnote the start time as estimated.

GPU-only at 336 users is a collapse, not a runner fault. The server's running plus waiting counts match the runner's outstanding questions on every GPU, so a question that times out is cancelled on the server. Demand (4.2 questions a minute per GPU) is above what a GPU can rebuild (about 3.3 a minute), so the queue grows until its oldest question is close to 300 seconds old. The engine always takes the oldest, which then gives up mid-rebuild: 910 of the 934 that gave up were on a GPU, for a median 40 s each. At any moment about 3 sessions per GPU are rebuilding and about 16 are queued. Use 224 users as the main GPU-only overload step (673 of 781 answered); show 336 as the collapse.

Then copy `xe7740:/data/capsim/exports/p02-demo-20261117/*` into `replay-data/`. Source: `scripts/p02_demo_export.py`. Add later campaign ids from `/data2/p02-20261116b/<launch>/readiness.json`.

## The SLA is 60 seconds

The service target is a first word inside 60 seconds for 95% of questions (the user's decision, 18 Nov). The 300-second deadline in the runs is not the SLA: it was set long so slow requests run and are recorded instead of timing out. Each step file has the score in `summary.wait.within_first_token["60"]` (and `within_answer["60"]` for the complete answer); the same maps hold 2, 5, 10 and 30 seconds.

At 96,000 tokens and 5-minute think time GPU-only meets the SLA through 96 users (95.6%) and misses at 144 (86.8%); the host-memory arm meets it at every step through 640, the highest run. At 60-second think time it is 56 against 336 or more. At 48,000 tokens both arms meet it at every step run, so use the 5-second view or the complete-answer score there. Full table: "Scored against the 60-second SLA" in `docs/p02-situation-ladder-plan.md`.

## Measured results (96,000 tokens, 5-minute think time, final 15 minutes)

G = GPU memory only, H = with host memory.

| Users | Arm | First word p50 / p95 (s) | Inside 5 s | Answer p95 (s) | Correct per min | From GPU / host / rebuilt (% of questions) | GPU busy | Wall power (W) | Wh per correct answer |
|---|---|---|---|---|---|---|---|---|---|
| 56 | G | 0.33 / 0.35 | 98.4% | 4.5 | 12.5 | 99 / 0 / 1 | 9% | 1,882 | 2.50 |
| 56 | H | 0.33 / 0.35 | 98.4% | 4.6 | 12.5 | 99 / 1 / 0 | 9% | 1,864 | 2.48 |
| 64 | G | 0.33 / 18.3 | 87.3% | 22.2 | 13.5 | 92 / 0 / 8 | 14% | 2,044 | 2.52 |
| 64 | H | 0.33 / 0.35 | 99.5% | 4.8 | 14.1 | 99 / 1 / 0 | 10% | 1,863 | 2.20 |
| 72 | G | 0.34 / 19.8 | 65.0% | 39.5 | 14.4 | 73 / 0 / 27 | 23% | 2,441 | 2.83 |
| 72 | H | 0.33 / 0.35 | 100% | 4.6 | 15.5 | 100 / 0 / 0 | 11% | 1,871 | 2.01 |
| 80 | G | 3.28 / 28.7 | 49.8% | 68.1 | 15.1 | 59 / 0 / 41 | 33% | 3,009 | 3.31 |
| 80 | H | 0.33 / 0.35 | 99.2% | 4.9 | 17.0 | 99 / 1 / 0 | 12% | 1,863 | 1.83 |
| 96 | G | 18.3 / 52.3 | 29.3% | 121.8 | 17.9 | 39 / 0 / 61 | 50% | 4,270 | 3.98 |
| 96 | H | 0.33 / 0.61 | 99.0% | 5.0 | 20.3 | 82 / 18 / 0 | 14% | 1,920 | 1.58 |
| 144 | G | 32.3 / 75.5 | 6.0% | 209.1 | 22.1 | 8 / 0 / 92 | 81% | 5,383 | 4.07 |
| 144 | H | 0.35 / 0.63 | 98.7% | 5.3 | 31.3 | 52 / 48 / 0 | 21% | 2,045 | 1.09 |
| 224 | G | 81.0 / 300+ | 0% | 300+ | 22.4 | 0 / 0 / 100 | 99% | 6,113 | 4.55 |
| 224 | H | 0.60 / 0.71 | 99.3% | 5.9 | 47.7 | 32 / 68 / 0 | 30% | 2,230 | 0.78 |

Reading it:
- Both arms are the same at 56 users. GPU-only starts to slip at 64 and has lost half its questions to the 5-second target by 80.
- GPU-only at 224 users: 10.6% of questions hit the 300-second deadline, the GPUs are 99% busy, and correct answers per minute stop rising (22.1 at 144, 22.4 at 224). "300+" means the value is cut off at the deadline.
- The host-memory arm holds about 99% inside 5 seconds through 224 users. Host cache is 49% full at 224 with no evictions.
- The few misses in the host-memory arm are wrong answers, which count as misses. They are not slow answers.

Still to come on the server (the queue belongs to the original session, do not touch it):
- Host memory only at 336 (running at the time of writing), 480 and 640 users. Host memory holds about 610 conversations, so 640 should show the tier's own limit.
- GPU-only at 336. It qualified to run because fewer than half of the 224-user questions timed out.
- Return surge: 224 users quiet for 10 minutes, then 10%, 25% and 50% of them come back within 60 seconds.

## What is measured and what is estimated

- **Users slider, at the steps above:** measured. Replay the recorded requests.
- **Users slider, between steps:** estimate, footnoted. Interpolate the hero numbers between neighbours. For the animation, either morph between neighbouring replays or run the simple model below.
- **Context slider:** only 96,000 tokens is measured in this dataset. Other lengths (the plan lists 48,000 and 24,000, about 8 hours of server time, not run) are estimates from the sizing arithmetic: about 897,000 tokens of cache per GPU, 192 GiB of host memory per replica at about 26 KB per token, rebuild about 0.45 / 2.6 / 6.5 / 18.3 s at 8,000 / 32,000 / 64,000 / 96,000 tokens. Label every non-96,000 value as estimated. If the estimate feels too thin to show, ship the first version with the context slider fixed at 96,000 and say so.
- **Other models:** not measured. Out of scope for the first version.
- **Tooltips on what is typical:** no sourced figures yet. Either find sources or word the tooltips as examples.

A simple model for the in-between animation, checked against the labels:
- Host-memory arm: "each GPU keeps its 10 most recently used conversations" reproduces 95 to 98% of the GPU-or-host labels at 80 to 224 users.
- GPU-only arm: the same rule fits poorly once rebuilds start (an effective 6 to 7 conversations fits best), because a rebuild in progress occupies cache. Prefer morphing between measured replays on this side.

## Caveats the demo must respect

- One 30-minute run per step. Call the results preliminary.
- `rebuilt_tokens_per_question` in `summary.cache` exceeds the context length at 144 and 224 users GPU-only (the server counter counts repeated work). Do not show it. Use the per-request `source` shares.
- The white paper quotes token shares (35% rebuilt at 72 users). The table above quotes shares of questions (27%). Pick one and label it.
- Residency is not recorded. Which conversations sit on which GPU at a given moment is inferred from the `source` labels and the replica assignment.
- The three sessions that answer wrongly do so in both arms. It is the model, not the cache.
- Power is whole-server wall power from the PSUs. There is no separate memory or CPU power reading.

## House rules

- Use the `dell-assets` skill and `<DELL>/Templates/writing-style-instructions.md`. Build from a script in `source/`, never hand-edit HTML. No run ids or test-tool names in copy. Write "Intel Xeon" with the marks the style guide requires.
- Benefit must be credible: real situations, best-case baseline, limits shown.
- Suggested output: `<DELL>/2026 Q4 Deliverables/Xeon KV Cache Offload/xeon-kv-cache-offload-replay-demo.html`, built by a script beside `wp2.py`. Embed the step files or a trimmed copy so the page works offline.

## Done looks like

- Opens as one self-contained page, works at 1440 and 375 wide, no console errors.
- Users slider snaps to 56, 64, 72, 80, 96, 144, 224 and replays both arms in sync, with a speed control.
- Hero numbers at each measured step match the table above.
- Estimated values are visibly marked and footnoted.
- Adding the later steps (336, 480, 640, surge) needs only a rerun of the export and the build.
