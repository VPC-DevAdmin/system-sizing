# P11 results: US model survey on the XE7740 (2026-10-26, corrected 2026-10-28)

Scope and method: [p11-us-stack-scoping.md](p11-us-stack-scoping.md). Driver
[`scripts/p11_us_survey.py`](../scripts/p11_us_survey.py), summary
[`scripts/p11_summary.py`](../scripts/p11_summary.py), full data
[`p11-results.json`](p11-results.json) (corrected run) and
[`p11-results-2026-10-26-prefixcached.json`](p11-results-2026-10-26-prefixcached.json) (first
run, kept for the before/after table). Raw AIPerf and engine artifacts stay on the box under
`/data/capsim/p11/results-nocache` (corrected) and `/data/capsim/p11/results` (first run).

No latency target is applied anywhere in this document. Every rung is recorded; a
rung only counts as failed on a real error. Apply targets to the tables below.

## Correction 2026-10-28: the first run's prefill was served from cache

The 2026-10-26 numbers were too high for every model that supports prefix caching. AIPerf
generates a fixed set of 100 synthetic prompts and recycles them, so after the first few
seconds of a rung the engine found almost every 1,024-token prompt in its prefix cache and
skipped the prefill. vLLM logged the hit rate: a median of 97–99.7% for gpt-oss-20b,
gpt-oss-120b, Gemma-4-26B, Beam and Beam+DFlash2, 88% for Gemma-4-31B, and 0% for the two
Nemotron models (hybrid Mamba stacks, which cannot prefix-cache). SGLang's radix cache does
not log a rate, but behaves the same. The first run therefore measured decode-only
throughput for most models and full prefill-plus-decode for Nemotron, and the published
comparisons were skewed against Nemotron and in favour of whichever engine cached better.

Everything was rerun on 2026-10-27/28 with the cache off (`--no-enable-prefix-caching` on
vLLM, `--disable-radix-cache` on SGLang; same images, model revisions, load and ladder).
The two Nemotron vLLM configs reproduce within 1–7%, which is the control: the correction
only moved models that had been caching. Quality is unchanged (one answer each way on
Gemma-4-26B and Nemotron-3-Super at the long length, run-to-run noise).

| Config | First run (prefix-cached) | Corrected (no prefix cache) | Corrected ÷ first |
|---|---|---|---|
| gpt-oss-20b, SGLang | 85.2k (4,096) | 59.3k (4,096) | 0.70 |
| gpt-oss-20b, vLLM | 99.7k (2,048) | 47.3k (4,096) | 0.47 |
| Gemma-4-26B-A4B NVFP4, vLLM | 96.5k (4,096) | 42.1k (4,096) | 0.44 |
| Gemma-4-26B-A4B NVFP4, SGLang | 73.1k (4,096) | 33.9k (4,096) | 0.46 |
| gpt-oss-120b, vLLM | 73.8k (4,096) | 25.4k (2,048) | 0.34 |
| Gemma-4-31B NVFP4 (dense), vLLM | 29.0k (2,048) | 8.3k (2,048) | 0.29 |
| Nemotron-3-Super NVFP4, vLLM | 7.7k (2,048) | 7.7k (2,048) | 1.00 |
| Beam 1.0 NVFP4, vLLM | 6.6k (4,096) | 2.0k (4,096) | 0.31 |
| Beam 1.0 NVFP4 + DFlash2 drafter, vLLM | 3.0k (4,096) | 1.5k (1,024) | 0.50 |
| Nemotron-3-Ultra NVFP4, vLLM | 1.1k (512) | 1.1k (512) | 1.07 |
| Gemma-4-31B NVFP4 (dense), SGLang | 21.7k (2,048) | rerun pending † | |
| gpt-oss-120b, SGLang | 9.6k (4,096) | rerun pending † | |
| Nemotron-3-Super NVFP4, SGLang | 5.1k (4,096) | rerun pending † | |
| Nemotron-3-Ultra NVFP4, SGLang | 1.2k (4,096) | rerun pending † | |

† SGLang v0.5.20 refuses to start these four configs without its radix cache (it then sizes
the KV pools from the request cap and runs out of memory). They are being rerun with the
cache on and defeated by unique prompts instead; until then the tables below carry their
first-run figures, marked †, and the per-rung tables keep them separate.

What the correction changes in the conclusions:

- **Throughput halves or worse for the cached models.** The small MoE models drop from
  roughly 100k to 42–59k gen tok/s, gpt-oss-120b on vLLM from 74k to 25k, Gemma-4-31B on
  vLLM from 29k to 8.3k, Beam from 6.6k to 2.0k. With 1,024 input and 256 output tokens,
  80% of the tokens are prefill, and the engines now compute them.
- **Nemotron was never behind by as much as it looked.** Its numbers stand; the peak gap
  between gpt-oss-120b and Nemotron-3-Super, both on vLLM, shrinks from 9.6× to 3.3×.
- **The engine ranking flips on gpt-oss-20b.** SGLang now leads vLLM there (59k vs 47k at
  4,096 users); vLLM still leads on Gemma-4-26B (42k vs 34k). The other engine pairs await
  the pending SGLang reruns.
- **First-token latency at moderate load is seconds, not tens of milliseconds.** At 1,024
  users, p50 TTFT is 0.5–1.7 s for the small models and 3 s for Gemma-4-31B, where the first
  run showed 60–260 ms. Per-token latency at 4,096 users is 1.3–5× higher.

## Setup

- Dell XE7740, 8 × RTX PRO 6000 Blackwell Server 96 GB (SM120), PCIe only, no NVLink.
- Engines: vLLM v0.29.0, SGLang v0.5.20, and Reflection's `vllm/vllm-openai:reflection-beam`
  image for Beam (Beam is not in a vLLM release yet). Both engines at a 0.90 GPU memory share,
  otherwise defaults, except where noted under "Engine settings".
- Placement: smallest tensor-parallel size that fits, copied to fill all 8 GPUs. AIPerf
  round-robins across copies, so throughput is per server, measured.
- Load: AIPerf 0.13.0, synthetic prompts, 1,024 input and 256 output tokens (`ignore_eos`),
  closed-loop at 8, 32, 128, 256, 512, 1,024, 2,048 and 4,096 concurrent users. 30 s warmup,
  120 s window. Four AIPerf processes per rung, records merged. Prefix caching is off in
  both engines, so every request computes its 1,024-token prefill (see the correction).
- Quality: 32 synthetic logistics documents, exact-match JSON extraction of 4 records, at
  about 6.5K–7.8K and 26K–31K prompt tokens (token counts differ by tokenizer), temperature 0,
  reasoning effort low where the model supports it.

## Peak throughput and quality, by model and engine

| Model | Size (total / active) | Engine | GPUs per copy × copies | Peak gen tok/s (at users) | Max requests the engine ran at once | Quality, short / long |
|---|---|---|---|---|---|---|
| gpt-oss-20b | 21B / 4B | SGLang | 1 × 8 | 59.3k (4,096) | 3,920 | 32/32, 32/32 |
| gpt-oss-20b | 21B / 4B | vLLM | 1 × 8 | 47.3k (4,096) | 3,973 | 32/32, 32/32 |
| Gemma-4-26B-A4B NVFP4 | 26B / 4B | vLLM | 1 × 8 | 42.1k (4,096) | 3,993 | 32/32, 31/32 |
| Gemma-4-26B-A4B NVFP4 | 26B / 4B | SGLang | 1 × 8 | 33.9k (4,096) | 4,092 | 32/32, 32/32 |
| gpt-oss-120b | 117B / 5B | vLLM | 1 × 8 | 25.4k (2,048) | 2,409 | 32/32, 32/32 |
| Gemma-4-31B NVFP4 (dense) | 31B | vLLM | 1 × 8 | 8.3k (2,048) | 912 | 31/32, 31/32 |
| Nemotron-3-Super NVFP4 | 120B / 12B | vLLM | 2 × 4 | 7.7k (2,048) | 1,848 | 32/32, 32/32 |
| Beam 1.0 NVFP4 | 501B / 23B | vLLM | 8 × 1 | 2.0k (4,096) | 1,017 | 32/32, 32/32 |
| Beam 1.0 NVFP4 + DFlash2 drafter | 501B / 23B | vLLM | 8 × 1 | 1.5k (1,024) | 290 | 32/32, 32/32 |
| Nemotron-3-Ultra NVFP4 | 550B / 55B | vLLM | 8 × 1 | 1.1k (512) | 629 | 32/32, 32/32 |
| Gemma-4-31B NVFP4 (dense) † | 31B | SGLang | 1 × 8 | 21.7k (2,048) | 3,338 | 31/32, 30/32 |
| gpt-oss-120b † | 117B / 5B | SGLang | 1 × 8 | 9.6k (4,096) | 235 | 32/32, 32/32 |
| Nemotron-3-Super NVFP4 † | 120B / 12B | SGLang | 2 × 4 | 5.1k (4,096) | 216 | 32/32, 32/32 |
| Nemotron-3-Ultra NVFP4 † | 550B / 55B | SGLang | 8 × 1 | 1.2k (4,096) | 84 | 32/32, 32/32 |

† First-run figure with the prefix cache on; corrected rerun pending (see the correction).

Peak throughput is the highest rung, regardless of latency. The per-rung tables at the
end give latency at every load so a target can be applied.

## What the data shows

- **Every model runs on this box.** All seven US models load and serve on SM120 under both
  engines (Beam on vLLM only), with no errors above 0.2% at any rung.
- **Small MoE models carry the throughput.** gpt-oss-20b reaches 47k (vLLM) and 59k (SGLang)
  gen tok/s per server, Gemma-4-26B 42k and 34k, gpt-oss-120b 25k on vLLM. Each fits one GPU,
  so the box runs eight independent copies. Throughput still climbs with load on gpt-oss-20b
  and Gemma-4-26B at 4,096 users; gpt-oss-120b on vLLM flattens from 2,048 up.
- **Dense Gemma-4-31B is prefill-bound.** On vLLM it peaks at 8.3k from 1,024 users up and
  holds about 910 requests in flight whatever is offered; everything above queues (p50 TTFT
  32 s at 2,048 users).
- **Quality is flat across models on this test.** 31–32 of 32 everywhere. Gemma-4-31B's
  misses are real: it returned a fifth, wrongly selected record. This test checks
  long-document extraction only; it does not rank general model quality.
- **The giants fit, at low concurrency.** Beam (501B) and Nemotron-3-Ultra (550B) each run as
  one copy across all 8 GPUs over PCIe. Beam peaks at about 2.0k gen tok/s and Nemotron-3-Ultra
  at about 1.1k; the two are now within 2× of each other where the first run had 6×. Per-token
  latency at 8 users is 16 ms for Beam and 24 ms for Ultra. Beam's prefill over PCIe shows in
  first-token time: 2.7 s at 32 users and 4.7 s at 128, against 0.6–1.2 s for Nemotron-3-Super.
- **Engine choice depends on the model and the load, so both curves are kept.**
  - SGLang has the higher peak on gpt-oss-20b (59k vs 47k) and lower per-token latency at
    4,096 users (59 ms vs 78 ms); vLLM has the higher peak on Gemma-4-26B (42k vs 34k).
    Below 512 users the two engines are within 15% on both models.
  - The first run's SGLang concurrency ceilings (about 232 requests for gpt-oss-120b, 216 for
    Nemotron-3-Super, 84 for Nemotron-3-Ultra) and the Nemotron-3-Ultra engine comparison
    come from the † configs and are re-examined when their reruns land.
- **Beam's DFlash2 drafter helps only at low load here.** At 32 users it gives +21%
  throughput (919 vs 761) and a 5× lower first-token time (0.55 s vs 2.7 s), with an
  acceptance of about 3.2–3.3 tokens per verification step (vendor figure: 3.7). From 128
  users up it is slower, and the drafter's memory leaves room for about 290 concurrent
  requests instead of 1,017. On SM120 it only ran on FLASHINFER with adaptive verification
  off (see below).

## Engine settings and fixes made during the run

| Item | Setting or fix | Why |
|---|---|---|
| SGLang NVFP4 MoE models (Gemma-4-26B, Nemotron-3) | `--moe-runner-backend flashinfer_cutlass` | SGLang's default MoE runner refuses NVFP4 experts on SM120 and names this one |
| SGLang replicas | spaced ports, explicit `--nccl-port` | 8 replicas on the host network collided on a random rendezvous port |
| Max concurrent requests per copy | 4,096 ÷ copies, capped at 1,024 | Beam ran out of memory at startup with a 4,096 cap (warm-up activations) |
| Beam DFlash2 drafter | `attention_backend` FLASHINFER, adaptive verification off | The recipe's FLASH_ATTN refuses an FP8 KV cache on SM120; the drafter accepts only FLASH_ATTN or FLASHINFER; FLASHINFER cannot do adaptive verification |
| Quality grading | take the final answer after a `<think>` block or gpt-oss's `final` channel | Engines without a reasoning parser return the reasoning in `content` (Nemotron on vLLM, gpt-oss on SGLang); all saved responses were regraded |
| Prefix caching | vLLM `--no-enable-prefix-caching`, SGLang `--disable-radix-cache`. Four SGLang configs (gpt-oss-120b, Gemma-4-31B, Nemotron-3-Super, Nemotron-3-Ultra) keep the radix cache on and defeat it instead: 20,000 distinct AIPerf prompts per shard, a new seed per rung, `/flush_cache` before each rung | AIPerf recycles a 100-prompt synthetic set, so the cache served the prefill in the first run (vLLM logged 89–99.7% hit rates). SGLang v0.5.20 refuses to start those four without the radix cache: it then sizes the SWA and Mamba pools from `--max-running-requests` and runs out of memory |
| Slow rungs | 600 s window and 600 s grace, marked `window_s` in the data | Beam at 4,096 and Nemotron-3-Ultra from 512 (SGLang) or 1,024 (vLLM) up: answers took minutes, so a 120 s window recorded none |

A first pass used a single AIPerf process per rung. It topped out near 400 requests/s, so
the engines were under-loaded at high concurrency (vLLM ran ~1,340 of 2,048 offered on
gpt-oss-20b). Those results were discarded; everything here uses four AIPerf processes, and
`engine_running_p50` in each rung shows how many requests the engine actually held.

## Caveats

- Synthetic prompts with fixed 1,024/256 lengths and `ignore_eos`. For reasoning models
  (gpt-oss, Beam, Nemotron) the 256 output tokens include reasoning, and "time to first
  token" is the first reasoning token, not the first answer token.
- The ladder doubles each step, so limits fall somewhere between two rungs.
- Engines run at defaults plus the settings above; no per-model tuning. The 1,024 cap on
  Beam is a driver choice; Beam's throughput at 2,048–4,096 users is bounded by it.
- Where answers take longer than the window, AIPerf counts only finished requests. The
  "Engine tok/s" column (the engine's own generation counter) is the cross-check; the two
  agree within about 3% except at a few overloaded rungs (e.g. Nemotron-3-Super on vLLM at
  4,096: 5.6k vs 7.6k).
- Nemotron-3-Ultra on SGLang at 128 and 256 users ran with the 120 s window while p95 first
  token was 37–73 s; its engine counter reads about 10% below AIPerf there.
- With prefill computed, the engines hold fewer requests than offered at the top rungs on
  some models (Gemma-4-31B on vLLM: 912 of 2,048; gpt-oss-120b on vLLM: 2,409 of 4,096), and
  first-token time there is the queue, not the model: 32 s and 12 s at p50.
- No power measurement (no wall-power instrumentation on this box).

## Provenance

| Image | Version label | Image ID |
|---|---|---|
| `vllm/vllm-openai:latest` | v0.29.0 | `sha256:c2914767605584b6d8f45686b82de173ecc99e781897aa3d0a66dacd72c51ae1` |
| `lmsysorg/sglang:latest` | v0.5.20 | `sha256:06e4f2ed21afde4ff513cda65070124e727ba23ccaeff7712b8c40e1097d611f` |
| `vllm/vllm-openai:reflection-beam` | (dev build) | `sha256:c1a6be6ad08016f8cebcb449146b3d992791fe1868f9a26f3e48a51040517b77` |

| Model | Hugging Face revision |
|---|---|
| openai/gpt-oss-20b | `6cee5e81ee83917806bbde320786a8fb61efebee` |
| openai/gpt-oss-120b | `b5c939de8f754692c1647ca79fbf85e8c1e70f8a` |
| nvidia/Gemma-4-26B-A4B-NVFP4 | `a19cfe00be84568a6867111c9a68c9c44fdcffe6` |
| nvidia/Gemma-4-31B-IT-NVFP4 | `4135a98a9b728a548947683219633b25682223ac` |
| nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4 | `ff433f5493e25d631c9f12b5d55c674229923d02` |
| nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-NVFP4 | `02462641f13d3af838b904f48195b9bb8a1e4ebc` |
| reflection/Beam-1.0-501B-A23B-NVFP4 | `f8ecac0e16045babf60156f6ba996a77fcb48cf9` |

## Per-rung results

TTFT is time to first token; ITL is inter-token latency. "Running" is the median number of
requests the engine held during the window; "Window" is the measurement window (600 s where
answers took minutes).

### openai/gpt-oss-20b on sglang (tp1 x 8)

| Users | Gen tok/s | Engine tok/s | TTFT p50 / p95 | ITL p50 / p95 | Running (p50) | Window | Errors |
|---|---|---|---|---|---|---|---|
| 8 | 1.7k | 1.7k | 35 ms / 40 ms | 4 ms / 5 ms | 8 | 120 s | 0.00% |
| 32 | 4.8k | 4.7k | 43 ms / 62 ms | 7 ms / 7 ms | 32 | 120 s | 0.00% |
| 128 | 13.2k | 13.0k | 69 ms / 268 ms | 9 ms / 10 ms | 128 | 120 s | 0.05% |
| 256 | 22.0k | 21.7k | 224 ms / 524 ms | 11 ms / 12 ms | 256 | 120 s | 0.00% |
| 512 | 34.0k | 33.5k | 493 ms / 965 ms | 13 ms / 15 ms | 512 | 120 s | 0.00% |
| 1024 | 46.2k | 45.8k | 939 ms / 2.0 s | 18 ms / 21 ms | 1024 | 120 s | 0.00% |
| 2048 | 51.9k | 51.2k | 800 ms / 3.1 s | 32 ms / 55 ms | 1962 | 120 s | 0.07% |
| 4096 | 59.3k | 58.6k | 2.5 s / 5.8 s | 59 ms / 67 ms | 3920 | 120 s | 0.00% |

### openai/gpt-oss-20b on vllm (tp1 x 8)

| Users | Gen tok/s | Engine tok/s | TTFT p50 / p95 | ITL p50 / p95 | Running (p50) | Window | Errors |
|---|---|---|---|---|---|---|---|
| 8 | 1.8k | 1.9k | 51 ms / 53 ms | 4 ms / 4 ms | 8 | 120 s | 0.00% |
| 32 | 4.8k | 4.9k | 65 ms / 92 ms | 6 ms / 6 ms | 32 | 120 s | 0.00% |
| 128 | 12.7k | 12.9k | 79 ms / 379 ms | 9 ms / 10 ms | 128 | 120 s | 0.02% |
| 256 | 20.2k | 20.5k | 134 ms / 711 ms | 12 ms / 12 ms | 255 | 120 s | 0.00% |
| 512 | 29.0k | 29.5k | 238 ms / 1.1 s | 16 ms / 17 ms | 505 | 120 s | 0.01% |
| 1024 | 37.2k | 38.0k | 583 ms / 1.6 s | 24 ms / 26 ms | 1012 | 120 s | 0.03% |
| 2048 | 43.9k | 44.8k | 1.1 s / 4.0 s | 41 ms / 44 ms | 2000 | 120 s | 0.03% |
| 4096 | 47.3k | 48.3k | 1.3 s / 10.8 s | 78 ms / 80 ms | 3973 | 120 s | 0.15% |

### nvidia/Gemma-4-26B-A4B-NVFP4 on vllm (tp1 x 8)

| Users | Gen tok/s | Engine tok/s | TTFT p50 / p95 | ITL p50 / p95 | Running (p50) | Window | Errors |
|---|---|---|---|---|---|---|---|
| 8 | 1.2k | 1.2k | 39 ms / 43 ms | 6 ms / 7 ms | 8 | 120 s | 0.00% |
| 32 | 3.6k | 3.6k | 68 ms / 101 ms | 9 ms / 9 ms | 32 | 120 s | 0.00% |
| 128 | 10.6k | 10.4k | 138 ms / 282 ms | 12 ms / 12 ms | 128 | 120 s | 0.00% |
| 256 | 16.7k | 16.4k | 247 ms / 421 ms | 14 ms / 15 ms | 256 | 120 s | 0.00% |
| 512 | 25.0k | 24.6k | 359 ms / 561 ms | 19 ms / 20 ms | 512 | 120 s | 0.02% |
| 1024 | 32.4k | 31.9k | 451 ms / 706 ms | 30 ms / 30 ms | 1024 | 120 s | 0.00% |
| 2048 | 38.9k | 38.3k | 484 ms / 2.8 s | 51 ms / 51 ms | 2048 | 120 s | 0.00% |
| 4096 | 42.1k | 41.5k | 621 ms / 8.1 s | 94 ms / 95 ms | 3993 | 120 s | 0.20% |

### nvidia/Gemma-4-26B-A4B-NVFP4 on sglang (tp1 x 8)

| Users | Gen tok/s | Engine tok/s | TTFT p50 / p95 | ITL p50 / p95 | Running (p50) | Window | Errors |
|---|---|---|---|---|---|---|---|
| 8 | 1.1k | 1.1k | 53 ms / 63 ms | 7 ms / 8 ms | 8 | 120 s | 0.00% |
| 32 | 3.2k | 3.2k | 60 ms / 98 ms | 10 ms / 10 ms | 32 | 120 s | 0.00% |
| 128 | 9.2k | 9.1k | 105 ms / 404 ms | 14 ms / 14 ms | 128 | 120 s | 0.00% |
| 256 | 14.5k | 14.2k | 186 ms / 728 ms | 17 ms / 18 ms | 256 | 120 s | 0.01% |
| 512 | 20.8k | 20.5k | 402 ms / 1.6 s | 23 ms / 26 ms | 512 | 120 s | 0.05% |
| 1024 | 26.9k | 26.5k | 1.4 s / 3.2 s | 32 ms / 39 ms | 1024 | 120 s | 0.00% |
| 2048 | 31.3k | 30.9k | 2.7 s / 6.2 s | 54 ms / 66 ms | 2044 | 120 s | 0.00% |
| 4096 | 33.9k | 34.5k | 6.4 s / 12.3 s | 96 ms / 116 ms | 4092 | 120 s | 0.00% |

### openai/gpt-oss-120b on vllm (tp1 x 8)

| Users | Gen tok/s | Engine tok/s | TTFT p50 / p95 | ITL p50 / p95 | Running (p50) | Window | Errors |
|---|---|---|---|---|---|---|---|
| 8 | 1.4k | 1.3k | 86 ms / 87 ms | 6 ms / 6 ms | 8 | 120 s | 0.00% |
| 32 | 3.1k | 3.1k | 136 ms / 244 ms | 10 ms / 10 ms | 32 | 120 s | 0.00% |
| 128 | 6.9k | 6.8k | 316 ms / 725 ms | 17 ms / 18 ms | 128 | 120 s | 0.00% |
| 256 | 10.2k | 10.0k | 621 ms / 1.4 s | 23 ms / 25 ms | 256 | 120 s | 0.00% |
| 512 | 14.7k | 14.5k | 1.2 s / 2.0 s | 30 ms / 34 ms | 512 | 120 s | 0.00% |
| 1024 | 19.7k | 19.4k | 1.7 s / 4.0 s | 44 ms / 49 ms | 1022 | 120 s | 0.00% |
| 2048 | 25.4k | 25.1k | 1.9 s / 9.1 s | 72 ms / 76 ms | 1966 | 120 s | 0.00% |
| 4096 | 25.2k | 24.9k | 12.4 s / 32.0 s | 93 ms / 108 ms | 2409 | 120 s | 0.03% |

### nvidia/Gemma-4-31B-IT-NVFP4 on vllm (tp1 x 8)

| Users | Gen tok/s | Engine tok/s | TTFT p50 / p95 | ITL p50 / p95 | Running (p50) | Window | Errors |
|---|---|---|---|---|---|---|---|
| 8 | 315 | 310 | 121 ms / 124 ms | 25 ms / 25 ms | 8 | 120 s | 0.00% |
| 32 | 1.2k | 1.1k | 315 ms / 418 ms | 27 ms / 27 ms | 32 | 120 s | 0.00% |
| 128 | 3.5k | 3.4k | 915 ms / 1.5 s | 33 ms / 35 ms | 128 | 120 s | 0.00% |
| 256 | 5.4k | 5.3k | 1.7 s / 2.5 s | 41 ms / 46 ms | 256 | 120 s | 0.00% |
| 512 | 7.4k | 7.2k | 2.1 s / 4.8 s | 62 ms / 67 ms | 512 | 120 s | 0.00% |
| 1024 | 8.2k | 8.1k | 3.1 s / 18.9 s | 101 ms / 104 ms | 912 | 120 s | 0.00% |
| 2048 | 8.3k | 8.2k | 32.0 s / 48.4 s | 101 ms / 104 ms | 912 | 120 s | 0.00% |
| 4096 | 7.8k | 8.0k | 66.7 s / 112.5 s | 101 ms / 104 ms | 912 | 120 s | 0.00% |

### nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4 on vllm (tp2 x 4)

| Users | Gen tok/s | Engine tok/s | TTFT p50 / p95 | ITL p50 / p95 | Running (p50) | Window | Errors |
|---|---|---|---|---|---|---|---|
| 8 | 668 | 657 | 298 ms / 312 ms | 11 ms / 12 ms | 8 | 120 s | 0.00% |
| 32 | 1.8k | 1.7k | 621 ms / 647 ms | 16 ms / 18 ms | 32 | 120 s | 0.00% |
| 128 | 3.7k | 3.7k | 1.2 s / 1.6 s | 30 ms / 32 ms | 128 | 120 s | 0.00% |
| 256 | 4.9k | 4.8k | 1.4 s / 2.9 s | 46 ms / 49 ms | 256 | 120 s | 0.00% |
| 512 | 6.3k | 6.2k | 1.6 s / 6.7 s | 75 ms / 78 ms | 512 | 120 s | 0.00% |
| 1024 | 7.3k | 7.2k | 1.7 s / 15.1 s | 130 ms / 134 ms | 964 | 120 s | 0.00% |
| 2048 | 7.7k | 7.6k | 5.9 s / 34.6 s | 222 ms / 232 ms | 1817 | 120 s | 0.00% |
| 4096 | 5.6k | 7.6k | 59.7 s / 97.2 s | 228 ms / 235 ms | 1848 | 120 s | 0.00% |

### reflection/Beam-1.0-501B-A23B-NVFP4 on vllm (tp8 x 1)

| Users | Gen tok/s | Engine tok/s | TTFT p50 / p95 | ITL p50 / p95 | Running (p50) | Window | Errors |
|---|---|---|---|---|---|---|---|
| 8 | 403 | 397 | 917 ms / 931 ms | 16 ms / 16 ms | 8 | 120 s | 0.00% |
| 32 | 761 | 738 | 2.7 s / 3.2 s | 31 ms / 36 ms | 32 | 120 s | 0.00% |
| 128 | 1.4k | 1.2k | 4.7 s / 10.9 s | 78 ms / 85 ms | 113 | 120 s | 0.00% |
| 256 | 1.6k | 1.5k | 5.6 s / 25.9 s | 141 ms / 150 ms | 241 | 120 s | 0.00% |
| 512 | 1.7k | 1.7k | 7.5 s / 53.3 s | 269 ms / 270 ms | 470 | 120 s | 0.00% |
| 1024 | 2.0k | 1.9k | 44.7 s / 92.6 s | 325 ms / 492 ms | 660 | 120 s | 0.00% |
| 2048 | 1.9k | 1.7k | 144.5 s / 184.4 s | 532 ms / 534 ms | 1017 | 600 s | 0.00% |
| 4096 | 2.0k | 1.8k | 425.4 s / 484.2 s | 534 ms / 535 ms | 1017 | 600 s | 0.00% |

### reflection/Beam-1.0-501B-A23B-NVFP4 + DFlash2 drafter on vllm (tp8 x 1)

| Users | Gen tok/s | Engine tok/s | TTFT p50 / p95 | ITL p50 / p95 | Running (p50) | Window | Errors |
|---|---|---|---|---|---|---|---|
| 8 | 416 | 410 | 302 ms / 402 ms | 19 ms / 24 ms | 8 | 120 s | 0.00% |
| 32 | 919 | 902 | 552 ms / 1.4 s | 33 ms / 44 ms | 31 | 120 s | 0.00% |
| 128 | 1.3k | 1.3k | 1.4 s / 6.2 s | 94 ms / 127 ms | 124 | 120 s | 0.00% |
| 256 | 1.4k | 1.4k | 3.2 s / 17.6 s | 162 ms / 233 ms | 247 | 120 s | 0.00% |
| 512 | 1.3k | 1.4k | 42.2 s / 71.7 s | 196 ms / 276 ms | 287 | 120 s | 0.00% |
| 1024 | 1.5k | 1.3k | 54.0 s / 129.6 s | 215 ms / 280 ms | 290 | 120 s | 0.00% |
| 2048 | 1.4k | 1.4k | 331.9 s / 334.7 s | 215 ms / 279 ms | 290 | 600 s | 0.00% |
| 4096 | 1.4k | 1.4k | 527.5 s / 720.8 s | 218 ms / 282 ms | 290 | 600 s | 0.00% |

### nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-NVFP4 on vllm (tp8 x 1)

| Users | Gen tok/s | Engine tok/s | TTFT p50 / p95 | ITL p50 / p95 | Running (p50) | Window | Errors |
|---|---|---|---|---|---|---|---|
| 8 | 264 | 260 | 1.6 s / 1.7 s | 24 ms / 29 ms | 8 | 120 s | 0.00% |
| 32 | 449 | 443 | 2.8 s / 4.6 s | 61 ms / 66 ms | 32 | 120 s | 0.00% |
| 128 | 763 | 743 | 4.7 s / 14.1 s | 155 ms / 167 ms | 115 | 120 s | 0.00% |
| 256 | 913 | 856 | 5.5 s / 35.3 s | 276 ms / 287 ms | 240 | 120 s | 0.00% |
| 512 | 1.1k | 894 | 23.2 s / 53.3 s | 345 ms / 526 ms | 341 | 120 s | 0.00% |
| 1024 | 437 | 820 | 46.7 s / 103.7 s | 495 ms / 660 ms | 431 | 120 s | 0.00% |
| 2048 | 1.1k | 924 | 384.8 s / 420.2 s | 676 ms / 677 ms | 629 | 600 s | 0.00% |
| 4096 | 949 | 951 | 523.9 s / 901.5 s | 676 ms / 677 ms | 629 | 600 s | 0.00% |

### nvidia/Gemma-4-31B-IT-NVFP4 on sglang (tp1 x 8) — first run, prefix-cached; corrected rerun pending †

| Users | Gen tok/s | Engine tok/s | TTFT p50 / p95 | ITL p50 / p95 | Running (p50) | Window | Errors |
|---|---|---|---|---|---|---|---|
| 8 | 304 | 301 | 177 ms / 190 ms | 26 ms / 26 ms | 8 | 120 s | 0.00% |
| 32 | 1.1k | 1.1k | 202 ms / 397 ms | 28 ms / 28 ms | 32 | 120 s | 0.00% |
| 128 | 3.8k | 3.7k | 182 ms / 1.6 s | 32 ms / 34 ms | 128 | 120 s | 0.00% |
| 256 | 6.1k | 6.1k | 208 ms / 3.3 s | 38 ms / 44 ms | 256 | 120 s | 0.00% |
| 512 | 10.1k | 10.0k | 245 ms / 5.6 s | 46 ms / 53 ms | 512 | 120 s | 0.00% |
| 1024 | 17.3k | 17.2k | 287 ms / 7.7 s | 54 ms / 59 ms | 1024 | 120 s | 0.00% |
| 2048 | 21.7k | 21.5k | 299 ms / 10.0 s | 80 ms / 102 ms | 1989 | 120 s | 0.04% |
| 4096 | 19.0k | 18.8k | 10.1 s / 20.7 s | 153 ms / 236 ms | 3338 | 120 s | 0.00% |

### openai/gpt-oss-120b on sglang (tp1 x 8) — first run, prefix-cached; corrected rerun pending †

| Users | Gen tok/s | Engine tok/s | TTFT p50 / p95 | ITL p50 / p95 | Running (p50) | Window | Errors |
|---|---|---|---|---|---|---|---|
| 8 | 1.2k | 1.2k | 70 ms / 75 ms | 6 ms / 6 ms | 8 | 120 s | 0.00% |
| 32 | 3.0k | 2.9k | 79 ms / 151 ms | 10 ms / 11 ms | 32 | 120 s | 0.00% |
| 128 | 6.6k | 6.5k | 126 ms / 415 ms | 19 ms / 21 ms | 128 | 120 s | 0.03% |
| 256 | 9.2k | 9.1k | 550 ms / 4.6 s | 24 ms / 25 ms | 232 | 120 s | 0.00% |
| 512 | 9.1k | 9.0k | 7.0 s / 12.8 s | 23 ms / 25 ms | 232 | 120 s | 0.00% |
| 1024 | 9.2k | 9.1k | 19.8 s / 25.9 s | 23 ms / 25 ms | 232 | 120 s | 0.00% |
| 2048 | 9.5k | 9.4k | 45.7 s / 51.2 s | 23 ms / 25 ms | 232 | 120 s | 0.05% |
| 4096 | 9.6k | 9.6k | 81.6 s / 100.6 s | 23 ms / 25 ms | 235 | 120 s | 0.00% |

### nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4 on sglang (tp2 x 4) — first run, prefix-cached; corrected rerun pending †

| Users | Gen tok/s | Engine tok/s | TTFT p50 / p95 | ITL p50 / p95 | Running (p50) | Window | Errors |
|---|---|---|---|---|---|---|---|
| 8 | 756 | 744 | 299 ms / 315 ms | 9 ms / 9 ms | 8 | 120 s | 0.00% |
| 32 | 2.1k | 2.1k | 295 ms / 314 ms | 14 ms / 14 ms | 32 | 120 s | 0.00% |
| 128 | 4.6k | 4.6k | 659 ms / 936 ms | 25 ms / 25 ms | 128 | 120 s | 0.00% |
| 256 | 4.7k | 4.7k | 3.0 s / 12.5 s | 35 ms / 42 ms | 216 | 120 s | 0.00% |
| 512 | 4.8k | 4.7k | 14.7 s / 24.7 s | 35 ms / 42 ms | 216 | 120 s | 0.00% |
| 1024 | 4.9k | 4.9k | 44.2 s / 48.6 s | 34 ms / 42 ms | 216 | 120 s | 0.00% |
| 2048 | 4.9k | 4.9k | 84.9 s / 102.0 s | 35 ms / 41 ms | 216 | 120 s | 0.00% |
| 4096 | 5.1k | 5.2k | 86.2 s / 161.7 s | 34 ms / 39 ms | 216 | 120 s | 0.00% |

### nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-NVFP4 on sglang (tp8 x 1) — first run, prefix-cached; corrected rerun pending †

| Users | Gen tok/s | Engine tok/s | TTFT p50 / p95 | ITL p50 / p95 | Running (p50) | Window | Errors |
|---|---|---|---|---|---|---|---|
| 8 | 281 | 277 | 1.5 s / 1.6 s | 23 ms / 26 ms | 8 | 120 s | 0.00% |
| 32 | 512 | 505 | 3.6 s / 6.0 s | 47 ms / 60 ms | 32 | 120 s | 0.00% |
| 128 | 799 | 726 | 15.0 s / 37.2 s | 76 ms / 105 ms | 76 | 120 s | 0.00% |
| 256 | 775 | 700 | 47.6 s / 72.9 s | 76 ms / 105 ms | 84 | 120 s | 0.00% |
| 512 | 775 | 784 | 145.5 s / 166.9 s | 72 ms / 99 ms | 84 | 600 s | 0.00% |
| 1024 | 1.0k | 925 | 257.8 s / 279.2 s | 60 ms / 94 ms | 84 | 600 s | 0.00% |
| 2048 | 957 | 908 | 536.6 s / 568.7 s | 59 ms / 97 ms | 84 | 600 s | 0.00% |
| 4096 | 1.2k | 1.1k | 683.3 s / 864.7 s | 54 ms / 75 ms | 84 | 600 s | 0.00% |
