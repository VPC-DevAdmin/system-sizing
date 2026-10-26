# P11 results: US model survey on the XE7740 (2026-10-26)

Scope and method: [p11-us-stack-scoping.md](p11-us-stack-scoping.md). Driver
[`scripts/p11_us_survey.py`](../scripts/p11_us_survey.py), summary
[`scripts/p11_summary.py`](../scripts/p11_summary.py), full data
[`p11-results.json`](p11-results.json). Raw AIPerf and engine artifacts stay on the
box under `/data/capsim/p11/results` (8.5 GB).

No latency target is applied anywhere in this document. Every rung is recorded; a
rung only counts as failed on a real error. Apply targets to the tables below.

## Setup

- Dell XE7740, 8 × RTX PRO 6000 Blackwell Server 96 GB (SM120), PCIe only, no NVLink.
- Engines: vLLM v0.29.0, SGLang v0.5.20, and Reflection's `vllm/vllm-openai:reflection-beam`
  image for Beam (Beam is not in a vLLM release yet). Both engines at a 0.90 GPU memory share,
  otherwise defaults, except where noted under "Engine settings".
- Placement: smallest tensor-parallel size that fits, copied to fill all 8 GPUs. AIPerf
  round-robins across copies, so throughput is per server, measured.
- Load: AIPerf 0.13.0, synthetic prompts, 1,024 input and 256 output tokens (`ignore_eos`),
  closed-loop at 8, 32, 128, 256, 512, 1,024, 2,048 and 4,096 concurrent users. 30 s warmup,
  120 s window. Four AIPerf processes per rung, records merged.
- Quality: 32 synthetic logistics documents, exact-match JSON extraction of 4 records, at
  about 6.5K–7.8K and 26K–31K prompt tokens (token counts differ by tokenizer), temperature 0,
  reasoning effort low where the model supports it.

## Peak throughput and quality, by model and engine

| Model | Size (total / active) | Engine | GPUs per copy × copies | Peak gen tok/s (at users) | Max requests the engine ran at once | Quality, short / long |
|---|---|---|---|---|---|---|
| gpt-oss-20b | 21B / 4B | vLLM | 1 × 8 | 99.7k (2,048) | ~1,610 | 32/32, 32/32 |
| gpt-oss-20b | | SGLang | 1 × 8 | 85.2k (4,096) | 4,096 | 32/32, 32/32 |
| Gemma-4-26B-A4B NVFP4 | 26B / 4B | vLLM | 1 × 8 | 96.5k (4,096) | ~3,920 | 32/32, 32/32 |
| Gemma-4-26B-A4B NVFP4 | | SGLang | 1 × 8 | 73.1k (4,096) | 4,096 | 32/32, 32/32 |
| gpt-oss-120b | 117B / 5B | vLLM | 1 × 8 | 73.8k (4,096) | 4,096 | 32/32, 32/32 |
| gpt-oss-120b | | SGLang | 1 × 8 | 9.6k (4,096) | ~232 | 32/32, 32/32 |
| Gemma-4-31B NVFP4 (dense) | 31B | vLLM | 1 × 8 | 29.0k (2,048) | 2,048 (collapses to ~930 at 4,096) | 31/32, 31/32 |
| Gemma-4-31B NVFP4 (dense) | | SGLang | 1 × 8 | 21.7k (2,048) | ~3,340 | 31/32, 30/32 |
| Nemotron-3-Super NVFP4 | 120B / 12B | vLLM | 2 × 4 | 7.7k (2,048) | ~1,850 | 32/32, 31/32 |
| Nemotron-3-Super NVFP4 | | SGLang | 2 × 4 | 5.1k (4,096) | 216 | 32/32, 32/32 |
| Beam 1.0 NVFP4 | 501B / 23B | vLLM | 8 × 1 | 6.6k (4,096) | 1,024 (cap set by the driver) | 32/32, 32/32 |
| Beam 1.0 NVFP4 + DFlash2 drafter | | vLLM | 8 × 1 | 3.0k (4,096) | ~532 | 32/32, 32/32 |
| Nemotron-3-Ultra NVFP4 | 550B / 55B | SGLang | 8 × 1 | 1.2k (4,096) | 84 | 32/32, 32/32 |
| Nemotron-3-Ultra NVFP4 | | vLLM | 8 × 1 | 1.1k (512) | ~629 | 32/32, 32/32 |

Peak throughput is the highest rung, regardless of latency. The per-rung tables at the
end give latency at every load so a target can be applied.

## What the data shows

- **Every model runs on this box.** All seven US models load and serve on SM120 under both
  engines (Beam on vLLM only), with no errors above 0.5% at any rung.
- **Small MoE models carry the throughput.** gpt-oss-20b and Gemma-4-26B reach roughly
  96–100k gen tok/s per server; gpt-oss-120b reaches 74k. Each fits one GPU, so the box runs
  eight independent copies.
- **Quality is flat across models on this test.** 30–32 of 32 everywhere. Gemma-4-31B's
  misses are real: it returned a fifth, wrongly selected record. This test checks
  long-document extraction only; it does not rank general model quality.
- **The giants fit, at low concurrency.** Beam (501B) and Nemotron-3-Ultra (550B) each run as
  one copy across all 8 GPUs over PCIe. Beam peaks at about 6.6k gen tok/s; Ultra at about
  1.1–1.2k. Per-token latency at 8 users is 16 ms for Beam and 23–24 ms for Ultra.
- **Engine choice depends on the model and the load, so both curves are kept.**
  - vLLM has the higher peak for gpt-oss, Gemma and Nemotron-3-Super.
  - SGLang has a lower concurrency ceiling on several models: about 232 requests for
    gpt-oss-120b, 216 for Nemotron-3-Super and 84 for Nemotron-3-Ultra. Past that, extra
    users queue and first-token latency grows.
  - SGLang is faster at low load on Nemotron-3-Super: up to 128 users it gives about 25%
    more throughput, with lower latency.
  - On Nemotron-3-Ultra the two engines trade per-token speed against users served at once.
    SGLang runs 84 requests at 54–105 ms per token; vLLM runs about 629 at 678 ms per token.
    Total throughput is about the same.
- **Beam's DFlash2 drafter helps only at low load here.** At 32 users it gives +34%
  throughput (1.24k vs 0.93k) and lower first-token latency, with an acceptance of about
  3.3 tokens per verification step (vendor figure: 3.7). From 128 users up it is slower,
  and the drafter's memory leaves room for about 532 concurrent requests instead of 1,024.
  On SM120 it only ran on FLASHINFER with adaptive verification off (see below).

## Engine settings and fixes made during the run

| Item | Setting or fix | Why |
|---|---|---|
| SGLang NVFP4 MoE models (Gemma-4-26B, Nemotron-3) | `--moe-runner-backend flashinfer_cutlass` | SGLang's default MoE runner refuses NVFP4 experts on SM120 and names this one |
| SGLang replicas | spaced ports, explicit `--nccl-port` | 8 replicas on the host network collided on a random rendezvous port |
| Max concurrent requests per copy | 4,096 ÷ copies, capped at 1,024 | Beam ran out of memory at startup with a 4,096 cap (warm-up activations) |
| Beam DFlash2 drafter | `attention_backend` FLASHINFER, adaptive verification off | The recipe's FLASH_ATTN refuses an FP8 KV cache on SM120; the drafter accepts only FLASH_ATTN or FLASHINFER; FLASHINFER cannot do adaptive verification |
| Quality grading | take the final answer after a `<think>` block or gpt-oss's `final` channel | Engines without a reasoning parser return the reasoning in `content` (Nemotron on vLLM, gpt-oss on SGLang); all saved responses were regraded |
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
requests the engine held during the window.

### openai/gpt-oss-20b on vllm (tp1 x 8)

| Users | Gen tok/s | Engine tok/s | TTFT p50 / p95 | ITL p50 / p95 | Running (p50) | Errors |
|---|---|---|---|---|---|---|
| 8 | 1.9k | 1.9k | 24 ms / 51 ms | 4 ms / 4 ms | 8.0 | 0.00% |
| 32 | 5.1k | 5.2k | 38 ms / 42 ms | 6 ms / 6 ms | 32.0 | 0.00% |
| 128 | 15.3k | 15.5k | 49 ms / 59 ms | 8 ms / 8 ms | 128.0 | 0.00% |
| 256 | 27.0k | 27.7k | 57 ms / 75 ms | 9 ms / 9 ms | 256.0 | 0.00% |
| 512 | 47.1k | 48.4k | 70 ms / 101 ms | 10 ms / 10 ms | 509.0 | 0.01% |
| 1024 | 77.7k | 79.6k | 92 ms / 323 ms | 12 ms / 12 ms | 1013.0 | 0.00% |
| 2048 | 99.7k | 102.2k | 889 ms / 2.9 s | 15 ms / 16 ms | 1570.0 | 0.13% |
| 4096 | 97.6k | 100.1k | 5.4 s / 9.4 s | 15 ms / 28 ms | 1613.5 | 0.12% |

### nvidia/Gemma-4-26B-A4B-NVFP4 on vllm (tp1 x 8)

| Users | Gen tok/s | Engine tok/s | TTFT p50 / p95 | ITL p50 / p95 | Running (p50) | Errors |
|---|---|---|---|---|---|---|
| 8 | 1.2k | 1.2k | 39 ms / 45 ms | 6 ms / 7 ms | 8.0 | 0.00% |
| 32 | 3.8k | 3.7k | 28 ms / 35 ms | 8 ms / 9 ms | 32.0 | 0.00% |
| 128 | 11.6k | 11.5k | 37 ms / 48 ms | 11 ms / 11 ms | 128.0 | 0.00% |
| 256 | 19.3k | 19.0k | 43 ms / 59 ms | 13 ms / 13 ms | 256.0 | 0.00% |
| 512 | 33.0k | 32.5k | 56 ms / 86 ms | 15 ms / 15 ms | 512.0 | 0.00% |
| 1024 | 54.7k | 53.9k | 74 ms / 165 ms | 18 ms / 19 ms | 1024.0 | 0.05% |
| 2048 | 79.8k | 78.6k | 113 ms / 311 ms | 25 ms / 26 ms | 2048.0 | 0.16% |
| 4096 | 96.5k | 95.1k | 382 ms / 1.9 s | 39 ms / 42 ms | 3915.5 | 0.03% |

### openai/gpt-oss-20b on sglang (tp1 x 8)

| Users | Gen tok/s | Engine tok/s | TTFT p50 / p95 | ITL p50 / p95 | Running (p50) | Errors |
|---|---|---|---|---|---|---|
| 8 | 1.7k | 1.7k | 36 ms / 41 ms | 4 ms / 5 ms | 8.0 | 0.00% |
| 32 | 4.8k | 4.7k | 43 ms / 61 ms | 7 ms / 7 ms | 32.0 | 0.00% |
| 128 | 13.9k | 13.7k | 45 ms / 71 ms | 9 ms / 10 ms | 128.0 | 0.02% |
| 256 | 23.3k | 23.0k | 48 ms / 104 ms | 11 ms / 12 ms | 256.0 | 0.07% |
| 512 | 42.1k | 41.4k | 78 ms / 204 ms | 11 ms / 14 ms | 512.0 | 0.00% |
| 1024 | 52.3k | 51.6k | 57 ms / 217 ms | 19 ms / 23 ms | 1019.0 | 0.03% |
| 2048 | 63.5k | 62.9k | 101 ms / 748 ms | 23 ms / 54 ms | 1985.5 | 0.07% |
| 4096 | 85.2k | 84.0k | 507 ms / 1.6 s | 46 ms / 48 ms | 4096.0 | 0.00% |

### openai/gpt-oss-120b on vllm (tp1 x 8)

| Users | Gen tok/s | Engine tok/s | TTFT p50 / p95 | ITL p50 / p95 | Running (p50) | Errors |
|---|---|---|---|---|---|---|
| 8 | 1.4k | 1.4k | 80 ms / 82 ms | 6 ms / 6 ms | 8.0 | 0.00% |
| 32 | 3.4k | 3.3k | 56 ms / 63 ms | 9 ms / 9 ms | 32.0 | 0.00% |
| 128 | 8.3k | 8.1k | 96 ms / 110 ms | 15 ms / 15 ms | 128.0 | 0.00% |
| 256 | 12.9k | 12.7k | 122 ms / 141 ms | 19 ms / 20 ms | 256.0 | 0.05% |
| 512 | 20.9k | 20.6k | 154 ms / 181 ms | 24 ms / 24 ms | 512.0 | 0.00% |
| 1024 | 34.0k | 33.4k | 196 ms / 286 ms | 29 ms / 30 ms | 1024.0 | 0.00% |
| 2048 | 54.1k | 53.3k | 265 ms / 501 ms | 37 ms / 38 ms | 2048.0 | 0.00% |
| 4096 | 73.8k | 72.8k | 566 ms / 2.8 s | 52 ms / 54 ms | 4096.0 | 0.00% |

### nvidia/Gemma-4-26B-A4B-NVFP4 on sglang (tp1 x 8)

| Users | Gen tok/s | Engine tok/s | TTFT p50 / p95 | ITL p50 / p95 | Running (p50) | Errors |
|---|---|---|---|---|---|---|
| 8 | 1.1k | 1.1k | 54 ms / 61 ms | 7 ms / 7 ms | 8.0 | 0.00% |
| 32 | 3.2k | 3.2k | 57 ms / 95 ms | 10 ms / 10 ms | 32.0 | 0.00% |
| 128 | 9.5k | 9.3k | 70 ms / 140 ms | 13 ms / 14 ms | 128.0 | 0.00% |
| 256 | 15.8k | 15.5k | 102 ms / 160 ms | 16 ms / 18 ms | 256.0 | 0.00% |
| 512 | 27.2k | 26.8k | 126 ms / 218 ms | 18 ms / 22 ms | 512.0 | 0.05% |
| 1024 | 47.1k | 46.4k | 264 ms / 422 ms | 20 ms / 23 ms | 1024.0 | 0.01% |
| 2048 | 66.6k | 65.8k | 528 ms / 788 ms | 29 ms / 30 ms | 2048.0 | 0.00% |
| 4096 | 73.1k | 72.2k | 914 ms / 1.7 s | 52 ms / 55 ms | 4096.0 | 0.00% |

### nvidia/Gemma-4-31B-IT-NVFP4 on vllm (tp1 x 8)

| Users | Gen tok/s | Engine tok/s | TTFT p50 / p95 | ITL p50 / p95 | Running (p50) | Errors |
|---|---|---|---|---|---|---|
| 8 | 315 | 310 | 122 ms / 124 ms | 25 ms / 25 ms | 8.0 | 0.00% |
| 32 | 1.2k | 1.2k | 85 ms / 410 ms | 26 ms / 27 ms | 32.0 | 0.00% |
| 128 | 4.2k | 4.2k | 89 ms / 105 ms | 30 ms / 30 ms | 128.0 | 0.00% |
| 256 | 7.2k | 7.1k | 102 ms / 133 ms | 35 ms / 35 ms | 256.0 | 0.00% |
| 512 | 11.8k | 11.6k | 136 ms / 626 ms | 42 ms / 46 ms | 512.0 | 0.00% |
| 1024 | 19.3k | 19.0k | 169 ms / 5.2 s | 50 ms / 53 ms | 1024.0 | 0.00% |
| 2048 | 29.0k | 28.5k | 257 ms / 5.5 s | 67 ms / 67 ms | 2048.0 | 0.00% |
| 4096 | 11.2k | 11.7k | 6.1 s / 85.7 s | 107 ms / 212 ms | 930.0 | 0.01% |

### nvidia/Gemma-4-31B-IT-NVFP4 on sglang (tp1 x 8)

| Users | Gen tok/s | Engine tok/s | TTFT p50 / p95 | ITL p50 / p95 | Running (p50) | Errors |
|---|---|---|---|---|---|---|
| 8 | 304 | 301 | 177 ms / 190 ms | 26 ms / 26 ms | 8.0 | 0.00% |
| 32 | 1.1k | 1.1k | 202 ms / 397 ms | 28 ms / 28 ms | 32.0 | 0.00% |
| 128 | 3.8k | 3.7k | 182 ms / 1.6 s | 32 ms / 34 ms | 128.0 | 0.00% |
| 256 | 6.1k | 6.1k | 208 ms / 3.3 s | 38 ms / 44 ms | 256.0 | 0.00% |
| 512 | 10.1k | 10.0k | 245 ms / 5.6 s | 46 ms / 53 ms | 512.0 | 0.00% |
| 1024 | 17.3k | 17.2k | 287 ms / 7.7 s | 54 ms / 59 ms | 1024.0 | 0.00% |
| 2048 | 21.7k | 21.5k | 299 ms / 10.0 s | 80 ms / 102 ms | 1989.0 | 0.04% |
| 4096 | 19.0k | 18.8k | 10.1 s / 20.7 s | 153 ms / 236 ms | 3337.5 | 0.00% |

### openai/gpt-oss-120b on sglang (tp1 x 8)

| Users | Gen tok/s | Engine tok/s | TTFT p50 / p95 | ITL p50 / p95 | Running (p50) | Errors |
|---|---|---|---|---|---|---|
| 8 | 1.2k | 1.2k | 70 ms / 75 ms | 6 ms / 6 ms | 8.0 | 0.00% |
| 32 | 3.0k | 2.9k | 79 ms / 151 ms | 10 ms / 11 ms | 32.0 | 0.00% |
| 128 | 6.6k | 6.5k | 126 ms / 415 ms | 19 ms / 21 ms | 128.0 | 0.03% |
| 256 | 9.2k | 9.1k | 550 ms / 4.6 s | 24 ms / 25 ms | 232.0 | 0.00% |
| 512 | 9.1k | 9.0k | 7.0 s / 12.8 s | 23 ms / 25 ms | 232.0 | 0.00% |
| 1024 | 9.2k | 9.1k | 19.8 s / 25.9 s | 23 ms / 25 ms | 232.0 | 0.00% |
| 2048 | 9.5k | 9.4k | 45.7 s / 51.2 s | 23 ms / 25 ms | 232.0 | 0.05% |
| 4096 | 9.6k | 9.6k | 81.6 s / 100.6 s | 23 ms / 25 ms | 235.0 | 0.00% |

### nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4 on vllm (tp2 x 4)

| Users | Gen tok/s | Engine tok/s | TTFT p50 / p95 | ITL p50 / p95 | Running (p50) | Errors |
|---|---|---|---|---|---|---|
| 8 | 661 | 650 | 299 ms / 310 ms | 11 ms / 12 ms | 8.0 | 0.00% |
| 32 | 1.7k | 1.7k | 622 ms / 649 ms | 16 ms / 18 ms | 32.0 | 0.00% |
| 128 | 3.7k | 3.6k | 1.3 s / 1.6 s | 30 ms / 32 ms | 128.0 | 0.00% |
| 256 | 4.9k | 4.8k | 1.5 s / 2.9 s | 46 ms / 49 ms | 256.0 | 0.00% |
| 512 | 6.3k | 6.2k | 1.6 s / 6.7 s | 75 ms / 78 ms | 512.0 | 0.00% |
| 1024 | 7.3k | 7.2k | 1.7 s / 15.2 s | 131 ms / 134 ms | 964.0 | 0.00% |
| 2048 | 7.7k | 7.6k | 5.9 s / 34.5 s | 222 ms / 232 ms | 1817.0 | 0.00% |
| 4096 | 5.6k | 7.6k | 59.7 s / 95.2 s | 229 ms / 235 ms | 1848.0 | 0.00% |

### reflection/Beam-1.0-501B-A23B-NVFP4 on vllm (tp8 x 1)

| Users | Gen tok/s | Engine tok/s | TTFT p50 / p95 | ITL p50 / p95 | Running (p50) | Errors |
|---|---|---|---|---|---|---|
| 8 | 411 | 404 | 890 ms / 907 ms | 16 ms / 16 ms | 8.0 | 0.00% |
| 32 | 928 | 914 | 281 ms / 1.8 s | 30 ms / 38 ms | 32.0 | 0.00% |
| 128 | 2.7k | 2.6k | 381 ms / 907 ms | 47 ms / 47 ms | 128.0 | 0.00% |
| 256 | 3.8k | 3.8k | 573 ms / 1.7 s | 65 ms / 65 ms | 256.0 | 0.00% |
| 512 | 5.2k | 5.1k | 1.2 s / 2.0 s | 95 ms / 96 ms | 512.0 | 0.00% |
| 1024 | 6.3k | 6.0k | 3.6 s / 8.4 s | 145 ms / 159 ms | 1013.0 | 0.00% |
| 2048 | 6.1k | 6.3k | 40.7 s / 47.6 s | 149 ms / 155 ms | 1024.0 | 0.00% |
| 4096 | 6.6k | 6.5k | 121.8 s / 128.6 s | 149 ms / 160 ms | 1024.0 | 0.49% |

### nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4 on sglang (tp2 x 4)

| Users | Gen tok/s | Engine tok/s | TTFT p50 / p95 | ITL p50 / p95 | Running (p50) | Errors |
|---|---|---|---|---|---|---|
| 8 | 756 | 744 | 299 ms / 315 ms | 9 ms / 9 ms | 8.0 | 0.00% |
| 32 | 2.1k | 2.1k | 295 ms / 314 ms | 14 ms / 14 ms | 32.0 | 0.00% |
| 128 | 4.6k | 4.6k | 659 ms / 936 ms | 25 ms / 25 ms | 128.0 | 0.00% |
| 256 | 4.7k | 4.7k | 3.0 s / 12.5 s | 35 ms / 42 ms | 216.0 | 0.00% |
| 512 | 4.8k | 4.7k | 14.7 s / 24.7 s | 35 ms / 42 ms | 216.0 | 0.00% |
| 1024 | 4.9k | 4.9k | 44.2 s / 48.6 s | 34 ms / 42 ms | 216.0 | 0.00% |
| 2048 | 4.9k | 4.9k | 84.9 s / 102.0 s | 35 ms / 41 ms | 216.0 | 0.00% |
| 4096 | 5.1k | 5.2k | 86.2 s / 161.7 s | 34 ms / 39 ms | 216.0 | 0.00% |

### reflection/Beam-1.0-501B-A23B-NVFP4 + DFlash2 drafter on vllm (tp8 x 1)

| Users | Gen tok/s | Engine tok/s | TTFT p50 / p95 | ITL p50 / p95 | Running (p50) | Errors |
|---|---|---|---|---|---|---|
| 8 | 418 | 406 | 301 ms / 403 ms | 18 ms / 25 ms | 8.0 | 0.00% |
| 32 | 1.2k | 1.2k | 290 ms / 587 ms | 24 ms / 37 ms | 31.0 | 0.00% |
| 128 | 2.3k | 2.3k | 675 ms / 715 ms | 52 ms / 70 ms | 125.0 | 0.00% |
| 256 | 2.7k | 2.7k | 1.2 s / 1.2 s | 88 ms / 119 ms | 250.0 | 0.00% |
| 512 | 3.0k | 2.8k | 2.2 s / 2.7 s | 158 ms / 217 ms | 498.0 | 0.00% |
| 1024 | 2.9k | 3.0k | 41.3 s / 48.2 s | 175 ms / 230 ms | 532.0 | 0.00% |
| 2048 | 2.6k | 2.9k | 56.2 s / 129.0 s | 180 ms / 235 ms | 533.0 | 0.00% |
| 4096 | 3.0k | 2.9k | 310.3 s / 314.1 s | 182 ms / 238 ms | 532.0 | 0.00% |

### nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-NVFP4 on sglang (tp8 x 1)

| Users | Gen tok/s | Engine tok/s | TTFT p50 / p95 | ITL p50 / p95 | Running (p50) | Errors |
|---|---|---|---|---|---|---|
| 8 | 281 | 277 | 1.5 s / 1.6 s | 23 ms / 26 ms | 8.0 | 0.00% |
| 32 | 512 | 505 | 3.6 s / 6.0 s | 47 ms / 60 ms | 32.0 | 0.00% |
| 128 | 799 | 726 | 15.0 s / 37.2 s | 76 ms / 105 ms | 76.0 | 0.00% |
| 256 | 775 | 700 | 47.6 s / 72.9 s | 76 ms / 105 ms | 84.0 | 0.00% |
| 512 | 775 | 784 | 145.5 s / 166.9 s | 72 ms / 99 ms | 84.0 | 0.00% |
| 1024 | 1.0k | 925 | 257.8 s / 279.2 s | 60 ms / 94 ms | 84.0 | 0.00% |
| 2048 | 957 | 908 | 536.6 s / 568.7 s | 59 ms / 97 ms | 84.0 | 0.00% |
| 4096 | 1.2k | 1.1k | 683.3 s / 864.7 s | 54 ms / 75 ms | 84.0 | 0.00% |

### nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-NVFP4 on vllm (tp8 x 1)

| Users | Gen tok/s | Engine tok/s | TTFT p50 / p95 | ITL p50 / p95 | Running (p50) | Errors |
|---|---|---|---|---|---|---|
| 8 | 262 | 258 | 1.7 s / 1.7 s | 24 ms / 29 ms | 8.0 | 0.00% |
| 32 | 446 | 437 | 3.0 s / 4.2 s | 60 ms / 67 ms | 32.0 | 0.00% |
| 128 | 763 | 756 | 4.3 s / 15.6 s | 157 ms / 168 ms | 123.0 | 0.00% |
| 256 | 912 | 905 | 4.4 s / 35.2 s | 278 ms / 288 ms | 242.0 | 0.00% |
| 512 | 1.1k | 983 | 23.6 s / 68.1 s | 368 ms / 522 ms | 370.0 | 0.00% |
| 1024 | 967 | 860 | 123.2 s / 173.5 s | 678 ms / 678 ms | 629.0 | 0.00% |
| 2048 | 1.0k | 898 | 385.1 s / 421.1 s | 678 ms / 679 ms | 629.0 | 0.00% |
| 4096 | 957 | 970 | 576.6 s / 900.9 s | 678 ms / 679 ms | 629.0 | 0.00% |
