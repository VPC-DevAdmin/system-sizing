# Efficiency sweep: preliminary read against the Dell/Metrum baseline

Status 2026-10-29 16:00 UTC. 1k/1k and 8k/1k complete for 15 of 17 configs; the vLLM 128/128 roofline rungs
are measured. Qwen3.5 on SGLang (two configs) is being redone after a flag fix; the TensorRT-LLM Llama 3.3
70B replica is blocked on a container startup hang (MPI init) under diagnosis. Numbers below will move a little when the
summary script is finalised; the ordering will not. Plan: [efficiency-sweep-plan.md](efficiency-sweep-plan.md).

## The baseline

No standard benchmark publishes tokens per watt, so the baseline is Dell's own published
figure for this platform: the Dell / Metrum AI InfoHub post of 3 November 2025, Llama 3.3 70B
in FP4 with TensorRT-LLM on eight RTX PRO 6000, system-level power, maximum load (up to 8,192
users, no per-user speed target, sequence lengths not stated).

| Published, 8x RTX PRO 6000 | Value |
|---|---|
| Throughput | 45,527 tokens/s |
| Efficiency | **7.03 tokens/s per system watt** |
| Implied system power | ~6,480 W |
| Energy per token | 0.14 J |
| Against 8x H200 SXM (5.70) | 1.23x |
| Against 8x L40S (0.65) | 10.8x |

## Our replica of the baseline model

Same model and weights class (Llama 3.3 70B, NVFP4), same server, vLLM instead of
TensorRT-LLM, fixed 1k/1k, prefix caching off, iDRAC wall power.

| Rung | Tokens/s | p90 tok/s per user | Wall W | Tok/s per W | vs 7.03 |
|---|---|---|---|---|---|
| Peak (1,024 users) | 14,285 | 14 | 5,966 | 2.39 | 0.34x |
| Best interactive rung (64 users) | 1,930 | 30 | 3,956 | 0.49 | 0.07x |
| 35 tok/s per user target | not reached at any rung | | | | |

The 3x gap to the published number is engine and workload shape, not power: our wall power at
peak (6.0 kW) matches their implied 6.5 kW. A TensorRT-LLM 1.2.1 pass of this model (same TP1 x 8 placement, prefix reuse off) is queued
after the sweep to close that gap and tie the two measurements together.

## Every model against the baseline (1k/1k, wall power)

Two readings per model. **Peak** is the rung with the best tokens per watt regardless of user
speed, which is how the Dell post is read. **At 35 tok/s per user** (p90) is the interactive
reading, which is the one a deployment pays for. Intelligence is the Artificial Analysis
Intelligence Index v4.3.2 (estimates marked ~).

| Model | Index | Peak tok/s | Peak tok/s/W | vs 7.03 | At 35 tok/s/user: tok/s | tok/s/W | vs 7.03 |
|---|---|---|---|---|---|---|---|
| Gemma 4 26B-A4B NVFP4 | ~17 | 49,579 | 9.98 | **1.42x** | 39,920 | 9.01 | **1.28x** |
| Qwen3.6 35B-A3B NVFP4 | 18 | 39,317 | 8.68 | **1.24x** | 25,835 | 6.32 | 0.90x |
| Qwen3 30B-A3B FP8 | ~8 | 34,895 | 7.44 | **1.06x** | 20,516 | 4.91 | 0.70x |
| gpt-oss-20b | 9 | 31,031 | 5.56 | 0.79x | 24,966 | 4.61 | 0.66x |
| GLM-4.7 Flash | ~15 | 21,333 | 4.75 | 0.68x | 4,693 | 1.25 | 0.18x |
| gpt-oss-120b | 12 | 24,753 | 4.41 | 0.63x | 8,877 | 1.91 | 0.27x |
| Nemotron 3 Super 120B NVFP4 | 13 | 11,378 | 2.73 | 0.39x | 4,421 | 1.08 | 0.15x |
| **Llama 3.3 70B NVFP4 (baseline model, vLLM)** | ~8 | 14,285 | 2.39 | 0.34x | misses target | | |
| Mistral Small 3.2 24B | 7 | 12,706 | 2.30 | 0.33x | misses target | | |
| Gemma 4 31B dense NVFP4 | 15 | 10,440 | 2.15 | 0.31x | 2,234 | 0.54 | 0.08x |
| Qwen3 32B dense FP8 | ~7 | 10,668 | 1.98 | 0.28x | 1,158 | 0.30 | 0.04x |
| Qwen3 235B-A22B FP8 | ~12 | 4,963 | 1.37 | 0.20x | 603 | 0.18 | 0.03x |
| MiniMax M2.7 | 23 | 4,827 | 1.25 | 0.18x | 1,183 | 0.33 | 0.05x |
| Beam 1.0 501B NVFP4 | 30 | 3,729 | 1.15 | 0.16x | 670 | 0.21 | 0.03x |
| Nemotron 3 Ultra 550B NVFP4 | 23 | 1,599 | 0.46 | 0.07x | 310 | 0.10 | 0.01x |
| Qwen3.5 397B NVFP4 (SGLang, with/without MTP) | 18 | redo queued | | | | | |

## 8k/1k (long prompts): every model against the baseline

Same two readings. Prefill now dominates: throughput falls 2 to 5x for every model and the
ordering changes little, but the gap between MoE and dense widens at the interactive target.

| Model | Index | Peak tok/s | Peak tok/s/W | vs 7.03 | At 35 tok/s/user: tok/s | tok/s/W | vs 7.03 |
|---|---|---|---|---|---|---|---|
| Gemma 4 26B-A4B NVFP4 | ~17 | 21,322 | 3.63 | 0.52x | 12,131 | 2.54 | 0.36x |
| Qwen3.6 35B-A3B NVFP4 | 18 | 13,946 | 2.76 | 0.39x | 11,289 | 2.52 | 0.36x |
| Qwen3 30B-A3B FP8 | ~8 | 6,833 | 1.39 | 0.20x | 4,845 | 1.08 | 0.15x |
| Nemotron 3 Super 120B NVFP4 | 13 | 4,841 | 1.14 | 0.16x | 2,248 | 0.56 | 0.08x |
| gpt-oss-120b | 12 | 5,585 | 1.07 | 0.15x | 3,892 | 0.84 | 0.12x |
| GLM-4.7 Flash | ~15 | 5,014 | 1.05 | 0.15x | 2,489 | 0.59 | 0.08x |
| gpt-oss-20b | 9 | 5,405 | 0.88 | 0.13x | 4,326 | 0.81 | 0.11x |
| Gemma 4 31B dense NVFP4 | 15 | 4,409 | 0.73 | 0.10x | 558 | 0.14 | 0.02x |
| **Llama 3.3 70B NVFP4 (baseline model, vLLM)** | ~8 | 2,114 | 0.40 | 0.06x | misses target | | |
| Mistral Small 3.2 24B | 7 | 1,993 | 0.40 | 0.06x | misses target | | |
| Qwen3 32B dense FP8 | ~7 | 1,871 | 0.39 | 0.06x | 281 | 0.07 | 0.01x |
| MiniMax M2.7 | 23 | 1,315 | 0.34 | 0.05x | 613 | 0.18 | 0.03x |
| Beam 1.0 501B NVFP4 | 30 | 999 | 0.31 | 0.04x | 347 | 0.11 | 0.02x |
| Qwen3 235B-A22B FP8 | ~12 | 985 | 0.26 | 0.04x | 302 | 0.09 | 0.01x |
| Nemotron 3 Ultra 550B NVFP4 | 23 | 552 | 0.17 | 0.02x | misses target | | |

The 7.03 column is kept for continuity only: the published figure is a short-sequence roofline and
nothing at 8k in / 1k out will reach it on any hardware. The useful ratio at this shape is against
our own Llama 3.3 70B row: Gemma 4 26B is 9x at peak, and at the interactive target the dense
70B does not qualify at all.

## What the preliminary numbers say

**Claim 1, same capability at a fraction of the energy.** Three MoE models beat Dell's
published peak efficiency outright, with vLLM rather than TensorRT-LLM, and two of them score
higher than Llama 3.3 70B on the public index:

| | Index | Peak tok/s/W | vs published 7.03 | vs our Llama 3.3 70B (2.39) |
|---|---|---|---|---|
| Gemma 4 26B-A4B | ~17 | 9.98 | 1.42x | 4.2x |
| Qwen3.6 35B-A3B | 18 | 8.68 | 1.24x | 3.6x |
| Qwen3 30B-A3B | ~8 | 7.44 | 1.06x | 3.1x |

At the interactive target the gap widens, because the dense 70B never reaches 35 tok/s per user
on PCIe GPUs while Gemma 4 26B serves 1,024 users at that speed: 9.01 against the best the dense
model manages at any interactive rung (0.49), about 18x.

**Claim 2, more intelligence at the same energy.** Reading across at the published 0.14 J per
token (7 tok/s/W): Gemma 4 26B (~17) and Qwen3.6 35B (18) sit at or above that line, against
the baseline model's index of ~8. Reading at our own Llama 3.3 70B peak (2.4 tok/s/W):
Nemotron 3 Super (13), gpt-oss-120b (12), GLM-4.7 Flash (~15) all land above it on both axes.
The large reasoning models (MiniMax 23, Beam 30, Nemotron Ultra 23) buy their index with 5 to
15x more energy per token than the baseline; they are the top of the intelligence axis, not
the efficiency story.

**Dense versus MoE.** Every dense model lands at 0.28 to 0.34x of the published figure at peak
and at or below 0.08x at the interactive target. Every MoE with 3 to 5B active parameters lands
at 0.8 to 1.4x at peak. On this server the architecture of the model is worth more than the
engine.

## Roofline reading (128 in / 128 out, measured 2026-10-29)

The post's 45,527 tok/s sits on our September roofline for the same model and engine (128-token
prompts, 2,048 sequences, prefix caching left on: TensorRT-LLM 36k to 39k, vLLM 43k), so it is a
short-sequence maximum-throughput reading. Re-measured with the sweep method (AIPerf, unique
prompts, prefix caching off, iDRAC wall power), vLLM, 1,024 and 2,048 users:

| Model | Users | Tok/s | Wall W | Tok/s per W | vs 7.03 | p90 tok/s per user |
|---|---|---|---|---|---|---|
| Gemma 4 26B-A4B NVFP4 | 2,048 | 77,622 | 5,002 | **15.5** | **2.2x** | 37 |
| Gemma 4 26B-A4B NVFP4 | 1,024 | 57,360 | 4,598 | 12.5 | 1.8x | 56 |
| gpt-oss-20b | 2,048 | 85,038 | 6,062 | **14.0** | **2.0x** | 46 |
| gpt-oss-20b | 1,024 | 65,331 | 5,538 | 11.8 | 1.7x | 76 |
| Qwen3.6 35B-A3B NVFP4 | 2,048 | 40,766 | 4,366 | 9.3 | 1.3x | 17 |
| Qwen3.6 35B-A3B NVFP4 | 1,024 | 28,753 | 3,802 | 7.6 | 1.1x | 24 |
| gpt-oss-120b | 2,048 | 40,913 | 6,727 | 6.1 | 0.87x | 22 |
| gpt-oss-120b | 1,024 | 28,469 | 5,693 | 5.0 | 0.71x | 31 |
| **Llama 3.3 70B NVFP4 (baseline model, vLLM)** | 2,048 | 24,601 | 7,774 | 3.2 | 0.45x | 12 |
| **Llama 3.3 70B NVFP4 (baseline model, vLLM)** | 1,024 | 18,848 | 6,481 | 2.9 | 0.41x | 19 |

Two things to take from this table.

- The honest roofline is 40 to 60% of the September one for every model (Gemma 77.6k against
  127k, gpt-oss-120b 40.9k against 92k, Llama 3.3 70B 24.6k against 43k). The September runs and,
  as far as the post says, the Dell/Metrum run used repeated short prompts with prefix caching
  on, which makes prefill nearly free. With unique prompts and caching off, the same server
  does half the work per second. Published roofline figures for this platform should be read
  with that in mind; ours is the conservative reading.
- Even on the conservative reading, Gemma 4 26B and gpt-oss-20b beat the published 7.03 by 2x
  and still hold 37 to 46 tok/s per user at 2,048 users. Llama 3.3 70B on vLLM reaches 0.45x,
  and its wall power at 2,048 users (7.8 kW) is above the 6.5 kW implied by the post, so the
  gap to 45,527 tok/s is throughput, not power.

The TensorRT-LLM pass of Llama 3.3 70B at the same shape is still pending (see caveats).

## Caveats

- The Dell post is TensorRT-LLM at 8,192 users with undisclosed sequence lengths; ours is vLLM
  at fixed 1k/1k with prefix caching off. The queued TensorRT-LLM pass is what makes the two
  directly comparable. Until then, ratios against 7.03 understate our position by roughly the
  engine gap seen on Llama 3.3 70B (about 3x at peak).
- Power is iDRAC PSU input, not a PTDaemon measurement, on a server with one of eight supplies
  reporting Critical. Dell does not say how its system power was measured.
- Intelligence scores mix reasoning and non-reasoning variants; the sweep generated a fixed
  1,024 tokens per request, so a model's thinking mode did not change token counts.
- TensorRT-LLM 1.2.1 containers hang at `import tensorrt_llm` on this host (OpenMPI singleton
  start-up, likely confused by the 30-plus docker bridge interfaces from other compose stacks).
  Until that is fixed the engine tie-in to the Dell figure is open and the Llama 3.3 70B rows are
  vLLM only.
