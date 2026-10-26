# P11 scoping: US model survey (2026-10-25)

Status: measured on the XE7740 2026-10-25/26; results in [p11-results.md](p11-results.md). In `catalog.json` as P11.

## The ask

A quick survey of four US open-weight model families on one Dell XE7740, served by vLLM
or SGLang and benchmarked with AIPerf. It's a showcase of what an all-US model lineup
delivers on one server, with no comparison to non-US models.

Story for buyers who won't run Chinese models: "Here's what a single XE7740 serves with
US models only."

## Models

| Family | Model | Size (total / active) | Weights | Fit on XE7740 (8 × 96 GB) |
|---|---|---|---|---|
| OpenAI | gpt-oss-120b, gpt-oss-20b | 117B / 5B, 21B / 4B | MXFP4 | 1 GPU each |
| Google | Gemma 4 (26B already in the roofline at 127k gen tok/s) | — | NVFP4 / BF16 | 1 GPU |
| Reflection | Beam 1.0 501B-A23B | 501B / 23B MoE | NVFP4, about 255 GB | tp4 (two copies per box) or tp8. DFlash2 drafter on/off. |
| NVIDIA | Nemotron 3 Ultra 550B-A55B, Nemotron 3 Super | 550B / 55B hybrid MoE | NVFP4, about 330 GB for Ultra | Ultra on tp8, Super smaller |

Beam weights went public 2026-10-16 under Apache 2.0, with vLLM and SGLang support at
launch.

## Method

1. Preflight each model: does it load and serve on vLLM or SGLang? The RTX PRO 6000 is
   SM120. Beam's long-context attention and Nemotron's Mamba layers may lack SM120
   kernels. Report a failure as "unsupported on this platform"; don't drop it silently.
2. Run an AIPerf ladder per model and engine on the same workload, recording latency and
   throughput at every rung. Latency targets are applied afterwards, not during the run.
3. Run a short quality check with the same prompts for every model. gpt-oss already
   failed our 96K quality check in P02, so report the context each model passes at.
4. Record image digests and model revisions so the lineup can be reproduced.

## Test plan (as executed, driver `scripts/p11_us_survey.py`)

| Setting | Value |
|---|---|
| Box | XE7740, 8 × RTX PRO 6000 Blackwell Server 96 GB, PCIe only |
| Engines | vLLM `vllm/vllm-openai:latest` (v0.29.0), SGLang `lmsysorg/sglang:latest` (v0.5.20), Beam on `vllm/vllm-openai:reflection-beam`. Image digests are recorded per run. |
| Placement | Smallest TP that fits, replicated to fill all 8 GPUs. AIPerf round-robins requests across replicas, so per-server numbers are measured rather than multiplied up from one GPU. |
| Load | AIPerf 0.13.0, synthetic chat prompts, 1,024 input / 256 output tokens, `ignore_eos`, closed-loop concurrency 8, 32, 128, 256, 512, 1,024, 2,048, 4,096, 120 s per rung after 30 s warmup. Four AIPerf processes per rung (one alone tops out near 400 requests/s), per-request records merged. |
| Targets | None applied during the run. Every rung is recorded; a ladder stops only on real errors (AIPerf failure or > 5% failed requests). Latency targets are applied afterwards. |
| Engines | Full ladder on both vLLM and SGLang for every model (Beam: vLLM only). A 1,024-user screen on both engines ran first. |
| Cross-check | Engine-side generation-token counter from /metrics per rung, compared with AIPerf throughput |
| Quality | 32 synthetic documents at about 8K and about 32K tokens, exact-match JSON extraction (`simulator/infrastructure/workload.py`), temperature 0, reasoning effort low where supported |
| Recorded per rung | Gen tok/s, requests/s, TTFT and inter-token latency p50/p95, engine running requests, error fraction |

| Config | Model | TP × replicas |
|---|---|---|
| gpt-oss-20b | openai/gpt-oss-20b | 1 × 8 |
| gpt-oss-120b | openai/gpt-oss-120b | 1 × 8 |
| gemma-4-26b | nvidia/Gemma-4-26B-A4B-NVFP4 | 1 × 8 |
| gemma-4-31b | nvidia/Gemma-4-31B-IT-NVFP4 | 1 × 8 |
| nemotron-3-super | nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4 | 2 × 4 |
| nemotron-3-ultra | nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-NVFP4 | 8 × 1 |
| beam / beam-dflash | reflection/Beam-1.0-501B-A23B-NVFP4 (DFlash2 drafter off/on), vLLM only | 8 × 1 |

Beam's documented minimum is 353 GB of GPU memory, so tp4 (384 GB) leaves almost no KV
cache on this box, and it runs at tp8.

## Outputs

- **O1:** Capacity curve per XE7740 for each model and engine (targets applied afterwards).
- **O2:** One survey table with each model's capacity, quality, and GPUs used.

## Out of scope

Comparisons with non-US models, audio/speech/image models, TensorRT-LLM, and security
claims beyond "US-origin models on pinned open-source engines".
