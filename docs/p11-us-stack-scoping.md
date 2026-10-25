# P11 scoping: US model survey (2026-10-25)

Status: planned, not measured. In `catalog.json` as P11.

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
2. Run an AIPerf sweep per model on the same workload and SLO, picking the better of vLLM
   or SGLang for each model. Record the best concurrency at SLO and gen tok/s per server.
3. Run a short quality check with the same prompts for every model. gpt-oss already
   failed our 96K quality check in P02, so report the context each model passes at.
4. Record image digests and model revisions so the lineup can be reproduced.

## Outputs

- **O1:** Capacity per XE7740 at SLO for each model.
- **O2:** One survey table with each model's capacity, quality, and GPUs used.

## Out of scope

Comparisons with non-US models, audio/speech/image models, TensorRT-LLM, and security
claims beyond "US-origin models on pinned open-source engines".
