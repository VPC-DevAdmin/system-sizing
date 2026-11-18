# Efficiency sweep: intelligence per wall kilowatt-hour on one XE7740

Started 2026-10-27. Driver [`scripts/efficiency_sweep.py`](../scripts/efficiency_sweep.py),
wall power [`scripts/idrac_power.py`](../scripts/idrac_power.py), summary
[`scripts/efficiency_summary.py`](../scripts/efficiency_summary.py). Raw artifacts stay on
the box under `/data/capsim/effsweep/results`.

## The claim

Starting point: Dell's own published efficiency result for this platform, the Dell / Metrum AI
post "Breaking the Performance Barrier: Dell PowerEdge XE7745 & XE7740 with NVIDIA RTX PRO 6000
Blackwell" (Dell InfoHub, 3 November 2025). It ran Llama 3.3 70B in FP4 with TensorRT-LLM on
eight RTX PRO 6000 at up to 8,192 concurrent users and reported, at the system level:

| | Tokens/s | Tokens/s per W | Implied system W | J per token |
|---|---|---|---|---|
| 8x RTX PRO 6000, FP4 | 45,527 | 7.03 | ~6,480 | 0.14 |
| 8x H200 SXM (XE9680), FP8 | 48,655 | 5.70 | ~8,540 | 0.18 |
| 8x L40S, FP8 | 3,929 | 0.65 | ~6,040 | 1.54 |

That post makes the hardware claim (1.2x H200, 10.8x L40S per watt). It does not state sequence
lengths or any per-user speed target, and it fixes the model at a dense 70B. This sweep keeps the
server and asks what the choice of model does to the same metric, measured one way for every
model and read at an interactivity target. The two statements:

1. **Same capability at a fraction of the energy per token.** Models that match or beat
   Llama 3.3 70B on public benchmarks use several times less energy per output token on the
   same server, and do so while meeting an interactive speed the dense 70B cannot.
2. **Significantly more intelligence at the same energy per token.** At the energy per token of
   the published Llama 3.3 70B result, today's MoE models score far higher.

Nothing about the hardware changed between the points. The improvement is models and serving
software, which is the argument that the server appreciates in useful work per kilowatt-hour
over its life.

Llama 3.3 70B NVFP4 is in the sweep twice: on vLLM like every other model, and on
TensorRT-LLM, the anchor's engine, so our wall-power measurement ties to Dell's figure directly. Llama 2 70B was dropped on 2026-10-28 (the published
MLPerf rows for it carry no power, so it anchored nothing).

## Method, kept simple

- **Load.** AIPerf closed loop, fixed sequence shapes 1k/1k (1,024 in, 1,024 out) and
  8k/1k (8,192 in, 1,024 out), `ignore_eos`, server-wide concurrency ladder
  8, 16, 32, 64, 128, 256, 512, 1,024, 2,048 (capped lower for models that run as one
  copy across all eight GPUs). Every rung is recorded. The headline is read at the rung
  where 90% of users still receive at least 35 tokens per second, InferenceX's default
  interactivity target, so the numbers also compare to that site.
- **Prefix caching off** in every engine. AIPerf recycles a small prompt set and the engines
  otherwise serve the prefill from cache (measured: 100% hit rate), which is not a
  serving condition anyone pays for.
- **Power.** Two numbers per rung, both averaged over AIPerf's measurement window:
  GPU board power from `nvidia-smi` at 1 Hz, and server input power from the iDRAC
  (sum of the eight PSU inputs over Redfish, about one fresh sample every 2 s). Wall
  energy per token is the headline; GPU energy per token is kept so the rows read against
  InferenceX, which has GPU power only.
- **Reasoning.** Throughput runs generate a fixed 1,024 tokens per request, so a model's
  thinking mode does not change the token count; whatever a model emits counts. Intelligence
  scores are taken for the mode closest to how the model was served.
- **Placement and engine.** Smallest tensor-parallel size that fits, copied to fill the
  eight GPUs; vLLM v0.29 for most models, SGLang v0.5.20 for Qwen3.5 (NVIDIA's own
  InferenceX recipe, with multi-token prediction), Reflection's vLLM image for Beam. Engine
  settings carried over from the P11 survey are listed in
  [p11-results.md](p11-results.md).
- **Intelligence axis.** Public scores, not our own evaluation: the Artificial Analysis
  Intelligence Index where the model is listed, otherwise the mean of MMLU-Pro, GPQA Diamond
  and LiveCodeBench from the model card. Recorded with source and date in the results JSON.

## Models

30B to 550B total parameters, mostly mixture-of-experts, plus the reference model and a few
dense models so the dense-versus-MoE difference is visible. All were already cached on the box.

| Group | Model | Total / active params | Weights | Engine, placement |
|---|---|---|---|---|
| Reference | Llama-3.3-70B-Instruct NVFP4 (the published anchor's model) | 70B dense | NVFP4 | vLLM, TP1 x 8; and TensorRT-LLM 1.2.1, TP1 x 8 (the anchor's engine; prefix reuse off via options YAML) |
| Dense | Gemma-4-31B-IT NVFP4 | 31B | NVFP4 | vLLM, TP1 x 8 |
| Dense | Qwen3-32B FP8 | 32B | FP8 | vLLM, TP1 x 8 |
| Dense | Mistral-Small-3.2-24B | 24B | BF16 | vLLM, TP1 x 8 |
| MoE | gpt-oss-20b | 21B / 4B | MXFP4 | vLLM, TP1 x 8 |
| MoE | gpt-oss-120b | 117B / 5B | MXFP4 | vLLM, TP1 x 8 |
| MoE | Gemma-4-26B-A4B NVFP4 | 26B / 4B | NVFP4 | vLLM, TP1 x 8 |
| MoE | Qwen3-30B-A3B-Instruct-2507 FP8 | 30B / 3B | FP8 | vLLM, TP1 x 8 |
| MoE | Qwen3.6-35B-A3B NVFP4 | 35B / 3B | NVFP4 | vLLM, TP1 x 8 |
| MoE | GLM-4.7-Flash | 30B / 3B | BF16 | vLLM, TP1 x 8 |
| MoE | Nemotron-3-Super-120B-A12B NVFP4 | 120B / 12B | NVFP4 | vLLM, TP2 x 4 |
| MoE | Qwen3-235B-A22B-Instruct-2507 FP8 | 235B / 22B | FP8 | vLLM, TP4 x 2 |
| MoE | MiniMax-M2.7 | 230B / 10B | FP8 | vLLM, TP4 x 2 |
| MoE | Qwen3.5-397B-A17B NVFP4 | 397B / 17B | NVFP4 | SGLang, TP4 x 2, with and without MTP |
| MoE | Beam-1.0-501B-A23B NVFP4 | 501B / 23B | NVFP4 | vLLM (Reflection image), TP8 x 1 |
| MoE | Nemotron-3-Ultra-550B-A55B NVFP4 | 550B / 55B | NVFP4 | vLLM, TP8 x 1 |

Not included: GLM-5.3-Flash and Qwen3.8-Flash-Next (architectures not in vLLM v0.29),
Kimi-Linear-48B (hybrid, needs remote code; already covered in the x100k runs), anything
above 550B (one copy across eight GPUs at 1 to 5k tokens per second adds little to the chart).

## Order and time

Pass 1: 1k/1k for every model, smallest first. Pass 2: 8k/1k for every model. Then the
TensorRT-LLM Llama 3.3 70B pass at both shapes (queued 2026-10-28, `efficiency_queue4.sh`).
Then a roofline pass in the post's own reading, 128 in / 128 out at 1,024 and 2,048 users with wall
power and no speed target, for Gemma 4 26B, gpt-oss-20b, gpt-oss-120b, Qwen3.6 35B and Llama 3.3 70B
on vLLM and TensorRT-LLM (`efficiency_queue5.sh`). Our September roofline on that shape put
TensorRT-LLM Llama 3.3 70B at 36k to 39k tok/s and vLLM at 43k, against the post's 45,527, which is
what identifies the post as a short-sequence roofline.
About 45 to 60 minutes per model per shape including engine start, roughly 30 hours in all.
The queue starts after the P11 no-cache rerun and the InferenceX Qwen3.5 sweeps release the
GPUs. Everything resumes per rung if interrupted.

## Output

- `results/<shape>/<config>/result.json` (shapes 1k1k, 8k1k, 128x128): every rung with throughput, latency percentiles,
  GPU and wall power, energy per token, tokens per wall kW.
- A chart: intelligence score (y) against output tokens per wall kWh at the 35 tok/s/user
  target (x, log scale), one point per model, Llama 3.3 70B marked as the anchor, with the two claims drawn
  as arrows from it and the published 7.03 tok/s per W drawn as a reference line in the
  GPU-free (wall) units it was given in. A second version uses GPU-only energy so it reads against InferenceX.
- A table per model: headline rung, tokens/s per server, wall W, GPU W, GPU share of wall,
  J per output token (wall and GPU), tokens per wall kWh, intelligence score and source.

## Caveats to carry into the write-up

- Wall power is PSU input via iDRAC, not a calibrated analyzer: an MLPerf power result would
  need SPEC PTDaemon. iDRAC's chassis-level sensor was found to report only one PSU most
  of the time; the per-PSU sum is used instead. PS2 reported Critical at 0 W during setup,
  so the server ran on seven supplies.
- AIPerf uses fixed prompt lengths; InferenceX's client draws random lengths around the
  same means.
- Intelligence scores are third-party and change as leaderboards update; the date is
  recorded.
- One server, one driver and firmware state, one week. No claim about other servers.
