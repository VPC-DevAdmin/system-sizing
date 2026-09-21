# KTransformers: two generations behind one engine

capsim's `ktransformers` engine serves MoE models whose weights exceed
the GPUs by running the experts on the CPU (2× Xeon 6787P with AMX,
2 TB RAM on the XE7740) and attention on the GPUs. Two very different
upstream lines sit behind the one engine name, and
`engines/ktransformers.py` picks between them per launch.

| | v0.3 (archived) | v0.7 (current) |
|---|---|---|
| Server | `python -m ktransformers.server.main` (own scheduler, `balance_serve`) | `python -m sglang.launch_server` from the kvcache-ai SGLang fork (`sglang-kt`) with `--kt-*` flags |
| CPU kernels | compiled into the server image (AVX512 build) | `kt-kernel` (AMX INT4/INT8, native FP8/BF16/INT4, llamafile) |
| Weights | GGUF only; HF dir supplies config + tokenizer | the HF checkpoint itself (FP8, BF16, RAWINT4, MXFP4), or a converted AMX directory, or a GGUF (LLAMAFILE) |
| Architectures | an optimize-rule file each: `deepseek_v3`, `qwen3_moe` in the image | every FusedMoE model the fork loads: DeepSeek V3/V3.1/V3.2/V4, Kimi K2/K2-Thinking/K2.5, Qwen3-MoE/Next/3.5, GLM-4.5–5.x, MiniMax M2/M3 |
| Image | `approachingai/ktransformers:v0.3.2-AVX512` (8.7 GB) | `approachingai/ktransformers:DSV4-specific` (12.0 GB) |
| Readiness | `/v1/models` | `/health` (SGLang) |
| Metrics | none | SGLang `/metrics` with `--enable-metrics` (`sglang:` names) |

Config: `ktransformers_generation: auto | v0.3 | v0.7`. `auto` takes an
explicit `ktransformers_image` tag as the answer, otherwise looks at
what is staged: a safetensors checkpoint or `ktransformers_amx_weight_path`
means v0.7, a lone GGUF companion means v0.3.

## What was established, and where

Read on 2026-09-20 from the `v0.7.1` tag of
[kvcache-ai/ktransformers](https://github.com/kvcache-ai/ktransformers)
(tags: v0.7.1, v0.7.0.post4, v0.7.0, v0.6.4 … v0.5.3) and the
[kvcache-ai/sglang](https://github.com/kvcache-ai/sglang) fork (`main`).

**(a) The serving command.** The top-level README at v0.7.1 says the
original integrated framework "has been archived to the `archive/`
directory" and that inference is now kt-kernel "for SGLang and other
frameworks" ([README.md](https://github.com/kvcache-ai/ktransformers/blob/v0.7.1/README.md)).
[kt-kernel/README.md § Integration with SGLang](https://github.com/kvcache-ai/ktransformers/blob/v0.7.1/kt-kernel/README.md)
gives the launch: `python -m sglang.launch_server [normal SGLang
parameters] --kt-method … --kt-weight-path … --kt-cpuinfer …
--kt-threadpool-count … --kt-num-gpu-experts …
--kt-max-deferred-experts-per-token …`, and insists on the fork:
"Use `sglang-kt` (kvcache-ai fork), not the official `sglang` package."
The fork's
[server_args.py](https://github.com/kvcache-ai/sglang/blob/main/python/sglang/srt/server_args.py)
("Ktransformer server args" block) defines `--kt-weight-path`,
`--kt-method` (default `AMXINT4`), `--kt-cpuinfer`,
`--kt-threadpool-count` (default 2), `--kt-numa-nodes`,
`--kt-num-gpu-experts` (per MoE layer), `--kt-gpu-experts-ratio`,
`--kt-max-deferred-experts-per-token`, `--kt-gpu-prefill-token-threshold`,
`--kt-enable-dynamic-expert-update`, `--kt-expert-placement-strategy`
(`uniform | frequency | front-loading | random`), plus the LoRA paths.
Context, batch, port and metrics are SGLang's own `--context-length`,
`--max-running-requests`, `--chunked-prefill-size`/`--max-prefill-tokens`,
`--port`, `--enable-metrics`. There is no KT-specific server any more;
a `kt run <model>` CLI exists but is marked "under active development"
([doc/en/kt-kernel/kt-cli.md](https://github.com/kvcache-ai/ktransformers/blob/v0.7.1/doc/en/kt-kernel/kt-cli.md)).
Every tutorial launch also carries `--attention-backend flashinfer
--disable-shared-experts-fusion --enable-mixed-chunk --trust-remote-code`,
and the fork's `_init_kt_gpu_experts_masks` only *warns* when neither
`--kt-num-gpu-experts` nor `--kt-gpu-experts-ratio` is given, so capsim
always passes the count.

**(b) Weight format.** Chosen by `--kt-method`
([kt-kernel/README.md § KT-Kernel Parameters](https://github.com/kvcache-ai/ktransformers/blob/v0.7.1/kt-kernel/README.md),
[Native-Precision-Tutorial.md](https://github.com/kvcache-ai/ktransformers/blob/v0.7.1/doc/en/kt-kernel/Native-Precision-Tutorial.md)):

* Native, no conversion — `--kt-weight-path` is the same directory as
  `--model`: `FP8` (block-wise; DeepSeek-V3/R1/V3.2, MiniMax-M2, and
  Kimi-K2-Instruct which shares the deepseek_v3 block-FP8 layout),
  `FP8_PERCHANNEL` (GLM-4.7-FP8), `BF16` (Qwen3-235B, GLM-4.7),
  `RAWINT4` (Kimi-K2-Thinking / K2.5 compressed-tensors INT4 —
  [Kimi-K2-Thinking-Native.md](https://github.com/kvcache-ai/ktransformers/blob/v0.7.1/doc/en/kt-kernel/Kimi-K2-Thinking-Native.md),
  [Kimi-K2.5.md](https://github.com/kvcache-ai/ktransformers/blob/v0.7.1/doc/en/Kimi-K2.5.md)),
  `MXFP4` (DeepSeek-V4-Flash).
* `AMXINT4` / `AMXINT8` — CPU experts pre-quantized by
  `kt-kernel/scripts/convert_cpu_weights.py --input-path … --input-type
  fp8|bf16|fp16 --output … --quant-method int4|int8`
  ([scripts/README.md](https://github.com/kvcache-ai/ktransformers/blob/v0.7.1/kt-kernel/scripts/README.md));
  `--no-merge-safetensor` and `--resume-layer` exist for RAM-bound
  hosts. The
  [DeepSeek-V3.2 tutorial](https://github.com/kvcache-ai/ktransformers/blob/v0.7.1/doc/en/kt-kernel/deepseek-v3.2-sglang-tutorial.md)
  converts FP8→INT4 with `--cpuinfer-threads 60 --threadpool-count 2`
  and then serves with `--kt-method AMXINT4` (~350 GB RAM, ~27 GB VRAM
  at 1 GPU expert per layer). The scripts README warns that FP8→INT4/8
  "may cause significant accuracy degradation".
* `LLAMAFILE` — a GGUF directory (Q4_K_M etc.), portable AVX2/AVX512
  kernels, not AMX.

The fork's [kt_ep_wrapper.py](https://github.com/kvcache-ai/sglang/blob/main/python/sglang/srt/layers/moe/kt_ep_wrapper.py)
wraps `Fp8MoEMethod` (block), `CompressedTensorsW8A8Fp8` (block and
channel), `UnquantizedFusedMoEMethod` (BF16), `CompressedTensorsWNA16`
(RAWINT4) and `DeepSeekMxfp4MoEMethod`; ModelOpt NVFP4 / AWQ / GPTQ
checkpoints have no wrapper, so capsim refuses them (`kt_method_for`).

**(c) Architectures.** The KT path is a wrapper on SGLang's shared
FusedMoE layer, not per-model code, so coverage is "whatever the fork
loads with one of the quant methods above". Upstream's own tables list
DeepSeek-V3/R1/V3.2/V4-Flash, Kimi-K2 / K2-Thinking / K2.5 (K2.6 via the
same path), Qwen3-30B-A3B / 235B-A22B / Next-80B / 3.5 / Coder-Next,
GLM-4.7 / 5 / 5.1 / 5.2 / 5.3-Flash, MiniMax-M2 / M2.1 / M2.5 / M3
(tutorials under `doc/en/kt-kernel/`). DeepSeek-V3.2's sparse-attention
indexer — which the v0.3 image had no rule for — is served here
(`--attention-backend triton` in that tutorial on an L20; flashinfer on
Blackwell per the V4-Flash tutorial).

**(d) Image for Blackwell + AMX.** Docker Hub
[approachingai/ktransformers](https://hub.docker.com/r/approachingai/ktransformers/tags)
has NO v0.6.x/v0.7.x tags: after `v0.5.3` (2026-04-02, 15.6 GB, the
conda/LLaMA-Factory SFT image with ktransformers 0.5.3 on SGLang 0.5.6)
the only newer builds are `DSV4-specific` (2026-09-01, 12.0 GB
compressed), `DSV4-specific-6b8a81c`, `DSV4-specific-a3803fd`,
`dsv4-specific-avx2-tp-77e5bf0` and `dsv4-api-codex-20260909-final`
(2026-09-09, 12.1 GB). No `approachingai/kt-kernel` repository exists.
The registry config of `DSV4-specific` (read without pulling) and the
`dsv4` stage of
[docker/Dockerfile](https://github.com/kvcache-ai/ktransformers/blob/v0.7.1/docker/Dockerfile)
pin: ktransformers / kt-kernel / sglang-kt **0.7.0.post1**, torch
2.9.1+cu128, CUDA 12.8.1, cuDNN 9.8, flashinfer 0.6.15.post1,
transformers 4.57.1, tilelang 0.1.10, Python 3.11 venv at `/opt/venv`,
`CPUINFER_BUILD_ALL_VARIANTS=1` (AMX/AVX512/AVX2 chosen at runtime) and
`CPUINFER_CUDA_ARCHS=80;86;89;90;100;120`. The
[DeepSeek-V4-Flash tutorial](https://github.com/kvcache-ai/ktransformers/blob/v0.7.1/doc/en/DeepSeek-V4-Flash.md)
validates it on an RTX 5090 (SM_120). By contrast the `kt-kernel` PyPI
wheel is built for SM 80/86/89/90 only (kt-kernel/README.md § CUDA
Installation), so the image, not pip, is the Blackwell route.

The image's `ENTRYPOINT` is `/usr/local/bin/entrypoint-dsv4`; it
`exec "$@"` when the first argument does not start with `--`, otherwise
assembles a V4-Flash launch from `TP`, `MEM_FRACTION`,
`CONTEXT_LENGTH`, `MAX_RUNNING_REQUESTS`, `KT_GPU_EXPERTS` (default 0),
`KT_CPUINFER_THREADS` (default physical cores − 4) and
`KT_THREADPOOL_COUNT` (NUMA nodes), runs under `numactl --interleave=all`
and expects `--ipc host --cap-add SYS_NICE`. It derives
`TORCH_CUDA_ARCH_LIST=12.0+PTX` / `FLASHINFER_CUDA_ARCH_LIST=12.0a` for
SM120; capsim passes its own CMD and therefore sets those two variables
itself (`ktransformers_cuda_arch`, default `12.0`). The HEALTHCHECK
polls `/health` with a 15-minute start period.

**(e) Readiness and metrics.** SGLang's: `/health` answers once the
scheduler holds the model (the tutorials verify with `/v1/models`);
`/metrics` exists only with `--enable-metrics` and carries the
`sglang:` counters (`prompt_tokens_total`, `generation_tokens_total`,
`num_running_reqs`, `num_queue_reqs`, `token_usage`,
`num_retracted_reqs`) that `engines/base.py` already maps for
`sglang_cuda`. The v0.3 server exposed no metrics at all.

## Staging expectations on the box

* **Native formats (the common case).** Stage the *full* HF checkpoint
  — config, tokenizer and safetensors — in the HF cache. capsim's
  `kt_only` catalog entries are staged config-only today (the v0.3
  convention); for the v0.7 line the safetensors are the CPU weights.
  Kimi-K2-Thinking / K2.5: ~600 GB INT4 (`RAWINT4`, ≥600 GB RAM per
  upstream). Kimi-K2-Instruct-0905, DeepSeek-V3.1/V3.2: ~650–690 GB
  block-FP8 (`FP8`). RAM budget: the routed experts, i.e. roughly the
  checkpoint size minus what `--kt-num-gpu-experts` moves to VRAM.
* **AMX INT8/INT4 (optional, explicit).** Run
  `kt-kernel/scripts/convert_cpu_weights.py` inside the image
  (`docker run --rm -v …:/model -v …:/out approachingai/ktransformers:DSV4-specific python /workspace/ktransformers/kt-kernel/scripts/convert_cpu_weights.py --input-path /model --input-type fp8 --output /out --quant-method int8 --cpuinfer-threads 170 --threadpool-count 2 --no-merge-safetensor`),
  then set `ktransformers_kt_method: AMXINT8` and
  `ktransformers_amx_weight_path`. Upstream publishes no timing; the
  converter is a single-process CPU pass over every expert tensor, so
  expect hours for a 1T model and RAM for one layer at a time with
  `--no-merge-safetensor`. capsim never runs it; a missing directory
  is refused before docker.
* **GGUF (LLAMAFILE).** Set `ktransformers_kt_method: LLAMAFILE` with
  `ktransformers_gguf_path`; the HF dir must still hold the GPU-side
  weights (attention, dense layers), so config-only staging is not
  enough here either.

## Defaults capsim applies

* `--kt-cpuinfer`: physical cores − 2 (`default_cpu_infer`), read from
  `/proc/cpuinfo` on the launching host; upstream says physical cores,
  never hyperthreads. The image's own entrypoint uses cores − 4.
* `--kt-threadpool-count`: NUMA node count from
  `/sys/devices/system/node`; the fork's default is 2.
* `--kt-num-gpu-experts`: `ktransformers_gpu_experts`, default 0 (a
  searchable lever; upstream's Kimi table steps 0/8/30/80 per layer
  from one to eight 48 GB cards).
* `--mem-fraction-static`: translated from `gpu_memory_utilization` the
  way `sglang_cuda` does it (`engines/vram.py`).
* `--watchdog-timeout 3000`: a 1T load plus a lazy first layerwise
  prefill outruns SGLang's 300 s default.
* `--max-running-requests` ← `max_num_seqs`; `--chunked-prefill-size`
  and `--max-prefill-tokens` ← `max_num_batched_tokens`;
  `--context-length` ← `max_model_len`.

## First run on the box

1. `docker pull approachingai/ktransformers:DSV4-specific` — 12.0 GB
   compressed, roughly 25–30 GB extracted (the layers are mostly the
   CUDA 12.8 devel base plus a torch 2.9.1 venv).
2. Stage a full native checkpoint (Kimi-K2-Thinking is the smallest 1T
   at ~600 GB; DeepSeek-V3.2 FP8 ~690 GB).
3. `ktransformers_generation` may stay `auto`; set
   `ktransformers_gpu_experts` (start at 8 on a 4×96 GB group) and
   optionally `ktransformers_gpu_prefill_threshold: 2048` with
   `ktransformers_dynamic_expert_update: true` for long prompts.
4. Expect 2–5 minutes to first `/health` (upstream: 2–3 min for
   Kimi-K2-Thinking, 4–5 min for V4-Flash with CUDA-graph capture).
