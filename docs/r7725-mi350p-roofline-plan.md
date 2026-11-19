# R7725 / MI350P: roofline on leading-edge AMD software

2026-11-15, revision 5. Box: Dell R7725, 1x AMD Instinct MI350P (PCIe).

**Goal:** the best performance this pre-release card can show, on the
newest and most capable models that fit it, using the newest AMD software
and the new techniques AMD is launching for it. Pre-release builds and
open pull requests are in scope; every result names exactly what it ran.

Revision 1 tuned settings on the software the box already had. Revision 2
found that software was a month behind AMD's MI350P work and added a stack
refresh. Revision 3 added AMD's unreleased techniques (section 3).
Revision 4 restricted the models to recent ones and added a search for
the best shape and serving settings (Phase 1b). Revision 5 sets the
model window to all of 2026.

**Which models.** Only models released in 2026 with a mixture-of-experts
design or a newer attention mechanism, and room for higher performance.
Llama-3.3-70B and gpt-oss (both 2025 or earlier) are out, as are
Kimi-Linear, Qwen3-30B-A3B, Qwen3-Next-80B and GLM-4.5-Air.

Candidates from two searches of Hugging Face on 2026-11-15 (some
checkpoint that fits one 144 GB card and is not NVIDIA-only NVFP4).
Each is smoke-tested on both engines first; a model that does not start
is recorded as an engine gap and dropped. Release dates, designs and
pull-request numbers come from the search and are checked only where a
result depends on them.

Released since mid-May:

| Model (checkpoint) | Released | GB | Design | Risk |
|---|---|---|---|---|
| Qwen3.8-27B (`amd/...-Quark-AWQ-MXFP4`, `Qwen/...-FP8`) | 08-05 | 20 / 31 | Dense, 48 linear-attention + 16 full layers, MTP head | Low: AMD's tuned model |
| Qwen3.8-Flash-Next (`amd/...-Quark-TokenPLE-AWQ-UINT4`) | 08-24 | 80 | MoE 512 experts / 10 active, linear + full attention, MTP | vLLM support is an open PR (#64369) |
| Muse-Glimmer-30B (`amd/...-w8a8-llmcompressor`) | 08-09 | 34 | Dense, 39 sliding-window + 13 full layers | AITER has gfx950 Muse work (#6507) |
| North-Mini-Code-1.0 (`CohereLabs/...-fp8`) | 06-05 | 32 | MoE 128 / 8, sliding-window + global attention | vLLM support merged (#62101) |
| Nemotron-3.5-Lightning-30B-A3B (`RedHatAI/...-FP8`) | 08-01 | 32 | Mamba-2 + MoE 128 / 6 + attention | AITER Mamba-2 kernel is an open issue |
| Laguna-XS-2.1 (`poolside/...-FP8`) | 06-20 | 35 | MoE 256 / 8, sliding-window + global attention | vLLM support is an open PR (#52800) |
| Raon-1-mini-preview (`KRAFTON/...`) | 10-21 | 33 | MoE 128 / 6, delta attention + gated MLA | vLLM support is an open PR (#64672) |
| Instella-MoE-16B-A3B (`amd/...-SFT`) | 07-23 | 32 | AMD's own MoE 64 / 6 with gated MLA | No engine support found |
| Arete-27B (`NousResearch/...`) | 11-06 | 55 | Qwen3.8-27B fine-tune, BF16 | Low |

Released January to mid-May (all already on the box except LFM2):

| Model (checkpoint) | Released | GB | Design | Risk |
|---|---|---|---|---|
| Step-3.5-Flash (`amd/...-MXFP4`) | 02-01 | 114 | MoE, sliding-window + full attention, MTP | The largest that fits; AITER's Step-3.5 MoE work is still open |
| Gemma-4-31B (`amd/...-MXFP4`) | 03-11 | 20 | Dense, 50 sliding-window + 10 full layers | Low |
| Gemma-4-26B-A4B (`RedHatAI/...-FP8-dynamic`) | 03-11 | 29 | MoE 128 experts, sliding-window + full | Failed on SGLang in the baseline; MI350P MoE tables are open PRs |
| Qwen3.5-35B-A3B (`amd/...-MXFP4`) | 02-24 | 25 | MoE 256 / 8, linear + full attention, MTP | Low |
| Qwen3.6-35B-A3B (`Qwen/...-FP8`) | 04-15 | 38 | Same design as Qwen3.5-35B-A3B | Ran in the baseline |
| Qwen3.6-27B (`Qwen/...-FP8`) | 04-21 | 31 | Dense, linear + full attention | Ran in the baseline |
| Qwen3-Coder-Next (`amd/...-MXFP4`) | 01-30 | 45 | MoE 512 / 10, linear + full attention | Low |
| GLM-4.7-Flash (`zai-org/...`, BF16) | 01-19 | 62 | MoE 64 / 4, MLA | Ran in the baseline |
| Nemotron-3-Super-120B-A12B (`nvidia/...-FP8`) | 03-10 | 128 | Mamba-2 + MoE 512 / 22 + attention | Little room for KV; failed on SGLang in the baseline |
| LFM2-24B-A2B (`LiquidAI/...`, BF16) | 02-24 | 48 | MoE 64 / 4, convolution + attention | No AMD build |

Qualifying but not runnable here:
- NVFP4 only under the size limit, which ROCm cannot load: Ripple-65B,
  Laguna-S-2.1, Nemotron-3.5-Super-VL, Qwen3.5-122B-A10B (its FP8 is
  127 GB and failed in the baseline), Mistral-Small-4-119B.
- Too large for one card: MiniMax-M2.5 and M2.7 MXFP4 (127 GB each),
  Step-3.7-Flash (129 GB), DeepSeek-V4-Flash (159 GB), GLM-5.3-Flash
  (185 GB), DeepSeek-V4.1-Flash, MiniMax-M3.1-Flash, Inkling-2-Small,
  Kimi-K3, GLM-5 and 5.1.

## 1. Review of revision 1 against the goal

| Gap | Finding (checked 2026-11-15) | Change |
|---|---|---|
| The kernel library is a month old | Both images carry AITER from 2026-10-17/19 (`amd-aiter 0.1.26`). AITER learned to identify the MI350P on 10-27 (#5182) and has added 128-CU tuned tables since: Qwen3.8-27B FP8 (#6939) and MXFP4 (#7075), Gemma-4-31B FP8-block (#7254), and 825 rows for gpt-oss, Qwen3-235B, DeepSeek-V4-Flash, GLM-5.2, MiniMax-M3 and Kimi-K2 (#7218, merged 11-13). Main is 640 commits ahead of what we ran. | New Phase 0: refresh the stack before any A/B. |
| Phase 2 planned to hand-tune what AMD has since tuned | The "no 128-CU rows" finding was true of our images, not of AITER. | Phase 2 shrinks to the shapes upstream still lacks. |
| Phase 1 would have A/B-tested the old stack | Its winners would not carry over to a stack with different kernels and tables. | Phase 1 runs on the Phase 0 stack; the old stack is one reference arm. |
| MXFP4 was last | MXFP4 is the card's headline precision (4.6 PF vs 2.3 PF FP8), AMD publishes the checkpoints, and AMD's tuning effort is on it. Revision 1 had one MXFP4 model in Phase 1 and deferred the rest to Phase 4. | MXFP4 leads every phase. |
| The test models were the NVIDIA list | Phase 1 used Llama-3.3-70B FP8 and Qwen3-30B FP8, which AMD has not tuned for this card. | Phase 1 uses the models AMD has tuned for 128 CUs, plus one untuned control. |
| AMD's own settings were only partly used | AMD's vLLM guide documents switches revision 1 missed: `VLLM_ROCM_SHUFFLE_KV_CACHE_LAYOUT=1` (the high-concurrency decode kernel for `ROCM_AITER_FA`) and `VLLM_ROCM_USE_AITER_FP4_ASM_GEMM` (MXFP4). | Added to the Phase 1 arms. |

What revision 1 got right and keeps: the first run is an untuned
baseline; vLLM ran with AITER off; graphs stopped at batch 512; the card
is power-bound at 450 W; the cross-check guard.

## 2. The software stack: what we ran and what is current

| Layer | First run | Newest available | Plan |
|---|---|---|---|
| vLLM | 0.33.0 (`vllm/vllm-openai-rocm:v0.33.0-rocm100`) | 0.34.0rc4; daily `nightly-rocm100` builds, which pin AITER v0.1.29 | Today's nightly, pinned by commit |
| SGLang | 0.5.23 (`lmsysorg/sglang:v0.5.23-rocm10-mi35x`, built 10-30) | 0.5.23 is the latest release; AMD's daily `rocm/sgl-dev:v0.5.23-rocm10-mi35x-<date>` | Today's daily |
| AITER | 0.1.26 (10-19) in vLLM; commit `285579b6` (10-17) in SGLang | v0.1.29 (11-09); main has the 11-11 and 11-13 MI350P tables. SGLang's Dockerfile still pins `285579b6` even in the daily. | Rebuild AITER at one pinned main commit (at or after #7218) in both images |
| ROCm in the containers | 10.0.0 | 10.1.0 (10-05), but every MI35x engine build is still on 10.0; the 10.1 engine builds are for MI45x, and AMD's `rocm/vllm` 10.1 image carries vLLM 0.29 | Stay on 10.0 |
| Host driver | amdgpu 7.1.3 (31.50) | 31.60 is the newest listed for the MI350 series; 31.50 is supported | No change. Optional, sudo (Phase 5) |
| Host ROCm | 10.0.0 | 10.1.0. Its notes list nothing for MI350P inference (virtualisation only) | No change |
| Card firmware | IFWI build 2026-06-12 | AMD lists PLDM bundle 01.26.01.03 or later; the mapping to our IFWI is not stated | Ask Dell (Phase 5) |

Nightly and main builds are pre-release. Every result records the image
digest, vLLM or SGLang commit and AITER commit, and the summary states
them. When vLLM 0.34 and the next AITER release ship, re-run the Phase 4a
headline cells on the released builds.

## 3. New AMD techniques for this card

Checked on GitHub and Hugging Face, 2026-11-15.

| Technique | What it is | State | How we get it |
|---|---|---|---|
| 128-CU tuned tables | GEMM and fused-MoE kernel choices tuned for the MI350P | Merged in AITER main | Phase 0 AITER rebuild |
| FlyDSL kernels | AITER's new kernel language; the newest MXFP4 GEMM and MoE kernels are written in it | Merged in AITER main | Phase 0 AITER rebuild |
| Mono decode | One persistent kernel per decoder layer, one thread group resident on each CU, in place of about 21 small launches. For Qwen3.8-27B MXFP4. AMD claims +12 to +19 % at 1 to 4 requests on an MI350P | vLLM #61573 (open) and #65078 (draft, the 128-CU version). `VLLM_ROCM_MONO_DECODE=1` | Build from the #65078 branch |
| MXFP4 activation-quant fusion | Activation, multiply and MXFP4 quantization in one kernel ahead of the GEMM; closes gap #60581 | vLLM #61902 (open). `VLLM_ROCM_USE_AITER_FP4_ACT_QUANT_FUSION=1`, gfx950 only | Build from the #61902 branch |
| DFlash2 speculative decoding | AMD's draft model `amd/Qwen3.8-27B-Quark-DFlash2` (published 11-10): 7 draft tokens a step, 2.31x on Spec-Bench | Checkpoint is public; vLLM has DFlash, DFlash2 support in the nightly is unverified | Smoke-test on the nightly; else find the branch |
| MTP decode for hybrids | Fused multi-token-prediction kernel for Qwen3.5/3.8 | Merged (vLLM #61855) | In the nightly |
| Unified attention rework | New masking for gpt-oss's windowed attention on gfx950 | Merged in AITER main (#7683); 128-CU split values still open (#7749) | Phase 0 AITER rebuild |
| Preshuffled FP8 block-scale GEMM | Faster FP8 GEMM, reverted once for an accuracy bug, now being re-landed | vLLM #64142 (draft) | Only if FP8 matters after Phase 1 |
| Gemma-4-26B-A4B MoE table | MI350P fused-MoE tiles, prefill only (+5 %) | vLLM #65134 (draft) | Config file only; add for Phase 4 |

Mono decode and speculative decoding raise speed per request at low
concurrency. They do not raise the high-batch peak, and mono decode has a
row cap. So the results get a second axis beside the roofline peak:
**interactive speed** (tokens per second per stream at 1 to 32 streams,
8k in / 1k out, the shape AMD quotes).

Not usable on one card: AMD's mono decode for DeepSeek-V4.1, Kimi-K3 and
Muse Spark, the PCIe ring all-reduce (AITER #5461) and MoRI are all
multi-GPU. `amd/Step-3.5-Flash_gpuep_1.0_gpu_128` (11-13) is an ONNX
Runtime build, not a vLLM or SGLang checkpoint.

## 4. The card

| | MI350P (this box) | MI355X (what AMD tuned first) |
|---|---|---|
| XCDs / CUs | 4 / 128 | 8 / 256 |
| HBM3E | 144 GB, 4 TB/s | 288 GB, 8 TB/s |
| Dense BF16 / FP8 / MXFP4 | 1.15 / 2.3 / 4.6 PF | 2.5 / 5.0 / 10 PF |
| Power | 600 W board, 450 W in the R7725 | 1,400 W |
| Partitions | SPX, DPX, CPX; NPS1 only | SPX, DPX, QPX, CPX; NPS1/2 |

- Decode at our shapes is memory-bandwidth bound (StorageReview measured
  3.29 TB/s at 450 W). MXFP4 halves weight bytes versus FP8, so it is the
  biggest single lever for weight-heavy models.
- 450 W is the binding limit at high batch: the GFX clock runs about
  1.4 GHz of 2.2 GHz. Tokens per watt is reported beside tokens per second.

## 5. The plan

Each phase ends with a gate. Prefix caching stays on throughout.

### Phase 0: refresh the stack (about 5 hours, about 1 GPU-hour)

1. Pulls and builds run beside the baseline's confirm phase (no GPU).
   When the images are ready, stop the baseline if it is still confirming
   and archive it as "untuned ROCm baseline (vLLM 0.33 / SGLang 0.5.23 /
   AITER 0.1.26)". Its 140 settled cells are kept; the remaining
   confirmations can be resumed later.
2. Pull the newest published builds:
   `vllm/vllm-openai-rocm:nightly-rocm100-<commit>` and
   `rocm/sgl-dev:v0.5.23-rocm10-mi35x-20261115`.
3. Build one derived image per engine with AITER rebuilt at a single
   pinned main commit. SGLang's Dockerfile takes `AITER_COMMIT`; vLLM's
   base Dockerfile takes `AITER_BRANCH`. Same commit in both.
4. Smoke each of the four images on gpt-oss-120b and Qwen3.8-27B MXFP4:
   it starts, the text is sane, one 64-stream rung runs, and the count of
   `not found tuned config` lines is recorded.
5. Build the technique images on top of the vLLM AITER-main image: one
   from the #65078 branch (mono decode), one from the #61902 branch
   (MXFP4 fusion). Both are drafts and #65078 does not merge cleanly, so
   each is its own image. Two hours each at most; a branch that will not
   build or start is recorded as "not ready" and dropped.
6. Smoke DFlash2 on the nightly with AMD's draft checkpoint.
7. Add an image override and the stack versions to capsim's roofline cell,
   so each cell names the image it ran.

Gate, per engine: use the AITER-main build if it passes the smoke test,
else the newest published build, else the current image. If neither new
build beats the current image's tuned-table hit rate, stop and report.

### Phase 1: settings and techniques on the new stack (about 9 GPU-hours)

One cell each (128 in / 256 out). Models:

- Qwen3.8-27B MXFP4 (128-CU tables exist; a hybrid, so both engines run
  it at a cap of 1024: vLLM refuses 2048 for it)
- Qwen3.8-27B FP8: the same model, for the FP8 to MXFP4 step
- Gemma-4-31B MXFP4 (dense, no 128-CU tables: the untuned control) and
  Step-3.5-Flash MXFP4 (the largest that fits), at a cap of 2048

The rows below that name gpt-oss apply only if a model with windowed
attention is in the final list.

| Engine | Arm | Settings |
|---|---|---|
| vLLM | S0 | Old image, defaults: the untuned reference, re-measured |
| vLLM | S2 | New image, `VLLM_ROCM_USE_AITER=1`. Dense: `--attention-backend ROCM_AITER_FA` with `VLLM_ROCM_SHUFFLE_KV_CACHE_LAYOUT=1`. gpt-oss: `ROCM_AITER_UNIFIED_ATTN --block-size 64` |
| vLLM | S3 | S2 + graph capture up to `max_num_seqs` |
| vLLM | S4 | S3 + `VLLM_ROCM_USE_AITER_FP4_ASM_GEMM=1`, the two dense MXFP4 models only. AMD says it wins at small batch, so expect no gain at 2048; one cell each settles it |
| SGLang | S0 | Old image, defaults |
| SGLang | S1 | New image, defaults (AITER is already on) |
| SGLang | S2 | S1 + `--cuda-graph-max-bs-decode 2048` |
| SGLang | gpt-oss | AITER attention first: #7683 reworked the path that faulted. `--attention-backend triton` only if it still faults |

Technique arms, Qwen3.8-27B MXFP4 on vLLM, each against S3:

| Arm | Settings | Measured on |
|---|---|---|
| T1 | Mono decode image, `VLLM_ROCM_MONO_DECODE=1` | Interactive ladder; one roofline cell to confirm the peak is unchanged |
| T2 | MXFP4 fusion image, fusion flag on | Roofline cell and interactive ladder |
| T3 | DFlash2 draft, 7 speculative tokens | Interactive ladder |
| T4 | MTP, 3 draft tokens | Interactive ladder |
| T5 | The best of T1 to T4 combined, if the builds allow | Interactive ladder |

The interactive ladder is 1, 2, 4, 8, 16 and 32 streams at 8k in / 1k
out, about 20 minutes an arm.

About 27 cells plus 6 technique runs, about 9 GPU-hours. Rules:

- An arm counts only if its log shows the requested backend and no
  fallback.
- AMD's guide says the FP4 kernels need an MI350X or MI355X. If they
  refuse the 128-CU card, record it as an upstream gap, not a result.
- Keep a setting that gains at least 5 % on rungs that pass the
  cross-check.

- A technique result is reported with its output checked for sanity
  (same prompts, readable answers); speculative arms report acceptance
  length.

Gate: carry the winning settings and techniques forward. The
gap between the tuned models and the untuned control decides how much
Phase 2 is worth.

### Phase 1b: best shape and serving settings (about 6 GPU-hours)

Phase 1 holds the shape fixed (128 in / 256 out) to compare kernels. The
headline number should come from the shape and serving settings that
suit the card, so this step searches them on the Phase 1 winners.

Already in place: every saturation request forces the full output length
(`ignore_eos`, or `min_tokens` where the model's parser needs it), so no
stream ends early.

| Lever | Values | Why |
|---|---|---|
| Output length | 256, 512, 1024 | At 128 in / 256 out the engine prefills one token for every two it generates. Longer outputs cut that share, so more of each step is decode |
| Input length | 32, 128 | Less prefill and less KV per stream. 32 is reported separately as a best-case shape, not as the standard one |
| Engine cap | 1024, 2048, 4096 | The reference runs peaked at 512 streams and fell at 2048; the best cap is not known yet |
| Prefill budget | vLLM `--max-num-batched-tokens` 8192 (default), 32768, 65536; SGLang `--max-prefill-tokens` likewise | AMD's guide recommends 32k or more for throughput; the default starves prefill at 2048 streams |
| Scheduling and front end | vLLM `--async-scheduling`; SGLang `--stream-interval` | At 10k+ tokens a second the API process and per-token streaming can be the limit |

Shapes first (6 per model), then the serving levers one at a time on the
best shape. Results report two numbers per model: the standard shape
(128 / 256, comparable with the XE7740) and the best shape found, each
with its shape stated.

### Phase 2: tune only what upstream has not (up to 8 GPU-hours)

1. From the Phase 1 logs and one smoke start per remaining model, list
   the shapes that still miss a tuned row, for the models in the final
   list only.
2. Tune those with AITER's in-tree tuners, MXFP4 (a4w4, fp4 MoE) first.
   Known traps: the a4w4 tuner faults on N=96 and some CK-tuned rows are
   unusable on 128 CUs (AITER #6255); unaligned rows are being removed
   upstream (#7944).
3. Keep the CSVs in `/data/capsim/aiter-tuned/gfx950-128cu/<aiter-commit>/`
   and load them through the `AITER_CONFIG_*` variables.

Gate: keep a table that gains at least 5 % or fixes a startup failure.
Stop at 8 GPU-hours whatever is left; untuned models are labelled so.

### Phase 3: make it capsim's AMD default (code, tests)

- An AMD profile per engine, applied when `gpu_vendor.is_amd()`: the
  Phase 0 images, the Phase 1 winners, tuned-table mounts when present.
- Each cell records the image digest, engine and AITER commits, and the
  attention, MoE and GEMM backends parsed from the engine log.
- Hybrid models: clamp `max_num_seqs` to the Mamba block count instead of
  failing the cell.
- Move the cross-check into capsim's summary. Tests for each.

### Phase 4: the AMD roofline, headline first

Grid as in the baseline (caps 1024 and 2048, outputs 128 and 256, input
128), so results pair with the baseline and with the XE7740.

| Part | Models | Cells | GPU-hours |
|---|---|---|---|
| 4a Headline | Qwen3.8-27B in MXFP4 and FP8 (AMD's showcase model for this card: tuned tables, mono decode, DFlash2, MTP) and Arete-27B. Both engines, the roofline grid, the best shape from Phase 1b, and the interactive ladder with the Phase 1 techniques | about 30 | about 9 |
| 4b Other 2026 models | Every candidate in the two tables at the top that passes its smoke test, AMD's MXFP4 checkpoints first | up to 140 | up to 33 |
| 4c Baseline grid | Optional: the baseline's remaining FP8 and BF16 models on the new stack | about 100 | about 23 |

4a and 4b are the deliverable. 4c runs only if you ask for it.

The newest large models do not fit one 144 GB card: DeepSeek-V4.1-Flash
(510 GB), GLM-5.3-Flash (657 GB), Qwen3.8-Flash-Next (182 GB in MXFP4),
MiniMax-M3.1-Flash (240 GB), Muse Spark Open, Kimi-K3.1. AMD shows those
on four to eight cards, and has open MI350P work for exactly that
(vLLM #67580 serves Muse Spark Open on 4x MI350P). They need a
multi-card box.

Summary pairs: the same model in FP8 and MXFP4, on both engines, against
the untuned baseline, with tokens per watt.

### Phase 5: needs your decision or sudo (not scheduled)

| Option | Expected effect | Cost and risk |
|---|---|---|
| 600 W power cap | The card is pinned at 450 W; AMD reportedly expects about +10 % at 600 W (secondhand) | **Check with Dell first** that the R7725 supports 600 W for this card. Report it as a separate variant |
| Card firmware and driver 31.60 | Unknown; none documented for inference | Ask Dell for the current MI350P firmware bundle; driver update needs sudo and a reboot |
| CPX partitions (4 x 32 CUs, about 36 GB each) | Four replicas for models under about 30 GB (gpt-oss-20b, Qwen3.8-27B MXFP4, Gemma-4-31B MXFP4) | Driver reconfiguration with sudo; AITER has no 32-CU gfx950 tables |
| Host kernel arguments (`iommu=pt`, `pci=realloc=off`, C-states) | Small at one GPU | Reboot with sudo |

### Partition test (queued 2026-11-18, waits for the mode switch)

Goal: "one card, four isolated servers". Show that four engines, one per
CPX partition, hold the unpartitioned card's total throughput, per model
as a percentage, including the cases where partitions are slower.

What the driver reports today (read without root):

- Modes offered: SPX, DPX, CPX. Current: SPX. Memory mode: NPS1 only.
- `compute_partition_mem_alloc_mode` is `CAPPING`, so each partition is
  expected to be limited to its share of memory (about 36 GB in CPX,
  about 72 GB in DPX). `ALL` is the other setting. The cap is read from
  the setting's name and has to be confirmed after the switch.
- No memory-mode change is involved (only NPS1 exists), so no driver
  reload is expected. The card has to be idle when the mode changes.

Commands for the box owner (root; not run by the test session):

```bash
sudo amd-smi partition --accelerator          # read-only: profiles and resources
sudo amd-smi set -g 0 --accelerator-partition CPX
sudo amd-smi set -g 0 --accelerator-partition DPX
sudo amd-smi set -g 0 --accelerator-partition SPX   # back to the whole card
```

Models (all staged on the box) and expected fit:

| Model | On disk | CPX (about 36 GB) | Whole-card reference, output tok/s |
|---|---|---|---|
| LFM2.5-24B-A2B FP8 | 24 GB | fits | 40,732 (two engines, short reading) |
| Qwen3.5-35B-A3B MXFP4 | 23 GB | fits | 15,966 (short reading; 15,046 outside tool) |
| Qwen3.8-27B Quark-AWQ-MXFP4 | 19 GB | fits; stream count limited by per-stream state | 5,710 |
| Gemma-4-26B-A4B FP8 | 27 GB | marginal; DPX if it does not hold enough streams | 17,433 (short reading; 16,765 outside tool) |
| LFM2-24B-A2B bf16 | 45 GB | does not fit; DPX only | 42,081 (10-minute mean, two engines) |

Steps, 128 in / 256 out, same serving settings as the whole-card runs:

1. After each switch, `/data/capsim/partition/inspect.sh` records what
   amd-smi, sysfs and PyTorch in a container see (devices, memory per
   device), which decides whether the launcher needs a change to treat
   partitions as devices.
2. A fresh whole-card reference per model as a 10-minute hold, so both
   sides of each percentage use the same method (done 2026-11-18, table
   below).
3. CPX: each model alone, one engine per partition, all four under load;
   10-minute hold. Record total and per-partition tok/s, power, streams
   in flight, and whether AITER reports missing tables for 32 CUs.
4. DPX: LFM2 bf16 and any model that did not fit or ran badly in CPX,
   two engines.
5. Mixed: four different models, one per CPX partition, all under load;
   per-model tok/s. One capsim run serves one model, so this step drives
   each engine with `vllm bench serve` and compares with that tool's
   whole-card figure.

Open: Gemma in DPX or left out of the four-way picture; whether the mode
survives a reboot.

Whole-card references (SPX), 10-minute holds, 128 in / 256 out, 0 errors
in every run:

| Model | Engines | Streams | Output tok/s | GPU W | ms/token p50 | TTFT p95 | Earlier short reading | Run |
|---|---|---|---|---|---|---|---|---|
| LFM2-24B-A2B bf16 | 2 | 8,192 | 42,081 | 447 | 191 | 4.2–4.4 s | 45,519 | run_415–417 (mean of three) |
| LFM2.5-24B-A2B FP8 | 2 | 8,192 | 40,855 | 446 | 197 | 5.6 s | 40,732 | run_438 |
| Gemma-4-26B-A4B FP8 | 1 | 1,536 | 16,724 | 449 | 90 | 0.9 s | 17,433 | run_440 |
| Qwen3.5-35B-A3B MXFP4 | 1 | 2,048 | 15,114 | 446 | 134 | 1.7 s | 15,966 | run_439 |
| Qwen3.8-27B Quark-AWQ-MXFP4 | 1 | 1,024 | 4,389 | 448 | 225 | 7.9 s | 5,710 | run_441 |

- Qwen3.8-27B holds 4,389, 23% under its 15-second reading of 5,710. Its
  prompt processing is slow (TTFT p95 7.9 s), so a short window between
  prompt waves reads well above the sustained rate. Use 4,389.
- Gemma and Qwen3.5 land within 0.5% of their `vllm bench serve` figures
  (16,765 and 15,046).
- Four engines on the unpartitioned card (arms in
  `/data/capsim/partition/spx/skipped`) were not run. They would separate
  the effect of four engines from the effect of partitions; run the
  matching one only if a CPX result lands far from its reference.

### Partition and layout results (2026-11-18 to 19)

All rows are 10-minute holds at 128 in / 256 out with the card at its
450 W cap. Mode switches: `sudo amd-smi set -g 0 --accelerator-partition
SPX|DPX|CPX`, card idle, no driver reload.

What the card reports: CPX shows four devices of 36.0 GiB, DPX two of
72.0 GiB (memory mode `CAPPING`). Power and activity are reported on the
first partition only, for the whole card. In CPX the video decoder and
JPEG engines are shared between pairs of partitions.

Harness changes these runs needed (in `simulator/`, with tests):

- `gpu_vendor.py`: a partitioned card reports "N/A" blocks for every
  partition but the first and gives the first the whole card's memory;
  both are handled.
- `arrivals.py`: a new session goes to the engine with the fewest in
  flight. Taking turns piled 3,600 of 5,120 streams onto the slowest of
  four engines.
- `spread_s` on a hold: start the streams evenly over a period.

LFM2 layouts, output tok/s:

| Model | Mode | Engines | Streams offered / in flight | Output tok/s | TTFT p95 | Errors | Run |
|---|---|---|---|---|---|---|---|
| LFM2 bf16 | SPX | 2 | 8,192 / 8,060 | 42,081 | 4.2–4.4 s | about 0.1% stalled | run_415–417 |
| LFM2 bf16 | DPX | 1 per half | 8,192 / 8,069 | 42,831 | 4.2 s | 79 stalled | run_450 |
| LFM2 bf16 | DPX | 1 per half, step budget 32,768 | 8,192 / 7,936 | 41,018 | 5.2 s | 184 stalled | run_451 |
| LFM2 bf16 | DPX | 1 per half, step budget 16,384 | 12,288 / 9,179 | 36,129 | 32 s | 7,983 stalled | run_453 |
| LFM2 bf16 | DPX | 1 per half, step budget 16,384 | 16,384 / 9,973 | 35,803 | 61 s | 86,017 | run_453 |
| LFM2 bf16 | SPX | 2, step budget 6,144 | 8,192 / 8,024 | 42,184 | 5.0 s | 139 stalled | run_456 |
| LFM2 bf16 | SPX | 2, starts spread over 50 s | 8,192 / 8,128 | 42,278 | 0.86 s | 176 stalled | run_457 |
| LFM2 bf16 | DPX | 1 per half, starts spread over 50 s | 8,192 / 8,129 | 42,216 | 0.84 s | 201 stalled | run_459 |
| LFM2 bf16 | DPX | same, repeat | 8,192 / 8,127 | 42,182 | 0.84 s | 203 stalled | run_461 |
| LFM2 bf16 | DPX | same, session cache in bf16 (not FP8) | 8,192 / 6,016 | 34,535 | 22 s | 306 stalled | run_460 |
| LFM2 bf16 | SPX | 1, stream interval 32, 8 API servers | 4,096 / 4,049 | 27,745 (346 W) | 1.8 s | 0 | run_455 |
| LFM2.5 FP8 | SPX | 2 | 8,192 / 8,072 | 40,855 | 5.6 s | 0 | run_438 |
| LFM2.5 FP8 | SPX | 4, starts spread | 8,192 / 8,129 | 37,772 | 1.0 s | 0 | run_458 |
| LFM2.5 FP8 | SPX | 4 | 5,120 / 5,006 | 35,285 | 2.8 s | 0 | run_454 |
| LFM2.5 FP8 | DPX | 2 per half | 5,120 / 4,690 | 33,863 | 5.7 s | 0 | run_452 |
| LFM2.5 FP8 | CPX | 1 per quarter, even load | 5,120 / 3,351 | 27,792 | 24 s | 0 | run_447 |
| LFM2.5 FP8 | CPX | 1 per quarter, uneven load | 5,120 / 2,742 | 27,231 | 82 s | 0 | run_446 |
| LFM2.5 FP8 | CPX | 1 per quarter, stream cap 2,048 | 8,192 / 1,829 | 21,680 | 149 s | 2 stalled | run_442 |

Other models in CPX (one engine per quarter):

| Model | Whole card | CPX | Share | Note |
|---|---|---|---|---|
| Qwen3.8-27B MXFP4 | 4,389 | 2,786 (run_448; 2,119 before the stream cap was sized, run_444) | 63% | about 63 streams fit per partition |
| Qwen3.5-35B-A3B MXFP4 | 15,114 | not measured | | did not start until graph capture was cut to 11 sizes (10.1 to 2.7 GiB); stopped before a result |
| Gemma-4-26B-A4B FP8 | 16,724 | did not start | | 27.5 GiB model copy leaves no session memory in 36 GiB |

Findings:

- Two-way partitioning costs nothing: one engine per half gave 42,831
  against 42,081 on the whole card, and 42,216 / 42,182 with spread
  starts against 42,278. The 42,831 was not repeated; treat the two
  modes as equal.
- Two engines is the best count for LFM2. Four engines hold four model
  copies and repeat the fixed per-step work: 37,772 on the whole card at
  the same 8,192 streams (92% of two engines), 33,863 as two per half,
  27,792 as one per quarter.
- A 36 GiB partition is too small for 19–29 GB models at high stream
  counts. The loss is model size against partition size, not the
  boundary. In CPX the engine also reported far fewer cache tokens per
  GiB than in DPX or SPX (213k from 5.3 GiB against 526k from 5.25 GiB);
  not explained.
- Partition 0 in CPX ran about 22% slower than the other three at equal
  load; not explained.
- Throughput for LFM2 sits at 41,000–42,900 across whole card, two
  halves, three step budgets and spread starts, at the power cap each
  time. More streams, a larger step budget and a single engine with
  lighter host work are all worse or no better.
- Spreading the stream starts keeps the rate (42,278) and cuts TTFT p95
  from 4.2 s to 0.86 s; the 10-second rate stays within 40,000–44,600.
  The 48,000 seen between waves was decode with no prompts to process.
- Leaving the session cache in bf16 halves the cache (1.65M tokens
  against 3.3M) and the card then holds 6,016 streams: 34,535.
- Recommended setting for LFM2 bf16 on one MI350P: two engines (whole
  card or one per half), 4,096 streams each, FP8 session cache,
  `--stream-interval 8 --api-server-count 4`, default step budget,
  starts spread: about 42,200 output tok/s with TTFT p95 under 0.9 s.
- The card was returned to SPX after the last run.

### The LFM2 mechanisms on the other models (2026-11-19)

Question: do the three things that worked for LFM2 (two engines sharing
the card, stream starts spread over one request's duration, stream count
sized to the memory two engines leave) raise throughput on the other
models? Whole card (SPX), 10-minute holds, 128 in / 256 out, 450 W cap,
0 errors in every run. Arms and results:
`/data/capsim/partition/spx4`; references are run_438 to run_441.

| Model | Engines | Streams | Starts | Output tok/s | Against reference | TTFT p50 / p95 | ms/token p50 | Run |
|---|---|---|---|---|---|---|---|---|
| LFM2.5-24B-A2B FP8 | 2 | 8,192 | together | 40,855 | reference | 1.16 / 5.6 s | 197 | run_438 |
| LFM2.5-24B-A2B FP8 | 2 | 8,192 | spread 50 s | 40,026 | -2.0% | 0.84 / 0.96 s | 202 | run_462 |
| Gemma-4-26B-A4B FP8 | 1 | 1,536 | together | 16,724 | reference | 0.53 / 0.87 s | 90 | run_440 |
| Gemma-4-26B-A4B FP8 | 1 | 1,536 | spread 23 s | 16,599 | -0.7% | 0.29 / 0.33 s | 92 | run_463 |
| Gemma-4-26B-A4B FP8 | 2 | 2 x 768 | spread 23 s | 15,699 | -6.1% | 0.30 / 0.35 s | 97 | run_464 |
| Qwen3.5-35B-A3B MXFP4 | 1 | 2,048 | together | 15,114 | reference | 0.65 / 1.68 s | 134 | run_439 |
| Qwen3.5-35B-A3B MXFP4 | 1 | 2,048 | spread 34 s | 14,400 | -4.7% | 0.43 / 0.49 s | 141 | run_465 |
| Qwen3.5-35B-A3B MXFP4 | 2 | 2 x 768 | spread 34 s | 14,121 | -6.6% | 0.34 / 0.50 s | 91 | run_466 |
| Qwen3.5-35B-A3B MXFP4 | 2 | 2 x 896 | spread 34 s | 14,179 | -6.2% | 0.38 / 0.54 s | 116 | run_470 |
| Qwen3.8-27B MXFP4 | 1 | 1,024 | together | 4,389 | reference | 2.19 / 7.9 s | 225 | run_441 |
| Qwen3.8-27B MXFP4 | 1 | 1,024 | spread 58 s | 4,501 | +2.6% | 0.75 / 0.91 s | 226 | run_467 |
| Qwen3.8-27B MXFP4 | 1, serving settings | 1,024 | spread 58 s | 4,475 | +2.0% | 0.71 / 0.89 s | 227 | run_468 |
| Qwen3.8-27B MXFP4 | 2, serving settings | 2 x 320 | spread 58 s | 4,498 | +2.5% | 0.49 / 0.73 s | 141 | run_469 |
| Qwen3.8-27B MXFP4 | 2, serving settings | 2 x 416 | spread 58 s | 4,400 | +0.3% | 0.59 / 1.00 s | 187 | run_471 |

"Serving settings" is `--stream-interval 8 --api-server-count 4`, which
the Qwen3.8 reference did not have. The two-engine Qwen runs cut graph
capture to about a dozen sizes (`--compilation-config`
`cudagraph_capture_sizes`) so each engine keeps 41 to 46 GiB for
sessions.

- No mechanism raised throughput on these models. Every run is within 7%
  of its reference; the only results above it are Qwen3.8's, by 0.3 to
  2.6%, which one run each cannot separate from run-to-run spread.
- Two engines helped LFM2 because one engine left the card at about
  357 W with its scheduling process at 100% of a host core. Gemma,
  Qwen3.5 and Qwen3.8 hold the card at 445 to 450 W with one engine, so
  a second engine has no idle GPU time to use and costs a second model
  copy: Gemma -6%, Qwen3.5 -6 to -7%.
- Spread starts cut first-token p95 on every model: LFM2.5 5.6 to
  0.96 s, Gemma 0.87 to 0.33 s, Qwen3.5 1.68 to 0.49 s, Qwen3.8 7.9 to
  0.91 s. The rate moves by -4.7% to +2.6%.
- Qwen3.8 with two engines at 640 streams gave the single engine's total
  (4,498 against 4,501 at 1,024 streams) with each stream 60% faster
  (141 against 226 ms per token). Raising it to 832 streams did not
  raise the total (4,400).
- Filling the memory two engines leave did not help: Qwen3.5 at 2 x 896
  (session memory 92% used) gave 14,179 against 14,121 at 2 x 768.
- The two Qwen3.5 engines did not share the card evenly: 8,439 and 5,682
  tok/s at 2 x 768, 7,765 and 6,414 at 2 x 896, with the same settings,
  memory and streams in flight. Gemma, Qwen3.8 and LFM2.5 split evenly.
  Not explained.
- Gemma and the two Qwen models get no prefix-cache hits at this shape.
  The engine's cache block is 128 tokens for Gemma and 832 for the Qwen
  hybrids, and the shared system prompt is 71 tokens, so no block is
  ever shared. LFM2's 56% reuse (worth about 12%) is not available to
  them at 128 in.
- Recommended setting per model: keep each reference configuration and
  add spread starts. Use two engines only for LFM2 and LFM2.5. For
  Qwen3.8, two engines at 320 streams each is the choice when per-stream
  speed matters.

### Why the other models trail LFM2 (2026-11-19)

Question: LFM2-24B-A2B gives 42,000 output tok/s; Qwen3.5-35B-A3B has a
similar active-parameter count and gives 15,000. What is the difference,
and can the other models be raised? Data: `/data/capsim/dig` (arms,
results, `membw.log`), profile `/data/capsim/huggingface/prof/dig_qwen35*`,
runs run_472 to run_479. All holds are 10 minutes, one engine, whole
card, 450 W cap, starts spread, 0 errors unless noted.

Work the card does per second, from the spread-start holds. "Processed"
is output tokens plus prompt tokens the GPU computed (cache hits left
out). Active parameters are estimates from each model's `config.json`.

| Model | Active per token | Output tok/s | Processed tok/s | Processed x 2 x active, TFLOP/s | Run |
|---|---|---|---|---|---|
| LFM2-24B-A2B bf16 (two engines) | about 2.0B | 42,278 | 52,500 | about 210 | run_457 |
| Gemma-4-26B-A4B FP8 | about 3.8B | 16,599 | 26,100 | about 200 | run_463 |
| Qwen3.5-35B-A3B MXFP4 | about 2.85B | 14,400 | 22,400 | about 130 | run_465 |
| Qwen3.8-27B MXFP4 (dense) | 27B | 4,501 | 7,700 | about 410 | run_467 |

- Gemma is explained by arithmetic: it does about the same work per
  second as LFM2, each token costs 1.9 times as much, and it gets no
  prompt reuse where LFM2 gets 56% (worth 14% on LFM2).
- Qwen3.5-35B-A3B is the outlier: about 60% of the work per second.

Qwen3.5-35B-A3B profile (vLLM torch profiler, 30 engine steps at 2,037
streams in flight; 133 ms of GPU kernel time per step against a 141 ms
step):

| Kernel group | Share of GPU time |
|---|---|
| Linear-attention layers (`qwen_gdn_attention_core`) | 47% |
| of which the recurrent-state update | 37% |
| of which the causal convolution | 7% |
| Dense matrix multiplies (`aten::mm`, rocBLAS) | 18% |
| Expert layers (AITER fused MoE, MXFP4) | 12% |
| Sampling (top-k/top-p, Gumbel, temperature) | 6% |
| Full attention (AITER paged attention) | 5% |
| State copies for the prefix cache | 2% |

- 30 of the 40 layers are linear-attention (gated delta rule) layers.
  Each keeps 32 heads x 128 x 128 values of state per stream: 1.05 MB
  in bf16 per layer, 31 MB per stream. Every generated token reads and
  rewrites all of it: about 63 MB of memory traffic per token. LFM2's 30
  convolution layers keep a few kilobytes, and its 10 attention layers
  read about 10 KB per token of context.
- At 2,037 streams one layer's update moves 4.3 GB (read plus write) in
  1.63 ms: about 2.6 TB/s.
- Memory probe on this card at 450 W (`membw.py`, PyTorch, 2.14 GB
  tensor): scale in place 3.05 TB/s, copy 2.63 TB/s, fill 3.3 TB/s. The
  state update runs at about 86% of the fastest read-and-write pass, so
  a better kernel has at most about 14% to give on 37% of the time.
- Memory-controller activity is the highest of the four models on
  Qwen3.5 (37%; LFM2 25%, Gemma 30%, Qwen3.8 27%).
- The experts, the part the "A3B" label counts, are 12% of GPU time. The
  model is bound by memory traffic from its state, not by active
  parameters, and no serving setting removes that. The state is constant
  per stream whatever the context length, so 128 in / 256 out is this
  design's worst case; it should compare better at long contexts (not
  measured here).

Lever runs:

| Model | Change | Output tok/s | Against base | Note | Run |
|---|---|---|---|---|---|
| Gemma-4-26B | base (spread starts) | 16,599 | | | run_463 |
| Gemma-4-26B | `--generation-config vllm` (no top-k 64 / top-p 0.95) | 17,059 | +2.8% | | run_473 |
| Gemma-4-26B | `--block-size 8` | did not start | | attention kernel: "block_size not supported" | run_474 |
| Gemma-4-26B | 256 in / 256 out, reuse on (47% hits) | 14,984 | | | run_478 |
| Gemma-4-26B | 256 in / 256 out, reuse off | 12,999 | | reuse is worth +15% at this shape | run_479 |
| Qwen3.5-35B-A3B | base (spread starts) | 14,400 | | | run_465 |
| Qwen3.5-35B-A3B | `--generation-config vllm` (no top-k 20 / top-p 0.95) | 14,566 | +1.2% | | run_475 |
| Qwen3.5-35B-A3B | `--no-enable-prefix-caching` | 14,648 | +1.7% | no memory freed | run_476 |
| Qwen3.5-35B-A3B | MTP, 1 draft token, bf16 convolution state | 15,109 | +4.9% | 1,235 of 2,048 streams fit (cache 99% full), rest queued, TTFT p95 15 s, rung not marked steady; 81 ms per token against 139 | run_477 |

- Sampling differs by model. capsim sends temperature 0.7 and the engine
  adds each model's shipped defaults: Gemma top-k 64 / top-p 0.95, Qwen
  top-k 20 / top-p 0.95, LFM2.5 top-k 50 and a repetition penalty, LFM2
  bf16 none. `--generation-config vllm` puts them all on plain
  temperature sampling: +2.8% on Gemma, +1.2% on Qwen3.5 (one run each).
- Gemma cannot reuse the 71-token shared prompt at 128 in: its cache
  blocks are 64 and 128 tokens (hits land on 128) and the engine rejects
  a smaller block. At 256 in / 256 out the shared prompt covers one
  block and reuse gives +15% (14,984 against 12,999), but the larger
  shape is slower than 128 in without reuse (16,599). A larger shape
  does not raise Gemma's output rate.
- Multi-token prediction on Qwen3.5 gives the original reference's total
  (15,114) with 40% fewer streams, each 1.7 times faster. It needs a run
  sized to about 1,200 streams before the figure is quoted.
- Not raised by any lever tried: the gap to LFM2. Expected ceilings from
  what was measured: Gemma about 17,000 (plain sampling); Qwen3.5 about
  15,000.
- Not tried: the 600 W setting (the owner's decision); AITER tuning for
  the dense multiplies (18% of Qwen3.5's time, untuned shapes); Qwen3.5
  at a long-context shape.

## Results to date (2026-11-16)

Everything below is one GPU, 450 W cap, new images (vLLM nightly `9bc16ae6`
and SGLang daily `20261115`, both with AITER main `95d21128`) unless a row
says otherwise. Result files are on the box under `/data/capsim/phase1*/results`.

### Stack (Phase 0)

- Both rebuilt images pass. AITER main is required for Qwen3.8-27B MXFP4: the
  published SGLang daily and the old vLLM 0.33 image both fail in the untuned
  MXFP4 GEMM (shared-memory limit).

### Qwen3.8-27B MXFP4, 128 in / 256 out, output tok/s

| Engine | Setting | Tok/s | Streams |
|---|---|---|---|
| SGLang | default | 2,525 | 71 |
| SGLang | half-precision state | 3,689 | 139 |
| SGLang | half-precision state, state memory ratio 4 | 5,060 | 236 |
| vLLM | AITER on, default attention | 3,535 | 581 |
| vLLM | AITER on, half-precision state | 5,173 | 1,023 |
| vLLM | AITER on, half-precision state, `ROCM_AITER_FA` | 5,710 | 1,024 |

The linear-attention layers keep about 141 MB of recurrent state per stream at
float32, so the state, not the KV cache, sets how many streams fit.
`VLLM_ROCM_SHUFFLE_KV_CACHE_LAYOUT=1` fails on hybrid models.

### Low-concurrency techniques, Qwen3.8-27B MXFP4 on vLLM, 8k in / 1k out

Output tok/s. The first four columns use half-precision state; the last three
use float32 state, which mono decode requires.

| Streams | Reference | MXFP4 fusion | DFlash2 | MTP-3 | Reference f32 | Mono f32 | Mono + MTP-3 f32 |
|---|---|---|---|---|---|---|---|
| 1 | 38 | 35 | 60 | 71 | 50 | 66 | 103 |
| 2 | 176 | 175 | 243 | 331 | 171 | 213 | 320 |
| 4 | 324 | 325 | 415 | 564 | 321 | 385 | 542 |
| 8 | 536 | 532 | 478 | 872 | 518 | 520 | 806 |
| 16 | 972 | 966 | 698 | 995 | 826 | 774 | 929 |
| 32 | 1,732 | 1,560 | 801 | 1,462 | 730 | 978 | 1,222 |

- Mono decode (vLLM #65078): +25% at 2 streams and +20% at 4 against the
  like-for-like float32 reference; no effect past its 4-row cap. With MTP each
  request is 4 rows, so mono only covers one stream there (+16% at 1 stream).
  MTP-3 on float32 state alone: 89 / 309 / 553 / 803 / 941 / 1,210.
- MTP-3 (merged): +88% at 2 streams, +74% at 4, +63% at 8, level at 16, behind
  at 32. About 2.8 tokens accepted per step.
- DFlash2 draft: +38% at 2 streams, +28% at 4, behind from 8 streams.
- MXFP4 activation-quant fusion (#61902): no sites on this model, no change.
- Single-stream rows have few samples and are noisy.

### Small-active MoE models, 128 in / 256 out, output tok/s

Best steady rung per setting. vLLM unless the column says SGLang.

| Model | AITER on | AITER on, FA, cap 2048 | AITER off | SGLang | Old-stack baseline | XE7740 per GPU |
|---|---|---|---|---|---|---|
| Gemma-4-26B-A4B FP8 | 12,671 (cap 2048); 14,402 (cap 1024, BF16 KV) | FA rejected | 13,134 | fails | 15,205 | 15,919 |
| GLM-4.7-Flash BF16 | fails with FP8 KV; 12,750 (cap 1024, BF16 KV) | not applicable | 14,082 | 7,186 | 14,130 | 6,530 |
| Qwen3.6-35B-A3B FP8 | 11,462 (cap 1024) | 13,921 | | 8,896 | 8,647 | 6,802 |
| Qwen3.5-35B-A3B MXFP4 | 12,989 (cap 1024) | 12,258 steady; 14,709 unsteady | | 8,888 | | |
| Qwen3-Coder-Next MXFP4 | 9,745 (cap 1024) | 12,013 | | 5,597 | | |

- The stock ladder offers 512, 2048, 4096 and 8192 streams, with nothing
  between 512 and 2048, and several higher rungs fail the steady-state check
  (Gemma read 15,955 at an unsteady rung). Phase 1b sweeps the batch cap
  (768 to 2048) to place the peak.
- The "cap 1024, BF16 KV" numbers come from two runs that launched without
  their batch-cap and KV-cache flags (capsim bug: an empty `vllm_extra_flags`
  lever replaced the shape's flags; fixed 2026-11-16). They stand as
  measurements of that configuration.
- GLM-4.7-Flash with AITER on and an FP8 KV cache crashes in AITER's MLA
  reduce kernel (32 heads, head dimension 256 unsupported); it runs with a
  BF16 KV cache, slower than AITER off.
- Gemma-4-26B-A4B: `ROCM_AITER_FA` is rejected (multimodal attention), and with
  AITER's linear path off (`VLLM_ROCM_USE_AITER_LINEAR=0`) it reached 13,658.
- SGLang on Gemma-4-26B-A4B FP8 fails at graph capture: AITER has no
  `gemm_a8w8_bpreshuffle` kernel for M=256, N=2816, K=2112.
- Raising the cap to 2048 with `ROCM_AITER_FA` and half-precision state lifted
  Qwen3.6-35B-A3B by 21% and Qwen3-Coder-Next by 23%; both then fill the cache.

### Newer models, first runs, 128 in / 256 out, cap 1024, output tok/s

vLLM with AITER on; SGLang on the rebuilt daily. Hybrid models ran with
half-precision recurrent state. Best steady rung.

| Model | vLLM | SGLang | Note |
|---|---|---|---|
| Laguna-XS-2.1 FP8 | 13,778 | 16,054 | both queue-bound at the cap, cache 16% full |
| LFM2-24B-A2B | 15,103 | unsteady | vLLM peak at 512 streams; GPU power falls as streams rise |
| North-Mini-Code-1.0 FP8 | 14,420 | fails | vLLM peak at 512 streams; SGLang needs `SGLANG_USE_AITER_UNIFIED_ATTN=1` for the sliding window |
| Nemotron-3.5-Lightning-30B-A3B FP8 | 11,092 | 9,798 | vLLM queue-bound at the cap |
| Muse-Glimmer-30B w8a8 | 7,095 | fails | SGLang: `int8_scaled_mm` not defined |
| Arete-27B | 4,485 | 3,939 | dense hybrid; vLLM peak rung had no completed requests |
| Qwen3.6-27B FP8 | 3,654 | 4,186 | dense hybrid; vLLM peak rung had no completed requests |
| Nemotron-3-Super-120B-A12B FP8 | 1,740 | fails | SGLang: modelopt FP8 unsupported on ROCm |
| Step-3.5-Flash MXFP4 | 1,675 | fails | vLLM holds 72 streams (cache full); SGLang: rope config key missing |
| Qwen3.8-Flash-Next UINT4 | fails | fails | vLLM: no `per_token_table` in the n-gram embedding; SGLang: unsupported fused-MoE scheme |
| Instella-MoE-16B-A3B | fails | fails | vLLM: attention not dispatchable; SGLang: K/V layout mismatch |
| Raon-1-mini-preview | fails | fails | no implementation in either engine |

- LFM2's SGLang rungs were all unsteady and collapsed to 1,025 tok/s at 8,192
  offered streams; its 15,853 reading is not usable.
- On vLLM the fastest models (LFM2, North-Mini, Laguna) lose GPU power as
  streams rise (432 W to 348 W on LFM2), which points at a host-side limit.
  Serving settings (`--stream-interval`, `--api-server-count`) are being tried.

### Follow-ups on the newer models, 128 in / 256 out, output tok/s

| Model | Engine | Setting | Cap | Tok/s | GPU W |
|---|---|---|---|---|---|
| LFM2-24B-A2B | vLLM | first run | 1024 | 15,103 | 432 |
| LFM2-24B-A2B | vLLM | `--stream-interval 8` | 1024 | 16,571 | 351 |
| LFM2-24B-A2B | vLLM | `--api-server-count 4` | 1024 | 16,790 | 348 |
| LFM2-24B-A2B | vLLM | both | 1024 | 16,851 | 351 |
| LFM2-24B-A2B | vLLM | both | 2048 | 24,068 | 334 |
| North-Mini-Code-1.0 FP8 | SGLang | `SGLANG_USE_AITER_UNIFIED_ATTN=1` | 1024 | 19,943 | 431 |
| Laguna-XS-2.1 FP8 | vLLM | cap raised | 2048 | 13,395 | 442 |
| Laguna-XS-2.1 FP8 | SGLang | cap raised | 2048 | 11,412 | 312 |
| Nemotron-3.5-Lightning FP8 | vLLM | `ROCM_AITER_FA` | 1024 | 11,048 | 448 |
| Nemotron-3.5-Lightning FP8 | vLLM | `ROCM_AITER_FA` | 2048 | 11,469 | 443 |

- During a Gemma run the vLLM API server process sat at 100% of one core and
  the engine core at 200%, with the GPU at its 450 W cap. On LFM2 the GPU is
  at 334 W at the 24,068 peak, so that model is still host-bound.
- Laguna-XS does worse at a 2048 cap on both engines; its best stays 16,054
  on SGLang at 1024.

### Serving settings on the fast models, 128 in / 256 out, output tok/s

Serving settings on vLLM are `--stream-interval 8 --api-server-count 4`; on
SGLang, `--stream-interval 8`. "Before" is the best earlier result on that
engine at any cap.

| Model | Engine | Cap | Before | With serving settings | GPU W |
|---|---|---|---|---|---|
| LFM2-24B-A2B | vLLM | 4096 | 24,068 | 29,806 | 356 |
| LFM2-24B-A2B | vLLM (stream 16, 8 servers) | 4096 | 24,068 | 28,604 | 359 |
| North-Mini-Code-1.0 FP8 | SGLang, unified attention | 2048 | 19,943 | 25,211 | 449 |
| North-Mini-Code-1.0 FP8 | SGLang, unified attention | 1024 | 19,943 | 20,271 | 433 |
| North-Mini-Code-1.0 FP8 | vLLM | 2048 | 14,420 | 18,612 | 450 |
| Laguna-XS-2.1 FP8 | vLLM | 2048 | 13,778 | 19,477 | 450 |
| Laguna-XS-2.1 FP8 | vLLM | 1024 | 13,778 | 14,098 | 448 |
| Gemma-4-26B-A4B FP8 | vLLM | 1024 | 14,660 | 15,590 | 449 |
| Gemma-4-26B-A4B FP8 | vLLM | 768 | 14,660 | 14,683 | 439 |
| GLM-4.7-Flash | vLLM, AITER off | 2048 | 14,082 | 15,136 | 448 |
| Qwen3.6-35B-A3B FP8 | vLLM | 2048 | 13,921 | 14,633 | 446 |
| Nemotron-3.5-Lightning FP8 | vLLM, bf16 state | 2048 | 11,469 | 12,264 | 444 |

- With the serving settings the wider cap wins on every model, the reverse
  of the earlier runs: Laguna-XS at 2048 went from 13,395 to 19,477.
- LFM2 is the one model still short of the power cap. Its vLLM engine core
  process sits at 100% of one core; the four API servers are at 20 to 30%
  each, and eight servers with stream interval 16 do no better. The next
  lever there is the engine core, not the API servers.
- North-Mini on SGLang at a 2048 cap without batched streaming recorded
  30,759 at one rung with 959 requests in flight and the GPU at 366 W, after
  an unsteady rung. That reading is a backlog flush, not a rate, and is not
  used.
- SGLang with unified attention on Laguna-XS: 15,864, no change from 16,054
  without it.

### Second serving round and two engines on one GPU, 128 in / 256 out, output tok/s

Same serving settings as above. "Two engines" is capsim's `replicas_per_gpu: 2`:
two vLLM processes on the card, started one after the other, each with half
the memory share and its own batch cap.

| Model | Engine | Cap | Tok/s | GPU W |
|---|---|---|---|---|
| LFM2-24B-A2B | vLLM, one engine | 8192 | 28,030 | 357 |
| LFM2-24B-A2B | vLLM, one engine, `--async-scheduling` | 4096 | 28,559 | 365 |
| LFM2-24B-A2B | vLLM, two engines | 2 x 2048 | 37,653 | 450 |
| LFM2-24B-A2B | vLLM, two engines | 2 x 4096 | 45,519 (15 s reading; 10-minute mean 42,081) | 450 |
| North-Mini-Code-1.0 FP8 | SGLang, unified attention | 4096 | 25,147 | 447 |
| North-Mini-Code-1.0 FP8 | vLLM | 4096 | 23,358 | 444 |
| Laguna-XS-2.1 FP8 | vLLM | 4096 | 21,891 | 450 |
| Laguna-XS-2.1 FP8 | SGLang | 2048 | 26,100 | 447 |
| Laguna-XS-2.1 FP8 | SGLang | 1024 | 16,013 | 429 |
| Gemma-4-26B-A4B FP8 | vLLM | 1536 | 17,433 | 450 |
| Gemma-4-26B-A4B FP8 | vLLM | 2048 | 16,922 | 449 |
| GLM-4.7-Flash | vLLM, AITER off | 4096 | 14,983 | 448 |
| Qwen3.5-35B-A3B MXFP4 | vLLM, bf16 state | 2048 | 15,966 | 446 |
| Qwen3-Coder-Next MXFP4 | vLLM, bf16 state | 2048 | 11,647 | 445 |
| Nemotron-3.5-Lightning FP8 | SGLang, bf16 state | 2048 | 9,906 | 437 |
| Qwen3.6-35B-A3B FP8 | SGLang, bf16 state | 2048 | 8,788 | 409 |

- One LFM2 engine levels off at 28 to 30k (four runs) with its engine core
  at 100% of a host core and the GPU near 357 W. Two engines take the GPU to
  the 450 W cap. The 45,519 reading is the top rung of the ladder (8,192
  streams, 8,140 in flight). It is a 15-second reading; three 10-minute
  holds later gave 42,081 (hardening section below).
- Two engines started together fail: each counts the other's allocation
  while sizing its KV cache and the second exits with "No available memory
  for the cache blocks". capsim now starts shared-GPU engines one at a time.
- North-Mini on SGLang at a 4096 cap recorded nothing at the 4,096-stream
  rung (GPU idle at 102 W); the 25,147 is its 2,048-stream rung.
- Gemma at 17,433 is 9.5% above the XE7740's 15,919 per GPU.
- SGLang on the two linear-attention hybrids holds only about 450 to 530
  requests in flight whatever the cap, so batched streaming changes nothing
  there (Lightning 9,906, Qwen3.6-35B-A3B 8,788); vLLM stays ahead on both.
- Qwen3-Coder-Next held 1,586 in flight at a 2048 cap: memory for its
  recurrent state, not the cap, sets its concurrency.

### Shape search, Qwen3.6-35B-A3B FP8, vLLM AITER on, bf16 state, 128 in, output tok/s

No serving settings (run before they were found).

| Cap | 256 out | 1024 out |
|---|---|---|
| 1024 | 11,575 | 12,806 |
| 1536 | 13,540 | 14,900 |
| 2048 | 13,827 | 15,559 |

Longer outputs raise the rate on this model, the reverse of Gemma. The two
highest 1024-output cells had no request complete inside the measured window;
the next rung of the 2048 cell agrees (15,376).

### Shape search, Gemma-4-26B-A4B FP8, vLLM AITER on, 128 in, output tok/s

| Cap | 256 out | 1024 out |
|---|---|---|
| 768 | 14,660 | 11,825 |
| 1024 | 13,595 | 12,584 |
| 1536 | 12,838 | 12,212 |
| 2048 | 12,870 | 10,146 |

Longer outputs lower the rate; the peak is at the smallest cap tried. The GPU
is at 450 W throughout, so this model is bound by the card at the power cap.

### Profiling and tuning slot (scheduled 2026-11-16)

Order, and why: the profile decides which settings every later run uses, so
it goes before the shape search, the larger models and the third-party
benchmark campaigns, and after the serving round already in flight.

1. Profile (about 40 minutes of GPU time). LFM2 on one engine and on two,
   then Gemma-4-26B and Laguna-XS as card-bound cases. Each run records GPU
   power, activity and memory-controller activity once a second, a py-spy
   profile of each engine core, and a torch profiler trace of 30 engine
   steps. Arms `/data/capsim/phase1p`, sidecar `/data/capsim/phase0/prof_side.sh`.
2. Act on the profile, most likely in this order:
   - Tune AITER for the shapes with no tuned entry. On LFM2 the fused expert
     kernel runs on its default ("using 2stage default") and 3,531 matrix
     multiplies report "not found tuned config" across seven shape families.
   - Repeat the 45,519 reading and extend the ladder past 8,192 streams.
   - LFM2.5-24B-A2B FP8 (LiquidAI, 2026-10, 24.7 GB): newer model, half the
     weight bytes, room for three or four engines.
   - SGLang on LFM2 with batched streaming; host placement of engine cores.
3. Shape search, on the settings that come out of step 2.
4. Larger models (first-load checks can run earlier; they do not depend on
   tuning).
5. AIPerf and GuideLLM campaigns on the final settings.

Already checked without the GPU: vLLM runs LFM2 natively (`lfm2_moe`) and
its experts go through AITER's fused expert kernel, not a per-expert loop.
Needs a decision, not scheduled: raising the 450 W cap; `perf` is usable
as it stands (`perf_event_paranoid` is -1).

### Profile results (2026-11-16)

vLLM torch profiler, 30 engine steps at the stated streams in flight, 128 in
/ 256 out, serving settings on. Share of GPU kernel time:

| Kernel group | LFM2, one engine | LFM2, two engines | Gemma-4-26B | Laguna-XS |
|---|---|---|---|---|
| Streams in flight | 4,096 | 4,096 each | 1,536 | 4,096 |
| Expert layers (fused MoE) | 48% (AITER) | 50% (AITER) | 16% (vLLM's own kernel) | 20% (AITER FP8) |
| Attention | 19% (vLLM kernel) | 9% (vLLM kernel) | 42% (vLLM kernel) | 48% (vLLM kernels) |
| Dense matrix multiplies (rocBLAS) | 16% | 17% | 15% | 14% |
| Sampling (top-k/top-p, Gumbel) | 6% | n/a | 8% | 2% |
| GPU kernel time per step | 85 ms | 162 ms | 74 ms | 166 ms |
| Step time from throughput | 137 ms | 180 ms | 88 ms | 187 ms |
| GPU activity / power | 60% / 325 W | 100% / 450 W | 100% / 448 W | 100% / 450 W |
| Memory-controller activity | 19% | n/a | 24 to 35% | 19 to 23% |

- One LFM2 engine keeps the GPU busy 62% of each step; the rest is the
  engine core on the host (92 to 100% of one core). Two engines fill the
  GPU: each engine's kernels take 1.9 times as long because the two share
  the card. 4,096 tokens per 85 ms of kernel time puts the ceiling for this
  kernel mix near 48k tok/s; the two-engine 45,519 is 95% of it. More
  engines will not help LFM2 until the kernels get faster.
- Attention on vLLM runs in vLLM's own paged-attention kernel on all three
  models, 19 to 42% of GPU time; AITER's flash attention was never selected
  for them. SGLang, which uses AITER attention, is ahead of vLLM on
  Laguna-XS (26,100 against 21,891).
- Dense multiplies go to rocBLAS; the engine logs show AITER has no tuned
  entry for these shapes. LFM2's expert kernel runs on its default too.
- Memory-controller activity stays under 35%: the card is limited by compute
  at the 450 W cap, not by memory bandwidth.
- The engine-core CPU profile (py-spy) did not run: the sidecar picked the
  wrong process id. The process list shows one engine core at 92 to 100% of
  a core on LFM2 and about 190% on Gemma and Laguna-XS.

Runs queued from this (`/data/capsim/phase1c`, e and g arms): AITER flash
attention on Laguna-XS, North-Mini, LFM2 (two engines) and Gemma in
text-only mode; a repeat of the two-engine LFM2 peak; LFM2.5-24B-A2B FP8 on
one, two and three engines; first load of Qwen3.5-122B-A10B MXFP4.

### Runs from the profile, 128 in / 256 out, output tok/s

vLLM, serving settings on. "FA" is `--attention-backend ROCM_AITER_FA`.

| Model | Setup | Cap | Before | Tok/s | GPU W |
|---|---|---|---|---|---|
| Laguna-XS-2.1 FP8 | FA | 4096 | 21,891 | 24,973 | 449 |
| North-Mini-Code-1.0 FP8 | FA | 4096 | 23,358 | 24,985 | 438 |
| LFM2-24B-A2B | two engines, FA | 2 x 4096 | 45,519 | 40,213 | 449 |
| LFM2-24B-A2B | two engines, repeat | 2 x 4096 | 45,519 | 46,548 | 448 |
| LFM2.5-24B-A2B FP8 | one engine | 4096 | n/a | 28,612 | 345 |
| LFM2.5-24B-A2B FP8 | two engines | 2 x 4096 | n/a | 40,732 | 446 |
| LFM2.5-24B-A2B FP8 | three engines | 3 x 4096 | n/a | 39,814 | 442 |

- AITER flash attention gains 14% on Laguna-XS and 7% on North-Mini and
  brings vLLM level with SGLang on both (26,100 and 25,211). It loses 12%
  on LFM2, which keeps vLLM's default kernel.
- The two-engine LFM2 peak repeats: 45,519 and 46,548. Both are 15-second
  readings; the 10-minute mean is 42,081 (see the hardening section).
- LFM2.5 FP8 is slower than LFM2 in bf16 on two engines (40,732 against
  46,548) and a third engine adds nothing, as the profile predicted.
- Gemma-4-26B: flash attention refuses it in text-only mode too ("head_size
  not supported").
- Qwen3.5-122B-A10B "MXFP4" from olka-fi is compressed-tensors 4-bit, not
  AMD's Quark format; vLLM sends it to `gptq_marlin_repack`, which the ROCm
  build does not have. Laguna-S-2.1 MXFP4 from the same author is the same
  format.

### Larger models: candidates (searched 2026-11-16, not yet run)

Sizes are the weight files on Hugging Face. The card holds 143.7 GB; at a 0.95
share that is 136.5 GB for weights, workspace and KV cache together.

60B to 200B, released in 2026, with a checkpoint this stack can load:

| Model | Released | Params (total / active) | Checkpoint | Weights GB | On the box |
|---|---|---|---|---|---|
| Nemotron-Labs-3-Puzzle-75B-A9B | 2026-06 | 75B / 9B | nvidia FP8 | 83 | no |
| Qwen3.5-122B-A10B | 2026-02 | 122B / 10B | olka-fi MXFP4 (community) | 79 | no |
| Qwen3.5-122B-A10B | 2026-02 | 122B / 10B | Qwen FP8 | 127 | yes |
| Laguna-S-2.1 | 2026-07 | 118B | olka-fi MXFP4 (community) | 68 | no |
| Laguna-S-2.1 | 2026-07 | 118B | poolside FP8 | 131 | no |
| Nemotron-3.5-Super-VL-120B-A12B | 2026-10 | 120B / 12B | RedHatAI FP8 | 129 | no |
| Qwen3.8-Flash-Next | 2026-08 | 180B | amd AWQ UINT4 | 80 | yes (does not load) |
| Step-3.5-Flash | 2026-02 | 199B / 11B | amd MXFP4 | 115 | yes |
| MiniMax-M2.7 | 2026-04 | 229B / 10B | amd MXFP4 | 127 | no |

- An FP8 checkpoint of a 120B model is 127 to 131 GB and leaves under 10 GB
  for the KV cache. That is why Nemotron-3-Super FP8 measured 1,740 tok/s;
  the 4-bit checkpoints (68 to 83 GB) leave 50 GB and more.
- Qwen3.8-Flash-Next (`qwen4_exp`) fails to load on both current images.

Largest model on one card: MXFP4 weights run about 0.55 GB per billion
parameters. Qwen3.5-397B-A17B MXFP4 is 221.5 GB and K2-Horizon-375B-A23B
MXFP4 is 218 GB, so neither fits one card; each fits two (287 GB). The
ceiling for one card is about 230B parameters: MiniMax-M2.7 (229B, 127 GB)
with about 9 GB left for cache, or Step-3.5-Flash (199B, 115 GB) with about
22 GB. Neither fits a 96 GB card at 4 bits (MiniMax-M2.7 NVFP4 is 140 GB).

### Larger models: first loads (2026-11-16), 128 in / 256 out, output tok/s

vLLM, serving settings on, memory share 0.98 where the weights are over
100 GiB. No tuning yet.

| Model | Params | Checkpoint | Weights GiB | KV cache | Cap | In flight | Tok/s | GPU W |
|---|---|---|---|---|---|---|---|---|
| MiniMax-M2.7 | 229B / 10B | amd MXFP4 | 119 | 20 GiB, 147k tokens | 512 | 491 | 6,721 | 450 |
| MiniMax-M2.7 | 229B / 10B | amd MXFP4 | 119 | 20 GiB | 256 | 253 | 4,774 | 450 |
| Step-3.5-Flash | 199B / 11B | amd MXFP4 | 107 | 23 GiB, 73k tokens | 512 | 277 | 3,741 | 448 |
| Nemotron-Labs-3-Puzzle-75B-A9B | 75B / 9B | nvidia FP8 | 73 | 60 GiB | 1024 | 753 | 3,830 | 445 |
| Qwen3.5-122B-A10B | 122B / 10B | Qwen FP8 | 117 | 22 GiB | 256 | 126 | 2,460 | 450 |

- MiniMax-M2.7 is the largest model that runs on one card, and it does not
  fit a 96 GB card at 4 bits. Its KV cache holds about 500 requests of this
  shape, which is what limits it.
- Step-3.5-Flash holds only 277 requests (311 KB of cache per token) and one
  of its layer types falls back to an emulated MXFP4 path.
- Puzzle-75B runs its experts and its Mamba state update on Triton kernels,
  not AITER's; Qwen3.5-122B FP8 leaves room for 126 requests.
- Do not load: Qwen3.5-122B MXFP4 and Laguna-S MXFP4 from olka-fi
  (compressed-tensors 4-bit; vLLM has no ROCm kernel, SGLang fails on
  `w2_weight_packed`).
- Laguna-XS on SGLang at a 4096 cap: 26,347 (26,100 at 2048).

### After the first loads (2026-11-16)

| Model | Setup | Shape | Cap | In flight | Tok/s | GPU W |
|---|---|---|---|---|---|---|
| MiniMax-M2.7 MXFP4 | vLLM, FA | 128 / 256 | 640 | 460 | 5,697 | 450 |
| MiniMax-M2.7 MXFP4 | vLLM | 1,024 / 1,024 | 128 | 125 | 2,682 | 450 |

- Flash attention loses 15% on MiniMax-M2.7 (5,697 against 6,721), so it
  keeps vLLM's default kernel, like LFM2.
- At 1,024 in / 1,024 out one card holds 125 MiniMax-M2.7 requests and
  generates 2,682 tok/s. The XE7740 measured the FP8 checkpoint at 4,827
  tok/s for the box at that shape (two engines, four cards each, AIPerf
  with unique prompts), about 603 per card. The two runs differ in
  precision, engine layout and prompt reuse, so this is context and not a
  like-for-like ratio.
- Gemma-4-26B rejects `ROCM_ATTN` as well as `ROCM_AITER_FA`.
- Kernel tuning: AITER's fused-MoE tuner on LFM2's shapes found a gain only
  at 512 tokens per call (14%) and none from 1,024 to 8,192, which is where
  the headline load runs. The dense (bf16) tuner's search is 34,688 trials
  and was stopped unfinished after 25 minutes. Nothing was installed.

### Hardening the LFM2 two-engine figure (2026-11-16)

The 45,519 and 46,548 above are 15-second readings. At 8,192 streams the
rate swings between about 48,000 and about 25,000 as each wave of requests
finishes and the next wave's prompts are processed, so the harness gained a
fixed hold (`hold_s`) and a fixed batch (`batch_requests`), both measured
from the engines' counters at the two ends. Full write-up:
`docs/mi350p-lfm2-throughput-claim.md`.

| Run | What | Tok/s | In flight | Notes |
|---|---|---|---|---|
| run_415 to run_417 | 600 s hold, three engine starts | 42,040 / 42,024 / 42,179 | 8,060 | mean 42,081, range 0.4%; 447 W; TTFT p95 4.2 to 4.4 s |
| run_418 | batch of 50,000 requests | 40,909 incl. ramp and drain | 7,582 | client token count 99.91% of the engines' |
| run_419 | 600 s hold, prefix reuse off | 36,918 | 7,999 | 12% lower; no errors |
| run_420 to run_422 | 60 scored questions: bf16 KV idle, fp8 KV idle, fp8 KV under load | n/a | 0 / 0 / 8,192 | 60 of 60 in all three |
| run_423 | 600 s hold, 1,024 in / 1,024 out | 20,440 | 4,080 | over-driven: 8,192 offered, the rest timed out in the queue |
| run_426 | `vllm bench serve`, shared 71-token prefix, 100,000 requests | 41,566 steady on engine counters | 8,192 | 0 failed; within 1.2% of the holds; 49% cache hits |
| run_428 | `vllm bench serve`, unique prompts, 100,000 requests | 36,799 steady on engine counters | 8,192 | 0 failed; 3% cache hits; 0.3% from the reuse-off hold |
| run_429 | `vllm bench serve`, shared prefix, 60,000 requests | 36,618 steady on engine counters | 4,096 | 0 failed; 109 ms per token |
| run_430 | 600 s hold, 1,024 in / 1,024 out, 3,840 offered | 21,245 | 3,378 | 27% of requests cut: each wave's prompts take 45 to 50 s and stall running streams |
| benchmarks/fc7ee79ef6db | AIPerf 0.13.0, unique prompts, 600 s | 36,856 | 8,192 | 0 failed; no cache hits; 0.2% from the reuse-off hold |
| benchmarks/fc7ee79ef6db | AIPerf 0.13.0, unique prompts, 600 s | 32,936 | 4,096 | 0 failed; TTFT p95 3.0 s |

- The LFM2 two-engine figure to quote is 42,081 (10-minute mean), not
  45,519.
- Two outside tools agree with the harness: vLLM's own benchmark tool with
  a shared prefix (41,566) and AIPerf with unique prompts (36,856 against
  capsim's 36,918 with reuse off). The shared system prompt is worth about
  12% at this shape.
- Lead not followed: with no new prompts arriving, two engines at 2,037
  streams each decoded at about 61,000 tok/s (66 ms per token); under
  sustained load at that stream count they gave 36,618 (109 ms). Prompt
  processing interleaved with running streams costs more than its token
  share, at both shapes. Staggered arrivals and the prompt batch budget
  (`--max-num-batched-tokens`) are untested.
- capsim's AIPerf path needed two fixes on this box: the idle-GPU check
  crashed on ROCm 10's idle output (`simulator/gpu_vendor.py`), and the
  system Python has no `venv`, so the service runs with
  `CAPSIM_BENCHMARK_PYTHON` set to a uv-managed Python 3.12.
- About 0.1% of streams stop delivering after their first tokens and are
  closed by the client after 40 s; the engines still generate their tokens.
  None occurred with reuse off. Cause not established.

### Outside-tool check on the other models (2026-11-16 to 17)

`vllm bench serve` against each model's best recorded configuration, 128 in /
256 out, with the shared-prefix and unique split capsim used for that model,
about 8 minutes of load, streams at the engine cap. "Sustained" is the
engine's generated-token counter over the steady part of the run for vLLM
and the tool's own whole-batch figure for SGLang (its counter moves only
when a request completes). The capsim column is the earlier 15-second
reading. Data: `/data/capsim/vbench_models/` on the box.

| Model | Engine | Streams | capsim reading | Sustained | Difference | Failed | Cache hits | Run |
|---|---|---|---|---|---|---|---|---|
| MiniMax-M2.7 MXFP4 (229B) | vLLM | 512 | 6,721 | 6,633 | -1.3% | 0 of 12,500 | 58% | run_435 |
| Gemma-4-26B-A4B FP8 | vLLM | 1,536 | 17,433 | 16,765 | -3.8% | 0 of 32,000 | 1.5% | run_431 |
| Puzzle-75B-A9B FP8 | vLLM | 1,024 | 3,830 | 3,611 | -5.7% | 0 of 7,000 | 0% | run_436 |
| Qwen3.5-35B-A3B MXFP4 | vLLM | 2,048 | 15,966 | 15,046 | -5.8% | 0 of 30,000 | 0% | run_432 |
| North-Mini-Code-1.0 FP8 | SGLang | 2,048 | 25,147 | 21,699 | -14% | 0 of 47,000 | 53% | run_434 |
| Laguna-XS-2.1 FP8 | SGLang | 4,096 | 26,347 | 21,120 | -20% | 0 of 50,000 | 62% | run_433 |

- The vLLM rows hold up to within 1 to 6% over several minutes.
- The two SGLang rows do not: their rate swings between about 10,000 and
  33,000 across each request wave, and the 15-second reading caught the high
  part. Quote about 21,000 for both until a 10-minute hold says otherwise.
  vLLM with flash attention (24,973 and 24,985, also short readings) may now
  be the better configuration for both and has not had the same check.
- Gemma, Qwen3.5-35B-A3B and Puzzle-75B get no prefix-cache hits on vLLM, in
  capsim's runs as well as the tool's, so the 54% reuse setting did not
  apply to those rows.
- AIPerf with unique prompts on Laguna-XS (SGLang) gave 17,017, with a mean
  output of 253 tokens against the 256 requested
  (`runs/benchmarks/6b771cf8e0f6af8f8bba62de`).

### Not running yet

- Gemma-4-31B MXFP4: vLLM fails loading weights (shape [5376, 576] against
  [5376, 1152]); `ROCM_AITER_FA` rejects its multimodal attention.

## 6. What stays as it is

- **Prefix caching stays on.** Capsim's industry setting (54 % reuse)
  needs it. InferenceX disables caching, so its numbers are not directly
  comparable.
- **External numbers are context, not targets.** StorageReview's MI350P
  figures use 512 in / 512 out at 128 streams with undisclosed settings;
  MLPerf runs on 8-GPU MI355X nodes.
- **One GPU, no tensor parallelism.**

## 7. Measurement guard (every phase)

Every rung's counter rate is checked against two client-side bounds: the
completions in its window, and requests in flight divided by per-token
latency. A rung more than 25 % above both is excluded. The first run had
20 such rungs of 470.

## 8. Known gaps upstream

- vLLM #60581: on gfx950, MXFP4 activation quantization is not fused into
  the GEMM. PRs #59153 and #61902 are open.
- vLLM #63712, #65078, #63341: several decode kernels assume the 256-CU
  part; the MI350P versions are open PRs.
- vLLM #65134 and AITER #7325: Gemma-4-26B-A4B MoE tables for the MI350P
  are open, not merged.
- AITER #7749: 128-CU split values for gpt-oss's windowed unified
  attention are open.
- Mono decode exists for the MI350P only as draft vLLM #65078; SGLang's
  version (#43558) was closed unmerged.
- SGLang lists MI350P support as a Q4 2026 roadmap item (#46462) and
  still pins the October AITER.

## Sources

- AITER: github.com/ROCm/aiter, PRs #5182, #6939, #7075, #7254, #7218,
  #7683; issues #6255, #7325, #7749, #7944.
- vLLM: github.com/vllm-project/vllm, PRs #61573, #65078, #61902,
  #59153, #61855, #64142, #65134, #67580.
- Hugging Face: `amd/Qwen3.8-27B-Quark-DFlash2` model card; sizes of the
  `amd/*MXFP4` checkpoints and the newest open models.
- vLLM `docker/Dockerfile.rocm_base` (AITER v0.1.29) and SGLang
  `docker/rocm.Dockerfile` (AITER `285579b6`), both on main, 2026-11-15.
- Docker Hub tag lists for `vllm/vllm-openai-rocm`, `rocm/sgl-dev`,
  `rocm/vllm`, `rocm/pytorch`.
- AMD, "vLLM V1 performance optimization",
  rocm.docs.amd.com/projects/ai-ecosystem.
- ROCm 10.1.0 release notes and compatibility matrix, rocm.docs.amd.com.
- Card figures: ROCm 10.0 MI350 architecture page; ServeTheHome
  (2026-05-07); StorageReview first look (2026-11-03).
