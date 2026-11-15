# R7725 / MI350P: roofline on leading-edge AMD software

2026-11-15, revision 3. Box: Dell R7725, 1x AMD Instinct MI350P (PCIe).

**Goal:** the best performance this pre-release card can show, on the
newest and most capable models that fit it, using the newest AMD software
and the new techniques AMD is launching for it. Pre-release builds and
open pull requests are in scope; every result names exactly what it ran.

Revision 1 tuned settings on the software the box already had. Revision 2
found that software was a month behind AMD's MI350P work and added a stack
refresh. Revision 3 adds AMD's unreleased techniques (section 3) and
re-picks the models (Phase 4).

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

One cell each (cap 2048, 128 in / 256 out). Models:

- gpt-oss-120b (MXFP4 MoE; 128-CU tables now exist)
- Qwen3.8-27B MXFP4 (128-CU tables exist; hybrid, so it runs at its
  Mamba block cap)
- Llama-3.3-70B MXFP4 and FP8 (dense, no 128-CU tables: the control that
  shows what tuning is worth, and the FP8 to MXFP4 step)

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
| T2 | MXFP4 fusion image, fusion flag on | Roofline cell and interactive ladder. Llama-3.3-70B MXFP4 too |
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
tuned-versus-control gap on Llama decides how much Phase 2 is worth.

### Phase 2: tune only what upstream has not (up to 8 GPU-hours)

1. From the Phase 1 logs and one smoke start per remaining model, list
   the shapes that still miss a tuned row. Expected: Llama-3.3-70B,
   Gemma-4-31B MXFP4, gpt-oss-20b, Qwen3.5-35B-A3B, Qwen3-Coder-Next,
   Step-3.5-Flash.
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
| 4a Headline | Qwen3.8-27B MXFP4 (AMD's showcase model for this card: tuned tables, mono decode, DFlash2, MTP); gpt-oss-120b; Step-3.5-Flash MXFP4 (114 GB, the largest that fits); Gemma-4-31B MXFP4. Both engines, roofline grid plus the interactive ladder with the Phase 1 techniques | about 40 | about 12 |
| 4b Newest models | Arete-27B (NousResearch, 11-06, a Qwen3.8-27B fine-tune) and Nanbeige4.3-33B (10-24), BF16 as published; Qwen3-Coder-Next, Qwen3.5-35B-A3B, Llama-3.3-70B and gpt-oss-20b in MXFP4 | about 48 | about 11 |
| 4c FP8 | The baseline's FP8 models and its three fixable failures | about 90 | about 21 |
| 4d BF16 | Optional; the baseline's BF16 numbers stand as "untuned" | about 80 | about 19 |

4a and 4b are the deliverable. 4c and 4d run only if you want the full
grid on the new stack.

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
