# P11 scoping: an all-US model and software stack on Dell (draft, 2026-10-25)

Status: scoping only. Nothing measured yet. In `catalog.json` as P11 (planned,
implementation pending).

Decision (2026-10-25): US-only showcase. We test the best US model in each tier on the
XE7740 and don't compare against non-US models.

## The story

Many enterprise and public-sector buyers won't run Chinese-origin models, because of
procurement rules, legal review, or just perception. They want to know what they can
run entirely with US models and software, and how much one server delivers.

Headline we want to be able to say:

> "A single Dell XE7740 running only US-origin models and US-stewarded serving software
> serves N users of a frontier-class model at our latency target. It runs air-gapped
> and ships with a complete software bill of materials."

The claim is about capacity and quality. "Security" is shown with evidence (where
everything comes from, and a working offline deploy), never with a "% more secure"
number. P07 uses the same rule.

## Threat model (be honest about it)

Open weights running on-prem don't phone home. So "Chinese models steal data" is not
the risk we should claim. These risks are real and worth naming:

- **Compliance and procurement.** Federal, state, and corporate policies that ban or
  restrict PRC-origin AI. This is the main reason people buy.
- **Model behavior.** Training-set bias or censorship, and the possibility of
  hidden behavior that is hard to audit. This applies to every model. Knowing where a
  model came from lowers the risk; it doesn't remove it.
- **Software supply chain.** Serving code and `trust_remote_code` model loaders run with
  full privileges. Fix: pinned container digests, an SBOM, no remote code.

## US-origin models (as of 2026-10-25)

| Tier | Model | Size (total / active) | Format for XE7740 | Notes |
|---|---|---|---|---|
| Frontier | **Reflection Beam 1.0** | 501B / 23B MoE | NVFP4 build, about 255 GB | Weights out 2026-10-16. Apache 2.0, 1M context, includes a 2B DFlash2 drafter. vLLM/SGLang/Dynamo support at launch. Reflection names Dell as a partner for air-gapped deploys. |
| Frontier | NVIDIA Nemotron 3 Ultra | 550B / 55B hybrid MoE | NVFP4 build, about 330 GB | Top US open model on Artificial Analysis. TRT-LLM and vLLM deploy guides exist. |
| Frontier | Arcee Trinity Large | about 400B | TBD | OpenMDW license |
| Frontier | Thinking Machines Inkling | 975B MoE | TBD | Apache 2.0. Probably too big for a comfortable single-box run. |
| Mid | Nemotron 3 Super, gpt-oss-120b | — | NVFP4 / MXFP4 | gpt-oss already failed our 96K quality check in P02. |
| Small | Gemma 4 26B, gpt-oss-20b | — | NVFP4 | Gemma 4 26B already in the roofline (127k gen tok/s) |
| Speech | Whisper large-v3, NVIDIA Parakeet/Canary | — | — | For pipeline projects |

Sizing on the XE7740 (8 × 96 GB = 768 GB, PCIe only):
- Beam NVFP4 (255 GB) fits at tp4 with about 120 GB left for KV cache, or at tp8 with about 500 GB. That allows two tp4 copies per box, the same replicas-versus-TP question we asked for the other giants.
- Nemotron 3 Ultra NVFP4 (330 GB) is tight at tp4 (about 50 GB left for KV), so tp8.

Gate: the RTX PRO 6000 is SM120, not SM100. NVFP4 MoE already works on SM120
(DeepSeek-V3.1 and Kimi-K2 under SGLang tp8). Beam's attention for 1M context and
Nemotron's Mamba layers may still need kernels that SM120 doesn't have. The preflight
checks this before any campaign.

## US-stewarded serving software

| Component | Steward | Use in this arm |
|---|---|---|
| vLLM | PyTorch Foundation (Berkeley origin) | yes |
| SGLang | LMSYS | yes |
| TensorRT-LLM, Dynamo, NIXL | NVIDIA | yes |
| LMCache | TensorMesh / UChicago | yes (KV tier) |
| KTransformers | Tsinghua / Approaching.AI | **excluded** |
| Mooncake | Moonshot AI | **excluded** |

Caveat for the white paper: open-source projects have contributors from around the
world. "US-stewarded" means a US organization governs the project and we can pin and
audit the code. It does not mean every contributor is American.

## Test design (kept simple)

Run each tier's candidates on the same box, workload, and SLO, reusing the roofline and
open-loop harness we already have. Each candidate gets its own best supported engine and
parallelism, and the tier winner becomes the headline:

| Tier | Candidates | Configurations |
|---|---|---|
| Frontier | Beam NVFP4, Nemotron 3 Ultra NVFP4, Trinity Large (if a build fits) | Beam: tp4 × 2 replicas vs tp8, DFlash2 drafter on/off. Nemotron: tp8. |
| Mid | Nemotron 3 Super, gpt-oss-120b | best engine, replicas vs TP |
| Small | Gemma 4 26B, gpt-oss-20b | one replica per GPU |

For each candidate:
1. Preflight: does it load, serve, and pass numerical checks? If not, report it as unsupported on this platform; don't drop it silently.
2. Capacity: users at SLO and gen tok/s per server (open-loop roofline).
3. Quality: the same quality adapter and prompts for every candidate (coding, long-context recall, tool use).

## Outputs

- **O1:** US-only capacity per XE7740 at SLO for the frontier tier winner.
- **O2:** Per-tier table: best US model, its capacity, and its quality score.
- **O3:** Evidence pack: container digests, SBOM, model license and origin, and an offline install log (reuses the ssh `deploy` branch path). Pass/fail scope only.

## Open questions

- Should Llama/Meta Muse be included (custom licenses), or only Apache/OpenMDW?
- Should P11 stay its own project, or become a model-provenance arm of P07 (confidential inference)?
