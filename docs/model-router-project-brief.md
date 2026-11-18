# Model router on AMD Instinct MI350P: project brief

Written 2026-11-16 as a handoff to a new session and a new, separate project.
It is self-contained: the reader is assumed to know nothing of the earlier work.

## 1. What this is

A technical marketing project for AMD. Build and measure a model router: one
server holds several language models of different sizes, all loaded at once,
and a router sends each request to the smallest model that can answer it well.

The claim to prove: **the same hardware serves several times more requests at
the same answer quality where quality matters, with no cold starts, because
every model stays resident in the card's 144 GB.**

The project must stand alone. It must not depend on the capacity simulator
(`capsim`, in the `system-sizing` repository), because the router is expected
to change shape over time. It may borrow measured numbers and launch settings
from that work (section 6).

## 2. Audience and rules

- Output feeds technical marketing (white paper, demo, sizing guidance), not
  a research paper. Harden for credibility; no research-grade reproducibility.
- Keep the method simple and explainable. Do not spend long test time chasing
  a metric.
- Maximise the benefit shown, never manufacture it. Use realistic workloads,
  give the baseline its best configuration, and show the limits.
- **No published AMD-versus-NVIDIA comparison** (a Dell constraint). Compare
  the system against itself, or against cloud pricing.
- **Off limits, already done for Intel:** agentic CPU-to-GPU ratio, video
  translation, AI video assistants, host-memory (EPYC) KV cache tiers, a
  real-time voice assistant. Other video work is allowed.
- Prefix-cache reuse in any benchmark must be what a real deployment would
  see (shared template or system prompt, a session's own history). Recycled
  prompt sets inflated earlier results 1.4 to 3.5 times.
- The goal is to help AMD win deals against cloud APIs and NVIDIA. Favour
  things that work out of the box over clever engineering.

## 3. Hardware

Available now: Dell R7725, 2 x EPYC 9455 (96 cores, 192 threads), 377 GB RAM,
**one** AMD Instinct MI350P.

- MI350P: 144 GB HBM3E, PCIe card, 450 W cap in this server (600 W card).
- Ubuntu 24.04, ROCm 10.0. Access: `ssh amdgpu` (jumps through `xe7740`).
  sudo needs the user's password; the user runs any root step.
- Storage: `/data` (model cache at `/data/capsim/huggingface/hub`, about
  340 GB free) and `/data2` (about 2.5 TB free; large checkpoints live in
  `/data2/capsim/hf-overflow` and are symlinked into the hub).
- The box is shared with a benchmark queue. Check that nothing is running
  before using the GPU (`docker ps`, and `curl localhost:8321/api/status`).

Possible later: an 8 x MI350P system. Not confirmed.

Card facts that shape the design:

- No bridge connectors. A model split across cards talks over PCIe, and AMD's
  kernels for that were immature in autumn 2026. Treat multi-card models as
  unproven until tested on the 8-GPU system.
- The card can be split into four isolated 36 GB hardware partitions. Not
  tried here.
- Several engine processes can share one card with a memory share each. This
  works and is how the fastest result was reached.

## 4. Design

### Tiers

On one card (build now):

| Tier | Role | Candidate | Weights |
|---|---|---|---|
| Large | Hard reasoning, long answers | Qwen3.5-122B-A10B MXFP4 or Laguna-S-2.1 MXFP4 | 74 GB / 68 GB |
| Mid | General work | Gemma-4-26B-A4B FP8 or Qwen3.5-35B-A3B MXFP4 | about 29 GB |
| Fast | Simple, high-volume requests | LFM2.5-24B-A2B FP8 | 24 GB |
| Router | Classify each request | A small model, or a rule set first | small |
| Interactive lane (optional) | One-user speed | Qwen3.8-27B MXFP4 with multi-token prediction | to check |

Weights plus KV cache for every tier must fit in about 136 GB (a 0.95 share
of 143.7 GB). Work out the memory budget before choosing; the table above is
a starting point, not a verified fit. The large-tier checkpoints are
community-made MXFP4 builds and have not been loaded on this stack yet.

On eight cards (later, if the system arrives): a 400B-class model split over
four cards, a 200B-class model over two, a fast model on one card, and a
utility card running several engines. This is the version the user most wants
to show; the one-card build is the proof of concept.

### Router

Start with the simplest thing that works: a rule or a small classifier that
labels each request easy, general or hard and picks the tier. A learned
router or an open-source routing layer can replace it later. The router must
expose one OpenAI-compatible endpoint so an application sees a single model.

### No on-demand loading

Loading on demand is not viable: weights load fast (44 GB in about 8 s) but
an engine takes about 100 s to become ready. The design keeps every tier
loaded, and "no cold starts" is part of the message.

## 5. Benchmark

**Comparison, against itself:** the same hardware configured as the large
model alone (as many copies as fit) serving all traffic, versus the routed
mix. Give the baseline its best serving settings.

**Workload:** a mixed stream with a stated share of easy, general and hard
requests. Use public prompt sets with known answers or accepted scoring so
quality is not judged by hand. State the mix and where it came from; show at
least two mixes so the reader sees how the gain depends on it.

**Measure:**

| Metric | Why |
|---|---|
| Requests per second and tokens per second at a latency target | Capacity |
| Time to first token and time to complete, by tier | User experience |
| Answer quality on the scored prompts, routed versus all-large | The "same quality" half of the claim |
| Share of requests sent to each tier, and misroutes | Whether the router is doing the work |
| GPU power | Feeds a cost-per-request figure |

**Honesty checks:** report quality lost to misrouting, not only capacity
gained. Include a mix that is mostly hard requests, where routing helps
least. Count the router's own cost.

## 6. What is already known (reuse, do not re-measure)

Measured on this box in November 2026, 128 tokens in / 256 out unless noted.
Current figures and settings are in
`docs/r7725-mi350p-roofline-plan.md`, section "Results to date", in the
`system-sizing` repository. Read that first.

- Engine images that work: `capsim/vllm-rocm:nightly-9bc16ae6-aiter-95d21128`
  and `capsim/sglang-rocm:20261115-aiter-95d21128`.
- vLLM settings that mattered: `VLLM_ROCM_USE_AITER=1`,
  `HSA_NO_SCRATCH_RECLAIM=1`, `--stream-interval 8`, `--api-server-count 4`.
  On fast models the serving host, not the card, was the limit until these
  were set.
- SGLang: `--stream-interval 8`; some models need
  `SGLANG_USE_AITER_UNIFIED_ATTN=1`.
- Throughput per card: LFM2-24B-A2B about 42,000 tok/s sustained on two
  engines; Laguna-XS 26,100 and North-Mini-Code 25,200 on SGLang;
  Gemma-4-26B 17,400, Qwen3.5-35B-A3B 16,000, GLM-4.7-Flash 15,100 on vLLM.
- Low latency, Qwen3.8-27B MXFP4, 8k in / 1k out, one stream: 38 tok/s
  standard, 71 with multi-token prediction, 103 with multi-token prediction
  plus AMD's single-pass decode kernel (needs a separate image build). The
  gain is gone by 8 to 16 streams.
- Two engines sharing a card must start one after the other. Started
  together, the second fails with "No available memory for the cache blocks".
- An FP8 checkpoint of a 120B model (127 to 131 GB) leaves almost no cache
  and measured 1,740 tok/s. Use 4-bit checkpoints for the large tier.
- Models that failed to load on these images: Qwen3.8-Flash-Next, Instella,
  Raon, Gemma-4-31B MXFP4.

## 7. Phases

0. **Fit check (half a day).** Load each candidate tier model alone, record
   memory and a quick throughput number. Fix the tier list and memory split.
1. **All tiers resident on one card.** Launch them together with explicit
   memory shares, one at a time. Confirm each still performs acceptably while
   the others are busy.
2. **Router and single endpoint.** Simplest router first.
3. **Benchmark** as in section 5, routed versus all-large.
4. **Interactive lane** if phases 1 to 3 hold.
5. **Demo.** A page that shows requests being routed live, with running
   capacity and quality figures.
6. **Eight-card version** when hardware arrives; first test whether a model
   split across cards works at all.

## 8. Open questions for the user

- Which request mix to lead with (customer support, coding, general
  enterprise assistant), and is there a source for a realistic mix?
- Is a quality benchmark required in the published asset, or only internally?
- Is the 8-GPU system confirmed, and when?
- Deliverable format: white paper, demo, or both?

## 9. Risks

- The large-tier checkpoints may not load, or may be slow, on this stack.
- Several engines on one card compete for it; per-tier numbers measured alone
  will not hold when all are busy.
- The capacity gain depends heavily on the request mix. A mix chosen to
  flatter the router would break the "never manufactured" rule.
- Multi-card models over PCIe are unproven on this card.
- Shared box: runs here can collide with the benchmark queue.

## 10. Not in scope

Head-to-head with NVIDIA, on-demand model loading, fine-tuning, host-memory
cache tiers, voice, video translation, AI video assistants.
