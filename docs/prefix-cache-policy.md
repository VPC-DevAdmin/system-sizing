# Prefix-cache reuse in Capsim measurements (2026-11-04, reuse setting added 2026-11-12)

Inference engines skip the prefill of any prompt prefix they have already computed.
That is a real saving in production, so a benchmark should not switch it off, but a
benchmark must only earn the reuse a real deployment would see. Prompts that repeat
across simulated users (a small recycled prompt set, or texts that open with the same
paragraphs) let the engine skip work no deployment skips, and overstate throughput.
This happened in the September 2026 headline sweeps (35-75% hit rates) and in the
2026-10-26 AIPerf runs (90-98.5%; docs/x100k-aiperf-results.md).

Implementation: [`simulator/prefix_policy.py`](../simulator/prefix_policy.py).

## What production reuse looks like

| Source | Workload | Prompt tokens served from cache |
|---|---|---|
| Wang et al., *KVCache Cache in the Wild*, USENIX ATC '25 ([arXiv 2506.02634](https://arxiv.org/abs/2506.02634)), Fig. 4 | Aliyun Tongyi consumer traffic (chat, file, multimodal, search), infinite cache | 62% |
| same | Aliyun business API traffic (OpenAI-compatible), infinite cache | 54% |
| Yuan et al., *DualMap*, ICLR '26 ([arXiv 2602.06502](https://arxiv.org/abs/2602.06502)), Table 1, on Mooncake's public traces (Qin et al., FAST '25) | Kimi multi-turn conversation | 40% |
| same | Kimi tool & agent | 59% |
| vLLM blog, *Serving Agentic Workloads at Scale with vLLM x Mooncake* (2026-05-06), on Inferact's open Codex traces ([HF dataset](https://huggingface.co/datasets/Inferact/codex_swebenchpro_traces)) | Codex coding agent on SWE-bench Pro: 610 sessions, median 33 turns; benchmark-derived, not production | 92.2-94.2% |
| Huang et al., *FlatCake* ([arXiv 2610.38355](https://arxiv.org/abs/2610.38355), 2026-10-28) | Production Claude Code trace, achieved with the paper's own cache design | 94% |

Where the reuse comes from, per the ATC '25 paper: almost all of it is from the same
user (multi-turn history) or from a shared system prompt on single-turn API calls;
reuse across different users is "extremely low". The same paper notes that synthetic
workloads report more than 80%, above anything its chat and API traffic reaches. The
ATC and Mooncake figures are ideal (infinite-cache) reuse; a real cache of finite size
reaches less. Long-running coding agents are the exception: each turn resends a large
tool prefix plus the whole session so far, so 90%+ is normal for them. The agent
evidence is thinner (one benchmark-derived trace set and one new paper's production
trace) than the chat and API evidence.

## Our choices

1. **Allowed reuse:** the model's chat template, a fixed system prompt that every user
   of an application shares (instructions and tool definitions), and a session's own
   earlier turns. Nothing else. Every user message opens with a unique reference
   (`simulator/tokenizer_corpus.py`, since 2026-10-29), so different users never share
   prompt text.
2. **Headline sweeps** (the saturation benchmark behind roofline and white-paper
   numbers) keep the prefix cache on and are built to reach a set reuse level, the way
   production reaches it: a system prompt every user shares, then user text nobody else
   sends. The two together are the sweep's prompt length (128 tokens stays 128). The
   level is the `simulation.prefix_reuse` setting, also on the headline and roofline
   forms in the UI:

   | Setting | Target | Source |
   |---|---|---|
   | `industry` (default) | 54% | business API traffic, ATC '25 Fig. 4 |
   | `consumer_chat` | 62% | consumer chat, ATC '25 Fig. 4 |
   | `conversation` | 40% | Kimi multi-turn conversation, DualMap Table 1 |
   | `template` | chat template only | unique prompts, no system prompt |
   | a number, e.g. `0.45` | as given | our choice; flagged if above 62% |

   The default is the business-API figure because a saturation sweep is single-turn API
   traffic. Each sweep sizes the system prompt from the model's own tokenizer so the
   shared prefix (chat template + system prompt, in 16-token blocks for vLLM, token-level
   for SGLang) lands closest to the target, keeping at least 32 unique user tokens. With
   128-token prompts and 16-token blocks the reachable rates are about 11 points apart,
   so a sweep lands within a few points of the target, not on it. The summary records
   the setting, target, source, the system and user token counts and the rate the split
   gives (`prefix_cache.template_floor`). Every rung records its measured hit rate
   (`prefix_hit_rate`, from the engine's hit and query counters summed over replicas
   across the published chunk) and is labelled `template` or `above_template` (more than
   5 points over the designed rate). A roofline cell's level is part of its identity, so
   cells measured at another level are measured again, not reused. An engine that cannot
   prefix-cache (Kimi-Linear) reads 0% whatever the setting.
3. **Persona workloads** (planner and open-loop runs) add a fixed system prompt per
   application type and multi-turn history on purpose. The sizes (0.4K for short bots,
   0.8K chat, 1-1.5K document work, 2K IDE and customer assistants, 4-6K tool-using
   agents, 10-24K for full agent harnesses; `scripts/planner_campaign.py`) are our
   choice, not a measured distribution. Their capacity points carry the window's hit
   rate and are labelled `within_production` or `above_production` against the ceiling
   for their kind: **62%** for chat, API and document work (the highest chat/API figure
   above), **94%** for the long-running agent personas (`agent_task_runner`,
   `coding_agent`, `h_coding_agent`, `h_research_agent`, `h_service_agent`). Mixes are
   judged as chat.
4. **Publication:** every throughput number carries its hit rate and label (the sizing
   export has `prefix_cache_hit_rate` and `prefix_reuse` per curve point and at the
   capacity point, and `meta.prefix_cache` per document). A number labelled
   `above_template` or `above_production` is not published without saying why. Any
   publication states these choices.

Third-party harnesses (AIPerf, GuideLLM, vllm bench) are reported as they ran, with
their settings: dataset size, seed, and whether prefix caching was on.

## Measurement notes

- vLLM's `prefix_cache_hits` and `prefix_cache_queries` counters (checked in the v0.29.0
  image) count only new requests. A preempted request that recomputes is tracked
  separately and does not raise them, so saturated rungs with preemptions are not
  inflated. SGLang publishes only a hit-rate gauge, which the sweep averages; it has not
  been checked for the same.
- gpt-oss's chat template carries the current date. A sweep that runs past midnight
  changes its template prefix once, which lowers that rung's hits slightly; it never
  raises them.
- The template floor is computed with the Hugging Face tokenizer's chat template. For the
  five models checked (gpt-oss-20b/120b, Gemma-4-26B, Qwen3-30B-A3B, Kimi-Linear-48B) its
  templated prompt lengths match the engines' own prompt-token counts in the 2026-10-26
  run exactly (195, 141, 136, 135).
