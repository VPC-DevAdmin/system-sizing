"""How much prefix-cache reuse a Capsim measurement may contain.

A benchmark earns cache hits only from reuse a real deployment would see
(docs/prefix-cache-policy.md):

* every request repeats the model's chat template and any fixed system
  prompt, so those tokens are fair game;
* a multi-turn session resends its own history, so that is fair too;
* different users sending the same prompt text is not: production traces
  show cross-user reuse is "extremely low" (Wang et al., USENIX ATC '25).

Headline sweeps send unique user text behind a system prompt every user
shares, sized so the reuse lands on ``simulation.prefix_reuse`` (default: the
published business-API level; ``resolve_reuse``, ``choose_system_tokens``).
``template_floor`` works out the reuse that design gives for a model;
``headline_reuse`` flags a rung whose measured hit rate sits above it. Persona workloads add system prompts and history on
purpose; ``workload_reuse`` flags one above the range production traces
reach.
"""

from __future__ import annotations

# Measured reuse, as a share of prompt tokens served from cache:
#   Aliyun Tongyi, infinite cache: 62% consumer chat, 54% business API
#     (Wang et al., "KVCache Cache in the Wild", USENIX ATC '25, Fig. 4)
#   Mooncake public traces: 40% conversation, 59% tool & agent
#     (Yuan et al., "DualMap", ICLR '26, Table 1, on Qin et al.'s traces)
#   Long-running coding agents, which resend a growing history each turn:
#     92.2-94.2% on Inferact's Codex / SWE-bench Pro traces (vLLM blog,
#     2026-05-06), 94% on a production Claude Code trace (FlatCake,
#     arXiv 2610.38355)
# The ATC paper notes synthetic workloads report over 80%, above anything
# its chat and API traffic reaches.
PRODUCTION_REUSE_CEILING = {"chat": 0.62, "agent": 0.94}

# What a headline sweep aims for unless told otherwise
# (``simulation.prefix_reuse``). A saturation sweep is single-turn API-style
# traffic, so the default is the published figure for exactly that: 54% of
# prompt tokens served from cache on Aliyun's business API traffic. The other
# named levels are the other published chat figures. Each sweep reaches its
# level the way production does: a system prompt every user shares, then user
# text nobody else sends.
REUSE_LEVELS = {
    "industry": (0.54, "business API traffic, 54% (Wang et al., KVCache Cache "
                       "in the Wild, USENIX ATC '25, Fig. 4)"),
    "consumer_chat": (0.62, "consumer chat, 62% (Wang et al., USENIX ATC '25, "
                            "Fig. 4)"),
    "conversation": (0.40, "multi-turn conversation, 40% (Yuan et al., DualMap, "
                           "ICLR '26, Table 1, on Mooncake's traces)"),
    # Unique prompts and no system prompt: only the chat template repeats.
    "template": (None, "chat template only (no shared system prompt)"),
    # No reuse at all: no shared system prompt, and the engine's prefix
    # cache switched off at launch (``engine_cache_off_flags``), so not
    # even the chat template is served from cache. A sensitivity setting,
    # not a production level: every deployment reuses its template.
    "off": (0.0, "prefix caching disabled in the engine; no shared system prompt"),
}
# A share of 0 means the same thing as "off".
OFF = "off"
DEFAULT_REUSE = "industry"
# A shared-prompt sweep keeps at least this many unique user tokens.
MIN_UNIQUE_USER_TOKENS = 32


def resolve_reuse(setting) -> dict:
    """``simulation.prefix_reuse`` as {setting, target, source,
    within_production}. Accepts a named level or a share of prompt tokens
    (0.4, "0.4" or "40%"); raises ValueError otherwise. ``setting`` is the
    canonical string, which also identifies the level in a roofline cell."""
    if setting is None or setting == "":
        setting = DEFAULT_REUSE
    if isinstance(setting, str) and setting.strip().lower() in REUSE_LEVELS:
        name = setting.strip().lower()
        target, source = REUSE_LEVELS[name]
        return {"setting": name, "target": target, "source": source,
                "within_production": True}
    try:
        if isinstance(setting, bool):
            raise ValueError
        text = str(setting).strip()
        value = float(text[:-1]) / 100 if text.endswith("%") else float(text)
    except ValueError:
        raise ValueError(
            f"simulation.prefix_reuse must be one of {sorted(REUSE_LEVELS)} or "
            f"a share between 0 and 0.95, got {setting!r}") from None
    if value > 1:
        value /= 100  # 54 means 54%
    if not 0 <= value <= 0.95:
        raise ValueError(
            f"simulation.prefix_reuse must be between 0 and 0.95, got {setting!r}")
    if value == 0:
        # Zero reuse is zero: template-only reuse is its own named
        # setting ("template"), and used to be what 0 silently meant.
        return resolve_reuse(OFF)
    ceiling = PRODUCTION_REUSE_CEILING["chat"]
    return {"setting": f"{value:.2f}", "target": round(value, 4),
            "source": "chosen for this run; published chat and API traffic "
                      f"reaches {ceiling:.0%} at most",
            "within_production": value <= ceiling}


def choose_system_tokens(floor_at, max_system: int, target: float | None,
                         step: int = 8) -> int:
    """Length of the shared system prompt (a multiple of ``step``, at most
    ``max_system``) whose reuse ``floor_at(n)`` lands closest to ``target``.
    ``floor_at`` rises with n and returns None when it cannot tell (no chat
    template, or one that refuses a system message): then 0, as for no
    target or a template that already reaches it."""
    if target is None or max_system < step:
        return 0
    base = floor_at(0)
    if base is None or base >= target:
        return 0
    lo, hi = 0, max_system // step        # floor(lo) < target
    top = floor_at(hi * step)
    if top is None:
        return 0
    if top < target:
        return hi * step
    while hi - lo > 1:                    # floor(hi) >= target
        mid = (lo + hi) // 2
        f = floor_at(mid * step)
        if f is None:
            return 0
        if f >= target:
            hi = mid
        else:
            lo = mid
    below, above = floor_at(lo * step), floor_at(hi * step)
    if below is None or above is None:
        return 0
    return (lo if target - below <= above - target else hi) * step


# Persona workloads that are long-running agents (many turns over a large
# shared tool prefix); everything else is judged as chat, API or document
# work. Mixes count as chat.
AGENT_WORKLOADS = frozenset({
    "agent_task_runner", "coding_agent", "h_coding_agent",
    "h_research_agent", "h_service_agent",
})

# A headline rung may exceed its template floor by this much before it is
# flagged: engines count hits per block, the first request on each replica
# misses, and chat templates add a few variable tokens.
TEMPLATE_TOLERANCE = 0.05

# vLLM caches full 16-token blocks by default; SGLang's radix cache matches
# token by token.
VLLM_BLOCK_TOKENS = 16


def block_tokens(engine_type: str | None) -> int:
    return 1 if engine_type and "sglang" in engine_type else VLLM_BLOCK_TOKENS


def template_floor(tokenizer, texts: list[str], block: int,
                   system_prompt: str | None = None) -> float | None:
    """Share of a headline prompt the engine can legitimately serve from
    cache: the prefix two of the sweep's own (unique) user messages share
    once templated (with the run's fixed system prompt, if it has one),
    rounded down to whole cache blocks, over the full templated prompt.
    Returns None without a usable chat template.
    """
    if tokenizer is None or not getattr(tokenizer, "chat_template", None):
        return None
    if len(texts) < 2:
        return None
    head = [{"role": "system", "content": system_prompt}] if system_prompt else []
    try:
        ids = [tokenizer.apply_chat_template(
            head + [{"role": "user", "content": t}], tokenize=True,
            add_generation_prompt=True) for t in texts[:2]]
    except Exception:  # noqa: BLE001
        return None
    # Newer transformers return a BatchEncoding (a mapping, not a dict).
    ids = [list(x["input_ids"]) if hasattr(x, "keys") else list(x) for x in ids]
    shared = 0
    for a, b in zip(*ids, strict=False):
        if a != b:
            break
        shared += 1
    shared -= shared % max(1, block)
    total = len(ids[0])
    return round(shared / total, 4) if total else None


def headline_reuse(hit_rate: float | None, floor: float | None) -> str | None:
    """'template' if a rung's reuse is what its chat template explains,
    'above_template' if prompts were reused beyond it, None if unknown."""
    if hit_rate is None or floor is None:
        return None
    return "above_template" if hit_rate > floor + TEMPLATE_TOLERANCE else "template"


def workload_class(workload_id: str | None) -> str:
    return "agent" if workload_id in AGENT_WORKLOADS else "chat"


def workload_reuse(hit_rate: float | None, kind: str = "chat") -> str | None:
    """'within_production' or 'above_production' for a workload that
    reuses system prompts and history on purpose, against the ceiling for
    its kind ('chat' or 'agent'); None if unknown."""
    if hit_rate is None:
        return None
    ceiling = PRODUCTION_REUSE_CEILING[kind]
    return "above_production" if hit_rate > ceiling else "within_production"


def cache_off(setting) -> bool:
    """Does this reuse setting ask for the engine's prefix cache off?"""
    try:
        return resolve_reuse(setting)["setting"] == OFF
    except ValueError:
        return False


def engine_cache_off_flags(engine_type: str) -> list[str]:
    """Launch flags that disable ``engine_type``'s prefix cache, or [] for
    an engine capsim has no switch for (the run then records whatever the
    engine served from cache, and ``headline_reuse`` flags it)."""
    if engine_type in ("vllm", "vllm_cuda", "vllm_cuda_multi", "vllm_dual_socket"):
        return ["--no-enable-prefix-caching"]
    if engine_type == "sglang_cuda":
        return ["--disable-radix-cache"]
    return []
