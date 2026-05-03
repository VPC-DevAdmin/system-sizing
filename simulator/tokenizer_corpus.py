"""Generates filler text targeting specific token counts.

Builds a small lazy-loaded corpus of paragraphs and concatenates / truncates
to land within tolerance of a target token count. Falls back to a whitespace
heuristic (~0.75 tokens/word) when the model's tokenizer can't be loaded.
"""

from __future__ import annotations

import random
from functools import lru_cache

# Lorem-ipsum-style paragraphs, deliberately varied so prefix-cache behaviour
# isn't dominated by any single repeated chunk.
_BASE_PARAGRAPHS = [
    "The system distributes workload across heterogeneous compute resources, balancing latency and throughput per request type.",
    "Field engineers reported intermittent stalls during peak load, traced back to oversubscription of memory bandwidth on socket one.",
    "Quarterly capacity planning revisited the cohort assumptions; the drafter persona had drifted upward in average input length.",
    "Document retrieval ran in parallel with summarisation, with the orchestration layer enforcing a strict per-tenant concurrency budget.",
    "The release candidate showed regressions only on long-context workloads, suggesting the prefix cache eviction policy needed revisiting.",
    "Operations reviewed the incident report, focusing on the cascading queue growth observed once mean think-time fell below ten seconds.",
    "Compiler optimisations reordered the AMX tiles to better overlap with weight loads, lifting effective utilisation by a measurable margin.",
    "Customer success surfaced renewal risk for accounts whose usage profile had migrated from chat-heavy toward document-heavy patterns.",
    "Telemetry confirmed memory bandwidth saturation preceded TTFT degradation by roughly twenty seconds across all measured runs.",
    "Engineering documented the new launch recipe, noting the exact thread-binding string required for deterministic numa placement.",
]


def _heuristic_token_count(text: str) -> int:
    # Rough approximation: english text ~0.75 tokens/word
    words = max(1, len(text.split()))
    return int(round(words / 0.75))


@lru_cache(maxsize=4)
def _load_tokenizer(model_id: str):
    try:
        from transformers import AutoTokenizer
        return AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    except Exception:
        return None


class TokenCorpus:
    """Generate prompt text near a target token count for a given model."""

    def __init__(self, model_id: str):
        self.model_id = model_id
        self._tokenizer = _load_tokenizer(model_id)

    def count(self, text: str) -> int:
        if self._tokenizer is None:
            return _heuristic_token_count(text)
        try:
            return len(self._tokenizer.encode(text, add_special_tokens=False))
        except Exception:
            return _heuristic_token_count(text)

    def make_text(self, target_tokens: int, rng: random.Random) -> str:
        target_tokens = max(1, int(target_tokens))
        # Build by repeating shuffled paragraphs until we exceed target,
        # then trim from the end at word boundaries.
        chunks: list[str] = []
        current_tokens = 0
        guard = 0
        while current_tokens < target_tokens and guard < 1000:
            para = rng.choice(_BASE_PARAGRAPHS)
            chunks.append(para)
            current_tokens += self.count(para) + 1  # newline
            guard += 1
        text = "\n".join(chunks)
        # Trim if overshot
        if self._tokenizer is not None:
            try:
                ids = self._tokenizer.encode(text, add_special_tokens=False)
                if len(ids) > target_tokens:
                    ids = ids[:target_tokens]
                    text = self._tokenizer.decode(ids, skip_special_tokens=True)
            except Exception:
                pass
        else:
            words = text.split()
            est_words = int(target_tokens * 0.75)
            if len(words) > est_words:
                text = " ".join(words[:est_words])
        return text
