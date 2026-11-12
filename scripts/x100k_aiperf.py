#!/usr/bin/env python3
"""AIPerf check of the 100,000 tokens/s white paper on the XE7740.

Reuses the P11 survey driver (p11_us_survey.py, same directory) at the paper's
workload: 128 input / 256 output tokens with ignore_eos, vLLM with one copy of
the model per GPU, AIPerf round-robin over the eight copies. Only the settings
below differ from P11; launch, load, merging and recording are P11's.

    x100k_aiperf.py list
    x100k_aiperf.py sweep --no-quality <config>...
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import p11_us_survey as p  # noqa: E402

# The 2026-10-26 run (100 recycled prompts per process, 90-98.5% prefix-cache
# hits) stays in /data/capsim/x100k/results; this design writes beside it.
p.OUT = Path(os.environ.get("X100K_OUT", "/data/capsim/x100k/results-shared-prompt"))
p.ISL, p.OSL = 128, 256  # the white paper's shape
p.LADDER = [256, 512, 1024, 2048, 4096, 8192]
p.SCREEN_CONCURRENCY = 4096
# Prompts are short, so a small context leaves the most memory for KV.
p.MAX_MODEL_LEN = 4096
# One AIPerf process per copy: 8,192 streaming users is ~400 requests/s and
# ~100k SSE tokens/s, more than four processes should be asked to parse.
p.SHARDS = 8
# 1,024 running requests per copy, so 8 copies can hold the 8,192 rung.
p.max_seqs = lambda c: 1024

FP8_KV = ["--kv-cache-dtype", "fp8"]  # as in the paper's configurations
MODELS = {
    "gemma-4-26b": "nvidia/Gemma-4-26B-A4B-NVFP4",
    "gpt-oss-20b": "openai/gpt-oss-20b",
    "qwen3-30b-a3b-fp8": "Qwen/Qwen3-30B-A3B-Instruct-2507-FP8",
    "qwen3-4b": "Qwen/Qwen3-4B-Instruct-2507",
    "gpt-oss-120b": "openai/gpt-oss-120b",
    "kimi-linear-48b": "Firworks/Kimi-Linear-48B-A3B-Instruct-nvfp4",
}
# Repos whose tokenizer is custom code (Kimi's tiktoken tokenizer). Reviewed
# 2026-10-26: Moonshot's tokenizer, no network or subprocess use; vLLM runs its
# own Kimi-Linear model code and only needs the config and tokenizer.
TRUST_REMOTE_CODE = {"kimi-linear-48b"}
# Kimi-Linear's MLA layers run on TRITON_MLA, the only MLA backend on SM120;
# with an FP8 KV cache its decode kernel asks for 102,400 B of shared memory
# against a 101,376 B limit and the engine dies at startup. Default KV works.
DEFAULT_KV = {"kimi-linear-48b"}
p.CONFIGS = {
    f"{k}-vllm": p.cfg(m, "vllm", 1, 8,
                       args=([] if k in DEFAULT_KV else FP8_KV)
                       + (["--trust-remote-code"] if k in TRUST_REMOTE_CODE else []))
    for k, m in MODELS.items()
}

# Prefix cache on, at a rate an enterprise chat/API deployment would see (54-62%,
# docs/prefix-cache-policy.md): a system prompt shared by every user, then user
# text that never repeats, together the paper's 128 input tokens. Split per model
# so the shared part (chat template + system prompt, in vLLM's 16-token blocks)
# is about 55% of the prompt; worked out 2026-11-12 from each model's tokenizer.
# gpt-oss needs a shorter system prompt because its template already shares 64
# tokens. Kimi-Linear cannot prefix-cache. (system tokens, expected hit rate)
SHARED_PROMPT = {
    "gemma-4-26b": (80, 0.548),        # 80 of 146 prompt tokens
    "qwen3-30b-a3b-fp8": (80, 0.567),  # 80 of 141
    "qwen3-4b": (80, None),            # not worked out
    "gpt-oss-20b": (48, 0.552),        # 112 of 203
    "gpt-oss-120b": (48, 0.552),
    "kimi-linear-48b": (80, 0.0),
}
for _k, (_system, _expected) in SHARED_PROMPT.items():
    p.CONFIGS[f"{_k}-vllm"].update(
        unique_prompts=True, system_tokens=_system, expected_hit_rate=_expected)


def trust_aiperf_tokenizer():
    """AIPerf loads the tokenizer too; pass it the same permission, only for
    the configs that need it, through a shim in place of the binary."""
    shim = p.OUT.parent / "aiperf-trust-remote-code.sh"
    shim.parent.mkdir(parents=True, exist_ok=True)
    shim.write_text(f'#!/bin/sh\nexec {p.AIPERF} "$@" --tokenizer-trust-remote-code\n')
    shim.chmod(0o755)
    p.AIPERF = str(shim)


if __name__ == "__main__":
    if any(a.removesuffix("-vllm") in TRUST_REMOTE_CODE for a in sys.argv[2:]):
        trust_aiperf_tokenizer()
    p.main()
