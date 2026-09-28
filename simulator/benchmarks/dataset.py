"""One immutable prompt corpus for both harnesses; no network during measurement."""

from __future__ import annotations

import json
import random
from pathlib import Path

from .storage import sha256, write_json


def build_dataset(tokenizer_path: str, spec: dict, directory: Path) -> dict:
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(
        tokenizer_path, local_files_only=True, trust_remote_code=False
    )
    rng = random.Random(spec["seed"])
    words = (
        "system model capacity memory compute network latency throughput request response "
        "engine server performance explain compare describe analyze training inference "
        "document question answer architecture benchmark measurement context hardware"
    ).split()
    rows = []
    counts = []
    target = spec["input_tokens"]
    for _ in range(spec["dataset_size"]):
        text = " ".join(rng.choices(words, k=target * 2))
        ids = tok.encode(text, add_special_tokens=False)[:target]
        text = tok.decode(ids, skip_special_tokens=True)
        counts.append(len(tok.encode(text, add_special_tokens=False)))
        rows.append(
            {
                "text": text,
                "input_tokens_count": counts[-1],
                "output_tokens_count": spec["output_tokens"],
            }
        )
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "prompts.jsonl"
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    meta = {
        "sha256": sha256(path),
        "rows": len(rows),
        "seed": spec["seed"],
        "tokenizer": tokenizer_path,
        "input_tokens_min": min(counts),
        "input_tokens_max": max(counts),
        "chat_template_overhead_included": False,
        "description": "Seeded synthetic single-turn text; same corpus in both harnesses. "
        "Prompt length excludes the serving engine's chat template. "
        "Warm cache may reuse prompts; not production traffic replay.",
    }
    write_json(directory / "dataset.json", meta)
    return meta
