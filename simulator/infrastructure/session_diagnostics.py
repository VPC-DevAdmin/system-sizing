"""Controlled correctness fixtures; fixed records/questions as context grows.

Run against an explicitly configured endpoint, once per pinned KV precision and
cache setting. This module does not alter or start serving processes.
"""

import argparse
import asyncio
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import httpx

from ..benchmarks.storage import write_json
from .driver import request_one
from .knowledge_workers import token_count
from .workload import build_corpus


def fixture(tokenizer, target, seed, question_index=0):
    records = build_corpus(1, 16, seed, answer_records=1)[0]
    rows = [
        json.loads(line) for line in records["prompt"].split("\nDocument:\n", 1)[1].splitlines()
    ]
    selected = [rows[question_index % 16], rows[(question_index + 7) % 16]]
    prefix = f"Workspace {seed}. Reference records:\n" + "\n".join(json.dumps(r) for r in rows)
    question = "\nReturn only a JSON array with every field of records, in order: " + json.dumps(
        [r["record_id"] for r in selected]
    )
    note = "\nBackground note: this workspace is used for document review. Preserve the reference records."

    def messages(n):
        return [{"role": "user", "content": prefix + note * n + question}]

    lo, hi = 0, target
    if token_count(tokenizer, messages(0)) > target:
        raise ValueError("Target too small for fixed fixture")
    while lo < hi:
        mid = (lo + hi) // 2
        if token_count(tokenizer, messages(mid)) < target:
            lo = mid + 1
        else:
            hi = mid
    msg = messages(lo)
    count = token_count(tokenizer, msg)
    if not target <= count <= target + 64:
        raise ValueError("Fixture target missed")
    return dict(
        id=f"{seed}-{target}-{question_index}",
        prompt=msg[0]["content"],
        messages=msg,
        expected=selected,
        input_tokens=count,
        target_tokens=target,
        prompt_sha256=hashlib.sha256(json.dumps(msg, sort_keys=True).encode()).hexdigest(),
    )


async def run(plan, tokenizer, directory):
    """Preserve every prompt and result; no automatic retries or score filtering."""
    directory.mkdir(parents=True, exist_ok=False)
    write_json(directory / "plan.json", {k: v for k, v in plan.items() if k != "api_key"})
    required = ("model_revision", "runtime_digest", "kv_dtype", "prefix_caching", "model", "url")
    if any(k not in plan for k in required):
        raise ValueError("Missing pinned endpoint metadata")
    if plan.get("mode") not in ("isolated", "reuse", "concurrent"):
        raise ValueError("Unknown diagnosis mode")
    if plan["mode"] == "isolated" and plan["prefix_caching"] is not False:
        raise ValueError(
            "Isolated control requires prefix caching disabled by the launch configuration"
        )
    spec = SimpleNamespace(
        seed=42, request_timeout_s=600, max_output_tokens=1024, structured_output=True
    )
    results = []
    seeds = plan.get("seeds", list(range(42, 52)))
    if len(seeds) < 10 or len(set(seeds)) != len(seeds):
        raise ValueError("Use at least ten distinct fixed seeds")
    concurrency = int(plan.get("concurrency", 4)) if plan["mode"] == "concurrent" else 1
    if not 1 <= concurrency <= 16:
        raise ValueError("Diagnostic concurrency outside 1..16")
    sem = asyncio.Semaphore(concurrency)
    async with httpx.AsyncClient(timeout=600) as client:

        async def send(item, attempt):
            async with sem:
                result = await request_one(
                    client,
                    plan["url"],
                    plan["model"],
                    item,
                    spec,
                    f"{item['id']}-{attempt}",
                    plan.get("api_key", "EMPTY"),
                )
                result.update(
                    fixture_id=item["id"], target_tokens=item["target_tokens"], attempt=attempt
                )
                results.append(result)
                with (directory / "requests.jsonl").open("a") as out:
                    out.write(json.dumps(result, allow_nan=False) + "\n")

        for target in (8192, 32768, 65536, 96000):
            items = [fixture(tokenizer, target, seed) for seed in seeds]
            for item in items:
                write_json(directory / (item["id"] + ".json"), item)
            for attempt in range(2 if plan["mode"] == "reuse" else 1):
                await asyncio.gather(*(send(item, attempt) for item in items))
    summary = []
    for target in (8192, 32768, 65536, 96000):
        rows = [r for r in results if r["target_tokens"] == target]
        summary.append(
            dict(
                context_tokens=target,
                attempted=len(rows),
                correct=sum(bool(r.get("transport_ok") and r.get("correct")) for r in rows),
            )
        )
    write_json(
        directory / "summary.json",
        dict(
            qualified=False,
            cells=summary,
            scope="Correctness diagnosis only; endpoint metadata must be verified against captured launch configuration",
        ),
    )
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("plan", type=Path)
    parser.add_argument("tokenizer")
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True)
    asyncio.run(run(json.loads(args.plan.read_text()), tokenizer, args.output))


if __name__ == "__main__":
    main()
