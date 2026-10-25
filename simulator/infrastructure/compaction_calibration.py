"""Phase 0: can the Xeon summarizer keep up with the compaction rate?

Runs against an explicitly configured summarizer endpoint (no GPUs). It builds
synthetic raw transcripts of the stage 2 shape, compacts them at several
concurrencies, and reports sustained compactions per second against the rate a
population needs. This module does not start or configure the summarizer.

    python -m simulator.infrastructure.compaction_calibration plan.json /path/to/tokenizer out/

plan.json: {"url": ..., "model": ..., "raw_tokens": 64000, "budget_tokens": 4096,
            "concurrencies": [1, 4, 8, 16], "population": 96, "cycle_minutes": 8,
            "seed": 42, "timeout_s": 900}
"""

import argparse
import asyncio
import itertools
import json
import random
import time
from pathlib import Path
from types import SimpleNamespace

import httpx

from ..benchmarks.storage import write_json
from .driver import percentile
from .knowledge_workers import make_session, token_count, turn_item
from .session_compaction import compact, empty_ledger


def synthetic_archive(tokenizer, raw_tokens, seed, growth_tokens=6000, context_tokens=4096):
    """Raw turns of the stage 2 shape, with fabricated correct answers, and the true facts."""
    spec = SimpleNamespace(
        seed=seed,
        document_records=16,
        session_context_tokens=context_tokens,
        session_answer_records=2,
        session_new_input_tokens=256,
        session_growth_tokens=growth_tokens,
        session_probe_mix=["reference", "session_recall", "knowledge_update", "abstention"],
        session_limit_policy="fail",
        max_output_tokens=1024,
        context_limit=131072,
        session_active_records=None,
    )
    session = make_session(tokenizer, spec, 0, 0)
    turn = 0
    while True:
        item = turn_item(session, tokenizer, spec, turn)
        answer = json.dumps(item["expected"])
        session["messages"] = item["messages"] + [{"role": "assistant", "content": answer}]
        session["archive"].append(dict(turn=turn, user=item["body"], assistant=answer))
        total = token_count(
            tokenizer,
            [
                {"role": "user", "content": "\n".join(e["user"] for e in session["archive"])},
                {
                    "role": "assistant",
                    "content": "\n".join(e["assistant"] for e in session["archive"]),
                },
            ],
        )
        turn += 1
        if total >= raw_tokens or turn >= 64:
            archive = session["archive"][1:]
            turns = {e["turn"] for e in archive}
            return archive, total, [f for f in session["facts"] if f["turn"] in turns]


def ledger_score(ledger, facts):
    """Facts the ledger states exactly, out of the facts in the summarized turns."""
    requested = {str(f["turn"]): f["record_ids"] for f in facts if f["kind"] == "requested"}
    latest = {}
    for f in sorted((f for f in facts if f["kind"] == "update"), key=lambda f: f["turn"]):
        latest[f["record_id"]] = f["allocation"]
    got_requested = (ledger or {}).get("requested_by_turn", {})
    got_alloc = (ledger or {}).get("working_allocations", {})
    hits = sum(1 for t, ids in requested.items() if got_requested.get(t) == ids)
    hits += sum(1 for rid, v in latest.items() if got_alloc.get(rid) == v)
    extra = sum(1 for t in got_requested if t not in requested) + sum(
        1 for rid in got_alloc if rid not in latest
    )
    return dict(facts=len(requested) + len(latest), exact=hits, invented=extra)


async def run(plan, tokenizer, directory):
    directory.mkdir(parents=True, exist_ok=True)
    spec = SimpleNamespace(
        seed=plan.get("seed", 42),
        session_compaction_model=plan["model"],
        session_compaction_url=plan["url"],
        session_compaction_budget_tokens=plan.get("budget_tokens", 4096),
        session_compaction_timeout_s=plan.get("timeout_s", 900),
        session_compaction_chunk_turns=plan.get("chunk_turns", 3),
    )
    # One distinct transcript per concurrent call: identical prompts would hit the
    # summarizer's prefix cache and overstate sustainable throughput.
    distinct = max(plan.get("concurrencies", [1, 4, 8, 16]))
    archives = [
        synthetic_archive(
            tokenizer,
            plan.get("raw_tokens", 64000),
            spec.seed + i,
            plan.get("growth_tokens", 6000),
            plan.get("context_tokens", 4096),
        )
        for i in range(distinct)
    ]
    archive, raw_tokens, _ = archives[0]
    write_json(
        directory / "transcript-shape.json",
        dict(raw_turns=len(archive), raw_tokens=raw_tokens, distinct_transcripts=distinct),
    )
    results = []
    async with httpx.AsyncClient(timeout=spec.session_compaction_timeout_s) as client:
        for concurrency in plan.get("concurrencies", [1, 4, 8, 16]):
            rng = random.Random(spec.seed + concurrency)
            start = time.monotonic()
            counter = itertools.count()
            rows = await asyncio.gather(
                *(
                    compact(client, spec, archives[i % len(archives)][0], empty_ledger(), counter)
                    for i in range(concurrency)
                )
            )
            elapsed = time.monotonic() - start
            for i, r in enumerate(rows):
                r["ledger_score"] = ledger_score(r["ledger"], archives[i % len(archives)][2])
            ok = [r for r in rows if r["ok"]]
            scores = [r["ledger_score"] for r in rows]
            facts = sum(x["facts"] for x in scores)
            prompt = [(r["usage"] or {}).get("prompt_tokens") for r in ok]
            completion = [(r["usage"] or {}).get("completion_tokens") for r in ok]
            walls = [r["wall_s"] for r in rows]
            cell = dict(
                concurrency=concurrency,
                attempted=len(rows),
                ok=len(ok),
                elapsed_s=elapsed,
                compactions_per_s=len(ok) / elapsed if elapsed else None,
                wall_p50_s=percentile(walls, 0.5),
                wall_p95_s=percentile(walls, 0.95),
                prompt_tokens_mean=sum(prompt) / len(prompt) if prompt and all(prompt) else None,
                completion_tokens_mean=sum(completion) / len(completion)
                if completion and all(completion)
                else None,
                finish_reasons={
                    r["finish_reason"]: sum(
                        1 for x in rows if x["finish_reason"] == r["finish_reason"]
                    )
                    for r in rows
                },
                errors=[r["error"] for r in rows if r["error"]],
                sample_summary=ok[0]["summary"][:2000] if ok else None,
                sample_ledger=ok[0]["ledger"] if ok else None,
                chunk_count=ok[0]["chunk_count"] if ok else None,
                # Failed compactions count as zero exact facts.
                ledger_exact_fraction=sum(x["exact"] for x in scores) / facts if facts else None,
                ledger_invented=sum(x["invented"] for x in scores),
                ledger_dropped_lines=sum(r.get("ledger_dropped_lines", 0) for r in rows),
                nonce=rng.random(),
            )
            results.append(cell)
            write_json(directory / f"concurrency-{concurrency}.json", dict(cell=cell, rows=rows))
    verdict = capacity_verdict(results, plan)
    write_json(directory / "calibration.json", dict(plan=plan, results=results, verdict=verdict))
    (directory / "calibration.md").write_text(render(plan, results, verdict))
    return results, verdict


def capacity_verdict(results, plan):
    population = plan.get("population", 96)
    cycle_s = plan.get("cycle_minutes", 8) * 60
    required = population / cycle_s
    best = max(
        (r for r in results if r["ok"] == r["attempted"] and r["attempted"]),
        key=lambda r: r["compactions_per_s"] or 0,
        default=None,
    )
    sustained = best["compactions_per_s"] if best else 0
    return dict(
        population=population,
        cycle_s=cycle_s,
        required_compactions_per_s=required,
        sustained_compactions_per_s=sustained,
        best_concurrency=best["concurrency"] if best else None,
        headroom=sustained / required if required else None,
        # The plan asks for under half utilization at the baseline population.
        passes=bool(best) and sustained >= 2 * required,
        in_flight_at_required=required * best["wall_p50_s"]
        if best and best["wall_p50_s"]
        else None,
    )


def render(plan, results, verdict):
    lines = [
        "# Summarizer calibration",
        "",
        f"Endpoint {plan['url']} model {plan['model']}; raw input about {plan.get('raw_tokens', 64000)} tokens, budget {plan.get('budget_tokens', 4096)}.",
        "",
        "| Concurrency | OK | Compactions/s | Wall p50 s | Wall p95 s | Prompt tok | Completion tok | Ledger exact | Invented | Dropped |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in results:
        lines.append(
            f"| {r['concurrency']} | {r['ok']}/{r['attempted']} | {r['compactions_per_s'] or 0:.3f} | {r['wall_p50_s'] or 0:.1f} | {r['wall_p95_s'] or 0:.1f} | {r['prompt_tokens_mean'] or 0:.0f} | {r['completion_tokens_mean'] or 0:.0f} | {r['ledger_exact_fraction'] or 0:.3f} | {r['ledger_invented']} | {r['ledger_dropped_lines']} |"
        )
    lines += [
        "",
        f"Required at {verdict['population']} workers, one compaction per {verdict['cycle_s'] / 60:.0f} min: {verdict['required_compactions_per_s']:.3f}/s.",
        f"Sustained (all calls OK): {verdict['sustained_compactions_per_s']:.3f}/s at concurrency {verdict['best_concurrency']}; headroom {verdict['headroom'] or 0:.2f}x; in flight at the required rate about {verdict['in_flight_at_required'] or 0:.1f}.",
        "",
        "Passes (at least 2x the required rate): " + ("yes" if verdict["passes"] else "no") + ".",
        "",
        "Calibration only. It sizes the summarizer; it is not a serving result.",
    ]
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("plan")
    parser.add_argument("tokenizer")
    parser.add_argument("out")
    args = parser.parse_args()
    from transformers import AutoTokenizer

    plan = json.loads(Path(args.plan).read_text())
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True)
    _, verdict = asyncio.run(run(plan, tokenizer, Path(args.out)))
    print(json.dumps(verdict, indent=2))


if __name__ == "__main__":
    main()
