"""Compact replay dataset for the P02 KV-offload demo. Standard library only.

Run on the XE7740, where the campaign evidence lives:

    python3 p02_demo_export.py OUT_DIR CAMPAIGN_ID[:-SUFFIX] [...]

Writes OUT_DIR/index.json (one summary row per finished step) and one
OUT_DIR/<arm>-<users>.json per step with per-request, per-second and per-GPU
series. Campaigns without cells.json (still running) are skipped, so the
script can be rerun as later steps finish.

Every request gets a `source` label, which the raw evidence does not carry:
  gpu      the conversation was still in GPU memory
  host     it was reloaded from host memory (tier arm only)
  rebuilt  the GPU reprocessed it from the text
GPU-only arm: from the server's cached-token count for the request.
Tier arm: from the replica's host-reload counter across the request's wait;
where another request on the same replica overlaps, from the first-token time.
"""

import bisect
import json
import pathlib
import sys

RUNS = pathlib.Path("/data/system-sizing/runs/infrastructure")
FAMILIES = ("prefix_cache_queries_total", "prefix_cache_hits_total", "external_prefix_cache_hits_total")
GAUGES = ("num_requests_running", "num_requests_waiting", "kv_cache_usage_perc")
ARM = {"baseline": "gpu_only", "treatment": "xeon_tier"}


def lines(path):
    if path.exists():
        with path.open() as handle:
            for line in handle:
                if line.strip():
                    yield json.loads(line)


def r(value, places=3):
    return None if value is None else round(value, places)


def replica_series(directory, started_at):
    """Per replica: sample times, cumulative counters, gauges, host-cache fill."""
    out = {}
    for sample in lines(directory / "native-queue-cache.jsonl"):
        at = sample["timestamp"] - started_at
        pool = {p["replica"]: p for p in sample.get("cpu_pool") or []}
        for rep in sample.get("samples", []):
            s = out.setdefault(rep["replica"], dict(t=[], q=[], g=[], c=[], run=[], wait=[], kv=[], fill=[]))
            v = dict.fromkeys(FAMILIES + GAUGES, 0.0)
            for key, value in rep.get("values", {}).items():
                family = key.split("vllm:", 1)[-1].split("{", 1)[0]
                if family in FAMILIES or family in GAUGES:
                    v[family] = value  # one engine per replica
            s["t"].append(at)
            s["q"].append(v[FAMILIES[0]])
            s["g"].append(v[FAMILIES[1]])
            s["c"].append(v[FAMILIES[2]])
            s["run"].append(v["num_requests_running"])
            s["wait"].append(v["num_requests_waiting"])
            s["kv"].append(v["kv_cache_usage_perc"])
            p = pool.get(rep["replica"])
            s["fill"].append(p["cached_blocks"] / p["total_blocks"] if p and p["total_blocks"] else None)
    return out


def label_requests(rows, series, arm, replicas):
    """Add replica and source to each request row; return label statistics."""
    by_replica = {}
    for row in rows:
        row["replica"] = row["session"] % replicas
        by_replica.setdefault(row["replica"], []).append(row)
    stats = dict(counter=0, by_wait=0, agree=0, compared=0)
    # First-token time that separates a GPU hit from a host reload: midpoint of the
    # two medians among requests the counters label unambiguously.
    clean = {"gpu": [], "host": []}
    pending = []
    for rep, group in by_replica.items():
        s = series.get(rep)
        for row in group:
            cached = (((row.get("usage") or {}).get("prompt_tokens_details") or {}).get("cached_tokens")) or 0
            prompt = (row.get("usage") or {}).get("prompt_tokens") or 0
            row["cached"], row["prompt"] = cached, prompt
            # Answer speed: output tokens over the time from first token to the end.
            # About 112 tokens/s when the GPU is free; a few tokens/s when every
            # engine step also carries a slice of another user's rebuild.
            row["output"] = (row.get("usage") or {}).get("completion_tokens") or 0
            span = (row.get("elapsed_s") or 0) - (row.get("ttft_s") or 0)
            row["answer_tok_s"] = row["output"] / span if row.get("ttft_s") is not None and span > 0 else None
            if arm == "gpu_only" or not s:
                row["source"] = "gpu" if prompt and cached >= 0.9 * prompt else "rebuilt"
                continue
            if row.get("ttft_s") is None:
                row["source"] = "rebuilt" if not (prompt and cached >= 0.9 * prompt) else None
                continue
            a, b = row["arrival_s"], row["arrival_s"] + row["ttft_s"]
            alone = not any(o is not row and o["arrival_s"] < b + 2 and o["finish_s"] > a - 2 for o in group)
            i = max(0, bisect.bisect_right(s["t"], a) - 1)
            j = min(len(s["t"]) - 1, bisect.bisect_left(s["t"], b + 0.5))
            host = s["c"][j] - s["c"][i]
            row["_host_delta"] = host
            if prompt and cached < 0.9 * prompt:
                row["source"] = "rebuilt"
            elif alone:
                row["source"] = "host" if host >= 0.5 * prompt else "gpu"
                clean[row["source"]].append(row["ttft_s"])
                stats["counter"] += 1
            else:
                pending.append(row)
    if arm == "xeon_tier":
        med = {k: sorted(v)[len(v) // 2] if v else None for k, v in clean.items()}
        cut = (med["gpu"] + med["host"]) / 2 if med["gpu"] and med["host"] else 0.45
        stats.update(gpu_first_token_p50_s=r(med["gpu"]), host_first_token_p50_s=r(med["host"]), cut_s=r(cut))
        for row in pending:
            row["source"] = "host" if row["ttft_s"] >= cut else "gpu"
            stats["by_wait"] += 1
        for name in ("gpu", "host"):
            for wait in clean[name]:
                stats["compared"] += 1
                stats["agree"] += ("host" if wait >= cut else "gpu") == name
    return stats


def gpu_starts(rows, series):
    """Estimate when each request got onto its GPU (the raw evidence has no such time).

    The engine serves each GPU's requests oldest first and reports how many it is
    working on. At every sample the N oldest outstanding requests on a replica
    are taken as the N the engine reports running; a request's start is the
    first sample where it is one of them. Requests that never reach the GPU
    before the deadline keep None.
    """
    by_replica = {}
    for row in rows:
        row["gpu_start_s"] = None
        by_replica.setdefault(row["replica"], []).append(row)
    for rep, group in by_replica.items():
        s = series.get(rep)
        if not s:
            continue
        group.sort(key=lambda x: x["arrival_s"])
        first = 0
        for at, running in zip(s["t"], s["run"]):
            while first < len(group) and group[first]["finish_s"] <= at:
                first += 1
            taken = 0
            for row in group[first:]:
                if row["arrival_s"] > at or taken >= running:
                    break
                if row["finish_s"] <= at:
                    continue
                taken += 1
                if row["gpu_start_s"] is None:
                    row["gpu_start_s"] = max(at, row["arrival_s"])
    for row in rows:
        # A measured first token proves the request was on the GPU by then.
        if row.get("ttft_s") is not None:
            seen = row["arrival_s"] + row["ttft_s"]
            if row["gpu_start_s"] is None or row["gpu_start_s"] > seen:
                row["gpu_start_s"] = row["arrival_s"] if row.get("source") != "rebuilt" else seen
    reached = [x for x in rows if not x.get("transport_ok") and x["gpu_start_s"] is not None]
    lost = sorted(x["finish_s"] - x["gpu_start_s"] for x in reached)
    return dict(
        gave_up=sum(not x.get("transport_ok") for x in rows),
        gave_up_on_gpu=len(reached),
        gave_up_gpu_seconds_p50=r(lost[len(lost) // 2], 1) if lost else None,
        gave_up_gpu_seconds_total=r(sum(lost), 0),
    )


def lru_check(rows, replicas):
    """How well 'the last N conversations on each GPU stay in GPU memory' predicts the labels."""
    out = {}
    for capacity in range(6, 13):
        recent, right, total = {i: [] for i in range(replicas)}, 0, 0
        for row in sorted(rows, key=lambda x: x["arrival_s"]):
            if row.get("source") is None:
                continue
            order = recent[row["replica"]]
            predicted = "gpu" if row["session"] in order else "miss"
            actual = "gpu" if row["source"] == "gpu" else "miss"
            right += predicted == actual
            total += 1
            if row["session"] in order:
                order.remove(row["session"])
            order.append(row["session"])
            del order[:-capacity]
        out[capacity] = r(right / total) if total else None
    return out


def export_cell(campaign, cell, out, suffix=""):
    directory = RUNS / campaign / pathlib.Path(cell["timeline"]).parent
    metrics = json.loads((directory / "metrics.json").read_text())
    started_at = metrics["started_at"]
    replicas = cell["gpus"]
    rows = list(lines(directory / "requests.jsonl"))
    series = replica_series(directory, started_at)
    stats = label_requests(rows, series, cell["arm"], replicas)
    lru = lru_check(rows, replicas)
    stats["gpu_start"] = gpu_starts(rows, series)

    requests = [
        [row["session"], row["turn"], row["replica"], r(row["arrival_s"], 2), r(row.get("ttft_s")),
         r(row.get("elapsed_s")), r(row.get("finish_s"), 2), int(bool(row.get("correct"))),
         int(bool(row.get("transport_ok"))), row["cached"], row["prompt"], row.get("source"),
         row["output"], r(row["answer_tok_s"], 1), r(row["gpu_start_s"], 1)]
        for row in sorted(rows, key=lambda x: x["arrival_s"])
    ]
    priming = [
        [row["session"], r(row["started_at"] - started_at, 2), r(row.get("ttft_s")), r(row.get("elapsed_s"))]
        for row in lines(directory / "priming-requests.jsonl")
    ]
    power = []
    for line in (directory / "power.csv").read_text().splitlines():
        parts = line.split(",")
        try:
            power.append([r(float(parts[0]) - started_at, 1), float(parts[1])])
        except (ValueError, IndexError):
            continue
    gpus = []
    for sample in lines(directory / "telemetry.jsonl"):
        devices = sample.get("gpu_devices") or []
        gpus.append([
            r(sample["timestamp"] - started_at, 1),
            sample.get("host_cpu_pct"),
            [d.get("sm_util_pct") for d in devices],
            [r(d.get("power_w"), 0) for d in devices],
        ])
    per_replica = {
        str(rep): dict(
            t=[r(x, 1) for x in s["t"]], running=s["run"], waiting=s["wait"],
            gpu_cache_used=[r(x) for x in s["kv"]], host_cache_used=[r(x, 4) for x in s["fill"]],
            gpu_hit_tokens=s["g"], host_hit_tokens=s["c"], query_tokens=s["q"],
        )
        for rep, s in sorted(series.items())
    }
    think = cell.get("pacing") == "think"
    length = 2 * metrics["measured_window_s"] if think else metrics["measured_window_s"]
    scored_from = length / 2 if think else 0
    sources = {}
    scored = {}
    for row in rows:
        sources[row.get("source")] = sources.get(row.get("source"), 0) + 1
        if row["arrival_s"] >= scored_from:
            scored[row.get("source")] = scored.get(row.get("source"), 0) + 1
    name = f"{cell['arm']}-{cell['users']}"
    # The plain name is the 96K, 5-minute think step; anything else says so.
    if cell.get("context_tokens") != 96000:
        name += f"-{round(cell['context_tokens'] / 1000)}k"
    if not think:
        name += f"-{cell['pacing']}"
    elif cell.get("think_time_s") != 300:
        name += f"-think{round(cell['think_time_s'])}"
    name += suffix
    doc = dict(
        summary={k: v for k, v in cell.items() if k != "timeline"},
        window=dict(length_s=r(length, 1), scored_from_s=r(scored_from, 1),
                    deadline_s=cell["wait"]["censored_at_s"], zero="measurement start; priming is negative time"),
        columns=dict(
            requests=["session", "turn", "replica", "arrival_s", "first_token_s", "answer_s", "finish_s",
                      "correct", "completed", "cached_tokens", "prompt_tokens", "source",
                      "output_tokens", "answer_tokens_per_s", "gpu_start_s"],
            priming=["session", "start_s", "first_token_s", "answer_s"],
            power=["t_s", "wall_w"],
            gpus=["t_s", "host_cpu_pct", "gpu_busy_pct[8]", "gpu_power_w[8]"],
        ),
        labels=dict(method=stats, source_counts=sources, source_counts_scored_window=scored,
                    lru_capacity_agreement=lru),
        requests=requests, priming=priming, power=power, gpus=gpus, replicas=per_replica,
        timeline=json.loads((RUNS / campaign / cell["timeline"]).read_text()),
    )
    (out / f"{name}.json").write_text(json.dumps(doc, separators=(",", ":")))
    row = dict(doc["summary"], file=f"{name}.json", source_counts_scored_window=scored,
               label_method=stats, lru_capacity_agreement=lru)
    return row


def main():
    out = pathlib.Path(sys.argv[1])
    out.mkdir(parents=True, exist_ok=True)
    index = []
    for campaign in sys.argv[2:]:
        campaign, _, suffix = campaign.partition(":")  # ID or ID:-suffix (for repeats)
        path = RUNS / campaign / "cells.json"
        if not path.exists():
            print("skip (no cells.json yet)", campaign)
            continue
        for cell in json.loads(path.read_text()):
            index.append(export_cell(campaign, cell, out, suffix))
            last = index[-1]
            print(last["file"], last["source_counts_scored_window"], last["label_method"], last["lru_capacity_agreement"])
    index.sort(key=lambda x: (x["file"].split("-", 2)[2:], x["users"], x["arm"]))
    (out / "index.json").write_text(json.dumps(index, indent=1))


if __name__ == "__main__":
    main()
