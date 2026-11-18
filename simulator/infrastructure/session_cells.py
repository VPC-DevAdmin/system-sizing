"""Flat per-cell export for the situation tool: one record per measured cell.

Everything here is derived from evidence already in the cell directory, so a
finished campaign can be re-exported without touching the server:

    python -m simulator.infrastructure.session_cells /path/to/campaign [situation]

Token counters come from the engine's native prefix-cache families. Under a deep
queue the engine counts a waiting request's prefix query on every scheduling
attempt, so the rebuilt share is only reliable in cells without timeouts; the
record carries the timeout share so a reader can tell.
"""

import json
import sys
from pathlib import Path

from ..benchmarks.storage import read_json, write_json
from .driver import percentile

BUCKET_S = 10
FAMILIES = (
    "prefix_cache_queries_total",
    "prefix_cache_hits_total",
    "external_prefix_cache_hits_total",
)


def _lines(path):
    if not path.exists():
        return
    with path.open() as handle:
        for line in handle:
            if line.strip():
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    continue  # a sampler killed mid-write leaves one partial line


def _counter_totals(sample):
    totals = dict.fromkeys(FAMILIES, 0.0)
    for replica in sample.get("samples", []):
        for key, value in replica.get("values", {}).items():
            family = key.split("vllm:", 1)[-1].split("{", 1)[0]
            if family in totals:
                totals[family] += value
    return totals


def _mean(values):
    values = [v for v in values if v is not None]
    return sum(values) / len(values) if values else None


def _power(directory, begin, end):
    """Wall power samples (epoch, watts) inside the window, from idrac_power.py."""
    out = []
    path = directory / "power.csv"
    if path.exists():
        for line in path.read_text().splitlines():
            parts = line.split(",")
            try:
                at, watts = float(parts[0]), float(parts[1])
            except (ValueError, IndexError):
                continue  # failed samples are written with an empty watts field
            if begin <= at <= end:
                out.append((at, watts))
    return out


def timeline(directory, started_at, length_s):
    """Ten-second buckets from the measurement start (wall clock)."""
    n = max(1, int(length_s // BUCKET_S) + 1)
    buckets = [
        dict(
            at_s=i * BUCKET_S,
            asked=0,
            answered=0,
            waiting=0,
            first_token_p50_s=None,
            gpu_hit_tokens=None,
            host_hit_tokens=None,
            rebuilt_tokens=None,
            host_cache_used_fraction=None,
            host_evicted_blocks=None,
            gpu_busy_pct=None,
            host_cpu_busy_pct=None,
            wall_power_w=None,
        )
        for i in range(n)
    ]

    def index(seconds):
        return min(n - 1, max(0, int(seconds // BUCKET_S)))

    waits = [[] for _ in range(n)]
    for row in _lines(directory / "requests.jsonl"):
        i = index(row["arrival_s"])
        buckets[i]["asked"] += 1
        if row.get("ttft_s") is not None:
            waits[i].append(row["ttft_s"])
        if row.get("transport_ok"):
            buckets[index(row["finish_s"])]["answered"] += 1
        for j in range(i, index(row["finish_s"])):
            buckets[j]["waiting"] += 1  # still outstanding at the end of bucket j
    for i, values in enumerate(waits):
        buckets[i]["first_token_p50_s"] = percentile(values, 0.5)

    last, pool_first = {}, None
    for sample in _lines(directory / "native-queue-cache.jsonl"):
        at = sample["timestamp"] - started_at
        if at < 0 or at > length_s + BUCKET_S:
            if at < 0:
                last["counters"] = _counter_totals(sample)
                pool_first = sum(r.get("evicted_blocks", 0) for r in sample.get("cpu_pool", []))
            continue
        b = buckets[index(at)]
        totals = _counter_totals(sample)
        if "counters" in last:
            q, g, c = (totals[k] - last["counters"][k] for k in FAMILIES)
            if min(q, g, c) >= 0:
                b["gpu_hit_tokens"] = (b["gpu_hit_tokens"] or 0) + g
                b["host_hit_tokens"] = (b["host_hit_tokens"] or 0) + c
                b["rebuilt_tokens"] = (b["rebuilt_tokens"] or 0) + max(0, q - g - c)
        last["counters"] = totals
        pool = sample.get("cpu_pool") or []
        if pool:
            total = sum(r["total_blocks"] for r in pool)
            b["host_cache_used_fraction"] = (
                sum(r["cached_blocks"] for r in pool) / total if total else None
            )
            b["host_evicted_blocks"] = sum(r["evicted_blocks"] for r in pool) - (pool_first or 0)

    gpu, cpu = [[] for _ in range(n)], [[] for _ in range(n)]
    for sample in _lines(directory / "telemetry.jsonl"):
        at = sample["timestamp"] - started_at
        if 0 <= at <= length_s:
            devices = sample.get("gpu_devices") or []
            gpu[index(at)].append(_mean([d.get("sm_util_pct") for d in devices]))
            cpu[index(at)].append(sample.get("host_cpu_pct"))
    watts = [[] for _ in range(n)]
    for at, value in _power(directory, started_at, started_at + length_s):
        watts[index(at - started_at)].append(value)
    for i, b in enumerate(buckets):
        b["gpu_busy_pct"] = _mean(gpu[i])
        b["host_cpu_busy_pct"] = _mean(cpu[i])
        b["wall_power_w"] = _mean(watts[i])
    return buckets


def cell_record(result, spec, directory, campaign_id, situation=None):
    m = result["metrics"]
    waits = m.get("waits") or {}
    started = m.get("started_at")
    length = max((r["finish_s"] for r in _lines(directory / "requests.jsonl")), default=0)
    series = timeline(directory, started, length) if started else []
    write_json(directory / "timeline.json", series)
    # Think cells score the final half by arrival; machine means use the same span.
    if spec.get("session_pacing") == "think":
        begin_s = spec["duration_s"] / 2
        end_s = spec["duration_s"]
    else:
        begin_s, end_s = 0, length
    scored = [b for b in series if begin_s <= b["at_s"] < end_s]
    correct = m.get("final_window_completions") or 0
    power = _mean([b["wall_power_w"] for b in scored])
    work = m.get("cache_work") or {}
    answered = work.get("prefix_query_tokens") and m.get("correct")
    pool = [b for b in series if b["host_cache_used_fraction"] is not None]
    return dict(
        situation=situation or spec.get("session_situation"),
        campaign_id=campaign_id,
        block=result["block"],
        arm="gpu_only" if result["arm"] == "baseline" else "xeon_tier",
        users=result["population"],
        context_tokens=spec.get("session_context_tokens"),
        pacing=spec.get("session_pacing", "open"),
        think_time_s=spec.get("session_think_s") if spec.get("session_pacing") == "think" else None,
        host_cache_gib_per_gpu=spec.get("session_cpu_gb") if result["arm"] == "treatment" else 0,
        gpus=len(spec.get("gpu_ids", [])),
        wait=dict(
            first_token_p50_s=waits.get("first_token_p50_s"),
            first_token_p95_s=waits.get("first_token_p95_s"),
            answer_p50_s=waits.get("answer_p50_s"),
            answer_p95_s=waits.get("answer_p95_s"),
            within_first_token=waits.get("within_first_token"),
            within_answer=waits.get("within_answer"),
            censored_at_s=waits.get("censored_at_s"),
        ),
        work=dict(
            questions=waits.get("questions"),
            answered_correctly_per_min=m["sustained_correct_requests_s"] * 60
            if m.get("sustained_correct_requests_s") is not None
            else None,
            correct_fraction=waits.get("correct_fraction"),
            timeout_fraction=waits.get("timeout_fraction"),
        ),
        cache=dict(
            # Whole measurement including drain, per answered question.
            gpu_hit_tokens_per_question=work["gpu_hit_tokens"] / m["correct"] if answered else None,
            host_hit_tokens_per_question=work["cpu_hit_tokens"] / m["correct"]
            if answered
            else None,
            rebuilt_tokens_per_question=work["uncached_prefix_tokens"] / m["correct"]
            if answered
            else None,
            counters_valid=work.get("accounting_valid"),
            host_cache_used_fraction_peak=max(
                (b["host_cache_used_fraction"] for b in pool), default=None
            ),
            host_evicted_blocks=pool[-1]["host_evicted_blocks"] if pool else None,
            host_reload_observed=m.get("cpu_reload_observed"),
        ),
        machine=dict(
            gpu_busy_pct=_mean([b["gpu_busy_pct"] for b in scored]),
            host_cpu_busy_pct=_mean([b["host_cpu_busy_pct"] for b in scored]),
            wall_power_w=power,
            watt_hours_per_correct_answer=power * (end_s - begin_s) / 3600 / correct
            if power and correct and end_s > begin_s
            else None,
        ),
        surges=m.get("surges"),
        timeline="/".join([result["evidence"], "timeline.json"]),
    )


def export_cells(path, situation=None):
    path = Path(path)
    manifest, state = read_json(path / "manifest.json"), read_json(path / "state.json")
    cells = [
        cell_record(r, manifest["spec"], path / r["evidence"], path.name, situation)
        for r in state["results"]
    ]
    write_json(path / "cells.json", cells)
    return cells


if __name__ == "__main__":
    out = export_cells(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else None)
    print(f"{len(out)} cells written")
