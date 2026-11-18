"""Fresh paired cache-capacity cells, NUMA-local independent GPU replicas."""

import traceback
import asyncio
import copy
import math
import time
from collections import Counter
from dataclasses import asdict
from pathlib import Path

import yaml

from ..benchmarks.assets import fingerprint, pin_model
from ..benchmarks.process import Lease, hardware_lease_path
from ..benchmarks.storage import read_json, write_json
from ..benchmarks.worker import capture_environment, probe, telemetry
from ..config import load_config
from .deployment import OwnedReplicas, finish_owned_cleanup, launch_owned
from .dynamo import arm_config
from .session_capacity import (
    compare_capacity,
    document_item,
    make_trace,
    padding_repeats,
    prime_documents,
    prompt_token_count,
    replay,
)
from .session_native import monitor_native, queue_delta
from .session_saturation import next_cells
from .sessions import OwnedSessions, tier_counters
from .topology import discover


class SessionReplicas(OwnedReplicas):
    def __init__(self, cfg, spec, arm, owner, inventory):
        super().__init__(cfg, owner)
        self.children = []
        for i, gpu in enumerate(spec.gpu_ids):
            child = copy.deepcopy(cfg)
            child.gpu_device_ids = [gpu]
            child.replica_devices = [[gpu]]
            child.port = cfg.port + i * 100
            cpus = inventory[gpu]["cpus"]
            node = inventory[gpu]["node"]
            child.docker_extra_args = list(child.docker_extra_args) + [
                "--cpuset-cpus",
                ",".join(str(c) for c in cpus),
                "--cpuset-mems",
                str(node),
            ]
            settings = spec.model_copy(update={"gpu_ids": [gpu]})
            if getattr(spec, "session_runtime", "dynamo") == "vllm":
                from .session_vllm import OwnedNativeSessions

                worker = OwnedNativeSessions(child, settings, arm, owner)
            else:
                worker = OwnedSessions(child, settings, arm, owner)
            worker.stopping = self.stopping
            self.children.append(worker)

    async def capture(self, directory, label):
        for i, child in enumerate(self.children):
            p = directory / f"replica-{i}"
            p.mkdir(parents=True, exist_ok=True)
            await child.capture_metrics(p, label)


def cache_evidence(directory, arm, backend="kvbm", prefix=""):
    # Prefix selects mature-session counters separately from initial context loading.
    rows = []
    for p in sorted(directory.glob("replica-*")):
        pairs = {}
        for label in ["before", "after"]:
            counters = {}
            for f in p.glob(prefix + label + "-worker-*.prom"):
                for line in f.read_text().splitlines():
                    if not line or line.startswith("#"):
                        continue
                    key, value = line.rsplit(" ", 1)
                    if any(
                        k in key
                        for k in [
                            "prefix_cache_hits_total",
                            "prefix_cache_queries_total",
                            "num_preemptions_total",
                            "prompt_tokens_total",
                        ]
                    ):
                        value = float(value)
                        if not math.isfinite(value) or value < 0:
                            raise ValueError("Invalid worker counter")
                        counters[f.name.split("-worker-", 1)[1] + ":" + key] = value
            pairs[label] = counters
        a, b = pairs["before"], pairs["after"]
        if not a or set(a) != set(b) or any(b[k] < a[k] for k in a):
            raise RuntimeError("Missing or reset worker cache counters")
        row = dict(replica=p.name, arm=arm, worker_deltas={k: b[k] - a[k] for k in a})
        if arm == "treatment":
            row["cpu_backend"] = backend
        if arm == "treatment" and backend == "simple_cpu":
            row["cpu_backend"] = backend
            row["external_matched_tokens"] = sum(
                v
                for k, v in row["worker_deltas"].items()
                if "external_prefix_cache_hits_total" in k
            )
            before = {r["pid"]: r for r in read_json(p / (prefix + "before-native-cpu.json"))}
            after = {r["pid"]: r for r in read_json(p / (prefix + "after-native-cpu.json"))}
            if not before or set(before) != set(after):
                raise RuntimeError("Native CPU worker identity changed")
            deltas = {
                k: sum(after[pid][k] - before[pid][k] for pid in before)
                for k in ("load_blocks", "store_blocks")
            }
            if any(after[pid][k] < before[pid][k] for pid in before for k in deltas):
                raise RuntimeError("Native CPU completion counters reset")
            row["completed_copy_deltas"] = deltas
            row["cpu_reload_observed"] = deltas["load_blocks"] > 0
            row["verification_scope"] = "Completed physical block copies, not distinct documents"
        elif arm == "treatment":
            a = tier_counters((p / (prefix + "before-kvbm.prom")).read_text())
            b = tier_counters((p / (prefix + "after-kvbm.prom")).read_text())
            if not a or set(a) != set(b) or any(b[k] < a[k] for k in a):
                raise RuntimeError("Missing or reset CPU tier counters")
            row["tier_deltas"] = {k: b[k] - a[k] for k in a}
            row["cpu_reload_observed"] = any(
                "onboard_blocks_h2d" in k and v > 0 for k, v in row["tier_deltas"].items()
            )
        rows.append(row)
    if not rows:
        raise RuntimeError("Missing replica evidence")
    write_json(directory / "cache-evidence.json", rows)
    return rows


async def run(path, manifest, spec):
    from .report import export

    state = read_json(path / "state.json")
    assets = None
    manifest_spec = spec  # padding repeats are resolved per cell from the frozen spec

    def update(phase, **kw):
        state.update(phase=phase, updated_at=time.time(), **kw)
        write_json(path / "state.json", state)
        write_json(path / "comparison.json", compare_capacity(state["results"], spec))

    try:
        with Lease(hardware_lease_path()):
            # Each client stream holds a socket; the default 1024 soft limit would
            # fail an 8192-stream replay mid-measurement.
            from .token_search import ensure_file_limit

            state["file_limit"] = ensure_file_limit(spec.concurrency)
            inventory = await asyncio.to_thread(discover, spec.gpu_ids)
            write_json(path / "topology.json", inventory)
            # The configured CPU cache is per GPU replica, not a shared allocation.
            if spec.session_cpu_gb * len(spec.gpu_ids) + 16 > next(
                int(line.split()[1]) / 1024**2
                for line in Path("/proc/meminfo").read_text().splitlines()
                if line.startswith("MemAvailable:")
            ):
                raise RuntimeError("Insufficient host memory for all CPU cache replicas")
            for node, count in Counter(inventory[g]["node"] for g in spec.gpu_ids).items():
                lines = (
                    Path(f"/sys/devices/system/node/node{node}/meminfo").read_text().splitlines()
                )

                def node_value(name, records=lines):
                    return next(
                        (int(line.split()[-2]) for line in records if f"{name}:" in line), 0
                    )

                # Linux node files have no MemAvailable. Use free plus reclaimable file cache,
                # exclude shared memory, and keep an explicit reserve; global MemAvailable is checked above.
                free_kib = node_value("MemFree") + max(
                    0, node_value("FilePages") - node_value("Shmem")
                )
                if free_kib / 1024**2 < spec.session_cpu_gb * count + 8:
                    raise RuntimeError(
                        f"NUMA node {node} lacks free memory for CPU caches plus reserve; reduce per-replica cache or release memory"
                    )
            pending = list(manifest["schedule"])
            refinements = {}
            while pending:
                cell = pending.pop(0)
                update("preflight", current=cell)
                directory = (
                    path
                    / f"block-{cell['block']:02d}-{cell['arm']}-users-{cell['population']}-rate-{cell['rate']:g}"
                )
                if cell.get("compaction"):
                    directory = directory.with_name(directory.name + f"-compaction-{cell['compaction']}")
                directory.mkdir()
                cfgdoc = arm_config(manifest["configuration"], spec, cell["arm"])
                cfgdoc["engine"]["max_num_seqs"] = spec.session_engine_max_seqs
                (directory / "config.yaml").write_text(yaml.safe_dump(cfgdoc))
                cfg = load_config(directory / "config.yaml")
                current = await asyncio.to_thread(fingerprint, cfg.engine)
                if not current.get("snapshot") or not current.get("image"):
                    raise ValueError("Pinned cached model and image required")
                if assets is not None and assets != current:
                    raise ValueError("Assets changed between paired cells")
                from .session_calibration import load_calibration

                calibration = load_calibration(spec, current, cfgdoc)
                if calibration:
                    frozen = (
                        read_json(path / "capacity-calibration.json")
                        if (path / "capacity-calibration.json").exists()
                        else None
                    )
                    if frozen is not None and frozen != calibration:
                        raise ValueError("Capacity calibration changed between cells")
                    write_json(path / "capacity-calibration.json", calibration)
                assets = current
                write_json(directory / "assets.json", assets)
                write_json(
                    directory / "environment.json",
                    await asyncio.to_thread(capture_environment, cfg),
                )
                cfg.engine.gpu_image = assets["image"]["id"]
                pin_model(cfg.engine, assets["snapshot"])
                (directory / "launch-resolved.yaml").write_text(yaml.safe_dump(asdict(cfg)))
                paced = spec.session_pacing != "open"
                # Paced users have no frozen arrival trace; one event per user
                # still drives the tokenizer preflight for every document.
                trace = (
                    [
                        dict(at_s=0.0, slot=slot, generation=0, turn=1)
                        for slot in range(cell["population"])
                    ]
                    if paced
                    else make_trace(spec, cell["population"], cell["rate"], cell["block"])
                )
                # Exact tokenizer preflight, including chat formatting and output headroom.
                from transformers import AutoTokenizer

                tokenizer = await asyncio.to_thread(
                    AutoTokenizer.from_pretrained, assets["snapshot"], local_files_only=True
                )
                if spec.session_document_padding:
                    repeats = await asyncio.to_thread(
                        padding_repeats, tokenizer, manifest_spec, cell["block"]
                    )
                    spec = manifest_spec.model_copy(
                        update={"session_document_pad_repeats": repeats}
                    )

                async def tokenize_trace(events, encoder, block):
                    # One tokenization per document, not per request: questions on a
                    # document differ only in a short suffix, and a 96K-token document
                    # replayed thousands of times would otherwise take hours to check.
                    first = {}
                    for e in events:
                        first.setdefault((e["slot"], e["generation"]), e)
                    lengths = []
                    for i, e in enumerate(first.values()):
                        # Record IDs are fixed width, so question suffixes differ by a
                        # few tokens at most; the slack covers that difference.
                        prompt = document_item(spec, e, block)["prompt"]
                        lengths.append(prompt_token_count(encoder, prompt) + 32)
                        if i % 16 == 0:
                            await asyncio.sleep(0)  # Honor cancellation during a large preflight.
                    return lengths

                if spec.session_workload == "knowledge_worker":
                    from .knowledge_workers import make_session, turn_item

                    example = await asyncio.to_thread(
                        make_session, tokenizer, spec, 0, cell["block"]
                    )
                    lengths = [turn_item(example, tokenizer, spec, 0)["input_tokens"]]
                else:
                    lengths = await tokenize_trace(trace, tokenizer, cell["block"])
                if not lengths or max(lengths) + spec.max_output_tokens > spec.context_limit:
                    raise ValueError(
                        "Trace is empty or tokenized document exceeds context/output budget"
                    )
                write_json(
                    directory / "token-lengths.json",
                    dict(
                        min=min(lengths),
                        max=max(lengths),
                        count=len(lengths),
                        pad_repeats=spec.session_document_pad_repeats,
                    ),
                )
                engine = SessionReplicas(cfg.engine, spec, cell["arm"], path.name, inventory)
                sampler = None
                power = None
                try:
                    if spec.session_power_env:
                        from .wall_power import sample_power

                        power = asyncio.create_task(
                            sample_power(spec.session_power_env, directory / "power.csv")
                        )
                    update("launching", current=cell)
                    await launch_owned(engine, directory)
                    await probe(engine, 60)

                    # Readiness probe uses unrelated content. All document cold starts are measured.
                    async def monitor(active_engine, active_dir):
                        tasks = [
                            asyncio.create_task(
                                telemetry(active_dir, active_engine, 4 * 1024**3, time.monotonic())
                            ),
                            asyncio.create_task(monitor_native(active_engine, active_dir)),
                        ]
                        try:
                            done, _ = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
                            for task in done:
                                await task
                            raise RuntimeError("Monitoring stopped unexpectedly")
                        finally:
                            for task in tasks:
                                task.cancel()
                            await asyncio.gather(*tasks, return_exceptions=True)

                    sampler = asyncio.create_task(monitor(engine, directory))
                    await engine.capture(directory, "before")
                    update("measuring", current=cell)
                    if spec.session_prime_documents:
                        await prime_documents(
                            engine, spec, cell["population"], cell["block"], directory
                        )
                    if spec.session_workload == "knowledge_worker":
                        from .knowledge_workers import replay_workers

                        work = replay_workers(
                            engine,
                            spec,
                            cell["population"],
                            cell["block"],
                            directory,
                            tokenizer,
                            cell.get("compaction", "none"),
                        )
                    elif paced:
                        from .session_paced import replay_paced

                        work = replay_paced(
                            engine, spec, cell["population"], cell["block"], directory
                        )
                    else:
                        work = replay(engine, spec, trace, cell["block"], directory)
                    measurement = asyncio.create_task(work)
                    try:
                        done, _ = await asyncio.wait(
                            [measurement, sampler], return_when=asyncio.FIRST_COMPLETED
                        )
                        if sampler in done:
                            await sampler
                            raise RuntimeError("Telemetry stopped before measurement completed")
                        metrics = await measurement
                        write_json(directory / "metrics.json", metrics)
                    finally:
                        measurement.cancel()
                        await asyncio.gather(measurement, return_exceptions=True)
                    await engine.capture(directory, "after")
                    if spec.session_prime_documents:
                        await engine.capture(directory, "steady-after")
                    metrics.update(
                        queue_delta(
                            directory,
                            "steady-"
                            if spec.session_workload == "knowledge_worker"
                            or spec.session_prime_documents
                            else "",
                        )
                    )
                    evidence = cache_evidence(
                        directory,
                        cell["arm"],
                        spec.session_cpu_backend,
                        "steady-"
                        if spec.session_workload == "knowledge_worker"
                        or spec.session_prime_documents
                        else "",
                    )
                    metrics["cpu_reload_observed"] = any(
                        r.get("cpu_reload_observed") for r in evidence
                    )
                    from .session_saturation import cache_work_metrics

                    metrics.update(cache_work_metrics(evidence, metrics.get("correct", 0)))
                    write_json(directory / "metrics.json", metrics)
                    state["results"].append(
                        dict(cell, metrics=metrics, evidence=str(directory.relative_to(path)))
                    )
                    update("measured", current=cell)
                    if paced:
                        try:
                            from .session_cells import export_cells

                            await asyncio.to_thread(export_cells, path)
                        except Exception as exc:  # the export is re-runnable from evidence
                            state["cells_export_error"] = str(exc)
                    if not pending and spec.session_workload != "knowledge_worker":
                        pending = next_cells(state["results"], spec, refinements)
                        state.setdefault("adaptive_schedule", []).extend(pending)
                        state["refinement_rounds"] = refinements
                        update("scheduled", current=None)
                finally:
                    for task in (sampler, power):
                        if task:
                            task.cancel()
                            await asyncio.gather(task, return_exceptions=True)
                    await finish_owned_cleanup([engine])
            comparison = compare_capacity(state["results"], spec)
            resolved = all(g["saturation_resolved"] for g in comparison.get("by_working_set", []))
            update(
                "complete",
                current=None,
                outcome="needs_review" if resolved else "saturation_unresolved",
            )
    except asyncio.CancelledError:
        update("cancelled", current=None, outcome="incomplete")
        raise
    except Exception as exc:
        update("failed", current=None, error=str(exc), traceback=traceback.format_exc(), outcome="incomplete")
    finally:
        try:
            await asyncio.to_thread(export, path, compare_capacity(state["results"], spec))
        except Exception as exc:
            update("failed", error="Export failed: " + str(exc))
    return state
