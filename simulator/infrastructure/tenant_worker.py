"""Dedicated, software-shared and MIG placement for a fixed two-service set."""

import asyncio
import time
from dataclasses import asdict

import yaml

from ..benchmarks.assets import fingerprint, pin_model
from ..benchmarks.process import Lease, hardware_lease_path
from ..benchmarks.storage import read_json, write_json
from ..benchmarks.worker import assert_gpu_idle, capture_environment, probe, telemetry
from ..config import load_config
from .capabilities import inspect_host
from .deployment import OwnedReplicas, finish_owned_cleanup, launch_owned
from .driver import capture_metrics, measure
from .partition import Partition
from .tenants import measure_tenant


def config(configuration, spec, arm, devices):
    from .worker import arm_config

    cfg = arm_config(configuration, spec, "baseline")
    e = cfg["engine"]
    e["replica_devices"] = [[d] for d in devices]
    e["gpu_device_ids"] = devices
    e["tensor_parallel_size"] = 1
    e["type"] = "vllm_cuda_multi"
    # The shared physical GPU splits its memory budget; explicit KV bytes keep
    # each service's cache allocation equal across all three placements.
    e["gpu_memory_utilization"] = 0.45 if arm == "shared" else 0.95
    flags = e["vllm_extra_flags"]
    clean = []
    i = 0
    while i < len(flags):
        key = flags[i].split("=", 1)[0]
        if key in ("--gpu-memory-utilization", "--kv-cache-memory-bytes"):
            if "=" not in flags[i]:
                i += 1
        else:
            clean.append(flags[i])
        i += 1
    e["vllm_extra_flags"] = clean + ["--kv-cache-memory-bytes", str(spec.tenant_kv_gib * 1024**3)]
    return cfg


def compare_tenants(rows, spec):
    summaries = {}
    for arm, gpus in [("dedicated", 2), ("shared", 1), ("mig", 1)]:
        cells = [r for r in rows if r["arm"] == arm]
        expected = {
            (b, phase, t)
            for b in range(spec.paired_blocks)
            for phase in ("steady", "noisy")
            for t in range(2)
        }
        keys = {(r["block"], r["phase"], r["tenant"]) for r in cells}
        if len(keys) != len(cells):
            raise ValueError("Duplicate tenant cell")
        steady = [r for r in cells if r["phase"] == "steady"]
        all_pass = keys == expected and all(r["metrics"]["gates_passed"] for r in steady)
        summaries[arm] = {
            "physical_gpus": gpus,
            "steady_contract_passed": all_pass,
            "services_per_gpu": 2 / gpus if all_pass else None,
            "per_tenant_results": cells,
        }
    return {
        "qualified": False,
        "reason": "Exploratory fixed service-set placement; not maximum tenant density",
        "scope": "Two identical independent endpoints at fixed offered load; software sharing is a separate comparator",
        "placements": summaries,
        "metrics": {},
    }


async def run(path, manifest, spec):
    from .report import export

    state = read_json(path / "state.json")

    def update(phase, **values):
        state.update(phase=phase, updated_at=time.time(), **values)
        write_json(path / "state.json", state)

    comparison = {"qualified": False, "metrics": {}, "reason": "Tenant experiment incomplete"}
    partition = None
    engine = None
    sampler = None
    try:
        with Lease(hardware_lease_path()):
            update("preflight")
            write_json(path / "host.json", await asyncio.to_thread(inspect_host))
            from types import SimpleNamespace

            await asyncio.to_thread(
                assert_gpu_idle, SimpleNamespace(cfg=SimpleNamespace(type="vllm_cuda"))
            )
            corpus = read_json(path / "corpus.json")
            assets = None
            from .workload import build_corpus

            warmup = build_corpus(spec.corpus_size, spec.document_records, spec.seed ^ 65535)
            for cell in manifest["schedule"]:
                folder = path / f"block-{cell['block']:02d}-{cell['arm']}"
                folder.mkdir()
                selected = spec.gpu_ids if cell["arm"] == "dedicated" else [spec.gpu_ids[-1]] * 2
                try:
                    if cell["arm"] == "mig":
                        update("partitioning", current=cell)
                        partition = Partition(
                            folder / "partition.json", path.name, spec.gpu_ids[-1]
                        )
                        task = asyncio.create_task(asyncio.to_thread(partition.create))
                        try:
                            selected = await asyncio.shield(task)
                        except BaseException:
                            try:
                                await task
                            except Exception:
                                pass
                            raise
                    cfg_doc = config(manifest["configuration"], spec, cell["arm"], selected)
                    (folder / "config.yaml").write_text(yaml.safe_dump(cfg_doc))
                    cfg = load_config(folder / "config.yaml")
                    current = await asyncio.to_thread(fingerprint, cfg.engine)
                    if not current.get("snapshot") or not current.get("image"):
                        raise ValueError("Tenant assets must be cached and fingerprinted")
                    if assets is not None and assets != current:
                        raise ValueError("Tenant assets changed")
                    assets = current
                    write_json(folder / "assets.json", assets)
                    write_json(
                        folder / "environment.json",
                        await asyncio.to_thread(capture_environment, cfg),
                    )
                    cfg.engine.gpu_image = assets["image"]["id"]
                    pin_model(cfg.engine, assets["snapshot"])
                    (folder / "launch-resolved.yaml").write_text(yaml.safe_dump(asdict(cfg)))
                    engine = OwnedReplicas(cfg.engine, owner=path.name)
                    update("launching", current=cell)
                    await launch_owned(engine, folder)
                    await probe(engine, 60)
                    sampler = asyncio.create_task(
                        telemetry(folder, engine, 4 * 1024**3, time.monotonic())
                    )
                    for phase in ("steady", "noisy"):
                        update("measuring", current=dict(cell, phase=phase))
                        window = folder / phase
                        window.mkdir()
                        if spec.warmup_s:
                            await asyncio.gather(
                                *(
                                    measure(
                                        child,
                                        warmup,
                                        spec,
                                        window / f"warmup-{t}.jsonl",
                                        spec.warmup_s,
                                    )
                                    for t, child in enumerate(engine.children)
                                )
                            )
                        await capture_metrics(engine, window, "before")
                        start_at = time.monotonic() + 0.2

                        async def measurement(
                            window=window, start_at=start_at, phase=phase, engine=engine
                        ):
                            return await asyncio.gather(
                                *(
                                    measure_tenant(
                                        child,
                                        corpus,
                                        spec,
                                        window / f"tenant-{t}.jsonl",
                                        start_at,
                                        offset=t * spec.corpus_size // 2,
                                        rate=spec.tenant_requests_per_second
                                        * (
                                            spec.tenant_noise_multiplier
                                            if phase == "noisy" and t == 1
                                            else 1
                                        ),
                                    )
                                    for t, child in enumerate(engine.children)
                                )
                            )

                        job = asyncio.create_task(measurement())
                        try:
                            done, _ = await asyncio.wait(
                                [job, sampler], return_when=asyncio.FIRST_COMPLETED
                            )
                            if sampler in done:
                                await sampler
                                raise RuntimeError("Telemetry stopped")
                            measured = await job
                        finally:
                            job.cancel()
                            await asyncio.gather(job, return_exceptions=True)
                        await capture_metrics(engine, window, "after")
                        for tenant, metrics in enumerate(measured):
                            state["results"].append(
                                dict(cell, phase=phase, tenant=tenant, metrics=metrics)
                            )
                        write_json(window / "metrics.json", measured)
                        update("measuring", current=dict(cell, phase=phase))
                finally:
                    await finish_owned_cleanup([engine], sampler, partition)
                    sampler = engine = partition = None
            comparison = compare_tenants(state["results"], spec)
            write_json(path / "comparison.json", comparison)
            update("complete", current=None, outcome="needs_review")
    except asyncio.CancelledError:
        update("cancelled", current=None, outcome="incomplete")
        raise
    except Exception as exc:
        update("failed", error=str(exc), current=None, outcome="incomplete")
    finally:
        try:
            await asyncio.to_thread(export, path, comparison)
        except Exception as exc:
            update("failed", error="Evidence export failed: " + str(exc), outcome="incomplete")
