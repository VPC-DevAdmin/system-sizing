"""Owned P03 cache grid execution; every failed cell stays in the evidence."""

import asyncio
import time
from dataclasses import asdict

import yaml

from ..benchmarks.assets import fingerprint, pin_model
from ..benchmarks.process import Lease, hardware_lease_path
from ..benchmarks.storage import read_json, write_json
from ..benchmarks.worker import capture_environment, probe, telemetry
from ..config import load_config
from .cache import cache_config, cells, compare_cache, observe_runtime
from .capabilities import inspect_host
from .deployment import OwnedReplicas, finish_owned_cleanup, launch_owned
from .driver import capture_metrics, measure, measure_finite
from .report import export
from .workload import build_corpus


async def run(path, manifest, spec):
    state = read_json(path / "state.json")

    def update(phase, **kwargs):
        state.update(phase=phase, updated_at=time.time(), **kwargs)
        write_json(path / "state.json", state)

    comparison = {"qualified": False, "reason": "Cache experiment incomplete", "metrics": {}}
    engine = None
    sampler = None
    try:
        with Lease(hardware_lease_path()):
            update("preflight")
            write_json(path / "host.json", await asyncio.to_thread(inspect_host))
            corpora = read_json(path / "corpus.json")
            assets = None
            for arm in manifest["schedule"]:
                directory = path / f"block-{arm['block']:02d}-{arm['arm']}"
                directory.mkdir()
                update("preparing", current=arm)
                launch_spec = spec.model_copy(update={"concurrency": max(spec.cache_concurrencies)})
                config_doc = cache_config(manifest["configuration"], launch_spec, arm["arm"])
                (directory / "config.yaml").write_text(yaml.safe_dump(config_doc))
                cfg = load_config(directory / "config.yaml")
                current = await asyncio.to_thread(fingerprint, cfg.engine)
                if not current.get("snapshot") or not current.get("image"):
                    raise ValueError(
                        "Cache test requires locally cached, fingerprinted model and runtime"
                    )
                if assets is not None and current != assets:
                    raise ValueError("Assets changed between cache arms")
                assets = current
                write_json(directory / "assets.json", assets)
                write_json(
                    directory / "environment.json",
                    await asyncio.to_thread(capture_environment, cfg),
                )
                cfg.engine.gpu_image = assets["image"]["id"]
                pin_model(cfg.engine, assets["snapshot"])
                (directory / "launch-resolved.yaml").write_text(yaml.safe_dump(asdict(cfg)))
                engine = OwnedReplicas(cfg.engine, owner=path.name)
                try:
                    update("launching", current=arm)
                    await launch_owned(engine, directory)
                    await probe(engine, 60)
                    observations = []
                    for logfile in sorted(directory.glob("replica-*/engine.log")):
                        observations.append(
                            dict(
                                replica=logfile.parent.name,
                                **observe_runtime(logfile.read_text(errors="replace"), arm["arm"]),
                            )
                        )
                    if len(observations) != len(spec.gpu_ids):
                        raise ValueError(
                            "Missing effective cache settings from one or more replicas"
                        )
                    write_json(directory / "runtime-observed.json", observations)
                    for records, concurrency in cells(spec, arm["arm"]):
                        cell = dict(arm, document_records=records, concurrency=concurrency)
                        window = directory / f"records-{records}-concurrency-{concurrency}"
                        window.mkdir()
                        settings = spec.model_copy(
                            update={"document_records": records, "concurrency": concurrency}
                        )
                        warmup = build_corpus(spec.corpus_size, records, spec.seed ^ 0xFFFF)
                        write_json(window / "warmup-corpus.json", warmup)
                        sampler = asyncio.create_task(
                            telemetry(window, engine, 4 * 1024**3, time.monotonic())
                        )
                        measurement = None
                        try:
                            if spec.warmup_s:
                                update("warming", current=cell)
                                await measure(
                                    engine, warmup, settings, window / "warmup.jsonl", spec.warmup_s
                                )
                            await capture_metrics(engine, window, "before")
                            update("measuring", current=cell)
                            measurement = asyncio.create_task(
                                measure_finite(
                                    engine,
                                    corpora[str(records)][: spec.cache_samples],
                                    settings,
                                    window / "requests.jsonl",
                                )
                            )
                            done, _ = await asyncio.wait(
                                [measurement, sampler], return_when=asyncio.FIRST_COMPLETED
                            )
                            if sampler in done:
                                await sampler
                                raise RuntimeError("Telemetry stopped during cache measurement")
                            metrics = await measurement
                            await capture_metrics(engine, window, "after")
                            write_json(window / "metrics.json", metrics)
                            state["results"].append(dict(cell, metrics=metrics))
                            update("measuring", current=cell)
                        finally:
                            if measurement:
                                measurement.cancel()
                                await asyncio.gather(measurement, return_exceptions=True)
                            sampler.cancel()
                            await asyncio.gather(sampler, return_exceptions=True)
                            sampler = None
                finally:
                    if engine:
                        update("cleaning", current=arm)
                        await finish_owned_cleanup([engine])
                        engine = None
            comparison = compare_cache(state["results"], spec)
            write_json(path / "comparison.json", comparison)
            update("complete", current=None, outcome="needs_review")
    except asyncio.CancelledError:
        update("cancelled", current=None, outcome="incomplete")
        raise
    except Exception as exc:
        update("failed", current=None, error=str(exc), outcome="incomplete")
    finally:
        try:
            await asyncio.to_thread(export, path, comparison)
        except Exception as exc:
            update("failed", error=f"Evidence export failed: {exc}", outcome="incomplete")
