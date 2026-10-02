"""Owned paired deployments. Only one arm runs at a time under the host lease."""

import asyncio
import contextlib
import copy
import json
import os
import signal
import sys
import time
from dataclasses import asdict
from pathlib import Path

import yaml

from ..benchmarks.assets import fingerprint, pin_model
from ..benchmarks.process import Lease, hardware_lease_path
from ..benchmarks.storage import read_json, write_json
from ..benchmarks.worker import capture_environment, probe, telemetry
from ..config import load_config
from .campaign import verify
from .capabilities import inspect_host
from .comparison import compare
from .deployment import OwnedReplicas, finish_owned_cleanup, launch_owned
from .driver import capture_metrics, measure
from .report import export
from .spec import ExperimentSpec
from .workload import build_corpus


def arm_config(configuration, spec, arm):
    if spec.project_id == "P10":
        from .attention import config

        return config(configuration, spec, arm)
    if spec.project_id in ("P01", "P02"):
        from .dynamo import arm_config as dynamo_config

        return dynamo_config(configuration, spec, arm)
    cfg = copy.deepcopy(configuration)
    e = cfg["engine"]
    e.update(
        type="vllm_cuda_multi" if len(spec.gpu_ids) > 1 else "vllm_cuda",
        tensor_parallel_size=1,
        gpu_device_ids=spec.gpu_ids,
        replica_devices=[[g] for g in spec.gpu_ids],
        port=19200,
        max_model_len=spec.context_limit,
        max_num_seqs=max(32, spec.concurrency, *spec.speculative_concurrencies),
    )
    flags = e.get("vllm_extra_flags", [])
    # Do not allow a recorded speculative or prefix-cache override to contaminate A/B.
    forbidden = ("--speculative-config", "--enable-prefix-caching", "--no-enable-prefix-caching")
    if any(any(f == x or f.startswith(x + "=") for x in forbidden) for f in flags):
        raise ValueError("Source has speculative/cache overrides; choose a clean recorded source")
    # Effective engine args, not only dataclass fields, must reflect the selected controls.
    managed = {"--max-num-seqs", "--max-model-len", "--tensor-parallel-size", "--port"}
    cleaned = []
    i = 0
    while i < len(flags):
        flag = flags[i]
        key = flag.split("=", 1)[0]
        if key in managed:
            if "=" not in flag:
                if i + 1 >= len(flags) or flags[i + 1].startswith("--"):
                    raise ValueError("Malformed recorded engine flag: " + key)
                i += 1
        else:
            cleaned.append(flag)
        i += 1
    e["vllm_extra_flags"] = cleaned + [
        "--max-num-seqs",
        str(e["max_num_seqs"]),
        "--no-enable-prefix-caching",
    ]
    if arm == "treatment":
        draft = {
            "method": spec.speculative_method,
            "num_speculative_tokens": spec.speculative_tokens,
        }
        if spec.speculative_method == "ngram":
            draft.update(prompt_lookup_min=2, prompt_lookup_max=5)
        else:
            draft["moe_backend"] = "triton"
        e["vllm_extra_flags"] += ["--speculative-config", json.dumps(draft)]
    return cfg


async def campaign(path):
    manifest = verify(path)
    spec = ExperimentSpec(**manifest["spec"])
    if spec.project_id == "P06":
        if spec.tenant_workflow == "post_call":
            from .call_worker import run

            return await run(path, manifest, spec)
        from .tenant_worker import run

        return await run(path, manifest, spec)
    if spec.project_id == "P02" and spec.session_mode == "capacity":
        from .session_capacity_worker import run

        return await run(path, manifest, spec)
    if spec.project_id == "P05":
        from .diffusion_worker import run

        return await run(path, manifest, spec)
    if spec.project_id == "P03":
        from .cache_worker import run

        return await run(path, manifest, spec)
    state = read_json(path / "state.json")

    def update(phase, **kw):
        state.update(phase=phase, updated_at=time.time(), **kw)
        write_json(path / "state.json", state)

    engine = None
    sampler = None
    comparison = {"qualified": False, "reason": "Experiment incomplete", "metrics": {}}
    try:
        with Lease(hardware_lease_path()):
            update("preflight")
            write_json(path / "host.json", await asyncio.to_thread(inspect_host))
            if spec.project_id == "P01" and (
                spec.p01_placement != "pooled" or spec.p01_placement_grid != ["pooled"]
            ):
                from .topology import discover, pool_plan

                inventory = await asyncio.to_thread(discover, spec.gpu_ids)
                plans = []
                for candidate in manifest["schedule"]:
                    placement = candidate.get("placement", spec.p01_placement)
                    if placement != "pooled":
                        plans.append(
                            dict(
                                candidate=candidate,
                                topology=pool_plan(
                                    spec.gpu_ids,
                                    candidate.get("prefill") or spec.prefill_workers,
                                    candidate["arm"],
                                    placement,
                                    inventory,
                                ),
                            )
                        )
                write_json(path / "topology-preflight.json", plans)
            corpus = read_json(path / "corpus.json")
            warmup = build_corpus(spec.corpus_size, spec.document_records, spec.seed ^ 0xFFFF)
            if spec.p01_goal_seek:
                from .goal_search import shapes

                warmup = {
                    shape["id"]: build_corpus(
                        spec.corpus_size, shape["records"], spec.seed ^ 0xFFFF, shape["answers"]
                    )
                    for shape in shapes(spec)
                }
            write_json(path / "warmup-corpus.json", warmup)
            assets = None
            original_spec = spec
            for cell in manifest["schedule"]:
                spec = original_spec.model_copy(
                    update={
                        "prefill_workers": cell.get("prefill") or original_spec.prefill_workers,
                        "p01_placement": cell.get("placement", original_spec.p01_placement),
                    }
                )
                directory = path / f"block-{cell['block']:02d}-{cell['arm']}"
                if spec.p01_prefill_grid:
                    directory = path / f"block-{cell['block']:02d}-{cell['arm']}-p{cell['prefill']}"
                if spec.p01_goal_seek and spec.p01_placement != "pooled":
                    directory = directory.with_name(directory.name + "-" + spec.p01_placement)
                directory.mkdir()
                update("preparing", current=cell)
                config_doc = arm_config(manifest["configuration"], spec, cell["arm"])
                if spec.project_id == "P10" and cell["arm"] == "profile":
                    from .attention import config as attention_config

                    config_doc = attention_config(
                        manifest["configuration"], spec, "baseline", directory / "traces"
                    )
                (directory / "config.yaml").write_text(yaml.safe_dump(config_doc))
                cfg = load_config(directory / "config.yaml")
                current = await asyncio.to_thread(fingerprint, cfg.engine)
                if not current.get("snapshot") or not current.get("image"):
                    raise ValueError(
                        "Model snapshot and local image must be available and fingerprinted"
                    )
                if spec.project_id == "P04" and spec.speculative_method == "mtp":
                    model_config = read_json(Path(current["snapshot"]) / "config.json")
                    text_config = model_config.get("text_config", model_config)
                    if not (
                        text_config.get("mtp_num_hidden_layers", 0)
                        or text_config.get("num_nextn_predict_layers", 0)
                    ):
                        raise ValueError("The cached checkpoint does not declare native MTP layers")
                    index = read_json(Path(current["snapshot"]) / "model.safetensors.index.json")
                    if not any("mtp" in name.lower() for name in index.get("weight_map", {})):
                        raise ValueError("The cached checkpoint has no indexed MTP tensors")
                if assets is not None and current != assets:
                    raise ValueError("Assets changed between paired arms")
                assets = current
                write_json(directory / "assets.json", assets)
                environment = await asyncio.to_thread(capture_environment, cfg)
                write_json(directory / "environment.json", environment)
                cfg.engine.gpu_image = assets["image"]["id"]
                pin_model(cfg.engine, assets["snapshot"])
                (directory / "launch-resolved.yaml").write_text(yaml.safe_dump(asdict(cfg)))
                if spec.project_id in ("P01", "P02"):
                    from .dynamo import OwnedDynamo

                    if spec.project_id == "P02":
                        from .sessions import OwnedSessions

                        engine = OwnedSessions(cfg.engine, spec, cell["arm"], path.name)
                    else:
                        engine = OwnedDynamo(cfg.engine, spec, cell["arm"], path.name)
                else:
                    engine = OwnedReplicas(cfg.engine, owner=path.name)
                try:
                    update("launching", current=cell)
                    await launch_owned(engine, directory)
                    await probe(engine, 60)
                    if spec.project_id == "P10":
                        from .attention import logits, profile, verify_backend

                        verify_backend(
                            directory,
                            spec.attention_treatment
                            if cell["arm"] == "treatment"
                            else spec.attention_baseline,
                            spec.attention_kv_dtype,
                        )
                        if cell["arm"] == "profile":
                            update("profiling", current=cell)
                            await profile(engine, corpus, spec, directory)
                            continue
                        await logits(engine, corpus, directory)
                    sampler = asyncio.create_task(
                        telemetry(directory, engine, 4 * 1024**3, time.monotonic())
                    )
                    loads = (
                        spec.speculative_concurrencies
                        if spec.project_id == "P04" and spec.speculative_concurrencies
                        else spec.p01_request_rates
                        if spec.p01_prefill_grid
                        else [spec.concurrency]
                    )
                    if spec.p01_goal_seek:
                        from .goal_search import windows

                        loads = windows(spec, cell, state["results"])
                    for load in loads:
                        concurrency = load["rate"] if isinstance(load, dict) else load
                        active_corpus, active_warmup = corpus, warmup
                        settings = spec.model_copy(
                            update={
                                "concurrency": spec.concurrency
                                if spec.p01_prefill_grid
                                else concurrency
                            }
                        )
                        point = (
                            dict(cell, concurrency=concurrency)
                            if spec.speculative_concurrencies
                            else cell
                        )
                        window = (
                            directory / f"concurrency-{concurrency}"
                            if spec.speculative_concurrencies
                            else directory
                        )
                        if spec.p01_prefill_grid:
                            point = dict(cell, rate=concurrency)
                            window = directory / f"rate-{concurrency:g}"
                        if spec.p01_goal_seek:
                            from .goal_search import client_concurrency

                            settings = settings.model_copy(
                                update={"concurrency": client_concurrency(spec, concurrency)}
                            )
                            point = dict(
                                point,
                                shape=load["id"],
                                document_records=load["records"],
                                answer_records=load["answers"],
                            )
                            window = directory / load["id"] / f"rate-{concurrency:g}"
                            active_corpus, active_warmup = corpus[load["id"]], warmup[load["id"]]
                            write_json(
                                directory / "search-decision.json",
                                dict(next=point, completed_cells=len(state["results"])),
                            )
                        window.mkdir(parents=True, exist_ok=True)
                        if spec.p01_goal_seek:
                            write_json(
                                window / "search-decision.json",
                                dict(
                                    next=point,
                                    completed_cells=len(state["results"]),
                                    rule="Endpoint probing followed by rate-grid bisection",
                                ),
                            )
                        if settings.warmup_s:
                            update("warming", current=point)
                            await measure(
                                engine,
                                active_warmup,
                                settings,
                                window / "warmup.jsonl",
                                settings.warmup_s,
                            )
                        await capture_metrics(engine, window, "before")
                        update("measuring", current=point)
                        if spec.p01_prefill_grid:
                            from .tenants import measure_tenant

                            operation = measure_tenant(
                                engine,
                                active_corpus,
                                settings,
                                window / "requests.jsonl",
                                time.monotonic() + 0.2,
                                rate=concurrency,
                            )
                        elif spec.project_id == "P02":
                            from .sessions import measure_sessions

                            operation = measure_sessions(engine, corpus, settings, window)
                        else:
                            operation = measure(engine, corpus, settings, window / "requests.jsonl")
                        measurement = asyncio.create_task(operation)
                        try:
                            done, _ = await asyncio.wait(
                                [measurement, sampler], return_when=asyncio.FIRST_COMPLETED
                            )
                            if sampler in done:
                                await sampler
                                raise RuntimeError(
                                    "Telemetry stopped before the measurement completed"
                                )
                            metrics = await measurement
                        finally:
                            measurement.cancel()
                            await asyncio.gather(measurement, return_exceptions=True)
                        if spec.p01_goal_seek:
                            metrics["effective_client_concurrency"] = settings.concurrency
                            metrics["client_limit_detected"] = (
                                metrics.get("queue_p95_s") or 0
                            ) > 0.1
                        await capture_metrics(engine, window, "after")
                        if spec.project_id == "P01" and cell["arm"] == "treatment":
                            from .dynamo import verify_transfer

                            verify_transfer(window, spec, getattr(engine, "worker_roles", None))
                        if spec.project_id == "P01":
                            from .dynamo import verify_worker_activity

                            verify_worker_activity(window, spec)
                        if spec.project_id == "P04" and cell["arm"] == "treatment":
                            from .speculative import verify_activity

                            verify_activity(window, len(engine.replica_urls))
                        write_json(window / "metrics.json", metrics)
                        state["results"].append(dict(point, metrics=metrics))
                        update("measured", current=point)
                    update("cleaning", current=cell)
                finally:
                    await finish_owned_cleanup([engine], sampler)
                    sampler = engine = None
            if spec.p01_goal_seek:
                from .goal_search import compare_goal

                comparison = compare_goal(state["results"], original_spec)
            elif spec.p01_prefill_grid:
                from .search_report import compare_search

                comparison = compare_search(state["results"], original_spec)
            else:
                comparison = compare(state["results"], spec)
            if spec.project_id == "P10":
                from .attention import parity

                numerical = parity(path, spec)
                comparison["numerical_parity"] = numerical
                if not numerical["passed"]:
                    comparison.update(
                        qualified=False, reason="Next-token numerical parity gate failed"
                    )
            if spec.project_id == "P04":
                from .speculative import export_candidate

                export_candidate(path, manifest, comparison)
            write_json(path / "comparison.json", comparison)
            update(
                "complete",
                current=None,
                outcome="qualified" if comparison["qualified"] else "needs_review",
            )
    except asyncio.CancelledError:
        update("cancelled", current=None, outcome="incomplete")
        raise
    except Exception as exc:
        update("failed", error=str(exc), current=None, outcome="incomplete")
    finally:
        if spec.p01_goal_seek:
            from .goal_search import compare_goal

            try:
                comparison = compare_goal(state["results"], ExperimentSpec(**manifest["spec"]))
            except Exception as exc:
                comparison = {
                    "qualified": False,
                    "reason": f"Goal search comparison failed: {exc}",
                    "metrics": {},
                }
                update("failed", error=str(exc), current=None, outcome="incomplete")
            write_json(path / "comparison.json", comparison)
        # All deployment teardown precedes evidence checksums.
        try:
            await asyncio.to_thread(export, path, comparison)
        except Exception as exc:
            update("failed", error=f"Evidence export failed: {exc}", outcome="incomplete")


async def main(path):
    parent = os.getppid()
    with Lease(path / ".worker.lock"):
        task = asyncio.create_task(campaign(path))
        loop = asyncio.get_running_loop()

        cancellation_requested = False

        def request_cancel():
            nonlocal cancellation_requested
            # The process-group guard may forward a second termination signal.
            if not cancellation_requested:
                cancellation_requested = True
                task.cancel()

        for sig in (signal.SIGINT, signal.SIGTERM):
            loop.add_signal_handler(sig, request_cancel)

        async def watch():
            while not task.done():
                if os.getppid() != parent:
                    request_cancel()
                    return
                await asyncio.sleep(1)

        watcher = asyncio.create_task(watch())
        try:
            await asyncio.wait_for(
                task, read_json(path / "manifest.json")["spec"]["max_hours"] * 3600
            )
        finally:
            watcher.cancel()
            await asyncio.gather(watcher, return_exceptions=True)


if __name__ == "__main__":
    with contextlib.suppress(asyncio.CancelledError):
        asyncio.run(main(Path(sys.argv[1])))
