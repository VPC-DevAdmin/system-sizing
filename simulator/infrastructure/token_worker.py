"""Persistent allocation launches, shared roofline instrument, owned cleanup."""

import asyncio
import copy
import time
from dataclasses import asdict

import httpx
import yaml

from ..benchmarks.assets import fingerprint, pin_model
from ..benchmarks.process import Lease, hardware_lease_path
from ..benchmarks.storage import read_json, write_json
from ..benchmarks.worker import capture_environment, probe
from ..config import load_config
from ..headline_search import _cell_persona_spec
from ..headline_sweep import run_headline_sweep
from ..persona_loader import parse_persona
from ..personas import PERSONAS, Cohort
from .allocation_recovery import AllocationFailure, run_allocation
from .deployment import finish_owned_cleanup, launch_owned
from .driver import capture_metrics
from .dynamo import OwnedDynamo, arm_config, verify_transfer, verify_worker_activity
from .report import export
from .token_search import CounterView, compare, ensure_file_limit, wait_idle


async def run(path, manifest, spec):
    state = read_json(path / "state.json")
    state.setdefault("results", [])
    state.setdefault("attempts", [])
    assets = None

    def update(phase, **kw):
        state.update(phase=phase, updated_at=time.time(), **kw)
        write_json(path / "state.json", state)
        write_json(path / "comparison.json", compare(state["results"]))

    try:
        with Lease(hardware_lease_path()):
            write_json(path / "file-limits.json", ensure_file_limit(max(spec.p01_streams)))
            for cell in manifest["schedule"]:
                settings = spec.model_copy(
                    update={
                        "prefill_workers": cell["prefill"] or spec.prefill_workers,
                        "p01_placement": cell["placement"],
                    }
                )
                directory = (
                    path
                    / f"block-{cell['block']:02d}-{cell['arm']}-p{cell['prefill']}-{cell['placement']}"
                )

                async def attempt(folder, number, cell=cell, settings=settings):
                    nonlocal assets
                    folder.mkdir()
                    config = arm_config(manifest["configuration"], settings, cell["arm"])
                    (folder / "config.yaml").write_text(yaml.safe_dump(config))
                    cfg = load_config(folder / "config.yaml")
                    current = await asyncio.to_thread(fingerprint, cfg.engine)
                    if not current.get("snapshot") or not current.get("image"):
                        raise ValueError("A cached model and pinned runtime image are required")
                    if assets is not None and current != assets:
                        raise ValueError("Assets changed between allocations")
                    assets = current
                    write_json(folder / "assets.json", current)
                    write_json(
                        folder / "environment.json",
                        await asyncio.to_thread(capture_environment, cfg),
                    )
                    cfg.engine.gpu_image = current["image"]["id"]
                    pin_model(cfg.engine, current["snapshot"])
                    (folder / "launch-resolved.yaml").write_text(yaml.safe_dump(asdict(cfg)))
                    engine = OwnedDynamo(cfg.engine, settings, cell["arm"], path.name)
                    try:
                        update("launching", current=cell)
                        await launch_owned(engine, folder)
                        await probe(engine, 60)
                        for inp, out in spec.p01_token_shapes:
                            shape = f"i{inp}-o{out}"
                            if any(
                                r["block"] == cell["block"]
                                and r["prefill"] == cell["prefill"]
                                and r["placement"] == cell["placement"]
                                and r["shape"] == shape
                                for r in state["results"]
                            ):
                                continue
                            await wait_idle(engine, spec.request_timeout_s)
                            window = folder / shape
                            window.mkdir()
                            pid = "headline_infrastructure_" + path.name
                            definition = _cell_persona_spec(
                                inp, out, name=shape, description="Fixed shape saturation probe"
                            )
                            PERSONAS[pid] = parse_persona(pid, definition)
                            cohort = Cohort(
                                pid, shape, "Fixed-token roofline comparison", {pid: 1.0}
                            )
                            measured = copy.deepcopy(cfg)
                            # Capacity affects stop heuristics, not the already-launched engine.
                            measured.engine.replica_devices = [[g] for g in spec.gpu_ids]
                            measured.simulation.headline_clear_s = spec.warmup_s
                            measured.simulation.headline_measure_s = max(10, spec.duration_s // 2)
                            measured.simulation.headline_measure_max_s = max(
                                spec.duration_s * 3, 60
                            )
                            measured.simulation.request_timeout_s = spec.request_timeout_s
                            progress = {}

                            async def publish(shape=shape, progress=progress):
                                while True:
                                    update(
                                        "measuring",
                                        current=dict(
                                            cell, shape=shape, progress=copy.deepcopy(progress)
                                        ),
                                    )
                                    await asyncio.sleep(5)

                            await capture_metrics(engine, window, "before")
                            watcher = asyncio.create_task(publish())
                            try:
                                summary_path = await run_headline_sweep(
                                    measured,
                                    cohort,
                                    run_dir=window,
                                    ladder_override=spec.p01_streams,
                                    existing_engine=CounterView(engine),
                                    persona_specs={pid: definition},
                                    progress=progress,
                                )
                            finally:
                                watcher.cancel()
                                await asyncio.gather(watcher, return_exceptions=True)
                                PERSONAS.pop(pid, None)
                            result = read_json(summary_path)
                            if result["status"] != "ok":
                                raise AllocationFailure(result["stop_reason"])
                            await capture_metrics(engine, window, "after")
                            if cell["prefill"]:
                                verify_transfer(window, settings, engine.worker_roles)
                            verify_worker_activity(window, settings)
                            state["results"].append(
                                dict(
                                    cell,
                                    shape=shape,
                                    input_tokens=inp,
                                    output_tokens=out,
                                    rungs=result["rungs"],
                                    peak=result["peak"],
                                    stop_reason=result["stop_reason"],
                                    evidence=str(window.relative_to(path)),
                                    deployment_attempt=number,
                                )
                            )
                            update("measured", current=dict(cell, shape=shape))
                    except (RuntimeError, httpx.HTTPError, TimeoutError, ConnectionError) as exc:
                        raise AllocationFailure(str(exc)) from exc
                    finally:
                        await finish_owned_cleanup([engine])

                def failure(folder, number, exc, cell=cell):
                    state["attempts"].append(
                        dict(
                            allocation=cell, attempt=number, error=str(exc), will_retry=number == 0
                        )
                    )
                    write_json(folder / "failure.json", state["attempts"][-1])
                    update("recovering", current=cell)

                succeeded = await run_allocation(directory, attempt, failure, enabled=True)
                if not succeeded and cell["arm"] == "baseline":
                    raise RuntimeError(
                        "Aggregate baseline failed after bounded recovery; remaining allocations require a working baseline"
                    )
            expected = len(manifest["schedule"]) * len(spec.p01_token_shapes)
            complete = len(state["results"]) == expected
            update(
                "complete" if complete else "failed",
                current=None,
                outcome="needs_review" if complete else "incomplete",
                complete_search=complete,
                error=None
                if complete
                else "Search has missing allocation-shape measurements after bounded recovery",
            )
    except asyncio.CancelledError:
        update("cancelled", current=None, outcome="incomplete")
        raise
    except Exception as exc:
        update("failed", error=str(exc), current=None, outcome="incomplete")
    finally:
        try:
            await asyncio.to_thread(export, path, compare(state["results"]))
        except Exception as exc:
            update("failed", error="Evidence export failed: " + str(exc), outcome="incomplete")
    return state
