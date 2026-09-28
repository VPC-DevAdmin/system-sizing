"""Serial, owned image-generation jobs under the common hardware lease."""

import asyncio
import hashlib
import json
import os
import subprocess
import threading
import time
import uuid
from pathlib import Path

from ..benchmarks.process import Lease, hardware_lease_path, module_argv, run_process
from ..benchmarks.storage import read_json, write_json
from ..benchmarks.worker import assert_gpu_idle
from ..config import EngineConfig
from .capabilities import inspect_host
from .deployment import OwnedVllm, finish_owned_cleanup
from .diffusion import raw_comparison


async def run(path, manifest, spec):
    from .report import export

    state = read_json(path / "state.json")

    def update(phase, **values):
        state.update(phase=phase, updated_at=time.time(), **values)
        write_json(path / "state.json", state)

    comparison = {"qualified": False, "reason": "Diffusion experiment incomplete", "metrics": {}}
    try:
        with Lease(hardware_lease_path()):
            idle_cfg = EngineConfig()
            idle_cfg.type = "vllm_cuda"
            await asyncio.to_thread(assert_gpu_idle, OwnedVllm(idle_cfg, threading.Event()))
            update("preparing")
            write_json(path / "host.json", await asyncio.to_thread(inspect_host))
            package_root = str(Path(__file__).resolve().parents[2])
            rc = await run_process(
                module_argv(
                    package_root, "simulator.infrastructure.diffusion", str(path.resolve())
                ),
                path / "prepare.log",
                timeout=3600,
            )
            if rc:
                raise RuntimeError("Diffusion preparation failed; see prepare.log")
            assets = read_json(path / "assets.json")
            prompts = read_json(path / "corpus.json")
            for cell in manifest["schedule"]:
                folder = path / f"block-{cell['block']:02d}-{cell['arm']}-steps-{cell['steps']}"
                directory = folder / "replica-0"
                directory.mkdir(parents=True)
                asset = assets[cell["arm"]]
                plan = dict(cell, snapshot=asset["snapshot"], prompts=prompts, seed=spec.seed)
                write_json(directory / "plan.json", plan)
                runtime = Path(__file__).with_name("diffusion_runtime.py")
                snapshot = Path(asset["snapshot"])
                cache = snapshot.parent.parent.parent
                # Use the common ownership format so interrupted-run recovery applies.
                name = "capsim-infra-" + uuid.uuid4().hex
                journal = directory / "resource.json"
                write_json(journal, {"owner": path.name, "name": name, "phase": "creating"})
                cfg = EngineConfig()
                engine = OwnedVllm(cfg, threading.Event(), path.name)
                engine._container_id = name
                engine.journal = journal
                argv = [
                    "docker",
                    "run",
                    "-d",
                    "--name",
                    name,
                    "--label",
                    f"capsim.infrastructure={path.name}",
                    "--gpus",
                    f"device={spec.gpu_ids[0]}",
                    "--ipc=host",
                    "--network=none",
                    "--user",
                    f"{os.getuid()}:{os.getgid()}",
                    "-e",
                    "HOME=/tmp",
                    "-e",
                    "HF_HOME=/tmp/huggingface",
                    "-v",
                    f"{cache}:{cache}:ro",
                    "-v",
                    f"{directory.resolve()}:/evidence",
                    "-v",
                    f"{runtime}:/runtime.py:ro",
                    assets["image"],
                    "/runtime.py",
                ]
                write_json(directory / "command.json", argv)
                try:
                    update("measuring", current=cell)
                    launching = asyncio.create_task(
                        asyncio.to_thread(
                            subprocess.run, argv, capture_output=True, text=True, timeout=120
                        )
                    )
                    try:
                        launched = await asyncio.shield(launching)
                    except asyncio.CancelledError:
                        await launching
                        raise
                    (directory / "launch.txt").write_text(launched.stdout + launched.stderr)
                    if launched.returncode:
                        raise RuntimeError("Diffusion container launch failed")
                    while True:
                        proc = await asyncio.to_thread(
                            subprocess.run,
                            ["docker", "inspect", "--format", "{{json .State}}", name],
                            capture_output=True,
                            text=True,
                            timeout=15,
                        )
                        if proc.returncode:
                            raise RuntimeError("Owned diffusion container disappeared")
                        status = json.loads(proc.stdout)
                        if not status["Running"]:
                            logs = await asyncio.to_thread(
                                subprocess.run,
                                ["docker", "logs", name],
                                capture_output=True,
                                text=True,
                                timeout=30,
                            )
                            (directory / "engine.log").write_text(logs.stdout + logs.stderr)
                            if status["ExitCode"] != 0:
                                raise RuntimeError("Diffusion runtime failed; see engine.log")
                            break
                        await asyncio.sleep(2)
                    rows = [
                        json.loads(r) for r in (directory / "images.jsonl").read_text().splitlines()
                    ]
                    if len(rows) != len(prompts):
                        raise RuntimeError("Incomplete diffusion cohort")
                    for row in rows:
                        if (
                            row["ok"]
                            and hashlib.sha256((directory / row["image"]).read_bytes()).hexdigest()
                            != row["sha256"]
                        ):
                            raise RuntimeError("Generated image integrity mismatch")
                    metrics = {
                        "attempted": len(rows),
                        "completed": sum(r["ok"] for r in rows),
                        "elapsed_s": sum(r["elapsed_s"] for r in rows),
                        "acceptance": "pending_review",
                    }
                    state["results"].append(dict(cell, metrics=metrics))
                    update("cleaning", current=cell)
                    if metrics["completed"] / metrics["attempted"] < spec.min_quality:
                        raise RuntimeError(
                            "Diffusion technical completion gate failed; all attempts retained"
                        )
                finally:
                    try:
                        logs = await asyncio.to_thread(
                            subprocess.run,
                            ["docker", "logs", name],
                            capture_output=True,
                            text=True,
                            timeout=30,
                        )
                        (directory / "engine.log").write_text(logs.stdout + logs.stderr)
                    finally:
                        await finish_owned_cleanup([engine])
            comparison = raw_comparison(state["results"])
            write_json(path / "comparison.json", comparison)
            update("complete", current=None, outcome="pending_review")
    except asyncio.CancelledError:
        update("cancelled", current=None, outcome="incomplete")
        raise
    except Exception as exc:
        update("failed", current=None, outcome="incomplete", error=str(exc))
    finally:
        try:
            await asyncio.to_thread(export, path, comparison)
        except Exception as exc:
            update("failed", outcome="incomplete", error="Evidence export failed: " + str(exc))
