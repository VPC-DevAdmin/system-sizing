"""Isolated campaign supervisor; survives browser disconnects, not owner death."""

from __future__ import annotations

import asyncio
import contextlib
import importlib.metadata
import json
import os
import platform
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

import httpx
import psutil
import yaml

from .adapters import ArtifactError, command, parse, verdict
from .assets import IMAGE_FIELDS, fingerprint, local_snapshot, pin_model
from .dataset import build_dataset
from .process import Lease, run_process
from .proxy import refresh_routing, routing_endpoint, worker_count
from .report import export_bundle
from .runtime import prepare
from .storage import read_json, sha256, write_json


def configure_file_limit(spec):
    """Each routed stream owns a client and upstream socket; children inherit the limit."""
    import resource

    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    required = max(spec.get("concurrency") or [10240]) * 2 + 4096
    if hard != resource.RLIM_INFINITY and hard < required:
        raise RuntimeError(
            f"Open-file hard limit {hard} cannot support this load; at least {required} is required"
        )
    target = max(soft, min(65536, hard) if hard != resource.RLIM_INFINITY else 65536, required)
    if soft != resource.RLIM_INFINITY and soft < target:
        resource.setrlimit(resource.RLIMIT_NOFILE, (target, hard))
    return {"soft": resource.getrlimit(resource.RLIMIT_NOFILE)[0], "hard": hard}


def capture_environment(cfg) -> dict:
    info = {
        "hostname": platform.node(),
        "platform": platform.platform(),
        "python": sys.version,
        "supervisor_runtime": {
            "executable": sys.executable,
            "packages": sorted(
                [
                    {"name": dist.metadata["Name"], "version": dist.version}
                    for dist in importlib.metadata.distributions()
                    if dist.metadata["Name"]
                ],
                key=lambda item: (item["name"].casefold(), item["version"]),
            ),
        },
        "logical_cpus": os.cpu_count(),
        "ram_bytes": psutil.virtual_memory().total,
        "model_id": cfg.engine.model_id,
        "client_environment": {
            key: os.getenv(key)
            for key in (
                "OMP_NUM_THREADS",
                "MKL_NUM_THREADS",
                "OPENBLAS_NUM_THREADS",
                "TOKENIZERS_PARALLELISM",
            )
        },
    }
    for name, argv in {
        "gpu_inventory": [
            "nvidia-smi",
            "--query-gpu=uuid,name,memory.total,driver_version",
            "--format=csv,noheader",
        ],
        "cpu_inventory": ["lscpu"],
    }.items():
        try:
            p = subprocess.run(argv, capture_output=True, text=True, timeout=15)
            info[name] = p.stdout.strip() if p.returncode == 0 else "unavailable"
        except (OSError, subprocess.SubprocessError):
            info[name] = "unavailable"
    images = {}
    for key, value in vars(cfg.engine).items():
        if key == IMAGE_FIELDS.get(cfg.engine.type) and value:
            try:
                p = subprocess.run(
                    ["docker", "image", "inspect", value, "--format", "{{.Id}}"],
                    capture_output=True,
                    text=True,
                    timeout=15,
                )
                if p.returncode == 0:
                    images[key] = {"reference": value, "id": p.stdout.strip()}
                    setattr(cfg.engine, key, p.stdout.strip())
            except (OSError, subprocess.SubprocessError):
                pass
    info["images"] = images
    return info


def assert_gpu_idle(engine):
    if engine.cfg.type not in ("vllm_cuda", "vllm_cuda_multi", "sglang_cuda", "trtllm"):
        return
    try:
        proc = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=15,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise RuntimeError(f"Cannot verify exclusive GPU access before benchmark launch: {exc}") from exc
    if proc.returncode:
        raise RuntimeError("Cannot verify exclusive GPU access before benchmark launch")
    if proc.stdout.strip():
        raise RuntimeError(
            "GPU compute processes are already active. Stop their owning job before benchmarking; no containers were removed"
        )


async def safe_launch(engine, directory):
    # A cancelled asyncio.to_thread does not stop the launch thread. Always join it
    # before cleanup, so cancellation cannot leave a late-starting server behind.
    await asyncio.to_thread(assert_gpu_idle, engine)
    task = asyncio.create_task(asyncio.to_thread(engine.launch, log_dir=directory))
    try:
        await asyncio.shield(task)
    except BaseException:
        with contextlib.suppress(Exception):
            await task
        await asyncio.to_thread(engine.shutdown)
        raise


async def probe(engine, timeout_s: float):
    async with httpx.AsyncClient(timeout=timeout_s) as client:
        for url in engine.replica_urls:
            root = url.rstrip("/").removesuffix("/v1")
            r = await client.post(
                root + "/v1/chat/completions",
                headers={"Authorization": f"Bearer {engine.api_key}"},
                json={
                    "model": engine.api_model_name,
                    "messages": [{"role": "user", "content": "Hello"}],
                    "max_tokens": 1,
                    "stream": False,
                },
            )
            r.raise_for_status()


async def telemetry(directory: Path, engine, budget: int, started: float):
    from ..collectors.gpu import GpuCollector

    collector = GpuCollector()
    try:
        with (directory / "telemetry.jsonl").open("w") as f:
            while True:
                sample = {
                    "timestamp": time.time(),
                    "host_cpu_pct": psutil.cpu_percent(),
                    "host_memory_pct": psutil.virtual_memory().percent,
                    "gpu_devices": await asyncio.to_thread(collector.sample_devices),
                    "engine_metrics": await asyncio.to_thread(engine.get_metrics),
                }
                f.write(json.dumps(sample) + "\n")
                f.flush()
                if time.monotonic() - started > 10:
                    size = await asyncio.to_thread(
                        lambda: sum(
                            p.stat().st_size
                            for p in directory.rglob("*")
                            if p.is_file() and not p.is_symlink()
                        )
                    )
                    if size > budget:
                        raise RuntimeError("Attempt exceeded disk budget")
                await asyncio.sleep(2)
    finally:
        collector.close()


async def run_measurement(argv, folder, spec, engine):
    proc_task = asyncio.create_task(
        run_process(
            argv,
            folder / "harness.log",
            timeout=spec["duration_s"]
            + spec["warmup_s"]
            + spec["timeout_grace_s"]
            + spec.get("processing_grace_s", 120),
            env={
                **os.environ,
                "HF_HUB_OFFLINE": "1",
                "TRANSFORMERS_OFFLINE": "1",
                "TOKENIZERS_PARALLELISM": "false",
                "HF_HOME": spec["_hf_home"],
            },
            cwd=folder,
        )
    )
    samples = asyncio.create_task(
        telemetry(folder, engine, int(spec["disk_budget_gb"] * 1024**3), time.monotonic())
    )
    try:
        done, _ = await asyncio.wait([proc_task, samples], return_when=asyncio.FIRST_COMPLETED)
        if samples in done:
            await samples
            raise RuntimeError("Telemetry supervision stopped unexpectedly")
        return await proc_task
    finally:
        for task in (proc_task, samples):
            task.cancel()
        await asyncio.gather(proc_task, samples, return_exceptions=True)


async def campaign(directory: Path, base: Path):
    from ..config import load_config
    from ..engines import make_engine

    manifest = read_json(directory / "manifest.json")
    state = read_json(directory / "state.json")
    spec = manifest["recipe"]
    if manifest.get("adapter_sources"):
        current_sources = {p.name: sha256(p) for p in sorted(Path(__file__).parent.glob("*.py"))}
        if current_sources != manifest["adapter_sources"]:
            state.update(
                phase="failed",
                outcome="incomplete",
                error="Benchmark adapter changed; create a fresh experiment",
            )
            write_json(directory / "state.json", state)
            return
    for a in state["attempts"]:
        if a.get("outcome") == "running":
            a.update(outcome="interrupted", reasons=["Previous worker stopped during this attempt"])

    def update(phase, **kwargs):
        state.update(phase=phase, updated_at=time.time(), **kwargs)
        write_json(directory / "state.json", state)

    engine = None
    try:
        update("preparing", error=None)
        if shutil.disk_usage(directory).free < spec["min_free_gb"] * 1024**3:
            raise RuntimeError("Insufficient free disk space for selected benchmark budget")
        runtimes = {h: await prepare(base, h) for h in spec["harnesses"]}
        for name, runtime in runtimes.items():
            write_json(directory / f"{name}-runtime.json", runtime)
        config_path = directory / "engine.yaml"
        if not config_path.exists():
            config_path.write_text(yaml.safe_dump(manifest["configuration"]))
        cfg = load_config(config_path)
        if cfg.engine.model_local_path and cfg.engine.type != "mock":
            raise RuntimeError(
                "Custom in-container model paths need a verified host snapshot mapping; use a catalog model profile"
            )
        current_assets = await asyncio.to_thread(fingerprint, cfg.engine)
        if manifest.get("source_assets") and current_assets != manifest["source_assets"]:
            raise RuntimeError(
                "Model or image assets changed since the selected roofline result; remeasure the source cell"
            )
        file_limits = configure_file_limit(spec)
        environment = await asyncio.to_thread(capture_environment, cfg)
        environment["open_file_limit"] = file_limits
        old_environment = directory / "environment.json"
        if old_environment.exists() and read_json(old_environment) != environment:
            raise RuntimeError(
                "Hardware, Python dependencies, or runtime images changed; create a fresh experiment"
            )
        write_json(old_environment, environment)
        tokenizer = await asyncio.to_thread(local_snapshot, cfg.engine.model_id)
        tokenizer_files = {
            p.name: sha256(p)
            for p in Path(tokenizer).glob("*")
            if p.is_file() and ("token" in p.name or p.name == "config.json")
        }
        assets = {"snapshot": tokenizer, "tokenizer_files": tokenizer_files}
        if (directory / "assets.json").exists() and read_json(directory / "assets.json") != assets:
            raise RuntimeError("Model/tokenizer snapshot changed; create a fresh experiment")
        write_json(directory / "assets.json", assets)
        if not (directory / "dataset.json").exists():
            await asyncio.to_thread(build_dataset, tokenizer, spec, directory)
        if sha256(directory / "prompts.jsonl") != read_json(directory / "dataset.json")["sha256"]:
            raise RuntimeError("Dataset checksum mismatch")
        # AIPerf 0.13.0 offline loading resolves HF repository IDs, even when
        # given a local path. Stage a private tokenizer-only cache under a stable
        # local alias. Both harnesses still use the same captured tokenizer.
        cache = directory / "tokenizer-cache"
        snapshot = cache / "hub" / "models--capsim--benchmark-tokenizer" / "snapshots" / "local"
        if not snapshot.exists():
            from transformers import AutoTokenizer

            local_tok = await asyncio.to_thread(
                AutoTokenizer.from_pretrained,
                tokenizer,
                local_files_only=True,
                trust_remote_code=False,
            )
            await asyncio.to_thread(local_tok.save_pretrained, snapshot)
            refs = snapshot.parent.parent / "refs"
            refs.mkdir(exist_ok=True)
            (refs / "main").write_text("local")
        entry = directory / "guidellm_entry.py"
        entry_source = Path(__file__).with_name("guidellm_entry.py").read_bytes()
        if entry.exists() and entry.read_bytes() != entry_source:
            raise RuntimeError("GuideLLM corpus adapter changed; create a fresh experiment")
        entry.write_bytes(entry_source)
        spec = {
            **spec,
            "_hf_home": str(cache),
            "_guidellm_entry": str(entry),
            "_reasoning_effort": cfg.engine.reasoning_effort if cfg.engine.reasoning else None,
        }
        pin_model(cfg.engine, tokenizer)
        (directory / "launch-resolved.yaml").write_text(
            yaml.safe_dump(__import__("dataclasses").asdict(cfg))
        )
        engine = make_engine(cfg.engine.type, cfg.engine)
        launched = False
        points = spec["rates"] if spec["recipe"] == "slo" else spec["concurrency"]
        jobs = [
            (h, point, rep)
            for rep in range(1, spec["repetitions"] + 1)
            for point in points
            for h in (spec["harnesses"] if rep % 2 else list(reversed(spec["harnesses"])))
        ]
        for harness, point, rep in jobs:
            prior = [
                a
                for a in state["attempts"]
                if (a["harness"], a["point"], a["repetition"]) == (harness, point, rep)
            ]
            if any(
                a["outcome"] in ("measured", "failed_requirements", "inconclusive", "unsupported")
                for a in prior
            ):
                continue
            used = len([a for a in prior if a["outcome"] != "interrupted"])
            for retry in range(used, spec["max_attempts"]):
                size = sum(
                    p.stat().st_size
                    for p in directory.rglob("*")
                    if p.is_file() and not p.is_symlink()
                )
                if (
                    size > spec["disk_budget_gb"] * 1024**3
                    or shutil.disk_usage(directory).free < spec["min_free_gb"] * 1024**3
                ):
                    raise RuntimeError("Campaign disk budget reached or free space below threshold")
                number = len(state["attempts"]) + 1
                folder = directory / "attempts" / f"{number:04d}-{harness}"
                folder.mkdir(parents=True)
                attempt = {
                    "number": number,
                    "harness": harness,
                    "point": point,
                    "repetition": rep,
                    "outcome": "running",
                    "directory": str(folder.relative_to(directory)),
                    "started_at": time.time(),
                    "reasons": [],
                }
                state["attempts"].append(attempt)
                update("launching", current=number)
                try:
                    if launched and spec["cache_policy"] == "restart":
                        await asyncio.to_thread(engine.shutdown)
                        launched = False
                    if not launched:
                        await safe_launch(engine, folder)
                        launched = True
                    await probe(engine, spec["timeout_grace_s"])
                    async with routing_endpoint(
                        engine,
                        spec["timeout_grace_s"],
                        workers=worker_count(max(spec["concurrency"] or [10240])),
                        directory=folder / "router",
                    ) as (
                        endpoint,
                        routing,
                    ):
                        binary = runtimes[harness]["binary"]

                        def args(out, duration, harness=harness, binary=binary, point=point):
                            return command(
                                harness,
                                binary,
                                spec,
                                endpoint,
                                engine.api_model_name,
                                (
                                    "capsim/benchmark-tokenizer"
                                    if harness == "aiperf"
                                    else tokenizer
                                ),
                                directory / "prompts.jsonl",
                                out,
                                point,
                                duration,
                            )

                        if spec["warmup_s"]:
                            update("warming")
                            warm = folder / "warmup"
                            warm.mkdir()
                            warm_argv = args(warm, spec["warmup_s"])
                            write_json(warm / "command.json", warm_argv)
                            if await run_measurement(warm_argv, warm, spec, engine):
                                raise RuntimeError("Harness warmup failed; see warmup/harness.log")
                        await refresh_routing(routing)
                        if spec["warmup_s"]:
                            write_json(warm / "routing.json", routing)
                        if routing["active"]:
                            raise RuntimeError("Warmup did not drain")
                        await refresh_routing(routing, reset=True)
                        update("measuring")
                        argv = args(folder, spec["duration_s"])
                        write_json(folder / "command.json", argv)
                        rc = await run_measurement(argv, folder, spec, engine)
                        await refresh_routing(routing)
                        write_json(folder / "routing.json", routing)
                        if rc:
                            raise RuntimeError(
                                f"Harness exited with code {rc}; inspect harness.log"
                            )
                        update("analyzing")
                        metrics = parse(harness, folder)
                        outcome, reasons = verdict(metrics, {**spec, "_load_point": point}, routing)
                        attempt.update(metrics=metrics, outcome=outcome, reasons=reasons)
                        write_json(folder / "normalized.json", metrics)
                except (ArtifactError, KeyError, json.JSONDecodeError) as e:
                    attempt.update(outcome="unsupported", reasons=[f"Artifact validation: {e}"])
                except asyncio.CancelledError:
                    attempt.update(
                        outcome="interrupted", reasons=["Stopped before measurement completed"]
                    )
                    raise
                except Exception as e:
                    attempt.update(outcome="failed", reasons=[f"{type(e).__name__}: {e}"])
                    if launched:
                        await asyncio.to_thread(engine.shutdown)
                        launched = False
                finally:
                    attempt["ended_at"] = time.time()
                    update("analyzing")
                if attempt["outcome"] != "failed":
                    break
                if retry + 1 < spec["max_attempts"]:
                    await asyncio.sleep(min(2**retry, 10))
        successful_jobs = {
            (a["harness"], a["point"], a["repetition"])
            for a in state["attempts"]
            if a["outcome"] == "measured"
        }
        update(
            "exporting",
            current=None,
            outcome="measured" if len(successful_jobs) == len(jobs) else "needs_review",
        )
    except asyncio.CancelledError:
        update("interrupted", outcome="incomplete", current=None)
        raise
    except Exception as e:
        update("failed", outcome="incomplete", current=None, error=f"{type(e).__name__}: {e}")
    finally:
        try:
            if engine:
                await asyncio.to_thread(engine.shutdown)
        except Exception as e:
            update("failed", outcome="incomplete", current=None, error=f"Engine cleanup: {e}")
        # Docker log streamers can append shutdown records. Seal evidence only
        # after they stop, so its checksums also match the retained live files.
        if state["phase"] == "exporting":
            update("complete")
        try:
            await asyncio.to_thread(export_bundle, directory)
        except Exception as e:
            update("failed", outcome="incomplete", current=None, error=f"Evidence export: {e}")


async def main(directory: Path, base: Path, parent: int):
    with Lease(directory / ".worker.lock"):
        task = asyncio.create_task(campaign(directory, base))
        loop = asyncio.get_running_loop()
        for sig in (signal.SIGTERM, signal.SIGINT):
            loop.add_signal_handler(sig, task.cancel)

        async def watch_parent():
            while not task.done():
                if os.getppid() != parent:
                    task.cancel()
                    return
                await asyncio.sleep(1)

        watcher = asyncio.create_task(watch_parent())
        try:
            timeout = read_json(directory / "manifest.json")["recipe"]["max_hours"] * 3600
            await asyncio.wait_for(task, timeout)
        finally:
            watcher.cancel()
            await asyncio.gather(watcher, return_exceptions=True)


if __name__ == "__main__":
    with contextlib.suppress(asyncio.CancelledError):
        asyncio.run(main(Path(sys.argv[1]), Path(sys.argv[2]), os.getppid()))
