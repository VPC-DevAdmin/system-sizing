"""Owned container lifecycle with cancellable startup and no stale-container sweep."""

import asyncio
import contextlib
import copy
import socket
import subprocess
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from ..benchmarks.storage import read_json, write_json
from ..benchmarks.worker import assert_gpu_idle
from ..engines.base import redact_argv
from ..engines.vllm_cuda import VllmCudaEngine


class OwnedVllm(VllmCudaEngine):
    def __init__(self, cfg, stopping, owner=None):
        super().__init__(cfg)
        self.stopping = stopping
        self.owner = owner
        self.journal = None
        self.log_stream = None

    def launch(self, log_dir):
        path = Path(log_dir)
        path.mkdir(parents=True, exist_ok=True)
        self._log_path = path / "engine.log"
        # Refuse a listener before launch; never mistake another server's health for ours.
        with socket.socket() as sock:
            sock.settimeout(1)
            if sock.connect_ex((self.cfg.host, self.cfg.port)) == 0:
                raise RuntimeError(f"Port {self.cfg.port} is occupied")
        argv = self._build_docker_command()
        name = "capsim-infra-" + uuid.uuid4().hex
        argv[argv.index("--name") + 1] = name
        self._container_id = name
        if self.owner:
            self.journal = path / "resource.json"
            write_json(self.journal, {"owner": self.owner, "name": name, "phase": "creating"})
            position = argv.index("--name") + 2
            argv[position:position] = ["--label", f"capsim.infrastructure={self.owner}"]
        (path / "command.txt").write_text(str(redact_argv(argv)))
        try:
            result = subprocess.run(argv, capture_output=True, text=True, timeout=120)
            if result.returncode:
                (path / "launch-error.txt").write_text(result.stderr)
                raise RuntimeError(
                    f"Owned container launch failed; see {path / 'launch-error.txt'}"
                )
            self.log_stream = self._log_path.open("ab")
            self._log_streamer = subprocess.Popen(
                ["docker", "logs", "-f", name], stdout=self.log_stream, stderr=subprocess.STDOUT
            )
            deadline = time.monotonic() + self.cfg.startup_timeout_s
            while time.monotonic() < deadline:
                if self.stopping.is_set():
                    raise RuntimeError("Experiment launch cancelled")
                if self.health_check():
                    return
                status = subprocess.run(
                    ["docker", "inspect", "--format", "{{.State.Running}}", name],
                    capture_output=True,
                    text=True,
                    timeout=10,
                )
                if status.returncode or status.stdout.strip() != "true":
                    raise RuntimeError(
                        f"Owned engine exited before readiness; see {self._log_path}"
                    )
                self.stopping.wait(2)
            raise TimeoutError("Engine startup deadline exceeded")
        except BaseException:
            self.shutdown()
            raise

    def shutdown(self):
        name = self._container_id
        if name:
            try:
                subprocess.run(
                    ["docker", "stop", "-t", "20", name], capture_output=True, timeout=30
                )
            except subprocess.TimeoutExpired:
                pass
            # A unique name is reserved before docker run, so even a timed-out run
            # can only cause this experiment's container to be removed.
            result = subprocess.run(
                ["docker", "inspect", "--format", "{{.State.Running}}", name],
                capture_output=True,
                text=True,
                timeout=10,
            )
            if result.returncode == 0:
                try:
                    failed = subprocess.run(
                        ["docker", "rm", "-f", "-v", name], capture_output=True, timeout=20
                    ).returncode
                except subprocess.TimeoutExpired:
                    failed = True
                if failed:
                    # --rm may remove the container between inspect and rm, and a
                    # replica with a large pinned host cache takes minutes to release
                    # it. Verify absence instead of treating either as failure.
                    deadline = time.monotonic() + 600
                    while True:
                        final = subprocess.run(
                            ["docker", "inspect", name], capture_output=True, text=True, timeout=10
                        )
                        if final.returncode and "no such" in final.stderr.lower():
                            break
                        if final.returncode or time.monotonic() >= deadline:
                            raise RuntimeError(
                                f"Could not verify removal of owned container {name}"
                            )
                        time.sleep(2)
            elif "no such" not in result.stderr.lower():
                raise RuntimeError(f"Could not verify cleanup of owned container {name}")
            self._container_id = None
            if self.journal:
                resource = read_json(self.journal)
                resource["phase"] = "removed"
                write_json(self.journal, resource)
        if self._log_streamer:
            self._log_streamer.terminate()
            try:
                self._log_streamer.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self._log_streamer.kill()
                self._log_streamer.wait()
            self._log_streamer = None
        if self.log_stream:
            self.log_stream.close()
            self.log_stream = None


class OwnedReplicas:
    def __init__(self, cfg, owner=None):
        self.cfg = cfg
        self.stopping = threading.Event()
        self.children = []
        for index, devices in enumerate(cfg.replica_devices or [[g] for g in cfg.gpu_device_ids]):
            child_cfg = copy.deepcopy(cfg)
            child_cfg.type = "vllm_cuda"
            child_cfg.gpu_device_ids = devices
            child_cfg.tensor_parallel_size = len(devices)
            child_cfg.port = cfg.port + index
            self.children.append(OwnedVllm(child_cfg, self.stopping, owner))

    @property
    def replica_urls(self):
        return [url for child in self.children for url in child.replica_urls]

    @property
    def api_model_name(self):
        return self.children[0].api_model_name

    @property
    def api_key(self):
        return self.children[0].api_key

    def launch(self, log_dir):
        failure = None
        with ThreadPoolExecutor(max_workers=len(self.children)) as pool:
            jobs = [
                pool.submit(child.launch, Path(log_dir) / f"replica-{index}")
                for index, child in enumerate(self.children)
            ]
            for job in as_completed(jobs):
                try:
                    job.result()
                except BaseException as exc:
                    failure = failure or exc
                    self.stopping.set()
            # All launch threads have joined before cleanup can release resources.
        if failure:
            self.shutdown()
            raise failure

    def shutdown(self):
        self.stopping.set()
        with ThreadPoolExecutor(max_workers=len(self.children)) as pool:
            jobs = [pool.submit(child.shutdown) for child in self.children]
            errors = []
            for job in jobs:
                try:
                    job.result()
                except Exception as exc:
                    errors.append(str(exc))
        if errors:
            raise RuntimeError("Owned deployment cleanup failed: " + "; ".join(errors))

    def get_metrics(self):
        return {str(i): child.get_metrics() for i, child in enumerate(self.children)}


async def launch_owned(engine, directory):
    await asyncio.to_thread(assert_gpu_idle, engine)
    task = asyncio.create_task(asyncio.to_thread(engine.launch, directory))
    try:
        await asyncio.shield(task)
    except BaseException:
        engine.stopping.set()
        with contextlib.suppress(Exception):
            await task
        raise


async def finish_owned_cleanup(engines, sampler=None, partition=None):
    """Join all owned teardown even when Stop arrives during normal cleanup."""

    async def cleanup():
        if sampler:
            sampler.cancel()
            await asyncio.gather(sampler, return_exceptions=True)
        errors = []
        for engine in engines:
            if engine:
                try:
                    await asyncio.to_thread(engine.shutdown)
                except Exception as exc:
                    errors.append(str(exc))
        if partition:
            try:
                await asyncio.to_thread(partition.restore)
            except Exception as exc:
                errors.append(str(exc))
        if errors:
            raise RuntimeError("Owned cleanup failed: " + "; ".join(errors))

    task = asyncio.create_task(cleanup())
    cancelled = False
    while True:
        try:
            await asyncio.shield(task)
            break
        except asyncio.CancelledError:
            if task.cancelled():
                raise
            cancelled = True
    if cancelled:
        raise asyncio.CancelledError
