"""P02 capacity cells on a plain ``vllm serve`` container (no Dynamo).

The Dynamo runtime (``dynamo.py`` + ``dynamo_runtime.py``) starts a
frontend and one ``dynamo.vllm`` worker per GPU inside NVIDIA's pinned
image. A capacity replica is one GPU, so there is nothing to route, and
on a host with no Dynamo build (AMD ROCm) the same comparison runs on
the stock vLLM server instead:

* the vLLM engine flags the Dynamo worker gets for sessions --
  ``--max-num-seqs``, ``--enable-prefix-caching``, ``--kv-cache-dtype``,
  no artificial GPU KV cap -- are passed explicitly, identically in
  both arms;
* the treatment adds exactly one ``--kv-transfer-config`` pair naming
  the observed ``SimpleCPUOffloadConnector`` (native_cpu_connector.py),
  mounted at /capsim_native_cpu.py with its evidence written to the
  replica directory, as under Dynamo;
* the API, ``/health`` and ``/metrics`` share the server port, so
  metrics come from there rather than per-worker Dynamo system ports;
* gpt-oss uses vLLM's own Harmony parsing (``--reasoning-parser
  openai_gptoss``) where the Dynamo path needed a frontend shim.

The plan, the exact service command and the image's package versions
are written beside the engine log so the report states this runtime
rather than Dynamo's.
"""

import json
import subprocess
from pathlib import Path

from ..benchmarks.storage import write_json
from ..engines.vllm_cuda import VllmCudaEngine
from ..gpu_vendor import engine_image, vendor
from .deployment import OwnedVllm
from .sessions import OwnedSessions

CONNECTOR = Path(__file__).with_name("native_cpu_connector.py")
VERSION_PROBE = (
    "import importlib.metadata as m, json\n"
    "out = {}\n"
    "for p in ('vllm', 'torch', 'transformers', 'triton', 'aiter', 'amdsmi'):\n"
    "    try:\n"
    "        out[p] = m.version(p)\n"
    "    except m.PackageNotFoundError:\n"
    "        pass\n"
    "import torch\n"
    "out['torch.version.hip'] = getattr(torch.version, 'hip', None)\n"
    "out['torch.version.cuda'] = torch.version.cuda\n"
    "print(json.dumps(out))\n"
)


def session_args(cfg, spec, arm):
    """vLLM server flags for one session replica. The treatment's argv is
    the baseline's plus the connector pair, nothing else."""
    argv = [
        "--max-num-seqs", str(cfg.max_num_seqs),
        "--enable-prefix-caching",
        "--kv-cache-dtype", cfg.kv_cache_dtype or "fp8",
    ]
    if spec.session_gpu_gb is not None:
        argv += ["--kv-cache-memory-bytes", str(spec.session_gpu_gb * 1024**3)]
    if "gpt-oss" in (cfg.model_local_path or cfg.model_id or "").lower():
        argv += ["--reasoning-parser", "openai_gptoss"]
    if arm == "treatment":
        if spec.session_tier != "cpu" or spec.session_cpu_backend != "simple_cpu":
            raise ValueError("The plain vLLM runtime offloads to the CPU tier only")
        argv += ["--kv-transfer-config", json.dumps({
            "kv_connector": "ObservedCPUOffloadConnector",
            "kv_connector_module_path": "capsim_native_cpu",
            "kv_role": "kv_both",
            "kv_connector_extra_config": {"cpu_bytes_to_use": spec.session_cpu_gb * 1024**3},
        })]
    return argv


class OwnedNativeSessions(OwnedSessions):
    """One GPU's P02 replica as a plain vLLM server."""

    @property
    def replica_urls(self):
        return [f"http://{self.cfg.host}:{self.cfg.port}/v1"]

    @property
    def metrics_urls(self):
        # Same port as the API; health_check derives /health from this.
        return [f"http://{self.cfg.host}:{self.cfg.port}/metrics"]

    def launch(self, log_dir):
        self._session_preflight()
        self.directory = Path(log_dir) / "replica-0"
        self.directory.mkdir(parents=True)
        plan = {
            "runtime": "vllm",
            "gpu_vendor": vendor(),
            "arm": self.arm,
            "gpu_ids": self.spec.gpu_ids,
            "port": self.cfg.port,
            "image": self.cfg.gpu_image,
            "model": self.cfg.model_local_path or self.cfg.model_id,
            "served_model_name": self.api_model_name,
            "context_limit": self.spec.context_limit,
            "max_num_seqs": self.cfg.max_num_seqs,
            "gpu_memory_utilization": self.cfg.gpu_memory_utilization,
            "kv_cache_dtype": self.cfg.kv_cache_dtype or "fp8",
            "quantization": self.cfg.quantization_kind or self.cfg.quantization,
            "sessions": {k: v for k, v in self.spec.model_dump().items()
                         if k.startswith("session_")},
        }
        write_json(self.directory / "session-plan.json", plan)
        OwnedVllm.launch(self, self.directory)
        self._record_mounts()
        # The command actually run, as OwnedVllm recorded it (redacted).
        write_json(self.directory / "service-commands.json",
                   [{"name": "vllm-serve",
                     "command": (self.directory / "command.txt").read_text()}])
        write_json(self.directory / "runtime-versions.json",
                   {"packages": self._image_versions()})

    def _image_versions(self):
        try:
            r = subprocess.run(
                ["docker", "exec", self._container_id, "python3", "-c", VERSION_PROBE],
                capture_output=True, text=True, timeout=60,
            )
            return json.loads(r.stdout.strip().splitlines()[-1]) if r.returncode == 0 else {}
        except (OSError, subprocess.TimeoutExpired, ValueError, IndexError):
            return {}

    def _build_docker_command(self):
        # The plain vLLM server command (GPU flags per vendor, cache mounts,
        # pinned image), not the Dynamo supervisor OwnedDynamo builds.
        argv = VllmCudaEngine._build_docker_command(self)
        pos = argv.index(engine_image("vllm", self.cfg.gpu_image))
        argv[pos:pos] = [
            # A large pinned host buffer needs unlimited locked memory.
            "--ulimit", "memlock=-1",
            "-v", f"{CONNECTOR}:/capsim_native_cpu.py:ro",
            "-v", f"{self.directory.resolve()}:/evidence",
            "-e", "PYTHONPATH=/",
            "-e", "CAPSIM_CPU_EVIDENCE_DIR=/evidence",
        ]
        return argv + session_args(self.cfg, self.spec, self.arm)
