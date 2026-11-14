"""Which GPU stack this host has — NVIDIA (nvidia-smi, docker --gpus)
or AMD ROCm (amd-smi, /dev/kfd + /dev/dri passthrough).

Everything vendor-specific that capsim needs to launch an engine
container and read the cards lives here, so the engines, preflight,
arena and collector ask one question instead of each probing their
own way:

* ``vendor()`` — "nvidia", "amd" or "none". ``CAPSIM_GPU_VENDOR``
  overrides the probe.
* ``docker_gpu_args(device_ids)`` — the ``docker run`` flags that hand
  those devices to a container.
* ``query_gpus()`` — (name, VRAM MiB) per device.
* ``amd_sample_devices()`` — one per-device telemetry reading from
  ``amd-smi metric --json`` in the collector's dict shape.
* ``engine_image(engine, image)`` — swaps the stock CUDA image for its
  ROCm build on an AMD host; an explicitly chosen image is kept.
"""

from __future__ import annotations

import functools
import json
import os
import shutil
import subprocess
from typing import Optional


@functools.lru_cache(maxsize=1)
def vendor() -> str:
    forced = os.environ.get("CAPSIM_GPU_VENDOR", "").strip().lower()
    if forced in ("nvidia", "amd", "none"):
        return forced
    if shutil.which("nvidia-smi"):
        return "nvidia"
    if os.path.exists("/dev/kfd") and shutil.which("amd-smi"):
        return "amd"
    return "none"


def is_amd() -> bool:
    return vendor() == "amd"


def docker_gpu_args(device_ids: Optional[list[int]] = None) -> list[str]:
    """``docker run`` flags giving the container ``device_ids`` (all
    devices when None/empty).

    NVIDIA: ``--gpus`` (CSV-parsed by docker, so a multi-device list
    carries embedded quotes). AMD: pass /dev/kfd and /dev/dri through
    with the video/render groups, and narrow with HIP_VISIBLE_DEVICES
    — the ROCm container runtime is not assumed to be installed.
    """
    ids = [int(i) for i in (device_ids or [])]
    if not is_amd():
        if not ids:
            return ["--gpus", "all"]
        arg = "device=" + ",".join(str(i) for i in ids)
        return ["--gpus", f'"{arg}"' if len(ids) > 1 else arg]
    args = ["--device=/dev/kfd", "--device=/dev/dri",
            "--group-add", "video", "--group-add", "render",
            "--security-opt", "seccomp=unconfined", "--cap-add=SYS_PTRACE"]
    if ids:
        vis = ",".join(str(i) for i in ids)
        # HIP only: ROCR_VISIBLE_DEVICES as well would renumber twice
        # (ROCR filters to 2,3, then HIP=2,3 points past the end).
        args += ["-e", f"HIP_VISIBLE_DEVICES={vis}"]
    return args


def _num(v) -> Optional[float]:
    """amd-smi reports {"value": x, "unit": u} or "N/A"."""
    if isinstance(v, dict):
        v = v.get("value")
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _amd_json(*args: str, timeout: int = 10) -> list[dict]:
    r = subprocess.run(["amd-smi", *args, "--json"], capture_output=True,
                       text=True, timeout=timeout)
    if r.returncode != 0 or not r.stdout.strip():
        return []
    d = json.loads(r.stdout)
    return d.get("gpu_data", d) if isinstance(d, dict) else d


def query_gpus() -> Optional[list[tuple[str, float]]]:
    """[(name, vram_mib)] per device; None when no GPU stack answers."""
    v = vendor()
    try:
        if v == "nvidia":
            r = subprocess.run(
                ["nvidia-smi", "--query-gpu=name,memory.total",
                 "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=15)
            if r.returncode != 0:
                return None
            out = []
            for line in r.stdout.strip().splitlines():
                parts = [p.strip() for p in line.split(",")]
                if len(parts) >= 2:
                    out.append((parts[0], _num(parts[1]) or 0.0))
            return out
        if v == "amd":
            out = []
            for g in _amd_json("static", "--asic", "--vram", timeout=15):
                name = (g.get("asic") or {}).get("market_name") or "AMD GPU"
                size = (g.get("vram") or {}).get("size")
                out.append((name, _num(size) or 0.0))   # amd-smi "MB" is MiB
            return out
    except (OSError, subprocess.TimeoutExpired, json.JSONDecodeError):
        return None
    return None


def amd_sample_devices() -> list[dict]:
    """Per-device readings in GpuCollector's dict shape."""
    devices: list[dict] = []
    for g in _amd_json("metric", "--usage", "--power", "--clock",
                       "--temperature", "--mem-usage"):
        usage = g.get("usage") or {}
        mem = g.get("mem_usage") or {}
        power = g.get("power") or {}
        temp = g.get("temperature") or {}
        clocks = g.get("clock") or {}
        gfx = [_num((c or {}).get("clk")) for k, c in clocks.items()
               if k.startswith("gfx_") and isinstance(c, dict)]
        gfx = [c for c in gfx if c is not None]
        used, total = _num(mem.get("used_vram")), _num(mem.get("total_vram"))
        devices.append({
            "index": int(g.get("gpu", len(devices))),
            "sm_util_pct": _num(usage.get("gfx_activity")),
            "mem_util_pct": _num(usage.get("umc_activity")),
            "vram_used_gb": used / 1024.0 if used is not None else None,
            "vram_total_gb": total / 1024.0 if total is not None else None,
            "power_w": _num(power.get("socket_power")),
            "sm_clock_mhz": max(gfx) if gfx else None,
            "throttled": None,
            "temperature_c": _num(temp.get("hotspot")) or _num(temp.get("edge")),
        })
    return devices


# Stock CUDA images capsim defaults to, and their ROCm counterparts for
# gfx950 (MI350/MI355). Overridable per engine via env so a newer tag
# needs no code change.
_CUDA_DEFAULTS = {
    "vllm/vllm-openai:latest", "lmsysorg/sglang:latest",
    "ghcr.io/ggml-org/llama.cpp:server-cuda",
}
ROCM_IMAGES = {
    "vllm": os.environ.get("CAPSIM_ROCM_VLLM_IMAGE", "vllm/vllm-openai-rocm:v0.33.0-rocm100"),
    "sglang": os.environ.get("CAPSIM_ROCM_SGLANG_IMAGE",
                             "lmsysorg/sglang:v0.5.23-rocm10-mi35x"),
    "llamacpp": os.environ.get("CAPSIM_ROCM_LLAMACPP_IMAGE",
                               "ghcr.io/ggml-org/llama.cpp:server-rocm"),
}


# capsim engine key -> ROCm image family. Engines absent here have no
# ROCm build and are not offered on an AMD host.
ROCM_ENGINE_FAMILY = {"vllm_cuda_multi": "vllm", "vllm_cuda": "vllm",
                      "sglang_cuda": "sglang", "llamacpp": "llamacpp"}


def engine_image(family: str, image: Optional[str]) -> Optional[str]:
    """The image to run for engine ``family`` ("vllm" | "sglang" |
    "llamacpp"). On an AMD host the stock CUDA default becomes the ROCm
    build; anything the operator chose explicitly is left alone."""
    if not is_amd():
        return image
    if image is None or image in _CUDA_DEFAULTS:
        return ROCM_IMAGES.get(family, image)
    return image
