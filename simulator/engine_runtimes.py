"""Which engine runtimes this host can actually launch.

The arena's rule is that feasibility is derived, not declared: a
model never appears at a TP that its weights cannot fit. Engines earn
the same treatment. An engine whose container image is not on the box
is not a choice the operator has — offering it would mean a search
that spends its first candidate discovering a 59 GB download it
cannot do mid-run.

So the image is the gate, and staging one in the Prepare phase is
what makes that engine appear in Optimize and Benchmark. The
dependency runs one way and is visible in the UI.

Image size matters here in a way it usually doesn't: TensorRT-LLM's
release image is ~59 GB extracted. On a box whose container store
sits on a small root volume this is the difference between a working
pull and a wedged host, which is why the status includes where the
image store actually lives.
"""

from __future__ import annotations

import logging
import shutil
import subprocess

log = logging.getLogger(__name__)

# Engines the benchmark/optimizer may choose between. ``image`` is the
# gate; ``approx_gb`` is the extracted size, for the pull warning.
RUNTIMES: dict[str, dict] = {
    "vllm_cuda_multi": {
        "label": "vLLM",
        "image": "vllm/vllm-openai:latest",
        "approx_gb": 31,
        "blurb": "The default. Broad model coverage and the engine "
                 "every existing capsim result was measured on.",
    },
    "sglang_cuda": {
        "label": "SGLang",
        "image": "lmsysorg/sglang:latest",
        "approx_gb": 22,
        "blurb": "RadixAttention prefix caching and an aggressive "
                 "scheduler. Often the strongest on workloads with "
                 "shared prefixes, which is most chat traffic.",
    },
    "ktransformers": {
        "label": "KTransformers",
        "image": "approachingai/ktransformers:latest",
        "approx_gb": 18,
        "blurb": "Heterogeneous: MoE experts on the CPU, attention on "
                 "the GPU. Serves models far larger than VRAM — the "
                 "reason this box has 2 TB of RAM and AMX.",
    },
    "trtllm": {
        "label": "TensorRT-LLM",
        "image": "nvcr.io/nvidia/tensorrt-llm/release:1.2.1",
        "approx_gb": 59,
        "blurb": "NVIDIA's own server. Often faster on NVIDIA silicon, "
                 "and the engine vendor headline numbers are quoted "
                 "on — worth measuring rather than assuming.",
    },
}


def local_images() -> set[str]:
    """Image tags present in the local container store."""
    if shutil.which("docker") is None:
        return set()
    try:
        r = subprocess.run(
            ["docker", "images", "--format", "{{.Repository}}:{{.Tag}}"],
            capture_output=True, text=True, timeout=30,
        )
    except (OSError, subprocess.SubprocessError) as e:
        log.debug("docker images failed: %s", e)
        return set()
    if r.returncode != 0:
        return set()
    return {ln.strip() for ln in r.stdout.splitlines() if ln.strip()}


def image_store_root() -> dict:
    """Where images actually land, and how much room is left there.

    Docker's ``data-root`` does NOT govern image content when the
    containerd snapshotter is in use — images live under containerd's
    root instead, which is a different filesystem on many hosts. A
    "plenty of space" reading from the wrong directory is how a 59 GB
    pull fills a root volume.
    """
    out: dict = {"path": None, "free_gb": None, "note": ""}
    if shutil.which("docker") is None:
        return out
    driver = ""
    try:
        r = subprocess.run(["docker", "info", "--format",
                            "{{.DriverStatus}}|{{.DockerRootDir}}"],
                           capture_output=True, text=True, timeout=30)
        driver, _, root = r.stdout.strip().partition("|")
        out["path"] = root or None
    except (OSError, subprocess.SubprocessError):
        return out
    if "containerd" in driver:
        # The containerd snapshotter ignores data-root for image
        # content. Its own root is the one that fills up.
        out["note"] = ("images are stored by containerd, not under "
                       "Docker's data-root")
    if out["path"]:
        try:
            usage = shutil.disk_usage(out["path"])
            out["free_gb"] = round(usage.free / 1e9, 1)
        except OSError:
            pass
    return out


def runtime_status(images: set[str] | None = None) -> list[dict]:
    """One row per engine: staged or not, with what a pull would cost."""
    from .engines.knobs import caveats
    have = local_images() if images is None else images
    rows = []
    for key, meta in RUNTIMES.items():
        rows.append({
            "engine": key,
            "label": meta["label"],
            "image": meta["image"],
            "approx_gb": meta["approx_gb"],
            "blurb": meta["blurb"],
            "staged": meta["image"] in have,
            # Where this engine's numbers are NOT interchangeable with
            # another's. Surfaced next to the choice, not buried in a
            # doc nobody reads mid-benchmark.
            "caveats": caveats(key),
        })
    return rows


def available_engines(images: set[str] | None = None) -> list[str]:
    """Engines with their image staged — what Optimize and Benchmark
    may offer. Never empty-guesses: an unstaged engine is simply not
    a choice."""
    return [r["engine"] for r in runtime_status(images) if r["staged"]]
