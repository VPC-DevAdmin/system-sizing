"""Pinned diffusion assets and a bounded prompt corpus for P05."""

import hashlib
import json
import shutil
import subprocess
from pathlib import Path

from ..benchmarks.storage import write_json
from ..models import hf_cache_dir
from .dynamo import DYNAMO_IMAGE

MODELS = {
    "baseline": ("Qwen/Qwen-Image", "75e0b4be04f60ec59a75f475837eced720f823b6"),
    "treatment": ("nvidia/Qwen-Image-Flash", "eafac15f6140e6dd9c6031217d658ac10bfb604b"),
}
IMAGE = "capsim-diffusion:qwen-flash-v1"
DOCKERFILE = f"""FROM {DYNAMO_IMAGE}
USER root
RUN python3 -m venv --system-site-packages /opt/capsim-diffusion && /opt/capsim-diffusion/bin/pip install --no-cache-dir diffusers==0.38.0 transformers==5.12.1 accelerate==1.12.0
ENV HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
ENTRYPOINT ["/opt/capsim-diffusion/bin/python"]
"""
PROMPTS = [
    "A studio product photograph of one red ceramic mug with a curved handle, on a plain white background. No text.",
    "A studio product photograph of two green glass bottles side by side, on a plain beige background. No labels or text.",
    "A catalog photograph of a blue backpack with black straps, front view, on a plain white background. No text.",
    "A catalog photograph of a yellow hard hat on a gray workbench, clean industrial setting, no people or text.",
    "A square advertisement concept showing three oranges in a white bowl on a wooden table. No text.",
    "A studio photograph of a black desk lamp with a circular base on a white desk. No text.",
    "A clean product photograph of a pair of white running shoes with blue laces, on a gray background. No text.",
    "An overhead photograph of a white plate holding a croissant beside a cup of black coffee. No text.",
]


def corpus(spec):
    return [
        {
            "id": f"image-{i:04d}",
            "seed": (spec.seed + i) % 2**32,
            "prompt": PROMPTS[i % len(PROMPTS)],
        }
        for i in range(spec.diffusion_samples)
    ]


def prepare(directory):
    """Idempotent local preparation; pinned models and image build evidence retained."""
    from huggingface_hub import hf_hub_download, snapshot_download

    cache = hf_cache_dir() / "hub"
    cache.mkdir(parents=True, exist_ok=True)
    expected = json.loads(Path(__file__).with_name("diffusion_files.json").read_text())
    missing = False
    for arm, (model, revision) in MODELS.items():
        for name in expected[arm]:
            try:
                hf_hub_download(
                    model, name, revision=revision, cache_dir=str(cache), local_files_only=True
                )
            except OSError:
                missing = True
    reserve = (150 if missing else 20) * 1024**3
    if shutil.disk_usage(cache).free < reserve:
        raise RuntimeError(
            f"Diffusion preparation requires {reserve // 1024**3} GiB free in the model cache"
        )
    assets = {}
    for arm, (model, revision) in MODELS.items():
        path = Path(
            snapshot_download(
                model, revision=revision, cache_dir=str(cache), allow_patterns=expected[arm]
            )
        )
        for name in expected[arm]:
            file = path / name
            if not file.is_file() or file.stat().st_size == 0:
                raise RuntimeError("Missing pinned diffusion asset: " + name)
        if path.name != revision:
            raise RuntimeError("Diffusion revision resolution mismatch")
        assets[arm] = {
            "model": model,
            "revision": revision,
            "snapshot": str(path),
            "files": {
                str(p.relative_to(path)): {"bytes": p.stat().st_size, "blob": p.resolve().name}
                for p in path.rglob("*")
                if p.is_file()
            },
        }
    build = directory / "build"
    build.mkdir(exist_ok=True)
    (build / "Dockerfile").write_text(DOCKERFILE)
    with (directory / "image-build.log").open("w") as log:
        subprocess.run(
            ["docker", "build", "-t", IMAGE, str(build)],
            stdout=log,
            stderr=subprocess.STDOUT,
            check=True,
            timeout=1800,
        )
    image = subprocess.check_output(
        ["docker", "image", "inspect", IMAGE, "--format", "{{.Id}}"], text=True, timeout=15
    ).strip()
    assets["image"] = image
    assets["recipe_sha256"] = hashlib.sha256(DOCKERFILE.encode()).hexdigest()
    write_json(directory / "assets.json", assets)
    return assets


def raw_comparison(results):
    groups = {}
    for row in results:
        groups.setdefault(f"{row['arm']}-{row['steps']}", []).append(row["metrics"])
    return {
        "qualified": False,
        "reason": "Image acceptance review required; raw timing is exploratory",
        "scope": "1024x1024 Qwen-Image versus its released FastGen-distilled student; one image at a time",
        "metrics": {},
        "raw_generation": {
            k: {
                "generated": sum(r["completed"] for r in v),
                "attempted": sum(r["attempted"] for r in v),
                "images_per_hour": 3600
                * sum(r["completed"] for r in v)
                / sum(r["elapsed_s"] for r in v),
            }
            for k, v in groups.items()
            if sum(r["elapsed_s"] for r in v) > 0
        },
    }


if __name__ == "__main__":
    import sys

    prepare(Path(sys.argv[1]))
