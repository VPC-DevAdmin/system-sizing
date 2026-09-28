"""Isolated Qwen-Image teacher/student inference; invoked inside the pinned image."""

import hashlib
import importlib.metadata
import json
import time
from pathlib import Path


def main():
    import numpy as np
    import torch
    from diffusers import QwenImagePipeline

    root = Path("/evidence")
    plan = json.loads((root / "plan.json").read_text())
    pipe = QwenImagePipeline.from_pretrained(
        plan["snapshot"], torch_dtype=torch.bfloat16, local_files_only=True
    ).to("cuda")
    pipe.set_progress_bar_config(disable=True)
    student = plan["arm"] == "treatment"
    settings = dict(
        width=1024,
        height=1024,
        num_inference_steps=plan["steps"],
        true_cfg_scale=1.0 if student else 4.0,
        guidance_scale=None,
        negative_prompt=None if student else "",
    )
    effective = {
        "settings": settings,
        "scheduler": dict(pipe.scheduler.config),
        "packages": {
            n: importlib.metadata.version(n) for n in ["torch", "diffusers", "transformers"]
        },
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0),
        "scope": "FastGen-distilled released student; Diffusers inference; no local distillation",
    }
    (root / "effective.json").write_text(json.dumps(effective, indent=2, default=str))

    def generate(prompt, seed):
        torch.cuda.synchronize()
        start = time.perf_counter()
        with torch.inference_mode():
            image = pipe(
                prompt=prompt,
                generator=torch.Generator(device="cuda").manual_seed(seed),
                **settings,
            ).images[0]
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        if student:
            sigmas = pipe.scheduler.sigmas.detach().cpu().float().tolist()
            if len(sigmas) != 5 or not np.allclose(sigmas, [1, 0.9, 0.75, 0.5, 0], atol=1e-5):
                raise RuntimeError(
                    "Student scheduler did not execute the published four-step trajectory"
                )
        return image, elapsed

    # Disjoint prompt and seed; warmup is retained but excluded from steady-state rate.
    _, warmup = generate("A plain blue ceramic bowl on a wooden table.", plan["seed"] ^ 65535)
    (root / "warmup.json").write_text(json.dumps({"elapsed_s": warmup}))
    torch.cuda.reset_peak_memory_stats()
    with (root / "images.jsonl").open("w") as out:
        for item in plan["prompts"]:
            row = {"id": item["id"], "seed": item["seed"], "prompt": item["prompt"], "ok": False}
            start = time.perf_counter()
            try:
                image, elapsed = generate(item["prompt"], item["seed"])
                pixels = np.asarray(image)
                if (
                    image.size != (1024, 1024)
                    or not np.isfinite(pixels).all()
                    or float(pixels.std()) < 1
                ):
                    raise ValueError("Generated image failed technical integrity check")
                target = root / (item["id"] + ".png")
                image.save(target)
                row.update(
                    ok=True,
                    elapsed_s=elapsed,
                    image=target.name,
                    sha256=hashlib.sha256(target.read_bytes()).hexdigest(),
                )
            except Exception as exc:
                row.update(
                    elapsed_s=time.perf_counter() - start,
                    error=type(exc).__name__ + ": " + str(exc),
                )
                if isinstance(exc, torch.cuda.OutOfMemoryError):
                    torch.cuda.empty_cache()
            out.write(json.dumps(row, allow_nan=False) + "\n")
            out.flush()
    (root / "memory.json").write_text(
        json.dumps(
            {
                "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
                "scope": "PyTorch allocation; not whole-server memory",
            }
        )
    )


if __name__ == "__main__":
    main()
