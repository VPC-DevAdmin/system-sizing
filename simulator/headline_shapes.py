"""Per-model-family store of optimal headline shapes.

The shape search finds the (input, output) pair that jointly maximizes
concurrency and output throughput — but that optimum belongs to the
MODEL FAMILY (architecture + size), not to one run. This module keeps
a small JSON file (``config/headline_shapes.json``, beside the persona
overlay directory) mapping a normalized family key to the winning
shape, and applies a stored shape to the "Headline: Generation"
persona so the workload the user picks IS the optimized one.

Family normalization strips the org prefix and quantization/precision
suffixes: ``Qwen/Qwen3-30B-A3B-Instruct-2507`` and its ``-FP8``
sibling share one optimum — the KV/batch geometry that decides the
shape is set by the architecture, not the weight format.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import yaml

GENERATION_PERSONA_ID = "headline_generation"

# Headline workloads are saturation benchmarks, not capacity models —
# picking one swaps the whole measurement instrument (see
# simulator/headline_sweep.py), so the id prefix is load-bearing.
HEADLINE_PREFIX = "headline_"


def is_headline_persona(persona_id) -> bool:
    return bool(persona_id) and str(persona_id).startswith(HEADLINE_PREFIX)

# Weight-format / quantization tokens that do not change which shape
# is optimal — siblings differing only in these share a family.
_QUANT_TOKENS = {
    "fp8", "fp4", "nvfp4", "mxfp4", "int4", "int8", "awq", "gptq",
    "gguf", "bnb", "w8a8", "w8a16", "w4a16", "bf16", "fp16",
    "dynamic", "e4m3", "e5m2", "marlin", "4bit", "8bit", "quantized",
}


def model_family(model_id: str) -> str:
    base = (model_id or "").strip().split("/")[-1].lower()
    toks = [t for t in re.split(r"[-_.]+", base)
            if t and t not in _QUANT_TOKENS]
    return "-".join(toks) or base


def shapes_path(catalog_dir: Path) -> Path:
    """The store lives beside the persona overlay dir (config/)."""
    return Path(catalog_dir).parent / "headline_shapes.json"


def load_shapes(path: Path) -> dict:
    try:
        data = json.loads(Path(path).read_text())
        return data if isinstance(data, dict) else {}
    except (OSError, ValueError):
        return {}


def shape_for(path: Path, model_id: str) -> dict | None:
    return load_shapes(path).get(model_family(model_id))


def save_shape(path: Path, model_id: str, shape: dict) -> str:
    """Record a winning shape under the model's family key; returns
    the key. The full model_id is kept inside the record so the user
    can see which sibling produced it."""
    path = Path(path)
    shapes = load_shapes(path)
    family = model_family(model_id)
    shapes[family] = {**shape, "model_id": model_id}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(shapes, indent=2))
    return family


def apply_shape_to_generation(catalog_dir: Path, inp: int, out: int) -> None:
    """Make (inp, out) the Headline: Generation persona's shape.

    Serializes the current merged persona and overrides only the token
    distributions, so any other edits the user made to the workload
    survive. Written as an overlay file because load-generator worker
    subprocesses resolve personas from the catalog on disk.
    """
    from .persona_loader import USER_CATALOG_DIR, serialize_persona
    from .personas import PERSONAS, reload_personas

    catalog_dir = Path(catalog_dir)
    p = PERSONAS[GENERATION_PERSONA_ID]
    spec = serialize_persona(p)
    spec["input_tokens"] = {"constant": int(inp)}
    spec["output_tokens"] = {"constant": int(out)}
    catalog_dir.mkdir(parents=True, exist_ok=True)
    (catalog_dir / f"{GENERATION_PERSONA_ID}.yaml").write_text(
        yaml.safe_dump({"personas": {GENERATION_PERSONA_ID: spec}},
                       sort_keys=False))
    reload_personas(
        user_dir=None if catalog_dir == USER_CATALOG_DIR else catalog_dir)


def generation_shape() -> tuple[int, int] | None:
    """The persona's current shape, when it is a fixed one."""
    from .distributions import Constant
    from .personas import PERSONAS

    p = PERSONAS.get(GENERATION_PERSONA_ID)
    if p is None:
        return None
    if isinstance(p.input_tokens, Constant) \
            and isinstance(p.output_tokens, Constant):
        return int(p.input_tokens.value), int(p.output_tokens.value)
    return None
