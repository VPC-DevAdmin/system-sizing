"""One builder for a custom engine shape, shared by every launch path.

The benchmark form, the roofline and the arena driver all start from
the same request: a model, an engine, a replica count, a TP width and
the canonical knobs (``engines/knobs.py``) plus any engine-specific
levers (``engine_notes.py``). Until this module existed the service
turned that into an engine config while the arena driver kept a
private copy of the docker argv construction -- and the copy drifted:
it never applied the TensorRT-LLM levers the search enumerates, and
it passed vLLM's share-of-total-VRAM number straight through to
engines whose flag means something else (``engines/vram.py``).

``custom_engine`` is the pure part: a request dict in, the ``engine``
section of a config out, ``ShapeError`` when the request cannot be
honoured on this host. The service wraps the error as a 422; the
driver records the candidate as unreachable without launching.
"""

from __future__ import annotations

from dataclasses import fields as _fields
from typing import Optional

from ..config import EngineConfig
from ..search import assign_devices
from .knobs import GPU_ENGINES, canonical, to_engine_config, unsupported
from .vram import weights_per_gpu_gb

LEVER_PREFIXES = ("trtllm_", "sglang_", "ktransformers_")

# Which engine owns each lever family. A lever whose prefix belongs to
# another engine is meaningless on this candidate: the search collapses
# it to its default (search.normalize) and the driver never passes it.
ENGINE_LEVER_PREFIX = {
    "trtllm": "trtllm_",
    "sglang_cuda": "sglang_",
    "ktransformers": "ktransformers_",
}


def lever_owner(key: str) -> Optional[str]:
    """The engine a lever name belongs to, or None for a plain knob."""
    for engine, prefix in ENGINE_LEVER_PREFIX.items():
        if key.startswith(prefix):
            return engine
    return None

# Engine-specific levers are real EngineConfig fields, not flags, and
# only names the dataclass declares are accepted -- so a typo in a
# request cannot inject a silent setting.
LEVER_FIELDS: dict[str, str] = {
    f.name: str(f.type) for f in _fields(EngineConfig)
    if f.name.startswith(LEVER_PREFIXES)
}


class ShapeError(ValueError):
    """The requested shape cannot be built as specified."""


def _coerce_lever(name: str, value):
    """A lever value in the type its EngineConfig field declares.

    The arena spells its categorical dimensions as strings ("on" /
    "off", "0" / "4"); the benchmark form sends JSON booleans and
    numbers. ``llm_api_options`` tests the bool levers with a bare
    ``if``, so the string "off" arriving on a bool field would switch
    the lever ON -- coerce rather than trust.
    """
    kind = LEVER_FIELDS.get(name, "")
    if kind == "bool":
        if isinstance(value, str):
            return value.strip().lower() in ("on", "true", "1", "yes")
        return bool(value)
    if kind == "int":
        try:
            return int(value)
        except (TypeError, ValueError) as e:
            raise ShapeError(f"{name} must be an integer, got {value!r}") from e
    return value


def levers_from(custom: dict) -> dict:
    """The engine-specific levers a request carries, typed."""
    return {k: _coerce_lever(k, v) for k, v in custom.items()
            if k in LEVER_FIELDS and v not in (None, "")}


def _model_facts(model_id: str, tp: int,
                 catalog: Optional[list[dict]]) -> tuple[Optional[float],
                                                          Optional[str]]:
    """(weights per GPU in GB, catalog precision label) -- inputs to
    the one-memory-knob translation. Unknown is (None, None), which
    every engine treats as "use your own default", never a guess."""
    try:
        if catalog is None:
            from ..model_catalog import load_model_catalog
            catalog = load_model_catalog()
        for e in catalog:
            if e.get("id") == model_id:
                return (weights_per_gpu_gb(e.get("approx_size_gb"), tp),
                        e.get("quant"))
    except Exception:  # noqa: BLE001
        pass
    return None, None


def cpu_engine(custom: dict) -> dict:
    """Conservative CPU-only vLLM -- for boxes without GPUs (or explicit
    CPU comparisons). No searched dimensions apply."""
    model_id = str(custom.get("model_id") or "")
    if "/" not in model_id:
        raise ShapeError("custom.model_id must be an org/name id")
    return {
        "type": "vllm",
        "model_id": model_id,
        "max_model_len": int(custom.get("max_model_len") or 8192),
        "vllm_extra_flags": (["--trust-remote-code"]
                             if custom.get("trust_remote_code") else []),
        "port": 9100,
        "host": "127.0.0.1",
        "startup_timeout_s": 1800,
    }


def custom_engine(custom: dict, *, hw: Optional[dict] = None,
                  catalog: Optional[list[dict]] = None) -> dict:
    """The ``engine`` section for a custom shape.

    ``hw`` is ``{"count", "device_groups", "vram_per_gpu_gb"}`` --
    the detected topology (``arena.hardware()``) when omitted, or a
    search space's pinned groups when the driver calls. ``catalog``
    likewise defaults to the packaged model catalog.
    """
    model_id = str(custom.get("model_id") or "")
    if "/" not in model_id:
        raise ShapeError("custom.model_id must be an org/name id")
    if custom.get("device") == "cpu":
        return cpu_engine(custom)

    replicas = int(custom.get("replicas") or 1)
    tp = int(custom.get("tp") or 1)
    placement = (custom.get("placement")
                 if custom.get("placement") in ("pack", "spread") else "pack")
    if hw is None:
        # Through the module attribute: tests pin the topology by
        # monkeypatching arena.hardware.
        from .. import arena
        hw = arena.hardware()
    if not hw.get("count"):
        raise ShapeError("custom engine shapes need a GPU host")
    devices = assign_devices(tp, replicas, placement, hw["device_groups"])
    if devices is None:
        raise ShapeError(
            f"{replicas} replicas × tp{tp} does not fit "
            f"{hw['count']} GPUs in domains {hw['device_groups']}")
    engine_type = str(custom.get("engine") or "vllm_cuda_multi")
    if engine_type not in GPU_ENGINES:
        raise ShapeError(
            f"unknown engine {engine_type!r} — expected one of "
            f"{', '.join(GPU_ENGINES)}")

    levers = levers_from(custom)
    knobs = canonical(custom)
    weights, model_quant = _model_facts(model_id, tp, catalog)
    why = unsupported(engine_type, knobs)
    if why:
        # Refuse rather than approximate: measuring "close enough" here
        # answers a different question than the one asked.
        raise ShapeError(f"{engine_type} cannot run this shape — {why}")

    engine: dict = {
        "model_id": model_id,
        "tensor_parallel_size": tp,
        "port": 9100,
        "host": "127.0.0.1",
        "startup_timeout_s": 1800,
        # Inputs to the one-memory-knob translation: the operator sets
        # a share of TOTAL VRAM and each engine gets whatever its own
        # flag needs to mean the same allocation (engines/vram.py).
        "vram_per_gpu_gb": hw.get("vram_per_gpu_gb"),
        "model_weights_gb": weights,
        "model_quant": model_quant,
        **to_engine_config(engine_type, knobs),
        **levers,
    }
    if custom.get("served_model_name"):
        engine["served_model_name"] = str(custom["served_model_name"])
    if engine_type != "vllm_cuda_multi":
        # Every non-vLLM engine is a DockerReplicaEngine, so one code
        # path covers any replica count -- even a single replica is
        # described by replica_devices.
        #
        # This MUST NOT be a list of known engines with a fall-through
        # to vLLM. It was, and adding SGLang and KTransformers to the
        # picker silently routed both to vllm_cuda_multi: the runs
        # launched, measured and reported as though the requested
        # engine had been used. Anything not explicitly vLLM keeps its
        # own type, so a new engine cannot be quietly absorbed again.
        engine["type"] = engine_type
        engine["replica_devices"] = devices
    elif replicas > 1:
        engine["type"] = "vllm_cuda_multi"
        engine["gpu_image"] = "vllm/vllm-openai:latest"
        engine["replica_devices"] = devices
    else:
        engine["type"] = "vllm_cuda"
        engine["gpu_image"] = "vllm/vllm-openai:latest"
        engine["gpu_device_ids"] = devices[0]
    return engine


def config_doc(engine: dict, runs_base) -> dict:
    """The whole config document around an engine section -- same
    schema as a promoted profile."""
    if engine.get("type") == "vllm":
        telemetry = {"enable_engine_metrics": True}
    else:
        telemetry = {"enable_pmu": True, "enable_memory_bandwidth": True,
                     "enable_power": True, "enable_engine_metrics": True,
                     "enable_gpu": True}
    return {
        "engine": engine,
        "telemetry": telemetry,
        "output": {"db_directory": str(runs_base)},
    }
