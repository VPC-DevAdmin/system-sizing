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
from pathlib import Path
from typing import Optional

from ..config import EngineConfig
from ..search import assign_devices
from .knobs import (
    GGUF_ENGINES,
    GPU_ENGINES,
    canonical,
    to_engine_config,
    unsupported,
)
from .vram import weights_per_gpu_gb

LEVER_PREFIXES = ("trtllm_", "sglang_", "ktransformers_", "llamacpp_")

# Which engine owns each lever family. A lever whose prefix belongs to
# another engine is meaningless on this candidate: the search collapses
# it to its default (search.normalize) and the driver never passes it.
ENGINE_LEVER_PREFIX = {
    "trtllm": "trtllm_",
    "sglang_cuda": "sglang_",
    "ktransformers": "ktransformers_",
    "llamacpp": "llamacpp_",
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
    if kind == "bool | None":
        # Tri-state: "auto" (or an empty value) leaves the engine to
        # decide; everything else is the bool below.
        if value is None or (isinstance(value, str)
                             and value.strip().lower() in ("", "auto")):
            return None
        kind = "bool"
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


def _catalog_entry(model_id: str, catalog: Optional[list[dict]]) -> dict:
    try:
        if catalog is None:
            from ..model_catalog import load_model_catalog
            catalog = load_model_catalog()
        return next((e for e in catalog if e.get("id") == model_id), None) or {}
    except Exception:  # noqa: BLE001
        return {}


def _kt_only(model_id: str, catalog: Optional[list[dict]]) -> bool:
    """Does the catalog mark ``model_id`` as beyond the GPUs (``kt_only``:
    served by the GGUF engines alone)?"""
    return bool(_catalog_entry(model_id, catalog).get("kt_only"))


def gguf_engine_excluded(engine_type: str, model_id: str,
                         catalog: Optional[list[dict]]) -> Optional[str]:
    """Why a GGUF engine may not run this model's companion, or None.

    A companion's ``engines`` allow-list narrows the pair when one
    engine cannot load the architecture: the KTransformers v0.3.2
    image has injection rules for deepseek_v3 and qwen3_moe only, so
    DeepSeek-V3.2, Kimi-K2 and GLM-5.3 name llamacpp alone. A cell
    scheduled against the list would die at load, after the 30-minute
    health timeout."""
    spec = _catalog_entry(model_id, catalog).get("gguf") or {}
    allowed = spec.get("engines")
    if not spec or not allowed or engine_type in allowed:
        return None
    return (f"the catalog marks its GGUF companion for "
            f"{', '.join(allowed)} only")


def llamacpp_gguf_missing(gguf_path: object) -> Optional[str]:
    """Why a llama-server launch cannot proceed, or None when it can:
    the GGUF directory must exist and hold a loadable file (one .gguf,
    or a first shard) -- checked here so a half-staged quant is refused
    in milliseconds rather than after the health timeout."""
    if not gguf_path:
        return ("llama-server loads weights from GGUF, and no "
                "llamacpp_gguf_path is configured; stage the model's GGUF "
                "companion (e.g. the unsloth/*-GGUF repo) and point "
                "llamacpp_gguf_path at its directory")
    if not Path(str(gguf_path)).is_dir():
        return (f"llamacpp_gguf_path {gguf_path!r} is not a directory on "
                "this host")
    from .llamacpp import gguf_entry_file
    try:
        gguf_entry_file(gguf_path)
    except ValueError as e:
        return str(e)
    return None


def ktransformers_gguf_missing(gguf_path: object) -> Optional[str]:
    """Why a KTransformers launch cannot proceed, or None when it can.

    The v0.3.2 server loads weights from GGUF only (the HF directory
    supplies config and tokenizer). Without ``--gguf_path`` it falls
    back to a hard-coded ``./DeepSeek-V2-Lite-Chat-GGUF``, raises
    FileNotFoundError, and its scheduler process then LINGERS instead
    of exiting -- so an unstaged launch used to burn the whole
    30-minute health timeout per roofline cell (observed on the
    XE7740, 2026-09-20). Refuse up front instead.
    """
    if not gguf_path:
        return ("KTransformers loads weights from GGUF, and no "
                "ktransformers_gguf_path is configured; stage a GGUF of the "
                "model (e.g. the unsloth/*-GGUF repo) and point "
                "ktransformers_gguf_path at its directory")
    if not Path(str(gguf_path)).exists():
        return (f"ktransformers_gguf_path {gguf_path!r} does not exist on "
                "this host")
    return None


def resolve_gguf_companion(model_id: str,
                           catalog: Optional[list[dict]] = None,
                           engine: str = "ktransformers") -> Optional[str]:
    """The staged GGUF companion's directory for ``model_id``, from
    the catalog entry's ``gguf`` block -- None when the model has no
    companion (the caller then falls back to the generic "configure
    <engine>_gguf_path" refusal). One directory serves both GGUF
    engines: KTransformers reads every .gguf under it, llama-server
    opens the first shard.

    A companion that exists but is not staged raises ShapeError
    naming the fix (Prepare's "Download GGUF" button): telling the
    operator to point <engine>_gguf_path somewhere would be the
    wrong advice for a model capsim already knows how to stage."""
    from ..models import gguf_status
    if catalog is not None:
        entry = next((e for e in catalog if e.get("id") == model_id), None)
        status = gguf_status(entry) if entry else None
    else:
        status = gguf_status(model_id)
    if status is None:
        return None
    if not status["cached"]:
        raise ShapeError(
            f"{engine} cannot run {model_id} — its GGUF companion "
            f"{status['repo']}/{status['file']} is not staged; stage the "
            f"GGUF companion in Prepare (Download GGUF) or set "
            f"{engine}_gguf_path explicitly")
    return status["path"]


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
                 if custom.get("placement") in ("pack", "spread", "span")
                 else "pack")
    if hw is None:
        # Through the module attribute: tests pin the topology by
        # monkeypatching arena.hardware.
        from .. import arena
        hw = arena.hardware()
    if not hw.get("count"):
        raise ShapeError("custom engine shapes need a GPU host")
    engine_type = str(custom.get("engine") or "vllm_cuda_multi")
    if engine_type == "llamacpp" and replicas == 1 and tp <= 1:
        # llama-server has no tensor parallel; its default is a
        # pipelined split by layer across every GPU it sees, and that
        # split crosses PCIe domains without an all-reduce. So one
        # replica at "tp1" means the whole box, not one card -- the
        # shape the roofline asks for. A tp > 1 confines the replica
        # to that many cards inside one domain, as for any engine.
        devices = [sorted(d for g in hw["device_groups"] for d in g)]
    else:
        devices = assign_devices(tp, replicas, placement, hw["device_groups"])
    if devices is None:
        raise ShapeError(
            f"{replicas} replicas × tp{tp} does not fit "
            f"{hw['count']} GPUs in domains {hw['device_groups']}")
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
    if engine_type not in GGUF_ENGINES and _kt_only(model_id, catalog):
        # The catalog says the weights exceed the GPUs outright; a
        # vLLM launch would spend the 30-minute health timeout
        # discovering that. Only the CPU-expert engines run it.
        raise ShapeError(
            f"{engine_type} cannot run {model_id} — the catalog marks it "
            f"kt_only (weights beyond the GPUs); only the GGUF engines "
            f"({', '.join(GGUF_ENGINES)}) serve it")
    if engine_type in GGUF_ENGINES:
        why = gguf_engine_excluded(engine_type, model_id, catalog)
        if why and not custom.get(f"{engine_type}_gguf_path"):
            # An explicit path is the operator's own GGUF, outside the
            # catalog's judgement; the companion's allow-list binds
            # only the companion.
            raise ShapeError(f"{engine_type} cannot run {model_id} — {why}")
        key = f"{engine_type}_gguf_path"
        gguf_path = custom.get(key)
        if not gguf_path:
            gguf_path = resolve_gguf_companion(model_id, catalog, engine_type)
            if gguf_path:
                levers[key] = gguf_path
        missing = (ktransformers_gguf_missing if engine_type == "ktransformers"
                   else llamacpp_gguf_missing)
        why = missing(gguf_path)
        if why:
            raise ShapeError(f"{engine_type} cannot run {model_id} — {why}")

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
