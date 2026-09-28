"""Engine launchers."""

from .base import Engine
from .mock import MockEngine
from .remote import RemoteEngine
from .sglang import SGLangEngine
from .vllm import VLLMEngine
from .vllm_cuda import VllmCudaEngine
from .vllm_dual_socket import VllmDualSocketEngine


def _make_engine(engine_type: str, config) -> Engine:
    if engine_type == "vllm":
        return VLLMEngine(config)
    if engine_type == "vllm_cuda":
        return VllmCudaEngine(config)
    if engine_type == "vllm_cuda_multi":
        from .vllm_cuda_multi import VllmCudaMultiEngine
        return VllmCudaMultiEngine(config)
    if engine_type == "sglang_cuda":
        from .sglang_cuda import SGLangCudaEngine
        return SGLangCudaEngine(config)
    if engine_type == "ktransformers":
        from .ktransformers import KTransformersEngine
        return KTransformersEngine(config)
    if engine_type == "llamacpp":
        from .llamacpp import LlamaCppEngine
        return LlamaCppEngine(config)
    if engine_type == "trtllm":
        from .trtllm import TrtLlmEngine
        return TrtLlmEngine(config)
    if engine_type == "sglang":
        return SGLangEngine(config)
    if engine_type == "vllm_dual_socket":
        return VllmDualSocketEngine(config)
    if engine_type == "remote":
        return RemoteEngine(config)
    if engine_type == "mock":
        return MockEngine(config)
    raise ValueError(f"Unknown engine type: {engine_type}")


__all__ = [
    "Engine",
    "VLLMEngine",
    "VllmCudaEngine",
    "SGLangEngine",
    "VllmDualSocketEngine",
    "RemoteEngine",
    "MockEngine",
    "make_engine",
]


def make_engine(engine_type: str, config) -> Engine:
    """One physical host owner across CLI and service processes."""
    from ..benchmarks.process import Lease, hardware_lease_path

    engine = _make_engine(engine_type, config)
    if engine_type == "mock":
        return engine
    launch, shutdown = engine.launch, engine.shutdown
    lease = Lease(hardware_lease_path())
    held = False
    def guarded_launch(*args, **kwargs):
        nonlocal held
        if held:
            raise RuntimeError("Engine already owns the hardware lease")
        lease.__enter__()
        held = True
        try:
            return launch(*args, **kwargs)
        except BaseException:
            lease.__exit__()
            held = False
            raise
    def guarded_shutdown(*args, **kwargs):
        nonlocal held
        try:
            return shutdown(*args, **kwargs)
        finally:
            if held:
                lease.__exit__()
                held = False
    engine.launch = guarded_launch
    engine.shutdown = guarded_shutdown
    return engine
