"""Engine launchers."""

from .base import Engine
from .mock import MockEngine
from .remote import RemoteEngine
from .sglang import SGLangEngine
from .vllm import VLLMEngine
from .vllm_cuda import VllmCudaEngine
from .vllm_dual_socket import VllmDualSocketEngine


def make_engine(engine_type: str, config) -> Engine:
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
