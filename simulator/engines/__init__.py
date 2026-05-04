"""Engine launchers."""

from .base import Engine
from .vllm import VLLMEngine
from .sglang import SGLangEngine
from .vllm_dual_socket import VllmDualSocketEngine


def make_engine(engine_type: str, config) -> Engine:
    if engine_type == "vllm":
        return VLLMEngine(config)
    if engine_type == "sglang":
        return SGLangEngine(config)
    if engine_type == "vllm_dual_socket":
        return VllmDualSocketEngine(config)
    raise ValueError(f"Unknown engine type: {engine_type}")


__all__ = [
    "Engine",
    "VLLMEngine",
    "SGLangEngine",
    "VllmDualSocketEngine",
    "make_engine",
]
