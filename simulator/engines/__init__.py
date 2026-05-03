"""Engine launchers."""

from .base import Engine
from .vllm import VLLMEngine
from .sglang import SGLangEngine


def make_engine(engine_type: str, config) -> Engine:
    if engine_type == "vllm":
        return VLLMEngine(config)
    if engine_type == "sglang":
        return SGLangEngine(config)
    raise ValueError(f"Unknown engine type: {engine_type}")


__all__ = ["Engine", "VLLMEngine", "SGLangEngine", "make_engine"]
