import os

from .base import GenerationRequest, ModelBackend
from .mock import MockBackend


def create_backend(name: str | None = None) -> ModelBackend:
    """ONKIOSK_BACKEND = mock(기본) | hf | rkllm"""
    name = (name or os.getenv("ONKIOSK_BACKEND", "mock")).lower()
    if name == "mock":
        return MockBackend()
    if name == "hf":
        from .hf import HFBackend

        return HFBackend()
    if name == "rkllm":
        from .rkllm import RKLLMBackend

        return RKLLMBackend()
    raise ValueError(f"알 수 없는 백엔드: {name}")


__all__ = ["GenerationRequest", "ModelBackend", "MockBackend", "create_backend"]
