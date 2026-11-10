"""RKLLM(RK3588 NPU, W8A8) 백엔드 자리 — 보드 수령 후 구현.

TODO: rkllm 런타임 로드, 프롬프트 → 생성, 타임아웃 시 NPU 작업 취소 처리.
"""
from .base import GenerationRequest


class RKLLMBackend:
    name = "rkllm"

    def __init__(self) -> None:
        raise NotImplementedError("RKLLM 백엔드는 보드 수령 후 구현합니다 (ONKIOSK_BACKEND=mock 사용)")

    async def generate(self, req: GenerationRequest) -> str:  # pragma: no cover
        raise NotImplementedError
