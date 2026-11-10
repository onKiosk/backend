"""모델 백엔드 공통 인터페이스. Mock → HF → RKLLM 순서로 같은 자리에 끼워 쓴다."""
from dataclasses import dataclass, field
from typing import Protocol


@dataclass
class GenerationRequest:
    user_text: str
    context: dict
    messages: list[dict] = field(default_factory=list)  # [{"role","content"}, ...] (SP01 포함)


class ModelBackend(Protocol):
    name: str

    async def generate(self, req: GenerationRequest) -> str:
        """모델 출력 텍스트(JSON 문자열)를 돌려준다. 파싱·검증은 서비스가 한다."""
        ...
