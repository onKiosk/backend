"""HuggingFace(Llama-3.2-1B + LoRA) 백엔드 자리 — GPU 서버/노트북에서 모델 출력 확인용.

TODO: generate()에서 req.messages를 tokenizer.apply_chat_template로 변환해 생성하고 텍스트를 돌려준다.
      (블로킹 호출이므로 asyncio.to_thread 로 감쌀 것)
"""
from .base import GenerationRequest


class HFBackend:
    name = "hf"

    def __init__(self) -> None:
        raise NotImplementedError("HF 백엔드는 아직 구현 전입니다 (ONKIOSK_BACKEND=mock 사용)")

    async def generate(self, req: GenerationRequest) -> str:  # pragma: no cover
        raise NotImplementedError
