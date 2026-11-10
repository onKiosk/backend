"""SP01 시스템 프롬프트 + 모델 입력 조립.

!! 학습 때 쓴 형식과 한 글자도 다르면 안 된다 !!
- prompts/sp01.txt 는 데이터셋 v5 README의 SP01 전문이다. v5.5(answer_question 추가)에서는 SP01이 바뀌었을 수 있으니
  학습에 실제로 쓴 SP01 파일로 교체해야 한다.
- build_user_content()의 형식(Context/발화를 어떻게 합치는지)은 민재 님의 학습 파이프라인(JSONL 변환 코드)과 맞춰야 한다.
  지금 형식은 임시(TODO) 값이다.
"""
import json
import re
from pathlib import Path

_SP01_PATH = Path(__file__).resolve().parent.parent / "prompts" / "sp01.txt"
SP01 = _SP01_PATH.read_text(encoding="utf-8").strip()

# 전화번호 등 숫자열이 모델 입력/로그로 가지 않도록 가린다 (스키마 개인정보 정책).
_DIGITS = re.compile(r"\d[\d\s\-]{2,}\d")


def redact_digits(text: str) -> str:
    return _DIGITS.sub("[숫자]", text)


def build_user_content(user_text: str, context: dict) -> str:
    # TODO(학습 형식 확인): 학습 JSONL의 user 메시지 구성과 동일하게 맞출 것
    ctx = json.dumps(context, ensure_ascii=False, separators=(",", ":"))
    return f"Context: {ctx}\n사용자 발화: {user_text}"


def build_messages(user_text: str, context: dict) -> list[dict]:
    return [
        {"role": "system", "content": SP01},
        {"role": "user", "content": build_user_content(user_text, context)},
    ]
