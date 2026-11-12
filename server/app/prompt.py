"""SP01 시스템 프롬프트 + 모델 입력 조립.

학습(v7)·RKLLM 양자화에 쓴 형식과 한 글자도 다르면 안 된다.
- prompts/sp01.txt : v7 학습에 쓴 SP01 원문 (19종 Action, answer_question 포함). 줄바꿈 없는 한 단락.
- build_user_content() : 학습 JSONL의 user 메시지와 같은 형식
    "[CONTEXT]\n<압축 JSON>\n[USER]\n<발화>"
  (JSON은 separators=(",", ":"), ensure_ascii=False. JSON 바로 다음 줄에 [USER], 빈 줄 없음 — 민재 님 확정 규격.)
형식을 바꾸면 tests/test_prompt.py(학습 JSONL 샘플과 비교)가 깨지도록 해 두었다.
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
    ctx = json.dumps(context, ensure_ascii=False, separators=(",", ":"))
    return f"[CONTEXT]\n{ctx}\n[USER]\n{user_text}"


def build_messages(user_text: str, context: dict) -> list[dict]:
    return [
        {"role": "system", "content": SP01},
        {"role": "user", "content": build_user_content(user_text, context)},
    ]
