"""모델 출력 텍스트 → Action 객체. (코드펜스·앞뒤 설명문이 섞여도 첫 JSON 객체만 추출)"""
import json
import re

from pydantic import ValidationError

from .schemas import ACTION_ADAPTER, Action

_FENCE = re.compile(r"^```(?:json)?\s*|\s*```$", re.IGNORECASE)


class ParseError(Exception):
    pass


def extract_json(text: str) -> dict:
    cleaned = _FENCE.sub("", (text or "").strip())
    start = cleaned.find("{")
    if start < 0:
        raise ParseError("JSON 객체를 찾지 못함")
    try:
        obj, _ = json.JSONDecoder().raw_decode(cleaned[start:])
    except json.JSONDecodeError as e:
        raise ParseError(f"JSON 파싱 실패: {e.msg}") from e
    if not isinstance(obj, dict):
        raise ParseError("최상위가 객체가 아님")
    return obj


def parse_action(text: str) -> Action:
    """ParseError(JSON 깨짐) 또는 ValidationError(스키마 위반)를 던진다."""
    return ACTION_ADAPTER.validate_python(extract_json(text))


__all__ = ["ParseError", "ValidationError", "extract_json", "parse_action"]
