"""폴백 규칙 표.

합의안(최종): 서버가 정상적으로 Action을 돌려주면(no_action 포함) 프론트는 정규식 폴백을 실행하지 않는다.
타임아웃 / 프로세스 오류 / JSON·스키마 검증 실패 / 정책 위반일 때만 `fallback: true`를 보내고,
프론트는 그때만 기존 resolveIntent(정규식)로 처리한다.
"""
from typing import Optional

from .schemas import ActionResponse

# 오류 코드 → 사람이 읽는 설명(로그용). 모두 프론트 폴백 대상.
ERROR_TABLE: dict[str, str] = {
    "TIMEOUT": "추론이 제한 시간을 넘김",
    "BACKEND_ERROR": "모델 백엔드(프로세스) 오류",
    "INVALID_JSON": "모델 출력에서 JSON을 읽지 못함",
    "INVALID_SCHEMA": "모델 출력이 Action 스키마와 맞지 않음",
    "POLICY_VIOLATION": "메뉴 정책 위반(고칠 수 없는 경우)",
}


def fallback_response(
    error_code: str,
    detail: Optional[str] = None,
    *,
    screen: Optional[str] = None,
    llm_latency_ms: Optional[int] = None,
) -> ActionResponse:
    return ActionResponse(
        ok=False,
        fallback=True,
        action=None,
        error_code=error_code,
        detail=detail or ERROR_TABLE.get(error_code),
        screen=screen,
        llm_latency_ms=llm_latency_ms,
    )
