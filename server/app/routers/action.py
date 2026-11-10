from fastapi import APIRouter, Request

from ..answer_texts import ANSWER_TEXTS
from ..schemas import ActionRequest, ActionResponse

router = APIRouter(prefix="/api", tags=["action"])


@router.post("/action", response_model=ActionResponse)
async def post_action(req: ActionRequest, request: Request) -> ActionResponse:
    """사용자 발화 + 현재 화면 Context → Action JSON.

    실패(타임아웃·검증 실패 등)도 HTTP 200으로 돌려주고 `fallback: true`로 알린다.
    """
    return await request.app.state.service.handle(req)


@router.get("/answer-texts")
def get_answer_texts() -> dict[str, str]:
    """answer_question의 answer_code → 안내 문구 표 (프론트와 공유)."""
    return ANSWER_TEXTS
