"""발화 → Action 처리 파이프라인.

요청 → 화면 정규화 → (연속 2회 실패 규칙) → 모델 백엔드(타임아웃) → JSON 추출 → 스키마 검증
     → 메뉴 정책 검증/보정 → 응답. 실패하면 fallback 응답(프론트가 정규식으로 처리).
"""
import asyncio
import logging
import time
from typing import Optional

from pydantic import ValidationError

from . import db_hooks
from .answer_texts import ANSWER_TEXTS
from .backends.base import GenerationRequest, ModelBackend
from .fallback import fallback_response
from .parser import ParseError, parse_action
from .policy import PolicyViolation, apply_policy
from .prompt import build_messages, redact_digits
from .schemas import ActionRequest, ActionResponse, CallStaff

log = logging.getLogger("onkiosk.service")


def _model_context(ctx: dict) -> dict:
    """모델에 넣을 Context. 값이 0/비어 있는 부가 키는 학습 데이터처럼 생략한다.

    TODO(학습 데이터와 키 구성 확인): 데이터셋 Context는 필요한 키만 담고 있다.
    """
    out = dict(ctx)
    if not out.get("recognition_failure_count"):
        out.pop("recognition_failure_count", None)
    if not out.get("draft"):
        out.pop("draft", None)
    return out


class ActionService:
    def __init__(self, backend: ModelBackend, timeout_s: float = 3.0):
        self.backend = backend
        self.timeout_s = timeout_s

    async def handle(self, req: ActionRequest) -> ActionResponse:
        raw_screen = req.context.screen
        screen = db_hooks.normalize_screen(raw_screen)
        ctx = req.context.model_dump(exclude_none=True)
        ctx["screen"] = screen
        if req.context.sold_out is None:
            ctx["sold_out"] = db_hooks.sold_out_menus()
        sold_out: list[str] = ctx["sold_out"]
        text = redact_digits(req.user_text)

        llm_ms: Optional[int] = None
        valid_schema = True

        # 1) 연속 2회 인식 실패 → 모델을 거치지 않고 직원 호출 (합의안)
        if int(ctx.get("recognition_failure_count") or 0) >= 2:
            action = CallStaff(action="call_staff", reason="VOICE_RECOGNITION_FAILED_TWICE")
            resp = ActionResponse(
                ok=True, fallback=False, action=action.model_dump(exclude_none=True),
                source="rule", screen=screen,
            )
            self._log(text, raw_screen, resp, True, req.stt_latency_ms, None)
            return resp

        # 2) 모델 호출 (타임아웃 / 백엔드 오류)
        model_ctx = _model_context(ctx)
        gen = GenerationRequest(text, model_ctx, build_messages(text, model_ctx))
        t0 = time.perf_counter()
        try:
            raw = await asyncio.wait_for(self.backend.generate(gen), self.timeout_s)
        except asyncio.TimeoutError:
            llm_ms = int((time.perf_counter() - t0) * 1000)
            resp = fallback_response("TIMEOUT", screen=screen, llm_latency_ms=llm_ms)
            self._log(text, raw_screen, resp, True, req.stt_latency_ms, llm_ms)
            return resp
        except Exception:
            log.exception("모델 백엔드 오류")
            llm_ms = int((time.perf_counter() - t0) * 1000)
            resp = fallback_response("BACKEND_ERROR", screen=screen, llm_latency_ms=llm_ms)
            self._log(text, raw_screen, resp, True, req.stt_latency_ms, llm_ms)
            return resp
        llm_ms = int((time.perf_counter() - t0) * 1000)

        # 3) JSON 추출 + 스키마 검증
        try:
            action = parse_action(raw)
        except ParseError as e:
            resp = fallback_response("INVALID_JSON", str(e), screen=screen, llm_latency_ms=llm_ms)
            self._log(text, raw_screen, resp, False, req.stt_latency_ms, llm_ms)
            return resp
        except ValidationError as e:
            first = e.errors()[0]
            detail = f"{'.'.join(str(p) for p in first['loc'])}: {first['msg']}"
            resp = fallback_response("INVALID_SCHEMA", detail, screen=screen, llm_latency_ms=llm_ms)
            self._log(text, raw_screen, resp, False, req.stt_latency_ms, llm_ms)
            return resp

        # 4) 메뉴 정책 검증 / 안전 보정
        try:
            action, repairs = apply_policy(action, ctx, sold_out)
        except PolicyViolation as e:
            resp = fallback_response(
                "POLICY_VIOLATION", f"{e.code}: {e.message}", screen=screen, llm_latency_ms=llm_ms
            )
            self._log(text, raw_screen, resp, True, req.stt_latency_ms, llm_ms)
            return resp

        payload = action.model_dump(exclude_none=True)
        resp = ActionResponse(
            ok=True,
            fallback=False,
            action=payload,
            answer_text=ANSWER_TEXTS.get(payload.get("answer_code", "")),
            repairs=repairs,
            source="model",
            screen=screen,
            llm_latency_ms=llm_ms,
        )
        self._log(text, raw_screen, resp, valid_schema, req.stt_latency_ms, llm_ms)
        return resp

    @staticmethod
    def _log(text, raw_screen, resp: ActionResponse, valid_schema, stt_ms, llm_ms) -> None:
        db_hooks.log_inference(
            utterance=text,
            predicted_action=(resp.action or {}).get("action"),
            response_obj=resp.action,
            screen_context=raw_screen,
            is_valid_schema=valid_schema,
            fallback_triggered=resp.fallback,
            stt_latency_ms=stt_ms,
            llm_latency_ms=llm_ms,
        )
