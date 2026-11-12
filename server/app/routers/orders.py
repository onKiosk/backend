import logging
import re

from fastapi import APIRouter
from fastapi.responses import JSONResponse

from .. import db_hooks
from ..schemas import OrderRequest, OrderResponse, StampRequest, StampResponse

log = logging.getLogger("onkiosk.orders")
router = APIRouter(prefix="/api", tags=["orders"])

_STATUS = {"SOLD_OUT": 409, "INVALID_ORDER": 422, "DB_UNAVAILABLE": 503, "ORDER_FAILED": 500,
           "INVALID_PHONE": 422, "STAMP_FAILED": 500}


def _fail(model, code: str, detail: str) -> JSONResponse:
    return JSONResponse(status_code=_STATUS[code], content=model(ok=False, error_code=code, detail=detail).model_dump())


@router.post("/orders", response_model=OrderResponse)
def post_order(req: OrderRequest):
    """결제 확정 시 호출. 총액은 서버(DB 가격표)가 계산하고 주문번호(A001…)를 발급한다."""
    items = [i.model_dump(exclude_none=True) for i in req.items]
    try:
        out = db_hooks.create_order(items, req.payment_method, req.pickup_type, req.print_receipt)
    except db_hooks.DbUnavailable:
        return _fail(OrderResponse, "DB_UNAVAILABLE", "database.py의 create_order를 사용할 수 없습니다.")
    except db_hooks.OrderRejected as e:
        return _fail(OrderResponse, e.code, e.detail)
    except Exception:
        log.exception("create_order 실패")
        return _fail(OrderResponse, "ORDER_FAILED", "주문 생성 중 오류가 발생했습니다.")
    total = out.get("total_price")
    return OrderResponse(
        ok=True,
        order_number=out.get("order_number"),
        total_price=int(total) if isinstance(total, (int, float)) else None,
        order=out,
    )


@router.post("/stamps", response_model=StampResponse)
def post_stamp(req: StampRequest):
    """전화번호(4/11자리)로 스탬프 적립. 번호는 로그·응답·LLM에 남기지 않는다."""
    digits = re.sub(r"\D", "", req.phone_number)
    raw = req.phone_number.strip()
    ok_format = re.fullmatch(r"\d{4}|\d{11}|\d{3}-\d{4}-\d{4}", raw) is not None  # 4자리 / 11자리 / 010-1234-5678
    if not ok_format:
        return _fail(StampResponse, "INVALID_PHONE", "전화번호는 숫자 4자리 또는 11자리여야 합니다.")
    try:
        stamp = db_hooks.add_stamp(digits, req.count)
    except db_hooks.DbUnavailable:
        return _fail(StampResponse, "DB_UNAVAILABLE", "database.py의 add_stamp를 사용할 수 없습니다.")
    except Exception as e:
        log.error("add_stamp 실패: %s", type(e).__name__)  # 번호가 섞일 수 있어 메시지는 남기지 않는다
        return _fail(StampResponse, "STAMP_FAILED", "적립 중 오류가 발생했습니다.")
    return StampResponse(ok=True, stamp=stamp)


@router.get("/aliases")
def get_aliases():
    """별칭 사전 (예: 아아 → 아메리카노). DB가 없으면 빈 사전."""
    try:
        return db_hooks.alias_map()
    except db_hooks.DbUnavailable:
        return {}


@router.get("/sold-out")
def get_sold_out():
    """현재 품절 메뉴명 (프론트 화면 표시와 서버 판단을 맞추는 단일 출처)."""
    return {"sold_out": db_hooks.sold_out_menus()}
