"""파트 3 database.py 연동 지점 (모듈이 없어도 서버가 동작하도록 느슨하게 연결).

database.py를 server/ 폴더(또는 PYTHONPATH)에 두면 자동으로 사용한다.
환경변수 ONKIOSK_DB=off 로 끌 수 있다.
"""
import logging
import os
import sys
from pathlib import Path
from typing import Any, Optional

from .policy import DEFAULT_SOLD_OUT
from .screens import local_normalize

log = logging.getLogger("onkiosk.db")

_db: Any = None
if os.getenv("ONKIOSK_DB", "auto").lower() != "off":
    try:
        import database as _db  # type: ignore  # noqa: F401
    except ModuleNotFoundError:
        # server/ 안에 없으면 저장소 루트(server/ 바로 밖)의 database.py를 찾는다.
        _root = str(Path(__file__).resolve().parents[2])
        if _root not in sys.path:
            sys.path.append(_root)
        try:
            import database as _db  # type: ignore  # noqa: F401,F811
        except Exception:
            _db = None
    except Exception:  # database.py 자체의 import 오류
        _db = None


def enabled() -> bool:
    return _db is not None


def init() -> None:
    """서버 시작 시 DB 파일·테이블을 준비한다(없으면 생성, 있으면 그대로). 실패해도 서버는 뜬다."""
    fn = getattr(_db, "init_db", None) if _db is not None else None
    if fn is None:
        return
    try:
        fn()
    except Exception:
        log.exception("init_db 실패")


def normalize_screen(raw: str) -> str:
    if _db is not None and hasattr(_db, "normalize_screen_context"):
        try:
            return _db.normalize_screen_context(raw)
        except Exception:
            log.exception("normalize_screen_context 실패, 로컬 규칙 사용")
    return local_normalize(raw)


def sold_out_menus() -> list[str]:
    if _db is not None and hasattr(_db, "get_sold_out_menu_names"):
        try:
            return list(_db.get_sold_out_menu_names())
        except Exception:
            log.exception("get_sold_out_menu_names 실패, 기본값 사용")
    return list(DEFAULT_SOLD_OUT)


def log_inference(**kwargs: Any) -> None:
    """추론 로그. 실패해도 주문 흐름에 영향이 없도록 예외를 삼킨다."""
    if _db is None or not hasattr(_db, "log_inference"):
        log.info("inference %s", {k: v for k, v in kwargs.items() if k != "response_obj"})
        return
    # database.log_inference 는 지연시간을 int 로 비교(<0)하므로 None(미측정)은 0 으로 보낸다.
    for key in ("stt_latency_ms", "llm_latency_ms"):
        if kwargs.get(key) is None:
            kwargs[key] = 0
    # database.log_inference 는 predicted_action 이 비어 있으면 ValueError 를 낸다(폴백 응답은 action 이 없음).
    if not kwargs.get("predicted_action"):
        kwargs["predicted_action"] = "FALLBACK"
    if kwargs.get("response_obj") is None:
        kwargs["response_obj"] = {}
    try:
        _db.log_inference(**kwargs)
    except Exception:
        log.exception("log_inference 실패")


# ---------------------------------------------------------------- 주문·적립·별칭 (합의안 6절)
class DbUnavailable(Exception):
    """database.py가 없거나 해당 함수가 없음."""


class OrderRejected(Exception):
    """품절(SOLD_OUT) 또는 정책/옵션 위반(INVALID_ORDER). code에 사유."""

    def __init__(self, code: str, detail: str):
        super().__init__(detail)
        self.code = code
        self.detail = detail


def _require(name: str):
    fn = getattr(_db, name, None) if _db is not None else None
    if fn is None:
        raise DbUnavailable(name)
    return fn


def _as_dict(obj: Any) -> dict:
    """create_order 반환값(dict/객체/pydantic)을 dict로."""
    if obj is None:
        return {}
    if isinstance(obj, dict):
        return dict(obj)
    for attr in ("model_dump", "_asdict", "dict"):
        fn = getattr(obj, attr, None)
        if callable(fn):
            try:
                return dict(fn())
            except Exception:
                pass
    return {k: v for k, v in vars(obj).items() if not k.startswith("_")} if hasattr(obj, "__dict__") else {}


def create_order(items: list[dict], payment_method: str, pickup_type: str, print_receipt: bool,
                 total_price: Optional[int] = None) -> dict:
    """database.create_order 호출. 총액은 기본적으로 DB 가격표로 서버가 계산한다(total_price=None)."""
    fn = _require("create_order")
    try:
        result = fn(items=items, total_price=total_price, payment_method=payment_method,
                    pickup_type=pickup_type, print_receipt=print_receipt)
    except Exception as e:  # SoldOutError / InvalidOrderError는 이름으로 판별(모듈 없이도 테스트 가능)
        kind = type(e).__name__
        if kind == "SoldOutError":
            raise OrderRejected("SOLD_OUT", str(e)) from e
        if kind == "InvalidOrderError":
            raise OrderRejected("INVALID_ORDER", str(e)) from e
        raise
    out = _as_dict(result)
    # 주문번호 키 이름 방어 (order_number | order_no | number)
    if "order_number" not in out:
        for k in ("order_no", "number", "id"):
            if k in out:
                out["order_number"] = str(out[k])
                break
    return out


def add_stamp(phone_number: str, count: int = 1) -> dict:
    """전화번호는 여기서만 전달하고 로그·응답에 남기지 않는다."""
    fn = _require("add_stamp")
    out = _as_dict(fn(phone_number=phone_number, count=count))
    for k in list(out):
        if "phone" in k.lower() or "hash" in k.lower():
            out.pop(k)
    return out


def alias_map() -> dict[str, str]:
    fn = _require("get_alias_map")
    return dict(fn())
