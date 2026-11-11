"""파트 3 database.py 연동 지점 (모듈이 없어도 서버가 동작하도록 느슨하게 연결).

database.py를 server/ 폴더(또는 PYTHONPATH)에 두면 자동으로 사용한다.
환경변수 ONKIOSK_DB=off 로 끌 수 있다.
"""
import logging
import os
from typing import Any, Optional

from .policy import DEFAULT_SOLD_OUT
from .screens import local_normalize

log = logging.getLogger("onkiosk.db")

_db: Any = None
if os.getenv("ONKIOSK_DB", "auto").lower() != "off":
    try:
        import database as _db  # type: ignore  # noqa: F401
    except Exception:  # 모듈이 없거나 import 오류
        _db = None


def enabled() -> bool:
    return _db is not None


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
    try:
        _db.log_inference(**kwargs)
    except Exception:
        log.exception("log_inference 실패")
