"""프론트 화면 이름(9종) → 데이터셋 화면 이름(6종).

합의안에 따르면 database.py의 normalize_screen_context()가 같은 일을 한다.
DB 모듈이 없을 때를 위해 같은 규칙을 여기에도 둔다(DB 모듈이 있으면 그쪽을 우선 사용).
"""
SCREEN_MAP = {
    # 프론트 9종
    "start": "menu",
    "menu": "menu",
    "option": "menu",
    "cart": "cart",
    "point": "loyalty",
    "pay": "payment",
    "paying": "payment",
    "receipt": "receipt_choice",
    "done": "complete",
    # 데이터셋 6종은 그대로 통과
    "loyalty": "loyalty",
    "payment": "payment",
    "receipt_choice": "receipt_choice",
    "complete": "complete",
}
DEFAULT_SCREEN = "menu"


def local_normalize(raw: str) -> str:
    return SCREEN_MAP.get((raw or "").strip().lower(), DEFAULT_SCREEN)
