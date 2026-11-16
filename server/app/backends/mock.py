"""Mock 백엔드 — 보드·모델 없이 프론트 연동을 시험하기 위한 가짜 모델.

1) 데이터셋 발화와 글자까지 같으면 그 정답 Action을 그대로 돌려준다(replay).
2) 아니면 간단한 키워드 규칙으로 Action을 만든다. (정확도 목적이 아님: 연동·예외 처리 시험용)
"""
import json
import os
import re
from pathlib import Path
from typing import Optional

from ..policy import DEFAULT_SOLD_OUT, MENUS, alternatives_for
from .base import GenerationRequest

DEFAULT_REPLAY = Path(__file__).resolve().parents[2] / "data" / "dataset_cases.json"

ALIASES = [
    ("아아", "아메리카노"),
    ("크루아상", "플레인크루아상"),
    ("아이스티", "복숭아 아이스티"),
    ("케이크", "치즈케이크"),
    ("쿠키", "초코쿠키"),
]
_NAMES = sorted(MENUS, key=lambda n: -len(n.replace(" ", "")))
_PAY = [
    ("삼성", "SAMSUNG_PAY"),
    ("카카오", "KAKAO_PAY"),
    ("네이버", "NAVER_PAY"),
    ("애플", "APPLE_PAY"),
    ("현금", "CASH"),
    ("카드", "CARD"),
]


def _squash(text: str) -> str:
    return re.sub(r"\s+", "", text)


def _find_menu(t: str) -> Optional[str]:
    for name in _NAMES:
        if name.replace(" ", "") in t:
            return name
    for alias, name in ALIASES:
        if alias in t:
            return name
    return None


def _qty(t: str) -> int:
    m = re.search(r"(?:^|\D)([1-9])(?:잔|개)", t)
    if m:
        return int(m.group(1))
    for i, w in enumerate(["한", "두", "세", "네", "다섯", "여섯", "일곱", "여덟", "아홉"], start=1):
        if f"{w}잔" in t or f"{w}개" in t:
            return i
    m = re.search(r"하나|둘|셋|넷", t)
    return ["하나", "둘", "셋", "넷"].index(m.group(0)) + 1 if m else 1


def rule_based(text: str, ctx: dict) -> dict:
    t = _squash(text)
    sold_out = ctx.get("sold_out") or list(DEFAULT_SOLD_OUT)
    menu = _find_menu(t)

    if re.search(r"직원|도와줘|도움", t):
        return {"action": "call_staff", "reason": "USER_REQUESTED_ASSISTANCE"}
    if re.search(r"고대비|큰글씨", t):
        return {"action": "set_accessibility", "mode": "HIGH_CONTRAST"}
    if re.search(r"휠체어|낮게|하단", t):
        return {"action": "set_accessibility", "mode": "LOWER_TOUCH"}
    if re.search(r"음성안내|소리로", t):
        return {"action": "set_accessibility", "mode": "VOICE_GUIDE"}

    if menu:
        if re.search(r"빼|삭제|취소", t):
            return {"action": "remove_from_cart", "target": menu}
        if menu in sold_out:
            return {
                "action": "inform_sold_out",
                "item": menu,
                "alternatives": alternatives_for(menu, sold_out),
            }
        pol = MENUS[menu]
        qty = _qty(t)
        if pol.temp == "NONE":
            return {"action": "add_to_cart", "items": [{"menu": menu, "quantity": qty}]}
        temp = (
            "HOT"
            if re.search(r"따뜻|뜨겁|뜨거|핫", t)
            else "ICE"
            if re.search(r"아이스|차갑|차가|시원|아아", t)
            else None
        )
        if pol.temp == "ICE_ONLY":
            temp = "ICE"
        size = (
            "LARGE"
            if re.search(r"라지|큰|크게", t)
            else "REGULAR"
            if re.search(r"레귤러|작은|기본", t)
            else None
        )
        if temp is None:
            return {
                "action": "ask_clarification",
                "field": "temperature",
                "item": menu,
                "choices": ["HOT", "ICE"],
            }
        if size is None:
            return {
                "action": "ask_clarification",
                "field": "size",
                "item": menu,
                "choices": ["REGULAR", "LARGE"],
            }
        item = {"menu": menu, "quantity": qty, "temperature": temp, "size": size}
        if temp == "ICE":
            item["options"] = {"ice": "NORMAL"}
        return {"action": "add_to_cart", "items": [item]}

    for key, method in _PAY:
        if key in t:
            return {"action": "start_payment", "method": method}
    _NO_RECEIPT = r"안|않|아니|필요없|괜찮|말고|싫"
    if "영수증" in t:
        return {"action": "set_receipt", "print": not re.search(_NO_RECEIPT, t)}
    # 영수증 선택 화면의 예/아니오 대답은 "영수증"이라는 말이 없어도 알아듣는다.
    if ctx.get("screen") == "receipt_choice":
        if re.search(_NO_RECEIPT, t):
            return {"action": "set_receipt", "print": False}
        if re.search(r"네|응|예|좋아|출력|뽑|인쇄|주세요|해줘", t):
            return {"action": "set_receipt", "print": True}
    if "쿠폰" in t:
        return {"action": "open_coupon_list"}
    if re.search(r"적립|스탬프", t):
        return {"action": "open_loyalty_input", "input_mode": "KEYPAD"}
    if re.search(r"비워|전체취소|처음부터", t):
        return {"action": "reset_order"}
    if "장바구니" in t:
        return {"action": "show_cart"}
    if re.search(r"주문확인|이대로|맞아요", t):
        return {"action": "confirm_order", "summary_required": True}
    # "주문할게요" 등 방법을 말하지 않은 진행 의사: 메뉴·장바구니 화면에서만 다음 단계(confirm_order)로.
    if ctx.get("screen") in ("menu", "cart") and re.search(
        r"(주문|결제|계산)(할게|할께|할래|하겠|해줘|해주세요|해요|하자)", t
    ):
        return {"action": "confirm_order", "summary_required": True}
    if "추천" in t:
        items = ["바닐라라떼", "초코라떼"]
        if re.search(r"시원|아이스|차가", t):
            items = ["자몽에이드", "레몬에이드"]
        elif re.search(r"디저트|달달", t):
            items = ["티라미수", "치즈케이크"]
        return {"action": "recommend_menu", "items": items, "confirmation_required": True}
    if "디저트" in t:
        return {"action": "show_category", "category": "DESSERT"}
    # "차"는 "차가운" 같은 말과 섞이지 않도록 한글에 붙지 않은 단독 단어일 때만(조사는 허용).
    if re.search(r"논커피|커피말고|라떼|에이드|차종류|차메뉴", t) or re.search(
        r"(?<![가-힣])차(?:[를은는도만로]|(?![가-힣]))", text
    ):
        return {"action": "show_category", "category": "NON_COFFEE"}
    if "커피" in t:
        return {"action": "show_category", "category": "COFFEE"}
    if ctx.get("order_number") and re.search(r"번호", t):
        return {
            "action": "announce_order",
            "order_number": ctx["order_number"],
            "pickup": ctx.get("pickup") or "VIBRATION_BELL",
        }
    return {"action": "no_action", "reason": "NO_COMMAND"}


class MockBackend:
    name = "mock"

    def __init__(self, replay_path: Optional[str] = None):
        path = Path(replay_path or os.getenv("ONKIOSK_MOCK_REPLAY") or DEFAULT_REPLAY)
        self._replay: dict[str, list[dict]] = {}
        if path.is_file():
            for case in json.loads(path.read_text(encoding="utf-8")):
                self._replay.setdefault(case["user_text"], []).append(case)

    async def generate(self, req: GenerationRequest) -> str:
        cases = self._replay.get(req.user_text)
        if cases:
            screen = req.context.get("screen")
            chosen = next((c for c in cases if c["screen"] == screen), cases[0])
            return json.dumps(chosen["target"], ensure_ascii=False)
        return json.dumps(rule_based(req.user_text, req.context), ensure_ascii=False)
