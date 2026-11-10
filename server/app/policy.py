"""메뉴 정책(25종) 검증과 안전 보정.

기준: 데이터셋 `메뉴_정책` 시트 / v5.5 합의안의 MENU_OPTION_POLICY.
- 보정(repair): 안전하게 고칠 수 있는 경우 (품절 메뉴 담기 → 품절 안내, 필수 옵션 누락 → 되묻기, ICE 전용 온도 채우기)
- 위반(PolicyViolation): 고칠 수 없는 경우 → 프론트가 기존 규칙으로 폴백
"""
from dataclasses import dataclass
from typing import Optional

from .schemas import (
    Action,
    AddToCart,
    AnnounceOrder,
    AskClarification,
    CallStaff,
    InformSoldOut,
    ModifyCart,
    RecommendMenu,
    RemoveFromCart,
)

DEFAULT_SOLD_OUT: tuple[str, ...] = ("콜드브루", "마카롱")


class PolicyViolation(Exception):
    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code
        self.message = message


@dataclass(frozen=True)
class MenuPolicy:
    name: str
    group: str  # coffee / latte / tea / ade / dessert (프론트 탭 기준, 대체 메뉴 추천용)
    category: str  # COFFEE / NON_COFFEE / DESSERT
    temp: str  # BOTH / ICE_ONLY / NONE
    shot: bool = False
    syrup: bool = False
    light: bool = False
    milk: bool = False


def _m(name, group, temp="BOTH", **opts) -> MenuPolicy:
    category = "COFFEE" if group == "coffee" else "DESSERT" if group == "dessert" else "NON_COFFEE"
    return MenuPolicy(name, group, category, temp, **opts)


_ALL = [
    _m("아메리카노", "coffee", shot=True, syrup=True, light=True),
    _m("카페라떼", "coffee", shot=True, syrup=True, light=True, milk=True),
    _m("바닐라라떼", "coffee", shot=True, syrup=True, light=True, milk=True),
    _m("카푸치노", "coffee", shot=True, syrup=True, light=True, milk=True),
    _m("카페모카", "coffee", shot=True, syrup=True, light=True, milk=True),
    _m("콜드브루", "coffee", "ICE_ONLY", shot=True, syrup=True),
    _m("초코라떼", "latte", syrup=True, milk=True),
    _m("녹차라떼", "latte", syrup=True, milk=True),
    _m("고구마라떼", "latte", syrup=True, milk=True),
    _m("딸기라떼", "latte", syrup=True, milk=True),
    _m("유자차", "tea"),
    _m("대추차", "tea"),
    _m("캐모마일", "tea"),
    _m("얼그레이", "tea"),
    _m("복숭아 아이스티", "tea", "ICE_ONLY", syrup=True),
    _m("자몽에이드", "ade", "ICE_ONLY", syrup=True),
    _m("레몬에이드", "ade", "ICE_ONLY", syrup=True),
    _m("청포도에이드", "ade", "ICE_ONLY", syrup=True),
    _m("치즈케이크", "dessert", "NONE"),
    _m("티라미수", "dessert", "NONE"),
    _m("초코쿠키", "dessert", "NONE"),
    _m("베이글", "dessert", "NONE"),
    _m("플레인크루아상", "dessert", "NONE"),
    _m("마카롱", "dessert", "NONE"),
    _m("소금빵", "dessert", "NONE"),
]
MENUS: dict[str, MenuPolicy] = {m.name: m for m in _ALL}
assert len(MENUS) == 25


def alternatives_for(item: str, sold_out: list[str], limit: int = 2) -> list[str]:
    pol = MENUS.get(item)
    if not pol:
        return []
    same = [m.name for m in _ALL if m.group == pol.group and m.name != item and m.name not in sold_out]
    return same[:limit]


# ---------------------------------------------------------------- 검증
def _check_options(opts, temperature: Optional[str], pol: MenuPolicy, where: str) -> None:
    if opts is None:
        return
    if opts.ice is not None and temperature == "HOT":
        raise PolicyViolation("HOT_WITH_ICE", f"{where}: HOT 음료에 얼음 양을 지정할 수 없음")
    if opts.shot and not pol.shot:
        raise PolicyViolation("OPTION_NOT_ALLOWED", f"{pol.name}: 샷 추가 불가")
    if opts.syrup and not pol.syrup:
        raise PolicyViolation("OPTION_NOT_ALLOWED", f"{pol.name}: 시럽 추가 불가")
    if opts.strength == "LIGHT" and not pol.light:
        raise PolicyViolation("OPTION_NOT_ALLOWED", f"{pol.name}: 연하게 불가")
    if opts.milk not in (None, "REGULAR") and not pol.milk:
        raise PolicyViolation("OPTION_NOT_ALLOWED", f"{pol.name}: 우유 변경 불가")


def _cart_menus(ctx: dict) -> dict[str, dict]:
    return {line.get("menu"): line for line in ctx.get("cart") or [] if isinstance(line, dict)}


def apply_policy(
    action: Action, ctx: dict, sold_out: list[str]
) -> tuple[Action, list[str]]:
    """(보정된 Action, 보정 내역). 고칠 수 없으면 PolicyViolation."""
    repairs: list[str] = []

    if isinstance(action, AddToCart):
        for item in action.items:
            if item.menu not in MENUS:
                raise PolicyViolation("UNKNOWN_MENU", f"알 수 없는 메뉴: {item.menu}")
        sold = [i.menu for i in action.items if i.menu in sold_out]
        if sold:
            repairs.append("sold_out_to_inform")
            return (
                InformSoldOut(
                    action="inform_sold_out",
                    item=sold[0],
                    alternatives=alternatives_for(sold[0], sold_out),
                ),
                repairs,
            )
        items = []
        for item in action.items:
            pol = MENUS[item.menu]
            if pol.temp == "NONE":
                if item.temperature or item.size or item.options:
                    raise PolicyViolation("DESSERT_WITH_OPTION", f"{item.menu}: 디저트는 수량만 사용")
                items.append(item)
                continue
            if pol.temp == "ICE_ONLY":
                if item.temperature == "HOT":
                    raise PolicyViolation("ICE_ONLY_WITH_HOT", f"{item.menu}: ICE 전용")
                if item.temperature is None:
                    item = item.model_copy(update={"temperature": "ICE"})
                    repairs.append("ice_only_temperature_filled")
            _check_options(item.options, item.temperature, pol, item.menu)
            items.append(item)
        for item in items:
            pol = MENUS[item.menu]
            if pol.temp == "NONE":
                continue
            if item.temperature is None:
                repairs.append("missing_temperature_to_clarification")
                return (
                    AskClarification(
                        action="ask_clarification",
                        field="temperature",
                        item=item.menu,
                        choices=["HOT", "ICE"],
                    ),
                    repairs,
                )
            if item.size is None:
                repairs.append("missing_size_to_clarification")
                return (
                    AskClarification(
                        action="ask_clarification",
                        field="size",
                        item=item.menu,
                        choices=["REGULAR", "LARGE"],
                    ),
                    repairs,
                )
        return action.model_copy(update={"items": items}), repairs

    if isinstance(action, ModifyCart):
        cart = _cart_menus(ctx)
        line = cart.get(action.target)
        if line is None:
            raise PolicyViolation("TARGET_NOT_IN_CART", f"장바구니에 없는 대상: {action.target}")
        pol = MENUS.get(action.target)
        if pol is None:
            raise PolicyViolation("UNKNOWN_MENU", f"알 수 없는 메뉴: {action.target}")
        ch = action.changes
        if pol.temp == "NONE" and (ch.temperature or ch.size or ch.ice or ch.shot or ch.syrup or ch.strength or ch.milk):
            raise PolicyViolation("DESSERT_WITH_OPTION", f"{pol.name}: 디저트는 수량만 변경 가능")
        if pol.temp == "ICE_ONLY" and ch.temperature == "HOT":
            raise PolicyViolation("ICE_ONLY_WITH_HOT", f"{pol.name}: ICE 전용")
        final_temp = ch.temperature or line.get("temperature") or ("ICE" if pol.temp == "ICE_ONLY" else None)
        if ch.ice is not None and final_temp == "HOT":
            raise PolicyViolation("HOT_WITH_ICE", f"{pol.name}: HOT 음료에 얼음 양을 지정할 수 없음")
        if ch.shot and not pol.shot:
            raise PolicyViolation("OPTION_NOT_ALLOWED", f"{pol.name}: 샷 추가 불가")
        if ch.syrup and not pol.syrup:
            raise PolicyViolation("OPTION_NOT_ALLOWED", f"{pol.name}: 시럽 추가 불가")
        if ch.strength == "LIGHT" and not pol.light:
            raise PolicyViolation("OPTION_NOT_ALLOWED", f"{pol.name}: 연하게 불가")
        if ch.milk not in (None, "REGULAR") and not pol.milk:
            raise PolicyViolation("OPTION_NOT_ALLOWED", f"{pol.name}: 우유 변경 불가")
        return action, repairs

    if isinstance(action, RemoveFromCart):
        if action.target not in _cart_menus(ctx):
            raise PolicyViolation("TARGET_NOT_IN_CART", f"장바구니에 없는 대상: {action.target}")
        return action, repairs

    if isinstance(action, RecommendMenu):
        for name in action.items:
            if name not in MENUS:
                raise PolicyViolation("UNKNOWN_MENU", f"알 수 없는 메뉴: {name}")
            if name in sold_out:
                raise PolicyViolation("RECOMMEND_SOLD_OUT", f"품절 메뉴 추천: {name}")
        return action, repairs

    if isinstance(action, InformSoldOut):
        pending = ctx.get("pending_sold_out_item")
        if action.item not in MENUS:
            raise PolicyViolation("UNKNOWN_MENU", f"알 수 없는 메뉴: {action.item}")
        if action.item not in sold_out and action.item != pending:
            raise PolicyViolation("NOT_SOLD_OUT", f"품절이 아닌 메뉴: {action.item}")
        for name in action.alternatives:
            if name not in MENUS or name in sold_out:
                raise PolicyViolation("BAD_ALTERNATIVE", f"대체 메뉴 오류: {name}")
        return action, repairs

    if isinstance(action, AnnounceOrder):
        number, pickup = ctx.get("order_number"), ctx.get("pickup")
        if number is None or action.order_number != number:
            raise PolicyViolation("ORDER_NUMBER_MISMATCH", "주문번호는 context 값만 사용")
        if pickup is not None and action.pickup != pickup:
            raise PolicyViolation("PICKUP_MISMATCH", "수령 방식은 context 값만 사용")
        return action, repairs

    if isinstance(action, CallStaff):
        if action.reason == "VOICE_RECOGNITION_FAILED_TWICE" and int(ctx.get("recognition_failure_count") or 0) < 2:
            raise PolicyViolation("FAILED_TWICE_MISUSE", "연속 2회 실패가 아닌데 해당 사유 사용")
        return action, repairs

    return action, repairs
