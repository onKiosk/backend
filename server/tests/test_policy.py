import pytest

from app.parser import ParseError, parse_action
from app.policy import PolicyViolation, alternatives_for, apply_policy
from app.schemas import ACTION_ADAPTER

SOLD = ["콜드브루", "마카롱"]


def act(d):
    return ACTION_ADAPTER.validate_python(d)


def run(d, ctx=None, sold=SOLD):
    return apply_policy(act(d), ctx or {"screen": "menu"}, sold)


def item(menu, **kw):
    return {"action": "add_to_cart", "items": [{"menu": menu, "quantity": 1, **kw}]}


# ------------------------------------------------------------ 보정(repair)
def test_sold_out_add_becomes_inform():
    a, repairs = run(item("마카롱"))
    assert a.action == "inform_sold_out" and a.item == "마카롱"
    assert "마카롱" not in a.alternatives and len(a.alternatives) == 2
    assert repairs == ["sold_out_to_inform"]


def test_missing_temperature_becomes_clarification():
    a, repairs = run(item("아메리카노", size="REGULAR"))
    assert (a.action, a.field, a.choices) == ("ask_clarification", "temperature", ["HOT", "ICE"])


def test_missing_size_becomes_clarification():
    a, _ = run(item("아메리카노", temperature="HOT"))
    assert (a.action, a.field) == ("ask_clarification", "size")


def test_ice_only_temperature_is_filled():
    a, repairs = run(item("자몽에이드", size="LARGE"))
    assert a.action == "add_to_cart" and a.items[0].temperature == "ICE"
    assert "ice_only_temperature_filled" in repairs


def test_dessert_needs_nothing():
    a, repairs = run(item("티라미수"))
    assert a.action == "add_to_cart" and repairs == []


# ------------------------------------------------------------ 위반(violation)
@pytest.mark.parametrize(
    "payload",
    [
        item("복숭아 아이스티", temperature="HOT", size="REGULAR"),  # ICE 전용에 HOT
        item("티라미수", temperature="HOT"),  # 디저트에 온도
        item("유자차", temperature="HOT", size="REGULAR", options={"syrup": 1}),  # 차 시럽 불가
        item("유자차", temperature="HOT", size="REGULAR", options={"strength": "LIGHT"}),  # 차 연하게 불가
        item("아메리카노", temperature="HOT", size="REGULAR", options={"milk": "SOY"}),  # 아메리카노 우유 불가
        item("초코라떼", temperature="HOT", size="REGULAR", options={"shot": 1}),  # 논커피 샷 불가
        item("카페라떼", temperature="HOT", size="REGULAR", options={"ice": "LESS"}),  # HOT에 얼음 양
        item("아아아메", temperature="ICE", size="REGULAR"),  # 없는 메뉴
    ],
)
def test_policy_violations(payload):
    with pytest.raises(PolicyViolation):
        run(payload)


def test_modify_target_must_be_in_cart():
    with pytest.raises(PolicyViolation):
        run({"action": "modify_cart", "target": "카페라떼", "changes": {"size": "LARGE"}},
            {"screen": "cart", "cart": [{"menu": "아메리카노", "quantity": 1}]})


def test_modify_ok_and_policy_checked():
    ctx = {"screen": "cart", "cart": [{"menu": "초코라떼", "quantity": 1, "temperature": "HOT", "size": "REGULAR"}]}
    a, _ = run({"action": "modify_cart", "target": "초코라떼", "changes": {"size": "LARGE"}}, ctx)
    assert a.action == "modify_cart"
    with pytest.raises(PolicyViolation):  # HOT 음료에 얼음 양
        run({"action": "modify_cart", "target": "초코라떼", "changes": {"ice": "LESS"}}, ctx)


def test_remove_target_must_be_in_cart():
    with pytest.raises(PolicyViolation):
        run({"action": "remove_from_cart", "target": "카페라떼"}, {"screen": "cart", "cart": []})


def test_recommend_cannot_include_sold_out():
    with pytest.raises(PolicyViolation):
        run({"action": "recommend_menu", "items": ["콜드브루", "카페라떼"], "confirmation_required": True})


def test_announce_order_uses_context_only():
    ctx = {"screen": "complete", "order_number": "A042", "pickup": "VIBRATION_BELL"}
    run({"action": "announce_order", "order_number": "A042", "pickup": "VIBRATION_BELL"}, ctx)
    with pytest.raises(PolicyViolation):
        run({"action": "announce_order", "order_number": "A099", "pickup": "VIBRATION_BELL"}, ctx)
    with pytest.raises(PolicyViolation):  # context에 번호가 없는데 지어냄
        run({"action": "announce_order", "order_number": "A001", "pickup": "VIBRATION_BELL"})


def test_failed_twice_reason_requires_count():
    with pytest.raises(PolicyViolation):
        run({"action": "call_staff", "reason": "VOICE_RECOGNITION_FAILED_TWICE"})
    run({"action": "call_staff", "reason": "VOICE_RECOGNITION_FAILED_TWICE"},
        {"screen": "menu", "recognition_failure_count": 2})


def test_inform_sold_out_must_match_context():
    with pytest.raises(PolicyViolation):
        run({"action": "inform_sold_out", "item": "아메리카노", "alternatives": ["카페라떼", "카푸치노"]})


# ------------------------------------------------------------ 스키마/파서
def test_answer_question_topic_code_pairs():
    ok = {"action": "answer_question", "topic": "PAYMENT_METHOD", "answer_code": "CASH_AVAILABLE"}
    assert act(ok).action == "answer_question"
    with pytest.raises(ValueError):
        act({**ok, "answer_code": "VIBRATION_BELL_PICKUP"})


def test_ask_clarification_choices_checked():
    with pytest.raises(ValueError):
        act({"action": "ask_clarification", "field": "size", "item": "아메리카노", "choices": ["HOT", "ICE"]})


@pytest.mark.parametrize(
    "text",
    [
        '{"action":"show_cart"}',
        '```json\n{"action":"show_cart"}\n```',
        '네, 알겠습니다. {"action":"show_cart"} 입니다.',
    ],
)
def test_parser_extracts_first_json(text):
    assert parse_action(text).action == "show_cart"


@pytest.mark.parametrize("text", ["죄송합니다", '{"action": "show_cart"', "[1,2]"])
def test_parser_rejects_garbage(text):
    with pytest.raises(ParseError):
        parse_action(text)


def test_alternatives_same_group_and_not_sold_out():
    assert alternatives_for("콜드브루", SOLD) == ["아메리카노", "카페라떼"]
    assert alternatives_for("마카롱", SOLD) == ["치즈케이크", "티라미수"]
