import asyncio
import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.backends.base import GenerationRequest
from app.backends.mock import MockBackend
from app.factory import create_app

CASES = json.loads((Path(__file__).resolve().parents[1] / "data" / "dataset_cases.json").read_text("utf-8"))


class FixedBackend:
    name = "fixed"

    def __init__(self, output):
        self.output = output
        self.seen: list[GenerationRequest] = []

    async def generate(self, req):
        self.seen.append(req)
        if isinstance(self.output, Exception):
            raise self.output
        return self.output


class SlowBackend:
    name = "slow"

    async def generate(self, req):
        await asyncio.sleep(1)
        return '{"action":"show_cart"}'


def client(backend=None, timeout_s=3.0):
    return TestClient(create_app(backend or MockBackend(), timeout_s=timeout_s))


def post(c, text, **ctx):
    body = {"user_text": text, "context": {"screen": "menu", **ctx}}
    r = c.post("/api/action", json=body)
    assert r.status_code == 200
    return r.json()


# ------------------------------------------------------------ 전체 흐름(데이터셋 replay)
def test_replay_all_240_dataset_cases():
    c = client()
    bad = []
    for case in CASES:
        r = c.post("/api/action", json={"user_text": case["user_text"], "context": case["context"]}).json()
        if not (r["ok"] and not r["fallback"] and r["action"] == case["target"]):
            bad.append((case["id"], r.get("error_code"), r.get("detail")))
    assert bad == []


# ------------------------------------------------------------ 규칙/폴백
def test_two_failures_returns_call_staff_without_model():
    backend = FixedBackend('{"action":"show_cart"}')
    r = post(client(backend), "아무 말", recognition_failure_count=2)
    assert r["action"] == {"action": "call_staff", "reason": "VOICE_RECOGNITION_FAILED_TWICE"}
    assert r["source"] == "rule" and backend.seen == []


def test_one_failure_still_goes_to_model():
    r = post(client(FixedBackend('{"action":"show_cart"}')), "장바구니", recognition_failure_count=1)
    assert r["action"]["action"] == "show_cart" and r["source"] == "model"


def test_timeout_triggers_fallback():
    r = post(client(SlowBackend(), timeout_s=0.05), "장바구니")
    assert (r["ok"], r["fallback"], r["error_code"]) == (False, True, "TIMEOUT")


def test_backend_exception_triggers_fallback():
    r = post(client(FixedBackend(RuntimeError("npu crashed"))), "장바구니")
    assert (r["fallback"], r["error_code"]) == (True, "BACKEND_ERROR")


@pytest.mark.parametrize(
    "output,code",
    [
        ("죄송합니다, 이해하지 못했어요", "INVALID_JSON"),
        ('{"action": "fly_to_moon"}', "INVALID_SCHEMA"),
        ('{"action":"add_to_cart","items":[{"menu":"유자차","quantity":1,"temperature":"HOT","size":"REGULAR","options":{"syrup":1}}]}', "POLICY_VIOLATION"),
    ],
)
def test_bad_model_output_triggers_fallback(output, code):
    r = post(client(FixedBackend(output)), "아무거나")
    assert (r["ok"], r["fallback"], r["error_code"]) == (False, True, code)
    assert r["action"] is None


def test_no_action_is_a_normal_response_not_fallback():
    r = post(client(FixedBackend('{"action":"no_action","reason":"NO_COMMAND"}')), "음...")
    assert (r["ok"], r["fallback"]) == (True, False)


# ------------------------------------------------------------ 보정
def test_sold_out_repair_uses_db_default_when_context_has_no_sold_out():
    r = post(client(FixedBackend('{"action":"add_to_cart","items":[{"menu":"마카롱","quantity":1}]}')), "마카롱 주세요")
    assert r["action"]["action"] == "inform_sold_out" and r["repairs"] == ["sold_out_to_inform"]


def test_client_sold_out_overrides_default():
    out = '{"action":"add_to_cart","items":[{"menu":"마카롱","quantity":1}]}'
    r = post(client(FixedBackend(out)), "마카롱 주세요", sold_out=[])
    assert r["action"]["action"] == "add_to_cart"


def test_missing_size_repair():
    out = '{"action":"add_to_cart","items":[{"menu":"아메리카노","quantity":1,"temperature":"HOT"}]}'
    r = post(client(FixedBackend(out)), "따뜻한 아메리카노")
    assert r["action"]["field"] == "size"


def test_code_fence_output_is_accepted():
    r = post(client(FixedBackend('```json\n{"action":"show_cart"}\n```')), "장바구니")
    assert r["ok"] and r["action"]["action"] == "show_cart"


# ------------------------------------------------------------ 입력 처리
def test_screen_names_are_normalized():
    for raw, expected in [("start", "menu"), ("option", "menu"), ("point", "loyalty"), ("pay", "payment"),
                          ("paying", "payment"), ("receipt", "receipt_choice"), ("done", "complete"), ("cart", "cart")]:
        r = client().post("/api/action", json={"user_text": "장바구니 보여줘", "context": {"screen": raw}}).json()
        assert r["screen"] == expected, raw


def test_digits_are_redacted_before_model_and_log():
    backend = FixedBackend('{"action":"open_loyalty_input","input_mode":"KEYPAD"}')
    post(client(backend), "01012345678 로 적립해줘", screen="point")
    sent = backend.seen[0]
    assert "0101234" not in sent.user_text
    assert all("0101234" not in m["content"] for m in sent.messages)


def test_model_context_omits_zero_failure_count_and_empty_draft():
    backend = FixedBackend('{"action":"show_cart"}')
    post(client(backend), "장바구니")
    ctx = backend.seen[0].context
    assert "recognition_failure_count" not in ctx and "draft" not in ctx
    assert ctx["screen"] == "menu" and ctx["sold_out"] == ["콜드브루", "마카롱"]


def test_prompt_contains_sp01_and_context():
    backend = FixedBackend('{"action":"show_cart"}')
    post(client(backend), "장바구니 보여줘")
    msgs = backend.seen[0].messages
    assert msgs[0]["role"] == "system" and "OnKiosk 음성 명령 파서" in msgs[0]["content"]
    assert "장바구니 보여줘" in msgs[1]["content"] and '"screen":"menu"' in msgs[1]["content"]


@pytest.mark.parametrize("body", [{"user_text": "", "context": {"screen": "menu"}}, {"context": {"screen": "menu"}}, {"user_text": "a"}])
def test_invalid_request_is_422(body):
    assert client().post("/api/action", json=body).status_code == 422


# ------------------------------------------------------------ Mock 규칙
@pytest.mark.parametrize(
    "text,expected",
    [
        ("따뜻한 카페모카 라지로 두 잔", {"action": "add_to_cart", "items": [{"menu": "카페모카", "quantity": 2, "temperature": "HOT", "size": "LARGE"}]}),
        ("아아 한 잔 주세요", {"action": "ask_clarification", "field": "size", "item": "아메리카노", "choices": ["REGULAR", "LARGE"]}),
        ("라떼 주세요", {"action": "ask_clarification", "field": "temperature", "item": "카페라떼", "choices": ["HOT", "ICE"]}),
        ("카카오페이로 할게요", {"action": "start_payment", "method": "KAKAO_PAY"}),
        ("콜드브루 주세요", {"action": "inform_sold_out", "item": "콜드브루", "alternatives": ["아메리카노", "카페라떼"]}),
        ("오늘 날씨 어때", {"action": "no_action", "reason": "NO_COMMAND"}),
    ],
)
def test_mock_rules(text, expected):
    r = post(client(), text)
    assert r["ok"] and r["action"] == expected


# ------------------------------------------------------------ 기타 엔드포인트
def test_health_and_answer_texts_and_cors():
    c = client()
    h = c.get("/health").json()
    assert h["status"] == "ok" and h["backend"] == "mock"
    assert len(c.get("/api/answer-texts").json()) == 15
    pre = c.options(
        "/api/action",
        headers={"Origin": "http://localhost:3000", "Access-Control-Request-Method": "POST",
                 "Access-Control-Request-Headers": "content-type"},
    )
    assert pre.headers.get("access-control-allow-origin") == "http://localhost:3000"


@pytest.mark.parametrize("screen", ["menu", "cart"])
@pytest.mark.parametrize("text", ["주문할게요", "주문 할게요", "결제할게요", "계산할게요"])
def test_mock_order_intent_confirms_on_menu_and_cart(screen, text):
    from app.backends.mock import rule_based

    ctx = {"screen": screen, "cart": [{"menu": "아메리카노", "quantity": 1}]}
    assert rule_based(text, ctx)["action"] == "confirm_order"


def test_mock_order_intent_ignored_on_payment_screen():
    from app.backends.mock import rule_based

    assert rule_based("결제할게요", {"screen": "payment", "cart": []})["action"] == "no_action"
    # 결제 수단을 말하면 기존처럼 결제 시작
    assert rule_based("카드로 결제할게요", {"screen": "cart", "cart": []})["action"] == "start_payment"
