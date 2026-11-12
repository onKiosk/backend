import types

import pytest
from fastapi.testclient import TestClient

from app import db_hooks
from app.backends.mock import MockBackend
from app.factory import create_app

ITEM = {"menu": "아메리카노", "quantity": 1, "temperature": "ICE", "size": "REGULAR"}
BODY = {"items": [ITEM], "payment_method": "KAKAO_PAY", "pickup_type": "VIBRATION_BELL", "print_receipt": True}


class SoldOutError(Exception): ...
class InvalidOrderError(Exception): ...


def fake_db(**overrides):
    calls = {"orders": [], "stamps": []}

    def create_order(items, total_price, payment_method, pickup_type, print_receipt):
        calls["orders"].append(dict(items=items, total_price=total_price, payment_method=payment_method,
                                    pickup_type=pickup_type, print_receipt=print_receipt))
        return {"order_number": "A001", "total_price": 4500, "status": "PAID"}

    def add_stamp(phone_number, count):
        calls["stamps"].append((phone_number, count))
        return {"stamps": 3, "phone_hash": "abc", "phone_number": phone_number}

    ns = types.SimpleNamespace(create_order=create_order, add_stamp=add_stamp,
                               get_alias_map=lambda: {"아아": "아메리카노"},
                               get_sold_out_menu_names=lambda: ["콜드브루"])
    for k, v in overrides.items():
        setattr(ns, k, v)
    return ns, calls


@pytest.fixture
def api(monkeypatch):
    def make(db=None):
        monkeypatch.setattr(db_hooks, "_db", db)
        return TestClient(create_app(MockBackend()))
    return make


def test_order_ok_passes_server_total_none(api):
    db, calls = fake_db()
    r = api(db).post("/api/orders", json=BODY)
    assert r.status_code == 200
    j = r.json()
    assert (j["ok"], j["order_number"], j["total_price"]) == (True, "A001", 4500)
    assert calls["orders"][0]["total_price"] is None  # 총액은 서버(DB)가 계산
    assert calls["orders"][0]["items"][0]["menu"] == "아메리카노"


def test_order_sold_out_and_invalid(api):
    def boom(exc):
        def f(**kw):
            raise exc("x")
        return f
    c = api(fake_db(create_order=boom(SoldOutError))[0])
    r = c.post("/api/orders", json=BODY)
    assert (r.status_code, r.json()["error_code"]) == (409, "SOLD_OUT")
    c = api(fake_db(create_order=boom(InvalidOrderError))[0])
    r = c.post("/api/orders", json=BODY)
    assert (r.status_code, r.json()["error_code"]) == (422, "INVALID_ORDER")
    c = api(fake_db(create_order=boom(RuntimeError))[0])
    assert c.post("/api/orders", json=BODY).json()["error_code"] == "ORDER_FAILED"


def test_order_without_db_and_bad_body(api):
    c = api(None)
    r = c.post("/api/orders", json=BODY)
    assert (r.status_code, r.json()["error_code"]) == (503, "DB_UNAVAILABLE")
    assert c.post("/api/orders", json={**BODY, "items": []}).status_code == 422
    assert c.post("/api/orders", json={**BODY, "payment_method": "BITCOIN"}).status_code == 422


@pytest.mark.parametrize("phone,sent", [("1234", "1234"), ("010-1234-5678", "01012345678"), ("01012345678", "01012345678")])
def test_stamp_ok_strips_phone_from_response(api, phone, sent):
    db, calls = fake_db()
    r = api(db).post("/api/stamps", json={"phone_number": phone, "count": 1})
    assert r.status_code == 200
    assert calls["stamps"] == [(sent, 1)]
    assert r.json()["stamp"] == {"stamps": 3}  # 번호·해시는 응답에 없다
    assert sent not in r.text


@pytest.mark.parametrize("phone", ["123", "12345", "0101234567", "abcd", "12-34", "010-1234-56789"])
def test_stamp_rejects_bad_phone(api, phone):
    db, calls = fake_db()
    r = api(db).post("/api/stamps", json={"phone_number": phone})
    assert (r.status_code, r.json()["error_code"]) == (422, "INVALID_PHONE")
    assert calls["stamps"] == []


def test_stamp_error_is_not_logged_with_number(api, caplog):
    def bad(phone_number, count):
        raise RuntimeError(f"fail {phone_number}")
    c = api(fake_db(add_stamp=bad)[0])
    r = c.post("/api/stamps", json={"phone_number": "1234"})
    assert r.json()["error_code"] == "STAMP_FAILED"
    assert "1234" not in caplog.text


def test_aliases_and_sold_out(api):
    c = api(fake_db()[0])
    assert c.get("/api/aliases").json() == {"아아": "아메리카노"}
    assert c.get("/api/sold-out").json() == {"sold_out": ["콜드브루"]}
    c = api(None)
    assert c.get("/api/aliases").json() == {}
    assert c.get("/api/sold-out").json()["sold_out"] == ["콜드브루", "마카롱"]


def test_log_inference_fills_empty_action_and_none_latency(monkeypatch):
    """폴백 응답(action 없음)과 미측정 지연시간(None)도 database.log_inference 규칙에 맞게 넘긴다."""
    from app import db_hooks

    got = {}

    class FakeDb:
        @staticmethod
        def log_inference(**kw):
            got.update(kw)

    monkeypatch.setattr(db_hooks, "_db", FakeDb)
    db_hooks.log_inference(utterance="x", predicted_action=None, response_obj=None,
                           stt_latency_ms=None, llm_latency_ms=None)
    assert got["predicted_action"] == "FALLBACK"
    assert got["response_obj"] == {}
    assert got["stt_latency_ms"] == 0 and got["llm_latency_ms"] == 0


def test_init_calls_init_db(monkeypatch):
    from app import db_hooks

    called = []

    class FakeDb:
        @staticmethod
        def init_db():
            called.append(1)

    monkeypatch.setattr(db_hooks, "_db", FakeDb)
    db_hooks.init()
    assert called == [1]
