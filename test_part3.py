# -*- coding: utf-8 -*-
"""
OnKiosk Backend - Part 3 검증 스크립트 (test_part3.py)
-------------------------------------------------------------------------------
실행:  python test_part3.py            (venv 의 python 사용 권장)
       python test_part3.py -v         (테스트별 결과 출력)

원칙
  * 모든 DB 테스트는 임시 폴더의 DB 를 쓴다 -> 실제 onkiosk.db 를 건드리지 않는다.
  * 음성 테스트는 '실제 모델'로 돌린다 (정답을 직접 넣어주는 모의 방식 금지).
    모델이 없으면 해당 테스트만 SKIP 으로 표시되며, 어떤 것이 건너뛰어졌는지 마지막에 출력된다.
  * 성공 케이스뿐 아니라 실패/경계 케이스(빈 입력, 품절, 동시성, 파일 삭제 등)를 함께 검증한다.
"""

import io
import os
import re
import sys
import json
import sqlite3
import tempfile
import threading
import unittest
import wave
from typing import List

import numpy as np

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import database as db
import speech_engine as se
from speech_engine import OnDeviceSpeechEngine

try:
    import soundfile as sf
except ImportError:  # pragma: no cover
    sf = None

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
KO_WAV = os.path.join(
    BASE_DIR, "models", "sherpa-onnx-sense-voice-zh-en-ja-ko-yue-int8-2024-07-17", "test_wavs", "ko.wav")
KO_REF = "조금만 생각을 하면서 살면 훨씬 편할 거야"   # SenseVoice 배포본 ko.wav 의 정답 문장

# 프론트 menu.ts 의 정답표 (database.py 와 독립적으로 다시 적어 둠: 값이 어긋나면 테스트가 잡아낸다)
EXPECTED_MENUS = {
    "아메리카노": 4500, "카페라떼": 5000, "바닐라라떼": 5500, "카푸치노": 5000, "카페모카": 5500, "콜드브루": 5000,
    "초코라떼": 5000, "녹차라떼": 5500, "고구마라떼": 5500, "딸기라떼": 6000,
    "유자차": 5000, "대추차": 5500, "캐모마일": 4500, "얼그레이": 4500, "복숭아 아이스티": 4500,
    "자몽에이드": 5500, "레몬에이드": 5500, "청포도에이드": 5500,
    "치즈케이크": 6000, "티라미수": 6500, "초코쿠키": 3000, "베이글": 3500, "플레인크루아상": 4000,
    "마카롱": 2500, "소금빵": 3500,
}
EXPECTED_SOLD_OUT = {"콜드브루", "마카롱"}


def cer(ref: str, hyp: str) -> float:
    """글자 오류율(CER): 공백/문장부호를 제거하고 편집거리를 정답 길이로 나눈다."""
    norm = lambda s: re.sub(r"[\s\.\,\?\!\'\"]", "", s)
    r, h = norm(ref), norm(hyp)
    if not r:
        return 0.0 if not h else 1.0
    prev = list(range(len(h) + 1))
    for i, rc in enumerate(r, 1):
        cur = [i]
        for j, hc in enumerate(h, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (rc != hc)))
        prev = cur
    return prev[-1] / len(r)


# ==============================================================================
# DB
# ==============================================================================
class DBTestCase(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.path = os.path.join(self._tmp.name, "test.db")
        db.init_db(self.path)

    def tearDown(self):
        self._tmp.cleanup()   # 연결이 안 닫혀 있으면 Windows 에서 여기서 PermissionError 가 난다


class TestMenus(DBTestCase):
    def test_seed_matches_frontend_exactly(self):
        menus = db.get_all_menus(db_path=self.path)
        self.assertEqual(len(menus), 25)
        self.assertEqual({m["name"]: m["price"] for m in menus}, EXPECTED_MENUS)

    def test_initial_sold_out_matches_frontend(self):
        self.assertEqual(set(db.get_sold_out_menu_names(db_path=self.path)), EXPECTED_SOLD_OUT)

    def test_categories_and_policies(self):
        counts = {c: len(db.get_all_menus(c, db_path=self.path)) for c in ("coffee", "latte", "tea", "ade", "dessert")}
        self.assertEqual(counts, {"coffee": 6, "latte": 4, "tea": 5, "ade": 3, "dessert": 7})
        self.assertEqual(db.get_menu("자몽에이드", self.path)["temp_policy"], "ICE_ONLY")
        self.assertEqual(db.get_menu("복숭아 아이스티", self.path)["temp_policy"], "ICE_ONLY")
        self.assertEqual(db.get_menu("콜드브루", self.path)["temp_policy"], "ICE_ONLY")
        self.assertEqual(db.get_menu("콜드브루", self.path)["allowed_options"],
                         {"shot": True, "syrup": True, "light": False, "milk": False})
        self.assertEqual(db.get_menu("유자차", self.path)["allowed_options"],
                         {"shot": False, "syrup": False, "light": False, "milk": False})
        self.assertEqual(db.get_menu("초코라떼", self.path)["allowed_options"],
                         {"shot": False, "syrup": True, "light": False, "milk": True})
        self.assertEqual(db.get_menu("소금빵", self.path)["temp_policy"], "NONE")
        self.assertEqual(db.get_menu("CF_AMERICANO", self.path)["name"], "아메리카노")
        self.assertEqual(db.get_menu("아메리카노", self.path)["aliases"], ["아아"])
        self.assertIsNone(db.get_menu("없는메뉴", self.path))

    def test_init_is_idempotent_and_keeps_changes(self):
        db.update_menu_sold_out("자몽에이드", True, self.path)
        db.init_db(self.path)
        db.init_db(self.path)
        self.assertEqual(len(db.get_all_menus(db_path=self.path)), 25)
        self.assertIn("자몽에이드", db.get_sold_out_menu_names(self.path))

    def test_sold_out_toggle_by_name_and_id(self):
        r = db.update_menu_sold_out("자몽에이드", True, self.path)
        self.assertTrue(r["is_sold_out"])
        r = db.update_menu_sold_out("ADE_GRAPEFRUIT", False, self.path)
        self.assertFalse(r["is_sold_out"])
        self.assertNotIn("자몽에이드", db.get_sold_out_menu_names(self.path))

    def test_sold_out_unknown_menu_returns_none(self):
        self.assertIsNone(db.update_menu_sold_out("없는메뉴", True, self.path))

    def test_alias_and_option_tables(self):
        self.assertEqual(db.get_alias_map()["아아"], "아메리카노")
        self.assertEqual(len(db.get_alias_map()), 6)
        self.assertEqual(db.get_extra_options()["shot"]["price"], 500)
        self.assertEqual(db.get_extra_options()["syrup"]["price"], 300)

    def test_old_schema_is_rejected_loudly(self):
        old = os.path.join(self._tmp.name, "old.db")
        c = sqlite3.connect(old)
        c.execute("CREATE TABLE menus (menu_id TEXT PRIMARY KEY, name TEXT)")
        c.commit(); c.close()
        with self.assertRaises(RuntimeError):
            db.init_db(old)


class TestOrders(DBTestCase):
    ITEMS = [{"menu": "아메리카노", "temperature": "HOT", "size": "REGULAR", "quantity": 1}]

    def test_order_numbers_are_sequential(self):
        nums = [db.create_order(self.ITEMS, 4500, db_path=self.path)["order_number"] for _ in range(3)]
        self.assertEqual(nums, ["A001", "A002", "A003"])

    def test_order_roundtrip(self):
        o = db.create_order(self.ITEMS, 4500, "kakao", "takeout", False, db_path=self.path)
        self.assertEqual(o["payment_method"], "KAKAO_PAY")
        got = db.get_order(o["order_id"], self.path)
        self.assertEqual(got["items"], self.ITEMS)
        self.assertEqual(got["pickup_type"], "TAKEOUT")
        self.assertFalse(got["print_receipt"])
        o2 = db.create_order(self.ITEMS, 4500, "card", "vibration_bell", True, db_path=self.path)
        self.assertEqual(o2["pickup_type"], "VIBRATION_BELL")
        self.assertIsNone(db.get_order(99999, self.path))

    def test_concurrent_orders_never_duplicate_numbers(self):
        errors: List[str] = []

        def worker():
            try:
                for _ in range(25):
                    db.create_order(self.ITEMS, 4500, db_path=self.path)
            except Exception as e:  # noqa
                errors.append(repr(e))

        threads = [threading.Thread(target=worker) for _ in range(8)]
        [t.start() for t in threads]
        [t.join() for t in threads]
        self.assertEqual(errors, [])
        with db.connect(self.path) as c:
            nums = [r[0] for r in c.execute("SELECT order_number FROM orders")]
        self.assertEqual(len(nums), 200)
        self.assertEqual(len(set(nums)), 200, "주문번호가 중복되었습니다")

    def test_deleting_a_row_does_not_cause_duplicates(self):
        db.create_order(self.ITEMS, 4500, db_path=self.path)
        db.create_order(self.ITEMS, 4500, db_path=self.path)
        with db.connect(self.path) as c:
            c.execute("DELETE FROM orders WHERE order_number = 'A001'")
        self.assertEqual(db.create_order(self.ITEMS, 4500, db_path=self.path)["order_number"], "A003")

    def test_failed_order_does_not_consume_a_number(self):
        with self.assertRaises(db.InvalidOrderError):
            db.create_order([], 1000, db_path=self.path)
        self.assertEqual(db.create_order(self.ITEMS, 4500, db_path=self.path)["order_number"], "A001")

    def test_invalid_orders_are_rejected(self):
        bad_cases = [
            ("empty items", dict(items=[], total_price=0)),
            ("items not list", dict(items="아메리카노", total_price=4500)),
            ("negative price", dict(items=self.ITEMS, total_price=-1)),
            ("float price", dict(items=self.ITEMS, total_price=4500.5)),
            ("bool price", dict(items=self.ITEMS, total_price=True)),
            ("unknown menu", dict(items=[{"menu": "없는메뉴", "quantity": 1}], total_price=1000)),
            ("missing menu key", dict(items=[{"quantity": 1}], total_price=1000)),
            ("zero qty", dict(items=[{"menu": "아메리카노", "quantity": 0}], total_price=0)),
            ("bool qty", dict(items=[{"menu": "아메리카노", "quantity": True}], total_price=4500)),
            ("bad payment", dict(items=self.ITEMS, total_price=4500, payment_method="BITCOIN")),
            ("bad pickup", dict(items=self.ITEMS, total_price=4500, pickup_type="DRONE")),
        ]
        for label, kwargs in bad_cases:
            with self.subTest(label):
                with self.assertRaises(db.InvalidOrderError):
                    db.create_order(db_path=self.path, **kwargs)
        with db.connect(self.path) as c:
            self.assertEqual(c.execute("SELECT COUNT(*) FROM orders").fetchone()[0], 0)

    def test_sold_out_menu_is_rejected(self):
        with self.assertRaises(db.SoldOutError):
            db.create_order([{"menu": "콜드브루", "quantity": 1}], 5000, db_path=self.path)
        db.update_menu_sold_out("콜드브루", False, self.path)
        self.assertEqual(
            db.create_order([{"menu": "콜드브루", "quantity": 1}], 5000, db_path=self.path)["order_number"], "A001")

    # ---- 서버 측 총액 재계산 / 정책 검증 ----
    def test_total_is_recomputed_by_server_when_omitted(self):
        o = db.create_order([{"menu": "아메리카노", "quantity": 2}], db_path=self.path)
        self.assertEqual(o["total_price"], 9000)
        self.assertEqual(db.get_order(o["order_id"], self.path)["total_price"], 9000)

    def test_wrong_total_is_rejected_and_does_not_consume_a_number(self):
        for wrong in (1, 4499, 4501, 0):
            with self.subTest(total=wrong):
                with self.assertRaises(db.InvalidOrderError):
                    db.create_order(self.ITEMS, wrong, db_path=self.path)
        self.assertEqual(db.create_order(self.ITEMS, 4500, db_path=self.path)["order_number"], "A001")

    def test_option_and_size_surcharges_match_frontend_formula(self):
        cases = [
            ("LARGE +500", {"menu": "아메리카노", "size": "LARGE"}, 5000),
            ("샷 +500", {"menu": "아메리카노", "options": {"shot": 1}}, 5000),
            ("시럽 +300", {"menu": "아메리카노", "options": {"syrup": 1}}, 4800),
            ("우유 변경 +500", {"menu": "카페라떼", "options": {"milk": "SOY"}}, 5500),
            ("우유 기본(REGULAR) 0원", {"menu": "카페라떼", "options": {"milk": "REGULAR"}}, 5000),
            ("연하게 0원", {"menu": "아메리카노", "options": {"strength": "LIGHT"}}, 4500),
            ("전부 선택", {"menu": "카페라떼", "size": "LARGE", "options": {"shot": 1, "syrup": 1, "milk": "OAT"}}, 5000 + 500 + 500 + 300 + 500),
            ("샷 개수 2도 선택 여부로 과금(프론트 동일)", {"menu": "아메리카노", "options": {"shot": 2}}, 5000),
            ("수량 3", {"menu": "아메리카노", "quantity": 3, "size": "LARGE"}, 15000),
        ]
        for label, item, expected in cases:
            with self.subTest(label):
                self.assertEqual(db.calculate_order_total([item], self.path), expected)
        mixed = [{"menu": "아메리카노", "quantity": 2}, {"menu": "치즈케이크", "quantity": 1}]
        self.assertEqual(db.calculate_order_total(mixed, self.path), 9000 + 6000)

    def test_ice_only_menu_cannot_be_ordered_hot(self):
        db.update_menu_sold_out("콜드브루", False, self.path)
        for menu in ("자몽에이드", "복숭아 아이스티", "콜드브루"):
            with self.subTest(menu):
                with self.assertRaises(db.InvalidOrderError):
                    db.create_order([{"menu": menu, "temperature": "HOT"}], db_path=self.path)
                self.assertGreater(db.create_order([{"menu": menu, "temperature": "ICE"}], db_path=self.path)["total_price"], 0)

    def test_per_menu_option_policy_matches_dataset(self):
        """데이터셋 '메뉴_정책' 시트 및 파트 2 합의안 기준 메뉴별 옵션 허용/불가 검증."""
        db.update_menu_sold_out("콜드브루", False, self.path)
        disallowed = [
            ("아메리카노 우유변경 불가", {"menu": "아메리카노", "options": {"milk": "SOY"}}),
            ("아메리카노 우유 기본값도 불가(우유 없음)", {"menu": "아메리카노", "options": {"milk": "REGULAR"}}),
            ("콜드브루 연하게 불가", {"menu": "콜드브루", "options": {"strength": "LIGHT"}}),
            ("콜드브루 light=True 불가", {"menu": "콜드브루", "options": {"light": True}}),
            ("콜드브루 우유변경 불가", {"menu": "콜드브루", "options": {"milk": "OAT"}}),
            ("초코라떼 샷추가 불가", {"menu": "초코라떼", "options": {"shot": 1}}),
            ("초코라떼 연하게 불가", {"menu": "초코라떼", "options": {"strength": "LIGHT"}}),
            ("유자차 시럽추가 불가", {"menu": "유자차", "options": {"syrup": 1}}),
            ("유자차 샷추가 불가", {"menu": "유자차", "options": {"shot": 1}}),
            ("유자차 연하게 불가", {"menu": "유자차", "options": {"strength": "LIGHT"}}),
            ("유자차 우유변경 불가", {"menu": "유자차", "options": {"milk": "SOY"}}),
            ("캐모마일 시럽추가 불가", {"menu": "캐모마일", "options": {"syrup": 1}}),
            ("복숭아 아이스티 샷추가 불가", {"menu": "복숭아 아이스티", "options": {"shot": 1}}),
            ("복숭아 아이스티 연하게 불가", {"menu": "복숭아 아이스티", "options": {"light": True}}),
            ("자몽에이드 샷추가 불가", {"menu": "자몽에이드", "options": {"shot": 1}}),
            ("자몽에이드 우유변경 불가", {"menu": "자몽에이드", "options": {"milk": "SOY"}}),
            ("HOT 음료 얼음 옵션 불가", {"menu": "아메리카노", "temperature": "HOT", "options": {"ice": "LESS"}}),
            ("잘못된 얼음 옵션값", {"menu": "아메리카노", "temperature": "ICE", "options": {"ice": "HALF"}}),
            ("잘못된 농도 옵션값", {"menu": "아메리카노", "options": {"strength": "STRONG"}}),
            ("잘못된 우유 옵션값", {"menu": "카페라떼", "options": {"milk": "ALMOND"}}),
            ("알 수 없는 옵션 키", {"menu": "아메리카노", "options": {"sugar": 1}}),
        ]
        for label, item in disallowed:
            with self.subTest(label):
                with self.assertRaises(db.InvalidOrderError):
                    db.calculate_order_total([item], self.path)

        # 허용 케이스: 얼음 NONE/MORE, 프론트 bool 형식(milk=True/False), 복숭아 아이스티·에이드 시럽 추가 등
        allowed = [
            ("ICE 얼음 NONE 허용", {"menu": "아메리카노", "temperature": "ICE", "options": {"ice": "NONE"}}, 4500),
            ("ICE 얼음 MORE 허용", {"menu": "콜드브루", "temperature": "ICE", "options": {"ice": "MORE", "shot": 1}}, 5500),
            ("프론트 bool 미선택(milk=False)은 아메리카노에서도 통과", {"menu": "아메리카노", "options": {"shot": False, "milk": False}}, 4500),
            ("프론트 bool 우유변경(milk=True) +500", {"menu": "초코라떼", "options": {"syrup": True, "milk": True}}, 5800),
            ("복숭아 아이스티 시럽 +300", {"menu": "복숭아 아이스티", "options": {"syrup": 1, "ice": "LESS"}}, 4800),
            ("자몽에이드 시럽 +300", {"menu": "자몽에이드", "options": {"syrup": 1}}, 5800),
        ]
        for label, item, expected in allowed:
            with self.subTest(label):
                self.assertEqual(db.calculate_order_total([item], self.path), expected)

    def test_dessert_cannot_have_temperature_size_or_options(self):
        for extra in ({"temperature": "HOT"}, {"size": "LARGE"}, {"options": {"shot": 1}}):
            with self.subTest(extra=extra):
                with self.assertRaises(db.InvalidOrderError):
                    db.create_order([{"menu": "치즈케이크", **extra}], db_path=self.path)
        self.assertEqual(db.create_order([{"menu": "치즈케이크", "temperature": None}], db_path=self.path)["total_price"], 6000)

    def test_quantity_and_line_limits(self):
        self.assertEqual(db.create_order([{"menu": "초코쿠키", "quantity": 99}], db_path=self.path)["total_price"], 3000 * 99)
        for qty in (100, 10 ** 9, 0, -1):
            with self.subTest(qty=qty):
                with self.assertRaises(db.InvalidOrderError):
                    db.create_order([{"menu": "초코쿠키", "quantity": qty}], db_path=self.path)
        with self.assertRaises(db.InvalidOrderError):
            db.create_order([{"menu": "초코쿠키"}] * (db.MAX_ORDER_LINES + 1), db_path=self.path)

    def test_bad_option_and_enum_values_are_rejected(self):
        bad_items = [
            {"menu": "아메리카노", "temperature": "WARM"},
            {"menu": "아메리카노", "size": "TALL"},
            {"menu": "아메리카노", "options": "shot"},
            {"menu": "아메리카노", "options": {"shot": -1}},
            {"menu": "아메리카노", "options": {"shot": "1"}},
            {"menu": "카페라떼", "options": {"milk": 3}},
        ]
        for it in bad_items:
            with self.subTest(item=it):
                with self.assertRaises(db.InvalidOrderError):
                    db.create_order([it], db_path=self.path)
        with db.connect(self.path) as c:
            self.assertEqual(c.execute("SELECT COUNT(*) FROM orders").fetchone()[0], 0)

    def test_calculate_order_total_rejects_sold_out_and_unknown(self):
        with self.assertRaises(db.SoldOutError):
            db.calculate_order_total([{"menu": "마카롱"}], self.path)
        with self.assertRaises(db.InvalidOrderError):
            db.calculate_order_total([{"menu": "없는메뉴"}], self.path)


class TestStamps(DBTestCase):
    def test_accumulates_and_reads_back(self):
        db.add_stamp("1234", 1, self.path)
        r = db.add_stamp("1234", 2, self.path)
        self.assertEqual(r["total_stamps"], 3)
        self.assertEqual(db.get_stamp_count("1234", self.path), 3)
        self.assertEqual(db.get_stamp_count("9999", self.path), 0)

    def test_formatting_variants_are_same_customer(self):
        db.add_stamp("010-1234-5678", 1, self.path)
        self.assertEqual(db.add_stamp("01012345678", 1, self.path)["total_stamps"], 2)

    def test_phone_number_is_never_stored_in_plain_text(self):
        db.add_stamp("010-1234-5678", 1, self.path)
        with db.connect(self.path) as c:
            row = dict(c.execute("SELECT * FROM stamps").fetchone())
        self.assertNotIn("01012345678", json.dumps(row))
        self.assertEqual(row["phone_hint"], "5678")

    def test_stored_key_is_not_a_plain_sha256_that_can_be_brute_forced(self):
        """DB 만 유출돼도 전화번호를 전수 대입으로 역산할 수 없어야 한다(비밀키가 섞여 있어야 함)."""
        import hashlib
        db.add_stamp("010-1234-5678", 1, self.path)
        db.add_stamp("5678", 1, self.path)
        with db.connect(self.path) as c:
            stored = {r[0] for r in c.execute("SELECT phone_hash FROM stamps")}
        self.assertEqual(len(stored), 2)
        self.assertNotIn(hashlib.sha256(b"01012345678").hexdigest(), stored)
        self.assertNotIn(hashlib.sha256(b"5678").hexdigest(), stored)

    def test_secret_key_file_is_created_once_and_reused(self):
        key_path = os.path.join(self._tmp.name, db.STAMP_KEY_FILENAME)
        self.assertFalse(os.path.exists(key_path))
        db.add_stamp("1234", 1, self.path)
        self.assertTrue(os.path.exists(key_path))
        with open(key_path, encoding="ascii") as f:
            first = f.read()
        self.assertGreaterEqual(len(first.strip()), 64)
        db.add_stamp("1234", 1, self.path)
        with open(key_path, encoding="ascii") as f:
            self.assertEqual(f.read(), first, "비밀키가 바뀌면 적립이 끊긴다")
        self.assertEqual(db.get_stamp_count("1234", self.path), 2)

    def test_different_secret_gives_different_key(self):
        db.add_stamp("1234", 1, self.path)
        old = os.environ.get("ONKIOSK_STAMP_SECRET")
        os.environ["ONKIOSK_STAMP_SECRET"] = "another-secret-value-for-test-0123456789"
        try:
            self.assertEqual(db.get_stamp_count("1234", self.path), 0)   # 다른 비밀키 -> 다른 키 -> 기록이 안 보임
        finally:
            if old is None:
                os.environ.pop("ONKIOSK_STAMP_SECRET", None)
            else:
                os.environ["ONKIOSK_STAMP_SECRET"] = old

    def test_corrupted_secret_file_is_reported_not_silently_replaced(self):
        with open(os.path.join(self._tmp.name, db.STAMP_KEY_FILENAME), "w") as f:
            f.write("abc")
        with self.assertRaises(db.KioskDBError):
            db.add_stamp("1234", 1, self.path)

    def test_concurrent_first_use_creates_a_single_consistent_key(self):
        errors: List[str] = []

        def worker():
            try:
                db.add_stamp("1234", 1, self.path)
            except Exception as e:  # noqa
                errors.append(repr(e))

        ts = [threading.Thread(target=worker) for _ in range(8)]
        [t.start() for t in ts]
        [t.join() for t in ts]
        self.assertEqual(errors, [])
        self.assertEqual(db.get_stamp_count("1234", self.path), 8)

    def test_known_limitation_full_number_and_last4_are_separate_records(self):
        """문서화된 한계: 같은 고객이 전체 번호와 뒤 4자리를 각각 입력하면 기록이 갈린다. 정책을 바꾸면 이 테스트도 바꿀 것."""
        db.add_stamp("010-1234-5678", 1, self.path)
        db.add_stamp("5678", 1, self.path)
        self.assertEqual(db.get_stamp_count("010-1234-5678", self.path), 1)
        self.assertEqual(db.get_stamp_count("5678", self.path), 1)

    def test_invalid_input_is_rejected(self):
        # 프론트 phoneSchema 와 동일하게 4자리 또는 11자리만 허용 (5~10자리 거부 포함)
        for bad in ("", "12", "12345", "12345678", "0101234567", "abc", "1" * 12, None):
            with self.subTest(phone=bad):
                with self.assertRaises(db.InvalidStampError):
                    db.add_stamp(bad, 1, self.path)
        for bad_count in (0, -1, True, 1.5):
            with self.subTest(count=bad_count):
                with self.assertRaises(db.InvalidStampError):
                    db.add_stamp("1234", bad_count, self.path)


class TestInferenceLogs(DBTestCase):
    def test_screen_context_normalization_matches_part2_spec(self):
        """프론트 9개 화면 키(start/menu/option/cart/point/pay/paying/receipt/done) -> 데이터셋 6개 화면 매핑 검증."""
        expected_map = {
            "start": "menu",
            "menu": "menu",
            "option": "menu",
            "cart": "cart",
            "point": "loyalty",
            "loyalty": "loyalty",
            "pay": "payment",
            "paying": "payment",
            "payment": "payment",
            "receipt": "receipt_choice",
            "receipt_choice": "receipt_choice",
            "done": "complete",
            "complete": "complete",
        }
        for raw_screen, expected in expected_map.items():
            with self.subTest(screen=raw_screen):
                self.assertEqual(db.normalize_screen_context(raw_screen), expected)
                log_id = db.log_inference("발화", "no_action", {"action": "no_action"},
                                          screen_context=raw_screen, db_path=self.path)
                with db.connect(self.path) as c:
                    saved = c.execute("SELECT screen_context FROM inference_logs WHERE id = ?", (log_id,)).fetchone()[0]
                self.assertEqual(saved, expected)

        with self.assertRaises(ValueError):
            db.normalize_screen_context("unknown_screen")
        with self.assertRaises(ValueError):
            db.log_inference("발화", "no_action", {"action": "no_action"},
                             screen_context="unknown_screen", db_path=self.path)
    def test_p95_uses_nearest_rank(self):
        for ms in range(1, 101):                                  # 1..100 ms -> 95번째 값 = 95
            db.log_inference("x", "no_action", {"a": 1}, stt_latency_ms=0.0, llm_latency_ms=float(ms), db_path=self.path)
        self.assertEqual(db.get_latency_stats(self.path)["p95_total_ms"], 95.0)

    def test_p95_small_samples(self):
        db.log_inference("x", "no_action", {"a": 1}, llm_latency_ms=100.0, db_path=self.path)
        self.assertEqual(db.get_latency_stats(self.path)["p95_total_ms"], 100.0)     # 1건이면 그 값
        db.log_inference("x", "no_action", {"a": 1}, llm_latency_ms=900.0, db_path=self.path)
        self.assertEqual(db.get_latency_stats(self.path)["p95_total_ms"], 900.0)     # 2건이면 큰 값(ceil(1.9)=2번째)

    def test_alias_is_not_accepted_by_create_order_but_alias_map_resolves_it(self):
        with self.assertRaises(db.InvalidOrderError):
            db.create_order([{"menu": "아아", "quantity": 1}], db_path=self.path)
        resolved = db.get_alias_map()["아아"]
        self.assertEqual(db.create_order([{"menu": resolved}], db_path=self.path)["total_price"], 4500)

    def test_empty_stats_are_none_not_fake_100_percent(self):
        s = db.get_latency_stats(self.path)
        self.assertEqual(s["total_calls"], 0)
        self.assertIsNone(s["avg_total_ms"])
        self.assertIsNone(s["valid_schema_rate"])

    def test_stats_values(self):
        for stt, llm, ok, fb in [(100, 900, True, False), (200, 1300, True, False),
                                 (150, 3000, False, True), (100, 1000, True, False)]:
            db.log_inference("발화", "add_to_cart", {"action": "add_to_cart"},
                             is_valid_schema=ok, fallback_triggered=fb,
                             stt_latency_ms=stt, llm_latency_ms=llm, db_path=self.path)
        s = db.get_latency_stats(self.path)
        self.assertEqual(s["total_calls"], 4)
        self.assertEqual(s["avg_total_ms"], round((1000 + 1500 + 3150 + 1100) / 4, 1))
        self.assertEqual(s["max_total_ms"], 3150.0)
        self.assertEqual(s["valid_schema_rate"], 75.0)
        self.assertEqual(s["within_target_rate"], 75.0)   # 2.5초 초과 1건
        self.assertEqual(s["fallback_count"], 1)

    def test_invalid_logs_are_rejected(self):
        with self.assertRaises(ValueError):
            db.log_inference("x", "add_to_cart", {}, stt_latency_ms=-1, db_path=self.path)
        with self.assertRaises(ValueError):
            db.log_inference("x", "", {}, db_path=self.path)

    def test_korean_json_is_stored_readable(self):
        db.log_inference("아아 주세요", "add_to_cart", {"menu": "아메리카노"}, db_path=self.path)
        with db.connect(self.path) as c:
            raw = c.execute("SELECT response_json FROM inference_logs").fetchone()[0]
        self.assertIn("아메리카노", raw)


class TestConnectionHygiene(unittest.TestCase):
    def test_db_file_can_be_deleted_right_after_use(self):
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "t.db")
            db.init_db(p)
            for _ in range(20):
                db.get_all_menus(db_path=p)
                db.get_sold_out_menu_names(p)
            db.create_order([{"menu": "아메리카노", "quantity": 1}], 4500, db_path=p)
            for f in os.listdir(d):
                os.remove(os.path.join(d, f))   # 연결 누수가 있으면 PermissionError

    def test_env_var_overrides_db_path(self):
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "env.db")
            os.environ["ONKIOSK_DB_PATH"] = p
            try:
                db.init_db()
                self.assertTrue(os.path.exists(p))
                self.assertEqual(len(db.get_all_menus()), 25)
            finally:
                del os.environ["ONKIOSK_DB_PATH"]


# ==============================================================================
# 음성 - 순수 함수 (모델 불필요)
# ==============================================================================
class TestSpeechTextHelpers(unittest.TestCase):
    def test_normalize_removes_tags_and_fixes_spelling(self):
        self.assertEqual(se.normalize_korean_utterance("<|ko|><|NEUTRAL|> 카모마일  주세요 "), "캐모마일 주세요")
        self.assertEqual(se.normalize_korean_utterance("크로와상 하나"), "크루아상 하나")
        self.assertEqual(se.normalize_korean_utterance("<|Speech|>"), "")
        self.assertEqual(se.normalize_korean_utterance(""), "")
        self.assertEqual(se.normalize_korean_utterance(None), "")

    def test_tts_text_prep_converts_latin_letters(self):
        self.assertEqual(se.prepare_tts_text("주문번호 A012번입니다"), "주문번호 에이012번입니다")
        self.assertEqual(se.prepare_tts_text("ICE 아메리카노"), "아이스 아메리카노")
        self.assertEqual(se.prepare_tts_text("HOT 라떼"), "핫 라떼")
        self.assertEqual(se.prepare_tts_text("결제 수단을 선택해 주세요"), "결제 수단을 선택해 주세요")

    def test_discard_zeroes_buffers_and_deletes_file(self):
        a = np.ones(100, dtype=np.float32)
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as f:
            f.write(b"x")
        self.assertTrue(OnDeviceSpeechEngine._discard([a], f.name, True))
        self.assertFalse(os.path.exists(f.name))
        self.assertEqual(float(np.abs(a).sum()), 0.0)

    def test_resample_changes_length_correctly(self):
        x = np.sin(np.linspace(0, 200, 48000)).astype(np.float32)
        self.assertAlmostEqual(len(se._lowpass_resample(x, 48000, 16000)), 16000, delta=2)
        self.assertEqual(len(se._lowpass_resample(x, 16000, 16000)), 48000)

    def test_cer_helper(self):
        self.assertEqual(cer("결제 수단", "결제수단"), 0.0)
        self.assertAlmostEqual(cer("아메리카노", "아메리카"), 0.2)


class TestSpeechFailureModes(unittest.TestCase):
    """모델이 없을 때 조용히 가짜 결과를 만들지 않고 명확히 실패하는지."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.engine = OnDeviceSpeechEngine(models_dir=self._tmp.name, mock=False)

    def tearDown(self):
        self._tmp.cleanup()

    def test_status_reports_missing_models(self):
        st = self.engine.status()
        self.assertFalse(st["stt_ready"])
        self.assertFalse(st["tts_ready"])
        self.assertTrue(any("STT" in m for m in st["missing"]))

    def test_stt_without_models_raises(self):
        with self.assertRaises(se.SpeechModelNotLoadedError):
            self.engine.transcribe_audio(samples=np.zeros(16000, dtype=np.float32), sample_rate=16000)

    def test_tts_without_models_raises(self):
        with self.assertRaises(se.SpeechModelNotLoadedError):
            self.engine.synthesize_speech("안녕하세요")

    def test_tts_validates_input_before_anything_else(self):
        for bad in ("", "   ", None, "가" * 301):
            with self.subTest(text=bad):
                with self.assertRaises(ValueError):
                    self.engine.synthesize_speech(bad)
        with self.assertRaises(ValueError):
            self.engine.synthesize_speech("안녕", speed=5.0)

    def test_tts_empty_audio_raises_synthesis_error(self):
        from types import SimpleNamespace
        eng = OnDeviceSpeechEngine(models_dir=self._tmp.name)
        eng._tts = SimpleNamespace(generate=lambda text, sid=0, speed=1.0: SimpleNamespace(samples=[], sample_rate=22050))
        with self.assertRaises(se.SpeechSynthesisError):
            eng.synthesize_speech("안녕하세요")

    def test_mock_mode_is_explicit_and_labelled(self):
        mock = OnDeviceSpeechEngine(models_dir=self._tmp.name, mock=True)
        with self.assertRaises(se.SpeechModelNotLoadedError):          # mock_text 없이는 가짜를 만들지 않음
            mock.transcribe_audio(samples=np.zeros(16000, dtype=np.float32), sample_rate=16000)
        r = mock.transcribe_audio(samples=np.zeros(16000, dtype=np.float32), sample_rate=16000, mock_text="아아 주세요")
        self.assertTrue(r["is_mock"])
        self.assertEqual(r["engine_mode"], "MOCK")
        t = mock.synthesize_speech("안녕하세요")
        self.assertTrue(t["is_mock"])

    def test_mock_mode_reports_whether_source_file_was_deleted(self):
        mock = OnDeviceSpeechEngine(models_dir=self._tmp.name, mock=True)
        fd, path = tempfile.mkstemp(suffix=".wav")
        os.close(fd)
        r = mock.transcribe_audio(wav_path=path, mock_text="아아 주세요")
        self.assertIs(r["source_file_deleted"], True)
        self.assertFalse(os.path.exists(path))
        r2 = mock.transcribe_audio(samples=np.zeros(16000, dtype=np.float32), sample_rate=16000, mock_text="아아")
        self.assertIsNone(r2["source_file_deleted"])        # 파일 입력이 아니면 해당 없음


# ==============================================================================
# 음성 - 실제 모델
# ==============================================================================
_probe = OnDeviceSpeechEngine()
_ps = _probe.status()
MODELS_READY = bool(_ps["stt_ready"] and _ps["tts_ready"]) and sf is not None
SKIP_MSG = f"음성 모델/패키지 없음 -> python tools/download_models.py 실행 필요 (missing={_ps['missing']})"
ENGINE = OnDeviceSpeechEngine(num_threads=2) if MODELS_READY else None


@unittest.skipUnless(MODELS_READY, SKIP_MSG)
class TestRealSpeech(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.engine = ENGINE
        cls.engine.warm_up()
        cls.ko = None
        if os.path.exists(KO_WAV):
            cls.ko, cls.ko_sr = sf.read(KO_WAV, dtype="float32")

    # ---- TTS ----
    def test_tts_produces_real_audible_speech(self):
        r = self.engine.synthesize_speech("아이스 아메리카노 라지 한 잔을 장바구니에 담았습니다.")
        self.assertFalse(r["is_mock"])
        wf = wave.open(io.BytesIO(r["wav_bytes"]))
        data = np.frombuffer(wf.readframes(wf.getnframes()), dtype="<i2").astype(np.float32) / 32768.0
        self.assertEqual(wf.getnchannels(), 1)
        self.assertGreater(len(data) / wf.getframerate(), 2.0)        # 최소 2초 이상의 발화
        self.assertGreater(float(np.abs(data).max()), 0.1)            # 무음이 아님
        self.assertGreater(float(np.sqrt((data ** 2).mean())), 0.01)  # 평균 에너지도 있음

    def test_tts_second_call_is_cached_and_identical(self):
        a = self.engine.synthesize_speech("결제 수단을 선택해 주세요.")
        b = self.engine.synthesize_speech("결제 수단을 선택해 주세요.")
        self.assertTrue(b["cached"])
        self.assertEqual(a["wav_bytes"], b["wav_bytes"])
        self.assertEqual(b["tts_latency_ms"], 0.0)

    def test_tts_handles_digits_and_order_number(self):
        for text in ("총 9,500원입니다.", "주문번호 A012번입니다.", "ICE 아메리카노 2잔을 담았습니다."):
            with self.subTest(text=text):
                self.assertGreater(self.engine.synthesize_speech(text)["duration_s"], 1.0)

    def test_tts_is_fast_enough(self):
        r = self.engine.synthesize_speech("카페라떼 따뜻하게 레귤러 한 잔 담았습니다.", speed=1.05)
        self.assertLess(r["rtf"], 1.0, "실시간보다 느린 합성은 키오스크에서 쓸 수 없다")

    # ---- STT ----
    def test_stt_recognizes_real_korean_wav(self):
        if self.ko is None:
            self.skipTest("ko.wav 없음")
        r = self.engine.transcribe_audio(samples=self.ko, sample_rate=self.ko_sr)
        self.assertTrue(r["speech_detected"])
        self.assertLessEqual(cer(KO_REF, r["text"]), 0.25, r["text"])
        self.assertNotIn("<|", r["text"])

    def test_stt_silence_and_noise_are_not_speech(self):
        rng = np.random.RandomState(0)
        for name, x in (("silence", np.zeros(32000, dtype=np.float32)),
                        ("noise", (rng.randn(32000) * 0.01).astype(np.float32))):
            with self.subTest(name):
                r = self.engine.transcribe_audio(samples=x, sample_rate=16000)
                self.assertFalse(r["speech_detected"])
                self.assertEqual(r["text"], "")

    def test_stt_too_short_input_is_handled(self):
        r = self.engine.transcribe_audio(samples=np.zeros(100, dtype=np.float32), sample_rate=16000)
        self.assertFalse(r["speech_detected"])

    def test_stt_accepts_wav_bytes_and_other_sample_rates(self):
        if self.ko is None:
            self.skipTest("ko.wav 없음")
        up = np.interp(np.linspace(0, len(self.ko) - 1, len(self.ko) * 3), np.arange(len(self.ko)), self.ko).astype(np.float32)
        buf = io.BytesIO()
        sf.write(buf, up, 48000, format="WAV", subtype="PCM_16")
        r = self.engine.transcribe_audio(audio_bytes=buf.getvalue())
        self.assertLessEqual(cer(KO_REF, r["text"]), 0.3, r["text"])

    def test_stt_accepts_stereo_and_raw_pcm16(self):
        if self.ko is None:
            self.skipTest("ko.wav 없음")
        stereo = np.stack([self.ko, self.ko], axis=1)
        r1 = self.engine.transcribe_audio(samples=stereo, sample_rate=self.ko_sr)
        self.assertLessEqual(cer(KO_REF, r1["text"]), 0.25)
        pcm = (np.clip(self.ko, -1, 1) * 32767).astype("<i2").tobytes()
        r2 = self.engine.transcribe_audio(audio_bytes=pcm, sample_rate=16000, raw_pcm16=True)
        self.assertLessEqual(cer(KO_REF, r2["text"]), 0.25)

    def test_stt_does_not_modify_callers_array(self):
        if self.ko is None:
            self.skipTest("ko.wav 없음")
        before = self.ko.copy()
        self.engine.transcribe_audio(samples=self.ko, sample_rate=self.ko_sr)
        np.testing.assert_array_equal(self.ko, before)

    def test_stt_deletes_source_file_after_success(self):
        if self.ko is None:
            self.skipTest("ko.wav 없음")
        fd, path = tempfile.mkstemp(suffix=".wav"); os.close(fd)
        sf.write(path, self.ko, self.ko_sr)
        r = self.engine.transcribe_audio(wav_path=path)
        self.assertFalse(os.path.exists(path), "음성 원본이 남아 있습니다")
        self.assertTrue(r["audio_discarded"])
        self.assertTrue(r["source_file_deleted"])

    def test_stt_keeps_file_when_asked(self):
        if self.ko is None:
            self.skipTest("ko.wav 없음")
        fd, path = tempfile.mkstemp(suffix=".wav"); os.close(fd)
        try:
            sf.write(path, self.ko, self.ko_sr)
            self.engine.transcribe_audio(wav_path=path, delete_source_file=False)
            self.assertTrue(os.path.exists(path))
        finally:
            os.remove(path)

    def test_stt_deletes_source_file_even_when_decoding_fails(self):
        fd, path = tempfile.mkstemp(suffix=".wav")
        with os.fdopen(fd, "wb") as f:
            f.write(b"this is not audio")
        with self.assertRaises(se.AudioDecodeError):
            self.engine.transcribe_audio(wav_path=path)
        self.assertFalse(os.path.exists(path), "오류가 나도 음성 원본은 삭제되어야 합니다")

    def test_stt_rejects_bad_inputs(self):
        with self.assertRaises(se.AudioDecodeError):
            self.engine.transcribe_audio()
        with self.assertRaises(se.AudioDecodeError):
            self.engine.transcribe_audio(audio_bytes=b"x", samples=np.zeros(10, dtype=np.float32), sample_rate=16000)
        with self.assertRaises(se.AudioDecodeError):
            self.engine.transcribe_audio(samples=np.zeros(10, dtype=np.float32))               # sample_rate 누락
        with self.assertRaises(se.AudioDecodeError):
            self.engine.transcribe_audio(audio_bytes=b"abc", sample_rate=16000, raw_pcm16=True)  # 홀수 바이트
        with self.assertRaises(se.AudioDecodeError):
            self.engine.transcribe_audio(audio_bytes=b"garbage-bytes")
        with self.assertRaises(se.AudioDecodeError):
            self.engine.transcribe_audio(samples=np.zeros(16000 * 31, dtype=np.float32), sample_rate=16000)

    def test_stt_rejects_oversized_input_before_decoding(self):
        # 상한만 단독으로 검증: 샘플레이트를 크게 잡아 '길이(초)' 검사는 통과하고 '바이트 크기'만 초과하게 만든다.
        n_bytes = se.MAX_AUDIO_BYTES + 2
        self.assertLess((n_bytes // 2) / 1_000_000, se.MAX_AUDIO_SECONDS)
        with self.assertRaisesRegex(se.AudioDecodeError, "너무 큽니다"):
            self.engine.transcribe_audio(audio_bytes=b"\x00" * n_bytes, sample_rate=1_000_000, raw_pcm16=True)

    def test_stt_rejects_oversized_wav_bytes_by_size_not_by_duration(self):
        buf = io.BytesIO()
        sf.write(buf, np.zeros(4_500_000, dtype=np.float32), 1_000_000, format="WAV", subtype="FLOAT")   # 4.5초, 약 18MB
        data = buf.getvalue()
        self.assertGreater(len(data), se.MAX_AUDIO_BYTES)
        with self.assertRaisesRegex(se.AudioDecodeError, "너무 큽니다"):
            self.engine.transcribe_audio(audio_bytes=data)

    def test_stt_rejects_oversized_file_and_still_deletes_it(self):
        fd, path = tempfile.mkstemp(suffix=".wav")
        os.close(fd)
        sf.write(path, np.zeros(4_500_000, dtype=np.float32), 1_000_000, subtype="FLOAT")             # 유효한 WAV, 약 18MB
        self.assertGreater(os.path.getsize(path), se.MAX_AUDIO_BYTES)
        with self.assertRaisesRegex(se.AudioDecodeError, "너무 큽니다"):
            self.engine.transcribe_audio(wav_path=path)
        self.assertFalse(os.path.exists(path), "거부된 입력 파일도 삭제되어야 합니다")

    def test_stt_missing_file_is_a_clean_error(self):
        with self.assertRaises(se.AudioDecodeError):
            self.engine.transcribe_audio(wav_path=os.path.join(tempfile.gettempdir(), "no_such_audio_file.wav"))

    def test_stt_is_faster_than_realtime(self):
        if self.ko is None:
            self.skipTest("ko.wav 없음")
        r = self.engine.transcribe_audio(samples=self.ko, sample_rate=self.ko_sr)
        self.assertLess(r["rtf"], 1.0)

    # ---- TTS -> STT 왕복 (실제 두 모델 연동) ----
    def test_tts_to_stt_roundtrip_on_kiosk_phrases(self):
        phrases = ["결제 수단을 선택해 주세요.", "카페라떼 따뜻하게 레귤러 한 잔 주세요.",
                   "죄송합니다. 잘 못 들었어요. 다시 말씀해 주세요."]
        scores = []
        for p in phrases:
            wav = self.engine.synthesize_speech(p)["wav_bytes"]
            out = self.engine.transcribe_audio(audio_bytes=wav)["text"]
            scores.append(cer(p, out))
        print(f"\n   [왕복 CER] " + ", ".join(f"{s:.2f}" for s in scores) + f"  (평균 {sum(scores)/len(scores):.2f})")
        self.assertLessEqual(sum(scores) / len(scores), 0.30)

    def test_thread_safety_parallel_stt_and_tts(self):
        if self.ko is None:
            self.skipTest("ko.wav 없음")
        errors: List[str] = []

        def stt_job():
            try:
                r = self.engine.transcribe_audio(samples=self.ko, sample_rate=self.ko_sr)
                if cer(KO_REF, r["text"]) > 0.3:
                    errors.append("stt mismatch: " + r["text"])
            except Exception as e:  # noqa
                errors.append(repr(e))

        def tts_job(i):
            try:
                self.engine.synthesize_speech(f"주문번호 {i}번 고객님 음료가 준비되었습니다.")
            except Exception as e:  # noqa
                errors.append(repr(e))

        ts = [threading.Thread(target=stt_job) for _ in range(4)] + [threading.Thread(target=tts_job, args=(i,)) for i in range(4)]
        [t.start() for t in ts]
        [t.join() for t in ts]
        self.assertEqual(errors, [])


@unittest.skipUnless(MODELS_READY, SKIP_MSG)
class TestEndToEndPart3(unittest.TestCase):
    """STT -> 로그 기록 -> 통계: 파트 3 두 모듈이 실제로 맞물려 동작하는지."""

    def test_stt_latency_flows_into_db_stats(self):
        if not os.path.exists(KO_WAV):
            self.skipTest("ko.wav 없음")
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "e2e.db")
            db.init_db(p)
            wav, sr = sf.read(KO_WAV, dtype="float32")
            for _ in range(3):
                s = ENGINE.transcribe_audio(samples=wav, sample_rate=sr)
                db.log_inference(s["text"], "no_action", {"action": "no_action"}, stt_latency_ms=s["stt_latency_ms"],
                                 llm_latency_ms=1000.0, db_path=p)
            st = db.get_latency_stats(p)
            self.assertEqual(st["total_calls"], 3)
            self.assertGreater(st["avg_stt_ms"], 0)
            self.assertEqual(st["within_target_rate"], 100.0)
            print(f"\n   [E2E] 평균 STT {st['avg_stt_ms']} ms / 평균 총 지연(LLM 1000ms 가정) {st['avg_total_ms']} ms")


class TestDownloaderSafeExtract(unittest.TestCase):
    """tools/download_models.py 의 압축 해제가 악성/비정상 tar 를 거부하는지 (네트워크 사용 안 함)."""
    TOOL = os.path.join(BASE_DIR, "tools", "download_models.py")

    @classmethod
    def setUpClass(cls):
        import importlib.util
        spec = importlib.util.spec_from_file_location("download_models", cls.TOOL)
        cls.tool = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.tool)

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = self._tmp.name
        self.dest = os.path.join(self.root, "models")
        os.makedirs(self.dest)

    def tearDown(self):
        self._tmp.cleanup()

    def _make_tar(self, entries):
        """entries: [(name, bytes | ('link', target))]"""
        import tarfile
        path = os.path.join(self.root, "a.tar.bz2")
        with tarfile.open(path, "w:bz2") as tf:
            for name, content in entries:
                info = tarfile.TarInfo(name)
                if isinstance(content, tuple):
                    info.type = tarfile.SYMTYPE
                    info.linkname = content[1]
                    tf.addfile(info)
                else:
                    info.size = len(content)
                    tf.addfile(info, io.BytesIO(content))
        return path

    def test_normal_archive_is_extracted(self):
        path = self._make_tar([("model-dir/model.onnx", b"abc"), ("model-dir/tokens.txt", b"x")])
        self.tool.safe_extract(path, self.dest)
        with open(os.path.join(self.dest, "model-dir", "model.onnx"), "rb") as f:
            self.assertEqual(f.read(), b"abc")

    def test_path_traversal_is_rejected_and_nothing_is_written_outside(self):
        path = self._make_tar([("ok/file.txt", b"1"), ("../evil.txt", b"pwned")])
        with self.assertRaises(SystemExit):
            self.tool.safe_extract(path, self.dest)
        self.assertFalse(os.path.exists(os.path.join(self.root, "evil.txt")))
        self.assertFalse(os.path.exists(os.path.join(self.dest, "ok", "file.txt")), "검사 전에 일부를 풀면 안 된다")

    def test_symlink_member_is_rejected(self):
        path = self._make_tar([("m/link", ("link", "/etc/passwd"))])
        with self.assertRaises(SystemExit):
            self.tool.safe_extract(path, self.dest)

    def test_every_asset_entry_is_complete(self):
        for a in self.tool.ASSETS:
            self.assertTrue(a["url"].startswith("https://"), a["name"])
            self.assertTrue(a["check"] and a["min_mb"] > 0, a["name"])


class TestMicEvalTool(unittest.TestCase):
    """tools/stt_mic_eval.py: 순수 함수는 단위 테스트, 실행 경로는 WAV 파일로 실제 실행(마이크 불필요)."""
    TOOL = os.path.join(BASE_DIR, "tools", "stt_mic_eval.py")

    @classmethod
    def setUpClass(cls):
        import importlib.util
        spec = importlib.util.spec_from_file_location("stt_mic_eval", cls.TOOL)
        cls.tool = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.tool)

    def test_cer_matches_reference_implementation(self):
        for ref, hyp in (("아메리카노 한 잔 주세요", "아메리카노 한잔 주세요"), ("카페라떼", "카페라테"), ("가나다", ""), ("", ""), ("", "a")):
            with self.subTest(ref=ref, hyp=hyp):
                self.assertAlmostEqual(self.tool.cer(ref, hyp), cer(ref, hyp))
        self.assertEqual(self.tool.cer("가나다", ""), 1.0)

    def test_summarize_values_and_empty_case(self):
        rows = [
            {"ref": "a", "text": "a", "cer": 0.0, "latency_ms": 100.0, "speech_detected": True},
            {"ref": "b", "text": "", "cer": 1.0, "latency_ms": 300.0, "speech_detected": False},
        ]
        s = self.tool.summarize(rows)
        self.assertEqual((s["n"], s["avg_cer"], s["exact_rate"], s["no_speech_rate"], s["avg_latency_ms"]),
                         (2, 0.5, 50.0, 50.0, 200.0))
        self.assertIsNone(self.tool.summarize([])["avg_cer"])        # 측정 없음을 가짜 100% 로 속이지 않는다

    @unittest.skipUnless(MODELS_READY and os.path.exists(KO_WAV), SKIP_MSG)
    def test_wav_mode_runs_end_to_end_with_real_model(self):
        import subprocess
        r = subprocess.run([sys.executable, self.TOOL, "--wav", KO_WAV, "--ref", KO_REF],
                           capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=120)
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        self.assertIn("평균 CER", r.stdout)
        self.assertIn("측정 1문장", r.stdout)

    def test_list_devices_does_not_crash(self):
        import subprocess
        r = subprocess.run([sys.executable, self.TOOL, "--list-devices"],
                           capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=60)
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)


if __name__ == "__main__":
    result = unittest.main(exit=False, verbosity=2 if "-v" in sys.argv else 1)
    skipped = len(result.result.skipped)
    print("\n" + "=" * 70)
    print(f" 실행 {result.result.testsRun}개 | 실패 {len(result.result.failures)} | 오류 {len(result.result.errors)} | 건너뜀 {skipped}")
    if skipped:
        print(" 건너뛴 테스트가 있습니다 (모델 미설치 가능):", {r[1] for r in result.result.skipped})
    print("=" * 70)
    sys.exit(0 if result.result.wasSuccessful() else 1)
