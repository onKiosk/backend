# -*- coding: utf-8 -*-
"""
OnKiosk Backend - Part 3: On-Device SQLite Database & CRUD Module (database.py)
-------------------------------------------------------------------------------
담당: 백엔드 파트 3 (김민재)

테이블 4종 + 카운터 1종
  - menus           : 메뉴 25종 / 가격 / 실시간 품절 (프론트 menu.ts 와 동일한 값이 시드)
  - orders          : 주문 내역 (일별 주문번호 A001, A002 ... 는 order_counters 로 원자적 발급)
  - order_counters  : 날짜별 마지막 주문 순번
  - stamps          : 전화번호 HMAC 키 기준 스탬프 적립 (원본 번호는 저장하지 않음, 비밀키는 DB 와 분리)
  - inference_logs  : 발화 / 예측 Action / 스키마 검증 / STT·LLM 지연시간

설계 원칙
  * 모든 함수는 호출마다 커넥션을 열고 반드시 close 한다 (Windows 파일 잠금/누수 방지).
  * 쓰기는 트랜잭션으로 묶고, 주문번호 발급은 BEGIN IMMEDIATE 로 동시 요청에도 중복되지 않게 한다.
  * 잘못된 입력은 조용히 저장하지 않고 InvalidOrderError 등으로 명확히 알린다 -> 파트 2 에서 폴백 처리.
  * DB 위치는 환경변수 ONKIOSK_DB_PATH 로 바꿀 수 있다 (테스트/배포용).
"""

import os
import re
import hmac
import json
import math
import sqlite3
import hashlib
import threading
from contextlib import contextmanager
from datetime import datetime
from typing import Any, Dict, Iterator, List, Optional

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DB_PATH = os.path.join(BASE_DIR, "onkiosk.db")
SCHEMA_VERSION = 2


class KioskDBError(Exception):
    """온키오스크 DB 계층의 공통 예외."""


class InvalidOrderError(KioskDBError, ValueError):
    """주문 내용이 규칙에 맞지 않을 때."""


class SoldOutError(InvalidOrderError):
    """품절 메뉴가 주문에 포함되었을 때."""


class InvalidStampError(KioskDBError, ValueError):
    """스탬프 적립 입력(전화번호/개수)이 올바르지 않을 때."""


def get_db_path() -> str:
    return os.environ.get("ONKIOSK_DB_PATH") or DEFAULT_DB_PATH


# ==============================================================================
# 프론트엔드(kiosk-one-dusky.vercel.app) menu.ts 와 동일한 메뉴 25종
#   (menu_id 는 데이터셋 '메뉴_정책' 시트 기준, 가격/품절/ICE전용은 프론트 기준)
# ==============================================================================
CATEGORY_LABELS = {
    "coffee": "커피", "latte": "라떼", "tea": "차", "ade": "에이드", "dessert": "디저트",
}

# (menu_id, category, name, price, initial_sold_out, ice_only)
_MENU_SEED = [
    ("CF_AMERICANO",         "coffee",  "아메리카노",      4500, False, False),
    ("CF_CAFE_LATTE",        "coffee",  "카페라떼",        5000, False, False),
    ("CF_VANILLA_LATTE",     "coffee",  "바닐라라떼",      5500, False, False),
    ("CF_CAPPUCCINO",        "coffee",  "카푸치노",        5000, False, False),
    ("CF_CAFE_MOCHA",        "coffee",  "카페모카",        5500, False, False),
    ("CF_COLD_BREW",         "coffee",  "콜드브루",        5000, True,  True),
    ("NC_CHOCOLATE_LATTE",   "latte",   "초코라떼",        5000, False, False),
    ("NC_GREEN_TEA_LATTE",   "latte",   "녹차라떼",        5500, False, False),
    ("NC_SWEET_POTATO_LATTE", "latte",  "고구마라떼",      5500, False, False),
    ("NC_STRAWBERRY_LATTE",  "latte",   "딸기라떼",        6000, False, False),
    ("TEA_YUJA",             "tea",     "유자차",          5000, False, False),
    ("TEA_JUJUBE",           "tea",     "대추차",          5500, False, False),
    ("TEA_CHAMOMILE",        "tea",     "캐모마일",        4500, False, False),
    ("TEA_EARL_GREY",        "tea",     "얼그레이",        4500, False, False),
    ("TEA_PEACH_ICED",       "tea",     "복숭아 아이스티", 4500, False, True),
    ("ADE_GRAPEFRUIT",       "ade",     "자몽에이드",      5500, False, True),
    ("ADE_LEMON",            "ade",     "레몬에이드",      5500, False, True),
    ("ADE_GREEN_GRAPE",      "ade",     "청포도에이드",    5500, False, True),
    ("DS_CHEESECAKE",        "dessert", "치즈케이크",      6000, False, False),
    ("DS_TIRAMISU",          "dessert", "티라미수",        6500, False, False),
    ("DS_CHOCO_COOKIE",      "dessert", "초코쿠키",        3000, False, False),
    ("DS_BAGEL",             "dessert", "베이글",          3500, False, False),
    ("DS_CROISSANT",         "dessert", "플레인크루아상",  4000, False, False),
    ("DS_MACARON",           "dessert", "마카롱",          2500, True,  False),
    ("DS_SALT_BREAD",        "dessert", "소금빵",          3500, False, False),
]

# 프론트 menu.ts 의 음성 별칭 (별칭 -> 정식 메뉴명)
MENU_ALIASES = {
    "아아": "아메리카노",
    "크루아상": "플레인크루아상",
    "아이스티": "복숭아 아이스티",
    "케이크": "치즈케이크",
    "쿠키": "초코쿠키",
    "라떼": "카페라떼",
}

# 프론트 menu.ts 의 추가 옵션 가격
EXTRA_OPTIONS = {
    "shot":  {"label": "샷 추가",   "price": 500},
    "syrup": {"label": "시럽 추가", "price": 300},
    "light": {"label": "연하게",    "price": 0},
    "milk":  {"label": "우유 변경", "price": 500},
}

# 데이터셋(v5.5) '메뉴_정책' 시트 기준 메뉴별 허용 추가 옵션 (shot / syrup / light / milk)
#   * 커피(아메리카노): 샷·시럽·연하게 가능, 우유변경 불가
#   * 커피(라떼·바닐라라떼·카푸치노·카페모카): 샷·시럽·연하게·우유변경 모두 가능
#   * 커피(콜드브루): ICE 고정, 샷·시럽 가능, 연하게·우유변경 불가
#   * 논커피 라떼(초코·녹차·고구마·딸기라떼): 시럽·우유변경 가능, 샷·연하게 불가
#   * 차(유자·대추·캐모마일·얼그레이): 샷·시럽·연하게·우유변경 모두 불가 (온도·사이즈·얼음량만 가능)
#   * 차(복숭아 아이스티) 및 에이드 3종: ICE 고정, 시럽 가능, 샷·연하게·우유변경 불가
#   * 디저트 7종: 모든 옵션 불가
MENU_OPTION_POLICY: Dict[str, Dict[str, bool]] = {
    "아메리카노":      {"shot": True,  "syrup": True,  "light": True,  "milk": False},
    "카페라떼":        {"shot": True,  "syrup": True,  "light": True,  "milk": True},
    "바닐라라떼":      {"shot": True,  "syrup": True,  "light": True,  "milk": True},
    "카푸치노":        {"shot": True,  "syrup": True,  "light": True,  "milk": True},
    "카페모카":        {"shot": True,  "syrup": True,  "light": True,  "milk": True},
    "콜드브루":        {"shot": True,  "syrup": True,  "light": False, "milk": False},
    "초코라떼":        {"shot": False, "syrup": True,  "light": False, "milk": True},
    "녹차라떼":        {"shot": False, "syrup": True,  "light": False, "milk": True},
    "고구마라떼":      {"shot": False, "syrup": True,  "light": False, "milk": True},
    "딸기라떼":        {"shot": False, "syrup": True,  "light": False, "milk": True},
    "유자차":          {"shot": False, "syrup": False, "light": False, "milk": False},
    "대추차":          {"shot": False, "syrup": False, "light": False, "milk": False},
    "캐모마일":        {"shot": False, "syrup": False, "light": False, "milk": False},
    "얼그레이":        {"shot": False, "syrup": False, "light": False, "milk": False},
    "복숭아 아이스티": {"shot": False, "syrup": True,  "light": False, "milk": False},
    "자몽에이드":      {"shot": False, "syrup": True,  "light": False, "milk": False},
    "레몬에이드":      {"shot": False, "syrup": True,  "light": False, "milk": False},
    "청포도에이드":    {"shot": False, "syrup": True,  "light": False, "milk": False},
    "치즈케이크":      {"shot": False, "syrup": False, "light": False, "milk": False},
    "티라미수":        {"shot": False, "syrup": False, "light": False, "milk": False},
    "초코쿠키":        {"shot": False, "syrup": False, "light": False, "milk": False},
    "베이글":          {"shot": False, "syrup": False, "light": False, "milk": False},
    "플레인크루아상":  {"shot": False, "syrup": False, "light": False, "milk": False},
    "마카롱":          {"shot": False, "syrup": False, "light": False, "milk": False},
    "소금빵":          {"shot": False, "syrup": False, "light": False, "milk": False},
}

VALID_PAYMENT_METHODS = {"CARD", "SAMSUNG_PAY", "KAKAO_PAY", "NAVER_PAY", "APPLE_PAY", "CASH"}
_PAYMENT_ALIASES = {
    "SAMSUNG": "SAMSUNG_PAY", "KAKAO": "KAKAO_PAY", "NAVER": "NAVER_PAY", "APPLE": "APPLE_PAY",
}
VALID_PICKUP_TYPES = {"DINE_IN", "TAKEOUT", "VIBRATION_BELL"}


# ==============================================================================
# 커넥션 / 트랜잭션
# ==============================================================================
@contextmanager
def connect(db_path: Optional[str] = None, immediate: bool = False) -> Iterator[sqlite3.Connection]:
    """
    트랜잭션이 걸린 커넥션을 빌려주고, 정상 종료 시 COMMIT / 예외 시 ROLLBACK 한 뒤 항상 close 한다.
    immediate=True 면 BEGIN IMMEDIATE 로 쓰기 잠금을 먼저 잡아 동시 쓰기 경쟁을 막는다.
    """
    conn = sqlite3.connect(db_path or get_db_path(), timeout=10, isolation_level=None)
    conn.row_factory = sqlite3.Row
    try:
        conn.execute("PRAGMA busy_timeout = 10000")
        conn.execute("PRAGMA foreign_keys = ON")
        conn.execute("BEGIN IMMEDIATE" if immediate else "BEGIN")
        try:
            yield conn
            conn.execute("COMMIT")
        except BaseException:
            if conn.in_transaction:
                conn.execute("ROLLBACK")
            raise
    finally:
        conn.close()


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def init_db(db_path: Optional[str] = None) -> None:
    """DB 파일과 테이블을 만들고, menus 가 비어 있으면 25종 메뉴를 적재한다. (여러 번 호출해도 안전)"""
    path = db_path or get_db_path()

    # WAL 모드는 DB 파일에 영구 저장되며 트랜잭션 밖에서 설정해야 한다. (읽기/쓰기 동시성 향상)
    raw = sqlite3.connect(path, timeout=10)
    try:
        raw.execute("PRAGMA journal_mode = WAL")
    finally:
        raw.close()

    with connect(path, immediate=True) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS menus (
                menu_id        TEXT PRIMARY KEY,
                category       TEXT NOT NULL,
                category_label TEXT NOT NULL,
                name           TEXT NOT NULL UNIQUE,
                price          INTEGER NOT NULL CHECK (price >= 0),
                is_ice_only    INTEGER NOT NULL DEFAULT 0,
                is_dessert     INTEGER NOT NULL DEFAULT 0,
                aliases_json   TEXT NOT NULL DEFAULT '[]',
                is_sold_out    INTEGER NOT NULL DEFAULT 0,
                updated_at     TEXT NOT NULL
            )""")
        conn.execute("""
            CREATE TABLE IF NOT EXISTS order_counters (
                order_date TEXT PRIMARY KEY,
                last_seq   INTEGER NOT NULL
            )""")
        conn.execute("""
            CREATE TABLE IF NOT EXISTS orders (
                id             INTEGER PRIMARY KEY AUTOINCREMENT,
                order_date     TEXT NOT NULL,
                order_number   TEXT NOT NULL,
                items_json     TEXT NOT NULL,
                total_price    INTEGER NOT NULL CHECK (total_price >= 0),
                payment_method TEXT NOT NULL,
                pickup_type    TEXT NOT NULL,
                print_receipt  INTEGER NOT NULL DEFAULT 1,
                created_at     TEXT NOT NULL,
                UNIQUE (order_date, order_number)
            )""")
        conn.execute("""
            CREATE TABLE IF NOT EXISTS stamps (
                phone_hash  TEXT PRIMARY KEY,
                phone_hint  TEXT NOT NULL,
                stamp_count INTEGER NOT NULL CHECK (stamp_count >= 0),
                updated_at  TEXT NOT NULL
            )""")
        conn.execute("""
            CREATE TABLE IF NOT EXISTS inference_logs (
                id                 INTEGER PRIMARY KEY AUTOINCREMENT,
                utterance          TEXT NOT NULL,
                screen_context     TEXT,
                predicted_action   TEXT NOT NULL,
                response_json      TEXT NOT NULL,
                is_valid_schema    INTEGER NOT NULL DEFAULT 1,
                fallback_triggered INTEGER NOT NULL DEFAULT 0,
                stt_latency_ms     REAL NOT NULL DEFAULT 0 CHECK (stt_latency_ms >= 0),
                llm_latency_ms     REAL NOT NULL DEFAULT 0 CHECK (llm_latency_ms >= 0),
                total_latency_ms   REAL NOT NULL DEFAULT 0 CHECK (total_latency_ms >= 0),
                created_at         TEXT NOT NULL
            )""")

        cols = {r["name"] for r in conn.execute("PRAGMA table_info(menus)")}
        if "category" not in cols:
            raise RuntimeError(
                "구버전 스키마의 onkiosk.db 가 발견되었습니다. 파일을 삭제한 뒤 다시 실행해 주세요: " + path
            )

        if conn.execute("SELECT COUNT(*) FROM menus").fetchone()[0] == 0:
            now = _now()
            alias_by_menu: Dict[str, List[str]] = {}
            for alias, target in MENU_ALIASES.items():
                alias_by_menu.setdefault(target, []).append(alias)
            for menu_id, category, name, price, sold_out, ice_only in _MENU_SEED:
                conn.execute(
                    """INSERT INTO menus (menu_id, category, category_label, name, price,
                                          is_ice_only, is_dessert, aliases_json, is_sold_out, updated_at)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (menu_id, category, CATEGORY_LABELS[category], name, price,
                     1 if (ice_only or category == "ade") else 0,
                     1 if category == "dessert" else 0,
                     json.dumps(alias_by_menu.get(name, []), ensure_ascii=False),
                     1 if sold_out else 0, now),
                )
        else:
            # 기존 DB 파일이 남아 있더라도 콜드브루는 데이터셋 메뉴_정책 기준 ICE 전용(is_ice_only=1)으로 보정
            conn.execute("UPDATE menus SET is_ice_only = 1 WHERE menu_id = 'CF_COLD_BREW' AND is_ice_only = 0")
        conn.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")


# ==============================================================================
# [CRUD 1] 메뉴 · 품절
# ==============================================================================
_NO_EXTRA_OPTIONS = {"shot": False, "syrup": False, "light": False, "milk": False}


def get_menu_option_policy(menu_name: str) -> Dict[str, bool]:
    """메뉴명의 허용 추가 옵션(shot/syrup/light/milk) 사전을 반환한다."""
    return dict(MENU_OPTION_POLICY.get(menu_name, _NO_EXTRA_OPTIONS))


def _menu_row_to_dict(r: sqlite3.Row) -> Dict[str, Any]:
    if r["is_dessert"]:
        temp_policy = "NONE"
    elif r["is_ice_only"]:
        temp_policy = "ICE_ONLY"
    else:
        temp_policy = "HOT/ICE"
    return {
        "menu_id": r["menu_id"],
        "category": r["category"],
        "category_label": r["category_label"],
        "name": r["name"],
        "price": r["price"],
        "is_ice_only": bool(r["is_ice_only"]),
        "is_dessert": bool(r["is_dessert"]),
        "temp_policy": temp_policy,
        "allowed_options": get_menu_option_policy(r["name"]),
        "aliases": json.loads(r["aliases_json"]),
        "is_sold_out": bool(r["is_sold_out"]),
        "updated_at": r["updated_at"],
    }


def get_all_menus(category: Optional[str] = None, db_path: Optional[str] = None) -> List[Dict[str, Any]]:
    """전체 메뉴 또는 카테고리(coffee/latte/tea/ade/dessert)별 메뉴 + 실시간 품절 상태."""
    with connect(db_path) as conn:
        if category:
            rows = conn.execute("SELECT * FROM menus WHERE category = ? ORDER BY rowid", (category,)).fetchall()
        else:
            rows = conn.execute("SELECT * FROM menus ORDER BY rowid").fetchall()
    return [_menu_row_to_dict(r) for r in rows]


def get_menu(identifier: str, db_path: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """menu_id 또는 메뉴명으로 1건 조회. 없으면 None."""
    with connect(db_path) as conn:
        r = conn.execute("SELECT * FROM menus WHERE menu_id = ? OR name = ?", (identifier, identifier)).fetchone()
    return _menu_row_to_dict(r) if r else None


def get_sold_out_menu_names(db_path: Optional[str] = None) -> List[str]:
    """현재 품절인 메뉴 이름 목록 (AI Context 의 sold_out 필드용)."""
    with connect(db_path) as conn:
        return [r["name"] for r in conn.execute("SELECT name FROM menus WHERE is_sold_out = 1 ORDER BY rowid")]


def update_menu_sold_out(identifier: str, is_sold_out: bool, db_path: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """menu_id 또는 메뉴명으로 품절 상태 변경. 존재하지 않는 메뉴면 None."""
    with connect(db_path, immediate=True) as conn:
        cur = conn.execute(
            "UPDATE menus SET is_sold_out = ?, updated_at = ? WHERE menu_id = ? OR name = ?",
            (1 if is_sold_out else 0, _now(), identifier, identifier),
        )
        if cur.rowcount == 0:
            return None
        r = conn.execute("SELECT * FROM menus WHERE menu_id = ? OR name = ?", (identifier, identifier)).fetchone()
    return _menu_row_to_dict(r)


def get_alias_map() -> Dict[str, str]:
    """
    프론트와 동일한 음성 별칭 -> 정식 메뉴명 사전.
    주의: create_order / get_menu 는 정식 메뉴명(또는 menu_id)만 받는다. 별칭("아아")을 정식명으로 바꾸는 일은
    호출자(파트 2 의 Action 검증/폴백 단계)가 이 사전으로 먼저 수행해야 한다.
    """
    return dict(MENU_ALIASES)


def get_extra_options() -> Dict[str, Dict[str, Any]]:
    """샷/시럽/연하게/우유변경 추가 옵션 가격표."""
    return {k: dict(v) for k, v in EXTRA_OPTIONS.items()}


# ==============================================================================
# [CRUD 2] 주문 · 스탬프
# ==============================================================================
MAX_ITEM_QUANTITY = 99          # 한 항목 최대 수량 (오입력/오인식 방어)
MAX_ORDER_LINES = 30            # 한 주문의 최대 항목 수
LARGE_SIZE_SURCHARGE = 500      # 프론트 menu.ts: LARGE 는 +500원
_VALID_TEMPERATURES = {"HOT", "ICE"}
_VALID_SIZES = {"REGULAR", "LARGE"}
_VALID_ICE_LEVELS = {"NORMAL", "LESS", "NONE", "MORE"}
_VALID_STRENGTHS = {"NORMAL", "LIGHT"}
_VALID_MILK_TYPES = {"REGULAR", "SOY", "OAT"}
_VALID_OPTION_KEYS = {"shot", "syrup", "light", "strength", "milk", "ice"}


def _option_flag(options: Dict[str, Any], key: str, line: Any) -> bool:
    """options 의 shot/syrup/light 값(0/1/개수/bool)을 '선택 여부'로 바꾼다. 프론트는 개수가 아닌 선택 여부로 과금한다."""
    value = options.get(key, 0)
    if value is None:
        return False
    if isinstance(value, bool):
        return value
    if isinstance(value, int) and value >= 0:
        return value > 0
    raise InvalidOrderError(f"옵션 {key} 는 0 이상의 정수 또는 bool 이어야 합니다: {line!r}")


def _price_line(row: sqlite3.Row, item: Dict[str, Any]) -> int:
    """
    주문 항목 1건을 검증하고 단가를 계산한다. (프론트 menu.ts 의 v() 및 데이터셋 '메뉴_정책' 기준)
      단가 = 메뉴 가격 + (LARGE 이면 500) + 선택된 추가 옵션 가격(샷 500 / 시럽 300 / 연하게 0 / 우유 변경 500)
    - 디저트에는 온도/사이즈/옵션을 붙일 수 없고, 아이스 전용 메뉴는 HOT 으로 주문할 수 없다.
    - 메뉴별로 허용되지 않는 추가 옵션(예: 유자차에 시럽, 아메리카노에 우유, 콜드브루에 연하게)은 거부한다.
    - 따뜻한(HOT) 음료에는 얼음(ice) 옵션을 붙일 수 없다.
    - 값이 있을 때만 검사한다(온도/사이즈 미선택은 허용: 선택 유도는 화면/대화 흐름의 책임).
    """
    temperature = item.get("temperature")
    size = item.get("size")
    options = item.get("options")
    if options is not None and not isinstance(options, dict):
        raise InvalidOrderError(f"options 는 객체여야 합니다: {item!r}")
    options = options or {}

    if row["is_dessert"]:
        if temperature is not None or size is not None or options:
            raise InvalidOrderError(f"디저트에는 온도/사이즈/옵션을 지정할 수 없습니다: {item!r}")
        return int(row["price"])

    if temperature is not None:
        if str(temperature).upper() not in _VALID_TEMPERATURES:
            raise InvalidOrderError(f"온도는 HOT/ICE 중 하나여야 합니다: {item!r}")
        if row["is_ice_only"] and str(temperature).upper() != "ICE":
            raise InvalidOrderError(f"{row['name']} 는 아이스로만 주문할 수 있습니다: {item!r}")
    if size is not None and str(size).upper() not in _VALID_SIZES:
        raise InvalidOrderError(f"사이즈는 REGULAR/LARGE 중 하나여야 합니다: {item!r}")

    unknown_keys = set(options) - _VALID_OPTION_KEYS
    if unknown_keys:
        raise InvalidOrderError(f"지원하지 않는 옵션 키입니다({sorted(unknown_keys)}): {item!r}")

    ice = options.get("ice")
    if ice is not None:
        if not isinstance(ice, str) or ice.strip().upper() not in _VALID_ICE_LEVELS:
            raise InvalidOrderError(f"옵션 ice 는 NORMAL/LESS/NONE/MORE 중 하나여야 합니다: {item!r}")
        if temperature is not None and str(temperature).upper() == "HOT":
            raise InvalidOrderError(f"따뜻한(HOT) 음료에는 얼음(ice) 옵션을 지정할 수 없습니다: {item!r}")

    policy = get_menu_option_policy(row["name"])
    unit = int(row["price"])
    if size is not None and str(size).upper() == "LARGE":
        unit += LARGE_SIZE_SURCHARGE
    if _option_flag(options, "shot", item):
        if not policy["shot"]:
            raise InvalidOrderError(f"{row['name']} 메뉴에는 샷 추가 옵션을 사용할 수 없습니다: {item!r}")
        unit += EXTRA_OPTIONS["shot"]["price"]
    if _option_flag(options, "syrup", item):
        if not policy["syrup"]:
            raise InvalidOrderError(f"{row['name']} 메뉴에는 시럽 추가 옵션을 사용할 수 없습니다: {item!r}")
        unit += EXTRA_OPTIONS["syrup"]["price"]

    has_light = _option_flag(options, "light", item)
    strength = options.get("strength")
    if strength is not None:
        if not isinstance(strength, str) or strength.strip().upper() not in _VALID_STRENGTHS:
            raise InvalidOrderError(f"옵션 strength 는 NORMAL/LIGHT 중 하나여야 합니다: {item!r}")
        if strength.strip().upper() == "LIGHT":
            has_light = True
    if has_light:
        if not policy["light"]:
            raise InvalidOrderError(f"{row['name']} 메뉴에는 연하게 옵션을 사용할 수 없습니다: {item!r}")
        unit += EXTRA_OPTIONS["light"]["price"]                  # 연하게: 무료(+0)

    milk = options.get("milk")
    if milk is not None and milk is not False:
        if isinstance(milk, bool):
            if not policy["milk"]:
                raise InvalidOrderError(f"{row['name']} 메뉴에는 우유 변경 옵션을 사용할 수 없습니다: {item!r}")
            unit += EXTRA_OPTIONS["milk"]["price"]
        elif isinstance(milk, str):
            milk_norm = milk.strip().upper()
            if milk_norm not in _VALID_MILK_TYPES:
                raise InvalidOrderError(f"옵션 milk 는 REGULAR/SOY/OAT 중 하나여야 합니다: {item!r}")
            if not policy["milk"]:
                raise InvalidOrderError(f"{row['name']} 메뉴에는 우유 변경 옵션을 사용할 수 없습니다: {item!r}")
            if milk_norm in ("SOY", "OAT"):                      # 우유 변경(두유 SOY / 오트 OAT)만 +500
                unit += EXTRA_OPTIONS["milk"]["price"]
        else:
            raise InvalidOrderError(f"옵션 milk 는 문자열 또는 bool 이어야 합니다: {item!r}")
    return unit


def _price_items(conn: sqlite3.Connection, items: Any) -> int:
    """항목 전체를 검증하고 서버가 계산한 총액을 돌려준다. 품절이면 SoldOutError."""
    if not isinstance(items, list) or not items:
        raise InvalidOrderError("주문 항목(items)이 비어 있습니다.")
    if len(items) > MAX_ORDER_LINES:
        raise InvalidOrderError(f"한 주문의 항목은 최대 {MAX_ORDER_LINES}개입니다: {len(items)}개")
    total = 0
    for it in items:
        if not isinstance(it, dict) or not it.get("menu"):
            raise InvalidOrderError(f"주문 항목 형식 오류(menu 필요): {it!r}")
        qty = it.get("quantity", 1)
        if isinstance(qty, bool) or not isinstance(qty, int) or not 1 <= qty <= MAX_ITEM_QUANTITY:
            raise InvalidOrderError(f"수량은 1~{MAX_ITEM_QUANTITY} 사이의 정수여야 합니다: {it!r}")
        row = conn.execute(
            "SELECT name, price, is_ice_only, is_dessert, is_sold_out FROM menus WHERE name = ?", (it["menu"],)).fetchone()
        if row is None:
            raise InvalidOrderError(f"존재하지 않는 메뉴: {it['menu']!r}")
        if row["is_sold_out"]:
            raise SoldOutError(f"품절된 메뉴입니다: {row['name']}")
        total += _price_line(row, it) * qty
    return total


def calculate_order_total(items: List[Dict[str, Any]], db_path: Optional[str] = None) -> int:
    """주문 항목의 총액을 서버(DB 가격표) 기준으로 계산한다. 잘못된 항목/품절이면 예외."""
    with connect(db_path) as conn:
        return _price_items(conn, items)


def create_order(
    items: List[Dict[str, Any]],
    total_price: Optional[int] = None,
    payment_method: str = "CARD",
    pickup_type: str = "DINE_IN",
    print_receipt: bool = True,
    db_path: Optional[str] = None,
) -> Dict[str, Any]:
    """
    주문을 저장하고 날짜별 주문번호(A001, A002 ...)를 발급한다.
    - items 는 비어 있으면 안 되고, 각 항목의 menu 는 DB 에 존재하는 정식 메뉴명이어야 한다. (별칭 -> 정식명 변환은 호출자 몫)
    - 총액은 서버가 DB 가격으로 다시 계산한다. total_price 를 넘기면 계산값과 같아야 하고(다르면 InvalidOrderError),
      None 이면 계산값을 사용한다.
    - 수량은 1~99, 아이스 전용 메뉴는 HOT 불가, 디저트에는 온도/사이즈/옵션 불가.
    - 품절 메뉴가 포함되면 SoldOutError.
    - 번호 발급은 같은 트랜잭션 안에서 원자적으로 이루어져 동시 주문에도 중복되지 않는다.
    """
    if not isinstance(items, list) or not items:
        raise InvalidOrderError("주문 항목(items)이 비어 있습니다.")
    if total_price is not None and (
            isinstance(total_price, bool) or not isinstance(total_price, int) or total_price < 0):
        raise InvalidOrderError(f"total_price 는 0 이상의 정수여야 합니다: {total_price!r}")

    method = str(payment_method).strip().upper()
    method = _PAYMENT_ALIASES.get(method, method)
    if method not in VALID_PAYMENT_METHODS:
        raise InvalidOrderError(f"지원하지 않는 결제수단: {payment_method!r}")
    pickup = str(pickup_type).strip().upper()
    if pickup not in VALID_PICKUP_TYPES:
        raise InvalidOrderError(f"지원하지 않는 수령 방식: {pickup_type!r}")

    now = datetime.now()
    order_date = now.strftime("%Y-%m-%d")

    with connect(db_path, immediate=True) as conn:
        computed = _price_items(conn, items)
        if total_price is not None and total_price != computed:
            raise InvalidOrderError(f"총액이 일치하지 않습니다: 요청 {total_price}원 / 서버 계산 {computed}원")
        total_price = computed

        conn.execute(
            """INSERT INTO order_counters (order_date, last_seq) VALUES (?, 1)
               ON CONFLICT(order_date) DO UPDATE SET last_seq = last_seq + 1""",
            (order_date,),
        )
        seq = conn.execute("SELECT last_seq FROM order_counters WHERE order_date = ?", (order_date,)).fetchone()[0]
        order_number = f"A{seq:03d}"
        created_at = now.isoformat(timespec="seconds")
        cur = conn.execute(
            """INSERT INTO orders (order_date, order_number, items_json, total_price,
                                   payment_method, pickup_type, print_receipt, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (order_date, order_number, json.dumps(items, ensure_ascii=False), total_price,
             method, pickup, 1 if print_receipt else 0, created_at),
        )
        order_id = cur.lastrowid

    return {
        "order_id": order_id, "order_number": order_number, "order_date": order_date,
        "items": items, "total_price": total_price, "payment_method": method,
        "pickup_type": pickup, "print_receipt": bool(print_receipt), "created_at": created_at,
    }


def get_order(order_id: int, db_path: Optional[str] = None) -> Optional[Dict[str, Any]]:
    with connect(db_path) as conn:
        r = conn.execute("SELECT * FROM orders WHERE id = ?", (order_id,)).fetchone()
    if r is None:
        return None
    return {
        "order_id": r["id"], "order_number": r["order_number"], "order_date": r["order_date"],
        "items": json.loads(r["items_json"]), "total_price": r["total_price"],
        "payment_method": r["payment_method"], "pickup_type": r["pickup_type"],
        "print_receipt": bool(r["print_receipt"]), "created_at": r["created_at"],
    }


def _normalize_phone(phone_number: str) -> str:
    digits = re.sub(r"\D", "", str(phone_number or ""))
    if len(digits) not in (4, 11):
        raise InvalidStampError("전화번호는 뒤 4자리(4자리) 또는 전체 번호(11자리) 숫자여야 합니다.")
    return digits


# ------------------------------------------------------------------------------
# 스탬프 키: 전화번호는 경우의 수가 ~10^8 뿐이라 '그냥 SHA-256' 은 DB 가 유출되면 수 초 만에 역산된다.
# 그래서 DB 와 분리된 서버 비밀키로 HMAC-SHA256 을 쓴다.
#   1순위: 환경변수 ONKIOSK_STAMP_SECRET
#   2순위: DB 파일 옆의 .onkiosk_stamp_key (없으면 최초 사용 시 무작위 32바이트 생성, .gitignore 대상)
# 주의: 비밀키를 잃어버리면 기존 스탬프 기록과 새 입력이 연결되지 않는다 -> DB 와 함께 백업할 것.
# ------------------------------------------------------------------------------
STAMP_KEY_FILENAME = ".onkiosk_stamp_key"
_stamp_key_lock = threading.Lock()


def _stamp_secret(db_path: Optional[str]) -> bytes:
    env = os.environ.get("ONKIOSK_STAMP_SECRET")
    if env:
        return env.encode("utf-8")
    key_path = os.path.join(os.path.dirname(os.path.abspath(db_path or get_db_path())), STAMP_KEY_FILENAME)
    with _stamp_key_lock:
        try:
            with open(key_path, "x", encoding="ascii") as f:   # 'x': 이미 있으면 덮어쓰지 않는다
                f.write(os.urandom(32).hex())
            try:
                os.chmod(key_path, 0o600)
            except OSError:
                pass
        except FileExistsError:
            pass
        with open(key_path, encoding="ascii") as f:
            secret = f.read().strip()
    if len(secret) < 32:
        raise KioskDBError(f"스탬프 비밀키 파일이 손상되었습니다(너무 짧음): {key_path}")
    return secret.encode("ascii")


def _phone_key(digits: str, db_path: Optional[str]) -> str:
    return hmac.new(_stamp_secret(db_path), digits.encode("utf-8"), hashlib.sha256).hexdigest()


def add_stamp(phone_number: str, count: int = 1, db_path: Optional[str] = None) -> Dict[str, Any]:
    """
    전화번호(또는 뒤 4자리) 기준으로 스탬프를 누적 적립한다.
    개인정보 보호: 원본 번호는 저장하지 않고 HMAC-SHA256(서버 비밀키) 값과 뒤 4자리(표시용)만 저장한다.
    한계(정책상 허용):
      * 뒤 4자리만 입력하면 서로 다른 고객의 번호가 같은 키로 합쳐질 수 있다.
      * 같은 고객이 전체 번호와 뒤 4자리를 각각 입력하면 서로 다른 기록으로 갈린다.
    """
    if isinstance(count, bool) or not isinstance(count, int) or count < 1:
        raise InvalidStampError(f"적립 개수는 1 이상의 정수여야 합니다: {count!r}")
    digits = _normalize_phone(phone_number)
    key = _phone_key(digits, db_path)
    hint = digits[-4:]
    now = _now()
    with connect(db_path, immediate=True) as conn:
        conn.execute(
            """INSERT INTO stamps (phone_hash, phone_hint, stamp_count, updated_at) VALUES (?, ?, ?, ?)
               ON CONFLICT(phone_hash) DO UPDATE
               SET stamp_count = stamp_count + excluded.stamp_count, updated_at = excluded.updated_at""",
            (key, hint, count, now),
        )
        total = conn.execute("SELECT stamp_count FROM stamps WHERE phone_hash = ?", (key,)).fetchone()[0]
    return {"phone_hint": hint, "added_stamps": count, "total_stamps": total, "updated_at": now}


def get_stamp_count(phone_number: str, db_path: Optional[str] = None) -> int:
    key = _phone_key(_normalize_phone(phone_number), db_path)
    with connect(db_path) as conn:
        r = conn.execute("SELECT stamp_count FROM stamps WHERE phone_hash = ?", (key,)).fetchone()
    return r["stamp_count"] if r else 0


# ==============================================================================
# [CRUD 3] AI 추론 로그 · 지연시간 통계
# ==============================================================================
LATENCY_TARGET_MS = 2500.0  # 중간발표 목표: 음성 입력 ~ 응답 2.0~2.5초 이내

# 프론트 9개 화면(ScreenKey) -> 데이터셋 6개 화면 컨텍스트(context.screen) 매핑
SCREEN_CONTEXT_MAP: Dict[str, str] = {
    "start":          "menu",
    "menu":           "menu",
    "option":         "menu",
    "cart":           "cart",
    "point":          "loyalty",
    "loyalty":        "loyalty",
    "pay":            "payment",
    "paying":         "payment",
    "payment":        "payment",
    "receipt":        "receipt_choice",
    "receipt_choice": "receipt_choice",
    "done":           "complete",
    "complete":       "complete",
}
VALID_SCREEN_CONTEXTS = {"menu", "cart", "loyalty", "payment", "receipt_choice", "complete"}


def normalize_screen_context(screen: Optional[str]) -> str:
    """
    프론트엔드의 9개 화면 키(start/menu/option/cart/point/pay/paying/receipt/done)와
    데이터셋의 6개 화면 컨텍스트(menu/cart/loyalty/payment/receipt_choice/complete)를
    데이터셋 기준 6개 값으로 정규화한다. 알 수 없는 화면 키면 ValueError 를 발생시킨다.
    """
    key = str(screen or "menu").strip().lower()
    if key not in SCREEN_CONTEXT_MAP:
        raise ValueError(f"지원하지 않는 screen_context 입니다: {screen!r}")
    return SCREEN_CONTEXT_MAP[key]


def log_inference(
    utterance: str,
    predicted_action: str,
    response_obj: Dict[str, Any],
    screen_context: str = "menu",
    is_valid_schema: bool = True,
    fallback_triggered: bool = False,
    stt_latency_ms: float = 0.0,
    llm_latency_ms: float = 0.0,
    db_path: Optional[str] = None,
) -> int:
    """발화 1건의 추론 결과와 단계별 지연시간(ms)을 기록하고 로그 id 를 반환한다."""
    if stt_latency_ms < 0 or llm_latency_ms < 0:
        raise ValueError("지연시간(ms)은 음수일 수 없습니다.")
    if not predicted_action:
        raise ValueError("predicted_action 이 비어 있습니다.")
    normalized_screen = normalize_screen_context(screen_context)
    total = round(stt_latency_ms + llm_latency_ms, 2)
    with connect(db_path, immediate=True) as conn:
        cur = conn.execute(
            """INSERT INTO inference_logs (utterance, screen_context, predicted_action, response_json,
                   is_valid_schema, fallback_triggered, stt_latency_ms, llm_latency_ms,
                   total_latency_ms, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (utterance or "", normalized_screen, predicted_action, json.dumps(response_obj, ensure_ascii=False),
             1 if is_valid_schema else 0, 1 if fallback_triggered else 0,
             round(stt_latency_ms, 2), round(llm_latency_ms, 2), total, _now()),
        )
        return cur.lastrowid


def get_latency_stats(db_path: Optional[str] = None) -> Dict[str, Any]:
    """
    누적 로그 기반 지연시간/성공률 통계. 로그가 없으면 비율 항목은 None(= 측정 없음)으로 돌려준다.
    within_target_rate 는 총 지연시간이 LATENCY_TARGET_MS(2.5초) 이하인 비율(%).
    """
    with connect(db_path) as conn:
        rows = conn.execute(
            "SELECT stt_latency_ms, llm_latency_ms, total_latency_ms, is_valid_schema, fallback_triggered "
            "FROM inference_logs"
        ).fetchall()

    n = len(rows)
    if n == 0:
        return {"total_calls": 0, "avg_stt_ms": None, "avg_llm_ms": None, "avg_total_ms": None,
                "p95_total_ms": None, "max_total_ms": None, "valid_schema_rate": None,
                "within_target_rate": None, "fallback_count": 0}

    totals = sorted(r["total_latency_ms"] for r in rows)
    p95 = totals[max(0, math.ceil(0.95 * n) - 1)]          # nearest-rank 방식 (1..100 -> 95)
    return {
        "total_calls": n,
        "avg_stt_ms": round(sum(r["stt_latency_ms"] for r in rows) / n, 1),
        "avg_llm_ms": round(sum(r["llm_latency_ms"] for r in rows) / n, 1),
        "avg_total_ms": round(sum(totals) / n, 1),
        "p95_total_ms": round(p95, 1),
        "max_total_ms": round(totals[-1], 1),
        "valid_schema_rate": round(100.0 * sum(r["is_valid_schema"] for r in rows) / n, 1),
        "within_target_rate": round(100.0 * sum(1 for t in totals if t <= LATENCY_TARGET_MS) / n, 1),
        "fallback_count": sum(r["fallback_triggered"] for r in rows),
    }
