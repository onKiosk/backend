"""Action 스키마 (데이터셋 `기능_스키마` 시트 기준) + API 요청/응답 모델.

- 18개 Action은 `action` 값으로 구분되는 Union(ACTION_ADAPTER)으로 검증한다.
- answer_question은 v5.5 합의안 기준(6 topic x 15 answer_code)이며, v5.5 엑셀 확인 후 필드명을 재검토한다.
"""
from typing import Annotated, Literal, Optional, Union

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, model_validator

Temperature = Literal["HOT", "ICE"]
Size = Literal["REGULAR", "LARGE"]
IceLevel = Literal["NORMAL", "LESS", "NONE", "MORE"]
Strength = Literal["NORMAL", "LIGHT"]
Milk = Literal["REGULAR", "SOY", "OAT"]


class Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


# ---------------------------------------------------------------- 공용 조각
class ItemOptions(Strict):
    ice: Optional[IceLevel] = None
    shot: Optional[int] = Field(None, ge=0, le=3)
    syrup: Optional[int] = Field(None, ge=0, le=3)
    strength: Optional[Strength] = None
    milk: Optional[Milk] = None


class CartItem(Strict):
    menu: str
    quantity: int = Field(ge=1, le=9)
    temperature: Optional[Temperature] = None
    size: Optional[Size] = None
    options: Optional[ItemOptions] = None


class CartChanges(Strict):
    quantity: Optional[int] = Field(None, ge=1, le=9)
    temperature: Optional[Temperature] = None
    size: Optional[Size] = None
    ice: Optional[IceLevel] = None
    shot: Optional[int] = Field(None, ge=0, le=3)
    syrup: Optional[int] = Field(None, ge=0, le=3)
    strength: Optional[Strength] = None
    milk: Optional[Milk] = None

    @model_validator(mode="after")
    def _not_empty(self):
        if not self.model_dump(exclude_none=True):
            raise ValueError("changes must not be empty")
        return self


# ---------------------------------------------------------------- Action 18종
class AddToCart(Strict):
    action: Literal["add_to_cart"]
    items: list[CartItem] = Field(min_length=1)


class AskClarification(Strict):
    action: Literal["ask_clarification"]
    field: Literal["temperature", "size"]
    item: str
    choices: list[str]

    @model_validator(mode="after")
    def _choices_match_field(self):
        expected = ["HOT", "ICE"] if self.field == "temperature" else ["REGULAR", "LARGE"]
        if self.choices != expected:
            raise ValueError(f"choices for {self.field} must be {expected}")
        return self


class RecommendMenu(Strict):
    action: Literal["recommend_menu"]
    items: list[str] = Field(min_length=1)
    confirmation_required: Literal[True]


class ShowCart(Strict):
    action: Literal["show_cart"]


class ModifyCart(Strict):
    action: Literal["modify_cart"]
    target: str
    changes: CartChanges


class RemoveFromCart(Strict):
    action: Literal["remove_from_cart"]
    target: str


class ResetOrder(Strict):
    action: Literal["reset_order"]


class ConfirmOrder(Strict):
    action: Literal["confirm_order"]
    summary_required: Literal[True]


class InformSoldOut(Strict):
    action: Literal["inform_sold_out"]
    item: str
    alternatives: list[str]


class ShowCategory(Strict):
    action: Literal["show_category"]
    category: Literal["COFFEE", "NON_COFFEE", "DESSERT"]


class StartPayment(Strict):
    action: Literal["start_payment"]
    method: Literal["CARD", "SAMSUNG_PAY", "KAKAO_PAY", "NAVER_PAY", "APPLE_PAY", "CASH"]


class OpenLoyaltyInput(Strict):
    action: Literal["open_loyalty_input"]
    input_mode: Literal["KEYPAD", "LAST4_KEYPAD"]


class OpenCouponList(Strict):
    action: Literal["open_coupon_list"]


class SetReceipt(Strict):
    action: Literal["set_receipt"]
    print: bool


class AnnounceOrder(Strict):
    action: Literal["announce_order"]
    order_number: str = Field(pattern=r"^[A-Z]\d{3}$")
    pickup: Literal["VIBRATION_BELL", "DINE_IN", "TAKEOUT"]


class SetAccessibility(Strict):
    action: Literal["set_accessibility"]
    mode: Literal["LOWER_TOUCH", "HIGH_CONTRAST", "VOICE_GUIDE"]


class CallStaff(Strict):
    action: Literal["call_staff"]
    reason: Literal[
        "USER_REQUESTED_ASSISTANCE", "VOICE_RECOGNITION_FAILED_TWICE", "KIOSK_MALFUNCTION"
    ]


class NoAction(Strict):
    action: Literal["no_action"]
    reason: Literal["OUT_OF_DOMAIN", "NO_COMMAND"]


# ---------------------------------------------------------------- v5.5 추가 Action
ANSWER_CODES: dict[str, tuple[str, ...]] = {
    "PAYMENT_METHOD": ("CASH_AVAILABLE", "CARD_AVAILABLE", "EASY_PAYMENT_AVAILABLE"),
    "MENU_OPTION": (
        "TEMPERATURE_OPTIONS_AVAILABLE",
        "SIZE_OPTIONS_AVAILABLE",
        "ITEM_OPTION_POLICY",
    ),
    "LOYALTY": (
        "PHONE_NUMBER_LOYALTY_AVAILABLE",
        "LOYALTY_INPUT_SECURITY",
        "COUPON_LIST_AVAILABLE",
    ),
    "RECEIPT": ("PAPER_RECEIPT_AVAILABLE", "RECEIPT_CHOICE_AFTER_PAYMENT"),
    "KIOSK_USAGE": ("VOICE_GUIDE_AVAILABLE", "STAFF_ASSISTANCE_AVAILABLE"),
    "ORDER_PICKUP": ("ORDER_NUMBER_ANNOUNCED", "VIBRATION_BELL_PICKUP"),
}


class AnswerQuestion(Strict):
    action: Literal["answer_question"]
    topic: Literal[
        "PAYMENT_METHOD", "MENU_OPTION", "LOYALTY", "RECEIPT", "KIOSK_USAGE", "ORDER_PICKUP"
    ]
    answer_code: Literal[
        "CASH_AVAILABLE",
        "CARD_AVAILABLE",
        "EASY_PAYMENT_AVAILABLE",
        "TEMPERATURE_OPTIONS_AVAILABLE",
        "SIZE_OPTIONS_AVAILABLE",
        "ITEM_OPTION_POLICY",
        "PHONE_NUMBER_LOYALTY_AVAILABLE",
        "LOYALTY_INPUT_SECURITY",
        "COUPON_LIST_AVAILABLE",
        "PAPER_RECEIPT_AVAILABLE",
        "RECEIPT_CHOICE_AFTER_PAYMENT",
        "VOICE_GUIDE_AVAILABLE",
        "STAFF_ASSISTANCE_AVAILABLE",
        "ORDER_NUMBER_ANNOUNCED",
        "VIBRATION_BELL_PICKUP",
    ]

    @model_validator(mode="after")
    def _code_matches_topic(self):
        if self.answer_code not in ANSWER_CODES[self.topic]:
            raise ValueError(f"{self.answer_code} is not allowed for topic {self.topic}")
        return self


Action = Annotated[
    Union[
        AddToCart,
        AskClarification,
        RecommendMenu,
        ShowCart,
        ModifyCart,
        RemoveFromCart,
        ResetOrder,
        ConfirmOrder,
        InformSoldOut,
        ShowCategory,
        StartPayment,
        OpenLoyaltyInput,
        OpenCouponList,
        SetReceipt,
        AnnounceOrder,
        SetAccessibility,
        CallStaff,
        NoAction,
        AnswerQuestion,
    ],
    Field(discriminator="action"),
]
ACTION_ADAPTER: TypeAdapter = TypeAdapter(Action)


# ---------------------------------------------------------------- API 요청/응답
class ActionContext(BaseModel):
    """프론트가 보내는 Context. 데이터셋 Context 키 + 프론트 화면 이름(9종)도 허용."""

    model_config = ConfigDict(extra="allow")

    screen: str
    cart: list[dict] = Field(default_factory=list)
    sold_out: Optional[list[str]] = None  # 없으면 DB(database.py)에서 조회
    recognition_failure_count: int = Field(0, ge=0, le=10)
    draft: Optional[dict] = None


class ActionRequest(BaseModel):
    user_text: str = Field(min_length=1, max_length=300)
    context: ActionContext
    stt_latency_ms: Optional[int] = Field(None, ge=0)


class ActionResponse(BaseModel):
    ok: bool
    fallback: bool  # True면 프론트가 기존 resolveIntent(정규식)로 처리
    action: Optional[dict] = None
    error_code: Optional[str] = None
    detail: Optional[str] = None
    answer_text: Optional[str] = None  # answer_question일 때 표시/TTS 문구
    repairs: list[str] = Field(default_factory=list)
    source: str = "model"  # model | rule
    screen: Optional[str] = None  # 정규화된 화면(6종)
    llm_latency_ms: Optional[int] = None
