"""answer_question의 answer_code → 한국어 안내 문구 (프론트와 공유할 표, 15개).

문구는 초안이다. 팀에서 확정한 문구로 바꾸면 되고, 프론트에는 GET /api/answer-texts 로 같은 표를 준다.
"""
ANSWER_TEXTS: dict[str, str] = {
    "CASH_AVAILABLE": "현금 결제도 가능해요. 결제 수단에서 현금을 고르고 카운터에서 결제해 주세요.",
    "CARD_AVAILABLE": "신용카드와 체크카드로 결제하실 수 있어요.",
    "EASY_PAYMENT_AVAILABLE": "삼성페이, 카카오페이, 네이버페이, 애플페이 같은 간편결제도 쓰실 수 있어요.",
    "TEMPERATURE_OPTIONS_AVAILABLE": "메뉴에 따라 따뜻하게 또는 차갑게 고르실 수 있어요.",
    "SIZE_OPTIONS_AVAILABLE": "사이즈는 레귤러와 라지 중에서 고르실 수 있어요.",
    "ITEM_OPTION_POLICY": "메뉴마다 얼음 양, 샷, 시럽, 우유 변경이 가능한지 달라요. 메뉴를 고르시면 선택할 수 있는 옵션이 나와요.",
    "PHONE_NUMBER_LOYALTY_AVAILABLE": "전화번호를 입력하시면 스탬프를 적립할 수 있어요.",
    "LOYALTY_INPUT_SECURITY": "전화번호는 화면의 보안 키패드로 입력하시고, 화면이나 음성으로 읽어 드리지 않아요.",
    "COUPON_LIST_AVAILABLE": "사용 가능한 쿠폰 목록을 확인하실 수 있어요.",
    "PAPER_RECEIPT_AVAILABLE": "종이 영수증을 출력해 드려요.",
    "RECEIPT_CHOICE_AFTER_PAYMENT": "결제가 끝나면 영수증 출력 여부를 고르실 수 있어요.",
    "VOICE_GUIDE_AVAILABLE": "음성 안내를 쓰실 수 있어요. 말씀으로도 주문하실 수 있어요.",
    "STAFF_ASSISTANCE_AVAILABLE": "어려우시면 언제든 “직원 불러 줘”라고 말씀해 주세요.",
    "ORDER_NUMBER_ANNOUNCED": "결제가 끝나면 주문번호를 음성으로 안내해 드려요.",
    "VIBRATION_BELL_PICKUP": "음료가 준비되면 진동벨이 울려요. 픽업대에서 받아 가세요.",
}
