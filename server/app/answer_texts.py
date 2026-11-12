"""answer_question의 answer_code → 한국어 안내 문구 (프론트·서버 공통 표, 15개).

문구는 합의안(최종 확정 통합본) 4.2절 표와 같다. 바꾸려면 합의안과 프론트를 함께 바꾼다.
프론트에는 GET /api/answer-texts 로 같은 표를 준다.
"""
ANSWER_TEXTS: dict[str, str] = {
    "CASH_AVAILABLE": "현금으로도 결제하실 수 있습니다.",
    "CARD_AVAILABLE": "신용카드와 체크카드로 결제하실 수 있습니다.",
    "EASY_PAYMENT_AVAILABLE": "삼성페이, 카카오페이, 네이버페이, 애플페이를 사용하실 수 있습니다.",
    "TEMPERATURE_OPTIONS_AVAILABLE": "메뉴에 따라 따뜻한 음료(HOT)와 차가운 음료(ICE)를 선택하실 수 있습니다.",
    "SIZE_OPTIONS_AVAILABLE": "음료 사이즈는 레귤러(보통)와 라지(큰 잔) 중에서 선택하실 수 있습니다.",
    "ITEM_OPTION_POLICY": "메뉴에 따라 얼음량, 샷 추가, 시럽 추가, 연하게, 두유 변경을 선택하실 수 있습니다.",
    "PHONE_NUMBER_LOYALTY_AVAILABLE": "전화번호를 입력하시면 스탬프를 적립하실 수 있습니다.",
    "LOYALTY_INPUT_SECURITY": "전화번호는 화면 키패드로 안전하게 입력하며 외부에 노출되지 않습니다.",
    "COUPON_LIST_AVAILABLE": "적립 화면에서 사용 가능한 쿠폰 목록을 확인하실 수 있습니다.",
    "PAPER_RECEIPT_AVAILABLE": "결제 후 종이 영수증을 출력하실 수 있습니다.",
    "RECEIPT_CHOICE_AFTER_PAYMENT": "결제가 완료된 뒤 영수증 출력 여부를 선택하실 수 있습니다.",
    "VOICE_GUIDE_AVAILABLE": "화면을 누를 때마다 소리로 읽어주는 음성 안내 기능을 사용하실 수 있습니다.",
    "STAFF_ASSISTANCE_AVAILABLE": "이용이 어려우실 때는 언제든 직원을 호출하실 수 있습니다.",
    "ORDER_NUMBER_ANNOUNCED": "결제가 완료되면 주문번호를 화면과 음성으로 안내해 드립니다.",
    "VIBRATION_BELL_PICKUP": "음료가 준비되면 진동벨로 알려드립니다.",
}
