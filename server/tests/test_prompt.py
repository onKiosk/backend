"""학습(v7) JSONL 한 줄과 서버가 만드는 모델 입력이 글자 단위로 같은지 검사한다.
형식(태그·줄바꿈·JSON 직렬화)이나 SP01을 바꾸면 이 테스트가 깨진다. 학습 파이프라인과 맞춘 뒤에만 고칠 것."""
import json

from app.parser import parse_action
from app.prompt import SP01, build_messages, build_user_content

CONTEXT = {"screen": "menu", "cart": [], "sold_out": []}
USER_TEXT = "어 저기 따뜻한 아메리카노 레귤러 한 잔 부탁드립니다"
# 민재 님 확정 규격: JSON 바로 다음 줄에 [USER] (빈 줄 없음). 학습 JSONL 원본과 한 번 더 대조할 것.
TRAIN_USER = '[CONTEXT]\n{"screen":"menu","cart":[],"sold_out":[]}\n[USER]\n어 저기 따뜻한 아메리카노 레귤러 한 잔 부탁드립니다'
TRAIN_ASSISTANT = (
    '{"action":"add_to_cart","items":[{"menu":"아메리카노","quantity":1,"temperature":"HOT",'
    '"size":"REGULAR","options":{"shot":0,"syrup":0,"strength":"NORMAL"}}]}'
)


def test_user_content_matches_training_format():
    assert build_user_content(USER_TEXT, CONTEXT) == TRAIN_USER


def test_messages_shape():
    msgs = build_messages(USER_TEXT, CONTEXT)
    assert [m["role"] for m in msgs] == ["system", "user"]
    assert msgs[0]["content"] == SP01
    assert msgs[1]["content"] == TRAIN_USER


def test_sp01_is_v7_with_answer_question():
    assert SP01.startswith("당신은 OnKiosk 음성 명령 파서입니다. 반드시 설명이나 마크다운 없이 JSON 객체 하나만 출력합니다.")
    assert "no_action, answer_question입니다." in SP01
    assert SP01.endswith("가능 여부 질문은 answer_question으로 응답합니다.")
    assert "\n" not in SP01  # 줄바꿈 없는 한 단락


def test_training_assistant_sample_parses():
    action = parse_action(TRAIN_ASSISTANT)
    assert action.action == "add_to_cart"
    assert json.loads(TRAIN_ASSISTANT)["items"][0]["options"]["shot"] == 0
