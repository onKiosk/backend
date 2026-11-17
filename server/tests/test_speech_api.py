import io
import json
import wave
from types import SimpleNamespace

from fastapi.testclient import TestClient

from app import speech_hooks
from app.backends.mock import MockBackend
from app.factory import create_app
from app.parser import parse_action


def _make_demo_wav(frames: bytes, sr: int = 24000) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(frames)
    return buf.getvalue()


def test_normalize_tts_korean_cart_prompt():
    out = speech_hooks.normalize_tts_korean("1잔 담았어요. 총 4,500원 맞으신가요?")
    assert out == "한 잔 담았어요. 총 사천오백원 맞으신가요?"


def test_normalize_tts_korean_order_number():
    out = speech_hooks.normalize_tts_korean("주문번호 A002입니다.")
    assert out == "주문번호 에이 공 공 이 번입니다."

    out_with_beon = speech_hooks.normalize_tts_korean("주문번호 A002번입니다.")
    assert out_with_beon == "주문번호 에이 공 공 이 번입니다."


def test_normalize_tts_korean_receipt_prompt():
    out = speech_hooks.normalize_tts_korean("결제가 완료됐어요. 영수증을 출력할까요?")
    assert out == "결제가 완료되었어요. 영수증을 출력할까요?"


def test_normalize_tts_korean_common_kiosk_prompts():
    assert speech_hooks.normalize_tts_korean("담으실 메뉴를 선택해주세요.") == "담으실 메뉴를 선택해 주세요."
    assert speech_hooks.normalize_tts_korean("결제방법을 골라주세요.") == "결제 방법을 골라 주세요."
    assert (
        speech_hooks.normalize_tts_korean("추가 옵션이 있으면 말씀해 주세요. 없으면 “담아 주세요”")
        == "추가 옵션이 있으면 말씀해 주세요. 없으면 담아 주세요"
    )


def test_normalize_stt_korean_fixes_splits_and_variants():
    assert speech_hooks.normalize_stt_korean("네 맞 아요.") == "네 맞아요."
    assert speech_hooks.normalize_stt_korean("내 맞습니다") == "네 맞습니다"
    assert speech_hooks.normalize_stt_korean("카페 라떼 레귤로 하나요") == "카페라떼 레귤러 하나요"
    assert speech_hooks.normalize_stt_korean("콜드부르 따듯한 거") == "콜드브루 따뜻한 거"
    assert speech_hooks.normalize_stt_korean("정립 안 할게요") == "적립 안 할게요"
    assert speech_hooks.normalize_stt_korean("내귤러로 주세요.") == "레귤러로 주세요."
    assert speech_hooks.normalize_stt_korean("내 글러 주세요.") == "레귤러 주세요."
    assert speech_hooks.normalize_stt_korean("낮으로 주세요.") == "라지로 주세요."
    assert speech_hooks.normalize_stt_korean("낮이 주세요.") == "라지 주세요."
    assert speech_hooks.normalize_stt_korean("네모네이드 주세요.") == "레몬에이드 주세요."
    assert speech_hooks.normalize_stt_korean("나떼 주세요.") == "라떼 주세요."
    assert speech_hooks.normalize_stt_korean("아이 수로 주세요.") == "아이스로 주세요."
    assert speech_hooks.normalize_stt_korean("샤추가!") == "샷 추가!"
    assert speech_hooks.normalize_stt_korean("셔추가.") == "샷 추가."
    assert speech_hooks.normalize_stt_korean("차추가!") == "샷 추가!"
    assert speech_hooks.normalize_stt_korean("에스프레소 셔츠가.") == "에스프레소 샷 추가."
    assert speech_hooks.normalize_stt_korean("에스프레소셔추가.") == "에스프레소 샷 추가."
    assert speech_hooks.normalize_stt_korean("샤타나 추가.") == "샷 하나 추가."
    assert speech_hooks.normalize_stt_korean("실업추.") == "시럽 추가."
    assert speech_hooks.normalize_stt_korean("오토 우유로 변경.") == "오트 우유로 변경."
    assert speech_hooks.normalize_stt_korean("우유변.") == "우유 변경."


def test_parser_normalizes_action_aliases_and_control_chars():
    a = parse_action('{"action":"ask",\n"field":"temperature","item":"아메리카노","choices":["HOT","ICE"]}')
    assert a.action == "ask_clarification"
    assert a.item == "아메리카노"

    b = parse_action('{"action":"confirm","summary_required":true}')
    assert b.action == "confirm_order"


def test_disk_tts_cache_single_and_concatenated(tmp_path):
    w1 = _make_demo_wav(b"\x01\x00" * 100)
    w2 = _make_demo_wav(b"\x02\x00" * 200)
    (tmp_path / "a.wav").write_bytes(w1)
    (tmp_path / "b.wav").write_bytes(w2)
    (tmp_path / "index.json").write_text(
        json.dumps({"1잔 담았어요.": "a.wav", "총 4,500원 맞으신가요?": "b.wav"}, ensure_ascii=False),
        "utf-8",
    )

    assert speech_hooks._lookup_tts_cache("1잔 담았어요.", cache_dir=tmp_path) == w1
    joined = speech_hooks._lookup_tts_cache("1잔 담았어요. 총 4,500원 맞으신가요?", cache_dir=tmp_path)
    assert joined is not None
    with wave.open(io.BytesIO(joined), "rb") as wf:
        # 100 frames + 120ms(2880 frames at 24kHz) + 200 frames
        assert wf.getnframes() == 100 + int(24000 * 0.12) + 200


def test_tts_endpoint_uses_normalized_text_and_default_speed_1(monkeypatch):
    seen: list[tuple[str, float]] = []

    class StubEngine:
        mock = True

        def warm_up(self):
            return {"stt_load_ms": 0.0, "tts_load_ms": 0.0}

        def synthesize_speech(self, text: str, speed: float = 1.0):
            seen.append((text, speed))
            return {
                "wav_bytes": b"RIFFdemoWAVEfmt ",
                "tts_latency_ms": 1.0,
                "cached": False,
            }

    monkeypatch.setattr(speech_hooks, "_se", SimpleNamespace(
        speech_engine=StubEngine(),
        SpeechModelNotLoadedError=RuntimeError,
        AudioDecodeError=ValueError,
        SpeechEngineError=RuntimeError,
        SpeechSynthesisError=RuntimeError,
    ))

    client = TestClient(create_app(MockBackend()))
    res = client.post("/api/tts", json={"text": "1잔 담았어요. 총 4,500원 맞으신가요?"})
    assert res.status_code == 200
    assert seen == [("한 잔 담았어요. 총 사천오백원 맞으신가요?", 1.0)]
