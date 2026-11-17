"""파트 3 speech_engine.py 연동 지점 (모델이 없어도 서버·mock 테스트가 동작하도록 느슨하게 연결).

speech_engine.py를 server/ 또는 저장소 루트(server/ 바로 밖)에서 찾는다.
환경변수 ONKIOSK_SPEECH=off 로 끌 수 있다.
TTS는 models/tts_cache/ (ko-KR-SunHiNeural 24kHz PCM16 WAV 디스크 캐시)를 우선 사용하고,
미캐시 문구는 온라인이면 edge-tts로 합성해 캐시에 저장하며, 오프라인/실패 시 온디바이스 VITS로 폴백한다.
"""
import asyncio
import hashlib
import io
import json
import logging
import os
import re
import sys
import threading
import wave
from pathlib import Path
from typing import Any

log = logging.getLogger("onkiosk.speech")

_ROOT_DIR = Path(__file__).resolve().parents[2]

_se: Any = None
if os.getenv("ONKIOSK_SPEECH", "auto").lower() != "off":
    try:
        import speech_engine as _se  # type: ignore  # noqa: F401
    except ModuleNotFoundError:
        _root = str(_ROOT_DIR)
        if _root not in sys.path:
            sys.path.append(_root)
        try:
            import speech_engine as _se  # type: ignore  # noqa: F401,F811
        except Exception:
            _se = None
    except Exception:
        _se = None

try:
    import edge_tts as _edge_tts  # type: ignore
except Exception:
    _edge_tts = None

try:
    import miniaudio as _miniaudio  # type: ignore
except Exception:
    _miniaudio = None

try:
    import numpy as _np  # type: ignore
except Exception:
    _np = None

_EDGE_VOICE = "ko-KR-SunHiNeural"
_EDGE_SR = 24000
_CACHE_DIR = Path(os.getenv("ONKIOSK_TTS_CACHE_DIR", str(_ROOT_DIR / "models" / "tts_cache")))
_cache_lock = threading.RLock()
_cache_index: dict[str, str] | None = None
_cache_index_dir: Path | None = None

_LETTER_NAMES = {
    "A": "에이", "B": "비", "C": "씨", "D": "디", "E": "이", "F": "에프", "G": "지", "H": "에이치",
    "I": "아이", "J": "제이", "K": "케이", "L": "엘", "M": "엠", "N": "엔", "O": "오", "P": "피",
    "Q": "큐", "R": "알", "S": "에스", "T": "티", "U": "유", "V": "브이", "W": "더블유", "X": "엑스",
    "Y": "와이", "Z": "제트",
}
_DIGIT_ORDER_WORDS = {
    "0": "공", "1": "일", "2": "이", "3": "삼", "4": "사",
    "5": "오", "6": "육", "7": "칠", "8": "팔", "9": "구",
}
_NATIVE_COUNTERS = {
    1: "한", 2: "두", 3: "세", 4: "네", 5: "다섯", 6: "여섯",
    7: "일곱", 8: "여덟", 9: "아홉", 10: "열", 11: "열한", 12: "열두",
}
_SINO_DIGITS = ["", "일", "이", "삼", "사", "오", "육", "칠", "팔", "구"]

# 긴 구문/구체적 표현을 먼저 치환해 부분 문자열 충돌을 피한다.
_PHONETIC_REPLACEMENTS: list[tuple[str, str]] = [
    ("결제방법을", "결제 방법을"),
    ("선택해주세요", "선택해 주세요"),
    ("골라주세요", "골라 주세요"),
    ("말씀해주세요", "말씀해 주세요"),
    ("눌러주세요", "눌러 주세요"),
    ("보여주세요", "보여 주세요"),
    ("됐습니다", "되었습니다"),
    ("됐어요", "되었어요"),
]

# SenseVoice 한국어 STT에서 자주 갈리는 띄어쓰기·음절 분리·철자 변이를 정규화한다.
_STT_REPLACEMENTS: list[tuple[str, str]] = [
    ("맞 아 요", "맞아요"),
    ("맞 아요", "맞아요"),
    ("복숭아 아이스 티", "복숭아 아이스티"),
    ("바닐라 라떼", "바닐라라떼"),
    ("바닐라라 때", "바닐라라떼"),
    ("고구마 라떼", "고구마라떼"),
    ("고구마라 때", "고구마라떼"),
    ("카페 라떼", "카페라떼"),
    ("카피 라떼", "카페라떼"),
    ("카페라 때", "카페라떼"),
    ("카피라떼", "카페라떼"),
    ("초코 라떼", "초코라떼"),
    ("초코라 때", "초코라떼"),
    ("녹차 라떼", "녹차라떼"),
    ("녹차라 때", "녹차라떼"),
    ("딸기 라떼", "딸기라떼"),
    ("딸기라 때", "딸기라떼"),
    ("아메리 카노", "아메리카노"),
    ("청포도 에이드", "청포도에이드"),
    ("자몽 에이드", "자몽에이드"),
    ("레몬 에이드", "레몬에이드"),
    ("카푸 치노", "카푸치노"),
    ("카페 모카", "카페모카"),
    ("치즈 케이크", "치즈케이크"),
    ("초코 쿠키", "초코쿠키"),
    ("소금 빵", "소금빵"),
    ("다마 주세요", "담아 주세요"),
    ("담화 주세요", "담아 주세요"),
    ("담아 드세요", "담아 주세요"),
    ("내 맞습니다", "네 맞습니다"),
    ("내 맞아요", "네 맞아요"),
    ("레귤로", "레귤러"),
    ("레 글로", "레귤러"),
    ("네 글로", "레귤러"),
    ("따듯", "따뜻"),
    ("정립", "적립"),
    ("결재", "결제"),
    ("영수중", "영수증"),
    ("염수증", "영수증"),
]

# SenseVoice 한국어 두음법칙(ㄹ->ㄴ) 및 연음으로 인한 외래어·옵션 왜곡 보정
_STT_REGEX_REPLACEMENTS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"[레네내]\s*[귤글길]\s*[러로]"), "레귤러"),
    (re.compile(r"[레네내]\s*[모몬]\s*에\s*이\s*드|[레네내]모네이드"), "레몬에이드"),
    (re.compile(r"(?<![가-힣])(?:나지지|낮지|나지)\s*[요오]"), "라지요"),
    (re.compile(r"(?<![가-힣])(?:낮|나지?)\s*으로"), "라지로"),
    (re.compile(r"(?<![가-힣])(?:낮이|낮지|나지)(?=\s*(?:주세요|주|요|로|할게|원|한|두|$|[.,?!]))"), "라지"),
    (re.compile(r"(?:나|낮|라)\s*[떼때]"), "라떼"),
    (re.compile(r"(카페|바닐라|초코|녹차|고구마|딸기)\s*라떼"), r"\1라떼"),
    (re.compile(r"아\s*이\s*수\s*로"), "아이스로"),
    (re.compile(r"아\s*이\s*수(?=\s*(?:주세요|요|로|할게|아메|카페|라떼|초코|녹차|$|[.,?!]))"), "아이스"),
    (re.compile(r"샤\s*타\s*나\s*추가"), "샷 하나 추가"),
    (re.compile(r"(?<![가-힣])[샤셔사차샷]\s*츠가"), "샷 추가"),
    (re.compile(r"[샤셔]\s*추(?:가|(?=$|[.,?!\s]))"), " 샷 추가"),
    (re.compile(r"(?<![가-힣])[사차삿삭샷]\s*추(?:가|(?=$|[.,?!\s]))"), "샷 추가"),
    (re.compile(r"(?<![가-힣])[샤셔](?=\s*(?:추가|하나|한\s*개|두\s*개|넣|주|$|[.,?!]))"), "샷"),
    (re.compile(r"실\s*업\s*추(?=[.,?!\s]|$)"), "시럽 추가"),
    (re.compile(r"실\s*업"), "시럽"),
    (re.compile(r"오\s*토(?=\s*(?:우유|밀크))"), "오트"),
    (re.compile(r"우유\s*변(?=[.,?!\s]|$)"), "우유 변경"),
    (
        re.compile(
            r"[콜골홀]드\s*(?:블\s*루|브\s*루|부\s*루|브\s*르|부\s*르|두\s*루|두\s*르|브\s*로|부\s*로|으\s*로|으\s*루|부|브|보)"
        ),
        "콜드브루",
    ),
]


def _under_10000_sino(n: int) -> str:
    parts: list[str] = []
    for unit_val, unit_name in ((1000, "천"), (100, "백"), (10, "십"), (1, "")):
        d = (n // unit_val) % 10
        if not d:
            continue
        if d == 1 and unit_val >= 10:
            parts.append(unit_name)
        else:
            parts.append(_SINO_DIGITS[d] + unit_name)
    return "".join(parts)


def _sino_korean_int(n: int) -> str:
    """정수를 자연스러운 한국어 한자어 수사(금액 읽기용)로 바꾼다."""
    if n <= 0:
        return "영"
    eok = n // 100_000_000
    man = (n % 100_000_000) // 10_000
    rest = n % 10_000
    parts: list[str] = []
    if eok:
        parts.append(_under_10000_sino(eok) + "억")
    if man:
        parts.append("만" if man == 1 else _under_10000_sino(man) + "만")
    if rest:
        parts.append(_under_10000_sino(rest))
    return "".join(parts)


def normalize_tts_korean(text: str) -> str:
    """VITS(espeak-ng) 한국어 합성이 자연스럽게 들리도록 수사와 연음/비음화를 전처리한다."""
    s = (text or "").strip()
    if not s:
        return s

    # 1. 따옴표 제거 및 가운뎃점 쉼표화
    s = s.replace("“", "").replace("”", "").replace('"', "").replace("'", "")
    s = s.replace("·", ", ")

    # 2. 주문번호 (예: A002입니다 / A002번입니다 -> 에이 공 공 이 번입니다)
    def _order_no_repl(m: "re.Match[str]") -> str:
        letter = _LETTER_NAMES.get(m.group(1).upper(), m.group(1))
        digits = " ".join(_DIGIT_ORDER_WORDS.get(ch, ch) for ch in m.group(2))
        return f"{letter} {digits} 번"

    s = re.sub(r"(?<![A-Za-z])([A-Za-z])(\d{3})(?!\d)(?:\s*번)?", _order_no_repl, s)

    # 3. 금액 (예: 4,500원 -> 사천오백원)
    def _won_repl(m: "re.Match[str]") -> str:
        val = int(m.group(1).replace(",", ""))
        return f"{_sino_korean_int(val)}원"

    s = re.sub(r"(\d{1,3}(?:,\d{3})+|\d+)\s*원", _won_repl, s)

    # 4. 수량 단위 (예: 1잔 -> 한 잔, 2개 -> 두 개)
    def _counter_repl(m: "re.Match[str]") -> str:
        num = int(m.group(1))
        unit = m.group(2)
        word = _NATIVE_COUNTERS.get(num)
        return f"{word} {unit}" if word else m.group(0)

    s = re.sub(r"(?<!\d)(\d{1,2})\s*(잔|개)", _counter_repl, s)

    # 5. 한국어 음운 변동/띄어쓰기 보정
    for src, dst in _PHONETIC_REPLACEMENTS:
        s = s.replace(src, dst)

    return re.sub(r"\s+", " ", s).strip()


def normalize_stt_korean(text: str) -> str:
    """STT 결과에서 메뉴·긍정 응답의 띄어쓰기 분리, 두음법칙(ㄹ->ㄴ) 및 발음 변이를 보정한다."""
    s = re.sub(r"\s+", " ", (text or "")).strip()
    if not s:
        return s
    for src, dst in _STT_REPLACEMENTS:
        s = s.replace(src, dst)
    for pattern, repl in _STT_REGEX_REPLACEMENTS:
        s = pattern.sub(repl, s)
    return re.sub(r"\s+", " ", s).strip()


def _norm_cache_key(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip())


def _get_cache_index(cache_dir: Path | None = None) -> dict[str, str]:
    global _cache_index, _cache_index_dir
    d = cache_dir or _CACHE_DIR
    with _cache_lock:
        if _cache_index is not None and _cache_index_dir == d:
            return _cache_index
        idx_path = d / "index.json"
        loaded: dict[str, str] = {}
        if idx_path.exists():
            try:
                raw = json.loads(idx_path.read_text("utf-8"))
                if isinstance(raw, dict):
                    loaded = {_norm_cache_key(str(k)): str(v) for k, v in raw.items()}
            except Exception:
                log.warning("TTS 캐시 index.json 읽기 실패: %s", idx_path, exc_info=True)
        _cache_index = loaded
        _cache_index_dir = d
        return _cache_index


def _read_wav_pcm(wav_bytes: bytes) -> tuple[int, bytes] | None:
    try:
        with wave.open(io.BytesIO(wav_bytes), "rb") as wf:
            if wf.getnchannels() != 1 or wf.getsampwidth() != 2:
                return None
            return wf.getframerate(), wf.readframes(wf.getnframes())
    except Exception:
        return None


def _pcm16_to_wav(pcm_bytes: bytes, sr: int) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(pcm_bytes)
    return buf.getvalue()


def _concat_wav_chunks(wav_list: list[bytes], gap_ms: int = 120) -> bytes | None:
    pcms: list[bytes] = []
    target_sr: int | None = None
    for wb in wav_list:
        parsed = _read_wav_pcm(wb)
        if parsed is None:
            return None
        sr, frames = parsed
        if target_sr is None:
            target_sr = sr
        elif sr != target_sr:
            return None
        pcms.append(frames)
    if target_sr is None or not pcms:
        return None
    silence = b"\x00" * (int(target_sr * gap_ms / 1000) * 2)
    return _pcm16_to_wav(silence.join(pcms), target_sr)


def _lookup_tts_cache(text: str, cache_dir: Path | None = None) -> bytes | None:
    d = cache_dir or _CACHE_DIR
    idx = _get_cache_index(d)
    if not idx:
        return None
    key = _norm_cache_key(text)
    if key in idx:
        p = d / idx[key]
        if p.exists():
            return p.read_bytes()

    parts = [p.strip() for p in re.split(r"(?<=[.?!])\s+", key) if p.strip()]
    if len(parts) >= 2:
        wavs: list[bytes] = []
        for part in parts:
            fname = idx.get(part)
            if not fname:
                return None
            fp = d / fname
            if not fp.exists():
                return None
            wavs.append(fp.read_bytes())
        return _concat_wav_chunks(wavs)
    return None


def _edge_synth_text(text: str) -> str:
    s = text.replace("“", "").replace("”", "").replace('"', "").replace("'", "")
    s = s.replace("·", ", ")

    def _ord_repl(m: "re.Match[str]") -> str:
        letter = _LETTER_NAMES.get(m.group(1).upper(), m.group(1))
        digits = " ".join(_DIGIT_ORDER_WORDS.get(ch, ch) for ch in m.group(2))
        beon = " 번" if m.group(3) else ""
        return f"{letter}, {digits}{beon}"

    s = re.sub(r"(?<![A-Za-z])([A-Za-z])(\d{3})(?!\d)\s*(번?)", _ord_repl, s)
    return re.sub(r"\s+", " ", s).strip()


def _synth_edge_and_cache(text: str, speed: float, cache_dir: Path | None = None) -> bytes | None:
    if _edge_tts is None or _miniaudio is None or os.getenv("ONKIOSK_EDGE_TTS", "auto").lower() == "off":
        return None
    pct = round((float(speed) - 1.0) * 100)
    rate = f"{pct:+d}%"
    spoken = _edge_synth_text(text)

    async def _run() -> bytes:
        comm = _edge_tts.Communicate(spoken, _EDGE_VOICE, rate=rate)
        mp3_buf = bytearray()
        async for chunk in comm.stream():
            if chunk["type"] == "audio":
                mp3_buf.extend(chunk["data"])
        return bytes(mp3_buf)

    try:
        mp3_bytes = asyncio.run(asyncio.wait_for(_run(), timeout=4.0))
        if not mp3_bytes:
            return None
        decoded = _miniaudio.decode(
            mp3_bytes,
            output_format=_miniaudio.SampleFormat.SIGNED16,
            nchannels=1,
            sample_rate=_EDGE_SR,
        )
        pcm = bytes(decoded.samples)
        if _np is not None:
            arr = _np.frombuffer(pcm, dtype="<i2")
            idx = _np.where(_np.abs(arr) > 180)[0]
            if len(idx) > 0:
                pad = int(_EDGE_SR * 0.035)
                pcm = arr[max(0, int(idx[0]) - pad) : min(len(arr), int(idx[-1]) + pad)].tobytes()
        wav_bytes = _pcm16_to_wav(pcm, _EDGE_SR)
    except Exception:
        log.debug("edge-tts 즉석 합성 실패(오프라인 또는 타임아웃) -> VITS 폴백", exc_info=True)
        return None

    if abs(float(speed) - 1.0) < 0.01:
        global _cache_index, _cache_index_dir
        d = cache_dir or _CACHE_DIR
        key = _norm_cache_key(text)
        fname = hashlib.sha1(key.encode("utf-8")).hexdigest()[:16] + ".wav"
        try:
            with _cache_lock:
                d.mkdir(parents=True, exist_ok=True)
                (d / fname).write_bytes(wav_bytes)
                idx = dict(_get_cache_index(d))
                idx[key] = fname
                _cache_index = idx
                _cache_index_dir = d
                (d / "index.json").write_text(
                    json.dumps(idx, ensure_ascii=False, indent=2), "utf-8"
                )
        except Exception:
            log.debug("TTS 동적 캐시 저장 건너뜀", exc_info=True)
    return wav_bytes


class SpeechUnavailable(Exception):
    """음성 엔진 또는 모델이 준비되지 않음 (HTTP 503)."""


class BadAudioInput(Exception):
    """잘못된 오디오 또는 텍스트 입력 (HTTP 400)."""


class SynthesisFailed(Exception):
    """TTS 음성 합성 실패 (HTTP 500)."""


def enabled() -> bool:
    return _se is not None and getattr(_se, "speech_engine", None) is not None


def _engine():
    if not enabled():
        raise SpeechUnavailable("speech_engine 모듈을 사용할 수 없습니다.")
    return _se.speech_engine


def warm_up() -> dict[str, Any] | None:
    """서버 시작 시 STT·TTS 모델을 적재하고 디스크 TTS 캐시 인덱스를 예열한다."""
    idx = _get_cache_index()
    log.info("TTS 디스크 캐시 적재: %d문구 (%s)", len(idx), _CACHE_DIR)
    if not enabled():
        return None
    try:
        eng = _engine()
        info = eng.warm_up()
        log.info("speech_engine warm_up 완료: %s", info)
        return info
    except Exception:
        log.exception("speech_engine warm_up 실패")
        return None


def transcribe(audio_bytes: bytes) -> dict[str, Any]:
    """WAV 오디오 바이트 → 한국어 STT 인식 결과."""
    eng = _engine()
    try:
        res = eng.transcribe_audio(audio_bytes=audio_bytes)
    except _se.SpeechModelNotLoadedError as e:
        raise SpeechUnavailable(str(e)) from e
    except (_se.AudioDecodeError, ValueError) as e:
        raise BadAudioInput(str(e)) from e
    except _se.SpeechEngineError as e:
        raise BadAudioInput(str(e)) from e

    fixed = normalize_stt_korean(res.get("text", ""))
    return {**res, "text": fixed, "speech_detected": bool(fixed)}


def synthesize(text: str, speed: float = 1.0) -> dict[str, Any]:
    """안내 문구 → 고품질 한국어 디스크 캐시 WAV 우선 반환, 미캐시 시 edge-tts/VITS 폴백."""
    eng = _engine()
    if not getattr(eng, "mock", False):
        if abs(float(speed) - 1.0) < 0.01:
            cached_wav = _lookup_tts_cache(text)
            if cached_wav is not None:
                return {"wav_bytes": cached_wav, "tts_latency_ms": 0.0, "cached": True}
        edge_wav = _synth_edge_and_cache(text, speed)
        if edge_wav is not None:
            return {"wav_bytes": edge_wav, "tts_latency_ms": 0.0, "cached": False}

    normalized = normalize_tts_korean(text)
    try:
        return eng.synthesize_speech(text=normalized, speed=speed)
    except _se.SpeechModelNotLoadedError as e:
        raise SpeechUnavailable(str(e)) from e
    except ValueError as e:
        raise BadAudioInput(str(e)) from e
    except _se.SpeechSynthesisError as e:
        raise SynthesisFailed(str(e)) from e
