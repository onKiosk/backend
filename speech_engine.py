# -*- coding: utf-8 -*-
"""
OnKiosk Backend - Part 3: On-Device STT & TTS Speech Module (speech_engine.py)
-------------------------------------------------------------------------------
담당: 백엔드 파트 3 (김민재)

구성 (모두 sherpa-onnx 기반, 인터넷/클라우드 불필요)
  STT : Silero VAD(발화 구간 검출) -> SenseVoice(한국어 인식) -> 후처리(태그 제거/표기 보정)
  TTS : 발음 전처리(영문/주문번호 -> 한글) -> VITS 한국어(mimic3 kss) -> 16bit WAV bytes

파트 2(main.py) 사용 예
    from speech_engine import speech_engine
    r = speech_engine.transcribe_audio(audio_bytes=wav_bytes)      # {'text': ..., 'speech_detected': ...}
    t = speech_engine.synthesize_speech("결제 수단을 선택해 주세요.")  # {'wav_bytes': ..., ...}

설계 원칙
  * 가짜 결과를 조용히 만들지 않는다. 모델이 없으면 SpeechModelNotLoadedError 를 던져
    파트 2 가 ask_clarification / call_staff 로 폴백할 수 있게 한다.
    (서버 연동 개발용으로만 환경변수 ONKIOSK_SPEECH_MOCK=1 을 명시했을 때 모의 모드가 켜지고,
     결과에 is_mock=True 가 항상 표시된다.)
  * 개인정보(best-effort): 이 모듈은 음성을 디스크에 쓰지 않고, 처리 직후 자신이 만든 디코딩 버퍼를 0 으로 덮어쓰며,
    파일(wav_path)로 받은 경우 처리 후 삭제한다. 다만 다음은 이 모듈이 지울 수 없다 -> 호출자(파트 2)가 주의할 것:
      - 호출자가 넘긴 audio_bytes / samples 원본 객체 (파이썬 객체라 호출자가 참조를 버려야 함)
      - sherpa-onnx 내부 복사본, OS 의 삭제된 파일 잔여 데이터(보안 삭제 아님)
    따라서 문서/발표에서는 '즉시 폐기'가 아니라 '처리 직후 폐기(best-effort)'로 표현한다.
  * 보안: wav_path 는 '서버가 직접 만든 임시 파일' 경로에만 써야 한다. 기본값(delete_source_file=True)에서는
    그 경로의 파일이 삭제되므로, 요청(클라이언트)에서 받은 경로를 그대로 넘기면 임의 파일 삭제 취약점이 된다.
    API 로는 audio_bytes 만 받고, 크기 상한(MAX_AUDIO_BYTES)은 여기서도 검사하지만 파트 2 에서 먼저 걸러 주는 것이 좋다.
  * 모델은 첫 사용 시 로드(lazy)되므로 import 만으로는 느려지지 않는다. 서버 시작 시 warm_up() 권장.
  * 입력 포맷: WAV/FLAC/OGG(soundfile 이 읽는 형식) 또는 raw PCM16(sample_rate 지정).
    브라우저 MediaRecorder 의 webm/opus 는 지원하지 않으므로 프론트에서 WAV(PCM16) 로 보내야 한다.

주의: sherpa-onnx 2.0 이상에서는 한국어 VITS(mimic3)의 espeak 음소 변환이 제거되어 TTS 가 동작하지 않는다.
      requirements-part3.txt 에 고정된 sherpa-onnx==1.13.8 을 사용할 것.
"""

import io
import os
import re
import time
import wave
import glob
import threading
from collections import OrderedDict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

try:
    import sherpa_onnx  # type: ignore
    SHERPA_VERSION: Optional[str] = getattr(sherpa_onnx, "__version__", "unknown")
except ImportError:  # pragma: no cover - 설치 안 된 환경 대응
    sherpa_onnx = None  # type: ignore
    SHERPA_VERSION = None

try:
    import soundfile as _sf  # type: ignore
except ImportError:  # pragma: no cover
    _sf = None

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STT_SAMPLE_RATE = 16000
MAX_AUDIO_SECONDS = 30.0      # VAD 버퍼 한도. 키오스크 발화는 보통 10초 이내
MAX_AUDIO_BYTES = 16 * 1024 * 1024   # 입력 파일/바이트 상한: 30초 48kHz 스테레오 float WAV(약 11.5MB)까지 허용
MAX_TTS_CHARS = 300


# ==============================================================================
# 예외
# ==============================================================================
class SpeechEngineError(Exception):
    """음성 모듈 공통 예외."""


class SpeechModelNotLoadedError(SpeechEngineError):
    """모델 파일/패키지가 없어 STT·TTS 를 쓸 수 없을 때 (파트 2 는 폴백 처리)."""


class AudioDecodeError(SpeechEngineError, ValueError):
    """입력 오디오를 해석할 수 없거나 규칙(길이 등)을 어겼을 때."""


class SpeechSynthesisError(SpeechEngineError):
    """TTS 합성이 실패(빈 오디오 등)했을 때."""


# ==============================================================================
# 텍스트 후처리 / 전처리
# ==============================================================================
_SENSEVOICE_TAG = re.compile(r"<\|[^|>]*\|>")
# 한국어 STT 에서 자주 달라지는 메뉴 표기. (정식명/별칭 해석은 프론트 규칙 파서·LLM 몫이므로 최소한만 보정)
_SPELLING_FIXES: List[Tuple["re.Pattern[str]", str]] = [
    (re.compile(r"카모마일"), "캐모마일"),
    (re.compile(r"크로와상|크로아상"), "크루아상"),
]


def normalize_korean_utterance(text: str) -> str:
    """SenseVoice 태그(<|ko|> 등) 제거, 공백 정리, 메뉴 표기 변이 보정."""
    cleaned = _SENSEVOICE_TAG.sub("", text or "")
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    for pattern, repl in _SPELLING_FIXES:
        cleaned = pattern.sub(repl, cleaned)
    return cleaned


_LETTER_NAMES = {
    "A": "에이", "B": "비", "C": "씨", "D": "디", "E": "이", "F": "에프", "G": "지", "H": "에이치",
    "I": "아이", "J": "제이", "K": "케이", "L": "엘", "M": "엠", "N": "엔", "O": "오", "P": "피",
    "Q": "큐", "R": "알", "S": "에스", "T": "티", "U": "유", "V": "브이", "W": "더블유", "X": "엑스",
    "Y": "와이", "Z": "제트",
}
_ENGLISH_WORDS = {"ICE": "아이스", "HOT": "핫", "REGULAR": "레귤러", "LARGE": "라지"}


def prepare_tts_text(text: str) -> str:
    """
    VITS 한국어 모델이 읽지 못하는 영문자를 한글 발음으로 바꾼다.
    예) "주문번호 A012번" -> "주문번호 에이012번", "ICE 아메리카노" -> "아이스 아메리카노"
    """
    def repl(m: "re.Match[str]") -> str:
        word = m.group(0)
        known = _ENGLISH_WORDS.get(word.upper())
        if known:
            return known
        return "".join(_LETTER_NAMES.get(ch.upper(), ch) for ch in word)

    return re.sub(r"\s+", " ", re.sub(r"[A-Za-z]+", repl, text or "")).strip()


# ==============================================================================
# 오디오 유틸
# ==============================================================================
def _lowpass_resample(x: np.ndarray, sr_in: int, sr_out: int) -> np.ndarray:
    """
    간단한 저역통과 FIR(윈도우드 싱크) + 선형보간 리샘플러. 다운샘플 시 앨리어싱을 줄인다.
    음성 인식용(16 kHz)으로는 충분한 품질이며 외부 의존성(scipy)이 필요 없다.
    """
    if sr_in == sr_out or len(x) == 0:
        return x.astype(np.float32, copy=False)
    y = x.astype(np.float32, copy=False)
    if sr_in > sr_out:
        cutoff = 0.45 * sr_out / sr_in            # 정규화 차단주파수 (Nyquist = 0.5)
        taps = 127
        n = np.arange(taps) - (taps - 1) / 2
        h = np.sinc(2 * cutoff * n) * np.hanning(taps)
        h /= h.sum()
        y = np.convolve(y, h.astype(np.float32), mode="same")
    n_out = int(round(len(y) * sr_out / sr_in))
    t_out = np.linspace(0, len(y) - 1, n_out)
    return np.interp(t_out, np.arange(len(y)), y).astype(np.float32)


def _decode_audio(
    audio_bytes: Optional[bytes],
    wav_path: Optional[str],
    samples: Optional[np.ndarray],
    sample_rate: Optional[int],
    raw_pcm16: bool,
) -> Tuple[np.ndarray, int]:
    """어떤 입력이든 (float32 mono, sample_rate) 로 통일한다."""
    given = sum(x is not None for x in (audio_bytes, wav_path, samples))
    if given != 1:
        raise AudioDecodeError("audio_bytes / wav_path / samples 중 정확히 하나만 지정해야 합니다.")

    # 디코딩(메모리 할당) 전에 크기부터 막는다 -> 거대한 입력으로 메모리를 고갈시키는 것을 방지
    if audio_bytes is not None and len(audio_bytes) > MAX_AUDIO_BYTES:
        raise AudioDecodeError(f"오디오 데이터가 너무 큽니다({len(audio_bytes):,} > {MAX_AUDIO_BYTES:,} bytes).")
    if wav_path is not None:
        try:
            file_size = os.path.getsize(wav_path)
        except OSError as e:
            raise AudioDecodeError(f"오디오 파일을 열 수 없습니다: {e}") from e
        if file_size > MAX_AUDIO_BYTES:
            raise AudioDecodeError(f"오디오 파일이 너무 큽니다({file_size:,} > {MAX_AUDIO_BYTES:,} bytes).")

    if samples is not None:
        if not sample_rate:
            raise AudioDecodeError("samples 를 넘길 때는 sample_rate 가 필요합니다.")
        data = np.asarray(samples, dtype=np.float32)
        if data.ndim == 2:
            data = data.mean(axis=1)
        return data.copy(), int(sample_rate)

    if audio_bytes is not None and raw_pcm16:
        if not sample_rate:
            raise AudioDecodeError("raw PCM16 입력은 sample_rate 가 필요합니다.")
        if len(audio_bytes) % 2:
            raise AudioDecodeError("PCM16 바이트 길이가 짝수가 아닙니다.")
        return np.frombuffer(audio_bytes, dtype="<i2").astype(np.float32) / 32768.0, int(sample_rate)

    if _sf is None:
        raise SpeechModelNotLoadedError("soundfile 패키지가 필요합니다: pip install soundfile")
    try:
        src = io.BytesIO(audio_bytes) if audio_bytes is not None else wav_path
        data, sr = _sf.read(src, dtype="float32", always_2d=True)
    except Exception as e:  # soundfile 은 여러 종류의 예외를 던진다
        raise AudioDecodeError(f"오디오를 읽을 수 없습니다(WAV/FLAC/OGG 만 지원): {e}") from e
    return data.mean(axis=1).astype(np.float32), int(sr)


def _float_to_wav_bytes(samples: np.ndarray, sample_rate: int) -> bytes:
    pcm = (np.clip(samples, -1.0, 1.0) * 32767.0).astype("<i2")
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm.tobytes())
    return buf.getvalue()


# ==============================================================================
# 엔진
# ==============================================================================
class OnDeviceSpeechEngine:
    """온디바이스 STT(VAD+SenseVoice) 와 TTS(VITS) 관리 클래스. 스레드 안전."""

    def __init__(self, models_dir: Optional[str] = None, num_threads: int = 2, mock: Optional[bool] = None) -> None:
        self.models_dir = models_dir or os.environ.get("ONKIOSK_MODELS_DIR") or os.path.join(BASE_DIR, "models")
        self.num_threads = num_threads
        self.mock = (os.environ.get("ONKIOSK_SPEECH_MOCK") == "1") if mock is None else bool(mock)

        self._stt_lock = threading.Lock()
        self._tts_lock = threading.Lock()
        self._cache_lock = threading.Lock()
        self._recognizer = None
        self._vad = None
        self._tts = None
        self._tts_cache: "OrderedDict[Tuple[str, float], Dict[str, Any]]" = OrderedDict()
        self._tts_cache_size = 64

    # ---------------- 모델 파일 탐색 ----------------
    def _stt_files(self) -> Dict[str, Optional[str]]:
        dirs = sorted(glob.glob(os.path.join(self.models_dir, "sherpa-onnx-sense-voice*")))
        model = tokens = None
        for d in dirs:
            for name in ("model.int8.onnx", "model.onnx"):
                if os.path.exists(os.path.join(d, name)):
                    model = os.path.join(d, name)
                    break
            if os.path.exists(os.path.join(d, "tokens.txt")):
                tokens = os.path.join(d, "tokens.txt")
            if model and tokens:
                break
        vad = os.path.join(self.models_dir, "silero_vad.onnx")
        return {"model": model, "tokens": tokens, "vad": vad if os.path.exists(vad) else None}

    def _tts_files(self) -> Dict[str, Optional[str]]:
        d = os.path.join(self.models_dir, "vits-mimic3-ko_KO-kss_low")
        paths = {
            "model": os.path.join(d, "ko_KO-kss_low.onnx"),
            "tokens": os.path.join(d, "tokens.txt"),
            "data_dir": os.path.join(d, "espeak-ng-data"),
        }
        return {k: (v if os.path.exists(v) else None) for k, v in paths.items()}

    def status(self) -> Dict[str, Any]:
        """현재 상태 점검 (서버 /health 용). 파일이 없으면 missing 목록으로 알려준다."""
        missing: List[str] = []
        if sherpa_onnx is None:
            missing.append("sherpa-onnx 패키지")
        stt, tts = self._stt_files(), self._tts_files()
        missing += [f"STT:{k}" for k, v in stt.items() if v is None]
        missing += [f"TTS:{k}" for k, v in tts.items() if v is None]
        return {
            "mode": "MOCK" if self.mock else "ON_DEVICE_SHERPA_ONNX",
            "is_mock": self.mock,
            "sherpa_onnx_version": SHERPA_VERSION,
            "stt_ready": sherpa_onnx is not None and all(stt.values()),
            "tts_ready": sherpa_onnx is not None and all(tts.values()),
            "stt_loaded": self._recognizer is not None,
            "tts_loaded": self._tts is not None,
            "missing": missing,
            "models_dir": self.models_dir,
        }

    # ---------------- 모델 로드 ----------------
    def _ensure_stt(self) -> float:
        """STT/VAD 를 로드하고, 이번 호출에서 실제로 로드했다면 걸린 시간(ms)을 반환(아니면 0)."""
        if self._recognizer is not None:
            return 0.0
        with self._stt_lock:
            if self._recognizer is not None:
                return 0.0
            if sherpa_onnx is None:
                raise SpeechModelNotLoadedError("sherpa-onnx 가 설치되어 있지 않습니다.")
            f = self._stt_files()
            missing = [k for k, v in f.items() if v is None]
            if missing:
                raise SpeechModelNotLoadedError(
                    f"STT 모델 파일 없음({missing}). `python tools/download_models.py` 로 받아주세요: {self.models_dir}")
            t0 = time.perf_counter()
            self._recognizer = sherpa_onnx.OfflineRecognizer.from_sense_voice(
                model=f["model"], tokens=f["tokens"], num_threads=self.num_threads,
                use_itn=True, language="ko")
            self._vad = sherpa_onnx.VoiceActivityDetector(
                sherpa_onnx.VadModelConfig(
                    silero_vad=sherpa_onnx.SileroVadModelConfig(
                        model=f["vad"], threshold=0.35, min_silence_duration=0.5,
                        min_speech_duration=0.12, window_size=512),
                    sample_rate=STT_SAMPLE_RATE, num_threads=1),
                buffer_size_in_seconds=MAX_AUDIO_SECONDS + 5)
            return (time.perf_counter() - t0) * 1000.0

    def _ensure_tts(self) -> float:
        if self._tts is not None:
            return 0.0
        with self._tts_lock:
            if self._tts is not None:
                return 0.0
            if sherpa_onnx is None:
                raise SpeechModelNotLoadedError("sherpa-onnx 가 설치되어 있지 않습니다.")
            f = self._tts_files()
            missing = [k for k, v in f.items() if v is None]
            if missing:
                raise SpeechModelNotLoadedError(
                    f"TTS 모델 파일 없음({missing}). `python tools/download_models.py` 로 받아주세요: {self.models_dir}")
            t0 = time.perf_counter()
            self._tts = sherpa_onnx.OfflineTts(sherpa_onnx.OfflineTtsConfig(
                model=sherpa_onnx.OfflineTtsModelConfig(
                    vits=sherpa_onnx.OfflineTtsVitsModelConfig(
                        model=f["model"], tokens=f["tokens"], data_dir=f["data_dir"],
                        noise_scale=0.333, noise_scale_w=0.333),
                    num_threads=self.num_threads, provider="cpu")))
            return (time.perf_counter() - t0) * 1000.0

    def warm_up(self) -> Dict[str, Any]:
        """서버 시작 시 호출하면 첫 요청 지연이 없다. 모의 모드에서는 아무것도 로드하지 않는다."""
        if self.mock:
            return {"stt_load_ms": 0.0, "tts_load_ms": 0.0, "is_mock": True}
        return {"stt_load_ms": round(self._ensure_stt(), 1), "tts_load_ms": round(self._ensure_tts(), 1), "is_mock": False}

    # ---------------- STT ----------------
    def _speech_segments(self, x16: np.ndarray) -> List[np.ndarray]:
        """VAD 로 발화 구간만 골라낸다. 단어 앞뒤가 잘리지 않도록 0.15초씩 여유를 두고 겹치면 합친다."""
        window = 512
        pad = (-len(x16)) % window
        padded = np.concatenate([x16, np.zeros(pad, dtype=np.float32)]) if pad else x16
        self._vad.reset()
        for i in range(0, len(padded), window):
            self._vad.accept_waveform(padded[i:i + window])
        self._vad.flush()

        spans: List[List[int]] = []
        margin = int(0.25 * STT_SAMPLE_RATE)
        while not self._vad.empty():
            seg = self._vad.front
            start = int(getattr(seg, "start", 0))
            n = len(seg.samples)
            spans.append([max(0, start - margin), min(len(x16), start + n + margin)])
            self._vad.pop()
        spans.sort()
        merged: List[List[int]] = []
        for s, e in spans:
            if merged and s <= merged[-1][1]:
                merged[-1][1] = max(merged[-1][1], e)
            else:
                merged.append([s, e])
        return [x16[s:e] for s, e in merged]

    def transcribe_audio(
        self,
        audio_bytes: Optional[bytes] = None,
        wav_path: Optional[str] = None,
        samples: Optional[np.ndarray] = None,
        sample_rate: Optional[int] = None,
        raw_pcm16: bool = False,
        delete_source_file: bool = True,
        mock_text: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        음성 -> 한국어 텍스트. 입력은 audio_bytes / wav_path / samples 중 하나.
        (보안) wav_path 는 서버가 만든 임시 파일에만 쓸 것: delete_source_file=True(기본)면 그 파일을 삭제한다.
              클라이언트 요청에서 받은 경로를 넘기지 말고 audio_bytes 를 사용한다.

        반환: text(정리된 인식 결과), raw_text, speech_detected, audio_seconds, speech_seconds,
              stt_latency_ms(VAD+인식 시간, 모델 로드 제외), rtf, model_load_ms(이번 호출에서 로드했을 때만),
              audio_discarded, source_file_deleted, engine_mode, is_mock
        speech_detected=False 면 text 는 "" 이다 -> 파트 2 는 '인식 실패 1회'로 세어 2회 누적 시 call_staff.
        """
        if self.mock:
            if mock_text is None:
                raise SpeechModelNotLoadedError("모의 모드(ONKIOSK_SPEECH_MOCK=1)에서는 mock_text 를 직접 지정해야 합니다.")
            deleted = self._discard(None, wav_path, delete_source_file)
            return {"text": normalize_korean_utterance(mock_text), "raw_text": mock_text, "speech_detected": True,
                    "audio_seconds": 0.0, "speech_seconds": 0.0, "stt_latency_ms": 0.0, "rtf": 0.0,
                    "audio_discarded": True, "source_file_deleted": deleted, "engine_mode": "MOCK", "is_mock": True}

        buffers: List[np.ndarray] = []
        try:
            # 예외가 나도(입력 오류 포함) finally 에서 음성 버퍼 폐기/원본 파일 삭제가 반드시 실행된다.
            result = self._transcribe_core(
                buffers, audio_bytes, wav_path, samples, sample_rate, raw_pcm16)
        finally:
            file_deleted = self._discard(buffers, wav_path, delete_source_file)
        result["audio_discarded"] = True
        result["source_file_deleted"] = file_deleted
        return result

    def _transcribe_core(
        self,
        buffers: List[np.ndarray],
        audio_bytes: Optional[bytes],
        wav_path: Optional[str],
        samples: Optional[np.ndarray],
        sample_rate: Optional[int],
        raw_pcm16: bool,
    ) -> Dict[str, Any]:
        model_load_ms = self._ensure_stt()
        raw, sr = _decode_audio(audio_bytes, wav_path, samples, sample_rate, raw_pcm16)
        buffers.append(raw)
        audio_seconds = len(raw) / sr if sr else 0.0
        if audio_seconds > MAX_AUDIO_SECONDS:
            raise AudioDecodeError(f"오디오가 너무 깁니다({audio_seconds:.1f}초 > {MAX_AUDIO_SECONDS:.0f}초).")

        t0 = time.perf_counter()
        x16 = np.clip(_lowpass_resample(raw, sr, STT_SAMPLE_RATE), -1.0, 1.0)
        buffers.append(x16)

        raw_text, speech_seconds = "", 0.0
        with self._stt_lock:
            segments = self._speech_segments(x16) if len(x16) >= 512 else []
            if segments:
                voiced = np.concatenate(segments)
            elif 0.15 <= audio_seconds <= 8.0 and float(np.abs(x16).max()) >= 0.06:
                voiced = x16
            else:
                voiced = None
            if voiced is not None and len(voiced) > 0:
                buffers.append(voiced)
                speech_seconds = len(voiced) / STT_SAMPLE_RATE
                stream = self._recognizer.create_stream()
                stream.accept_waveform(STT_SAMPLE_RATE, voiced)
                self._recognizer.decode_stream(stream)
                raw_text = stream.result.text
        latency_ms = (time.perf_counter() - t0) * 1000.0
        text = normalize_korean_utterance(raw_text)

        result = {
            "text": text, "raw_text": raw_text, "speech_detected": bool(text),
            "audio_seconds": round(audio_seconds, 2), "speech_seconds": round(speech_seconds, 2),
            "stt_latency_ms": round(latency_ms, 2),
            "rtf": round(latency_ms / 1000.0 / audio_seconds, 3) if audio_seconds else 0.0,
            "engine_mode": "ON_DEVICE_SHERPA_ONNX", "is_mock": False,
        }
        if model_load_ms:
            result["model_load_ms"] = round(model_load_ms, 1)
        return result

    @staticmethod
    def _discard(buffers: Optional[List[np.ndarray]], wav_path: Optional[str], delete_source_file: bool) -> Optional[bool]:
        """
        처리 직후 정리(best-effort): 이 모듈이 만든 디코딩 버퍼를 0 으로 덮어쓰고, 원본 파일이 있으면 삭제한다.
        호출자가 넘긴 원본 객체나 OS 의 삭제된 파일 잔여 데이터는 지우지 못한다(보안 삭제 아님).
        """
        for b in buffers or []:
            try:
                b.fill(0.0)
            except Exception:
                pass
        if wav_path and delete_source_file:
            try:
                if os.path.exists(wav_path):
                    os.remove(wav_path)
                return not os.path.exists(wav_path)
            except OSError:
                return False
        return None

    # ---------------- TTS ----------------
    def synthesize_speech(self, text: str, speed: float = 1.0) -> Dict[str, Any]:
        """
        안내 문구 -> 16bit PCM WAV bytes. 자주 쓰는 문구는 캐시되어 두 번째부터 즉시 반환된다.
        빈 문구/너무 긴 문구는 ValueError, 합성 결과가 비면 SpeechSynthesisError (조용히 무음을 만들지 않는다).
        """
        if not isinstance(text, str) or not text.strip():
            raise ValueError("TTS 텍스트가 비어 있습니다.")
        if len(text) > MAX_TTS_CHARS:
            raise ValueError(f"TTS 텍스트가 너무 깁니다({len(text)}자 > {MAX_TTS_CHARS}자).")
        if not 0.5 <= float(speed) <= 2.0:
            raise ValueError("speed 는 0.5 ~ 2.0 사이여야 합니다.")

        spoken = prepare_tts_text(text)
        if not spoken:
            raise ValueError("발음 가능한 텍스트가 없습니다.")

        if self.mock:
            sr = 22050
            wav = _float_to_wav_bytes(np.zeros(int(sr * 0.3), dtype=np.float32), sr)
            return {"text": text, "spoken_text": spoken, "wav_bytes": wav, "sample_rate": sr, "duration_s": 0.3,
                    "tts_latency_ms": 0.0, "rtf": 0.0, "cached": False, "engine_mode": "MOCK", "is_mock": True}

        key = (spoken, round(float(speed), 2))
        with self._cache_lock:
            cached = self._tts_cache.get(key)
            if cached is not None:
                self._tts_cache.move_to_end(key)
        if cached is not None:
            return {**cached, "text": text, "tts_latency_ms": 0.0, "rtf": 0.0, "cached": True}

        model_load_ms = self._ensure_tts()
        t0 = time.perf_counter()
        with self._tts_lock:
            audio = self._tts.generate(spoken, sid=0, speed=float(speed))
        latency_ms = (time.perf_counter() - t0) * 1000.0

        samples = np.asarray(audio.samples, dtype=np.float32)
        if len(samples) == 0 or not audio.sample_rate or float(np.abs(samples).max()) < 1e-4:
            raise SpeechSynthesisError(f"TTS 합성 결과가 비어 있거나 무음입니다: {spoken!r}")
        duration = len(samples) / audio.sample_rate
        result = {
            "text": text, "spoken_text": spoken,
            "wav_bytes": _float_to_wav_bytes(samples, audio.sample_rate),
            "sample_rate": int(audio.sample_rate), "duration_s": round(duration, 2),
            "tts_latency_ms": round(latency_ms, 2), "rtf": round(latency_ms / 1000.0 / duration, 3),
            "cached": False, "engine_mode": "ON_DEVICE_SHERPA_ONNX", "is_mock": False,
        }
        if model_load_ms:
            result["model_load_ms"] = round(model_load_ms, 1)
        with self._cache_lock:
            self._tts_cache[key] = {k: v for k, v in result.items() if k != "model_load_ms"}
            while len(self._tts_cache) > self._tts_cache_size:
                self._tts_cache.popitem(last=False)
        return result


# 파트 2(main.py)에서 `from speech_engine import speech_engine` 로 사용 (모델은 첫 사용/ warm_up() 때 로드)
speech_engine = OnDeviceSpeechEngine()
