# -*- coding: utf-8 -*-
"""
STT 인식률 측정 도구 (PC 마이크 또는 WAV 파일) - 파트 3

    python tools/stt_mic_eval.py --list-devices                 # 마이크 장치 번호 확인
    python tools/stt_mic_eval.py                                # 기본 키오스크 문장 10개를 차례로 읽으며 측정
    python tools/stt_mic_eval.py --refs my_sentences.txt        # 내가 정한 문장(한 줄에 하나)으로 측정
    python tools/stt_mic_eval.py --device 2 --seconds 6         # 장치/녹음 시간 지정
    python tools/stt_mic_eval.py --wav a.wav --ref "아메리카노 한 잔 주세요"   # 파일 하나만 확인(마이크 불필요)

측정 항목: 문장별 인식 결과, 글자 오류율(CER), 정확히 일치한 비율, 평균 STT 지연시간(ms).
개인정보: 녹음한 음성은 파일로 저장하지 않으며 인식 직후 메모리 버퍼를 폐기한다.
sounddevice 는 마이크 모드에서만 필요하다 (pip install sounddevice).
"""

import os
import re
import sys
import argparse
from typing import Dict, List, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import numpy as np  # noqa: E402

from speech_engine import OnDeviceSpeechEngine, SpeechEngineError, STT_SAMPLE_RATE  # noqa: E402

DEFAULT_PHRASES = [
    "아메리카노 한 잔 주세요",
    "따뜻한 카페라떼 큰 걸로 주세요",
    "아이스 아메리카노 두 잔 주세요",
    "샷 추가해서 바닐라라떼 주세요",
    "복숭아 아이스티 하나 주세요",
    "치즈케이크도 하나 담아 주세요",
    "장바구니 보여 주세요",
    "카드로 결제할게요",
    "영수증 필요 없어요",
    "직원 좀 불러 주세요",
]


def normalize(s: str) -> str:
    return re.sub(r"[\s\.\,\?\!\'\"]", "", s or "")


def cer(ref: str, hyp: str) -> float:
    """글자 오류율: 공백/문장부호를 뺀 편집거리 / 정답 길이."""
    r, h = normalize(ref), normalize(hyp)
    if not r:
        return 0.0 if not h else 1.0
    prev = list(range(len(h) + 1))
    for i, rc in enumerate(r, 1):
        cur = [i]
        for j, hc in enumerate(h, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (rc != hc)))
        prev = cur
    return prev[-1] / len(r)


def summarize(rows: List[Dict]) -> Dict[str, Optional[float]]:
    """rows: [{'ref','text','cer','latency_ms','speech_detected'}, ...] -> 요약 통계."""
    n = len(rows)
    if n == 0:
        return {"n": 0, "avg_cer": None, "exact_rate": None, "no_speech_rate": None, "avg_latency_ms": None}
    return {
        "n": n,
        "avg_cer": round(sum(r["cer"] for r in rows) / n, 3),
        "exact_rate": round(100.0 * sum(1 for r in rows if r["cer"] == 0.0) / n, 1),
        "no_speech_rate": round(100.0 * sum(1 for r in rows if not r["speech_detected"]) / n, 1),
        "avg_latency_ms": round(sum(r["latency_ms"] for r in rows) / n, 1),
    }


def record(seconds: float, device: Optional[int]) -> np.ndarray:
    try:
        import sounddevice as sd
    except ImportError:
        raise SystemExit("마이크 모드에는 sounddevice 가 필요합니다:  pip install sounddevice")
    audio = sd.rec(int(seconds * STT_SAMPLE_RATE), samplerate=STT_SAMPLE_RATE, channels=1,
                   dtype="float32", device=device)
    sd.wait()
    return audio[:, 0].copy()


def list_devices() -> None:
    try:
        import sounddevice as sd
    except ImportError:
        raise SystemExit("sounddevice 가 설치되어 있지 않습니다:  pip install sounddevice")
    devices = sd.query_devices()
    inputs = [(i, d) for i, d in enumerate(devices) if d["max_input_channels"] > 0]
    if not inputs:
        print("입력(마이크) 장치가 없습니다. 마이크 연결/Windows 개인정보 설정(마이크 허용)을 확인하세요.")
        return
    default_in = sd.default.device[0]
    for i, d in inputs:
        mark = " (기본)" if i == default_in else ""
        print(f"  [{i}] {d['name']}  - 입력 {d['max_input_channels']}채널, {int(d['default_samplerate'])} Hz{mark}")


def evaluate_one(engine: OnDeviceSpeechEngine, samples: np.ndarray, sr: int, ref: str) -> Dict:
    r = engine.transcribe_audio(samples=samples, sample_rate=sr)
    samples.fill(0.0)                       # 녹음 버퍼 폐기
    return {"ref": ref, "text": r["text"], "cer": cer(ref, r["text"]) if ref else 0.0,
            "latency_ms": r["stt_latency_ms"], "speech_detected": r["speech_detected"]}


def print_row(i: int, total: int, row: Dict) -> None:
    flag = "OK  " if row["cer"] == 0.0 else ("약간" if row["cer"] <= 0.25 else "오류")
    print(f"  [{i}/{total}] {flag} CER {row['cer']:.2f} | {row['latency_ms']:.0f} ms | 정답: {row['ref']}")
    print(f"           인식: {row['text'] or '(음성 없음)'}")


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="STT 인식률 측정 (마이크 / WAV)")
    ap.add_argument("--list-devices", action="store_true", help="마이크 장치 목록 출력 후 종료")
    ap.add_argument("--device", type=int, default=None, help="입력 장치 번호 (--list-devices 로 확인)")
    ap.add_argument("--seconds", type=float, default=5.0, help="문장당 녹음 시간(초), 기본 5")
    ap.add_argument("--refs", help="정답 문장 파일 (UTF-8, 한 줄에 한 문장)")
    ap.add_argument("--wav", help="마이크 대신 WAV 파일 하나를 평가")
    ap.add_argument("--ref", default="", help="--wav 와 함께 쓰는 정답 문장")
    args = ap.parse_args(argv)

    if args.list_devices:
        list_devices()
        return 0

    engine = OnDeviceSpeechEngine()
    status = engine.status()
    if not status["stt_ready"]:
        print("STT 모델이 준비되지 않았습니다:", status["missing"])
        print("먼저 실행하세요:  python tools/download_models.py")
        return 2
    engine.warm_up()

    rows: List[Dict] = []
    try:
        if args.wav:
            import soundfile as sf
            data, sr = sf.read(args.wav, dtype="float32", always_2d=True)
            rows.append(evaluate_one(engine, data.mean(axis=1).astype(np.float32), int(sr), args.ref))
            print_row(1, 1, rows[0])
        else:
            if args.refs:
                with open(args.refs, encoding="utf-8") as f:
                    phrases = [ln.strip() for ln in f if ln.strip()]
                if not phrases:
                    print("정답 문장 파일이 비어 있습니다:", args.refs)
                    return 2
            else:
                phrases = DEFAULT_PHRASES
            print(f"문장을 한 줄씩 보여 드립니다. Enter 를 누르고 {args.seconds:g}초 안에 또박또박 읽어 주세요. (Ctrl+C 로 중단)\n")
            for i, phrase in enumerate(phrases, 1):
                input(f"[{i}/{len(phrases)}] 읽을 문장: 「{phrase}」  - 준비되면 Enter ")
                print("   녹음 중...", flush=True)
                audio = record(args.seconds, args.device)
                row = evaluate_one(engine, audio, STT_SAMPLE_RATE, phrase)
                rows.append(row)
                print_row(i, len(phrases), row)
    except KeyboardInterrupt:
        print("\n중단했습니다. 지금까지의 결과로 요약합니다.")
    except SpeechEngineError as e:
        print("음성 처리 오류:", e)
        return 1

    s = summarize(rows)
    if s["n"]:
        print("\n" + "=" * 60)
        print(f" 측정 {s['n']}문장 | 평균 CER {s['avg_cer']} | 완전 일치 {s['exact_rate']}% | "
              f"음성 미검출 {s['no_speech_rate']}% | 평균 STT {s['avg_latency_ms']} ms")
        print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
