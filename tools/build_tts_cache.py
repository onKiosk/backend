# -*- coding: utf-8 -*-
"""
OnKiosk 오렌지파이 온디바이스 TTS 디스크 캐시 생성기 (method 2).

Microsoft Neural TTS(ko-KR-SunHiNeural)로 키오스크 안내 문구들을 미리 합성해
24kHz 16-bit mono PCM WAV 파일로 models/tts_cache/ 에 저장한다.
오렌지파이 서버는 인터넷 없이도 index.json 과 WAV 파일만으로 즉시(0ms)
아나운서 품질의 음성을 반환한다.
"""

import asyncio
import hashlib
import io
import json
import os
import re
import sys
import tempfile
import wave
from pathlib import Path

import edge_tts
import miniaudio
import numpy as np

ROOT_DIR = Path(__file__).resolve().parents[1]
CACHE_DIR = ROOT_DIR / "models" / "tts_cache"
INDEX_PATH = CACHE_DIR / "index.json"
VOICE = "ko-KR-SunHiNeural"
SAMPLE_RATE = 24000

MENUS = [
    "아메리카노", "카페라떼", "바닐라라떼", "카푸치노", "카페모카", "콜드브루",
    "초코라떼", "녹차라떼", "고구마라떼", "딸기라떼",
    "유자차", "대추차", "캐모마일", "얼그레이", "복숭아 아이스티",
    "자몽에이드", "레몬에이드", "청포도에이드",
    "치즈케이크", "티라미수", "초코쿠키", "베이글", "플레인크루아상", "마카롱", "소금빵",
]

ANSWER_TEXTS = [
    "현금으로도 결제하실 수 있습니다.",
    "신용카드와 체크카드로 결제하실 수 있습니다.",
    "삼성페이, 카카오페이, 네이버페이, 애플페이를 사용하실 수 있습니다.",
    "메뉴에 따라 따뜻한 음료(HOT)와 차가운 음료(ICE)를 선택하실 수 있습니다.",
    "음료 사이즈는 레귤러(보통)와 라지(큰 잔) 중에서 선택하실 수 있습니다.",
    "메뉴에 따라 얼음량, 샷 추가, 시럽 추가, 연하게, 두유 변경을 선택하실 수 있습니다.",
    "전화번호를 입력하시면 스탬프를 적립하실 수 있습니다.",
    "전화번호는 화면 키패드로 안전하게 입력하며 외부에 노출되지 않습니다.",
    "적립 화면에서 사용 가능한 쿠폰 목록을 확인하실 수 있습니다.",
    "결제 후 종이 영수증을 출력하실 수 있습니다.",
    "결제가 완료된 뒤 영수증 출력 여부를 선택하실 수 있습니다.",
    "화면을 누를 때마다 소리로 읽어주는 음성 안내 기능을 사용하실 수 있습니다.",
    "이용이 어려우실 때는 언제든 직원을 호출하실 수 있습니다.",
    "결제가 완료되면 주문번호를 화면과 음성으로 안내해 드립니다.",
    "음료가 준비되면 진동벨로 알려드립니다.",
]

_DIGIT_WORDS = {
    "0": "공", "1": "일", "2": "이", "3": "삼", "4": "사",
    "5": "오", "6": "육", "7": "칠", "8": "팔", "9": "구",
}
_LETTER_NAMES = {
    "A": "에이", "B": "비", "C": "씨", "D": "디", "E": "이", "F": "에프", "G": "지", "H": "에이치",
    "I": "아이", "J": "제이", "K": "케이", "L": "엘", "M": "엠", "N": "엔", "O": "오", "P": "피",
    "Q": "큐", "R": "알", "S": "에스", "T": "티", "U": "유", "V": "브이", "W": "더블유", "X": "엑스",
    "Y": "와이", "Z": "제트",
}


def has_batchim(word: str) -> bool:
    if not word:
        return False
    code = ord(word[-1]) - 0xAC00
    return 0 <= code < 11172 and (code % 28) > 0


def josa(word: str, with_batchim: str, without: str) -> str:
    return word + (with_batchim if has_batchim(word) else without)


def won(n: int) -> str:
    return f"{n:,}원"


def synth_input_text(text: str) -> str:
    """SunHiNeural이 자연스럽게 발음하도록 따옴표/기호/주문번호만 정리한다."""
    s = text.replace("“", "").replace("”", "").replace('"', "").replace("'", "")
    s = s.replace("·", ", ")

    def _ord_repl(m: "re.Match[str]") -> str:
        letter = _LETTER_NAMES.get(m.group(1).upper(), m.group(1))
        digits = " ".join(_DIGIT_WORDS.get(ch, ch) for ch in m.group(2))
        beon = " 번" if m.group(3) else ""
        return f"{letter}, {digits}{beon}"

    s = re.sub(r"(?<![A-Za-z])([A-Za-z])(\d{3})(?!\d)\s*(번?)", _ord_repl, s)
    return re.sub(r"\s+", " ", s).strip()


def trim_pcm16(pcm_bytes: bytes, sr: int = SAMPLE_RATE, pad_ms: int = 35, threshold: int = 180) -> bytes:
    arr = np.frombuffer(pcm_bytes, dtype="<i2")
    if len(arr) == 0:
        return pcm_bytes
    idx = np.where(np.abs(arr) > threshold)[0]
    if len(idx) == 0:
        return pcm_bytes
    pad = int(sr * pad_ms / 1000)
    start = max(0, int(idx[0]) - pad)
    end = min(len(arr), int(idx[-1]) + pad)
    return arr[start:end].tobytes()


def pcm16_to_wav(pcm_bytes: bytes, sr: int = SAMPLE_RATE) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(pcm_bytes)
    return buf.getvalue()


def collect_phrases() -> list[str]:
    phrases: list[str] = [
        # 화면 기본 안내 (dialog.ts)
        "담으실 메뉴를 선택해주세요.",
        "드실 메뉴를 말씀하시거나 눌러 주세요",
        "따뜻한 걸로 드릴까요, 차가운 걸로 드릴까요?",
        "사이즈는 레귤러, 라지 중 어떤 걸로 드릴까요?",
        "추가 옵션이 있으면 말씀해 주세요. 없으면 “담아 주세요”",
        "추가 옵션이 있으면 말씀해 주세요.",
        "없으면 “담아 주세요”",
        "장바구니가 비어 있어요. 메뉴를 담아 주세요.",
        "장바구니가 비어 있어요.",
        "메뉴를 담아 주세요.",
        "메뉴를 먼저 담아 주세요.",
        "장바구니가 비어 있어요. 메뉴를 먼저 담아 주세요.",
        "스탬프 적립할 전화번호를 눌러 주세요. 건너뛰려면 “적립 안 할게요”",
        "스탬프 적립할 전화번호를 눌러 주세요.",
        "건너뛰려면 “적립 안 할게요”",
        "결제방법을 골라주세요.",
        "결제 수단을 말씀하시거나 골라 주세요.",
        "결제 수단을 골라 주세요.",
        "결제 수단을 다시 골라 주세요.",
        "결제가 완료됐어요. 영수증을 출력할까요?",
        "결제가 완료됐어요.",
        "영수증을 출력할까요?",
        "더 필요하신 게 있나요?",
        # 결제 대기 화면 (payingCopy title + body 조합 및 조각)
        "IC카드를 꽂아 주세요 카드 삽입구는 화면 아래 오른쪽에 있어요. 결제가 끝날 때까지 빼지 마세요.",
        "IC카드를 꽂아 주세요 카드 삽입구는 화면 아래 오른쪽에 있어요.",
        "IC카드를 꽂아 주세요",
        "카드 삽입구는 화면 아래 오른쪽에 있어요.",
        "결제가 끝날 때까지 빼지 마세요.",
        "카운터에서 결제해 주세요 카운터 직원에게 현금 결제를 요청해 주세요.",
        "카운터에서 결제해 주세요",
        "카운터 직원에게 현금 결제를 요청해 주세요.",
        "결제 바코드를 보여 주세요 휴대폰의 결제 바코드를 화면 아래 오른쪽 스캐너에 보여 주세요.",
        "결제 바코드를 보여 주세요",
        "휴대폰의 결제 바코드를 화면 아래 오른쪽 스캐너에 보여 주세요.",
        "휴대폰을 대 주세요 결제 단말기는 화면 아래 오른쪽에 있어요.",
        "휴대폰을 대 주세요",
        "결제 단말기는 화면 아래 오른쪽에 있어요.",
        # 완료·수령 안내
        "화면 아래 왼쪽 배출구에서 진동벨을 가져가 주세요.",
        "벨이 울리면 픽업대로 와 주세요.",
        "진동벨이 울리면 픽업대로 와 주세요.",
        "자리에서 기다려 주세요.",
        "포장 음료는 픽업대에서 받아 가세요.",
        "아직 결제가 끝나지 않았어요.",
        # 직원 호출·인식 실패·상태 안내
        "음성을 두 번 인식하지 못했어요. 직원을 불러 드릴까요? 화면을 눌러 주문을 계속할 수도 있어요.",
        "음성을 두 번 인식하지 못했어요.",
        "직원을 불러 드릴까요?",
        "화면을 눌러 주문을 계속할 수도 있어요.",
        "직원을 호출했어요. 잠시만 기다려 주세요. 곧 도와드릴게요.",
        "직원을 호출했어요.",
        "잠시만 기다려 주세요.",
        "곧 도와드릴게요.",
        "잘 알아듣지 못했어요. 다시 한 번 말씀해 주세요.",
        "잘 알아듣지 못했어요.",
        "다시 한 번 말씀해 주세요.",
        "결제를 기다리고 있어요. 결제 수단을 바꾸려면 “결제 취소”라고 말씀해 주세요.",
        "결제를 기다리고 있어요.",
        "결제 수단을 바꾸려면 “결제 취소”라고 말씀해 주세요.",
        "장바구니를 모두 비웠어요.",
        "결제를 취소했어요.",
        "결제를 취소했어요. 결제 수단을 다시 골라 주세요.",
        "영수증은 결제가 끝난 뒤에 고르실 수 있어요.",
        "쿠폰 기능은 아직 준비 중이에요. 필요하시면 직원에게 문의해 주세요.",
        "쿠폰 기능은 아직 준비 중이에요.",
        "필요하시면 직원에게 문의해 주세요.",
        "죄송해요, 주문과 관련된 말씀만 도와드릴 수 있어요.",
        # 접근성 모드 전환
        "음성 모드예요. 말로만 주문할 수 있어요. 드실 메뉴를 말씀해 주세요.",
        "음성 모드예요.",
        "말로만 주문할 수 있어요.",
        "드실 메뉴를 말씀해 주세요.",
        "기본 모드로 바꿨어요.",
        "하단 모드로 바꿨어요.",
        "고대비 모드로 바꿨어요.",
        # 품절 대체 문구
        "다른 메뉴를 골라 주세요.",
        "대신 아메리카노, 카페라떼 중에서 골라 보시겠어요?",
        "대신 치즈케이크, 티라미수 중에서 골라 보시겠어요?",
    ]

    phrases.extend(ANSWER_TEXTS)

    # 수량 문구 (1~20)
    for n in range(1, 21):
        phrases.append(f"{n}잔 담았어요.")
        phrases.append(f"총 {n}잔 담았어요.")
        phrases.append(f"총 {n}잔 으로 바꿨어요.")
        phrases.append(f"스탬프 {n}개가 적립됐어요.")

    # 25개 메뉴별 담기/빼기/바꾸기/추천/품절 안내
    for name in MENUS:
        phrases.append(f"{josa(name, '을', '를')} 뺐어요.")
        phrases.append(f"장바구니에 {josa(name, '이', '가')} 없어요.")
        phrases.append(f"{josa(name, '을', '를')} 요청하신 대로 바꿨어요.")
        phrases.append(f"{josa(name, '을', '를')} 추천드려요.")
        phrases.append(f"주문하시려면 “{name} 주세요”라고 말씀해 주세요.")
        phrases.append(f"죄송해요, {josa(name, '은', '는')} 지금 품절이에요.")
        phrases.append(f"대신 {josa(name, '은', '는')} 어떠세요?")

    # 금액 문구 (2,500원 ~ 30,000원: 500원 단위 + 시럽 300/600/900원 조합)
    prices: set[int] = set()
    for base in range(2500, 30001, 500):
        for syrup_extra in (0, 300, 600, 900):
            total = base + syrup_extra
            if 2500 <= total <= 30000:
                prices.add(total)
    for p in sorted(prices):
        w = won(p)
        phrases.append(f"총 {w} 맞으신가요?")
        phrases.append(f"총 {w}입니다.")

    # 주문번호 A001 ~ A100
    for i in range(1, 101):
        code = f"A{i:03d}"
        phrases.append(f"주문번호 {code}입니다.")
        phrases.append(f"주문번호 {code}번입니다.")

    # 중복 제거 (순서 유지)
    return list(dict.fromkeys(p.strip() for p in phrases if p.strip()))


async def synthesize_one(sem: asyncio.Semaphore, phrase: str, existing: dict[str, str]) -> tuple[str, str]:
    key = phrase
    if key in existing and (CACHE_DIR / existing[key]).exists():
        return key, existing[key]

    fname = hashlib.sha1(key.encode("utf-8")).hexdigest()[:16] + ".wav"
    out_path = CACHE_DIR / fname
    if out_path.exists():
        return key, fname

    text_for_tts = synth_input_text(key)
    async with sem:
        for attempt in range(3):
            try:
                comm = edge_tts.Communicate(text_for_tts, VOICE)
                mp3_buf = bytearray()
                async for chunk in comm.stream():
                    if chunk["type"] == "audio":
                        mp3_buf.extend(chunk["data"])
                if not mp3_buf:
                    raise RuntimeError(f"Empty audio for {key!r}")
                decoded = miniaudio.decode(
                    bytes(mp3_buf),
                    output_format=miniaudio.SampleFormat.SIGNED16,
                    nchannels=1,
                    sample_rate=SAMPLE_RATE,
                )
                pcm = trim_pcm16(bytes(decoded.samples), sr=SAMPLE_RATE)
                out_path.write_bytes(pcm16_to_wav(pcm, sr=SAMPLE_RATE))
                return key, fname
            except Exception as e:
                if attempt == 2:
                    raise RuntimeError(f"Failed to synthesize {key!r}: {e}") from e
                await asyncio.sleep(0.5 * (attempt + 1))
    return key, fname


async def main() -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    existing: dict[str, str] = {}
    if INDEX_PATH.exists():
        try:
            existing = json.loads(INDEX_PATH.read_text("utf-8"))
        except Exception:
            existing = {}

    phrases = collect_phrases()
    print(f"대상 문구: {len(phrases)}개 (기존 캐시: {len(existing)}개)")

    sem = asyncio.Semaphore(8)
    done = 0
    index = dict(existing)

    tasks = [asyncio.create_task(synthesize_one(sem, p, existing)) for p in phrases]
    for fut in asyncio.as_completed(tasks):
        key, fname = await fut
        index[key] = fname
        done += 1
        if done % 100 == 0 or done == len(phrases):
            print(f"진행: {done}/{len(phrases)}")

    INDEX_PATH.write_text(json.dumps(index, ensure_ascii=False, indent=2), "utf-8")
    total_mb = sum(f.stat().st_size for f in CACHE_DIR.glob("*.wav")) / (1024 * 1024)
    print(f"완료: {len(index)}개 문구 저장 ({total_mb:.1f} MB) -> {INDEX_PATH}")


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    asyncio.run(main())
