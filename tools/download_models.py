# -*- coding: utf-8 -*-
"""
STT/TTS/VAD 모델 다운로드 도구 (models/ 폴더는 .gitignore 되어 있어 각자 한 번씩 실행해야 한다)

    python tools/download_models.py            # 없는 모델만 받음
    python tools/download_models.py --force    # 전부 다시 받음

받는 모델 (총 약 230MB)
  - SenseVoice int8 (zh/en/ja/ko/yue 다국어 STT, 한국어 사용)
  - Silero VAD (발화 구간 검출)
  - VITS mimic3 ko_KO-kss_low (한국어 TTS)
GitHub 가 큰 파일에서 일시적으로 504 를 주는 경우가 있어 자동 재시도한다.
"""

import os
import sys
import time
import shutil
import tarfile
import argparse
import urllib.request

BASE = "https://github.com/k2-fsa/sherpa-onnx/releases/download"
MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")

ASSETS = [
    {"name": "Silero VAD", "url": f"{BASE}/asr-models/silero_vad.onnx",
     "check": "silero_vad.onnx", "min_mb": 0.4, "archive": False},
    {"name": "SenseVoice int8 (STT)",
     "url": f"{BASE}/asr-models/sherpa-onnx-sense-voice-zh-en-ja-ko-yue-int8-2024-07-17.tar.bz2",
     "check": os.path.join("sherpa-onnx-sense-voice-zh-en-ja-ko-yue-int8-2024-07-17", "model.int8.onnx"),
     "min_mb": 150, "archive": True},
    {"name": "VITS Korean (TTS)", "url": f"{BASE}/tts-models/vits-mimic3-ko_KO-kss_low.tar.bz2",
     "check": os.path.join("vits-mimic3-ko_KO-kss_low", "ko_KO-kss_low.onnx"),
     "min_mb": 50, "archive": True},
]


def download(url: str, dest: str, retries: int = 8) -> None:
    for attempt in range(1, retries + 1):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "onkiosk-model-downloader"})
            with urllib.request.urlopen(req, timeout=60) as r, open(dest + ".part", "wb") as f:
                shutil.copyfileobj(r, f, length=1 << 20)
            os.replace(dest + ".part", dest)
            return
        except Exception as e:  # 504 등 일시 오류 재시도
            print(f"   시도 {attempt}/{retries} 실패: {e}")
            if os.path.exists(dest + ".part"):
                os.remove(dest + ".part")
            time.sleep(min(5 * attempt, 30))
    raise SystemExit(f"다운로드 실패: {url}")


def safe_extract(tar_path: str, dest_dir: str) -> None:
    """
    tar 압축을 풀되 dest_dir 밖으로 나가는 경로(../, 절대경로)와 링크/장치 파일은 거부한다.
    (Python 3.12+ 와 최신 3.10/3.11 에는 filter="data" 도 함께 적용한다. 구버전에서도 아래 검사로 막는다.)
    """
    base = os.path.realpath(dest_dir)
    with tarfile.open(tar_path, "r:bz2") as tf:
        members = tf.getmembers()
        for m in members:
            target = os.path.realpath(os.path.join(base, m.name))
            if os.path.commonpath([base, target]) != base:
                raise SystemExit(f"위험한 경로가 압축 파일에 들어 있어 중단합니다: {m.name}")
            if m.issym() or m.islnk() or m.isdev() or m.isfifo():
                raise SystemExit(f"링크/장치 파일이 압축 파일에 들어 있어 중단합니다: {m.name}")
        if hasattr(tarfile, "data_filter"):
            tf.extractall(base, members=members, filter="data")
        else:
            tf.extractall(base, members=members)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    os.makedirs(MODELS_DIR, exist_ok=True)

    for a in ASSETS:
        target = os.path.join(MODELS_DIR, a["check"])
        if os.path.exists(target) and not args.force:
            print(f"[건너뜀] {a['name']} - 이미 있음")
            continue
        print(f"[받는 중] {a['name']}")
        fname = os.path.join(MODELS_DIR, os.path.basename(a["url"]))
        download(a["url"], fname)
        if a["archive"]:
            try:
                safe_extract(fname, MODELS_DIR)
            finally:
                if os.path.exists(fname):
                    os.remove(fname)
        if not os.path.exists(target) or os.path.getsize(target) < a["min_mb"] * 1e6:
            raise SystemExit(f"검증 실패(파일 없음/크기 이상): {target}")
        print(f"   OK ({os.path.getsize(target) / 1e6:.1f} MB)")
    print("모든 모델 준비 완료:", MODELS_DIR)


if __name__ == "__main__":
    sys.exit(main())
