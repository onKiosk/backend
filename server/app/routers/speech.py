import asyncio
import logging

from fastapi import APIRouter, Request, Response
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from .. import speech_hooks

log = logging.getLogger("onkiosk.speech")
router = APIRouter(prefix="/api", tags=["speech"])

_MAX_AUDIO_BYTES = 16 * 1024 * 1024


class TtsRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=300)
    speed: float = Field(default=1.0, ge=0.5, le=2.0)


def _err(status_code: int, code: str, detail: str) -> JSONResponse:
    return JSONResponse(
        status_code=status_code,
        content={"ok": False, "error_code": code, "detail": detail},
    )


@router.post("/stt")
async def post_stt(request: Request):
    """브라우저에서 녹음한 WAV 바이트를 받아 한국어 텍스트로 변환한다."""
    body = await request.body()
    if not body:
        return _err(400, "EMPTY_AUDIO", "오디오 데이터가 비어 있습니다.")
    if len(body) > _MAX_AUDIO_BYTES:
        return _err(400, "AUDIO_TOO_LARGE", "오디오 데이터가 16MB 제한을 초과했습니다.")

    try:
        r = await asyncio.to_thread(speech_hooks.transcribe, body)
    except speech_hooks.SpeechUnavailable as e:
        return _err(503, "SPEECH_UNAVAILABLE", str(e))
    except speech_hooks.BadAudioInput as e:
        return _err(400, "BAD_AUDIO", str(e))

    log.info(
        "[STT] speech_detected=%s text=%r latency_ms=%s",
        r.get("speech_detected"),
        r.get("text"),
        r.get("stt_latency_ms"),
    )
    return {
        "ok": True,
        "text": r["text"],
        "speech_detected": r["speech_detected"],
        "stt_latency_ms": r["stt_latency_ms"],
    }


@router.post("/tts")
async def post_tts(req: TtsRequest):
    """안내 문구를 받아 한국어 VITS 음성(16bit WAV)을 반환한다."""
    try:
        r = await asyncio.to_thread(speech_hooks.synthesize, req.text, req.speed)
    except speech_hooks.SpeechUnavailable as e:
        return _err(503, "SPEECH_UNAVAILABLE", str(e))
    except (speech_hooks.BadAudioInput, speech_hooks.SynthesisFailed) as e:
        return _err(400, "TTS_FAILED", str(e))

    log.info(
        "[TTS] text=%r cached=%s latency_ms=%s",
        req.text,
        r.get("cached"),
        r.get("tts_latency_ms"),
    )
    return Response(
        content=r["wav_bytes"],
        media_type="audio/wav",
        headers={
            "X-TTS-Latency-Ms": str(r["tts_latency_ms"]),
            "X-TTS-Cached": str(r["cached"]).lower(),
        },
    )
