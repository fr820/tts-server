"""OpenAI-compatible TTS API: POST /v1/audio/speech and GET /v1/models."""

from __future__ import annotations

import logging

from fastapi import APIRouter, Request, Response
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from tts_server.errors import UnsupportedFeatureError
from tts_server.logging_config import truncate_text
from tts_server.models import AudioFormat, TTSRequest
from tts_server.streaming.audio import pcm_to_wav
from tts_server.streaming.pipeline import run_stream, synthesize_once

logger = logging.getLogger("tts_server.api.openai")
router = APIRouter()

_FORMAT_ALIASES = {"pcm": AudioFormat.PCM_S16LE, "wav": AudioFormat.WAV}


class SpeechRequest(BaseModel):
    model: str | None = None
    input: str
    voice: str = "default"
    response_format: str = "pcm"
    speed: float = Field(default=1.0, gt=0.0, le=4.0)
    instructions: str | None = None


@router.post("/v1/audio/speech")
async def create_speech(body: SpeechRequest, request: Request):
    backend = request.app.state.backend
    cfg = request.app.state.config
    fmt = _FORMAT_ALIASES.get(body.response_format)
    if fmt is None or fmt not in backend.capabilities.supported_audio_formats:
        raise UnsupportedFeatureError(
            f"response_format {body.response_format!r} not supported by "
            f"backend {backend.name!r}",
            capabilities=backend.capabilities,
        )

    tts_request = TTSRequest(
        text=body.input,
        voice=body.voice,
        speed=body.speed,
        instructions=body.instructions,
        sample_rate=cfg.audio.sample_rate,
        format=fmt,
    )
    logger.info(
        "speech request backend=%s format=%s text=%s",
        backend.name, fmt.value, truncate_text(body.input),
    )

    if fmt == AudioFormat.PCM_S16LE:
        async def audio_bytes():
            async for chunk in run_stream(
                backend, tts_request,
                timeout_s=cfg.server.request_timeout_s, api="openai",
            ):
                yield chunk.audio

        return StreamingResponse(
            audio_bytes(),
            media_type="application/octet-stream",
            headers={"X-Backend": backend.name},
        )

    result = await synthesize_once(
        backend, tts_request, timeout_s=cfg.server.request_timeout_s, api="openai"
    )
    return Response(
        content=pcm_to_wav(result.audio, result.sample_rate),
        media_type="audio/wav",
        headers={"X-Backend": backend.name},
    )


@router.get("/v1/models")
async def list_models(request: Request):
    info = request.app.state.backend.model_info()
    return {
        "object": "list",
        "data": [{"id": info.id, "object": "model", "owned_by": info.owned_by}],
    }
