"""Native API: full TTSRequest surface plus backend introspection."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Request, Response, WebSocket
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

from tts_server.backends.registry import available_backends
from tts_server.models import AudioFormat, TTSRequest
from tts_server.streaming.audio import pcm_to_wav
from tts_server.streaming.pipeline import run_stream, synthesize_once
from tts_server.streaming.websocket import StreamInputSession, run_ws_session

router = APIRouter(prefix="/api/v1")


class NativeTTSRequest(BaseModel):
    text: str
    voice: str = "default"
    speed: float = Field(default=1.0, gt=0.0, le=4.0)
    instructions: str | None = None
    sample_rate: int | None = None
    format: AudioFormat = AudioFormat.PCM_S16LE
    stream: bool = True
    extra: dict[str, Any] = Field(default_factory=dict)


@router.post("/tts")
async def native_tts(body: NativeTTSRequest, request: Request):
    backend = request.app.state.backend
    cfg = request.app.state.config
    tts_request = TTSRequest(
        text=body.text,
        voice=body.voice,
        speed=body.speed,
        instructions=body.instructions,
        sample_rate=body.sample_rate or cfg.audio.sample_rate,
        format=body.format,
        extra=body.extra,
    )
    headers = {"X-Backend": backend.name}

    if body.stream and body.format == AudioFormat.PCM_S16LE:
        async def audio_bytes():
            async for chunk in run_stream(
                backend, tts_request,
                timeout_s=cfg.server.request_timeout_s, api="native",
            ):
                yield chunk.audio

        return StreamingResponse(
            audio_bytes(), media_type="application/octet-stream", headers=headers
        )

    result = await synthesize_once(
        backend, tts_request, timeout_s=cfg.server.request_timeout_s, api="native"
    )
    if body.format == AudioFormat.WAV:
        return Response(
            content=pcm_to_wav(result.audio, result.sample_rate),
            media_type="audio/wav",
            headers=headers,
        )
    return Response(
        content=result.audio,
        media_type="application/octet-stream",
        headers=headers,
    )


@router.websocket("/tts/ws")
async def native_ws(websocket: WebSocket):
    await websocket.accept()
    backend = websocket.app.state.backend
    cfg = websocket.app.state.config
    params = websocket.query_params
    raw_sample_rate = params.get("sample_rate")
    if raw_sample_rate is None:
        sample_rate = cfg.audio.sample_rate
    else:
        try:
            sample_rate = int(raw_sample_rate)
        except ValueError:
            await websocket.close(code=1003, reason="sample_rate must be an integer")
            return
    session = StreamInputSession(
        backend,
        voice=params.get("voice", "default"),
        sample_rate=sample_rate,
        timeout_s=cfg.server.request_timeout_s,
    )
    await run_ws_session(websocket, session)


def _backend_summary(name: str, request: Request) -> dict:
    active = request.app.state.backend
    if name == active.name:
        return {
            "name": name,
            "active": True,
            "loaded": active.loaded,
            "capabilities": active.capabilities.model_dump(),
        }
    return {"name": name, "active": False, "loaded": None, "capabilities": None}


@router.get("/backends")
async def list_backends_endpoint(request: Request):
    return [_backend_summary(name, request) for name in available_backends()]


@router.get("/backends/{name}")
async def backend_detail(name: str, request: Request):
    if name not in available_backends():
        return JSONResponse(
            status_code=404,
            content={"error": {"code": "unknown_backend", "message": name}},
        )
    return _backend_summary(name, request)
