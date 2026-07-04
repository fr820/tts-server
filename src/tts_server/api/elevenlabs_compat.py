"""ElevenLabs-style WebSocket streaming API."""

from __future__ import annotations

from fastapi import APIRouter, WebSocket

from tts_server.streaming.websocket import StreamInputSession, run_ws_session

router = APIRouter()


@router.websocket("/v1/text-to-speech/{voice_id}/stream-input")
async def stream_input(websocket: WebSocket, voice_id: str):
    await websocket.accept()
    backend = websocket.app.state.backend
    cfg = websocket.app.state.config
    session = StreamInputSession(
        backend,
        voice=voice_id,
        sample_rate=cfg.audio.sample_rate,
        timeout_s=cfg.server.request_timeout_s,
    )
    await run_ws_session(websocket, session)
