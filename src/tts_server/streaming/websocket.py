"""WebSocket session state machine shared by the ElevenLabs-compatible and
native WS endpoints. Streaming-input backends synthesize each text segment
as it arrives; others buffer until flush/finalize."""

from __future__ import annotations

import json
import logging
import uuid
from collections.abc import Awaitable, Callable

from fastapi import WebSocket, WebSocketDisconnect
from starlette.websockets import WebSocketState

from tts_server.backends.base import TTSBackend
from tts_server.errors import TTSServerError
from tts_server.logging_config import request_id_var, truncate_text
from tts_server.models import TTSRequest
from tts_server.streaming.audio import encode_audio_b64
from tts_server.streaming.pipeline import run_stream

logger = logging.getLogger("tts_server.streaming.ws")

Send = Callable[[dict], Awaitable[None]]


class StreamInputSession:
    def __init__(
        self,
        backend: TTSBackend,
        *,
        voice: str,
        sample_rate: int,
        timeout_s: float,
        request_id: str | None = None,
    ) -> None:
        self._backend = backend
        self._voice = voice
        self._sample_rate = sample_rate
        self._timeout_s = timeout_s
        self.request_id = request_id or uuid.uuid4().hex
        self._buffer: list[str] = []

    @property
    def backend_name(self) -> str:
        return self._backend.name

    def _message(self, audio_b64: str, is_final: bool) -> dict:
        return {
            "audio": audio_b64,
            "isFinal": is_final,
            "backend": self._backend.name,
            "request_id": self.request_id,
        }

    def error_message(self, exc: TTSServerError) -> dict:
        return {
            "error": {"code": exc.code, "message": exc.message},
            "isFinal": True,
            "backend": self._backend.name,
            "request_id": self.request_id,
        }

    async def _synthesize_segment(self, text: str, send: Send) -> None:
        if not text:
            return
        logger.info("ws segment text=%s", truncate_text(text))
        request = TTSRequest(
            text=text,
            voice=self._voice,
            sample_rate=self._sample_rate,
            request_id=self.request_id,
        )
        async for chunk in run_stream(
            self._backend, request, timeout_s=self._timeout_s, api="elevenlabs"
        ):
            if chunk.audio:
                await send(self._message(encode_audio_b64(chunk.audio), False))

    async def flush(self, send: Send) -> None:
        buffered = "".join(self._buffer)
        self._buffer.clear()
        await self._synthesize_segment(buffered, send)

    async def handle_text(self, text: str, send: Send) -> bool:
        """Process one client text value. Returns True when session is done."""
        if text == "":
            await self.flush(send)
            await send(self._message("", True))
            return True
        if self._backend.capabilities.supports_streaming_input:
            await self._synthesize_segment(text, send)
        else:
            self._buffer.append(text)
        return False


async def run_ws_session(websocket: WebSocket, session: StreamInputSession) -> None:
    """Shared receive loop for the ElevenLabs-compatible and native WS
    endpoints. The caller is responsible for `websocket.accept()` and for
    constructing `session` (including validating any query params) before
    calling this.
    """
    request_id_var.set(session.request_id)

    async def send(message: dict) -> None:
        await websocket.send_text(json.dumps(message))

    try:
        while True:
            raw = await websocket.receive_text()
            try:
                data = json.loads(raw)
            except json.JSONDecodeError:
                data = None
            if not isinstance(data, dict):
                await websocket.close(code=1003, reason="messages must be JSON objects")
                return
            if data.get("flush"):
                await session.flush(send)
                continue
            done = await session.handle_text(data.get("text", ""), send)
            if done:
                break
    except WebSocketDisconnect:
        pass
    except TTSServerError as exc:
        logger.warning("ws session error code=%s message=%s", exc.code, exc.message)
        if websocket.application_state == WebSocketState.CONNECTED:
            await send(session.error_message(exc))
            await websocket.close(code=1011)
    finally:
        if websocket.application_state == WebSocketState.CONNECTED:
            await websocket.close()
