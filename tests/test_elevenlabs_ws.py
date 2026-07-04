import base64
import json

import pytest
from starlette.websockets import WebSocketDisconnect as StarletteWSDisconnect

from tts_server.backends.mock import MockBackend
from tts_server.config import AppConfig
from tts_server.errors import SynthesisError
from tts_server.streaming.websocket import StreamInputSession


def collect_until_final(ws) -> list[dict]:
    messages = []
    while True:
        msg = json.loads(ws.receive_text())
        messages.append(msg)
        if msg["isFinal"]:
            return messages


def test_stream_input_incremental_text(client):
    with client.websocket_connect(
        "/v1/text-to-speech/default/stream-input"
    ) as ws:
        ws.send_text(json.dumps({"text": "Hello, "}))
        ws.send_text(json.dumps({"text": "welcome to our realtime voice demo."}))
        ws.send_text(json.dumps({"text": ""}))
        messages = collect_until_final(ws)

    assert messages[-1]["isFinal"] is True
    audio = b"".join(base64.b64decode(m["audio"]) for m in messages)
    assert len(audio) > 1000
    assert all(m["backend"] == "mock" for m in messages)
    assert all(len(m["request_id"]) > 0 for m in messages)


def test_flush_message_produces_audio_without_closing(client):
    with client.websocket_connect(
        "/v1/text-to-speech/default/stream-input"
    ) as ws:
        ws.send_text(json.dumps({"text": "part one."}))
        ws.send_text(json.dumps({"flush": True}))
        first = json.loads(ws.receive_text())
        assert first["isFinal"] is False
        assert len(base64.b64decode(first["audio"])) > 0
        # session still open: finalize normally
        ws.send_text(json.dumps({"text": ""}))
        messages = collect_until_final(ws)
        assert messages[-1]["isFinal"] is True


def test_empty_session_finalizes_cleanly(client):
    with client.websocket_connect(
        "/v1/text-to-speech/default/stream-input"
    ) as ws:
        ws.send_text(json.dumps({"text": ""}))
        msg = json.loads(ws.receive_text())
        assert msg["isFinal"] is True and msg["audio"] == ""


def test_invalid_json_closes_with_1003(client):
    with client.websocket_connect(
        "/v1/text-to-speech/default/stream-input"
    ) as ws:
        ws.send_text("not json {")
        with pytest.raises(StarletteWSDisconnect) as exc:
            ws.receive_text()
        assert exc.value.code == 1003


def test_non_json_payload_closes_with_1003(client):
    with client.websocket_connect(
        "/v1/text-to-speech/default/stream-input"
    ) as ws:
        ws.send_text(json.dumps(["not", "an", "object"]))
        with pytest.raises(StarletteWSDisconnect) as exc:
            ws.receive_text()
        assert exc.value.code == 1003


class FailingMockBackend(MockBackend):
    """Mock variant whose streaming synthesis always raises, to exercise the
    WS error-message + 1011-close path."""

    async def synthesize_stream(self, request):
        raise SynthesisError("synthetic synthesis failure")
        yield  # pragma: no cover - make this an async generator


def test_stream_input_backend_error_sends_message_and_closes_1011(client):
    client.app.state.backend = FailingMockBackend(client.app.state.config)
    with client.websocket_connect(
        "/v1/text-to-speech/default/stream-input"
    ) as ws:
        ws.send_text(json.dumps({"text": "hello"}))
        msg = json.loads(ws.receive_text())
        assert msg["isFinal"] is True
        assert msg["error"]["code"] == "synthesis_error"
        assert msg["error"]["message"] == "synthetic synthesis failure"
        with pytest.raises(StarletteWSDisconnect) as exc:
            ws.receive_text()
        assert exc.value.code == 1011


class BufferingMockBackend(MockBackend):
    """Mock variant without streaming-input support: session must buffer."""

    capabilities = MockBackend.capabilities.model_copy(
        update={"supports_streaming_input": False}
    )


async def test_non_streaming_input_backend_buffers_until_finalize():
    cfg = AppConfig()
    cfg.backend.options.update({"first_chunk_delay_ms": 0, "chunk_interval_ms": 0})
    backend = BufferingMockBackend(cfg)
    await backend.load()
    session = StreamInputSession(
        backend, voice="default", sample_rate=24000, timeout_s=5.0
    )
    sent: list[dict] = []

    async def send(message: dict) -> None:
        sent.append(message)

    assert await session.handle_text("part one, ", send) is False
    assert sent == []  # buffered, nothing synthesized yet
    assert await session.handle_text("", send) is True
    assert sent[-1]["isFinal"] is True and sent[-1]["audio"] == ""
    assert any(m["audio"] for m in sent[:-1])  # flush produced audio
