import base64
import json

import pytest
from starlette.websockets import WebSocketDisconnect

from tts_server.errors import SynthesisError
from tts_server.backends.mock import MockBackend


class FailingMockBackend(MockBackend):
    """Mock variant whose streaming synthesis always raises, to exercise the
    WS error-message + 1011-close path."""

    async def synthesize_stream(self, request):
        raise SynthesisError("synthetic synthesis failure")
        yield  # pragma: no cover - make this an async generator


def test_native_tts_streaming(client):
    resp = client.post(
        "/api/v1/tts",
        json={"text": "native streaming test", "stream": True},
    )
    assert resp.status_code == 200
    assert resp.headers["x-backend"] == "mock"
    assert len(resp.content) > 1000


def test_native_tts_full_wav(client):
    resp = client.post(
        "/api/v1/tts",
        json={"text": "full result", "stream": False, "format": "wav"},
    )
    assert resp.status_code == 200
    assert resp.content[:4] == b"RIFF"


def test_native_tts_stream_true_with_wav_returns_complete_riff(client):
    # stream=true + format=wav is not actually streamed: a WAV file needs a
    # header with the final size up front, so this downgrades to a single
    # complete response (documented in the README).
    resp = client.post(
        "/api/v1/tts",
        json={"text": "wav downgrade", "stream": True, "format": "wav"},
    )
    assert resp.status_code == 200
    assert resp.content[:4] == b"RIFF"
    assert resp.headers["content-type"] == "audio/wav"


def test_native_ws(client):
    with client.websocket_connect("/api/v1/tts/ws?voice=default") as ws:
        ws.send_text(json.dumps({"text": "hello native"}))
        ws.send_text(json.dumps({"text": ""}))
        final_seen = False
        audio = b""
        while not final_seen:
            msg = json.loads(ws.receive_text())
            audio += base64.b64decode(msg["audio"])
            final_seen = msg["isFinal"]
    assert len(audio) > 0


def test_backends_listing(client):
    resp = client.get("/api/v1/backends")
    assert resp.status_code == 200
    by_name = {b["name"]: b for b in resp.json()}
    assert set(by_name) == {"mock", "qwen3"}
    assert by_name["mock"]["active"] is True
    assert by_name["mock"]["loaded"] is True
    assert by_name["mock"]["capabilities"]["streaming_mode"] == "native"
    assert by_name["qwen3"]["active"] is False
    assert by_name["qwen3"]["capabilities"] is None


def test_backend_detail_and_404(client):
    assert client.get("/api/v1/backends/mock").status_code == 200
    assert client.get("/api/v1/backends/nope").status_code == 404


def test_native_out_of_range_speed_is_422(client):
    resp = client.post("/api/v1/tts", json={"text": "hi", "speed": 99.0})
    assert resp.status_code == 422


def test_native_ws_invalid_json_closes_1003(client):
    with client.websocket_connect("/api/v1/tts/ws") as ws:
        ws.send_text("not json {")
        with pytest.raises(WebSocketDisconnect) as exc:
            ws.receive_text()
        assert exc.value.code == 1003


def test_native_ws_invalid_sample_rate_closes_1003(client):
    with client.websocket_connect("/api/v1/tts/ws?sample_rate=not-a-number") as ws:
        with pytest.raises(WebSocketDisconnect) as exc:
            ws.receive_text()
        assert exc.value.code == 1003


def test_native_ws_backend_error_sends_message_and_closes_1011(client):
    client.app.state.backend = FailingMockBackend(client.app.state.config)
    with client.websocket_connect("/api/v1/tts/ws") as ws:
        ws.send_text(json.dumps({"text": "hello"}))
        msg = json.loads(ws.receive_text())
        assert msg["isFinal"] is True
        assert msg["error"]["code"] == "synthesis_error"
        assert msg["error"]["message"] == "synthetic synthesis failure"
        with pytest.raises(WebSocketDisconnect) as exc:
            ws.receive_text()
        assert exc.value.code == 1011
