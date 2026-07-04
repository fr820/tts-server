def test_speech_pcm_streams_audio(client):
    resp = client.post(
        "/v1/audio/speech",
        json={"input": "Hello, welcome to our voice agent demo.", "voice": "default",
              "response_format": "pcm", "speed": 1.0},
    )
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("application/octet-stream")
    assert len(resp.content) > 1000
    assert len(resp.content) % 2 == 0


def test_speech_wav(client):
    resp = client.post(
        "/v1/audio/speech", json={"input": "hello", "response_format": "wav"}
    )
    assert resp.status_code == 200
    assert resp.headers["content-type"] == "audio/wav"
    assert resp.content[:4] == b"RIFF"


def test_speech_wav_increments_requests_counter(client):
    from tts_server import metrics

    counter = metrics.REQUESTS.labels(backend="mock", api="openai")
    before = counter._value.get()
    resp = client.post(
        "/v1/audio/speech", json={"input": "hello", "response_format": "wav"}
    )
    assert resp.status_code == 200
    assert counter._value.get() == before + 1


def test_speech_wav_stalling_backend_returns_504(client, monkeypatch):
    import asyncio

    async def stalling_synthesize(self, request):
        await asyncio.sleep(60)

    monkeypatch.setattr(
        "tts_server.backends.mock.MockBackend.synthesize", stalling_synthesize
    )
    client.app.state.config.server.request_timeout_s = 0.05
    resp = client.post(
        "/v1/audio/speech", json={"input": "hello", "response_format": "wav"}
    )
    assert resp.status_code == 504
    assert resp.json()["error"]["code"] == "request_timeout"


def test_unsupported_format_returns_400_with_capabilities(client):
    resp = client.post(
        "/v1/audio/speech", json={"input": "hi", "response_format": "mp3"}
    )
    assert resp.status_code == 400
    err = resp.json()["error"]
    assert err["code"] == "unsupported_feature"
    assert "supported_audio_formats" in err["capabilities"]


def test_missing_input_is_422(client):
    assert client.post("/v1/audio/speech", json={}).status_code == 422


def test_out_of_range_speed_is_422(client):
    resp = client.post(
        "/v1/audio/speech", json={"input": "hi", "speed": 10.0}
    )
    assert resp.status_code == 422


def test_models_listing(client):
    resp = client.get("/v1/models")
    assert resp.status_code == 200
    body = resp.json()
    assert body["object"] == "list"
    assert body["data"][0]["id"] == "mock"
    assert body["data"][0]["object"] == "model"
