from tts_server.errors import (
    RequestTimeoutError,
    TTSServerError,
    UnsupportedFeatureError,
)
from tts_server.backends.base import TTSBackend
from tts_server.models import (
    BackendHealth,
    TTSCapabilities,
    TTSRequest,
    TTSResult,
)


class DummyBackend(TTSBackend):
    """Non-streaming backend used to exercise the emulated-streaming default."""

    name = "dummy"
    capabilities = TTSCapabilities()

    async def load(self) -> None:
        self._loaded = True

    async def synthesize(self, request: TTSRequest) -> TTSResult:
        # 0.25s of silence at the requested rate
        pcm = b"\x00\x00" * (request.sample_rate // 4)
        return TTSResult(audio=pcm, sample_rate=request.sample_rate)

    async def close(self) -> None:
        self._loaded = False


async def test_emulated_stream_covers_full_audio_and_marks_final():
    backend = DummyBackend()
    await backend.load()
    req = TTSRequest(text="hi", sample_rate=24000)
    chunks = [c async for c in backend.synthesize_stream(req)]
    assert len(chunks) >= 2
    assert chunks[-1].is_final is True
    assert all(not c.is_final for c in chunks[:-1])
    assert [c.sequence for c in chunks] == list(range(len(chunks)))
    joined = b"".join(c.audio for c in chunks)
    assert joined == (await backend.synthesize(req)).audio


async def test_emulated_stream_empty_audio_yields_single_final_chunk():
    class EmptyBackend(DummyBackend):
        async def synthesize(self, request):
            return TTSResult(audio=b"", sample_rate=request.sample_rate)

    backend = EmptyBackend()
    chunks = [c async for c in backend.synthesize_stream(TTSRequest(text=""))]
    assert len(chunks) == 1
    assert chunks[0].is_final is True and chunks[0].audio == b""


async def test_default_health_and_loaded_flag():
    backend = DummyBackend()
    assert backend.loaded is False
    health = await backend.health()
    assert isinstance(health, BackendHealth)
    assert health.loaded is False
    await backend.load()
    assert (await backend.health()).loaded is True


def test_error_hierarchy_and_status_codes():
    err = UnsupportedFeatureError("no wav", capabilities=TTSCapabilities())
    assert isinstance(err, TTSServerError)
    assert err.status_code == 400
    assert err.code == "unsupported_feature"
    assert err.capabilities is not None
    assert RequestTimeoutError("slow").status_code == 504


def test_base_class_default_capabilities_label_emulated_streaming():
    assert TTSBackend.capabilities.streaming_mode == "emulated"
    assert TTSBackend.capabilities.supports_streaming_output is True
