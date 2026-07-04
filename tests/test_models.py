from tts_server.models import (
    AudioFormat,
    BackendHealth,
    TTSCapabilities,
    TTSChunk,
    TTSRequest,
    TTSResult,
)


def test_audio_format_values():
    assert AudioFormat.PCM_S16LE.value == "pcm_s16le"
    assert AudioFormat.WAV.value == "wav"


def test_capabilities_defaults_are_honest():
    caps = TTSCapabilities()
    assert caps.supports_streaming_input is False
    assert caps.supports_streaming_output is False
    assert caps.streaming_mode == "none"
    assert caps.supports_cuda is False
    assert caps.supports_cpu is True
    assert AudioFormat.PCM_S16LE in caps.supported_audio_formats


def test_request_defaults_and_request_id():
    req = TTSRequest(text="hello")
    assert req.voice == "default"
    assert req.speed == 1.0
    assert req.sample_rate == 24000
    assert req.format == AudioFormat.PCM_S16LE
    assert len(req.request_id) == 32
    assert req.extra == {}
    # request_id is unique per request
    assert TTSRequest(text="a").request_id != TTSRequest(text="b").request_id


def test_chunk_and_result():
    chunk = TTSChunk(audio=b"\x00\x01", sample_rate=24000, sequence=3)
    assert chunk.is_final is False
    result = TTSResult(audio=b"\x00\x01", sample_rate=24000)
    assert result.format == AudioFormat.PCM_S16LE


def test_backend_health():
    h = BackendHealth(ok=True, loaded=True)
    assert h.detail is None and h.gpu_memory_mb is None
