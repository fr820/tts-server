from tts_server.config import AppConfig
from tts_server.backends.mock import MockBackend
from tts_server.models import TTSRequest


def make_backend(**options) -> MockBackend:
    cfg = AppConfig()
    cfg.backend.options.update(options)
    return MockBackend(cfg)


async def test_synthesis_is_deterministic():
    backend = make_backend()
    await backend.load()
    req = TTSRequest(text="hello world", request_id="a" * 32)
    r1 = await backend.synthesize(req)
    r2 = await backend.synthesize(TTSRequest(text="hello world", request_id="b" * 32))
    assert r1.audio == r2.audio
    assert len(r1.audio) > 0
    assert len(r1.audio) % 2 == 0  # whole s16le samples


async def test_different_text_gives_different_audio():
    backend = make_backend()
    await backend.load()
    a = await backend.synthesize(TTSRequest(text="aaaaaaaaaaaa"))
    b = await backend.synthesize(TTSRequest(text="bbbbbbbbbbbb"))
    assert a.audio != b.audio


async def test_duration_scales_with_text_and_speed():
    backend = make_backend()
    await backend.load()
    short = await backend.synthesize(TTSRequest(text="hi" * 20))
    long = await backend.synthesize(TTSRequest(text="hi" * 60))
    fast = await backend.synthesize(TTSRequest(text="hi" * 60, speed=2.0))
    assert len(long.audio) > len(short.audio)
    assert len(fast.audio) < len(long.audio)


async def test_native_stream_matches_full_synthesis():
    backend = make_backend(first_chunk_delay_ms=0, chunk_interval_ms=0)
    await backend.load()
    req = TTSRequest(text="stream me", request_id="c" * 32)
    chunks = [c async for c in backend.synthesize_stream(req)]
    assert chunks[-1].is_final
    full = await backend.synthesize(TTSRequest(text="stream me"))
    assert b"".join(c.audio for c in chunks) == full.audio


def test_capabilities_report_native_streaming():
    caps = make_backend().capabilities
    assert caps.streaming_mode == "native"
    assert caps.supports_streaming_output is True
    assert caps.supports_streaming_input is True
    assert caps.supports_cuda is False
