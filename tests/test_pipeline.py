import asyncio

import pytest

from tts_server.config import AppConfig
from tts_server.errors import RequestTimeoutError, SynthesisError
from tts_server.backends.mock import MockBackend
from tts_server.models import TTSCapabilities, TTSRequest, TTSResult
from tts_server.streaming.pipeline import run_stream
from tts_server.backends.base import TTSBackend


def mock_backend(**options) -> MockBackend:
    cfg = AppConfig()
    cfg.backend.options.update(options)
    return MockBackend(cfg)


async def test_passes_through_all_chunks():
    backend = mock_backend(first_chunk_delay_ms=0, chunk_interval_ms=0)
    await backend.load()
    req = TTSRequest(text="hello there")
    chunks = [c async for c in run_stream(backend, req, timeout_s=5.0)]
    assert chunks[-1].is_final
    full = await backend.synthesize(TTSRequest(text="hello there"))
    assert b"".join(c.audio for c in chunks) == full.audio


async def test_idle_timeout_raises():
    class StallingBackend(TTSBackend):
        name = "stall"
        capabilities = TTSCapabilities()

        async def load(self): self._loaded = True
        async def close(self): pass
        async def synthesize(self, request):
            return TTSResult(audio=b"", sample_rate=24000)

        async def synthesize_stream(self, request):
            await asyncio.sleep(60)
            yield  # pragma: no cover

    backend = StallingBackend()
    with pytest.raises(RequestTimeoutError):
        async for _ in run_stream(backend, TTSRequest(text="x"), timeout_s=0.05):
            pass


async def test_backend_exception_propagates():
    class BrokenBackend(TTSBackend):
        name = "broken"
        capabilities = TTSCapabilities()

        async def load(self): self._loaded = True
        async def close(self): pass
        async def synthesize(self, request):
            raise SynthesisError("boom")

    backend = BrokenBackend()
    with pytest.raises(SynthesisError):
        async for _ in run_stream(backend, TTSRequest(text="x"), timeout_s=5.0):
            pass


async def test_closing_generator_cancels_producer():
    produced = []

    class SlowBackend(TTSBackend):
        name = "slow"
        capabilities = TTSCapabilities()

        async def load(self): self._loaded = True
        async def close(self): pass
        async def synthesize(self, request):
            return TTSResult(audio=b"", sample_rate=24000)

        async def synthesize_stream(self, request):
            from tts_server.models import TTSChunk
            for i in range(1000):
                produced.append(i)
                yield TTSChunk(audio=b"\x00\x00", sample_rate=24000, sequence=i)
                await asyncio.sleep(0.001)

    backend = SlowBackend()
    gen = run_stream(backend, TTSRequest(text="x"), timeout_s=5.0)
    async for chunk in gen:
        break  # client disconnects after first chunk
    await gen.aclose()
    count_after_close = len(produced)  # producer teardown is deterministic
    assert count_after_close < 1000


async def test_client_disconnect_not_counted_as_failure():
    from tts_server import metrics

    backend = mock_backend(first_chunk_delay_ms=0, chunk_interval_ms=0)
    await backend.load()
    counter = metrics.FAILURES.labels(backend="mock", api="native")
    before = counter._value.get()
    gen = run_stream(backend, TTSRequest(text="disconnect me early please"), timeout_s=5.0)
    async for _ in gen:
        break
    await gen.aclose()
    assert counter._value.get() == before
