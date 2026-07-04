"""Shared streaming pipeline: wraps any backend stream with backpressure,
idle timeout, cancellation on client disconnect, and metrics capture."""

from __future__ import annotations

import asyncio
import time
from collections.abc import AsyncIterator

from tts_server import metrics
from tts_server.backends.base import TTSBackend
from tts_server.errors import RequestTimeoutError
from tts_server.models import TTSChunk, TTSRequest, TTSResult

_END = None  # queue sentinel


async def run_stream(
    backend: TTSBackend,
    request: TTSRequest,
    *,
    timeout_s: float,
    api: str = "native",
    max_queue: int = 32,
) -> AsyncIterator[TTSChunk]:
    queue: asyncio.Queue[TTSChunk | Exception | None] = asyncio.Queue(
        maxsize=max_queue
    )

    async def _produce() -> None:
        try:
            async for chunk in backend.synthesize_stream(request):
                await queue.put(chunk)
            await queue.put(_END)
        except Exception as exc:  # forwarded to the consumer
            await queue.put(exc)

    producer = asyncio.create_task(_produce())
    metrics.REQUESTS.labels(backend=backend.name, api=api).inc()
    metrics.ACTIVE_SESSIONS.inc()
    start = time.perf_counter()
    first_chunk_at: float | None = None
    audio_bytes = 0
    outcome = "completed"
    try:
        while True:
            try:
                item = await asyncio.wait_for(queue.get(), timeout=timeout_s)
            except TimeoutError:
                raise RequestTimeoutError(
                    f"no audio chunk within {timeout_s}s (request {request.request_id})"
                ) from None
            if item is _END:
                break
            if isinstance(item, Exception):
                raise item
            if first_chunk_at is None:
                first_chunk_at = time.perf_counter()
                metrics.TTFA_SECONDS.labels(backend=backend.name).observe(
                    first_chunk_at - start
                )
            audio_bytes += len(item.audio)
            yield item
    except (GeneratorExit, asyncio.CancelledError):
        outcome = "cancelled"
        raise
    except BaseException:
        outcome = "failed"
        raise
    finally:
        producer.cancel()
        await asyncio.gather(producer, return_exceptions=True)
        metrics.ACTIVE_SESSIONS.dec()
        elapsed = time.perf_counter() - start
        if outcome == "failed":
            metrics.FAILURES.labels(backend=backend.name, api=api).inc()
        elif outcome == "completed":
            metrics.LATENCY_SECONDS.labels(backend=backend.name).observe(elapsed)
            audio_s = audio_bytes / (2 * request.sample_rate)
            if audio_s > 0:
                metrics.RTF.labels(backend=backend.name).observe(elapsed / audio_s)


async def synthesize_once(
    backend: TTSBackend,
    request: TTSRequest,
    *,
    timeout_s: float,
    api: str = "native",
) -> TTSResult:
    """Non-streaming counterpart to `run_stream`: bounds `backend.synthesize`
    with `timeout_s` and records the same request/failure/latency/RTF
    metrics the streaming path records.
    """
    metrics.REQUESTS.labels(backend=backend.name, api=api).inc()
    metrics.ACTIVE_SESSIONS.inc()
    start = time.perf_counter()
    try:
        try:
            result = await asyncio.wait_for(backend.synthesize(request), timeout=timeout_s)
        except TimeoutError:
            raise RequestTimeoutError(
                f"synthesis did not complete within {timeout_s}s "
                f"(request {request.request_id})"
            ) from None
    except BaseException:
        metrics.FAILURES.labels(backend=backend.name, api=api).inc()
        raise
    finally:
        metrics.ACTIVE_SESSIONS.dec()

    elapsed = time.perf_counter() - start
    metrics.LATENCY_SECONDS.labels(backend=backend.name).observe(elapsed)
    audio_s = len(result.audio) / (2 * result.sample_rate)
    if audio_s > 0:
        metrics.RTF.labels(backend=backend.name).observe(elapsed / audio_s)
    return result
