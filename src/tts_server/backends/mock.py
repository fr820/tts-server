"""Deterministic synthetic-audio backend for development, tests, CI,
and benchmark pipeline validation. No GPU, no downloads."""

from __future__ import annotations

import array
import asyncio
import hashlib
import math
from collections.abc import AsyncIterator

from tts_server.backends.base import TTSBackend
from tts_server.config import AppConfig
from tts_server.models import (
    TTSCapabilities,
    TTSChunk,
    TTSRequest,
    TTSResult,
)
from tts_server.streaming.audio import slice_pcm


class MockBackend(TTSBackend):
    name = "mock"
    capabilities = TTSCapabilities(
        supports_streaming_input=True,
        supports_streaming_output=True,
        streaming_mode="native",
        supports_cuda=False,
        supports_cpu=True,
    )

    def __init__(self, config: AppConfig) -> None:
        super().__init__()
        opts = config.backend.options
        self._first_chunk_delay_s = float(opts.get("first_chunk_delay_ms", 20)) / 1000
        self._chunk_interval_s = float(opts.get("chunk_interval_ms", 10)) / 1000
        self._seconds_per_char = float(opts.get("seconds_per_char", 0.06))

    async def load(self) -> None:
        self._loaded = True

    async def close(self) -> None:
        self._loaded = False

    def _generate_pcm(self, request: TTSRequest) -> bytes:
        duration_s = (
            min(max(len(request.text) * self._seconds_per_char, 0.2), 30.0)
            / request.speed
        )
        n_samples = int(duration_s * request.sample_rate)
        digest = hashlib.sha256(request.text.encode()).hexdigest()
        freq = 200.0 + (int(digest, 16) % 200)
        step = 2.0 * math.pi * freq / request.sample_rate
        samples = array.array(
            "h", (int(6553 * math.sin(step * i)) for i in range(n_samples))
        )
        return samples.tobytes()

    async def synthesize(self, request: TTSRequest) -> TTSResult:
        pcm = await asyncio.to_thread(self._generate_pcm, request)
        return TTSResult(audio=pcm, sample_rate=request.sample_rate)

    async def synthesize_stream(
        self, request: TTSRequest
    ) -> AsyncIterator[TTSChunk]:
        pcm = await asyncio.to_thread(self._generate_pcm, request)
        pieces = slice_pcm(pcm, request.sample_rate) or [b""]
        await asyncio.sleep(self._first_chunk_delay_s)
        for i, piece in enumerate(pieces):
            if i > 0:
                await asyncio.sleep(self._chunk_interval_s)
            yield TTSChunk(
                audio=piece,
                sample_rate=request.sample_rate,
                is_final=(i == len(pieces) - 1),
                sequence=i,
            )
