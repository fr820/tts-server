"""TTSBackend protocol. Backends without native streaming inherit an
emulated-streaming default that re-slices the full synthesis result."""

from __future__ import annotations

import abc
from collections.abc import AsyncIterator

from tts_server.models import (
    BackendHealth,
    ModelInfo,
    TTSCapabilities,
    TTSChunk,
    TTSRequest,
    TTSResult,
    VoiceInfo,
)
from tts_server.streaming.audio import slice_pcm


class TTSBackend(abc.ABC):
    name: str = "abstract"
    capabilities: TTSCapabilities = TTSCapabilities(
        supports_streaming_output=True, streaming_mode="emulated"
    )

    def __init__(self) -> None:
        self._loaded = False

    @property
    def loaded(self) -> bool:
        return self._loaded

    @abc.abstractmethod
    async def load(self) -> None: ...

    @abc.abstractmethod
    async def synthesize(self, request: TTSRequest) -> TTSResult: ...

    @abc.abstractmethod
    async def close(self) -> None: ...

    async def synthesize_stream(
        self, request: TTSRequest
    ) -> AsyncIterator[TTSChunk]:
        """Emulated streaming: full synthesis, then re-sliced chunks.

        Backends with native streaming MUST override this and set
        capabilities.streaming_mode = "native"; the base-class default
        capabilities honestly report "emulated".
        """
        result = await self.synthesize(request)
        pieces = slice_pcm(result.audio, result.sample_rate) or [b""]
        for i, piece in enumerate(pieces):
            yield TTSChunk(
                audio=piece,
                sample_rate=result.sample_rate,
                is_final=(i == len(pieces) - 1),
                sequence=i,
            )

    async def health(self) -> BackendHealth:
        return BackendHealth(ok=True, loaded=self._loaded)

    def list_voices(self) -> list[VoiceInfo]:
        return [VoiceInfo(id="default", name="default")]

    def model_info(self) -> ModelInfo:
        return ModelInfo(id=self.name)
