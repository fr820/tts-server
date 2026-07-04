"""Common data models shared by all backends and API surfaces."""

from __future__ import annotations

import enum
import uuid
from typing import Any, Literal

from pydantic import BaseModel, Field


class AudioFormat(str, enum.Enum):
    PCM_S16LE = "pcm_s16le"
    WAV = "wav"


class TTSCapabilities(BaseModel):
    supports_streaming_input: bool = False
    supports_streaming_output: bool = False
    streaming_mode: Literal["native", "emulated", "none"] = "none"
    supports_voice_cloning: bool = False
    supports_reference_audio: bool = False
    supports_emotion_or_style_control: bool = False
    supports_cuda: bool = False
    supports_cpu: bool = True
    supported_languages: list[str] = Field(default_factory=lambda: ["en"])
    supported_sample_rates: list[int] = Field(default_factory=lambda: [24000])
    supported_audio_formats: list[AudioFormat] = Field(
        default_factory=lambda: [AudioFormat.PCM_S16LE, AudioFormat.WAV]
    )


class TTSRequest(BaseModel):
    text: str
    voice: str = "default"
    speed: float = Field(default=1.0, gt=0.0, le=4.0)
    instructions: str | None = None
    sample_rate: int = 24000
    format: AudioFormat = AudioFormat.PCM_S16LE
    request_id: str = Field(default_factory=lambda: uuid.uuid4().hex)
    extra: dict[str, Any] = Field(default_factory=dict)


class TTSChunk(BaseModel):
    audio: bytes
    sample_rate: int
    is_final: bool = False
    sequence: int = 0


class TTSResult(BaseModel):
    audio: bytes
    sample_rate: int
    format: AudioFormat = AudioFormat.PCM_S16LE


class VoiceInfo(BaseModel):
    id: str
    name: str
    languages: list[str] = Field(default_factory=lambda: ["en"])


class ModelInfo(BaseModel):
    id: str
    owned_by: str = "tts-server"


class BackendHealth(BaseModel):
    ok: bool
    loaded: bool
    detail: str | None = None
    gpu_memory_mb: float | None = None
