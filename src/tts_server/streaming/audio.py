"""Pure-function PCM utilities: slicing, WAV wrapping, base64."""

from __future__ import annotations

import base64
import io
import wave


def slice_pcm(
    pcm: bytes,
    sample_rate: int,
    chunk_ms: int = 100,
    sample_width: int = 2,
    channels: int = 1,
) -> list[bytes]:
    frame_bytes = sample_width * channels
    chunk_bytes = int(sample_rate * chunk_ms / 1000) * frame_bytes
    if chunk_bytes <= 0:
        raise ValueError("chunk_ms and sample_rate must be positive")
    return [pcm[i : i + chunk_bytes] for i in range(0, len(pcm), chunk_bytes)]


def pcm_to_wav(
    pcm: bytes, sample_rate: int, channels: int = 1, sample_width: int = 2
) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(sample_width)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm)
    return buf.getvalue()


def encode_audio_b64(audio: bytes) -> str:
    return base64.b64encode(audio).decode("ascii")
