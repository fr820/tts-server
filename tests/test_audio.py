import base64
import io
import wave

from tts_server.streaming.audio import (
    encode_audio_b64,
    pcm_to_wav,
    slice_pcm,
)


def test_slice_pcm_chunk_sizes_and_alignment():
    # 1 second of 24kHz mono s16le = 48000 bytes; 100ms chunks = 4800 bytes
    pcm = b"\x00\x01" * 24000
    chunks = slice_pcm(pcm, sample_rate=24000, chunk_ms=100)
    assert len(chunks) == 10
    assert all(len(c) == 4800 for c in chunks)
    assert b"".join(chunks) == pcm


def test_slice_pcm_last_chunk_partial():
    pcm = b"\x00\x01" * 25000  # 50000 bytes
    chunks = slice_pcm(pcm, sample_rate=24000, chunk_ms=100)
    assert len(chunks) == 11
    assert len(chunks[-1]) == 50000 - 10 * 4800
    assert b"".join(chunks) == pcm


def test_slice_pcm_empty_returns_no_chunks():
    assert slice_pcm(b"", sample_rate=24000) == []


def test_pcm_to_wav_roundtrip():
    pcm = b"\x00\x01" * 2400
    data = pcm_to_wav(pcm, sample_rate=24000)
    with wave.open(io.BytesIO(data)) as wf:
        assert wf.getframerate() == 24000
        assert wf.getnchannels() == 1
        assert wf.getsampwidth() == 2
        assert wf.readframes(wf.getnframes()) == pcm


def test_encode_audio_b64():
    assert base64.b64decode(encode_audio_b64(b"abc")) == b"abc"
