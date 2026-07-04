"""Prometheus collectors. Import-time singletons; the streaming pipeline
records timings here and /metrics renders them."""

from __future__ import annotations

from prometheus_client import (
    CONTENT_TYPE_LATEST,
    Counter,
    Gauge,
    Histogram,
    generate_latest,
)

REQUESTS = Counter("tts_requests_total", "TTS requests", ["backend", "api"])
FAILURES = Counter("tts_failures_total", "Failed TTS requests", ["backend", "api"])
ACTIVE_SESSIONS = Gauge("tts_active_sessions", "In-flight TTS sessions")
TTFA_SECONDS = Histogram(
    "tts_ttfa_seconds", "Time to first audio chunk", ["backend"]
)
LATENCY_SECONDS = Histogram(
    "tts_latency_seconds", "Total synthesis latency", ["backend"]
)
RTF = Histogram("tts_rtf", "Real-time factor (synthesis time / audio time)", ["backend"])
GPU_MEMORY_MB = Gauge("tts_gpu_memory_mb", "Allocated CUDA memory in MB")


def _refresh_gpu_gauge() -> None:
    try:
        import torch  # noqa: PLC0415 — optional dependency

        if torch.cuda.is_available():
            GPU_MEMORY_MB.set(torch.cuda.memory_allocated() / 1e6)
    except ImportError:
        pass


def render_metrics() -> tuple[bytes, str]:
    _refresh_gpu_gauge()
    return generate_latest(), CONTENT_TYPE_LATEST
