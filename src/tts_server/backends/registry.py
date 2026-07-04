"""Backend registry. Values are import paths so heavy dependencies
(torch, transformers) are only imported when that backend is selected."""

from __future__ import annotations

import importlib

from tts_server.backends.base import TTSBackend
from tts_server.config import AppConfig
from tts_server.errors import UnknownBackendError

_BACKENDS: dict[str, str] = {
    "mock": "tts_server.backends.mock:MockBackend",
    "qwen3": "tts_server.backends.qwen3:Qwen3TTSBackend",
}


def available_backends() -> list[str]:
    return sorted(_BACKENDS)


def create_backend(config: AppConfig) -> TTSBackend:
    name = config.backend.name
    try:
        target = _BACKENDS[name]
    except KeyError:
        raise UnknownBackendError(
            f"Unknown backend {name!r}. Available: {', '.join(available_backends())}"
        ) from None
    module_name, _, attr = target.partition(":")
    module = importlib.import_module(module_name)
    cls = getattr(module, attr)
    return cls(config)
