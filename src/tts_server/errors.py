"""Exception hierarchy mapped to HTTP/WS error responses by the API layer."""

from __future__ import annotations

from tts_server.models import TTSCapabilities


class TTSServerError(Exception):
    code = "internal_error"
    status_code = 500

    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.message = message


class UnsupportedFeatureError(TTSServerError):
    code = "unsupported_feature"
    status_code = 400

    def __init__(
        self, message: str, capabilities: TTSCapabilities | None = None
    ) -> None:
        super().__init__(message)
        self.capabilities = capabilities


class UnknownBackendError(TTSServerError):
    code = "unknown_backend"
    status_code = 400


class BackendNotLoadedError(TTSServerError):
    code = "backend_not_loaded"
    status_code = 503


class SynthesisError(TTSServerError):
    code = "synthesis_error"
    status_code = 500


class RequestTimeoutError(TTSServerError):
    code = "request_timeout"
    status_code = 504
