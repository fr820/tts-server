import sys

import pytest

from tts_server.config import AppConfig
from tts_server.errors import UnknownBackendError
from tts_server.backends.registry import available_backends, create_backend


def test_available_backends():
    assert available_backends() == ["mock", "qwen3"]


def test_unknown_backend_raises_with_available_list():
    cfg = AppConfig()
    cfg.backend.name = "nope"
    with pytest.raises(UnknownBackendError) as exc:
        create_backend(cfg)
    assert "mock" in exc.value.message


def test_registry_does_not_import_qwen3_module_for_mock():
    cfg = AppConfig()  # backend name defaults to "mock"
    create_backend(cfg)
    assert "tts_server.backends.qwen3" not in sys.modules


def test_create_mock_backend():
    backend = create_backend(AppConfig())
    assert backend.name == "mock"
