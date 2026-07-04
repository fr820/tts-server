import sys
from unittest.mock import MagicMock

import pytest

from tts_server.config import AppConfig
from tts_server.errors import BackendNotLoadedError, SynthesisError
from tts_server.models import TTSRequest


@pytest.fixture(autouse=True)
def _unload_qwen3_module():
    """Keep this test file's imports from leaking into sys.modules for other
    test files (e.g. test_registry.py asserts qwen3 is NOT imported when the
    mock backend is selected -- alphabetical test-file ordering would run
    this file first and permanently cache the module otherwise)."""
    yield
    sys.modules.pop("tts_server.backends.qwen3", None)


def make_config(**backend_overrides) -> AppConfig:
    cfg = AppConfig()
    cfg.backend.name = "qwen3"
    for key, value in backend_overrides.items():
        setattr(cfg.backend, key, value)
    return cfg


def test_capabilities_are_honest():
    from tts_server.backends.qwen3 import Qwen3TTSBackend

    caps = Qwen3TTSBackend(make_config()).capabilities
    assert caps.streaming_mode == "emulated"
    assert caps.supports_streaming_output is True
    assert caps.supports_streaming_input is False
    assert caps.supports_cuda is True


async def test_synthesize_before_load_raises():
    from tts_server.backends.qwen3 import Qwen3TTSBackend

    backend = Qwen3TTSBackend(make_config())
    with pytest.raises(BackendNotLoadedError):
        await backend.synthesize(TTSRequest(text="hi"))


async def test_load_without_torch_gives_helpful_error(monkeypatch):
    from tts_server.backends.qwen3 import Qwen3TTSBackend

    monkeypatch.setitem(sys.modules, "torch", None)  # simulate missing dep
    backend = Qwen3TTSBackend(make_config())
    with pytest.raises(SynthesisError) as exc:
        await backend.load()
    assert "uv sync --extra qwen3" in exc.value.message


async def test_synthesize_delegates_to_generate(monkeypatch):
    from tts_server.backends import qwen3

    backend = qwen3.Qwen3TTSBackend(make_config())
    backend._loaded = True
    backend._generate = MagicMock(return_value=(b"\x00\x01" * 100, 24000))
    result = await backend.synthesize(TTSRequest(text="hello", voice="vivian"))
    assert result.audio == b"\x00\x01" * 100
    assert result.sample_rate == 24000
    backend._generate.assert_called_once()
