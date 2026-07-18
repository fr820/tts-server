import asyncio
import sys
import threading
import time
import types
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


async def test_synthesize_serializes_concurrent_inference():
    """The `_infer_lock` must keep at most one _generate in flight at a time,
    so concurrent requests never interleave on the shared model. _generate runs
    in a worker thread (asyncio.to_thread), so overlap is detected with a
    threading lock, not an asyncio one."""
    from tts_server.backends import qwen3

    backend = qwen3.Qwen3TTSBackend(make_config())
    backend._loaded = True
    active = 0
    peak = 0
    guard = threading.Lock()

    def slow_generate(request):
        nonlocal active, peak
        with guard:
            active += 1
            peak = max(peak, active)
        time.sleep(0.03)
        with guard:
            active -= 1
        return (b"\x00\x01" * 100, 24000)

    backend._generate = slow_generate
    await asyncio.gather(
        *(backend.synthesize(TTSRequest(text="x")) for _ in range(6))
    )
    assert peak == 1, f"expected serialized inference, saw {peak} concurrent"


async def test_health_reports_peak_gpu_memory(monkeypatch):
    from tts_server.backends import qwen3

    fake_torch = types.ModuleType("torch")
    fake_torch.cuda = types.SimpleNamespace(
        is_available=lambda: True,
        memory_allocated=lambda: 1_000_000.0,
        max_memory_allocated=lambda: 2_000_000.0,
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)

    backend = qwen3.Qwen3TTSBackend(make_config())
    health = await backend.health()
    assert health.gpu_memory_mb == pytest.approx(1.0)
    assert health.gpu_memory_peak_mb == pytest.approx(2.0)
