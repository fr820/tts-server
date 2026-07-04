# TTS Inference Server Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A model-agnostic, plugin-based, streaming TTS inference server with OpenAI-compatible HTTP, ElevenLabs-style WebSocket, and native APIs, a MockBackend for GPU-free dev/CI, and a CUDA-first Qwen3-TTS adapter.

**Architecture:** Single-process asyncio FastAPI server. Blocking model inference runs in worker threads and feeds a bounded `asyncio.Queue`; one shared streaming pipeline normalizes all backends to `AsyncIterator[TTSChunk]` with cancellation, timeout, backpressure, and timing capture. Backends implement a small ABC and are resolved by a lazy-import registry.

**Tech Stack:** Python 3.12+, uv, FastAPI/Starlette, uvicorn, Pydantic v2, PyYAML, prometheus-client, pytest + pytest-asyncio + httpx; optional extra `qwen3` (torch, transformers, accelerate).

## Global Constraints

- Python `>=3.12`; all dependency/project management through `uv` (never pip directly).
- Package lives at `src/tts_server/`; distribution name `tts-server`, import name `tts_server`.
- asyncio-native: no blocking calls in request handlers; blocking inference goes through `asyncio.to_thread`.
- Base install must run with NO GPU and NO model downloads (MockBackend). Heavy deps only via `uv sync --extra qwen3`.
- **Honesty rule:** capability metadata must describe what the adapter actually does. Emulated streaming is labeled `"emulated"`. No feature claimed without an implementing test.
- No secrets, tokens, or full user text in logs (request text truncated to 80 chars).
- Every server startup creates a new log file under `logs/`.
- Default audio: `pcm_s16le`, 24000 Hz, mono, 16-bit.
- Config precedence: defaults → YAML (`TTS_CONFIG` env or `config.yaml`) → env vars (`TTS_BACKEND`, `TTS_HOST`, `TTS_PORT`, `TTS_DEVICE`).
- TDD for every task; commit at the end of every task; all tests must pass without GPU.
- Benchmark output must carry the backend name; mock results explicitly labeled — never fabricate numbers.

---

### Task 1: Project scaffolding + core types

**Files:**
- Create: `pyproject.toml`, `.gitignore`, `src/tts_server/__init__.py`, `src/tts_server/models.py`
- Test: `tests/test_models.py`

**Interfaces:**
- Consumes: nothing (first task)
- Produces: `tts_server.models` exporting `AudioFormat` (str enum: `PCM_S16LE="pcm_s16le"`, `WAV="wav"`), `TTSCapabilities`, `TTSRequest`, `TTSChunk`, `TTSResult`, `VoiceInfo`, `ModelInfo`, `BackendHealth` — exact fields shown below. All later tasks import from here.

- [ ] **Step 1: Scaffold project files**

`pyproject.toml`:

```toml
[project]
name = "tts-server"
version = "0.1.0"
description = "Model-agnostic, plugin-based, streaming TTS inference server"
requires-python = ">=3.12"
dependencies = [
    "fastapi>=0.115",
    "uvicorn[standard]>=0.30",
    "pydantic>=2.7",
    "pyyaml>=6.0",
    "prometheus-client>=0.20",
]

[project.optional-dependencies]
qwen3 = ["torch>=2.4", "transformers>=4.46", "accelerate>=1.0"]

[dependency-groups]
dev = ["pytest>=8", "pytest-asyncio>=0.24", "httpx>=0.27", "websockets>=13"]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["src/tts_server"]

[tool.pytest.ini_options]
asyncio_mode = "auto"
testpaths = ["tests"]
```

`.gitignore`:

```
__pycache__/
*.pyc
.venv/
logs/
benchmarks/results/
.pytest_cache/
dist/
```

`src/tts_server/__init__.py`:

```python
__version__ = "0.1.0"
```

Run: `uv sync` — expect it to resolve and create `uv.lock` and `.venv`.

- [ ] **Step 2: Write the failing test**

`tests/test_models.py`:

```python
from tts_server.models import (
    AudioFormat,
    BackendHealth,
    TTSCapabilities,
    TTSChunk,
    TTSRequest,
    TTSResult,
)


def test_audio_format_values():
    assert AudioFormat.PCM_S16LE.value == "pcm_s16le"
    assert AudioFormat.WAV.value == "wav"


def test_capabilities_defaults_are_honest():
    caps = TTSCapabilities()
    assert caps.supports_streaming_input is False
    assert caps.supports_streaming_output is False
    assert caps.streaming_mode == "none"
    assert caps.supports_cuda is False
    assert caps.supports_cpu is True
    assert AudioFormat.PCM_S16LE in caps.supported_audio_formats


def test_request_defaults_and_request_id():
    req = TTSRequest(text="hello")
    assert req.voice == "default"
    assert req.speed == 1.0
    assert req.sample_rate == 24000
    assert req.format == AudioFormat.PCM_S16LE
    assert len(req.request_id) == 32
    assert req.extra == {}
    # request_id is unique per request
    assert TTSRequest(text="a").request_id != TTSRequest(text="b").request_id


def test_chunk_and_result():
    chunk = TTSChunk(audio=b"\x00\x01", sample_rate=24000, sequence=3)
    assert chunk.is_final is False
    result = TTSResult(audio=b"\x00\x01", sample_rate=24000)
    assert result.format == AudioFormat.PCM_S16LE


def test_backend_health():
    h = BackendHealth(ok=True, loaded=True)
    assert h.detail is None and h.gpu_memory_mb is None
```

- [ ] **Step 3: Run test to verify it fails**

Run: `uv run pytest tests/test_models.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'tts_server.models'`

- [ ] **Step 4: Write the implementation**

`src/tts_server/models.py`:

```python
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
```

- [ ] **Step 5: Run test to verify it passes**

Run: `uv run pytest tests/test_models.py -v`
Expected: 5 passed

- [ ] **Step 6: Commit**

```bash
git add pyproject.toml uv.lock .gitignore src/ tests/
git commit -m "feat: project scaffolding and core data models"
```

---

### Task 2: Audio utilities

**Files:**
- Create: `src/tts_server/streaming/__init__.py` (empty), `src/tts_server/streaming/audio.py`
- Test: `tests/test_audio.py`

**Interfaces:**
- Consumes: nothing
- Produces:
  - `slice_pcm(pcm: bytes, sample_rate: int, chunk_ms: int = 100, sample_width: int = 2, channels: int = 1) -> list[bytes]`
  - `pcm_to_wav(pcm: bytes, sample_rate: int, channels: int = 1, sample_width: int = 2) -> bytes`
  - `encode_audio_b64(audio: bytes) -> str`
  - `pcm_duration_seconds(pcm: bytes, sample_rate: int, sample_width: int = 2, channels: int = 1) -> float`

- [ ] **Step 1: Write the failing test**

`tests/test_audio.py`:

```python
import base64
import io
import wave

from tts_server.streaming.audio import (
    encode_audio_b64,
    pcm_duration_seconds,
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


def test_pcm_duration_seconds():
    pcm = b"\x00" * 48000  # 1s of 24kHz mono s16le
    assert pcm_duration_seconds(pcm, 24000) == 1.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_audio.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Write the implementation**

Create empty `src/tts_server/streaming/__init__.py`, then `src/tts_server/streaming/audio.py`:

```python
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


def pcm_duration_seconds(
    pcm: bytes, sample_rate: int, sample_width: int = 2, channels: int = 1
) -> float:
    return len(pcm) / (sample_rate * sample_width * channels)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_audio.py -v`
Expected: 6 passed

- [ ] **Step 5: Commit**

```bash
git add src/tts_server/streaming/ tests/test_audio.py
git commit -m "feat: PCM slicing, WAV wrapping, and base64 audio utilities"
```

---

### Task 3: Errors + backend protocol with emulated-streaming default

**Files:**
- Create: `src/tts_server/errors.py`, `src/tts_server/backends/__init__.py` (empty), `src/tts_server/backends/base.py`
- Test: `tests/test_backend_base.py`

**Interfaces:**
- Consumes: `models.py` types; `slice_pcm` from Task 2.
- Produces:
  - `errors.TTSServerError(message, *, code: str, status_code: int)` with subclasses `UnsupportedFeatureError` (400, accepts `capabilities: TTSCapabilities | None`), `BackendNotLoadedError` (503), `SynthesisError` (500), `RequestTimeoutError` (504), `UnknownBackendError` (400).
  - `backends.base.TTSBackend` ABC: class attrs `name: str`, `capabilities: TTSCapabilities`; abstract `async load()`, `async synthesize(request) -> TTSResult`, `async close()`; concrete overridable `synthesize_stream(request) -> AsyncIterator[TTSChunk]` (emulated default), `async health() -> BackendHealth`, `list_voices() -> list[VoiceInfo]`, `model_info() -> ModelInfo`, plus `loaded: bool` property backed by `self._loaded`.

- [ ] **Step 1: Write the failing test**

`tests/test_backend_base.py`:

```python
import pytest

from tts_server.errors import (
    RequestTimeoutError,
    TTSServerError,
    UnsupportedFeatureError,
)
from tts_server.backends.base import TTSBackend
from tts_server.models import (
    BackendHealth,
    TTSCapabilities,
    TTSRequest,
    TTSResult,
)


class DummyBackend(TTSBackend):
    """Non-streaming backend used to exercise the emulated-streaming default."""

    name = "dummy"
    capabilities = TTSCapabilities()

    async def load(self) -> None:
        self._loaded = True

    async def synthesize(self, request: TTSRequest) -> TTSResult:
        # 0.25s of silence at the requested rate
        pcm = b"\x00\x00" * (request.sample_rate // 4)
        return TTSResult(audio=pcm, sample_rate=request.sample_rate)

    async def close(self) -> None:
        self._loaded = False


async def test_emulated_stream_covers_full_audio_and_marks_final():
    backend = DummyBackend()
    await backend.load()
    req = TTSRequest(text="hi", sample_rate=24000)
    chunks = [c async for c in backend.synthesize_stream(req)]
    assert len(chunks) >= 2
    assert chunks[-1].is_final is True
    assert all(not c.is_final for c in chunks[:-1])
    assert [c.sequence for c in chunks] == list(range(len(chunks)))
    joined = b"".join(c.audio for c in chunks)
    assert joined == (await backend.synthesize(req)).audio


async def test_emulated_stream_empty_audio_yields_single_final_chunk():
    class EmptyBackend(DummyBackend):
        async def synthesize(self, request):
            return TTSResult(audio=b"", sample_rate=request.sample_rate)

    backend = EmptyBackend()
    chunks = [c async for c in backend.synthesize_stream(TTSRequest(text=""))]
    assert len(chunks) == 1
    assert chunks[0].is_final is True and chunks[0].audio == b""


async def test_default_health_and_loaded_flag():
    backend = DummyBackend()
    assert backend.loaded is False
    health = await backend.health()
    assert isinstance(health, BackendHealth)
    assert health.loaded is False
    await backend.load()
    assert (await backend.health()).loaded is True


def test_error_hierarchy_and_status_codes():
    err = UnsupportedFeatureError("no wav", capabilities=TTSCapabilities())
    assert isinstance(err, TTSServerError)
    assert err.status_code == 400
    assert err.code == "unsupported_feature"
    assert err.capabilities is not None
    assert RequestTimeoutError("slow").status_code == 504
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_backend_base.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'tts_server.errors'`

- [ ] **Step 3: Write the implementations**

`src/tts_server/errors.py`:

```python
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
```

Create empty `src/tts_server/backends/__init__.py`, then `src/tts_server/backends/base.py`:

```python
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
    capabilities: TTSCapabilities = TTSCapabilities()

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
        capabilities.streaming_mode = "native".
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_backend_base.py -v`
Expected: 4 passed

- [ ] **Step 5: Commit**

```bash
git add src/tts_server/errors.py src/tts_server/backends/ tests/test_backend_base.py
git commit -m "feat: error hierarchy and TTSBackend protocol with emulated streaming default"
```

---

### Task 4: Config loading (defaults → YAML → env)

**Files:**
- Create: `src/tts_server/config.py`
- Test: `tests/test_config.py`

**Interfaces:**
- Consumes: `AudioFormat` from models.
- Produces:
  - `ServerConfig(host="0.0.0.0", port=8000, request_timeout_s=120.0, log_dir="logs")`
  - `BackendConfig(name="mock", device="auto", dtype="bf16", model_path: str | None = None, compile=False, warmup=True, options: dict[str, Any] = {})`
  - `AudioConfig(default_format=AudioFormat.PCM_S16LE, sample_rate=24000, channels=1)`
  - `AppConfig(server: ServerConfig, backend: BackendConfig, audio: AudioConfig)`
  - `load_config(path: str | None = None) -> AppConfig` — path arg > `TTS_CONFIG` env > `./config.yaml` if it exists > pure defaults; then env overrides `TTS_BACKEND`, `TTS_HOST`, `TTS_PORT`, `TTS_DEVICE`.

- [ ] **Step 1: Write the failing test**

`tests/test_config.py`:

```python
import pytest

from tts_server.config import AppConfig, load_config


def test_defaults_without_yaml_or_env(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)  # no config.yaml here
    for var in ("TTS_CONFIG", "TTS_BACKEND", "TTS_HOST", "TTS_PORT", "TTS_DEVICE"):
        monkeypatch.delenv(var, raising=False)
    cfg = load_config()
    assert cfg.backend.name == "mock"
    assert cfg.server.port == 8000
    assert cfg.audio.sample_rate == 24000


def test_yaml_file_overrides_defaults(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("TTS_BACKEND", raising=False)
    (tmp_path / "myconf.yaml").write_text(
        "server:\n  port: 9000\nbackend:\n  name: qwen3\n  device: cuda\n"
    )
    cfg = load_config(str(tmp_path / "myconf.yaml"))
    assert cfg.server.port == 9000
    assert cfg.backend.name == "qwen3"
    assert cfg.backend.device == "cuda"
    assert cfg.audio.channels == 1  # untouched section keeps defaults


def test_env_overrides_yaml(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "config.yaml").write_text("backend:\n  name: qwen3\n")
    monkeypatch.setenv("TTS_BACKEND", "mock")
    monkeypatch.setenv("TTS_PORT", "8123")
    cfg = load_config()
    assert cfg.backend.name == "mock"
    assert cfg.server.port == 8123


def test_tts_config_env_points_to_file(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("TTS_BACKEND", raising=False)
    p = tmp_path / "elsewhere.yaml"
    p.write_text("server:\n  host: 127.0.0.1\n")
    monkeypatch.setenv("TTS_CONFIG", str(p))
    assert load_config().server.host == "127.0.0.1"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_config.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Write the implementation**

`src/tts_server/config.py`:

```python
"""Configuration: defaults -> YAML file -> environment variable overrides."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field

from tts_server.models import AudioFormat


class ServerConfig(BaseModel):
    host: str = "0.0.0.0"
    port: int = 8000
    request_timeout_s: float = 120.0
    log_dir: str = "logs"


class BackendConfig(BaseModel):
    name: str = "mock"
    device: str = "auto"
    dtype: str = "bf16"
    model_path: str | None = None
    compile: bool = False
    warmup: bool = True
    options: dict[str, Any] = Field(default_factory=dict)


class AudioConfig(BaseModel):
    default_format: AudioFormat = AudioFormat.PCM_S16LE
    sample_rate: int = 24000
    channels: int = 1


class AppConfig(BaseModel):
    server: ServerConfig = Field(default_factory=ServerConfig)
    backend: BackendConfig = Field(default_factory=BackendConfig)
    audio: AudioConfig = Field(default_factory=AudioConfig)


def _yaml_path(explicit: str | None) -> Path | None:
    if explicit:
        return Path(explicit)
    env = os.environ.get("TTS_CONFIG")
    if env:
        return Path(env)
    default = Path("config.yaml")
    return default if default.exists() else None


def load_config(path: str | None = None) -> AppConfig:
    data: dict[str, Any] = {}
    yaml_path = _yaml_path(path)
    if yaml_path is not None:
        data = yaml.safe_load(yaml_path.read_text()) or {}

    cfg = AppConfig.model_validate(data)

    if backend := os.environ.get("TTS_BACKEND"):
        cfg.backend.name = backend
    if host := os.environ.get("TTS_HOST"):
        cfg.server.host = host
    if port := os.environ.get("TTS_PORT"):
        cfg.server.port = int(port)
    if device := os.environ.get("TTS_DEVICE"):
        cfg.backend.device = device
    return cfg
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_config.py -v`
Expected: 4 passed

- [ ] **Step 5: Commit**

```bash
git add src/tts_server/config.py tests/test_config.py
git commit -m "feat: layered config loading (defaults, YAML, env overrides)"
```

---

### Task 5: Backend registry with lazy imports

**Files:**
- Create: `src/tts_server/backends/registry.py`
- Test: `tests/test_registry.py`

**Interfaces:**
- Consumes: `AppConfig` (Task 4), `TTSBackend` (Task 3), `UnknownBackendError` (Task 3).
- Produces:
  - `available_backends() -> list[str]` → `["mock", "qwen3"]` sorted
  - `create_backend(config: AppConfig) -> TTSBackend` — resolves `config.backend.name` via lazy `importlib` import; unknown name raises `UnknownBackendError` listing available names. Backend classes take `(config: AppConfig)` as their only constructor arg.

- [ ] **Step 1: Write the failing test**

`tests/test_registry.py`:

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_registry.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'tts_server.backends.registry'`

- [ ] **Step 3: Write the implementation**

`src/tts_server/backends/registry.py`:

```python
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
```

Note: this fails until MockBackend exists — so Step 4 here only passes after writing a minimal `src/tts_server/backends/mock.py` stub in the same step. Write the REAL MockBackend now as part of Task 6? No — Task 6 owns it. Instead, write this minimal placeholder-free stub which Task 6 replaces with the full implementation (the stub is itself a working backend):

`src/tts_server/backends/mock.py` (minimal working version; Task 6 extends it):

```python
from __future__ import annotations

from tts_server.backends.base import TTSBackend
from tts_server.config import AppConfig
from tts_server.models import TTSCapabilities, TTSRequest, TTSResult


class MockBackend(TTSBackend):
    name = "mock"
    capabilities = TTSCapabilities()

    def __init__(self, config: AppConfig) -> None:
        super().__init__()
        self._config = config

    async def load(self) -> None:
        self._loaded = True

    async def synthesize(self, request: TTSRequest) -> TTSResult:
        return TTSResult(audio=b"", sample_rate=request.sample_rate)

    async def close(self) -> None:
        self._loaded = False
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_registry.py -v`
Expected: 4 passed

- [ ] **Step 5: Commit**

```bash
git add src/tts_server/backends/registry.py src/tts_server/backends/mock.py tests/test_registry.py
git commit -m "feat: lazy-import backend registry"
```

---

### Task 6: MockBackend (deterministic, natively streaming)

**Files:**
- Modify: `src/tts_server/backends/mock.py` (replace the Task 5 stub entirely)
- Test: `tests/test_mock_backend.py`

**Interfaces:**
- Consumes: `TTSBackend`, models, `AppConfig` (reads `config.backend.options` keys `first_chunk_delay_ms` default 20, `chunk_interval_ms` default 10, `seconds_per_char` default 0.06).
- Produces: `MockBackend(config: AppConfig)` — deterministic sine PCM keyed on text hash; duration `min(max(len(text) * seconds_per_char, 0.2), 30.0) / speed`; native streaming output with real pacing; capabilities: `supports_streaming_input=True`, `supports_streaming_output=True`, `streaming_mode="native"`, `supports_cuda=False`, `supports_cpu=True`, `supports_emotion_or_style_control=False`.

- [ ] **Step 1: Write the failing test**

`tests/test_mock_backend.py`:

```python
from tts_server.config import AppConfig
from tts_server.backends.mock import MockBackend
from tts_server.models import TTSRequest


def make_backend(**options) -> MockBackend:
    cfg = AppConfig()
    cfg.backend.options.update(options)
    return MockBackend(cfg)


async def test_synthesis_is_deterministic():
    backend = make_backend()
    await backend.load()
    req = TTSRequest(text="hello world", request_id="a" * 32)
    r1 = await backend.synthesize(req)
    r2 = await backend.synthesize(TTSRequest(text="hello world", request_id="b" * 32))
    assert r1.audio == r2.audio
    assert len(r1.audio) > 0
    assert len(r1.audio) % 2 == 0  # whole s16le samples


async def test_different_text_gives_different_audio():
    backend = make_backend()
    await backend.load()
    a = await backend.synthesize(TTSRequest(text="aaaaaaaaaaaa"))
    b = await backend.synthesize(TTSRequest(text="bbbbbbbbbbbb"))
    assert a.audio != b.audio


async def test_duration_scales_with_text_and_speed():
    backend = make_backend()
    await backend.load()
    short = await backend.synthesize(TTSRequest(text="hi" * 20))
    long = await backend.synthesize(TTSRequest(text="hi" * 60))
    fast = await backend.synthesize(TTSRequest(text="hi" * 60, speed=2.0))
    assert len(long.audio) > len(short.audio)
    assert len(fast.audio) < len(long.audio)


async def test_native_stream_matches_full_synthesis():
    backend = make_backend(first_chunk_delay_ms=0, chunk_interval_ms=0)
    await backend.load()
    req = TTSRequest(text="stream me", request_id="c" * 32)
    chunks = [c async for c in backend.synthesize_stream(req)]
    assert chunks[-1].is_final
    full = await backend.synthesize(TTSRequest(text="stream me"))
    assert b"".join(c.audio for c in chunks) == full.audio


def test_capabilities_report_native_streaming():
    caps = make_backend().capabilities
    assert caps.streaming_mode == "native"
    assert caps.supports_streaming_output is True
    assert caps.supports_streaming_input is True
    assert caps.supports_cuda is False
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_mock_backend.py -v`
Expected: FAIL (stub returns empty audio → determinism/length assertions fail)

- [ ] **Step 3: Write the implementation**

Replace `src/tts_server/backends/mock.py` with:

```python
"""Deterministic synthetic-audio backend for development, tests, CI,
and benchmark pipeline validation. No GPU, no downloads."""

from __future__ import annotations

import array
import asyncio
import hashlib
import math
from collections.abc import AsyncIterator

from tts_server.backends.base import TTSBackend
from tts_server.config import AppConfig
from tts_server.models import (
    TTSCapabilities,
    TTSChunk,
    TTSRequest,
    TTSResult,
)
from tts_server.streaming.audio import slice_pcm


class MockBackend(TTSBackend):
    name = "mock"
    capabilities = TTSCapabilities(
        supports_streaming_input=True,
        supports_streaming_output=True,
        streaming_mode="native",
        supports_cuda=False,
        supports_cpu=True,
    )

    def __init__(self, config: AppConfig) -> None:
        super().__init__()
        opts = config.backend.options
        self._first_chunk_delay_s = float(opts.get("first_chunk_delay_ms", 20)) / 1000
        self._chunk_interval_s = float(opts.get("chunk_interval_ms", 10)) / 1000
        self._seconds_per_char = float(opts.get("seconds_per_char", 0.06))

    async def load(self) -> None:
        self._loaded = True

    async def close(self) -> None:
        self._loaded = False

    def _generate_pcm(self, request: TTSRequest) -> bytes:
        duration_s = (
            min(max(len(request.text) * self._seconds_per_char, 0.2), 30.0)
            / request.speed
        )
        n_samples = int(duration_s * request.sample_rate)
        digest = hashlib.sha256(request.text.encode()).hexdigest()
        freq = 200.0 + (int(digest, 16) % 200)
        step = 2.0 * math.pi * freq / request.sample_rate
        samples = array.array(
            "h", (int(6553 * math.sin(step * i)) for i in range(n_samples))
        )
        return samples.tobytes()

    async def synthesize(self, request: TTSRequest) -> TTSResult:
        pcm = await asyncio.to_thread(self._generate_pcm, request)
        return TTSResult(audio=pcm, sample_rate=request.sample_rate)

    async def synthesize_stream(
        self, request: TTSRequest
    ) -> AsyncIterator[TTSChunk]:
        pcm = await asyncio.to_thread(self._generate_pcm, request)
        pieces = slice_pcm(pcm, request.sample_rate) or [b""]
        await asyncio.sleep(self._first_chunk_delay_s)
        for i, piece in enumerate(pieces):
            if i > 0:
                await asyncio.sleep(self._chunk_interval_s)
            yield TTSChunk(
                audio=piece,
                sample_rate=request.sample_rate,
                is_final=(i == len(pieces) - 1),
                sequence=i,
            )
```

- [ ] **Step 4: Run all tests to verify pass (registry tests still green)**

Run: `uv run pytest -v`
Expected: all tests pass

- [ ] **Step 5: Commit**

```bash
git add src/tts_server/backends/mock.py tests/test_mock_backend.py
git commit -m "feat: deterministic MockBackend with native paced streaming"
```

---

### Task 7: Metrics module

**Files:**
- Create: `src/tts_server/metrics/__init__.py`
- Test: `tests/test_metrics.py`

**Interfaces:**
- Consumes: nothing internal (prometheus-client).
- Produces module-level collectors and helpers used by the pipeline (Task 8) and health API (Task 10):
  - `REQUESTS = Counter("tts_requests_total", ..., ["backend", "api"])`
  - `FAILURES = Counter("tts_failures_total", ..., ["backend", "api"])`
  - `ACTIVE_SESSIONS = Gauge("tts_active_sessions", ...)`
  - `TTFA_SECONDS = Histogram("tts_ttfa_seconds", ..., ["backend"])`
  - `LATENCY_SECONDS = Histogram("tts_latency_seconds", ..., ["backend"])`
  - `RTF = Histogram("tts_rtf", ..., ["backend"])`
  - `GPU_MEMORY_MB = Gauge("tts_gpu_memory_mb", ...)`
  - `render_metrics() -> tuple[bytes, str]` returning `(payload, content_type)`; refreshes GPU gauge from torch if importable and CUDA available, else leaves it at 0.

- [ ] **Step 1: Write the failing test**

`tests/test_metrics.py`:

```python
from tts_server import metrics


def test_counters_and_render():
    metrics.REQUESTS.labels(backend="mock", api="openai").inc()
    metrics.TTFA_SECONDS.labels(backend="mock").observe(0.05)
    metrics.RTF.labels(backend="mock").observe(0.3)
    payload, content_type = metrics.render_metrics()
    text = payload.decode()
    assert "tts_requests_total" in text
    assert "tts_ttfa_seconds" in text
    assert "tts_rtf" in text
    assert "tts_gpu_memory_mb" in text
    assert content_type.startswith("text/plain")


def test_active_sessions_gauge():
    metrics.ACTIVE_SESSIONS.inc()
    metrics.ACTIVE_SESSIONS.dec()
    payload, _ = metrics.render_metrics()
    assert "tts_active_sessions" in payload.decode()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_metrics.py -v`
Expected: FAIL with import error

- [ ] **Step 3: Write the implementation**

`src/tts_server/metrics/__init__.py`:

```python
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_metrics.py -v`
Expected: 2 passed

- [ ] **Step 5: Commit**

```bash
git add src/tts_server/metrics/ tests/test_metrics.py
git commit -m "feat: prometheus metrics module"
```

---

### Task 8: Streaming pipeline (cancellation, timeout, backpressure, timing)

**Files:**
- Create: `src/tts_server/streaming/pipeline.py`
- Test: `tests/test_pipeline.py`

**Interfaces:**
- Consumes: `TTSBackend.synthesize_stream`, `TTSChunk`, `RequestTimeoutError`, metrics collectors (Task 7), `pcm_duration_seconds` (Task 2).
- Produces: `run_stream(backend: TTSBackend, request: TTSRequest, *, timeout_s: float, api: str = "native", max_queue: int = 32) -> AsyncIterator[TTSChunk]`
  - Producer task drains `backend.synthesize_stream` into a bounded queue (backpressure).
  - `timeout_s` is a per-chunk idle timeout; exceeding it raises `RequestTimeoutError`.
  - Closing the generator (client disconnect) cancels the producer task.
  - Records REQUESTS on entry; TTFA on first chunk; LATENCY, RTF on completion; FAILURES on error. ACTIVE_SESSIONS incremented for the stream's lifetime.

- [ ] **Step 1: Write the failing test**

`tests/test_pipeline.py`:

```python
import asyncio

import pytest

from tts_server.config import AppConfig
from tts_server.errors import RequestTimeoutError, SynthesisError
from tts_server.backends.mock import MockBackend
from tts_server.models import TTSCapabilities, TTSRequest, TTSResult
from tts_server.streaming.pipeline import run_stream
from tts_server.backends.base import TTSBackend


def mock_backend(**options) -> MockBackend:
    cfg = AppConfig()
    cfg.backend.options.update(options)
    return MockBackend(cfg)


async def test_passes_through_all_chunks():
    backend = mock_backend(first_chunk_delay_ms=0, chunk_interval_ms=0)
    await backend.load()
    req = TTSRequest(text="hello there")
    chunks = [c async for c in run_stream(backend, req, timeout_s=5.0)]
    assert chunks[-1].is_final
    full = await backend.synthesize(TTSRequest(text="hello there"))
    assert b"".join(c.audio for c in chunks) == full.audio


async def test_idle_timeout_raises():
    class StallingBackend(TTSBackend):
        name = "stall"
        capabilities = TTSCapabilities()

        async def load(self): self._loaded = True
        async def close(self): pass
        async def synthesize(self, request):
            return TTSResult(audio=b"", sample_rate=24000)

        async def synthesize_stream(self, request):
            await asyncio.sleep(60)
            yield  # pragma: no cover

    backend = StallingBackend()
    with pytest.raises(RequestTimeoutError):
        async for _ in run_stream(backend, TTSRequest(text="x"), timeout_s=0.05):
            pass


async def test_backend_exception_propagates():
    class BrokenBackend(TTSBackend):
        name = "broken"
        capabilities = TTSCapabilities()

        async def load(self): self._loaded = True
        async def close(self): pass
        async def synthesize(self, request):
            raise SynthesisError("boom")

    backend = BrokenBackend()
    with pytest.raises(SynthesisError):
        async for _ in run_stream(backend, TTSRequest(text="x"), timeout_s=5.0):
            pass


async def test_closing_generator_cancels_producer():
    produced = []

    class SlowBackend(TTSBackend):
        name = "slow"
        capabilities = TTSCapabilities()

        async def load(self): self._loaded = True
        async def close(self): pass
        async def synthesize(self, request):
            return TTSResult(audio=b"", sample_rate=24000)

        async def synthesize_stream(self, request):
            from tts_server.models import TTSChunk
            for i in range(1000):
                produced.append(i)
                yield TTSChunk(audio=b"\x00\x00", sample_rate=24000, sequence=i)
                await asyncio.sleep(0.001)

    backend = SlowBackend()
    gen = run_stream(backend, TTSRequest(text="x"), timeout_s=5.0)
    async for chunk in gen:
        break  # client disconnects after first chunk
    await gen.aclose()
    await asyncio.sleep(0.05)
    count_after_close = len(produced)
    await asyncio.sleep(0.05)
    assert len(produced) == count_after_close  # producer stopped
    assert count_after_close < 1000
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_pipeline.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Write the implementation**

`src/tts_server/streaming/pipeline.py`:

```python
"""Shared streaming pipeline: wraps any backend stream with backpressure,
idle timeout, cancellation on client disconnect, and metrics capture."""

from __future__ import annotations

import asyncio
import time
from collections.abc import AsyncIterator

from tts_server import metrics
from tts_server.backends.base import TTSBackend
from tts_server.errors import RequestTimeoutError
from tts_server.models import TTSChunk, TTSRequest
from tts_server.streaming.audio import pcm_duration_seconds

_END = None  # queue sentinel


async def run_stream(
    backend: TTSBackend,
    request: TTSRequest,
    *,
    timeout_s: float,
    api: str = "native",
    max_queue: int = 32,
) -> AsyncIterator[TTSChunk]:
    queue: asyncio.Queue[TTSChunk | Exception | None] = asyncio.Queue(
        maxsize=max_queue
    )

    async def _produce() -> None:
        try:
            async for chunk in backend.synthesize_stream(request):
                await queue.put(chunk)
            await queue.put(_END)
        except Exception as exc:  # forwarded to the consumer
            await queue.put(exc)

    producer = asyncio.create_task(_produce())
    metrics.REQUESTS.labels(backend=backend.name, api=api).inc()
    metrics.ACTIVE_SESSIONS.inc()
    start = time.perf_counter()
    first_chunk_at: float | None = None
    audio_bytes = 0
    failed = False
    try:
        while True:
            try:
                item = await asyncio.wait_for(queue.get(), timeout=timeout_s)
            except TimeoutError:
                raise RequestTimeoutError(
                    f"no audio chunk within {timeout_s}s (request {request.request_id})"
                ) from None
            if item is _END:
                break
            if isinstance(item, Exception):
                raise item
            if first_chunk_at is None:
                first_chunk_at = time.perf_counter()
                metrics.TTFA_SECONDS.labels(backend=backend.name).observe(
                    first_chunk_at - start
                )
            audio_bytes += len(item.audio)
            yield item
    except BaseException:
        failed = True
        raise
    finally:
        producer.cancel()
        metrics.ACTIVE_SESSIONS.dec()
        elapsed = time.perf_counter() - start
        if failed:
            metrics.FAILURES.labels(backend=backend.name, api=api).inc()
        else:
            metrics.LATENCY_SECONDS.labels(backend=backend.name).observe(elapsed)
            audio_s = pcm_duration_seconds(
                b"\x00" * audio_bytes, request.sample_rate
            )
            if audio_s > 0:
                metrics.RTF.labels(backend=backend.name).observe(elapsed / audio_s)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_pipeline.py -v`
Expected: 4 passed

- [ ] **Step 5: Commit**

```bash
git add src/tts_server/streaming/pipeline.py tests/test_pipeline.py
git commit -m "feat: streaming pipeline with backpressure, timeout, cancellation, metrics"
```

---

### Task 9: Structured logging with request IDs

**Files:**
- Create: `src/tts_server/logging_config.py`
- Test: `tests/test_logging.py`

**Interfaces:**
- Consumes: nothing internal.
- Produces:
  - `request_id_var: ContextVar[str]` (default `"-"`)
  - `setup_logging(log_dir: str = "logs") -> Path` — creates dir, adds a JSON-lines file handler `logs/server-<YYYYmmdd-HHMMSS>.log` (new file per startup) and a console handler to the `tts_server` logger; returns the log file path. Idempotent per process (clears previous handlers).
  - `truncate_text(s: str, limit: int = 80) -> str` — for logging user text safely.
  - JSON records contain: `time`, `level`, `logger`, `message`, `request_id`.

- [ ] **Step 1: Write the failing test**

`tests/test_logging.py`:

```python
import json
import logging

from tts_server.logging_config import request_id_var, setup_logging, truncate_text


def test_creates_log_file_and_writes_json(tmp_path):
    log_file = setup_logging(str(tmp_path / "logs"))
    assert log_file.parent.name == "logs"
    logger = logging.getLogger("tts_server.test")
    request_id_var.set("req-123")
    logger.info("hello %s", "world")
    for handler in logging.getLogger("tts_server").handlers:
        handler.flush()
    record = json.loads(log_file.read_text().strip().splitlines()[-1])
    assert record["message"] == "hello world"
    assert record["request_id"] == "req-123"
    assert record["level"] == "INFO"


def test_new_file_per_setup(tmp_path):
    f1 = setup_logging(str(tmp_path / "logs"))
    import time

    time.sleep(1.1)  # filename has second granularity
    f2 = setup_logging(str(tmp_path / "logs"))
    assert f1 != f2


def test_truncate_text():
    assert truncate_text("short") == "short"
    long = "x" * 200
    out = truncate_text(long)
    assert len(out) <= 84 and out.endswith("...")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_logging.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Write the implementation**

`src/tts_server/logging_config.py`:

```python
"""Structured JSON logging. One log file per server startup; every record
carries the current request ID from a ContextVar. Never log secrets or
full user text — use truncate_text for text fields."""

from __future__ import annotations

import json
import logging
from contextvars import ContextVar
from datetime import datetime, timezone
from pathlib import Path

request_id_var: ContextVar[str] = ContextVar("request_id", default="-")


def truncate_text(s: str, limit: int = 80) -> str:
    return s if len(s) <= limit else s[:limit] + "..."


class _JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        return json.dumps(
            {
                "time": datetime.now(timezone.utc).isoformat(),
                "level": record.levelname,
                "logger": record.name,
                "message": record.getMessage(),
                "request_id": request_id_var.get(),
            }
        )


def setup_logging(log_dir: str = "logs") -> Path:
    directory = Path(log_dir)
    directory.mkdir(parents=True, exist_ok=True)
    log_file = directory / f"server-{datetime.now():%Y%m%d-%H%M%S}.log"

    logger = logging.getLogger("tts_server")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(_JsonFormatter())
    logger.addHandler(file_handler)

    console = logging.StreamHandler()
    console.setFormatter(
        logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
    )
    logger.addHandler(console)
    return log_file
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_logging.py -v`
Expected: 3 passed

- [ ] **Step 5: Commit**

```bash
git add src/tts_server/logging_config.py tests/test_logging.py
git commit -m "feat: structured JSON logging with per-startup files and request IDs"
```

---

### Task 10: App factory, lifespan, request-ID middleware, health + metrics endpoints

**Files:**
- Create: `src/tts_server/main.py`, `src/tts_server/api/__init__.py` (empty), `src/tts_server/api/health.py`
- Test: `tests/test_app.py`, `tests/conftest.py`

**Interfaces:**
- Consumes: `load_config`/`AppConfig`, `create_backend`, `setup_logging`/`request_id_var`, `render_metrics`, `TTSServerError`.
- Produces:
  - `create_app(config: AppConfig | None = None) -> FastAPI` — lifespan loads backend once (`await backend.load()`, optional warmup synthesis of `"warmup"` when `config.backend.warmup`), stores `app.state.backend` and `app.state.config`; teardown `await backend.close()`.
  - `app = create_app()` module-level for `uvicorn tts_server.main:app`.
  - Middleware: sets `request_id_var` from `X-Request-ID` header or new uuid4 hex; echoes it back as `X-Request-ID` response header.
  - Exception handler mapping `TTSServerError` → JSON `{"error": {"code", "message", "capabilities"?}}` with the error's `status_code`.
  - `GET /healthz` → `{"status": "ok"|"degraded", "backend": <BackendHealth>, "backend_name": str}`.
  - `GET /metrics` → Prometheus text.
  - Shared test fixture `client` (TestClient over `create_app` with mock backend, zero delays).

- [ ] **Step 1: Write the shared fixture and failing test**

`tests/conftest.py`:

```python
import pytest
from fastapi.testclient import TestClient

from tts_server.config import AppConfig


@pytest.fixture
def app_config() -> AppConfig:
    cfg = AppConfig()
    cfg.backend.name = "mock"
    cfg.backend.warmup = False
    cfg.backend.options.update({"first_chunk_delay_ms": 0, "chunk_interval_ms": 0})
    cfg.server.log_dir = "logs"
    return cfg


@pytest.fixture
def client(app_config, tmp_path):
    app_config.server.log_dir = str(tmp_path / "logs")
    from tts_server.main import create_app

    with TestClient(create_app(app_config)) as client:
        yield client
```

`tests/test_app.py`:

```python
def test_healthz_reports_backend(client):
    resp = client.get("/healthz")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["backend_name"] == "mock"
    assert body["backend"]["loaded"] is True


def test_metrics_endpoint(client):
    resp = client.get("/metrics")
    assert resp.status_code == 200
    assert "tts_requests_total" in resp.text


def test_request_id_echoed(client):
    resp = client.get("/healthz", headers={"X-Request-ID": "trace-me"})
    assert resp.headers["x-request-id"] == "trace-me"


def test_request_id_generated_when_absent(client):
    resp = client.get("/healthz")
    assert len(resp.headers["x-request-id"]) == 32
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_app.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'tts_server.main'`

- [ ] **Step 3: Write the implementation**

`src/tts_server/api/health.py`:

```python
from __future__ import annotations

from fastapi import APIRouter, Request, Response

from tts_server.metrics import render_metrics

router = APIRouter()


@router.get("/healthz")
async def healthz(request: Request):
    backend = request.app.state.backend
    health = await backend.health()
    return {
        "status": "ok" if health.ok and health.loaded else "degraded",
        "backend_name": backend.name,
        "backend": health.model_dump(),
    }


@router.get("/metrics")
async def metrics_endpoint():
    payload, content_type = render_metrics()
    return Response(content=payload, media_type=content_type)
```

`src/tts_server/main.py`:

```python
"""Application factory: config, logging, backend lifecycle, middleware,
error handling, and router registration."""

from __future__ import annotations

import logging
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from tts_server.api import health
from tts_server.backends.registry import create_backend
from tts_server.config import AppConfig, load_config
from tts_server.errors import TTSServerError, UnsupportedFeatureError
from tts_server.logging_config import request_id_var, setup_logging
from tts_server.models import TTSRequest

logger = logging.getLogger("tts_server.main")


def create_app(config: AppConfig | None = None) -> FastAPI:
    cfg = config or load_config()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        log_file = setup_logging(cfg.server.log_dir)
        logger.info("starting; backend=%s log_file=%s", cfg.backend.name, log_file)
        backend = create_backend(cfg)
        await backend.load()
        if cfg.backend.warmup:
            await backend.synthesize(
                TTSRequest(text="warmup", sample_rate=cfg.audio.sample_rate)
            )
            logger.info("warmup complete")
        app.state.backend = backend
        app.state.config = cfg
        try:
            yield
        finally:
            await backend.close()
            logger.info("backend closed; shutdown complete")

    app = FastAPI(title="tts-server", lifespan=lifespan)

    @app.middleware("http")
    async def request_id_middleware(request: Request, call_next):
        request_id = request.headers.get("X-Request-ID") or uuid.uuid4().hex
        request_id_var.set(request_id)
        response = await call_next(request)
        response.headers["X-Request-ID"] = request_id
        return response

    @app.exception_handler(TTSServerError)
    async def tts_error_handler(request: Request, exc: TTSServerError):
        body: dict = {"error": {"code": exc.code, "message": exc.message}}
        if isinstance(exc, UnsupportedFeatureError) and exc.capabilities:
            body["error"]["capabilities"] = exc.capabilities.model_dump()
        return JSONResponse(status_code=exc.status_code, content=body)

    app.include_router(health.router)
    return app


app = create_app()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_app.py -v`
Expected: 4 passed. Also sanity-check the server boots: `uv run uvicorn tts_server.main:app --port 8000 &`, `curl -s localhost:8000/healthz`, then kill it.

- [ ] **Step 5: Commit**

```bash
git add src/tts_server/main.py src/tts_server/api/ tests/conftest.py tests/test_app.py
git commit -m "feat: app factory with backend lifecycle, request IDs, health and metrics endpoints"
```

---

### Task 11: OpenAI-compatible API (`/v1/audio/speech`, `/v1/models`)

**Files:**
- Create: `src/tts_server/api/openai_compat.py`
- Modify: `src/tts_server/main.py` (add `app.include_router(openai_compat.router)` next to the health router include, and the matching import)
- Test: `tests/test_openai_api.py`

**Interfaces:**
- Consumes: `run_stream` (Task 8), `pcm_to_wav` (Task 2), backend from `request.app.state.backend`, config timeout, `UnsupportedFeatureError`.
- Produces:
  - `POST /v1/audio/speech` — body `{model?, input, voice="default", response_format="pcm", speed=1.0, instructions?}`. `response_format="pcm"` → chunked `StreamingResponse` of raw s16le (media type `application/octet-stream`); `"wav"` → full synthesis wrapped in WAV (`audio/wav`). Unknown `response_format` or format not in capabilities → 400 via `UnsupportedFeatureError` carrying capabilities. `model` omitted → active backend.
  - `GET /v1/models` — `{"object": "list", "data": [{"id": <backend model id>, "object": "model", "owned_by": "tts-server"}]}`.

- [ ] **Step 1: Write the failing test**

`tests/test_openai_api.py`:

```python
def test_speech_pcm_streams_audio(client):
    resp = client.post(
        "/v1/audio/speech",
        json={"input": "Hello, welcome to our voice agent demo.", "voice": "default",
              "response_format": "pcm", "speed": 1.0},
    )
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("application/octet-stream")
    assert len(resp.content) > 1000
    assert len(resp.content) % 2 == 0


def test_speech_wav(client):
    resp = client.post(
        "/v1/audio/speech", json={"input": "hello", "response_format": "wav"}
    )
    assert resp.status_code == 200
    assert resp.headers["content-type"] == "audio/wav"
    assert resp.content[:4] == b"RIFF"


def test_unsupported_format_returns_400_with_capabilities(client):
    resp = client.post(
        "/v1/audio/speech", json={"input": "hi", "response_format": "mp3"}
    )
    assert resp.status_code == 400
    err = resp.json()["error"]
    assert err["code"] == "unsupported_feature"
    assert "supported_audio_formats" in err["capabilities"]


def test_missing_input_is_422(client):
    assert client.post("/v1/audio/speech", json={}).status_code == 422


def test_models_listing(client):
    resp = client.get("/v1/models")
    assert resp.status_code == 200
    body = resp.json()
    assert body["object"] == "list"
    assert body["data"][0]["id"] == "mock"
    assert body["data"][0]["object"] == "model"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_openai_api.py -v`
Expected: FAIL — 404s (routes not registered)

- [ ] **Step 3: Write the implementation**

`src/tts_server/api/openai_compat.py`:

```python
"""OpenAI-compatible TTS API: POST /v1/audio/speech and GET /v1/models."""

from __future__ import annotations

import logging

from fastapi import APIRouter, Request, Response
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from tts_server.errors import UnsupportedFeatureError
from tts_server.logging_config import truncate_text
from tts_server.models import AudioFormat, TTSRequest
from tts_server.streaming.audio import pcm_to_wav
from tts_server.streaming.pipeline import run_stream

logger = logging.getLogger("tts_server.api.openai")
router = APIRouter()

_FORMAT_ALIASES = {"pcm": AudioFormat.PCM_S16LE, "wav": AudioFormat.WAV}


class SpeechRequest(BaseModel):
    model: str | None = None
    input: str
    voice: str = "default"
    response_format: str = "pcm"
    speed: float = 1.0
    instructions: str | None = None


@router.post("/v1/audio/speech")
async def create_speech(body: SpeechRequest, request: Request):
    backend = request.app.state.backend
    cfg = request.app.state.config
    fmt = _FORMAT_ALIASES.get(body.response_format)
    if fmt is None or fmt not in backend.capabilities.supported_audio_formats:
        raise UnsupportedFeatureError(
            f"response_format {body.response_format!r} not supported by "
            f"backend {backend.name!r}",
            capabilities=backend.capabilities,
        )

    tts_request = TTSRequest(
        text=body.input,
        voice=body.voice,
        speed=body.speed,
        instructions=body.instructions,
        sample_rate=cfg.audio.sample_rate,
        format=fmt,
    )
    logger.info(
        "speech request backend=%s format=%s text=%s",
        backend.name, fmt.value, truncate_text(body.input),
    )

    if fmt == AudioFormat.PCM_S16LE:
        async def audio_bytes():
            async for chunk in run_stream(
                backend, tts_request,
                timeout_s=cfg.server.request_timeout_s, api="openai",
            ):
                yield chunk.audio

        return StreamingResponse(
            audio_bytes(),
            media_type="application/octet-stream",
            headers={"X-Backend": backend.name},
        )

    result = await backend.synthesize(tts_request)
    return Response(
        content=pcm_to_wav(result.audio, result.sample_rate),
        media_type="audio/wav",
        headers={"X-Backend": backend.name},
    )


@router.get("/v1/models")
async def list_models(request: Request):
    info = request.app.state.backend.model_info()
    return {
        "object": "list",
        "data": [{"id": info.id, "object": "model", "owned_by": info.owned_by}],
    }
```

In `src/tts_server/main.py`, change the import line `from tts_server.api import health` to `from tts_server.api import health, openai_compat` and below `app.include_router(health.router)` add `app.include_router(openai_compat.router)`.

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_openai_api.py -v`
Expected: 5 passed

- [ ] **Step 5: Commit**

```bash
git add src/tts_server/api/openai_compat.py src/tts_server/main.py tests/test_openai_api.py
git commit -m "feat: OpenAI-compatible /v1/audio/speech and /v1/models"
```

---

### Task 12: WebSocket session + ElevenLabs-style stream-input endpoint

**Files:**
- Create: `src/tts_server/streaming/websocket.py`, `src/tts_server/api/elevenlabs_compat.py`
- Modify: `src/tts_server/main.py` (import + include `elevenlabs_compat.router`)
- Test: `tests/test_elevenlabs_ws.py`

**Interfaces:**
- Consumes: `run_stream`, `encode_audio_b64`, backend, `TTSRequest`.
- Produces:
  - `streaming.websocket.StreamInputSession(backend, *, voice: str, sample_rate: int, timeout_s: float, request_id: str)` with:
    - `async handle_text(text: str, send) -> bool` — returns True when the session is finished. Nonempty text: if `backend.capabilities.supports_streaming_input`, synthesize that segment immediately and stream its audio messages (`isFinal: false`); otherwise append to an internal buffer. Empty text: flush remaining buffer, send a final message `{"audio": "", "isFinal": true, ...}`, return True.
    - `async flush(send)` — synthesize buffered text (if any) without closing.
    - `send` is `async (dict) -> None`.
    - All server messages: `{"audio": <b64 str>, "isFinal": bool, "backend": str, "request_id": str}`.
  - `GET /v1/text-to-speech/{voice_id}/stream-input` WebSocket endpoint driving the session: client JSON messages `{"text": "..."}` (empty text finalizes) or `{"flush": true}`; disconnect also flushes-and-ends implicitly (no send after disconnect).

- [ ] **Step 1: Write the failing test**

`tests/test_elevenlabs_ws.py`:

```python
import base64
import json


def collect_until_final(ws) -> list[dict]:
    messages = []
    while True:
        msg = json.loads(ws.receive_text())
        messages.append(msg)
        if msg["isFinal"]:
            return messages


def test_stream_input_incremental_text(client):
    with client.websocket_connect(
        "/v1/text-to-speech/default/stream-input"
    ) as ws:
        ws.send_text(json.dumps({"text": "Hello, "}))
        ws.send_text(json.dumps({"text": "welcome to our realtime voice demo."}))
        ws.send_text(json.dumps({"text": ""}))
        messages = collect_until_final(ws)

    assert messages[-1]["isFinal"] is True
    audio = b"".join(base64.b64decode(m["audio"]) for m in messages)
    assert len(audio) > 1000
    assert all(m["backend"] == "mock" for m in messages)
    assert all(len(m["request_id"]) > 0 for m in messages)


def test_flush_message_produces_audio_without_closing(client):
    with client.websocket_connect(
        "/v1/text-to-speech/default/stream-input"
    ) as ws:
        ws.send_text(json.dumps({"text": "part one."}))
        ws.send_text(json.dumps({"flush": True}))
        first = json.loads(ws.receive_text())
        assert first["isFinal"] is False
        assert len(base64.b64decode(first["audio"])) > 0
        # session still open: finalize normally
        ws.send_text(json.dumps({"text": ""}))
        messages = collect_until_final(ws)
        assert messages[-1]["isFinal"] is True


def test_empty_session_finalizes_cleanly(client):
    with client.websocket_connect(
        "/v1/text-to-speech/default/stream-input"
    ) as ws:
        ws.send_text(json.dumps({"text": ""}))
        msg = json.loads(ws.receive_text())
        assert msg["isFinal"] is True and msg["audio"] == ""
```

Note: MockBackend has `supports_streaming_input=True`, so each nonempty text message synthesizes immediately — `collect_until_final` sees audio for both segments plus the final marker.

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_elevenlabs_ws.py -v`
Expected: FAIL — WebSocket route not found

- [ ] **Step 3: Write the implementation**

`src/tts_server/streaming/websocket.py`:

```python
"""WebSocket session state machine shared by the ElevenLabs-compatible and
native WS endpoints. Streaming-input backends synthesize each text segment
as it arrives; others buffer until flush/finalize."""

from __future__ import annotations

import logging
import uuid
from collections.abc import Awaitable, Callable

from tts_server.backends.base import TTSBackend
from tts_server.logging_config import truncate_text
from tts_server.models import TTSRequest
from tts_server.streaming.audio import encode_audio_b64
from tts_server.streaming.pipeline import run_stream

logger = logging.getLogger("tts_server.streaming.ws")

Send = Callable[[dict], Awaitable[None]]


class StreamInputSession:
    def __init__(
        self,
        backend: TTSBackend,
        *,
        voice: str,
        sample_rate: int,
        timeout_s: float,
        request_id: str | None = None,
    ) -> None:
        self._backend = backend
        self._voice = voice
        self._sample_rate = sample_rate
        self._timeout_s = timeout_s
        self.request_id = request_id or uuid.uuid4().hex
        self._buffer: list[str] = []

    def _message(self, audio_b64: str, is_final: bool) -> dict:
        return {
            "audio": audio_b64,
            "isFinal": is_final,
            "backend": self._backend.name,
            "request_id": self.request_id,
        }

    async def _synthesize_segment(self, text: str, send: Send) -> None:
        if not text:
            return
        logger.info("ws segment text=%s", truncate_text(text))
        request = TTSRequest(
            text=text,
            voice=self._voice,
            sample_rate=self._sample_rate,
            request_id=self.request_id,
        )
        async for chunk in run_stream(
            self._backend, request, timeout_s=self._timeout_s, api="elevenlabs"
        ):
            if chunk.audio:
                await send(self._message(encode_audio_b64(chunk.audio), False))

    async def flush(self, send: Send) -> None:
        buffered = "".join(self._buffer)
        self._buffer.clear()
        await self._synthesize_segment(buffered, send)

    async def handle_text(self, text: str, send: Send) -> bool:
        """Process one client text value. Returns True when session is done."""
        if text == "":
            await self.flush(send)
            await send(self._message("", True))
            return True
        if self._backend.capabilities.supports_streaming_input:
            await self._synthesize_segment(text, send)
        else:
            self._buffer.append(text)
        return False
```

`src/tts_server/api/elevenlabs_compat.py`:

```python
"""ElevenLabs-style WebSocket streaming API."""

from __future__ import annotations

import json

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from tts_server.streaming.websocket import StreamInputSession

router = APIRouter()


@router.websocket("/v1/text-to-speech/{voice_id}/stream-input")
async def stream_input(websocket: WebSocket, voice_id: str):
    await websocket.accept()
    backend = websocket.app.state.backend
    cfg = websocket.app.state.config
    session = StreamInputSession(
        backend,
        voice=voice_id,
        sample_rate=cfg.audio.sample_rate,
        timeout_s=cfg.server.request_timeout_s,
    )

    async def send(message: dict) -> None:
        await websocket.send_text(json.dumps(message))

    try:
        while True:
            raw = await websocket.receive_text()
            data = json.loads(raw)
            if data.get("flush"):
                await session.flush(send)
                continue
            done = await session.handle_text(data.get("text", ""), send)
            if done:
                break
    except WebSocketDisconnect:
        pass
    finally:
        await websocket.close()
```

In `main.py`, extend the import to `from tts_server.api import elevenlabs_compat, health, openai_compat` and add `app.include_router(elevenlabs_compat.router)`.

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_elevenlabs_ws.py -v`
Expected: 3 passed

- [ ] **Step 5: Commit**

```bash
git add src/tts_server/streaming/websocket.py src/tts_server/api/elevenlabs_compat.py src/tts_server/main.py tests/test_elevenlabs_ws.py
git commit -m "feat: ElevenLabs-style WebSocket stream-input API"
```

---

### Task 13: Native API (`/api/v1/tts`, `/api/v1/tts/ws`, `/api/v1/backends`)

**Files:**
- Create: `src/tts_server/api/native.py`
- Modify: `src/tts_server/main.py` (import + include `native.router`)
- Test: `tests/test_native_api.py`

**Interfaces:**
- Consumes: `run_stream`, `pcm_to_wav`, `StreamInputSession`, `available_backends()`, backend/capabilities.
- Produces:
  - `POST /api/v1/tts` — body `NativeTTSRequest {text, voice="default", speed=1.0, instructions?, sample_rate?, format="pcm_s16le", stream=true, extra={}}`. `stream=true` + pcm → chunked octet-stream; `stream=false` → full body (`audio/wav` for wav, octet-stream for pcm). Response header `X-Request-ID` comes from middleware; `X-Backend` set explicitly.
  - `GET /api/v1/tts/ws` — WebSocket, same message protocol as ElevenLabs endpoint but query params `voice` (default `"default"`) and `sample_rate` (optional).
  - `GET /api/v1/backends` — `[{"name", "active": bool, "loaded": bool|null, "capabilities": {...}|null}]`; capabilities/loaded only populated for the active backend (inactive ones are not instantiated — lazy imports stay lazy).
  - `GET /api/v1/backends/{name}` — detail for one name, 404 JSON error if unknown.

- [ ] **Step 1: Write the failing test**

`tests/test_native_api.py`:

```python
import base64
import json


def test_native_tts_streaming(client):
    resp = client.post(
        "/api/v1/tts",
        json={"text": "native streaming test", "stream": True},
    )
    assert resp.status_code == 200
    assert resp.headers["x-backend"] == "mock"
    assert len(resp.content) > 1000


def test_native_tts_full_wav(client):
    resp = client.post(
        "/api/v1/tts",
        json={"text": "full result", "stream": False, "format": "wav"},
    )
    assert resp.status_code == 200
    assert resp.content[:4] == b"RIFF"


def test_native_ws(client):
    with client.websocket_connect("/api/v1/tts/ws?voice=default") as ws:
        ws.send_text(json.dumps({"text": "hello native"}))
        ws.send_text(json.dumps({"text": ""}))
        final_seen = False
        audio = b""
        while not final_seen:
            msg = json.loads(ws.receive_text())
            audio += base64.b64decode(msg["audio"])
            final_seen = msg["isFinal"]
    assert len(audio) > 0


def test_backends_listing(client):
    resp = client.get("/api/v1/backends")
    assert resp.status_code == 200
    by_name = {b["name"]: b for b in resp.json()}
    assert set(by_name) == {"mock", "qwen3"}
    assert by_name["mock"]["active"] is True
    assert by_name["mock"]["loaded"] is True
    assert by_name["mock"]["capabilities"]["streaming_mode"] == "native"
    assert by_name["qwen3"]["active"] is False
    assert by_name["qwen3"]["capabilities"] is None


def test_backend_detail_and_404(client):
    assert client.get("/api/v1/backends/mock").status_code == 200
    assert client.get("/api/v1/backends/nope").status_code == 404
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_native_api.py -v`
Expected: FAIL — 404s

- [ ] **Step 3: Write the implementation**

`src/tts_server/api/native.py`:

```python
"""Native API: full TTSRequest surface plus backend introspection."""

from __future__ import annotations

import json
from typing import Any

from fastapi import APIRouter, Request, Response, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

from tts_server.backends.registry import available_backends
from tts_server.models import AudioFormat, TTSRequest
from tts_server.streaming.audio import pcm_to_wav
from tts_server.streaming.pipeline import run_stream
from tts_server.streaming.websocket import StreamInputSession

router = APIRouter(prefix="/api/v1")


class NativeTTSRequest(BaseModel):
    text: str
    voice: str = "default"
    speed: float = 1.0
    instructions: str | None = None
    sample_rate: int | None = None
    format: AudioFormat = AudioFormat.PCM_S16LE
    stream: bool = True
    extra: dict[str, Any] = Field(default_factory=dict)


@router.post("/tts")
async def native_tts(body: NativeTTSRequest, request: Request):
    backend = request.app.state.backend
    cfg = request.app.state.config
    tts_request = TTSRequest(
        text=body.text,
        voice=body.voice,
        speed=body.speed,
        instructions=body.instructions,
        sample_rate=body.sample_rate or cfg.audio.sample_rate,
        format=body.format,
        extra=body.extra,
    )
    headers = {"X-Backend": backend.name}

    if body.stream and body.format == AudioFormat.PCM_S16LE:
        async def audio_bytes():
            async for chunk in run_stream(
                backend, tts_request,
                timeout_s=cfg.server.request_timeout_s, api="native",
            ):
                yield chunk.audio

        return StreamingResponse(
            audio_bytes(), media_type="application/octet-stream", headers=headers
        )

    result = await backend.synthesize(tts_request)
    if body.format == AudioFormat.WAV:
        return Response(
            content=pcm_to_wav(result.audio, result.sample_rate),
            media_type="audio/wav",
            headers=headers,
        )
    return Response(
        content=result.audio,
        media_type="application/octet-stream",
        headers=headers,
    )


@router.websocket("/tts/ws")
async def native_ws(websocket: WebSocket):
    await websocket.accept()
    backend = websocket.app.state.backend
    cfg = websocket.app.state.config
    params = websocket.query_params
    session = StreamInputSession(
        backend,
        voice=params.get("voice", "default"),
        sample_rate=int(params.get("sample_rate") or cfg.audio.sample_rate),
        timeout_s=cfg.server.request_timeout_s,
    )

    async def send(message: dict) -> None:
        await websocket.send_text(json.dumps(message))

    try:
        while True:
            data = json.loads(await websocket.receive_text())
            if data.get("flush"):
                await session.flush(send)
                continue
            if await session.handle_text(data.get("text", ""), send):
                break
    except WebSocketDisconnect:
        pass
    finally:
        await websocket.close()


def _backend_summary(name: str, request: Request) -> dict:
    active = request.app.state.backend
    if name == active.name:
        return {
            "name": name,
            "active": True,
            "loaded": active.loaded,
            "capabilities": active.capabilities.model_dump(),
        }
    return {"name": name, "active": False, "loaded": None, "capabilities": None}


@router.get("/backends")
async def list_backends_endpoint(request: Request):
    return [_backend_summary(name, request) for name in available_backends()]


@router.get("/backends/{name}")
async def backend_detail(name: str, request: Request):
    if name not in available_backends():
        return JSONResponse(
            status_code=404,
            content={"error": {"code": "unknown_backend", "message": name}},
        )
    return _backend_summary(name, request)
```

In `main.py`, extend the import to include `native` and add `app.include_router(native.router)`.

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_native_api.py -v`
Expected: 5 passed. Then run the full suite: `uv run pytest` — all green.

- [ ] **Step 5: Commit**

```bash
git add src/tts_server/api/native.py src/tts_server/main.py tests/test_native_api.py
git commit -m "feat: native TTS API and backend introspection endpoints"
```

---

### Task 14: Qwen3TTSBackend (CUDA-first, honest capabilities, GPU-unverified)

**Files:**
- Create: `src/tts_server/backends/qwen3.py`
- Test: `tests/test_qwen3_backend.py`

**Interfaces:**
- Consumes: `TTSBackend`, `AppConfig` (`backend.device`, `backend.dtype`, `backend.model_path`, `backend.compile`), `SynthesisError`, `BackendNotLoadedError`.
- Produces: `Qwen3TTSBackend(config: AppConfig)` — lazy torch/transformers import inside `load()`; helpful `SynthesisError` if deps missing (mentions `uv sync --extra qwen3`); `synthesize()` runs model generation via `asyncio.to_thread`; capabilities: `streaming_mode="emulated"` (inherits the base emulated stream), `supports_streaming_output=True`, `supports_streaming_input=False`, `supports_cuda=True`, `supports_cpu=True`, `supports_emotion_or_style_control=True`.

**IMPORTANT — upstream verification step:** Before writing the model-calling code, check current Qwen3-TTS usage docs (context7 or the model card at `https://huggingface.co/Qwen` for the Qwen3-TTS model) and adapt `_load_model`/`_generate` to the actual published API. The code below is the reference shape based on the transformers auto-class pattern; the *structure* (lazy import, to_thread, dtype/device handling, capability honesty) is fixed, the two model-touching functions may need adjusting. If upstream exposes true incremental audio generation, override `synthesize_stream` natively and set `streaming_mode="native"` — otherwise keep `"emulated"`. Do not claim native streaming without implementing it.

- [ ] **Step 1: Write the failing test (no GPU, no torch required)**

`tests/test_qwen3_backend.py`:

```python
import sys
from unittest.mock import MagicMock

import pytest

from tts_server.config import AppConfig
from tts_server.errors import BackendNotLoadedError, SynthesisError
from tts_server.models import TTSRequest


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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_qwen3_backend.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'tts_server.backends.qwen3'`

- [ ] **Step 3: Verify upstream API, then write the implementation**

First check the current Qwen3-TTS docs/model card as described in the task note, then write `src/tts_server/backends/qwen3.py` (reference shape — adjust `_load_model`/`_generate` to the real API):

```python
"""Qwen3-TTS backend adapter. CUDA-first with CPU fallback.

STATUS: code-complete but UNVERIFIED ON REAL GPU HARDWARE. Streaming output
is emulated (full synthesis re-sliced); capabilities say so honestly.
Requires: uv sync --extra qwen3
"""

from __future__ import annotations

import asyncio
import logging

from tts_server.backends.base import TTSBackend
from tts_server.config import AppConfig
from tts_server.errors import BackendNotLoadedError, SynthesisError
from tts_server.models import (
    BackendHealth,
    TTSCapabilities,
    TTSRequest,
    TTSResult,
    VoiceInfo,
)

logger = logging.getLogger("tts_server.backends.qwen3")

_DEFAULT_MODEL = "Qwen/Qwen3-TTS"
_INSTALL_HINT = "Qwen3 dependencies missing; install with: uv sync --extra qwen3"


class Qwen3TTSBackend(TTSBackend):
    name = "qwen3"
    capabilities = TTSCapabilities(
        supports_streaming_input=False,
        supports_streaming_output=True,
        streaming_mode="emulated",
        supports_emotion_or_style_control=True,
        supports_cuda=True,
        supports_cpu=True,
        supported_languages=["en", "zh"],
    )

    def __init__(self, config: AppConfig) -> None:
        super().__init__()
        self._model_path = config.backend.model_path or _DEFAULT_MODEL
        self._device_pref = config.backend.device
        self._dtype_name = config.backend.dtype
        self._compile = config.backend.compile
        self._model = None
        self._processor = None
        self._device = "cpu"

    async def load(self) -> None:
        await asyncio.to_thread(self._load_model)
        self._loaded = True
        logger.info("qwen3 loaded on %s (dtype=%s)", self._device, self._dtype_name)

    def _load_model(self) -> None:
        try:
            import torch
            from transformers import AutoModel, AutoProcessor
        except (ImportError, AttributeError) as exc:
            raise SynthesisError(_INSTALL_HINT) from exc

        if self._device_pref == "auto":
            self._device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self._device = self._device_pref

        dtype = {
            "bf16": torch.bfloat16,
            "fp16": torch.float16,
            "fp32": torch.float32,
        }.get(self._dtype_name, torch.float32)
        if self._device == "cpu":
            dtype = torch.float32

        self._processor = AutoProcessor.from_pretrained(self._model_path)
        self._model = AutoModel.from_pretrained(
            self._model_path, torch_dtype=dtype
        ).to(self._device)
        self._model.eval()
        if self._compile:
            self._model = torch.compile(self._model)

    def _generate(self, request: TTSRequest) -> tuple[bytes, int]:
        """Run one synthesis. Returns (pcm_s16le_bytes, sample_rate)."""
        import numpy as np
        import torch

        inputs = self._processor(
            text=request.text,
            voice=request.voice,
            instructions=request.instructions,
            return_tensors="pt",
        ).to(self._device)
        with torch.inference_mode():
            output = self._model.generate(**inputs)
        waveform = output.audio[0].float().cpu().numpy()
        sample_rate = int(getattr(output, "sample_rate", request.sample_rate))
        pcm = (np.clip(waveform, -1.0, 1.0) * 32767).astype(np.int16).tobytes()
        return pcm, sample_rate

    async def synthesize(self, request: TTSRequest) -> TTSResult:
        if not self._loaded:
            raise BackendNotLoadedError("qwen3 backend not loaded")
        try:
            pcm, sample_rate = await asyncio.to_thread(self._generate, request)
        except Exception as exc:
            if isinstance(exc, SynthesisError):
                raise
            raise SynthesisError(f"qwen3 synthesis failed: {exc}") from exc
        return TTSResult(audio=pcm, sample_rate=sample_rate)

    async def close(self) -> None:
        self._model = None
        self._processor = None
        self._loaded = False
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass

    async def health(self) -> BackendHealth:
        gpu_mb = None
        try:
            import torch

            if torch.cuda.is_available():
                gpu_mb = torch.cuda.memory_allocated() / 1e6
        except ImportError:
            pass
        return BackendHealth(ok=True, loaded=self._loaded, gpu_memory_mb=gpu_mb)

    def list_voices(self) -> list[VoiceInfo]:
        return [
            VoiceInfo(id="default", name="default", languages=["en", "zh"]),
        ]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_qwen3_backend.py -v`
Expected: 4 passed (no torch installed — the missing-dep path is what's exercised)

- [ ] **Step 5: Commit**

```bash
git add src/tts_server/backends/qwen3.py tests/test_qwen3_backend.py
git commit -m "feat: Qwen3-TTS backend adapter (CUDA-first, GPU-unverified)"
```

---

### Task 15: Client examples

**Files:**
- Create: `examples/curl/speech.sh`, `examples/python/http_client.py`, `examples/python/ws_client.py`, `examples/javascript/ws_client.mjs`, `config.example.yaml`

**Interfaces:**
- Consumes: the running server's public APIs only.
- Produces: runnable examples referenced by the README.

- [ ] **Step 1: Write the examples**

`config.example.yaml`:

```yaml
server:
  host: "0.0.0.0"
  port: 8000

backend:
  name: "mock"        # mock | qwen3
  device: "auto"      # auto | cuda | cpu
  dtype: "bf16"
  model_path: null     # e.g. "Qwen/Qwen3-TTS"
  compile: false
  warmup: true

audio:
  default_format: "pcm_s16le"
  sample_rate: 24000
  channels: 1
```

`examples/curl/speech.sh`:

```bash
#!/usr/bin/env bash
# OpenAI-compatible speech request; writes out.wav
set -euo pipefail
BASE_URL="${BASE_URL:-http://localhost:8000}"

curl -sS "$BASE_URL/v1/audio/speech" \
  -H "Content-Type: application/json" \
  -d '{
    "input": "Hello, welcome to our voice agent demo.",
    "voice": "default",
    "response_format": "wav",
    "speed": 1.0
  }' -o out.wav

echo "wrote out.wav ($(wc -c < out.wav) bytes)"
curl -sS "$BASE_URL/v1/models"
echo
curl -sS "$BASE_URL/api/v1/backends"
echo
```

`examples/python/http_client.py`:

```python
"""Stream PCM from the OpenAI-compatible endpoint and report TTFA."""

import time

import httpx

BASE_URL = "http://localhost:8000"


def main() -> None:
    start = time.perf_counter()
    first_chunk_at = None
    total = 0
    with httpx.stream(
        "POST",
        f"{BASE_URL}/v1/audio/speech",
        json={"input": "Streaming pcm from python.", "response_format": "pcm"},
        timeout=60.0,
    ) as resp:
        resp.raise_for_status()
        for chunk in resp.iter_bytes():
            if first_chunk_at is None and chunk:
                first_chunk_at = time.perf_counter() - start
            total += len(chunk)
    print(f"TTFA: {first_chunk_at * 1000:.1f} ms, total bytes: {total}")


if __name__ == "__main__":
    main()
```

`examples/python/ws_client.py`:

```python
"""Send incremental text over the ElevenLabs-style WebSocket, collect audio."""

import asyncio
import base64
import json

import websockets

URL = "ws://localhost:8000/v1/text-to-speech/default/stream-input"


async def main() -> None:
    audio = b""
    async with websockets.connect(URL) as ws:
        for part in ["Hello, ", "welcome to our realtime voice demo.", ""]:
            await ws.send(json.dumps({"text": part}))
        while True:
            msg = json.loads(await ws.recv())
            audio += base64.b64decode(msg["audio"])
            if msg["isFinal"]:
                break
    print(f"received {len(audio)} bytes of pcm from backend")


if __name__ == "__main__":
    asyncio.run(main())
```

`examples/javascript/ws_client.mjs`:

```javascript
// node >= 21 (built-in WebSocket). Streams text in, collects base64 audio out.
const url = "ws://localhost:8000/v1/text-to-speech/default/stream-input";
const ws = new WebSocket(url);
let bytes = 0;

ws.onopen = () => {
  for (const text of ["Hello, ", "welcome to our realtime voice demo.", ""]) {
    ws.send(JSON.stringify({ text }));
  }
};

ws.onmessage = (event) => {
  const msg = JSON.parse(event.data);
  bytes += Buffer.from(msg.audio, "base64").length;
  if (msg.isFinal) {
    console.log(`received ${bytes} bytes of pcm from ${msg.backend}`);
    ws.close();
  }
};
```

- [ ] **Step 2: Verify against a live server**

```bash
uv run uvicorn tts_server.main:app --port 8000 &
sleep 2
bash examples/curl/speech.sh
uv run python examples/python/http_client.py
uv run python examples/python/ws_client.py
kill %1
```

Expected: `out.wav` written with RIFF header; python clients print TTFA/byte counts; no errors. (Skip the JS example if node isn't installed; note it in the commit message if so.)

- [ ] **Step 3: Commit**

```bash
git add examples/ config.example.yaml
git commit -m "docs: curl, python, and javascript client examples"
```

---

### Task 16: Benchmarks (HTTP + WS)

**Files:**
- Create: `benchmarks/__init__.py` (empty), `benchmarks/common.py`, `benchmarks/bench_http.py`, `benchmarks/bench_ws.py`
- Test: `tests/test_benchmarks.py` (stats helpers only — network runs are manual)

**Interfaces:**
- Consumes: running server URLs; `httpx`, `websockets` (dev group).
- Produces:
  - `benchmarks/common.py`: `percentiles(values: list[float]) -> dict` (`p50`, `p90`, `p95`, `mean`, `min`, `max`), `write_results(results: dict, out_dir: str = "benchmarks/results") -> tuple[Path, Path]` writing `<name>.json` and `<name>.md` where `name = f"{results['bench']}-{results['backend']}-c{results['concurrency']}"`.
  - CLI `uv run python benchmarks/bench_http.py --backend mock --concurrency 20 --requests 200 [--url ... --text ...]` measuring TTFA, e2e latency, RTF, throughput, failure rate.
  - CLI `uv run python benchmarks/bench_ws.py --backend mock --requests 20 [--streaming-input]` measuring WS TTFA/e2e.
  - Every result dict includes `backend`, and `"note": "mock backend — synthetic audio, not real model inference"` when backend is mock.

- [ ] **Step 1: Write the failing test for stats helpers**

`tests/test_benchmarks.py`:

```python
import json

from benchmarks.common import percentiles, write_results


def test_percentiles():
    stats = percentiles([float(i) for i in range(1, 101)])
    assert stats["p50"] == 50.5
    assert stats["p90"] == 90.1
    assert stats["p95"] == 95.05
    assert stats["min"] == 1.0 and stats["max"] == 100.0


def test_write_results_labels_mock(tmp_path):
    results = {
        "bench": "http",
        "backend": "mock",
        "concurrency": 2,
        "requests": 10,
        "failures": 0,
        "ttfa_ms": percentiles([10.0, 12.0]),
        "latency_ms": percentiles([100.0, 110.0]),
        "rtf": percentiles([0.1, 0.2]),
        "throughput_rps": 5.0,
    }
    json_path, md_path = write_results(results, out_dir=str(tmp_path))
    saved = json.loads(json_path.read_text())
    assert saved["backend"] == "mock"
    assert "mock backend" in saved["note"]
    assert "| p50" in md_path.read_text() or "p50" in md_path.read_text()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_benchmarks.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'benchmarks'`

- [ ] **Step 3: Write the implementations**

Create empty `benchmarks/__init__.py`, then `benchmarks/common.py`:

```python
"""Shared benchmark helpers: percentile stats and JSON+Markdown output."""

from __future__ import annotations

import json
import statistics
from pathlib import Path

MOCK_NOTE = "mock backend — synthetic audio, not real model inference"


def percentiles(values: list[float]) -> dict:
    if not values:
        return {"p50": None, "p90": None, "p95": None, "mean": None,
                "min": None, "max": None}
    qs = statistics.quantiles(values, n=100, method="inclusive")
    return {
        "p50": qs[49],
        "p90": qs[89],
        "p95": qs[94],
        "mean": statistics.fmean(values),
        "min": min(values),
        "max": max(values),
    }


def _markdown(results: dict) -> str:
    lines = [
        f"# Benchmark: {results['bench']} — backend `{results['backend']}`",
        "",
        f"- requests: {results['requests']}, concurrency: {results['concurrency']}",
        f"- failures: {results['failures']}",
        f"- throughput: {results.get('throughput_rps', 'n/a')} req/s",
    ]
    if results.get("note"):
        lines.append(f"- **note: {results['note']}**")
    lines += ["", "| metric | p50 | p90 | p95 | mean |", "|---|---|---|---|---|"]
    for key in ("ttfa_ms", "latency_ms", "rtf"):
        if key in results and results[key]["p50"] is not None:
            s = results[key]
            lines.append(
                f"| {key} | {s['p50']:.2f} | {s['p90']:.2f} "
                f"| {s['p95']:.2f} | {s['mean']:.2f} |"
            )
    return "\n".join(lines) + "\n"


def write_results(results: dict, out_dir: str = "benchmarks/results") -> tuple[Path, Path]:
    if results["backend"] == "mock":
        results["note"] = MOCK_NOTE
    directory = Path(out_dir)
    directory.mkdir(parents=True, exist_ok=True)
    name = f"{results['bench']}-{results['backend']}-c{results['concurrency']}"
    json_path = directory / f"{name}.json"
    md_path = directory / f"{name}.md"
    json_path.write_text(json.dumps(results, indent=2))
    md_path.write_text(_markdown(results))
    return json_path, md_path
```

`benchmarks/bench_http.py`:

```python
"""HTTP benchmark against POST /v1/audio/speech (streaming pcm).

Usage:
  uv run python benchmarks/bench_http.py --backend mock --concurrency 20 --requests 200
"""

from __future__ import annotations

import argparse
import asyncio
import time

import httpx

from benchmarks.common import percentiles, write_results

DEFAULT_TEXT = "Hello, welcome to our realtime voice agent demonstration."


async def one_request(client: httpx.AsyncClient, url: str, text: str) -> dict:
    start = time.perf_counter()
    ttfa = None
    audio_bytes = 0
    async with client.stream(
        "POST", url, json={"input": text, "response_format": "pcm"}
    ) as resp:
        resp.raise_for_status()
        async for chunk in resp.aiter_bytes():
            if ttfa is None and chunk:
                ttfa = time.perf_counter() - start
            audio_bytes += len(chunk)
    elapsed = time.perf_counter() - start
    audio_s = audio_bytes / (2 * 24000)
    return {
        "ttfa_ms": ttfa * 1000,
        "latency_ms": elapsed * 1000,
        "rtf": elapsed / audio_s if audio_s > 0 else None,
    }


async def run(args: argparse.Namespace) -> None:
    url = f"{args.url}/v1/audio/speech"
    semaphore = asyncio.Semaphore(args.concurrency)
    outcomes: list[dict | None] = []

    async def bounded(client: httpx.AsyncClient) -> None:
        async with semaphore:
            try:
                outcomes.append(await one_request(client, url, args.text))
            except Exception:
                outcomes.append(None)

    started = time.perf_counter()
    async with httpx.AsyncClient(timeout=120.0) as client:
        await asyncio.gather(*(bounded(client) for _ in range(args.requests)))
    wall = time.perf_counter() - started

    ok = [o for o in outcomes if o is not None]
    results = {
        "bench": "http",
        "backend": args.backend,
        "concurrency": args.concurrency,
        "requests": args.requests,
        "failures": args.requests - len(ok),
        "throughput_rps": round(len(ok) / wall, 2),
        "ttfa_ms": percentiles([o["ttfa_ms"] for o in ok]),
        "latency_ms": percentiles([o["latency_ms"] for o in ok]),
        "rtf": percentiles([o["rtf"] for o in ok if o["rtf"] is not None]),
    }
    json_path, md_path = write_results(results)
    print(md_path.read_text())
    print(f"saved: {json_path} and {md_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://localhost:8000")
    parser.add_argument("--backend", required=True,
                        help="label for the active server backend (mock/qwen3)")
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--requests", type=int, default=20)
    parser.add_argument("--text", default=DEFAULT_TEXT)
    asyncio.run(run(parser.parse_args()))


if __name__ == "__main__":
    main()
```

`benchmarks/bench_ws.py`:

```python
"""WebSocket benchmark against the ElevenLabs-style stream-input endpoint.

Usage:
  uv run python benchmarks/bench_ws.py --backend mock --requests 20 [--streaming-input]
"""

from __future__ import annotations

import argparse
import asyncio
import json
import time

import websockets

from benchmarks.common import percentiles, write_results

DEFAULT_TEXT = "Hello, welcome to our realtime voice agent demonstration."


async def one_session(url: str, text: str, streaming_input: bool) -> dict:
    start = time.perf_counter()
    ttfa = None
    async with websockets.connect(url) as ws:
        if streaming_input:
            for word in text.split(" "):
                await ws.send(json.dumps({"text": word + " "}))
        else:
            await ws.send(json.dumps({"text": text}))
        await ws.send(json.dumps({"text": ""}))
        while True:
            msg = json.loads(await ws.recv())
            if ttfa is None and msg["audio"]:
                ttfa = time.perf_counter() - start
            if msg["isFinal"]:
                break
    elapsed = time.perf_counter() - start
    return {"ttfa_ms": ttfa * 1000 if ttfa else None,
            "latency_ms": elapsed * 1000}


async def run(args: argparse.Namespace) -> None:
    url = f"{args.url}/v1/text-to-speech/default/stream-input"
    outcomes: list[dict | None] = []
    for _ in range(args.requests):
        try:
            outcomes.append(await one_session(url, args.text, args.streaming_input))
        except Exception:
            outcomes.append(None)
    ok = [o for o in outcomes if o is not None]
    results = {
        "bench": "ws-streaming-input" if args.streaming_input else "ws",
        "backend": args.backend,
        "concurrency": 1,
        "requests": args.requests,
        "failures": args.requests - len(ok),
        "ttfa_ms": percentiles([o["ttfa_ms"] for o in ok if o["ttfa_ms"]]),
        "latency_ms": percentiles([o["latency_ms"] for o in ok]),
    }
    json_path, md_path = write_results(results)
    print(md_path.read_text())
    print(f"saved: {json_path} and {md_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="ws://localhost:8000")
    parser.add_argument("--backend", required=True)
    parser.add_argument("--requests", type=int, default=20)
    parser.add_argument("--streaming-input", action="store_true")
    asyncio.run(run(parser.parse_args()))


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run tests, then a real mock benchmark run**

Run: `uv run pytest tests/test_benchmarks.py -v` — expected 2 passed.

Then verify end-to-end:

```bash
uv run uvicorn tts_server.main:app --port 8000 &
sleep 2
uv run python benchmarks/bench_http.py --backend mock --concurrency 5 --requests 20
uv run python benchmarks/bench_ws.py --backend mock --requests 5
kill %1
```

Expected: two Markdown tables printed with p50/p90/p95 rows and the mock-backend note; files under `benchmarks/results/`.

- [ ] **Step 5: Commit**

```bash
git add benchmarks/ tests/test_benchmarks.py
git commit -m "feat: reproducible HTTP and WebSocket benchmarks with JSON+Markdown output"
```

---

### Task 17: Docker

**Files:**
- Create: `Dockerfile`, `docker-compose.yml`, `.dockerignore`

**Interfaces:**
- Consumes: the packaged project.
- Produces: CUDA-ready image; compose service with optional GPU reservation; mock backend runs in the same image without GPU.

- [ ] **Step 1: Write the files**

`.dockerignore`:

```
.venv
logs
benchmarks/results
.git
__pycache__
.pytest_cache
docs
```

`Dockerfile`:

```dockerfile
# CUDA runtime base so the qwen3 extra can use the GPU; the mock backend
# runs in this same image with no GPU attached.
FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /usr/local/bin/

WORKDIR /app
ENV UV_PYTHON_INSTALL_DIR=/opt/python UV_LINK_MODE=copy
RUN uv python install 3.12

COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-install-project --no-dev

COPY src/ src/
COPY config.example.yaml ./config.yaml
RUN uv sync --frozen --no-dev

EXPOSE 8000
ENV TTS_BACKEND=mock
CMD ["uv", "run", "--no-sync", "uvicorn", "tts_server.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

`docker-compose.yml`:

```yaml
services:
  tts-server:
    build: .
    ports:
      - "8000:8000"
    environment:
      TTS_BACKEND: ${TTS_BACKEND:-mock}
    # Uncomment for CUDA (requires nvidia-container-toolkit):
    # deploy:
    #   resources:
    #     reservations:
    #       devices:
    #         - driver: nvidia
    #           count: 1
    #           capabilities: [gpu]
```

- [ ] **Step 2: Verify the build (if docker available)**

Run: `docker build -t tts-server .` then `docker run --rm -p 8000:8000 tts-server` and `curl -s localhost:8000/healthz`.
Expected: healthz returns `{"status": "ok", ...}` with mock backend.
If docker is unavailable on this machine, note that in the commit message and rely on CI/user verification.

- [ ] **Step 3: Commit**

```bash
git add Dockerfile docker-compose.yml .dockerignore
git commit -m "feat: CUDA-ready Dockerfile and compose service"
```

---

### Task 18: README and docs

**Files:**
- Create: `README.md`, `docs/adding-a-backend.md`

**Interfaces:**
- Consumes: everything shipped; must match actual behavior exactly (run the commands you document).

- [ ] **Step 1: Write `README.md`**

Required sections, in order — write real content for each (the positioning paragraph and capability/license matrices below are the canonical text; use them verbatim or improve wording without changing meaning):

1. **Title + positioning:** "A model-agnostic, plugin-based, CUDA-accelerated streaming TTS inference server for real-time voice agents. It exposes OpenAI-compatible HTTP TTS, ElevenLabs-style WebSocket streaming, and a native API for advanced model-specific features. It supports reproducible benchmarking across latency, RTF, throughput, GPU memory usage, and streaming stability." Never describe the project as a Qwen3-TTS wrapper; Qwen3-TTS is one backend adapter.
2. **Architecture** — Mermaid diagram (copy from the design spec `docs/superpowers/specs/2026-07-03-tts-server-design.md`).
3. **Supported APIs** — table of the endpoints from Tasks 10–13 with one-line descriptions.
4. **Backend capability matrix:**

```markdown
| Backend | Streaming output | Streaming input | CUDA | CPU | Voice cloning | Style control | Status |
|---|---|---|---|---|---|---|---|
| mock  | native   | yes | no  | yes | no | no  | stable (dev/CI) |
| qwen3 | emulated | no  | yes | yes | no | yes | experimental — unverified on real GPU hardware |
```

5. **Quickstart (mock, no GPU):**

```bash
uv sync
uv run uvicorn tts_server.main:app --port 8000
curl -sS localhost:8000/v1/audio/speech -H "Content-Type: application/json" \
  -d '{"input": "hello", "response_format": "wav"}' -o hello.wav
```

6. **Qwen3-TTS setup:** `uv sync --extra qwen3`, `TTS_BACKEND=qwen3`, config example, note about model download from Hugging Face and the GPU-unverified status.
7. **Docker/CUDA instructions** — build/run commands from Task 17, nvidia-container-toolkit note.
8. **API examples** — point to `examples/`, inline the curl + WS message shapes.
9. **Benchmarks** — the CLI commands from Task 16 and a statement that published numbers are from the mock backend until GPU runs exist; never compare mock numbers to real model numbers.
10. **Configuration** — YAML shape (from `config.example.yaml`) + env override table (`TTS_BACKEND`, `TTS_HOST`, `TTS_PORT`, `TTS_DEVICE`, `TTS_CONFIG`).
11. **What's implemented / experimental / future work:** implemented — mock backend, all four API surfaces, metrics, logging, benchmarks; experimental — qwen3 adapter (GPU-unverified), torch.compile flag; future — Kokoro/Piper adapters, native streaming for qwen3 if upstream supports it, reference-audio voice cloning.
12. **License matrix:**

```markdown
| Backend | Upstream license | Commercial use |
|---|---|---|
| mock  | (this repo's license) | yes |
| qwen3 | see the Qwen3-TTS model card on Hugging Face | verify upstream license before commercial deployment |

Model weights are never redistributed by this project. Always verify the
upstream model license before commercial use.
```

- [ ] **Step 2: Write `docs/adding-a-backend.md`**

Content: the under-30-minutes recipe — (1) subclass `TTSBackend` in `src/tts_server/backends/<name>.py` implementing `load`, `synthesize`, `close` (override `synthesize_stream` only for native streaming, and set `streaming_mode` honestly); (2) add one line to `_BACKENDS` in `registry.py`; (3) add optional deps as a `[project.optional-dependencies]` extra; (4) add a capability test modeled on `tests/test_qwen3_backend.py::test_capabilities_are_honest`. Include a complete minimal example backend (copy the DummyBackend shape from `tests/test_backend_base.py` with a filled-in sine `synthesize`).

- [ ] **Step 3: Verify every documented command**

Run each command that appears in the README against the repo (quickstart, curl, benchmark CLIs). Fix any doc/behavior mismatch — in the doc or the code, whichever is wrong.

- [ ] **Step 4: Full suite + commit**

Run: `uv run pytest` — all tests pass.

```bash
git add README.md docs/adding-a-backend.md
git commit -m "docs: README with capability/license matrices and backend-authoring guide"
```
