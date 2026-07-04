# Adding a backend

A new backend adapter should take well under 30 minutes: subclass
`TTSBackend`, register it, declare its optional dependencies, and add one
honesty test. This recipe walks through all four steps with a complete
minimal example.

## 1. Subclass `TTSBackend`

Create `src/tts_server/backends/<name>.py` implementing `load`,
`synthesize`, and `close` (all abstract on `tts_server.backends.base.TTSBackend`).
Only override `synthesize_stream` if your backend has *native* incremental
streaming — otherwise leave it alone and the base class emulates streaming
by synthesizing the full result and re-slicing it into chunks.

Set `capabilities` honestly. In particular:

- `streaming_mode` must be `"native"` only if you override `synthesize_stream`
  with a real incremental generator; otherwise leave it `"emulated"` (the
  base class default).
- Every other `supports_*` flag must reflect something the adapter actually
  does and that a test exercises — never claim an unimplemented feature.

### Complete minimal example: `SineBackend`

This backend has no real model — it emits a sine wave — but it is a fully
working adapter shape you can copy and fill in with a real model call.

```python
"""Minimal example backend: emits a sine wave. Not a real TTS model."""

from __future__ import annotations

import math
import struct

from tts_server.backends.base import TTSBackend
from tts_server.config import AppConfig
from tts_server.models import TTSCapabilities, TTSRequest, TTSResult

_FREQUENCY_HZ = 440.0


class SineBackend(TTSBackend):
    name = "sine"
    capabilities = TTSCapabilities(
        supports_streaming_output=True,
        streaming_mode="emulated",  # no native streaming call to wrap
        supports_cuda=False,
        supports_cpu=True,
    )

    def __init__(self, config: AppConfig) -> None:
        super().__init__()
        self._config = config

    async def load(self) -> None:
        # No model weights to load for this example.
        self._loaded = True

    async def synthesize(self, request: TTSRequest) -> TTSResult:
        sample_rate = request.sample_rate or 24000
        duration_s = max(len(request.text), 1) * 0.05
        n_samples = int(sample_rate * duration_s)
        samples = [
            int(32767 * 0.2 * math.sin(2 * math.pi * _FREQUENCY_HZ * i / sample_rate))
            for i in range(n_samples)
        ]
        pcm = struct.pack(f"<{len(samples)}h", *samples)
        return TTSResult(audio=pcm, sample_rate=sample_rate)

    async def close(self) -> None:
        self._loaded = False
```

(This mirrors the `DummyBackend` shape used in
`tests/test_backend_base.py`, with a real `synthesize` body filled in
instead of returning silence.)

## 2. Register it

Add one line to `_BACKENDS` in `src/tts_server/backends/registry.py`. The
value is an import path, not an eager import, so heavy dependencies (torch,
etc.) are only imported when that backend is actually selected:

```python
_BACKENDS: dict[str, str] = {
    "mock": "tts_server.backends.mock:MockBackend",
    "qwen3": "tts_server.backends.qwen3:Qwen3TTSBackend",
    "sine": "tts_server.backends.sine:SineBackend",
}
```

Select it with `TTS_BACKEND=sine` or `backend.name: "sine"` in `config.yaml`.

## 3. Add optional dependencies

If the backend needs extra packages (a model SDK, torch, etc.), add them as
a `[project.optional-dependencies]` extra in `pyproject.toml` rather than a
hard dependency, following the `qwen3` extra:

```toml
[project.optional-dependencies]
qwen3 = ["torch>=2.4", "transformers>=4.46", "accelerate>=1.0", "qwen-tts>=0.1.1"]
sine = []  # example backend above has no extra deps
```

Install with `uv sync --extra <name>`. Import the extra dependencies inside
`load`/the module's runtime code (not at module top level) so importing the
registry never fails for backends nobody selected — see how `qwen3.py`
imports `torch`/`qwen_tts` lazily inside `_load_model`.

## 4. Add a capability-honesty test

Model it on `tests/test_qwen3_backend.py::test_capabilities_are_honest`:
assert every `supports_*`/`streaming_mode` flag your backend declares
matches what it can actually do, e.g. (`make_config` is a small local
helper that builds an `AppConfig` with `backend.name` set, same as in
`test_qwen3_backend.py`):

```python
def test_capabilities_are_honest():
    from tts_server.backends.sine import SineBackend

    caps = SineBackend(make_config(name="sine")).capabilities
    assert caps.streaming_mode == "emulated"  # no native streaming override
    assert caps.supports_cuda is False        # this backend never touches a GPU
```

Run `uv run pytest` to confirm the new backend's tests pass alongside the
rest of the suite, and that selecting `mock` or any other backend still
does not import your new backend's module (lazy-import discipline, see
`tests/test_registry.py`).
