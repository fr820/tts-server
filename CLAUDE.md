# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

All project management goes through `uv` — never pip directly.

```bash
uv sync                                  # base install (mock backend only, no GPU/model deps)
uv sync --extra qwen3                    # adds torch/transformers/qwen-tts for the Qwen3 backend

uv run pytest                            # full suite (must pass with no GPU and no model downloads)
uv run pytest tests/test_pipeline.py -v  # one file
uv run pytest tests/test_pipeline.py::test_idle_timeout_raises -v  # one test

uv run uvicorn tts_server.main:app --port 8000   # run server (mock backend by default)
TTS_BACKEND=qwen3 uv run uvicorn tts_server.main:app --port 8000

uv run python benchmarks/bench_http.py --backend mock --concurrency 5 --requests 20
uv run python benchmarks/bench_ws.py --backend mock --requests 5
```

Config precedence: defaults → YAML (`TTS_CONFIG` env or `./config.yaml`) → env overrides (`TTS_BACKEND`, `TTS_HOST`, `TTS_PORT`, `TTS_DEVICE`). See `config.example.yaml`.

## Architecture

Single-process asyncio FastAPI server. One backend is active per process, resolved by name from `backends/registry.py` (a name → `"module:Class"` dict with **lazy imports** — selecting `mock` must never import `qwen3`'s heavy deps; a test enforces this) and loaded once in the FastAPI lifespan (`main.py`), with warmup and graceful `close()` on shutdown.

The flow every request takes:

- **Three API surfaces** (`api/openai_compat.py`, `api/elevenlabs_compat.py`, `api/native.py`) all funnel into the same two entry points in `streaming/pipeline.py`:
  - `run_stream()` — async generator wrapping any backend stream with a bounded queue (backpressure), per-chunk idle timeout, producer cancellation on client disconnect, and metrics. Cancellation (`GeneratorExit`/`CancelledError`) is deliberately *not* counted as a failure and records no latency.
  - `synthesize_once()` — the non-streaming path (WAV, `stream=false`); applies the same timeout + metrics. Do not call `backend.synthesize()` directly from an API handler.
- **Backends** subclass `TTSBackend` (`backends/base.py`): implement `load`/`synthesize`/`close`; `synthesize_stream` has an emulated default (full synthesis re-sliced into chunks). Blocking model inference must go through `asyncio.to_thread` — handlers are asyncio-native.
- **WebSocket endpoints** share `StreamInputSession` + `run_ws_session` in `streaming/websocket.py` (buffering vs. per-segment synthesis depends on `supports_streaming_input`; malformed input → close 1003; `TTSServerError` → error message then close 1011). Don't duplicate the receive loop in a new WS endpoint — reuse `run_ws_session`.
- **Errors**: raise `TTSServerError` subclasses (`errors.py`); the app-level handler in `main.py` maps them to `{"error": {code, message, capabilities?}}` with the right status. `UnsupportedFeatureError` should carry the backend's capabilities.

## The honesty rule (project-defining constraint)

Capability metadata (`TTSCapabilities`) must describe only what the code actually does. Emulated streaming is labeled `streaming_mode="emulated"` (the base-class default); only backends that genuinely pace/generate chunks incrementally may claim `"native"`. This extends to docs and benchmarks: README claims must match implemented behavior, and benchmark results from the mock backend are auto-labeled "mock backend — synthetic audio, not real model inference" (`benchmarks/common.py`) — never present mock numbers as real model performance.

## Backend status caveats

- `Qwen3TTSBackend` is code-complete but **unverified on real GPU hardware** (this machine has no CUDA; torch isn't installed). It uses the `qwen-tts` package's `Qwen3TTSModel.generate_custom_voice()` API — verified against upstream docs, not executed. Its tests run without torch by design (they exercise the missing-dep and not-loaded paths). Keep it importable without torch: heavy imports live inside methods.
- `MockBackend` is deterministic (sine PCM keyed on text hash) with configurable pacing via `config.backend.options` — tests and the `client` fixture (`tests/conftest.py`) set delays to zero.

## Adding a backend

Follow `docs/adding-a-backend.md`: subclass `TTSBackend`, add one line to `_BACKENDS` in `registry.py`, put optional deps behind a `[project.optional-dependencies]` extra, set capabilities honestly, and add a capability test (model it on `tests/test_qwen3_backend.py::test_capabilities_are_honest`).

## Logging

`logging_config.py` writes JSON lines to a new `logs/server-<timestamp>.log` per startup; every record carries the request ID from `request_id_var` (set by HTTP middleware; WS sessions set it from `session.request_id`). Never log full user text — pass it through `truncate_text()`.
