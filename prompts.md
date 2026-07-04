Build an open-source model-agnostic, CUDA-accelerated, low-latency streaming TTS inference server with a plugin-based adapter architecture.

Project goal:
Create a production-style but not over-engineered TTS server for real-time voice agents. The project should expose unified realtime TTS APIs while allowing different open-source TTS models to be plugged in through backend adapters.

This project is not a thin wrapper around one model. Qwen3-TTS should be implemented as one backend adapter, but the architecture must support other popular open-source TTS engines such as Kokoro, Piper, Fish Speech, F5-TTS, CosyVoice, XTTS, StyleTTS2, or future models.

The main value of the project is:

* unified TTS API compatibility
* plugin-based model adapters
* CUDA/GPU inference support
* low-latency streaming design
* reproducible latency/throughput/RTF benchmarks
* clear capability reporting per backend

Tech constraints:

* Use Python 3.12+.
* Use uv for dependency and project management.
* Use FastAPI / Starlette with uvicorn.
* Prefer asyncio-native design.
* Target CUDA GPU inference where supported.
* Load selected models once at server startup.
* Provide a lightweight mock/dev backend so the API can run without GPU or model downloads.
* Do not hardcode secrets, tokens, or vendor API keys.
* Keep the architecture simple, typed, readable, and suitable for a strong undergraduate open-source portfolio project.
* Avoid unnecessary abstractions, but design the backend interface cleanly enough for additional model adapters.

Core architecture:
Design a plugin-style backend interface.

Create a common TTS backend protocol similar to:

class TTSBackend:
name: str
capabilities: TTSCapabilities

```
async def load(self) -> None:
    ...

async def synthesize(self, request: TTSRequest) -> TTSResult:
    ...

async def synthesize_stream(self, request: TTSRequest) -> AsyncIterator[TTSChunk]:
    ...

async def close(self) -> None:
    ...
```

Define common data models:

* TTSRequest
* TTSResult
* TTSChunk
* VoiceInfo
* ModelInfo
* TTSCapabilities
* AudioFormat
* BackendHealth

Capabilities should explicitly describe:

* supports_streaming_input
* supports_streaming_output
* supports_voice_cloning
* supports_reference_audio
* supports_emotion_or_style_control
* supports_cuda
* supports_cpu
* supported_languages
* supported_sample_rates
* supported_audio_formats
* native_streaming or emulated_streaming

Important:
If a backend does not support true native streaming, do not pretend that it does. It may emulate chunked output after full synthesis, but the capability metadata and README must clearly label it as emulated streaming.

Backend adapters:
Implement at least:

1. MockBackend

* No GPU required.
* Generates deterministic synthetic PCM audio chunks.
* Used for local development, tests, CI, and benchmark pipeline validation.

2. Qwen3TTSBackend

* Real backend adapter for Qwen3-TTS.
* Load model once during startup.
* Support CUDA if available.
* Support streaming output if the underlying implementation supports it.
* Support basic voice selection and instructions/style text if available.
* Do not claim unsupported features.

3. Optional lightweight backend
   Implement one additional simple open-source backend only if feasible without making the project too large. Good candidates are Kokoro or Piper because they can be useful lightweight baselines. Keep this optional.

Backend registry:
Create a backend registry so the active backend can be selected by config:

TTS_BACKEND=mock
TTS_BACKEND=qwen3
TTS_BACKEND=kokoro
TTS_BACKEND=piper

Support config via environment variables and/or a simple YAML file.

Example config:

server:
host: "0.0.0.0"
port: 8000

backend:
name: "qwen3"
device: "cuda"
dtype: "bf16"
model_path: "Qwen/Qwen3-TTS"
compile: false
warmup: true

audio:
default_format: "pcm_s16le"
sample_rate: 24000
channels: 1

API compatibility:

1. OpenAI-compatible HTTP TTS API

Implement:

POST /v1/audio/speech

Request body should support at least:

{
"model": "qwen3-tts",
"input": "Hello, welcome to our voice agent demo.",
"voice": "default",
"response_format": "pcm",
"speed": 1.0,
"instructions": "Speak naturally and warmly."
}

Behavior:

* Return audio using HTTP chunked streaming where possible.
* Support pcm_s16le as the primary realtime format.
* Support wav if simple to implement.
* Make the endpoint easy to test with curl, Python, and JavaScript clients.
* Route the request to the selected backend adapter.
* If model is omitted, use the configured default backend model.
* If requested features are unsupported by the active backend, return a clear 400 error with capability details.

2. ElevenLabs-style WebSocket streaming API

Implement:

GET /v1/text-to-speech/{voice_id}/stream-input

Behavior:

* Accept incremental text chunks over WebSocket.
* Generate audio chunks progressively where supported.
* Return JSON messages containing base64-encoded audio chunks.
* Support finalization when the client sends an empty text message or an explicit flush/close message.
* Include an isFinal flag in server responses.
* Include request_id and backend name in server responses when useful.
* This API should be suitable for LLM-streaming-text -> TTS-streaming-audio workflows.

Example client messages:

{
"text": "Hello, "
}

{
"text": "welcome to our realtime voice demo."
}

{
"text": ""
}

Example server message:

{
"audio": "<base64-audio-chunk>",
"isFinal": false,
"backend": "qwen3",
"request_id": "..."
}

3. Native API for advanced backend features

Implement:

POST /api/v1/tts
GET /api/v1/tts/ws

Purpose:
Expose model-specific capabilities that do not fit OpenAI or ElevenLabs compatibility cleanly.

The native request should allow:

* backend selection
* voice selection
* reference audio metadata
* style/instruction text
* sample rate
* audio format
* streaming mode
* experimental backend-specific parameters

Do not make this too complex in the first version.

4. Utility APIs

Implement:

GET /v1/models
GET /api/v1/backends
GET /api/v1/backends/{backend_name}
GET /healthz
GET /metrics

/v1/models:
Return OpenAI-compatible model listing.

/api/v1/backends:
Return backend names, loaded state, and capabilities.

/healthz:
Return server status and active backend health.

/metrics:
Expose useful runtime counters and latency metrics in a simple Prometheus-compatible format if feasible.

Core system requirements:

* Support streaming audio chunks instead of waiting for full synthesis whenever possible.
* Track first-audio latency / TTFA.
* Track total synthesis latency.
* Track real-time factor / RTF.
* Track request count, failure count, active sessions, and backend-level errors.
* Include structured logs under logs/.
* Each server startup should create a separate log file.
* Logs must not contain secrets, full tokens, or sensitive user data.
* Include request IDs for tracing.
* Add graceful shutdown logic to release backend resources.

CUDA / inference requirements:

* Use GPU if available and supported by the backend.
* Use mixed precision where appropriate.
* Add clear feature flags for optional optimizations such as torch.compile, FlashAttention, bf16, fp16, or CUDA graph capture.
* Do not claim unsupported optimizations in README unless implemented and measured.
* Include warmup logic after model loading.
* Expose basic GPU memory statistics if available.
* Keep CPU fallback available through MockBackend and any CPU-friendly backend.

Streaming design:
Implement a shared streaming abstraction that can handle:

* true native streaming backend output
* emulated streaming from full audio result
* WebSocket text input buffering
* flush messages
* close messages
* cancellation
* timeout
* backpressure

Important:
Clearly separate streaming input from streaming output.

Some models may support:

* full text input -> full audio output
* full text input -> streaming audio output
* streaming text input -> streaming audio output

The server must report this honestly through TTSCapabilities.

Benchmark framework:
Create reproducible benchmark scripts under benchmarks/.

Benchmarks should measure:

* First audio latency / TTFA
* End-to-end latency
* Real-time factor / RTF
* p50 / p90 / p95 latency
* Throughput under concurrent requests
* Peak GPU memory usage
* Failure rate under load
* Difference between true streaming and emulated streaming
* Backend-to-backend comparison when multiple adapters are installed

The benchmark output should be saved as JSON and Markdown.

Do not fake benchmark numbers. If real model inference is unavailable, clearly label results as mock-backend results.

Benchmark CLI examples:

uv run python benchmarks/bench_http.py --backend qwen3 --concurrency 1 --requests 20
uv run python benchmarks/bench_http.py --backend mock --concurrency 20 --requests 200
uv run python benchmarks/bench_ws.py --backend qwen3 --streaming-input

Repository structure suggestion:

.
├── README.md
├── pyproject.toml
├── uv.lock
├── Dockerfile
├── docker-compose.yml
├── src/
│   └── tts_server/
│       ├── main.py
│       ├── config.py
│       ├── api/
│       │   ├── openai_compat.py
│       │   ├── elevenlabs_compat.py
│       │   ├── native.py
│       │   └── health.py
│       ├── backends/
│       │   ├── base.py
│       │   ├── registry.py
│       │   ├── mock.py
│       │   ├── qwen3.py
│       │   └── kokoro.py
│       ├── streaming/
│       │   ├── chunks.py
│       │   ├── websocket.py
│       │   └── audio.py
│       ├── metrics/
│       ├── logging_config.py
│       └── errors.py
├── examples/
│   ├── curl/
│   ├── python/
│   └── javascript/
├── benchmarks/
├── tests/
└── docs/

Testing requirements:
Add tests for:

* backend protocol behavior
* backend registry
* MockBackend synthesis
* OpenAI-compatible request validation
* WebSocket message protocol
* capability reporting
* metrics output
* error handling for unsupported features

Documentation requirements:
Write a strong README.

README must include:

* project positioning
* supported API compatibility
* supported backends
* backend capability matrix
* quickstart with uv
* mock backend quickstart
* Qwen3-TTS backend setup
* Docker/CUDA instructions
* API examples
* benchmark instructions
* Mermaid architecture diagram
* what is implemented
* what is experimental
* what is future work
* license notes for each backend

README positioning:
This project provides a model-agnostic, plugin-based, CUDA-accelerated streaming TTS inference server for real-time voice agents. It exposes OpenAI-compatible HTTP TTS, ElevenLabs-style WebSocket streaming, and a native API for advanced model-specific features. It supports reproducible benchmarking across latency, RTF, throughput, GPU memory usage, and streaming stability.

Do not describe the project as a Qwen3-TTS wrapper. Describe Qwen3-TTS as one backend adapter.

License and model notes:

* Do not assume all open-source TTS models are commercially usable.
* Add a backend license matrix.
* Tell users to verify upstream model licenses before commercial deployment.
* Do not redistribute model weights unless the upstream license permits it.

Quality bar:

* Code should be clean, typed, and readable.
* Prefer small composable modules.
* Avoid unnecessary abstractions.
* Make it easy for another developer to add a new backend adapter in under 30 minutes.
* The project should be understandable and impressive to an AI graduate-school reviewer or technical interviewer.

Implementation order:

1. Inspect the existing repository if any.
2. Preserve existing working code.
3. Implement the backend protocol and registry.
4. Implement MockBackend.
5. Implement OpenAI-compatible HTTP endpoint.
6. Implement ElevenLabs-style WebSocket endpoint.
7. Implement metrics, logging, and health checks.
8. Implement Qwen3TTSBackend.
9. Add examples and tests.
10. Add benchmarks.
11. Write README and docs.
12. Only then consider adding another real backend adapter.

Do not over-engineer the architecture. Implement the smallest complete version first, then expand.

