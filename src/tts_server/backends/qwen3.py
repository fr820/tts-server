"""Qwen3-TTS backend adapter. CUDA-first with CPU fallback.

Default model is the 0.6B CustomVoice variant, chosen for realtime use
(smallest talker that still covers the 10 major languages). The 1.7B
CustomVoice variant can be selected via `backend.model_path` — it also
honors `instruct` style control, which 0b6 models ignore (the
emotion/style capability is adjusted per loaded model accordingly).

STATUS: exercised on real CUDA hardware (NVIDIA A10, Ampere) via
`scripts/gpu_validate.py`; see `benchmarks/results/gpu_validation/` for
the latest real run. Streaming output remains emulated (full synthesis
re-sliced); capabilities say so honestly.

Upstream API verified via the `qwen-tts` PyPI package (v0.1.1) and the
QwenLM/Qwen3-TTS GitHub README. Model loading and generation go through
the `qwen_tts.Qwen3TTSModel` class, *not* a generic transformers
AutoModel/AutoProcessor pair:

    from qwen_tts import Qwen3TTSModel
    model = Qwen3TTSModel.from_pretrained(
        "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice",
        device_map="cuda:0", dtype=torch.bfloat16,
    )
    wavs, sr = model.generate_custom_voice(
        text=..., language=..., speaker=..., instruct=...,
    )

`generate_custom_voice` (and its siblings `generate_voice_design` /
`generate_voice_clone`) always return the *complete* waveform for the
requested text -- there is no incremental/token-by-token audio streaming
call exposed at this Python API layer (the "streaming" mentioned in
upstream marketing refers to the server-side OpenAI-compatible HTTP
wrapper, not this model class). Because of that, `synthesize_stream` is
NOT overridden here and streaming_mode stays "emulated" -- the base class's
re-slicing behavior is accurate to what this backend can actually do.

GPU install constraints (see `pyproject.toml` `[qwen3]` extra):
  * `numba>=0.59` -- qwen-tts -> librosa -> numba pulls numba transitively;
    older numba drags in llvmlite with no Python 3.12 wheel.
  * `torch>=2.4,<2.13` -- torch 2.13's triton stack references
    `torch._native.ops.bmm_outer_product.triton_kernels` (not shipped by
    the bundled triton), so generation fails at runtime. 2.12.1+cu130 is
    the verified-good version.
  * flash-attn is NOT required: without it qwen-tts falls back to a manual
    PyTorch attention path that works, just slower. Measured on the A10
    (2026-08-20, see `reports/2026-08-20/`): stock qwen-tts path RTF p90
    1.47 (host-bound on the nested code-predictor generate, ~24% GPU
    util); with the default `fast_subtalker` patch (CUDA-graphed sub-talker,
    `qwen3_fast.py`) RTF p90 0.58 at 46% util — realtime, same semantics.

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

_DEFAULT_MODEL = "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice"
_DEFAULT_SPEAKER = "Vivian"
_DEFAULT_LANGUAGE = "English"
_INSTALL_HINT = "Qwen3 dependencies missing; install with: uv sync --extra qwen3"

# The 10 major languages Qwen3-TTS documents support for.
_SUPPORTED_LANGUAGES = [
    "Chinese",
    "English",
    "Japanese",
    "Korean",
    "German",
    "French",
    "Russian",
    "Portuguese",
    "Spanish",
    "Italian",
]


class Qwen3TTSBackend(TTSBackend):
    name = "qwen3"
    capabilities = TTSCapabilities(
        supports_streaming_input=False,
        supports_streaming_output=True,
        streaming_mode="emulated",
        supports_emotion_or_style_control=True,
        supports_cuda=True,
        supports_cpu=True,
        supported_languages=_SUPPORTED_LANGUAGES,
    )

    def __init__(self, config: AppConfig) -> None:
        super().__init__()
        self._model_path = config.backend.model_path or _DEFAULT_MODEL
        self._device_pref = config.backend.device
        self._dtype_name = config.backend.dtype
        self._compile = config.backend.compile
        self._default_speaker = config.backend.options.get("speaker", _DEFAULT_SPEAKER)
        self._default_language = config.backend.options.get(
            "language", _DEFAULT_LANGUAGE
        )
        # Perf: replace qwen-tts's nested per-frame HF generate() on the code
        # predictor with a lean KV-cached loop (see qwen3_fast.py for the
        # measurements). Kill-switch for A-B benchmarking.
        self._fast_subtalker = config.backend.options.get("fast_subtalker", True)
        self._subtalker_greedy = config.backend.options.get("subtalker_greedy", False)
        self._model = None
        self._device = "cpu"
        # Honesty rule: the 0.6B CustomVoice model ignores `instruct`
        # (qwen-tts drops style instructions for 0b6 models), so the
        # emotion/style capability is only true for the 1.7B variant.
        # Resolved per loaded model in load().
        if "0.6b" in self._model_path.lower() or "0b6" in self._model_path.lower():
            caps = self.capabilities.model_copy()
            caps.supports_emotion_or_style_control = False
            self.capabilities = caps
        # Serialize inference on the shared model. HuggingFace `generate()` is
        # not documented thread-safe; on a single GPU concurrent calls only
        # serialize incidentally (CUDA default stream + GIL) and pile up with
        # no backpressure (observed: 4 in-flight requests -> ~30s time-to-first
        # audio). An explicit lock makes queuing deterministic and guarantees no
        # interleaved mutation of the model's generation state, at zero throughput
        # cost (CUDA serializes a single device regardless).
        self._infer_lock = asyncio.Lock()

    async def load(self) -> None:
        await asyncio.to_thread(self._load_model)
        self._loaded = True
        logger.info(
            "qwen3 loaded on %s (dtype=%s, fast_subtalker=%s, greedy=%s)",
            self._device,
            self._dtype_name,
            self._fast_subtalker,
            self._subtalker_greedy,
        )

    def _load_model(self) -> None:
        try:
            import torch
            from qwen_tts import Qwen3TTSModel
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

        device_map = (
            self._device
            if self._device == "cpu" or ":" in self._device
            else f"{self._device}:0"
        )

        self._model = Qwen3TTSModel.from_pretrained(
            self._model_path,
            device_map=device_map,
            dtype=dtype,
        )
        if self._fast_subtalker:
            try:
                from tts_server.backends.qwen3_fast import patch_code_predictor

                patch_code_predictor(
                    self._model.model.talker.code_predictor,
                    greedy=self._subtalker_greedy,
                )
            except AttributeError:
                # qwen-tts internals moved (version drift): stay on the stock,
                # slower path rather than failing to load.
                logger.warning(
                    "qwen3: fast sub-talker patch did not apply "
                    "(qwen-tts internals changed?); using stock path",
                    exc_info=True,
                )
        if self._compile:
            try:
                self._model = torch.compile(self._model)
            except Exception:
                logger.warning(
                    "qwen3: torch.compile unavailable/failed; continuing uncompiled",
                    exc_info=True,
                )

    def _generate(self, request: TTSRequest) -> tuple[bytes, int]:
        """Run one synthesis. Returns (pcm_s16le_bytes, sample_rate).

        `Qwen3TTSModel.generate_custom_voice` returns the complete waveform
        (a list of numpy float arrays, one per input text) plus a sample
        rate -- there is no incremental/streaming variant at this layer.
        """
        import numpy as np

        speaker = (
            request.voice
            if request.voice and request.voice != "default"
            else self._default_speaker
        )
        language = request.extra.get("language", self._default_language)

        wavs, sample_rate = self._model.generate_custom_voice(
            text=request.text,
            language=language,
            speaker=speaker,
            instruct=request.instructions or "",
        )
        waveform = np.asarray(wavs[0], dtype=np.float32)
        pcm = (np.clip(waveform, -1.0, 1.0) * 32767).astype(np.int16).tobytes()
        return pcm, int(sample_rate)

    async def synthesize(self, request: TTSRequest) -> TTSResult:
        if not self._loaded:
            raise BackendNotLoadedError("qwen3 backend not loaded")
        async with self._infer_lock:
            try:
                pcm, sample_rate = await asyncio.to_thread(self._generate, request)
            except Exception as exc:
                if isinstance(exc, SynthesisError):
                    raise
                raise SynthesisError(f"qwen3 synthesis failed: {exc}") from exc
        return TTSResult(audio=pcm, sample_rate=sample_rate)

    async def close(self) -> None:
        self._model = None
        self._loaded = False
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass

    async def health(self) -> BackendHealth:
        gpu_mb = None
        gpu_peak_mb = None
        try:
            import torch

            if torch.cuda.is_available():
                gpu_mb = torch.cuda.memory_allocated() / 1e6
                gpu_peak_mb = torch.cuda.max_memory_allocated() / 1e6
        except ImportError:
            pass
        return BackendHealth(
            ok=True,
            loaded=self._loaded,
            gpu_memory_mb=gpu_mb,
            gpu_memory_peak_mb=gpu_peak_mb,
        )

    def list_voices(self) -> list[VoiceInfo]:
        # After load, report the model's real speaker table (case-insensitive
        # upstream); before load, a known-valid subset of the default model.
        if self._model is not None:
            try:
                spk = self._model.model.config.talker_config.spk_id
                return [
                    VoiceInfo(id=name.title(), name=name.title(), languages=_SUPPORTED_LANGUAGES)
                    for name in sorted(spk)
                ]
            except AttributeError:
                pass
        return [
            VoiceInfo(id="Vivian", name="Vivian", languages=_SUPPORTED_LANGUAGES),
            VoiceInfo(id="Ryan", name="Ryan", languages=_SUPPORTED_LANGUAGES),
        ]
