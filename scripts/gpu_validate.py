"""Direct GPU validation harness for the Qwen3-TTS backend.

Exercises the REAL backend code path (``Qwen3TTSBackend``) — not just the
upstream ``qwen_tts`` API — on real CUDA hardware, real model weights, and real
audio output. Emits a structured PASS/FAIL report plus a JSON summary.

Checks (each independently PASS/FAIL):
  env        torch sees CUDA and can run a matmul on the A10
  load       backend.load() succeeds (downloads weights on first run)
  audio      single synthesis yields non-trivial, non-silent PCM at a sane sr;
             a WAV artifact is written for manual listening
  latency    warm p50/p95 latency and RTF over N iterations
  vram       peak GPU memory stays within the device ceiling
  concurrency K concurrent synthesize() calls all succeed (thread-safety)

Reproducible:
  uv pip install qwen-tts                       # or: uv sync --extra qwen3 (once fixed)
  TTS_BACKEND=qwen3 uv run python scripts/gpu_validate.py
  TTS_BACKEND=qwen3 uv run python scripts/gpu_validate.py --iters 20 --concurrency 4

NOTE: numbers are REAL model inference on this GPU. They are never labeled as
mock. If torch/CUDA is unavailable the script reports env=FAIL and skips the
model checks rather than fabricating results.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import sys
import time
import wave
from pathlib import Path

# Allow `python scripts/gpu_validate.py` without an installed package.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from tts_server.backends.qwen3 import Qwen3TTSBackend  # noqa: E402
from tts_server.config import AppConfig  # noqa: E402
from tts_server.models import TTSRequest  # noqa: E402

RESULTS_DIR = Path("benchmarks/results/gpu_validation")
SAMPLE_TEXT = "The quick brown fox jumps over the lazy dog. Welcome aboard."


class Report:
    def __init__(self) -> None:
        self.checks: list[dict] = []
        self.artifacts: dict[str, str] = {}

    def add(self, name: str, passed: bool, detail: str, **extra) -> None:
        self.checks.append(
            {"check": name, "status": "PASS" if passed else "FAIL", "detail": detail, **extra}
        )
        tag = "PASS" if passed else "FAIL"
        print(f"[{tag}] {name}: {detail}")

    def summary(self) -> str:
        n_pass = sum(1 for c in self.checks if c["status"] == "PASS")
        n_fail = sum(1 for c in self.checks if c["status"] == "FAIL")
        return f"{n_pass} PASS / {n_fail} FAIL out of {len(self.checks)} checks"


def _audio_stats(pcm: bytes, sample_rate: int) -> dict:
    """Compute basic PCM-s16le stats to prove the audio is real (not silence)."""
    import array

    samples = array.array("h")
    samples.frombytes(pcm)
    n = len(samples)
    if n == 0:
        return {"samples": 0, "duration_s": 0.0, "peak": 0, "rms": 0.0}
    peak = max(abs(s) for s in samples)
    mean_sq = sum(s * s for s in samples) / n
    rms = mean_sq**0.5
    return {
        "samples": n,
        "duration_s": round(n / sample_rate, 3),
        "peak": peak,
        "rms": round(rms, 1),
        "peak_ratio": round(peak / 32767, 4),
    }


def _write_wav(path: Path, pcm: bytes, sample_rate: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm)


async def main(args: argparse.Namespace) -> int:
    report = Report()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # ---- env ----
    try:
        import torch

        cuda_ok = torch.cuda.is_available()
        dev_name = torch.cuda.get_device_name(0) if cuda_ok else "n/a"
        caps = torch.cuda.get_device_capability(0) if cuda_ok else None
        info = {
            "torch": torch.__version__,
            "cuda_compiled": torch.version.cuda,
            "cuda_available": cuda_ok,
            "device": dev_name,
            "capability": caps,
        }
        if cuda_ok:
            x = torch.randn(64, 64, device="cuda")
            float((x @ x).sum())  # force a real kernel
            torch.cuda.synchronize()
        report.add("env", cuda_ok, f"torch={torch.__version__} cuda={torch.version.cuda} dev={dev_name}", **info)
    except Exception as exc:  # noqa: BLE001
        report.add("env", False, f"torch/CUDA probe failed: {exc!r}")
        report.artifacts["error"] = repr(exc)
        _write_report(args, report)
        return 1

    # ---- build backend via real factory path ----
    cfg = AppConfig()
    cfg.backend.name = "qwen3"
    cfg.backend.device = args.device
    cfg.backend.dtype = args.dtype
    cfg.backend.model_path = args.model or None
    cfg.backend.compile = args.compile
    cfg.audio.sample_rate = 24000
    backend = Qwen3TTSBackend(cfg)

    # ---- load ----
    t0 = time.perf_counter()
    try:
        await backend.load()
        load_s = time.perf_counter() - t0
        report.add("load", backend.loaded, f"loaded on {backend._device} in {load_s:.1f}s", load_s=round(load_s, 2), device=backend._device)
    except Exception as exc:  # noqa: BLE001
        report.add("load", False, f"load() failed: {exc!r}")
        report.artifacts["error"] = repr(exc)
        _write_report(args, report)
        await _safe_close(backend)
        return 1

    import torch as _t  # noqa: F811
    _t.cuda.reset_peak_memory_stats()

    # ---- single synthesis + real-audio validation ----
    req = TTSRequest(text=SAMPLE_TEXT, voice="Vivian")
    try:
        result = await backend.synthesize(req)
        stats = _audio_stats(result.audio, result.sample_rate)
        wav_path = RESULTS_DIR / "sample.wav"
        _write_wav(wav_path, result.audio, result.sample_rate)
        # Real speech: non-empty, non-silent (peak well above noise floor),
        # reasonable duration for the input length.
        passed = (
            stats["samples"] > 0
            and stats["peak"] > 1000            # not digital silence
            and 0.5 <= stats["duration_s"] <= 60  # sane bounds
        )
        report.add(
            "audio",
            passed,
            f"sr={result.sample_rate} dur={stats['duration_s']}s peak={stats['peak']} rms={stats['rms']}",
            sample_rate=result.sample_rate,
            **stats,
        )
        report.artifacts["wav"] = str(wav_path)
    except Exception as exc:  # noqa: BLE001
        report.add("audio", False, f"synthesis failed: {exc!r}")
        report.artifacts["error"] = repr(exc)
        _write_report(args, report)
        await _safe_close(backend)
        return 1

    # ---- latency / RTF (warm) ----
    latencies: list[float] = []
    audio_lens: list[float] = []
    for _ in range(args.iters):
        s = time.perf_counter()
        r = await backend.synthesize(req)
        _t.cuda.synchronize()
        dt = time.perf_counter() - s
        latencies.append(dt)
        audio_lens.append(len(r.audio) / (2 * r.sample_rate))
    rtfs = [l / a for l, a in zip(latencies, audio_lens) if a > 0]
    lat_pct = _pct(latencies)
    rtf_pct = _pct(rtfs)
    passed = lat_pct["p50"] is not None and lat_pct["p50"] < args.latency_budget_s
    report.add(
        "latency",
        passed,
        f"warm p50={lat_pct['p50']*1000:.0f}ms p95={lat_pct['p95']*1000:.0f}ms RTF p50={rtf_pct['p50']:.3f} (budget {args.latency_budget_s}s)",
        latency_ms={k: round(v * 1000, 1) for k, v in lat_pct.items() if v is not None},
        rtf={k: round(v, 3) for k, v in rtf_pct.items() if v is not None},
    )

    # ---- VRAM ----
    alloc_mb = _t.cuda.memory_allocated() / 1e6
    peak_mb = _t.cuda.max_memory_allocated() / 1e6
    total_mb = _t.cuda.get_device_properties(0).total_memory / 1e6
    passed = peak_mb < total_mb
    report.add(
        "vram",
        passed,
        f"allocated={alloc_mb:.0f}MB peak={peak_mb:.0f}MB / {total_mb:.0f}MB device total",
        allocated_mb=round(alloc_mb, 1),
        peak_mb=round(peak_mb, 1),
        device_total_mb=round(total_mb, 1),
    )

    # ---- concurrency (thread-safety of the shared model) ----
    reqs = [TTSRequest(text=SAMPLE_TEXT, voice="Vivian") for _ in range(args.concurrency)]
    t0 = time.perf_counter()
    outcomes = await asyncio.gather(
        *(backend.synthesize(q) for q in reqs), return_exceptions=True
    )
    wall = time.perf_counter() - t0
    errs = [o for o in outcomes if isinstance(o, BaseException)]
    ok_lens = [len(o.audio) for o in outcomes if not isinstance(o, BaseException)]
    passed = len(errs) == 0 and len(ok_lens) == args.concurrency
    report.add(
        "concurrency",
        passed,
        f"{args.concurrency} concurrent: {len(ok_lens)}/{args.concurrency} ok, {len(errs)} errors, wall={wall:.2f}s"
        + (f" first_err={errs[0]!r}" if errs else ""),
        concurrency=args.concurrency,
        ok=len(ok_lens),
        errors=len(errs),
        first_error=(repr(errs[0]) if errs else None),
        wall_s=round(wall, 2),
    )

    await _safe_close(backend)
    _write_report(args, report)
    return 0 if all(c["status"] == "PASS" for c in report.checks) else 2


def _pct(values: list[float]) -> dict:
    if not values:
        return {"p50": None, "p90": None, "p95": None, "mean": None, "min": None, "max": None}
    qs = statistics.quantiles(values, n=100, method="inclusive")
    return {
        "p50": qs[49],
        "p95": qs[94],
        "p99": qs[98],
        "mean": statistics.fmean(values),
        "min": min(values),
        "max": max(values),
    }


async def _safe_close(backend: Qwen3TTSBackend) -> None:
    try:
        await backend.close()
    except Exception:  # noqa: BLE001
        pass


def _write_report(args: argparse.Namespace, report: Report) -> None:
    payload = {
        "args": vars(args),
        "checks": report.checks,
        "artifacts": report.artifacts,
        "summary": report.summary(),
    }
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = RESULTS_DIR / "report.json"
    out.write_text(json.dumps(payload, indent=2))
    print("\n" + report.summary())
    print(f"report: {out}")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--device", default="auto")
    p.add_argument("--dtype", default="bf16")
    p.add_argument("--model", default=None, help="override model id/path")
    p.add_argument("--compile", action="store_true")
    p.add_argument("--iters", type=int, default=10, help="warm latency iterations")
    p.add_argument("--concurrency", type=int, default=3)
    p.add_argument("--latency-budget-s", type=float, default=30.0)
    return p.parse_args()


if __name__ == "__main__":
    raised = asyncio.run(main(_parse_args()))
    sys.exit(raised)
