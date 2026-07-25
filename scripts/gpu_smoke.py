"""End-to-end GPU smoke test against the running tts-server HTTP API.

Unlike ``scripts/gpu_validate.py`` (which drives the backend object directly),
this talks to the **deployed server** over HTTP — exercising the full path a
client takes: ``GET /healthz`` and ``GET /v1/models``, then ``POST
/v1/audio/speech`` with real text. It validates the returned WAV is real,
non-silent speech, and reports latency, GPU memory, and GPU utilization.

Runs on the host with stdlib only (no torch/httpx dependency), so it works
against a containerized server on ``http://localhost:8000`` exactly as a client
would. GPU memory/utilization come from the server's ``/healthz`` (process
allocations, when the backend reports them) and from sampling ``nvidia-smi``
(device-wide) during synthesis.

Reproducible:
  # server already running (mock or qwen3):
  python3 scripts/gpu_smoke.py --url http://localhost:8000
  python3 scripts/gpu_smoke.py --url http://localhost:8000 --iters 10

NOTE: this validates the TTS speech-synthesis API end to end and confirms the
synthesized audio is real (non-silence, sane duration). It does not run speech
*recognition* (ASR) — the server has no STT surface. Never label a mock-backend
run as real model inference.
"""

from __future__ import annotations

import argparse
import array
import json
import statistics
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
import wave
from pathlib import Path

RESULTS_DIR = Path("benchmarks/results/gpu_smoke")
DEFAULT_TEXT = "The quick brown fox jumps over the lazy dog. Welcome aboard."


class Report:
    def __init__(self) -> None:
        self.checks: list[dict] = []
        self.artifacts: dict[str, str] = {}

    def add(self, name: str, passed: bool, detail: str, **extra) -> None:
        self.checks.append(
            {"check": name, "status": "PASS" if passed else "FAIL", "detail": detail, **extra}
        )
        print(f"[{'PASS' if passed else 'FAIL'}] {name}: {detail}")

    def summary(self) -> str:
        n_pass = sum(1 for c in self.checks if c["status"] == "PASS")
        n_fail = sum(1 for c in self.checks if c["status"] == "FAIL")
        return f"{n_pass} PASS / {n_fail} FAIL out of {len(self.checks)} checks"


def _pct(values: list[float]) -> dict:
    if not values:
        return {"p50": None, "p90": None, "p95": None, "mean": None,
                "min": None, "max": None}
    if len(values) == 1:
        # statistics.quantiles needs >=2 points; a single sample is its own
        # every-percentile value (e.g. a fast mock run yields few GPU samples).
        v = values[0]
        return {"p50": v, "p90": v, "p95": v, "mean": v, "min": v, "max": v}
    qs = statistics.quantiles(values, n=100, method="inclusive")
    return {"p50": qs[49], "p90": qs[89], "p95": qs[94],
            "mean": statistics.fmean(values), "min": min(values), "max": max(values)}


def _http_json(url: str, timeout: float = 30.0) -> dict:
    with urllib.request.urlopen(url, timeout=timeout) as resp:
        return json.loads(resp.read().decode())


def _post_speech(url: str, text: str, timeout: float = 300.0) -> tuple[bytes, float]:
    """POST /v1/audio/speech (wav). Returns (wav_bytes, elapsed_s)."""
    body = json.dumps({"input": text, "response_format": "wav"}).encode()
    req = urllib.request.Request(
        f"{url}/v1/audio/speech", data=body,
        headers={"Content-Type": "application/json"}, method="POST",
    )
    start = time.perf_counter()
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        data = resp.read()
    return data, time.perf_counter() - start


def _audio_stats(wav_bytes: bytes) -> tuple[dict, int]:
    """Parse the WAV, return (stats, sample_rate). Proves the file is valid +
    non-silent. Raises on a corrupt/truncated WAV (caught by the caller)."""
    import io

    with wave.open(io.BytesIO(wav_bytes), "rb") as wf:
        sample_rate = wf.getframerate()
        n_channels = wf.getnchannels()
        sampwidth = wf.getsampwidth()
        pcm = wf.readframes(wf.getnframes())
    samples = array.array("h")
    samples.frombytes(pcm)
    n = len(samples)
    if n == 0:
        return {"samples": 0, "duration_s": 0.0, "peak": 0, "rms": 0.0}, sample_rate
    peak = max(abs(s) for s in samples)
    mean_sq = sum(s * s for s in samples) / n
    return {
        "samples": n,
        "channels": n_channels,
        "sampwidth": sampwidth,
        "duration_s": round(n / sample_rate, 3),
        "peak": peak,
        "rms": round(mean_sq**0.5, 1),
        "peak_ratio": round(peak / 32767, 4),
    }, sample_rate


def _nvidia_smi(*fields: str) -> str | None:
    """Run `nvidia-smi --query-gpu=... --format=csv,noheader,nounits`. None if
    nvidia-smi is absent (e.g. a non-GPU host)."""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=" + ",".join(fields),
             "--format=csv,noheader,nounits", "-i", "0"],
            capture_output=True, text=True, timeout=10,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None
    if out.returncode != 0:
        return None
    return out.stdout.strip() or None


class GpuSampler:
    """Sample device-wide GPU utilization + memory while synthesis runs."""

    def __init__(self, interval_ms: int) -> None:
        self.interval = interval_ms / 1000
        self.utils: list[float] = []
        self.mem_used: list[float] = []
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def __enter__(self) -> "GpuSampler":
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=5)

    def _run(self) -> None:
        while not self._stop.is_set():
            line = _nvidia_smi("utilization.gpu", "memory.used")
            if line:
                try:
                    util, mem = line.split(",")
                    self.utils.append(float(util))
                    self.mem_used.append(float(mem))
                except ValueError:
                    pass
            time.sleep(self.interval)

    def summary(self) -> dict:
        return {
            "util_pct": _pct(self.utils) if self.utils else {},
            "mem_used_mb": _pct(self.mem_used) if self.mem_used else {},
            "samples": len(self.utils),
        }


def main(args: argparse.Namespace) -> int:
    report = Report()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    url = args.url.rstrip("/")

    # ---- device probe (host nvidia-smi) ----
    dev = _nvidia_smi("name", "memory.total", "driver_version")
    if dev:
        report.add("device", True, f"nvidia-smi: {dev}", device=dev)
    else:
        report.add("device", False, "nvidia-smi not found on host; cannot report GPU util/mem")

    # ---- health ----
    try:
        health = _http_json(f"{url}/healthz", timeout=10)
        bh = health.get("backend", {})
        loaded = bh.get("loaded")
        gpu_mem = bh.get("gpu_memory_mb")
        gpu_peak = bh.get("gpu_memory_peak_mb")
        report.add(
            "health",
            health.get("status") == "ok" and loaded,
            f"status={health.get('status')} backend={health.get('backend_name')} loaded={loaded}"
            + (f" gpu_mem={gpu_mem:.0f}MB" if gpu_mem is not None else " gpu_mem=n/a"),
            backend=health.get("backend_name"),
            loaded=loaded,
            gpu_memory_mb=gpu_mem,
            gpu_memory_peak_mb=gpu_peak,
        )
    except (urllib.error.URLError, OSError) as exc:
        report.add("health", False, f"/healthz unreachable: {exc!r}")
        _write_report(args, report)
        return 1

    is_mock = health.get("backend_name") == "mock"
    if is_mock:
        report.artifacts["note"] = "mock backend — synthetic audio, not real model inference"

    # ---- models ----
    try:
        models = _http_json(f"{url}/v1/models", timeout=10)
        ids = [m.get("id") for m in models.get("data", [])]
        report.add("models", len(ids) > 0, f"models={ids}", models=ids)
    except (urllib.error.URLError, OSError) as exc:
        report.add("models", False, f"/v1/models unreachable: {exc!r}")

    # ---- single synthesis + real-audio validation ----
    try:
        wav_bytes, elapsed = _post_speech(url, args.text)
        stats, sample_rate = _audio_stats(wav_bytes)
        wav_path = RESULTS_DIR / "sample.wav"
        wav_path.write_bytes(wav_bytes)
        passed = (
            len(wav_bytes) > 44
            and stats["samples"] > 0
            and stats["peak"] > 1000           # not digital silence
            and 0.2 <= stats["duration_s"] <= 60
        )
        report.add(
            "speech",
            passed,
            f"{len(wav_bytes)}B sr={sample_rate} dur={stats['duration_s']}s peak={stats['peak']} rms={stats['rms']}",
            latency_ms=round(elapsed * 1000, 1),
            sample_rate=sample_rate,
            **stats,
        )
        report.artifacts["wav"] = str(wav_path)
    except Exception as exc:  # noqa: BLE001
        report.add("speech", False, f"synthesis failed: {exc!r}")
        report.artifacts["error"] = repr(exc)
        _write_report(args, report)
        return 1

    # ---- latency / RTF (warm) + GPU sampling during the loop ----
    latencies: list[float] = []
    audio_lens: list[float] = []
    with GpuSampler(args.sample_interval_ms) as sampler:
        for _ in range(args.iters):
            data, dt = _post_speech(url, args.text)
            latencies.append(dt)
            audio_lens.append(_duration(data))
    rtfs = [l / a for l, a in zip(latencies, audio_lens) if a > 0]
    lat_pct = _pct(latencies)
    rtf_pct = _pct(rtfs)
    report.add(
        "latency",
        lat_pct["p50"] is not None and lat_pct["p50"] < args.latency_budget_s,
        f"warm p50={lat_pct['p50']*1000:.0f}ms p95={lat_pct['p95']*1000:.0f}ms"
        f" RTF p50={(rtf_pct['p50'] or 0):.3f} (budget {args.latency_budget_s}s)",
        latency_ms={k: round(v * 1000, 1) for k, v in lat_pct.items() if v is not None},
        rtf={k: round(v, 3) for k, v in rtf_pct.items() if v is not None},
    )

    # ---- GPU utilization + device memory during synthesis ----
    s = sampler.summary()
    util = s["util_pct"]
    mem = s["mem_used_mb"]
    if util.get("p50") is not None:
        report.add(
            "gpu_util",
            True,
            f"util p50={util['p50']:.0f}% peak={util['max']:.0f}%"
            f" | dev mem used p50={mem['p50']:.0f}MB peak={mem['max']:.0f}MB"
            f" (over {s['samples']} samples)",
            utilization_pct={k: round(v, 1) for k, v in util.items()},
            device_memory_used_mb={k: round(v, 1) for k, v in mem.items()},
            samples=s["samples"],
        )
    else:
        report.add("gpu_util", not is_mock, "no nvidia-smi samples (device probe failed)")

    _write_report(args, report)
    return 0 if all(c["status"] == "PASS" for c in report.checks) else 2


def _duration(wav_bytes: bytes) -> float:
    """Audio duration of a returned WAV (used for per-iteration RTF)."""
    import io

    try:
        with wave.open(io.BytesIO(wav_bytes), "rb") as wf:
            return wf.getnframes() / wf.getframerate()
    except Exception:  # noqa: BLE001
        return 0.0


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
    p.add_argument("--url", default="http://localhost:8000")
    p.add_argument("--text", default=DEFAULT_TEXT)
    p.add_argument("--iters", type=int, default=5, help="warm latency iterations")
    p.add_argument("--sample-interval-ms", type=int, default=200,
                   help="nvidia-smi sampling cadence during synthesis")
    p.add_argument("--latency-budget-s", type=float, default=30.0)
    return p.parse_args()


if __name__ == "__main__":
    sys.exit(main(_parse_args()))
