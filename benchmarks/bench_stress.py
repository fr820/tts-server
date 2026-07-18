"""Escalating-concurrency stress test against a live tts-server.

Drives the OpenAI-compatible streaming endpoint (`POST /v1/audio/speech`,
`response_format=pcm`) at several concurrency levels back-to-back and reports,
per level: throughput, p50/p95/p99 latency, time-to-first-audio, and the
failure rate (HTTP non-200 / client errors). It also samples the server's
`tts_gpu_memory_peak_mb` metric before/after to surface VRAM growth (leaks).

Closed workload: `N` requests released through a bounded semaphore of size `C`
(`C` concurrent in-flight). On a single serialized GPU this is the right shape
for finding the queueing/timeout knee.

Numbers are REAL model inference. Never labeled mock.

Usage:
  uv run python benchmarks/bench_stress.py --url http://localhost:8000 \
      --levels 1:4,4:8,8:8,16:16 --sustained 8:24
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
import sys
import time
from pathlib import Path

import httpx

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from benchmarks.common import percentiles, write_results  # noqa: E402

DEFAULT_TEXT = "语音合成压力测试：这是一段用于负载测试的中文文本。"
RESULTS_DIR = Path("benchmarks/results")
_PEAK_RE = re.compile(r"^tts_gpu_memory_peak_mb\s+([0-9.]+)", re.M)


async def _gpu_peak(client: httpx.AsyncClient, url: str) -> float | None:
    try:
        r = await client.get(f"{url}/metrics", timeout=10.0)
        m = _PEAK_RE.search(r.text)
        return float(m.group(1)) if m else None
    except Exception:
        return None


async def _one(
    client: httpx.AsyncClient, url: str, text: str, timeout: float
) -> dict:
    start = time.perf_counter()
    ttfa = None
    nbytes = 0
    status = None
    err = None
    try:
        async with client.stream(
            "POST",
            f"{url}/v1/audio/speech",
            json={"input": text, "response_format": "pcm"},
            timeout=timeout,
        ) as r:
            status = r.status_code
            async for chunk in r.aiter_bytes():
                if ttfa is None and chunk:
                    ttfa = time.perf_counter() - start
                nbytes += len(chunk)
    except Exception as exc:  # client timeout / connection error
        err = exc.__class__.__name__
    elapsed = time.perf_counter() - start
    return {
        "ttfa_ms": (ttfa * 1000) if ttfa else None,
        "latency_ms": elapsed * 1000,
        "ok": status == 200 and err is None,
        "status": status,
        "err": err,
        "bytes": nbytes,
    }


async def _level(
    client: httpx.AsyncClient, url: str, text: str, c: int, n: int, timeout: float
) -> tuple[list[dict], float]:
    sem = asyncio.Semaphore(c)

    async def bounded() -> dict:
        async with sem:
            return await _one(client, url, text, timeout)

    start = time.perf_counter()
    results = await asyncio.gather(*(bounded() for _ in range(n)))
    return results, time.perf_counter() - start


def _summarize(name: str, c: int, n: int, results: list[dict], wall: float) -> dict:
    ok = [r for r in results if r["ok"]]
    fail = [r for r in results if not r["ok"]]
    lat = [r["latency_ms"] for r in ok]
    ttfa = [r["ttfa_ms"] for r in ok if r["ttfa_ms"] is not None]
    errs = {}
    for r in fail:
        key = f"HTTP {r['status']}" if r["status"] else (r["err"] or "error")
        errs[key] = errs.get(key, 0) + 1
    return {
        "name": name,
        "concurrency": c,
        "requests": n,
        "ok": len(ok),
        "failures": len(fail),
        "failure_rate": round(len(fail) / n, 3) if n else 0,
        "throughput_rps": round(len(ok) / wall, 3) if wall else 0,
        "wall_s": round(wall, 1),
        "latency_ms": percentiles(lat),
        "ttfa_ms": percentiles(ttfa),
        "error_breakdown": errs or None,
    }


def _parse_levels(s: str) -> list[tuple[int, int]]:
    out = []
    for part in s.split(","):
        c, _, n = part.partition(":")
        out.append((int(c), int(n)))
    return out


async def main(args: argparse.Namespace) -> int:
    timeout = args.timeout
    levels = [(n, "escalate", c, r) for n, (c, r) in enumerate(_parse_levels(args.levels), 1)]
    if args.sustained:
        c, r = _parse_levels(args.sustained)[0]
        levels.append((len(levels) + 1, "sustained", c, r))

    rows: list[dict] = []
    async with httpx.AsyncClient() as client:
        print(f"target={args.url}  text_bytes={len(args.text.encode())}  client_timeout={timeout}s")
        print(f"{'level':<22}{'C':>4}{'N':>5}{'ok':>5}{'fail':>6}{'rps':>8}"
              f"{'lat p50':>10}{'lat p95':>10}{'ttfa p95':>10}")
        print("-" * 90)
        peak0 = await _gpu_peak(client, args.url)
        for idx, tag, c, n in levels:
            results, wall = await _level(client, args.url, args.text, c, n, timeout)
            s = _summarize(f"L{idx}:{tag}", c, n, results, wall)
            peak_after = await _gpu_peak(client, args.url)
            s["gpu_peak_mb_after"] = peak_after
            rows.append(s)
            lat = s["latency_ms"]
            ttfa = s["ttfa_ms"]
            print(f"{s['name']:<22}{c:>4}{n:>5}{s['ok']:>5}{s['failures']:>6}"
                  f"{s['throughput_rps']:>8.3f}"
                  f"{(lat['p50'] or 0):>9.0f}ms{(lat['p95'] or 0):>9.0f}ms"
                  f"{(ttfa['p95'] or 0):>9.0f}ms")
            if s["error_breakdown"]:
                print(f"    errors: {s['error_breakdown']}")
        peak_end = await _gpu_peak(client, args.url)

    summary = {
        "bench": "stress",
        "backend": args.backend,
        "url": args.url,
        "client_timeout_s": timeout,
        "levels": rows,
        "gpu_peak_mb_start": peak0,
        "gpu_peak_mb_end": peak_end,
        "gpu_growth_mb": (round(peak_end - peak0, 1) if peak0 is not None and peak_end is not None else None),
    }
    json_path, md_path = write_results(summary)
    # write_results names by concurrency; rewrite a stable stress filename
    stress_json = RESULTS_DIR / "stress-qwen3.json"
    stress_md = RESULTS_DIR / "stress-qwen3.md"
    stress_json.write_text(json.dumps(summary, indent=2))
    stress_md.write_text(_markdown(summary))
    print(f"\ngpu_peak_mb: start={peak0} end={peak_end} growth={summary['gpu_growth_mb']} MB")
    print(f"saved: {stress_json} and {stress_md}")
    return 0


def _markdown(s: dict) -> str:
    lines = [
        f"# Stress test — backend `{s['backend']}`",
        "",
        f"- client timeout: {s['client_timeout_s']}s",
        f"- GPU peak: start={s['gpu_peak_mb_start']} MB, end={s['gpu_peak_mb_end']} MB "
        f"(growth {s['gpu_growth_mb']} MB — near-zero growth means no VRAM leak)",
        "",
        "| level | C | N | ok | fail | rps | lat p50 ms | lat p95 ms | lat p99 ms | ttfa p95 ms | errors |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for lv in s["levels"]:
        lat = lv["latency_ms"]
        ttfa = lv["ttfa_ms"]
        lines.append(
            f"| {lv['name']} | {lv['concurrency']} | {lv['requests']} | {lv['ok']} | "
            f"{lv['failures']} | {lv['throughput_rps']} | "
            f"{(lat['p50'] or 0):.0f} | {(lat['p95'] or 0):.0f} | {(lat['p99'] or 0):.0f} | "
            f"{(ttfa['p95'] or 0):.0f} | {lv['error_breakdown'] or ''} |"
        )
    return "\n".join(lines) + "\n"


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--url", default="http://localhost:8000")
    p.add_argument("--backend", default="qwen3")
    p.add_argument("--levels", default="1:4,4:8,8:8,16:16",
                   help="comma list of C:N concurrency:request-count")
    p.add_argument("--sustained", default=None, help="extra sustained level C:N")
    p.add_argument("--text", default=DEFAULT_TEXT)
    p.add_argument("--timeout", type=float, default=180.0,
                   help="client timeout; keep > server request_timeout to capture 504s")
    return p.parse_args()


if __name__ == "__main__":
    sys.exit(asyncio.run(main(_parse_args())))
