"""HTTP benchmark against POST /v1/audio/speech (streaming pcm).

Usage:
  uv run python benchmarks/bench_http.py --backend mock --concurrency 20 --requests 200
"""

from __future__ import annotations

import argparse
import asyncio
import sys
import time
from pathlib import Path

import httpx

if __package__ in (None, ""):
    # Allow running as `python benchmarks/bench_http.py` without installing
    # the repo root on sys.path first (the script's own dir is added by
    # default, not its parent).
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from benchmarks.common import percentiles, write_results

DEFAULT_TEXT = "Hello, welcome to our realtime voice agent demonstration."


async def one_request(
    client: httpx.AsyncClient, url: str, text: str, sample_rate: int
) -> dict:
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
    audio_s = audio_bytes / (2 * sample_rate)
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
                outcomes.append(
                    await one_request(client, url, args.text, args.sample_rate)
                )
            except (httpx.HTTPError, ConnectionError, TimeoutError) as exc:
                print(f"request failed: {exc!r}", file=sys.stderr)
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
    parser.add_argument("--sample-rate", type=int, default=24000,
                        help="audio sample rate used to compute RTF (must match the server's)")
    asyncio.run(run(parser.parse_args()))


if __name__ == "__main__":
    main()
