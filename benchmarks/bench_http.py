"""HTTP benchmark against POST /v1/audio/speech (streaming pcm).

Usage:
  uv run python benchmarks/bench_http.py --backend mock --concurrency 20 --requests 200
"""

from __future__ import annotations

import argparse
import asyncio
import json
import statistics
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


def load_texts(path: str) -> list[str]:
    """Load the ``text`` field from each line of a JSONL file (blanks skipped).

    Used to rotate real benchmark inputs across requests so length is the only
    controlled variable in a concurrency sweep.
    """
    texts: list[str] = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        texts.append(json.loads(line)["text"])
    if not texts:
        raise SystemExit(f"--texts-file {path} contained no records")
    return texts


def _p99(values: list[float]) -> float | None:
    """99th percentile (computed locally so common.py / bench_stress stay untouched)."""
    vals = [v for v in values if v is not None]
    if not vals:
        return None
    if len(vals) == 1:
        return vals[0]
    return statistics.quantiles(vals, n=100, method="inclusive")[98]


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

    texts = load_texts(args.texts_file) if args.texts_file else None

    async def bounded(client: httpx.AsyncClient, i: int) -> None:
        text = texts[i % len(texts)] if texts else args.text
        async with semaphore:
            try:
                outcomes.append(
                    await one_request(client, url, text, args.sample_rate)
                )
            except (httpx.HTTPError, ConnectionError, TimeoutError) as exc:
                print(f"request failed: {exc!r}", file=sys.stderr)
                outcomes.append(None)

    started = time.perf_counter()
    async with httpx.AsyncClient(timeout=args.timeout) as client:
        await asyncio.gather(*(bounded(client, i) for i in range(args.requests)))
    wall = time.perf_counter() - started

    ok = [o for o in outcomes if o is not None]
    ttfa = [o["ttfa_ms"] for o in ok]
    lat = [o["latency_ms"] for o in ok]
    results = {
        "bench": "http",
        "backend": args.backend,
        "dataset": args.dataset,
        "texts_file": str(args.texts_file) if args.texts_file else None,
        "text": None if texts else args.text,
        "concurrency": args.concurrency,
        "requests": args.requests,
        "client_timeout_s": args.timeout,
        "failures": args.requests - len(ok),
        "throughput_rps": round(len(ok) / wall, 2),
        "ttfa_ms": {**percentiles(ttfa), "p99": _p99(ttfa)},
        "latency_ms": {**percentiles(lat), "p99": _p99(lat)},
        "rtf": percentiles([o["rtf"] for o in ok if o["rtf"] is not None]),
    }
    json_path, md_path = write_results(results, out_dir=args.out_dir)
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
    parser.add_argument("--texts-file", default=None,
                        help="JSONL with {\"text\": ...}; rotates across requests (overrides --text)")
    parser.add_argument("--out-dir", default="benchmarks/results",
                        help="directory for result JSON/MD")
    parser.add_argument("--dataset", default=None,
                        help="dataset label for result metadata, e.g. bench-text-v1-en")
    parser.add_argument("--timeout", type=float, default=180.0,
                        help="client timeout (s); keep above the server request_timeout to capture 504s")
    parser.add_argument("--sample-rate", type=int, default=24000,
                        help="audio sample rate used to compute RTF (must match the server's)")
    asyncio.run(run(parser.parse_args()))


if __name__ == "__main__":
    main()
