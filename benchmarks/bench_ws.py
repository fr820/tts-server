"""WebSocket benchmark against the ElevenLabs-style stream-input endpoint.

Usage:
  uv run python benchmarks/bench_ws.py --backend mock --requests 20 [--streaming-input]
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path

import websockets

if __package__ in (None, ""):
    # Allow running as `python benchmarks/bench_ws.py` without installing
    # the repo root on sys.path first (the script's own dir is added by
    # default, not its parent).
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

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
        except (OSError, TimeoutError, websockets.exceptions.WebSocketException) as exc:
            print(f"session failed: {exc!r}", file=sys.stderr)
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
    parser.add_argument("--text", default=DEFAULT_TEXT)
    parser.add_argument("--streaming-input", action="store_true")
    asyncio.run(run(parser.parse_args()))


if __name__ == "__main__":
    main()
