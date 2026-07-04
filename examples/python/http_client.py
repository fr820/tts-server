"""Stream PCM from the OpenAI-compatible endpoint and report TTFA."""

import time

import httpx

BASE_URL = "http://localhost:8000"


def main() -> None:
    start = time.perf_counter()
    first_chunk_at = None
    total = 0
    with httpx.stream(
        "POST",
        f"{BASE_URL}/v1/audio/speech",
        json={"input": "Streaming pcm from python.", "response_format": "pcm"},
        timeout=60.0,
    ) as resp:
        resp.raise_for_status()
        for chunk in resp.iter_bytes():
            if first_chunk_at is None and chunk:
                first_chunk_at = time.perf_counter() - start
            total += len(chunk)
    print(f"TTFA: {first_chunk_at * 1000:.1f} ms, total bytes: {total}")


if __name__ == "__main__":
    main()
