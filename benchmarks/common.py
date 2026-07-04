"""Shared benchmark helpers: percentile stats and JSON+Markdown output."""

from __future__ import annotations

import json
import statistics
from pathlib import Path

MOCK_NOTE = "mock backend — synthetic audio, not real model inference"


def percentiles(values: list[float]) -> dict:
    if not values:
        return {"p50": None, "p90": None, "p95": None, "mean": None,
                "min": None, "max": None}
    qs = statistics.quantiles(values, n=100, method="inclusive")
    return {
        "p50": qs[49],
        "p90": qs[89],
        "p95": qs[94],
        "mean": statistics.fmean(values),
        "min": min(values),
        "max": max(values),
    }


def _markdown(results: dict) -> str:
    lines = [
        f"# Benchmark: {results['bench']} — backend `{results['backend']}`",
        "",
        f"- requests: {results['requests']}, concurrency: {results['concurrency']}",
        f"- failures: {results['failures']}",
        f"- throughput: {results.get('throughput_rps', 'n/a')} req/s",
    ]
    if results.get("note"):
        lines.append(f"- **note: {results['note']}**")
    lines += ["", "| metric | p50 | p90 | p95 | mean |", "|---|---|---|---|---|"]
    for key in ("ttfa_ms", "latency_ms", "rtf"):
        if key in results and results[key]["p50"] is not None:
            s = results[key]
            lines.append(
                f"| {key} | {s['p50']:.2f} | {s['p90']:.2f} "
                f"| {s['p95']:.2f} | {s['mean']:.2f} |"
            )
    return "\n".join(lines) + "\n"


def write_results(results: dict, out_dir: str = "benchmarks/results") -> tuple[Path, Path]:
    if results["backend"] == "mock":
        results["note"] = MOCK_NOTE
    directory = Path(out_dir)
    directory.mkdir(parents=True, exist_ok=True)
    name = f"{results['bench']}-{results['backend']}-c{results['concurrency']}"
    json_path = directory / f"{name}.json"
    md_path = directory / f"{name}.md"
    json_path.write_text(json.dumps(results, indent=2))
    md_path.write_text(_markdown(results))
    return json_path, md_path
