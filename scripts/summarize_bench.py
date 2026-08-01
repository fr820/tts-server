#!/usr/bin/env python3
"""Assemble the HTTP-matrix summary table from per-level result JSONs and the
nvidia-smi monitor CSVs. Run after the matrix completes:

    uv run python scripts/summarize_bench.py

Reads reports/2026-08-01/http-qwen3-c{1,2,4,8}.json + gpu_monitor/http_c*.csv,
prints a markdown table, and writes reports/2026-08-01/SUMMARY.md.
"""
from __future__ import annotations

import csv
import json
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "reports" / "2025-08-01"
LEVELS = [1, 2, 4, 8]


def peak_vram_mb(csv_path: Path) -> float | None:
    if not csv_path.exists():
        return None
    mx = 0.0
    with csv_path.open() as f:
        for row in csv.reader(f):
            if len(row) < 2:
                continue
            m = row[1].strip().strip('"').strip().split()[0]  # "5781 MiB" -> "5781"
            try:
                mx = max(mx, float(m))
            except ValueError:
                continue
    return mx or None


def fmt(v, unit="ms", nd=0):
    if v is None:
        return "n/a"
    return f"{v:.{nd}f}{unit}"


rows = []
for c in LEVELS:
    j = OUT / f"http-qwen3-c{c}.json"
    mon = OUT / "gpu_monitor" / f"http_c{c}.csv"
    if not j.exists():
        rows.append((c, None))
        continue
    d = json.loads(j.read_text())
    lat = d["latency_ms"]
    ttfa = d["ttfa_ms"]
    reqs = d["requests"]
    fail = d["failures"]
    rows.append((c, {
        "reqs": reqs,
        "ok": reqs - fail,
        "fail": fail,
        "err_pct": round(100 * fail / reqs, 1) if reqs else 0,
        "rps": d.get("throughput_rps"),
        "lat_p50": lat.get("p50"), "lat_p95": lat.get("p95"), "lat_p99": lat.get("p99"),
        "ttfa_p50": ttfa.get("p50"), "ttfa_p95": ttfa.get("p95"),
        "peak_vram": peak_vram_mb(mon),
    }))

lines = ["# HTTP concurrency matrix — backend `qwen3`, dataset `bench-text-v1-en`", "",
         "Per level: 50 requests, texts rotated from texts_main.jsonl (35–55 words). "
         "Client timeout 180s > server request_timeout 120s, so 504s are server-originated "
         "capacity signals (recorded as failures), not client cutoffs.", "",
         "| c | ok | fail | err% | rps | lat p50 | lat p95 | lat p99 | ttfa p50 | ttfa p95 | peak VRAM |",
         "|---|---|---|---|---|---|---|---|---|---|---|"]
for c, d in rows:
    if d is None:
        lines.append(f"| {c} | - | - | - | - | _missing_ | | | | | |")
        continue
    lines.append(
        f"| {c} | {d['ok']} | {d['fail']} | {d['err_pct']} | {d['rps']} | "
        f"{fmt(d['lat_p50'])} | {fmt(d['lat_p95'])} | {fmt(d['lat_p99'])} | "
        f"{fmt(d['ttfa_p50'])} | {fmt(d['ttfa_p95'])} | {fmt(d['peak_vram'], 'MB')} |"
    )

# stress + gpu_smoke appendix if present
stress = OUT / "stress-qwen3.json"
if stress.exists():
    s = json.loads(stress.read_text())
    lines += ["", "## bench_stress (1:50,2:50,4:50,8:50, timeout 180s)", "",
              "| level | c | n | ok | fail | err% | rps | lat p50 | lat p95 | lat p99 | gpu_peak_after MB |",
              "|---|---|---|---|---|---|---|---|---|---|---|"]
    for lv in s.get("levels", []):
        lat = lv["latency_ms"]
        lines.append(
            f"| {lv['name']} | {lv['concurrency']} | {lv['requests']} | {lv['ok']} | "
            f"{lv['failures']} | {round(lv['failure_rate']*100,1)} | {lv['throughput_rps']} | "
            f"{fmt(lat.get('p50'))} | {fmt(lat.get('p95'))} | {fmt(lat.get('p99'))} | "
            f"{fmt(lv.get('gpu_peak_mb_after'), 'MB')} |"
        )
    if s.get("gpu_growth_mb") is not None:
        lines += ["", f"GPU peak growth across the whole stress run: **{s['gpu_growth_mb']} MB** "
                      f"(start {s['gpu_peak_mb_start']} -> end {s['gpu_peak_mb_end']} MB)"]

smoke = OUT / "gpu_smoke" / "report.json"
if smoke.exists():
    sr = json.loads(smoke.read_text())
    sp = next((c for c in sr["checks"] if c["check"] == "speech"), None)
    lt = next((c for c in sr["checks"] if c["check"] == "latency"), None)
    lines += ["", "## gpu_smoke (real-audio sanity)", ""]
    if sp:
        lines.append(f"- speech: {sp['detail']}")
    if lt:
        lines.append(f"- {lt['detail']}")
    lines.append(f"- summary: {sr['summary']}")

table = "\n".join(lines) + "\n"
(OUT / "SUMMARY.md").write_text(table)
print(table)
print(f"wrote {OUT/'SUMMARY.md'}")
