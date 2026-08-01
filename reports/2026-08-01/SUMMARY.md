# HTTP concurrency matrix — backend `qwen3`, dataset `bench-text-v1-en`

Per level: 50 requests, texts rotated from texts_main.jsonl (35–55 words). Client timeout 180s > server request_timeout 120s, so 504s are server-originated capacity signals (recorded as failures), not client cutoffs.

| c | ok | fail | err% | rps | lat p50 | lat p95 | lat p99 | ttfa p50 | ttfa p95 | peak VRAM |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 50 | 0 | 0.0 | 0.04 | 25297ms | 37383ms | 39139ms | 25290ms | 37368ms | 5785MB |
| 2 | 50 | 0 | 0.0 | 0.04 | 53420ms | 73824ms | 80590ms | 53410ms | 73805ms | 5807MB |
| 4 | 37 | 13 | 26.0 | 0.03 | 97069ms | 117486ms | 119587ms | 97059ms | 117477ms | 5943MB |
| 8 | 5 | 45 | 90.0 | 0.01 | 66523ms | 110219ms | 114345ms | 66510ms | 110211ms | 6415MB |

## bench_stress (1:50,2:50,4:50,8:50, timeout 180s)

| level | c | n | ok | fail | err% | rps | lat p50 | lat p95 | lat p99 | gpu_peak_after MB |
|---|---|---|---|---|---|---|---|---|---|---|
| L1:escalate | 1 | 50 | 50 | 0 | 0.0 | 0.108 | 8491ms | 16760ms | n/a | n/a |
| L2:escalate | 2 | 50 | 50 | 0 | 0.0 | 0.11 | 17212ms | 24482ms | n/a | n/a |
| L3:escalate | 4 | 50 | 50 | 0 | 0.0 | 0.105 | 35876ms | 50727ms | n/a | n/a |
| L4:escalate | 8 | 50 | 50 | 0 | 0.0 | 0.108 | 73509ms | 80835ms | n/a | n/a |

## gpu_smoke (real-audio sanity)

- speech: 195884B sr=24000 dur=4.08s peak=12479 rms=2172.7
- warm p50=7539ms p95=8032ms RTF p50=1.497 (budget 30.0s)
- summary: 6 PASS / 0 FAIL out of 6 checks
