# Benchmark: http — backend `qwen3`

- requests: 50, concurrency: 8
- failures: 45
- throughput: 0.01 req/s

| metric | p50 | p90 | p95 | mean |
|---|---|---|---|---|
| ttfa_ms | 66510.28 | 105053.23 | 110210.88 | 68020.76 |
| latency_ms | 66522.51 | 105061.12 | 110218.78 | 68029.08 |
| rtf | 4.57 | 6.35 | 6.56 | 4.38 |
