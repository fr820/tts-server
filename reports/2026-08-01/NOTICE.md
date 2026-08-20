# NOTICE (added 2026-08-20)

The qwen3 numbers in this directory predate any verified real-GPU execution
on the machine that produced them and are **not reproducible from artifacts
kept here**: per-request raw samples were not persisted (see REVIEW.md),
part of the stress table was reconstructed after the fact (see git history),
and the host at the time had neither the qwen3 environment nor the model
weights to rerun them. Treat everything in `2026-08-01/` as unverified.

Superseded by `reports/2026-08-20/` — a full A-B rerun on the same repo
benchmarks (bench_http c=1 x30, gpu_smoke, gpu_validate) on an NVIDIA A10
with a pinned environment (`env.txt`), per-request artifacts, server-side
/metrics histograms, and reproduction commands.
