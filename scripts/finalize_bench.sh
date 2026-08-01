#!/usr/bin/env bash
# Auto-finalize the 2025-08-01 qwen3 benchmark once the HTTP matrix finishes.
# Waits for 'matrix done', then runs bench_stress, writes env.txt, builds the
# summary, commits + pushes, and drops a FINALIZE_DONE marker. Runs in tmux so
# it completes autonomously (SSH-drop / Claude-restart safe).
set -uo pipefail
cd /home/ecs-user/tts-server
export HF_ENDPOINT=https://hf-mirror.com
OUT=reports/2025-08-01
MARKER="$OUT/FINALIZE_DONE"
LOG="$OUT/finalize.log"
: > "$LOG"
log(){ echo "[$(date -u +%H:%M:%S)] $*" | tee -a "$LOG"; }

log "finalize watcher started; waiting for HTTP matrix to finish..."
while true; do
  if grep -q "^matrix done" "$OUT/matrix_progress.txt" 2>/dev/null; then break; fi
  sleep 15
done
log "matrix done -> starting stress."

log "healthz: $(curl -s --max-time 5 http://localhost:8000/healthz)"

log "running bench_stress --levels 1:50,2:50,4:50,8:50 --timeout 180 ..."
uv run python benchmarks/bench_stress.py --levels 1:50,2:50,4:50,8:50 --timeout 180 \
    > "$OUT/stress.stdout.log" 2>&1 || log "bench_stress exit non-zero (trailing p99 KeyError is expected; JSON already saved)"
cp -f benchmarks/results/stress-qwen3.json "$OUT/" 2>/dev/null && log "copied stress json"
cp -f benchmarks/results/stress-qwen3.md  "$OUT/" 2>/dev/null && log "copied stress md"

COMMIT=$(git rev-parse HEAD)
DRIVER=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader -i 0 | tr -d ' ')
GPU=$(nvidia-smi --query-gpu=name --format=csv,noheader -i 0)
UNAME=$(uname -srvmo)
TORCH=$(.venv/bin/python -c "import torch;print(torch.__version__)" 2>/dev/null)
QT=$(.venv/bin/python -c "import qwen_tts;print(getattr(qwen_tts,'__version__','unknown'))" 2>/dev/null || echo unknown)
cat > "$OUT/env.txt" <<EOF
date: 2026-08-01 (matrix started 2026-08-01T04:54:32Z UTC)
dataset: bench-text-v1-en
backend: qwen3 (Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice, bf16, CUDA, emulated streaming)
host: bare-metal GPU server (no container)
gpu: $GPU
driver: $DRIVER
torch: $TORCH
qwen_tts: $QT
uname: $UNAME
HF_ENDPOINT: https://hf-mirror.com
git_commit_at_finalize: $COMMIT
layout: qwen3 server in tmux 'tts'; HTTP matrix c=1/2/4/8 in tmux 'bench'; bench_stress after matrix.
notes: HTTP matrix + stress use 50 req/level, client timeout 180s > server request_timeout 120s so 504s are server-originated capacity signals. Result JSONs unmodified.
EOF
log "wrote env.txt"

log "building summary..."
uv run python scripts/summarize_bench.py > "$OUT/summary.stdout.log" 2>&1 || log "summarize non-zero"

log "committing + pushing..."
git add reports/2025-08-01 scripts/run_http_matrix.sh scripts/summarize_bench.py scripts/finalize_bench.sh
git commit -m "bench(qwen3): stress + env + summary for 2025-08-01 run" \
    -m "Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>" >/dev/null 2>&1 && log "committed" || log "nothing new to commit"
git push origin main > "$OUT/push.log" 2>&1 && log "PUSH OK" || log "PUSH FAILED (see push.log)"

echo "FINALIZE_DONE $(date -u +%FT%TZ)" > "$MARKER"
log "FINALIZE_DONE"
