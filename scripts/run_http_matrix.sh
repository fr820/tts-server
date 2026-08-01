#!/usr/bin/env bash
# Run the HTTP concurrency matrix c=1,2,4,8 (50 requests each) against the live
# qwen3 server, with per-level GPU monitoring and a per-level git commit.
# One GPU + serialized inference => levels run sequentially. Each level is
# ~50 * ~24s = ~20 min; the whole matrix is ~70-90 min. Run in the background.
set -uo pipefail
cd /home/ecs-user/tts-server
export HF_ENDPOINT=https://hf-mirror.com

OUT=reports/2026-08-01
mkdir -p "$OUT/gpu_monitor"
PROG="$OUT/matrix_progress.txt"
: > "$PROG"
echo "matrix start $(date -u +%FT%TZ)" >> "$PROG"

baseline_vram() { nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 0 | tr -d ' \n'; }

for C in 1 2 4 8; do
  echo "===== level c=$C $(date -u +%FT%TZ) =====" | tee -a "$PROG"
  BASE=$(baseline_vram)
  MON="$OUT/gpu_monitor/http_c${C}.csv"
  : > "$MON"
  # start the monitor (user-specified query)
  nvidia-smi --query-gpu=timestamp,memory.used,utilization.gpu --format=csv -l 5 > "$MON" 2>/dev/null &
  MONPID=$!
  echo "[c$C] baseline_vram_mb=$BASE" | tee -a "$PROG"

  T0=$(date +%s)
  uv run python benchmarks/bench_http.py --backend qwen3 --concurrency "$C" --requests 50 \
      --texts-file benchmarks/data/texts_main.jsonl --out-dir "$OUT" \
      --dataset bench-text-v1-en --url http://localhost:8000 \
      > "$OUT/http-qwen3-c${C}.stdout.log" 2>&1
  RC=$?
  T1=$(date +%s)
  kill "$MONPID" 2>/dev/null || true

  PEAK=$(awk -F',' 'NF>=2{m=$2;gsub(/[" ]/,"",m);if(m+0>mx)mx=m+0}END{print mx+0}' "$MON")
  echo "[c$C] rc=$RC wall_s=$((T1-T0)) peak_vram_mb=$PEAK" | tee -a "$PROG"

  # wait for device VRAM to settle back near baseline before the next level
  for _ in $(seq 1 30); do
    NOW=$(baseline_vram)
    if [ "$((NOW))" -le "$((BASE+200))" ]; then break; fi
    sleep 5
  done

  git add reports/2026-08-01 >/dev/null 2>&1
  git commit -m "bench(qwen3): http matrix c=$C (50 req, bench-text-v1-en)" \
      -m "Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>" >/dev/null 2>&1 \
      && echo "[c$C] committed" | tee -a "$PROG" || echo "[c$C] nothing to commit" | tee -a "$PROG"
done

echo "matrix done $(date -u +%FT%TZ)" | tee -a "$PROG"
echo "MATRIX_RUN_COMPLETE"
