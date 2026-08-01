#!/usr/bin/env bash
# Re-run specific HTTP matrix levels (here c=4 and c=8) with the empty-body fix.
# Original c=4/c=8 crashed (TypeError on None ttfa) so no JSON was written.
# Usage: rerun_http_levels.sh 4 8
set -uo pipefail
cd /home/ecs-user/tts-server
export HF_ENDPOINT=https://hf-mirror.com
OUT=reports/2025-08-01
PROG="$OUT/rerun_progress.txt"
: > "$PROG"
echo "rerun start $(date -u +%FT%TZ)" >> "$PROG"
baseline_vram(){ nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 0 | tr -d ' \n'; }

for C in "$@"; do
  echo "===== rerun c=$C $(date -u +%FT%TZ) =====" | tee -a "$PROG"
  BASE=$(baseline_vram)
  MON="$OUT/gpu_monitor/http_c${C}.csv"; : > "$MON"
  nvidia-smi --query-gpu=timestamp,memory.used,utilization.gpu --format=csv -l 5 > "$MON" 2>/dev/null &
  MONPID=$!
  T0=$(date +%s)
  uv run python benchmarks/bench_http.py --backend qwen3 --concurrency "$C" --requests 50 \
      --texts-file benchmarks/data/texts_main.jsonl --out-dir "$OUT" \
      --dataset bench-text-v1-en --url http://localhost:8000 \
      > "$OUT/http-qwen3-c${C}.stdout.log" 2>&1
  RC=$?
  T1=$(date +%s)
  kill "$MONPID" 2>/dev/null || true
  PEAK=$(awk -F',' 'NF>=2{m=$2;gsub(/[" ]/,"",m);sub(/MiB.*/,"",m);if(m+0>mx)mx=m+0} END{print mx+0}' "$MON")
  JSON=$([ -f "$OUT/http-qwen3-c${C}.json" ] && echo yes || echo NO)
  echo "[c$C] rc=$RC wall_s=$((T1-T0)) peak_vram_mb=$PEAK json_written=$JSON baseline=$BASE" | tee -a "$PROG"
  for _ in $(seq 1 30); do NOW=$(baseline_vram); [ "$((NOW))" -le "$((BASE+300))" ] && break; sleep 5; done
  git add reports/2025-08-01 >/dev/null 2>&1
  git commit -m "bench(qwen3): http matrix c=$C rerun (empty-body fix, 50 req, bench-text-v1-en)" \
      -m "Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>" >/dev/null 2>&1 \
      && echo "[c$C] committed" | tee -a "$PROG" || echo "[c$C] nothing to commit" | tee -a "$PROG"
done

echo "rerun done $(date -u +%FT%TZ)" | tee -a "$PROG"
echo "RERUN_DONE $(date -u +%FT%TZ)" > "$OUT/RERUN_DONE"
