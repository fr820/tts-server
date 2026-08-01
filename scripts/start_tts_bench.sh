#!/usr/bin/env bash
# Launcher for the qwen3 benchmark server (run inside a tmux session).
# Model weights download via the HF mirror on first start.
set -euo pipefail
export HF_ENDPOINT=https://hf-mirror.com
export TTS_BACKEND=qwen3
cd /home/ecs-user/tts-server
exec uv run uvicorn tts_server.main:app --host 0.0.0.0 --port 8000
