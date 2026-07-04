#!/usr/bin/env bash
# OpenAI-compatible speech request; writes out.wav
set -euo pipefail
BASE_URL="${BASE_URL:-http://localhost:8000}"

curl -sS "$BASE_URL/v1/audio/speech" \
  -H "Content-Type: application/json" \
  -d '{
    "input": "Hello, welcome to our voice agent demo.",
    "voice": "default",
    "response_format": "wav",
    "speed": 1.0
  }' -o out.wav

echo "wrote out.wav ($(wc -c < out.wav) bytes)"
curl -sS "$BASE_URL/v1/models"
echo
curl -sS "$BASE_URL/api/v1/backends"
echo
