#!/usr/bin/env bash
# Start the local model server (llama.cpp) with Qwen2.5-VL-3B.
# The agent talks to it over the OpenAI-compatible API at :8080/v1.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BIN_DIR="$ROOT/vendor/llama.cpp/llama-b9994"

exec env LD_LIBRARY_PATH="$BIN_DIR" "$BIN_DIR/llama-server" \
  --model "$ROOT/models/Qwen2.5-VL-3B-Instruct-Q4_K_M.gguf" \
  --mmproj "$ROOT/models/mmproj-Qwen2.5-VL-3B-Instruct-Q8_0.gguf" \
  --alias qwen2.5-vl-3b \
  --host 127.0.0.1 --port 8080 \
  --ctx-size 8192 \
  --threads "$(nproc)" \
  --no-warmup \
  "$@"
