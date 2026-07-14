#!/usr/bin/env bash
# One-shot setup: venv + deps + llama.cpp binary + Qwen2.5-VL-3B weights.
# Everything is open source; no API keys, no accounts, no GPU required.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

LLAMA_TAG="b9994"
HF_REPO="ggml-org/Qwen2.5-VL-3B-Instruct-GGUF"

echo "==> python venv + deps"
python3 -m venv .venv
.venv/bin/pip install -q --upgrade pip
.venv/bin/pip install -q -r requirements.txt

echo "==> llama.cpp ${LLAMA_TAG} (prebuilt, CPU)"
mkdir -p vendor/llama.cpp && cd vendor/llama.cpp
if [ ! -d "llama-${LLAMA_TAG}" ]; then
  curl -sL -o llama.tar.gz \
    "https://github.com/ggml-org/llama.cpp/releases/download/${LLAMA_TAG}/llama-${LLAMA_TAG}-bin-ubuntu-x64.tar.gz"
  tar xzf llama.tar.gz && rm llama.tar.gz
fi
cd "$ROOT"

echo "==> model weights from Hugging Face (~2.7 GB)"
mkdir -p models
for f in "Qwen2.5-VL-3B-Instruct-Q4_K_M.gguf" "mmproj-Qwen2.5-VL-3B-Instruct-Q8_0.gguf"; do
  [ -f "models/$f" ] || curl -L -o "models/$f" \
    "https://huggingface.co/${HF_REPO}/resolve/main/$f"
done

echo "==> done. start the model server with:  scripts/serve.sh"
echo "    then run an episode with:           .venv/bin/python -m lang2act.main"
