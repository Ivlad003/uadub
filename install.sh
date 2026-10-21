#!/usr/bin/env bash
# uadub on Linux with NVIDIA: uv → uadub[cuda] → Ollama + model → models → doctor.
set -euo pipefail

REPO="${UADUB_SOURCE:-git+https://github.com/Ivlad003/uadub.git}"

say() { printf '\n\033[1m%s\033[0m\n' "$*"; }

say "1/6 uv"
if ! command -v uv >/dev/null 2>&1; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi
uv --version

say "2/6 NVIDIA"
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
  EXTRA=cuda
else
  echo "nvidia-smi не знайдено: ставлю CPU-варіант (повільний). Для CUDA встановіть драйвер NVIDIA і запустіть скрипт знову."
  EXTRA=cpu
fi

say "3/6 uadub[$EXTRA]"
uv tool install --force --python 3.12 "uadub[$EXTRA] @ $REPO"
export PATH="$HOME/.local/bin:$PATH"

say "4/6 ffmpeg"
if ! command -v ffmpeg >/dev/null 2>&1; then
  echo "ffmpeg не знайдено. Встановіть: sudo apt install ffmpeg  (Debian/Ubuntu)  ·  sudo dnf install ffmpeg  (Fedora)"
fi

say "5/6 Ollama"
if ! command -v ollama >/dev/null 2>&1; then
  curl -fsSL https://ollama.com/install.sh | sh
fi
if ! curl -s http://127.0.0.1:11434/api/tags >/dev/null; then
  (ollama serve >/tmp/ollama.log 2>&1 &) ; sleep 3
fi

say "6/6 моделі"
uadub --prefetch
uadub --doctor || true
