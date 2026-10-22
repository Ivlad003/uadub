#!/usr/bin/env bash
# Proves the Linux dependency set and code paths without a GPU: run from the repo root on the Mac.
# Needs Docker. Installs uadub[cpu] from this checkout inside python:3.12, runs --doctor and an
# ASR-only pass on the sample clip with the onnx backend.
set -euo pipefail
cd "$(dirname "$0")/.."
docker run --rm -t -v "$PWD":/src:ro -v uadub-linux-cache:/root/.cache -w /tmp python:3.12-bookworm bash -euxc '
  apt-get update -qq && apt-get install -y -qq ffmpeg git >/dev/null
  curl -LsSf https://astral.sh/uv/install.sh | sh && export PATH=/root/.local/bin:$PATH
  # [tool.uv] of the installed package is ignored by `uv tool install` (also [tool.uv.sources]): pass the numpy override explicitly.
  printf "numpy>=2.0\n" > /tmp/overrides.txt
  uv tool install --python 3.12 --overrides /tmp/overrides.txt --torch-backend cpu "uadub[cpu] @ /src"
  uadub --doctor || true
  cp /src/examples/ab/clip.mp4 /tmp/clip.mp4
  uadub /tmp/clip.mp4 --only transcript --llm ollama:none --asr-backend onnx --stop-after asr --workdir /tmp/clip.uadub -o /tmp/clip.uk.mp4
  python3 -c "import json;t=json.load(open(\"/tmp/clip.uadub/transcript.json\"));print(\"sentences\", len(t)); assert len(t) > 20"
'
