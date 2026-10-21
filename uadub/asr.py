"""Speech recognition behind one seam: MLX on Apple Silicon, ONNX (Parakeet v3) and CTranslate2
(faster-whisper) everywhere else. Every adapter returns the same sentence dicts:
[{"start", "end", "text", "words": [{"w", "start", "end"}]}]."""

from __future__ import annotations

import importlib.util
from pathlib import Path

from .config import DEFAULT_ASR, WHISPER_ASR

BACKENDS = ("mlx", "onnx", "ctranslate")
ONNX_PARAKEET = "nemo-parakeet-tdt-0.6b-v3"  # onnx-asr model name (English and 24 more languages)
CT2_WHISPER = "large-v3-turbo"  # faster-whisper model name (other languages)


def has_mlx() -> bool:
    return all(importlib.util.find_spec(m) is not None for m in ("parakeet_mlx", "mlx_whisper"))


def choose_asr_backend(lang: str, *, platform: str, machine: str, has_mlx: bool, wanted: str = "auto") -> str:
    """auto → mlx on Apple Silicon with the MLX packages; else onnx (Parakeet) for English and
    ctranslate (Whisper) for every other language."""
    if wanted != "auto":
        if wanted not in BACKENDS:
            raise ValueError(f"невідомий ASR-бекенд «{wanted}»: {', '.join(BACKENDS)}")
        return wanted
    if platform == "darwin" and machine == "arm64" and has_mlx:
        return "mlx"
    return "onnx" if lang == "en" else "ctranslate"


def default_asr_model(backend: str, lang: str) -> str:
    if backend == "mlx":
        return DEFAULT_ASR if lang == "en" else WHISPER_ASR
    return ONNX_PARAKEET if backend == "onnx" else CT2_WHISPER
