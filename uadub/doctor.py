"""`uadub --doctor`: one pasteable report of the machine, the backends and what is missing."""

from __future__ import annotations

import importlib
import json
import shutil
import subprocess
import sys
import urllib.request
from pathlib import Path


def _version(module: str) -> str | None:
    try:
        return getattr(importlib.import_module(module), "__version__", "?")
    except Exception:
        return None


def collect(*, platform: str = sys.platform, machine: str | None = None, lang: str = "en") -> dict:
    import platform as _platform

    from . import __version__
    from .asr import choose_asr_backend, has_mlx
    from .config import CACHE_DIR, default_llm
    from .llm import detect_vram

    machine = machine or _platform.machine()
    info: dict = {"platform": platform, "machine": machine, "python": _platform.python_version(), "uadub": __version__,
                  "mlx": has_mlx(), "ffmpeg": shutil.which("ffmpeg"), "cache_dir": str(CACHE_DIR)}
    info["torch"] = _version("torch")
    try:
        import torch

        info["torch_cuda"] = bool(torch.cuda.is_available())
    except Exception:
        info["torch_cuda"] = False
    info["onnxruntime"] = _version("onnxruntime")
    try:
        import onnxruntime as ort

        info["ort_providers"] = ort.get_available_providers()
    except Exception:
        info["ort_providers"] = []
    info["ctranslate2"] = _version("ctranslate2")
    info["vram_gb"] = detect_vram()
    if shutil.which("nvidia-smi"):
        try:
            info["driver"] = subprocess.run(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
                                            capture_output=True, text=True, timeout=10).stdout.strip().splitlines()[0]
        except Exception:
            info["driver"] = None
    info["asr_backend"] = choose_asr_backend(lang, platform=platform, machine=machine, has_mlx=info["mlx"])
    info["llm"] = default_llm(platform=platform, machine=machine, vram_gb=info["vram_gb"])
    if info["llm"].startswith("ollama:"):
        try:
            with urllib.request.urlopen("http://127.0.0.1:11434/api/tags", timeout=3) as r:
                info["ollama_up"] = True
                info["ollama_models"] = [m["name"] for m in json.loads(r.read()).get("models", [])]
        except Exception:
            info["ollama_up"], info["ollama_models"] = False, []
    try:
        info["cache_gb"] = round(sum(p.stat().st_size for p in Path(CACHE_DIR).rglob("*") if p.is_file()) / 1e9, 1)
    except Exception:
        info["cache_gb"] = 0.0
    return info


def render(info: dict) -> tuple[str, bool]:
    """The report text and whether everything required is in place."""
    lines, problems = [], []
    yes = lambda v: "так" if v else "ні"  # noqa: E731
    lines.append(f"uadub {info.get('uadub', '?')} · Python {info.get('python', '?')} · "
                 f"{info.get('platform', '?')} {info.get('machine', '?')}")
    if info.get("mlx"):
        lines.append("Рушій: MLX (Apple Silicon)")
    else:
        lines.append(f"torch: {info.get('torch') or 'немає'} · CUDA: {yes(info.get('torch_cuda'))}"
                     + (f" · драйвер {info['driver']}" if info.get("driver") else "")
                     + (f" · VRAM {info['vram_gb']} ГБ" if info.get("vram_gb") else ""))
        lines.append(f"onnxruntime: {info.get('onnxruntime') or 'немає'} ({', '.join(info.get('ort_providers') or []) or '—'})"
                     f" · ctranslate2: {info.get('ctranslate2') or 'немає'}")
        if info.get("platform") == "darwin":
            problems.append("немає MLX: встановіть uadub з додатком [mac] і MLX-пакетами: uv pip install -e \".[mac,cpu,dev]\"")
        elif info.get("torch") and not info.get("torch_cuda") and info.get("driver"):
            problems.append("CUDA недоступна для torch: перевірте драйвер NVIDIA (nvidia-smi) і що uadub встановлено з [cuda]")
        if info.get("asr_backend") == "onnx" and not info.get("onnxruntime"):
            problems.append("немає onnxruntime: uv tool install \"uadub[cuda]\"")
        if info.get("asr_backend") == "ctranslate" and not info.get("ctranslate2"):
            problems.append("немає faster-whisper/ctranslate2: uv tool install \"uadub[cuda]\"")
    lines.append(f"Розпізнавання: {info.get('asr_backend', '?')} · Переклад: {info.get('llm', '?')}")
    llm = str(info.get("llm", ""))
    if llm.startswith("ollama:"):
        model = llm.partition(":")[2]
        if info.get("ollama_up") is False:
            problems.append("Ollama не запущений: ollama serve (або встановіть з https://ollama.com)")
        elif info.get("ollama_up") and model not in (info.get("ollama_models") or []):
            problems.append(f"у Ollama немає моделі {model}: ollama pull {model}")
    if info.get("ffmpeg"):
        lines.append(f"ffmpeg: {info['ffmpeg']}")
    else:
        problems.append("немає ffmpeg: winget install ffmpeg (Windows) · sudo apt install ffmpeg (Linux) · brew install ffmpeg (macOS)")
    if info.get("cache_dir"):
        lines.append(f"Кеш моделей: {info['cache_dir']} ({info.get('cache_gb', 0)} ГБ)")
    if problems:
        lines.append("\nЩо виправити:")
        lines += [f"  ! {p}" for p in problems]
    else:
        lines.append("\nУсе на місці.")
    return "\n".join(lines), not problems
