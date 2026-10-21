"""Which accelerator to use: one answer for the PyTorch engines and onnxruntime."""

from __future__ import annotations

import os

DEVICES = ("cuda", "mps", "cpu")


def choose(prefer: str | None, *, cuda: bool, mps: bool, env: str | None = None) -> str:
    """--device, then UADUB_DEVICE, then the best available accelerator."""
    wanted = prefer or env
    if wanted:
        if wanted not in DEVICES:
            raise ValueError(f"невідомий пристрій «{wanted}»: {', '.join(DEVICES)}")
        return wanted
    return "cuda" if cuda else "mps" if mps else "cpu"


def pick(prefer: str | None = None) -> str:
    import torch

    return choose(prefer, cuda=torch.cuda.is_available(), mps=torch.backends.mps.is_available(),
                  env=os.environ.get("UADUB_DEVICE"))


def ort_providers(device: str | None) -> list[str]:
    """onnxruntime execution providers for a device name (CPU is always the fallback)."""
    if device == "cuda":
        return ["CUDAExecutionProvider", "CPUExecutionProvider"]
    if device == "mps":
        return ["CPUExecutionProvider"]  # CoreML ran out of memory on Parakeet v3 (M1 Pro, exit -9)
    return ["CPUExecutionProvider"]
