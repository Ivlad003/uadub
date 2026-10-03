"""ffmpeg wrappers and numpy audio utilities (resample, trim, stretch, duck, mix)."""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import numpy as np


def _run(cmd: list[str]) -> None:
    proc = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
    if proc.returncode != 0:
        tail = "\n".join(proc.stderr.strip().splitlines()[-15:])
        raise RuntimeError(f"Команда завершилась з помилкою: {' '.join(cmd[:3])} …\n{tail}")


def require_ffmpeg() -> None:
    for tool in ("ffmpeg", "ffprobe"):
        if shutil.which(tool) is None:
            raise SystemExit(f"Не знайдено {tool}. Встановіть: brew install ffmpeg")


def probe(path: str | Path) -> dict:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_format", "-show_streams", "-of", "json", str(path)],
        capture_output=True, text=True, check=True,
    ).stdout
    return json.loads(out)


def duration(path: str | Path) -> float:
    return float(probe(path)["format"]["duration"])


def has_stream(path: str | Path, kind: str) -> bool:
    return any(s.get("codec_type") == kind for s in probe(path).get("streams", []))


def extract_audio(video: str | Path, out_wav: Path, sr: int = 44100) -> None:
    _run(["ffmpeg", "-y", "-i", str(video), "-vn", "-ac", "2", "-ar", str(sr), "-c:a", "pcm_f32le", str(out_wav)])


def to_mono_16k(in_wav: Path, out_wav: Path) -> None:
    _run(["ffmpeg", "-y", "-i", str(in_wav), "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", str(out_wav)])


# ---------------------------------------------------------------------------
# numpy helpers
# ---------------------------------------------------------------------------
def read(path: Path, sr: int | None = None, mono: bool = False) -> tuple[np.ndarray, int]:
    import soundfile as sf

    y, file_sr = sf.read(str(path), dtype="float32", always_2d=True)  # (n, ch)
    if mono:
        y = y.mean(axis=1, keepdims=True)
    if sr is not None and sr != file_sr:
        y = resample(y, file_sr, sr)
        file_sr = sr
    return (y[:, 0] if mono else y), file_sr


def write(path: Path, y: np.ndarray, sr: int) -> None:
    import soundfile as sf

    sf.write(str(path), y, sr, subtype="FLOAT")


def resample(y: np.ndarray, sr_in: int, sr_out: int) -> np.ndarray:
    if sr_in == sr_out:
        return y
    import soxr

    return soxr.resample(y, sr_in, sr_out, quality="HQ").astype(np.float32)


def trim_silence(y: np.ndarray, sr: int, thresh_db: float = -42.0, pad: float = 0.03) -> np.ndarray:
    """Cut leading/trailing silence relative to the clip's peak level."""
    if y.size == 0:
        return y
    peak = float(np.max(np.abs(y))) or 1.0
    hop = max(1, int(sr * 0.01))
    n = len(y) // hop
    if n == 0:
        return y
    frames = np.abs(y[: n * hop]).reshape(n, hop).max(axis=1)
    loud = np.where(frames > peak * 10 ** (thresh_db / 20))[0]
    if loud.size == 0:
        return y[:0]
    a = max(0, loud[0] * hop - int(pad * sr))
    b = min(len(y), (loud[-1] + 1) * hop + int(pad * sr))
    return y[a:b]


def fade(y: np.ndarray, sr: int, ms: float = 12.0) -> np.ndarray:
    n = min(len(y) // 2, int(sr * ms / 1000))
    if n > 0:
        ramp = np.linspace(0.0, 1.0, n, dtype=np.float32)
        y = y.copy()
        y[:n] *= ramp
        y[-n:] *= ramp[::-1]
    return y


def time_stretch(y: np.ndarray, sr: int, speed: float) -> np.ndarray:
    """Speed speech up by `speed` (>1 = faster/shorter) without changing pitch."""
    if abs(speed - 1.0) < 1e-3 or y.size == 0:
        return y
    from pedalboard import time_stretch as _ts

    out = _ts(y[np.newaxis, :].astype(np.float32), float(sr), stretch_factor=float(speed),
              high_quality=True, transient_mode="mixed", preserve_formants=True)
    return out[0]


def loudness(y: np.ndarray, sr: int) -> float | None:
    import pyloudnorm as pyln

    if y.shape[0] < int(0.5 * sr):
        return None
    val = pyln.Meter(sr).integrated_loudness(y)
    return None if not np.isfinite(val) else float(val)


def duck_gain(voice: np.ndarray, sr: int, duck_db: float, *, hold: float = 0.25, ramp: float = 0.15) -> np.ndarray:
    """Per-sample gain for the background: `duck_db` while the dub speaks, 0 dB otherwise."""
    hop = max(1, int(sr * 0.01))
    n = len(voice) // hop + 1
    padded = np.zeros(n * hop, dtype=np.float32)
    padded[: len(voice)] = voice
    rms = np.sqrt((padded.reshape(n, hop) ** 2).mean(axis=1) + 1e-12)
    active = (rms > 10 ** (-50 / 20)).astype(np.float32)
    k = int(hold / 0.01)
    if k > 0:
        active = (np.convolve(active, np.ones(2 * k + 1), mode="same") > 0).astype(np.float32)
    g_db = active * duck_db
    r = max(1, int(ramp / 0.01))
    g_db = np.convolve(g_db, np.ones(r) / r, mode="same")
    gain = 10 ** (g_db / 20)
    return np.interp(np.arange(len(voice)), np.arange(n) * hop, gain).astype(np.float32)


def limit(y: np.ndarray, sr: int, threshold_db: float = -1.0) -> np.ndarray:
    from pedalboard import Limiter, Pedalboard

    board = Pedalboard([Limiter(threshold_db=threshold_db, release_ms=120)])
    return board(y.T.astype(np.float32), sr).T


def mux(video: Path, audio_wav: Path, out: Path, *, srt: Path | None, keep_original: bool) -> None:
    has_video = has_stream(video, "video")
    has_audio = has_stream(video, "audio")
    sub_codec = "mov_text" if out.suffix.lower() in {".mp4", ".m4v", ".mov", ".m4a"} else "srt"
    cmd = ["ffmpeg", "-y", "-i", str(video), "-i", str(audio_wav)]
    if srt:
        cmd += ["-i", str(srt)]
    if has_video:
        cmd += ["-map", "0:v:0", "-c:v", "copy"]
    cmd += ["-map", "1:a:0"]
    keep = keep_original and has_audio
    if keep:
        cmd += ["-map", "0:a:0"]
    if srt:
        cmd += ["-map", "2:s:0", "-c:s", sub_codec]
    cmd += ["-c:a", "aac", "-b:a", "192k",
            "-metadata:s:a:0", "language=ukr", "-metadata:s:a:0", "title=Українська (AI)",
            "-disposition:a:0", "default"]
    if keep:
        cmd += ["-metadata:s:a:1", "language=eng", "-metadata:s:a:1", "title=English (original)",
                "-disposition:a:1", "0"]
    if srt:
        cmd += ["-metadata:s:s:0", "language=ukr", "-metadata:s:s:0", "title=Українські субтитри",
                "-disposition:s:0", "0"]
    cmd += [str(out)]
    _run(cmd)
