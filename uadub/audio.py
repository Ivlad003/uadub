"""ffmpeg wrappers and numpy audio utilities (resample, trim, stretch, duck, mix)."""

from __future__ import annotations

import functools
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


_DECODE_ERROR = "Error submitting packet to decoder"
SALVAGE_SKIP = 0.5  # seconds jumped past the point where the decoder died
MAX_LOST = 0.9  # fraction of silence above which a damaged track is useless


def extract_audio(video: str | Path, out_wav: Path, sr: int = 44100, *, start: float = 0.0,
                  length: float | None = None) -> None:
    """Decode the default audio track to stereo float WAV.

    A clean track goes through one ffmpeg call. A damaged one (corrupt AAC packets) makes ffmpeg
    either drop packets, which shifts everything after them, or die when a garbage packet "changes"
    the sample rate. Such a track is decoded by `_salvage_audio`: a fresh decoder is restarted past
    each failure and every piece is placed at its own timestamp, so the timeline stays aligned with
    the video and the broken spots become silence.

    start/length: a part of a long video (seconds in the input); the WAV then has exactly
    round(length * sr) frames, so the parts add up to the whole.
    """
    seek = ["-ss", f"{start:.6f}"] if start else []
    cut = ["-t", f"{length:.6f}"] if length is not None else []
    cmd = ["ffmpeg", "-nostdin", "-y", "-max_error_rate", "1.0", *seek, "-i", str(video), *cut, "-vn",
           "-ac", "2", "-ar", str(sr), "-c:a", "pcm_f32le", str(out_wav)]
    proc = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
    bad = proc.stderr.count(_DECODE_ERROR)
    if proc.returncode == 0 and not bad:
        if length is not None:
            _fit_file(Path(out_wav), round(length * sr))
        return
    if not bad:  # not a damaged stream: report the real ffmpeg error
        tail = "\n".join(proc.stderr.strip().splitlines()[-15:])
        raise RuntimeError(f"Команда завершилась з помилкою: {' '.join(cmd[:3])} …\n{tail}")
    print(f"   ! аудіодоріжка пошкоджена (декодер відкинув {bad} пакетів) — відновлюю з вирівнюванням "
          "за часом, пошкоджені місця стануть тишею", flush=True)
    y, runs = _salvage_audio(video, sr, start=start, length=length)
    lost = silent_fraction(y, sr)
    print(f"   ! відновлено: {len(y) / sr:.0f} с, без звуку ≈{lost * len(y) / sr:.0f} с ({lost:.0%}), "
          f"перезапусків декодера: {runs}", flush=True)
    if lost > MAX_LOST and length is not None:  # one part of a long video: the others must still go on
        print("   ! ця частина майже вся пошкоджена — вона буде переважно тишею", flush=True)
    elif lost > MAX_LOST:
        raise SystemExit(f"Аудіо у файлі {Path(video).name} майже повністю пошкоджене ({lost:.0%} без звуку) — "
                         "перекладати нічого. Знайдіть цілу копію відео.")
    import soundfile as sf

    sf.write(str(out_wav), y, sr, subtype="FLOAT")


def _salvage_audio(video: str | Path, sr: int, *, start: float = 0.0,
                   length: float | None = None) -> tuple[np.ndarray, int]:
    import tempfile

    total = length if length is not None else duration(video) - start
    with tempfile.TemporaryDirectory() as td:
        raw = Path(td) / "a.raw"

        def decode(pos: float) -> tuple[np.ndarray, bool]:
            # -reinit_filter 0: a garbage packet that claims another sample rate or layout stops the
            # run instead of being resampled into noise; the caller restarts past it.
            cut = ["-t", f"{total - pos:.3f}"] if length is not None else []
            proc = subprocess.run(
                ["ffmpeg", "-nostdin", "-v", "error", "-y", "-max_error_rate", "1.0", "-reinit_filter", "0",
                 "-ss", f"{start + pos:.3f}", "-i", str(video), *cut, "-vn",
                 "-af", "aresample=async=1:first_pts=0", "-ac", "2", "-ar", str(sr), "-f", "f32le", str(raw)],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            y = np.fromfile(raw, np.float32).reshape(-1, 2) if raw.exists() else np.zeros((0, 2), np.float32)
            raw.unlink(missing_ok=True)
            return y, proc.returncode == 0

        return salvage_timeline(decode, total, sr)


def salvage_timeline(decode, total: float, sr: int, skip: float = SALVAGE_SKIP,
                     max_runs: int = 2000) -> tuple[np.ndarray, int]:
    """Assemble `total` seconds of audio from `decode(pos) -> (samples from pos, finished)`.

    Each failed run is restarted `skip` seconds past the last decoded sample; gaps stay silent.
    """
    n = round(total * sr)
    out = np.zeros((n, 2), np.float32)
    pos, runs = 0.0, 0
    while pos < total and runs < max_runs:
        runs += 1
        y, finished = decode(pos)
        i = round(pos * sr)
        k = max(0, min(len(y), n - i))
        out[i:i + k] = y[:k]
        if finished:
            break
        pos += k / sr + skip
    return out, runs


def silent_fraction(y: np.ndarray, sr: int, window: float = 0.1) -> float:
    """Share of `window`-long stretches that are exact digital silence (how salvage fills gaps)."""
    w = max(1, int(sr * window))
    m = len(y) // w
    if m == 0:
        return 0.0
    peak = np.abs(y[:m * w]).reshape(m, w, -1).max(axis=(1, 2))
    return float(np.mean(peak == 0.0))


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


def write_flac(path: Path, y: np.ndarray, sr: int) -> None:
    """Lossless 24-bit copy: part mode keeps finished stems this way (about half the size of float WAV)."""
    import soundfile as sf

    sf.write(str(path), np.clip(y, -1.0, 1.0), sr, subtype="PCM_24")


def to_flac(wav: Path) -> Path:
    y, sr = read(wav)
    out = Path(wav).with_suffix(".flac")
    write_flac(out, y, sr)
    Path(wav).unlink()
    return out


def existing(path: Path) -> Path:
    """`path`, or its .flac twin when part cleanup has compressed it."""
    path = Path(path)
    if path.exists():
        return path
    flac = path.with_suffix(".flac")
    return flac if flac.exists() else path


def fit_length(y: np.ndarray, n: int) -> np.ndarray:
    if len(y) >= n:
        return y[:n]
    return np.pad(y, ((0, n - len(y)),) + ((0, 0),) * (y.ndim - 1))


def _fit_file(path: Path, n: int) -> None:
    y, sr = read(path)
    if len(y) != n:
        write(path, fit_length(y, n), sr)


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


def _ltas(y: np.ndarray, sr: int, nfft: int) -> np.ndarray:
    """Long-term average power spectrum of the louder frames (speech, not pauses)."""
    y = y.mean(axis=1) if y.ndim == 2 else y
    m = len(y) // nfft
    if m == 0:
        return np.ones(nfft // 2 + 1)
    frames = y[: m * nfft].reshape(m, nfft)
    rms = np.sqrt((frames**2).mean(axis=1) + 1e-12)
    frames = frames[rms > rms.max() * 10 ** (-40 / 20)]
    spec = np.abs(np.fft.rfft(frames * np.hanning(nfft), axis=1)) ** 2
    return spec.mean(axis=0) + 1e-12


def match_spectrum(y: np.ndarray, ref: np.ndarray, sr: int, *, max_db: float = 6.0, fmin: float = 80.0,
                   fmax: float = 10000.0, taps: int = 1023) -> np.ndarray:
    """Tilt the dub's long-term spectrum towards the original speaker's (zero-phase FIR).

    TTS voices come out darker and boomier than a real microphone; a static, gently smoothed
    correction (1/3-octave bands, clamped to ±`max_db`) makes the dub clearer and sits it in the
    original's background. Nothing is done above `fmax` (24 kHz TTS has no content there).
    """
    from scipy.signal import firwin2

    nfft = 2048
    mono = y.mean(axis=1) if y.ndim == 2 else y
    if mono.size < nfft * 4 or ref.size < nfft * 4:
        return y
    f = np.fft.rfftfreq(nfft, 1 / sr)
    diff = 10 * np.log10(_ltas(ref, sr, nfft) / _ltas(mono, sr, nfft))  # dB the dub lacks per bin
    edges = fmin * 2 ** (np.arange(0, 32) / 3)
    edges = edges[edges < min(fmax, sr / 2)]
    freqs, gains = [0.0, fmin * 0.7], [0.0, 0.0]
    for lo, hi in zip(edges[:-1], edges[1:]):
        sel = (f >= lo) & (f < hi)
        if sel.any():
            freqs.append(float(np.sqrt(lo * hi)))
            gains.append(float(np.clip(diff[sel].mean(), -max_db, max_db)))
    freqs += [min(fmax, sr / 2 * 0.95), sr / 2]
    gains += [0.0, 0.0]
    gains_db = np.interp(freqs, freqs, gains)
    taps = taps | 1
    fir = firwin2(taps, np.array(freqs) / (sr / 2), 10 ** (np.array(gains_db) / 20))
    pad = taps // 2
    out = np.empty_like(y)
    cols = [y] if y.ndim == 1 else [y[:, c] for c in range(y.shape[1])]
    for i, col in enumerate(cols):
        full = np.convolve(col, fir.astype(np.float32), mode="full")[pad : pad + len(col)]
        if y.ndim == 1:
            out = full.astype(np.float32)
        else:
            out[:, i] = full
    return out


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


PREVIEW_BITRATE = 6_000_000  # hardware encoder's bitrate when the source's is unknown


@functools.lru_cache(maxsize=1)
def _has_videotoolbox() -> bool:
    out = subprocess.run(["ffmpeg", "-hide_banner", "-encoders"], capture_output=True, text=True).stdout
    return "h264_videotoolbox" in out


def _video_encoder(rate: int | None = None) -> tuple[str, ...]:
    """Hardware H.264 on Apple Silicon, else a fast software encode (previews only).

    rate: the source's video bitrate; a preview never gets more (hours of previews must fit the disk)."""
    cap = lambda r: ("-maxrate", str(r), "-bufsize", str(2 * r)) if r else ()
    if _has_videotoolbox():  # -b:v alone overshoots by ~1.5× at low rates; the cap holds it
        rate = min(rate, PREVIEW_BITRATE) if rate else None
        return ("-c:v", "h264_videotoolbox", "-b:v", str(rate or PREVIEW_BITRATE), *cap(rate))
    return ("-c:v", "libx264", "-preset", "veryfast", "-crf", "20", *cap(rate))


def video_bitrate(info: dict) -> int | None:
    """Bitrate of the first video stream from ffprobe output, else the whole file's; None when unknown."""
    stream = next((s for s in info.get("streams", []) if s.get("codec_type") == "video"), {})
    for value in (stream.get("bit_rate"), info.get("format", {}).get("bit_rate")):
        try:
            if int(value) > 0:
                return int(value)
        except (TypeError, ValueError):
            pass
    return None


def mux(video: Path, audio_wav: Path, out: Path, *, srt: Path | None, keep_original: bool,
        clip: tuple[float, float] | None = None) -> None:
    """clip=(start, length): only that range of `video` (a part preview); the video is re-encoded so it
    starts exactly at `start` rather than at the previous keyframe."""
    info = probe(video)
    kinds = {s.get("codec_type") for s in info.get("streams", [])}
    has_video, has_audio = "video" in kinds, "audio" in kinds
    sub_codec = "mov_text" if out.suffix.lower() in {".mp4", ".m4v", ".mov", ".m4a"} else "srt"
    rng = ["-ss", f"{clip[0]:.6f}", "-t", f"{clip[1]:.6f}"] if clip else []
    cmd = ["ffmpeg", "-y", *rng, "-i", str(video), "-i", str(audio_wav)]
    if srt:
        cmd += ["-i", str(srt)]
    if has_video:
        cmd += ["-map", "0:v:0", *(_video_encoder(video_bitrate(info)) if clip else ("-c:v", "copy"))]
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
