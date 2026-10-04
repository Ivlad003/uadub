# Long Videos in Parts Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** `uadub long.mp4` dubs a 4–7-hour video in parts cut at pauses, with peak RAM bounded by
the part length, one shared brief, previews of finished parts, and one in-sync `long.uk.mp4` +
`long.uk.srt` at the end.

**Architecture:** A new module `uadub/parts.py` plans cuts from a streamed loudness envelope,
runs the existing `run_pipeline` once per part (each part is an ordinary work folder with a time
range in the original file), builds one brief for the whole video between ASR and translation,
and finally concatenates the per-part dub audio with ffmpeg and muxes it over the untouched
original video. Stages learn three part-only behaviours: extract a time range, mix to a fixed
length and loudness, and write `dub.flac` + a preview instead of the final mux.

**Tech Stack:** Python 3.12, numpy, soundfile, ffmpeg/ffprobe (subprocess), multiprocessing
`spawn`, pytest. No new dependencies.

**Spec:** `docs/superpowers/specs/2026-10-04-long-video-parts-design.md`

## Global Constraints

- macOS on Apple Silicon only; tests must run without models: `.venv/bin/python -m pytest -q tests/test_logic.py` (ffmpeg is available and may be used in tests).
- All new tests go into `tests/test_logic.py` (the documented test command runs only that file).
- Console messages in Ukrainian; code, comments, prompts, `spec.md`, `README.md` in English; keep `README.uk.md` in sync.
- Only one heavy model resident at a time: any LLM use outside a stage runs in a `spawn` subprocess (ADR-002).
- New `Options` fields enter a stage fingerprint only when set (normal runs keep their caches).
- Do **not** commit: the project's CLAUDE.md forbids commits unless the user asks. Each task ends with a test checkpoint instead.
- Part mode: automatic above **45 minutes** with **15-minute** parts; `--part-minutes N` splits when the input is longer than **1.5 × N**; `--part-minutes 0` disables.
- Cut search window **±3 minutes** (180 s), minimum pause **0.5 s**, loudness frame **50 ms**, last part never shorter than **1/3** of the target.
- Silence threshold: `max(p10 + 12, p99 − 35)` dB over real frames.
- Loudness target clip: **−26…−12 LUFS**. Edge context: **2,500** characters each side.
- Previews: `<stem>.uk.parts/NN.uk<ext>`, deleted after assembly unless `--keep-parts`.

## Review Focus

1. **Damaged audio inside a long recording** (the user's real case): the loudness pipe may stop early; the plan must still cover the full duration and each part's extract must still be sample-exact. → Task 3 `test_frame_db_pads_unknown_tail_with_nan`, Task 1 `test_salvage_audio_clip_is_sample_exact`.
2. **No usable pauses** (music or constant noise): every cut falls back to the quietest window near its target, flagged in the plan table. → Task 3 `test_find_cuts_prefers_longest_pause_and_falls_back`.
3. **A part fails or the run is interrupted**: other parts continue, assembly is skipped with a list, and a rerun redoes only what is missing. → Task 6 `test_run_long_isolates_failures_and_resumes`.
4. **The user edits one part's `review.md`**: that part re-dubs and the final file is re-assembled even though previews were deleted. → Task 6 `test_run_long_reassembles_when_a_part_changes`.
5. **Audio-only input** (podcast `.m4a`): previews and the final file have no video stream and must not fail. → Task 2 `test_mux_clip_preview_and_audio_only`.

---

### Task 1: Audio helpers — clip extraction, exact length, FLAC stems

**Files:**
- Modify: `uadub/audio.py` (`extract_audio`, `_salvage_audio`; new `existing`, `fit_length`, `write_flac`, `to_flac`, `_fit_file`)
- Modify: `uadub/stages.py` (new `_voice_src`; replace the 5 `vocals.wav … else audio.wav` expressions; `stage_separate` start; `stage_mix` reads)
- Test: `tests/test_logic.py`

**Interfaces:**
- Produces:
  - `A.existing(path: Path) -> Path` — `path` if it exists, else `path.with_suffix(".flac")` if that exists, else `path`.
  - `A.fit_length(y: np.ndarray, n: int) -> np.ndarray` — trim or zero-pad axis 0 to `n`.
  - `A.write_flac(path: Path, y: np.ndarray, sr: int) -> None` — 24-bit FLAC, clipped to ±1.
  - `A.to_flac(wav: Path) -> Path` — converts, deletes the WAV, returns the FLAC path.
  - `A.extract_audio(video, out_wav, sr=44100, *, start: float = 0.0, length: float | None = None)` — with `length`, the WAV has exactly `round(length * sr)` frames.
  - `stages._voice_src(w: Path) -> Path`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_logic.py`:

```python
def test_existing_and_fit_length(tmp_path):
    import numpy as np

    from uadub import audio as A

    wav = tmp_path / "vocals.wav"
    assert A.existing(wav) == wav  # nothing there: the name itself
    A.write_flac(tmp_path / "vocals.flac", np.zeros((10, 2), np.float32), 44100)
    assert A.existing(wav) == tmp_path / "vocals.flac"
    A.write(wav, np.zeros((10, 2), np.float32), 44100)
    assert A.existing(wav) == wav  # WAV wins
    y = np.ones((5, 2), np.float32)
    assert A.fit_length(y, 3).shape == (3, 2)
    padded = A.fit_length(y, 8)
    assert padded.shape == (8, 2) and not padded[5:].any()
    assert A.fit_length(np.ones(4, np.float32), 6).shape == (6,)


def test_to_flac_replaces_wav(tmp_path):
    import numpy as np
    import soundfile as sf

    from uadub import audio as A

    wav = tmp_path / "background.wav"
    A.write(wav, np.full((441, 2), 0.25, np.float32), 44100)
    out = A.to_flac(wav)
    assert out == tmp_path / "background.flac" and not wav.exists()
    y, sr = sf.read(str(out), dtype="float32")
    assert sr == 44100 and y.shape == (441, 2) and abs(float(y[0, 0]) - 0.25) < 1e-4


def _sine_m4a(path, seconds=4, rate=48000):
    import subprocess

    subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-y", "-f", "lavfi", "-i",
                    f"sine=frequency=440:sample_rate={rate}:duration={seconds}", "-c:a", "aac", str(path)],
                   check=True)
    return path


def test_extract_audio_clip_is_sample_exact(tmp_path):
    import soundfile as sf

    from uadub import audio as A

    src = _sine_m4a(tmp_path / "src.m4a")
    A.extract_audio(src, tmp_path / "a.wav", 44100, start=1.0, length=1.5)
    assert sf.info(str(tmp_path / "a.wav")).frames == 66150


def test_salvage_audio_clip_is_sample_exact(tmp_path):
    from uadub import audio as A

    src = _sine_m4a(tmp_path / "src.m4a")
    y, runs = A._salvage_audio(src, 44100, start=1.0, length=1.5)
    assert y.shape == (66150, 2) and runs == 1 and abs(y).max() > 0.1
```

- [ ] **Step 2: Run them to see them fail**

Run: `.venv/bin/python -m pytest -q tests/test_logic.py -k "existing_and_fit or to_flac or sample_exact"`
Expected: FAIL — `AttributeError: module 'uadub.audio' has no attribute 'existing'` / unexpected keyword `start`.

- [ ] **Step 3: Implement the helpers in `uadub/audio.py`**

Add after `write()`:

```python
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
```

Replace `extract_audio` and `_salvage_audio` with:

```python
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
    if lost > MAX_LOST:
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
```

- [ ] **Step 4: Route stage reads through `A.existing` in `uadub/stages.py`**

Add after `load_json`:

```python
def _voice_src(w: Path) -> Path:
    """Separated speech if there is one (WAV, or FLAC after part cleanup), else the full audio."""
    v = A.existing(w / "vocals.wav")
    return v if v.exists() else A.existing(w / "audio.wav")
```

Replace each of the five occurrences of
`w / "vocals.wav" if (w / "vocals.wav").exists() else w / "audio.wav"` (lines ~132, 149, 161, 311, 412)
with `_voice_src(w)`, keeping the variable each one assigns (`voice = _voice_src(w)`, `src = _voice_src(w)`).

In `stage_separate`, replace the first loop and add a re-extract guard:

```python
    for name in ("vocals.wav", "background.wav", "vocals.flac", "background.flac"):
        (w / name).unlink(missing_ok=True)
    if not opt.separate:
        log("   • пропущено (режим закадрового перекладу поверх оригіналу)")
        return
    if not (w / "audio.wav").exists():  # removed by part cleanup: decode the part again
        stage_extract(opt)
```

In `stage_mix`, replace the two reads:

```python
    separated = A.existing(w / "background.wav").exists()
    bg, _ = A.read(A.existing(w / ("background.wav" if separated else "audio.wav")), sr=sr)
```
and
```python
    ref = A.read(A.existing(w / "vocals.wav"), sr=sr)[0] if separated else bg
```

- [ ] **Step 5: Run the tests**

Run: `.venv/bin/python -m pytest -q tests/test_logic.py`
Expected: all pass (34 existing + 5 new).

- [ ] **Step 6: Checkpoint** — full test run green; no commit (project rule).

---

### Task 2: Part-aware options, fingerprints and stages (extract, mix, part output, cleanup)

**Files:**
- Modify: `uadub/config.py` (`Options` fields, `fingerprint`, new `_file_hash`)
- Modify: `uadub/audio.py` (`mux` gains `clip`; new `_video_encoder`)
- Modify: `uadub/stages.py` (`stage_extract`, `stage_mix`, `stage_mux`, new `_part_output`, `_cleanup_part`)
- Test: `tests/test_logic.py`

**Interfaces:**
- Consumes: Task 1 helpers.
- Produces:
  - `Options.part_minutes: float | None = None`, `keep_parts: bool = False`, `clip_start: float | None = None`, `clip_end: float | None = None`, `shared_brief: str | None = None`, `edge_context: str | None = None`, `loudness_target: float | None = None`.
  - Fingerprint keys (only when set): extract/mix/mux `"clip": [start, end]`; translate `"shared_brief"`, `"edge"` (12-char sha1 of file content); mix `"loudness_target"`.
  - `A.mux(video, audio, out, *, srt, keep_original, clip: tuple[float, float] | None = None)` — `clip=(start, length)` cuts the input range and re-encodes video.
  - A part's `stage_mux` writes `work/dub.flac` (exactly `round(length*sr)` frames), the preview at `opt.output` (+ `.srt`), then compresses stems.
  - `stages._cleanup_part(w: Path) -> None`.

- [ ] **Step 1: Write the failing tests**

```python
def test_part_fingerprints_only_when_set(tmp_path):
    from uadub.config import Options

    base = dict(input="a.mp4", output="a.uk.mp4", workdir=str(tmp_path), voice="st")
    plain = Options(**base)
    for stage in ("extract", "translate", "mix", "mux"):
        assert not {"clip", "shared_brief", "edge", "loudness_target"} & set(plain.fingerprint(stage))
    (tmp_path / "b.json").write_text("{}")
    (tmp_path / "e.json").write_text("{}")
    part = Options(**base, clip_start=10.0, clip_end=20.0, shared_brief=str(tmp_path / "b.json"),
                   edge_context=str(tmp_path / "e.json"), loudness_target=-19.5)
    assert part.fingerprint("extract")["clip"] == [10.0, 20.0]
    before = part.fingerprint("translate")
    (tmp_path / "b.json").write_text('{"a": 1}')
    after = part.fingerprint("translate")
    assert after["shared_brief"] != before["shared_brief"] and after["edge"] == before["edge"]
    assert part.fingerprint("mix")["loudness_target"] == -19.5
    assert part.fingerprint("mix")["clip"] == [10.0, 20.0] and part.fingerprint("mux")["clip"] == [10.0, 20.0]


def _test_video(path, seconds=6):
    import subprocess

    subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-y",
                    "-f", "lavfi", "-i", f"testsrc=size=160x120:rate=25:duration={seconds}",
                    "-f", "lavfi", "-i", f"sine=frequency=300:sample_rate=44100:duration={seconds}",
                    "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", "-shortest", str(path)], check=True)
    return path


def test_mux_clip_preview_and_audio_only(tmp_path):
    import numpy as np

    from uadub import audio as A

    dub = tmp_path / "dub.flac"
    A.write_flac(dub, np.zeros((44100 * 2, 2), np.float32), 44100)
    video = _test_video(tmp_path / "v.mp4")
    out = tmp_path / "p.uk.mp4"
    A.mux(video, dub, out, srt=None, keep_original=True, clip=(2.0, 2.0))
    assert abs(A.duration(out) - 2.0) < 0.15 and A.has_stream(out, "video")
    podcast = _sine_m4a(tmp_path / "pod.m4a", seconds=6)
    out = tmp_path / "p.uk.m4a"
    A.mux(podcast, dub, out, srt=None, keep_original=True, clip=(2.0, 2.0))
    assert abs(A.duration(out) - 2.0) < 0.15 and not A.has_stream(out, "video")


def test_cleanup_part_compresses_stems(tmp_path):
    import numpy as np

    from uadub import audio as A
    from uadub.stages import _cleanup_part

    for name in ("audio.wav", "vocals.wav", "background.wav", "mix.wav"):
        A.write(tmp_path / name, np.zeros((100, 2), np.float32), 44100)
    _cleanup_part(tmp_path)
    assert sorted(p.name for p in tmp_path.iterdir()) == ["background.flac", "vocals.flac"]
    A.write(tmp_path / "audio.wav", np.zeros((100, 2), np.float32), 44100)  # --no-separate part
    for name in ("vocals.flac", "background.flac"):
        (tmp_path / name).unlink()
    _cleanup_part(tmp_path)
    assert sorted(p.name for p in tmp_path.iterdir()) == ["audio.flac"]
```

- [ ] **Step 2: Run them to see them fail**

Run: `.venv/bin/python -m pytest -q tests/test_logic.py -k "part_fingerprints or mux_clip or cleanup_part"`
Expected: FAIL — unexpected keyword `clip_start` / `clip` / missing `_cleanup_part`.

- [ ] **Step 3: Options and fingerprints (`uadub/config.py`)**

Add the fields after `domain`:

```python
    part_minutes: float | None = None  # long videos: None = automatic, 0 = never split, N = parts of ~N min
    keep_parts: bool = False  # keep the part previews after the final assembly
    clip_start: float | None = None  # this run is one part of a long video: its range in the input (s)
    clip_end: float | None = None
    shared_brief: str | None = None  # brief of the whole long video (replaces the per-part brief)
    edge_context: str | None = None  # neighbouring parts' text for translation context
    loudness_target: float | None = None  # LUFS shared by all parts of a long video
```

Add a module-level helper above `Options`:

```python
def _file_hash(path: str | None) -> str:
    import hashlib

    try:
        return hashlib.sha1(Path(path).read_bytes()).hexdigest()[:12]
    except (OSError, TypeError):
        return ""
```

At the end of `fingerprint`, before `return fp`:

```python
        if self.clip_start is not None and stage in ("extract", "mix", "mux"):
            fp["clip"] = [self.clip_start, self.clip_end]
        if stage == "translate" and self.shared_brief:
            fp["shared_brief"] = _file_hash(self.shared_brief)
            fp["edge"] = _file_hash(self.edge_context)
        if stage == "mix" and self.loudness_target is not None:
            fp["loudness_target"] = round(self.loudness_target, 2)
```

- [ ] **Step 4: `A.mux` with a clip (`uadub/audio.py`)**

```python
@functools.lru_cache(maxsize=1)
def _video_encoder() -> tuple[str, ...]:
    """Hardware H.264 on Apple Silicon, else a fast software encode (previews only)."""
    out = subprocess.run(["ffmpeg", "-hide_banner", "-encoders"], capture_output=True, text=True).stdout
    if "h264_videotoolbox" in out:
        return ("-c:v", "h264_videotoolbox", "-b:v", "6M")
    return ("-c:v", "libx264", "-preset", "veryfast", "-crf", "20")
```

(`import functools` at the top.) Change `mux`:

```python
def mux(video: Path, audio_wav: Path, out: Path, *, srt: Path | None, keep_original: bool,
        clip: tuple[float, float] | None = None) -> None:
    """clip=(start, length): only that range of `video` (a part preview); the video is re-encoded so it
    starts exactly at `start` rather than at the previous keyframe."""
    has_video = has_stream(video, "video")
    has_audio = has_stream(video, "audio")
    sub_codec = "mov_text" if out.suffix.lower() in {".mp4", ".m4v", ".mov", ".m4a"} else "srt"
    rng = ["-ss", f"{clip[0]:.6f}", "-t", f"{clip[1]:.6f}"] if clip else []
    cmd = ["ffmpeg", "-y", *rng, "-i", str(video), "-i", str(audio_wav)]
    if srt:
        cmd += ["-i", str(srt)]
    if has_video:
        cmd += ["-map", "0:v:0", *(_video_encoder() if clip else ("-c:v", "copy"))]
```
(the rest of `mux` stays unchanged).

- [ ] **Step 5: Stages (`uadub/stages.py`)**

`stage_extract`:

```python
def stage_extract(opt: Options) -> None:
    w = opt.work
    if not A.has_stream(opt.input, "audio"):
        raise SystemExit("У файлі немає аудіодоріжки — нічого перекладати.")
    if opt.clip_start is not None:
        A.extract_audio(opt.input, w / "audio.wav", opt.sample_rate, start=opt.clip_start,
                        length=opt.clip_end - opt.clip_start)
    else:
        A.extract_audio(opt.input, w / "audio.wav", opt.sample_rate)
    save_json(w, "meta.json", {"duration": A.duration(w / "audio.wav")})
```

`stage_mix`: right after reading `bg`, add

```python
    if opt.clip_end is not None:  # a part of a long video: exactly its length, so the parts add up
        bg = A.fit_length(bg, round((opt.clip_end - opt.clip_start) * sr))
```

and replace the loudness block with

```python
    # loudness: dub speaks as loud as the original speech did (one target for all parts of a long video)
    if opt.loudness_target is not None:
        target = opt.loudness_target
    else:
        ref = A.read(A.existing(w / "vocals.wav"), sr=sr)[0] if separated else bg
        target = A.loudness(ref, sr)
        target = float(np.clip(target if target is not None else -18.0, -26.0, -12.0))
```

`stage_mux` and helpers:

```python
def stage_mux(opt: Options) -> None:
    if opt.clip_start is not None:
        _part_output(opt)
        return
    w = opt.work
    out = Path(opt.output)
    srt = w / "uk.srt"
    A.mux(Path(opt.input), w / "mix.wav", out, srt=srt if srt.exists() else None, keep_original=opt.keep_original)
    if srt.exists():
        shutil.copyfile(srt, out.with_suffix(".srt"))


def _part_output(opt: Options) -> None:
    """One part of a long video: keep its dub as FLAC for the final assembly and write a preview."""
    w = opt.work
    out = Path(opt.output)
    srt = w / "uk.srt"
    length = opt.clip_end - opt.clip_start
    src = w / "mix.wav" if (w / "mix.wav").exists() else w / "dub.flac"
    y, sr = A.read(src)
    A.write_flac(w / "dub.flac", A.fit_length(y, round(length * sr)), sr)
    out.parent.mkdir(parents=True, exist_ok=True)
    A.mux(Path(opt.input), w / "dub.flac", out, srt=srt if srt.exists() else None,
          keep_original=opt.keep_original, clip=(opt.clip_start, length))
    if srt.exists():
        shutil.copyfile(srt, out.with_suffix(".srt"))
    log(f"   • превʼю: {out}")
    _cleanup_part(w)


def _cleanup_part(w: Path) -> None:
    """Free disk after a part is done: the stems a later re-dub needs stay as FLAC, the rest goes."""
    (w / "mix.wav").unlink(missing_ok=True)
    separated = A.existing(w / "vocals.wav").exists()
    for name in ("vocals.wav", "background.wav"):
        if (w / name).exists():
            A.to_flac(w / name)
    if (w / "audio.wav").exists():
        if separated:
            (w / "audio.wav").unlink()  # separate re-extracts it if ever needed
        else:
            A.to_flac(w / "audio.wav")  # --no-separate: mix and asr read it
```

- [ ] **Step 6: Run the tests**

Run: `.venv/bin/python -m pytest -q tests/test_logic.py`
Expected: all pass.

- [ ] **Step 7: Checkpoint** — green; no commit.

---

### Task 3: Cut planning (`uadub/parts.py`, part 1)

**Files:**
- Create: `uadub/parts.py`
- Test: `tests/test_logic.py`

**Interfaces:**
- Produces:
  - Constants `FRAME = 0.05`, `AUTO_MINUTES = 15.0`, `AUTO_ABOVE_MINUTES = 45.0`, `WINDOW = 180.0`, `MIN_PAUSE = 0.5`.
  - `part_minutes_for(duration_s: float, part_minutes: float | None) -> float` (0 = no parts).
  - `frame_db(video: Path, total: float, frame: float = FRAME) -> np.ndarray` — one dB value per frame, length `ceil(total/frame)`, NaN where the decoder gave nothing.
  - `silence_threshold(db: np.ndarray) -> float`.
  - `find_cuts(db, total, part_s, *, frame=FRAME, window=WINDOW, min_pause=MIN_PAUSE) -> list[dict]` — each `{"time": float, "pause": float, "fallback": bool}`.
  - `make_plan(db, total, part_minutes, sr, *, frame=FRAME) -> dict` and `load_or_make_plan(work: Path, video: Path, part_minutes: float, sr: int, *, log=print) -> dict` — plan `{"input", "duration", "sample_rate", "part_minutes", "parts": [{"n", "start", "end", "pause", "fallback"}]}`; `pause`/`fallback` describe the cut that ends the part (`None`/`False` for the last).
  - `format_plan(plan: dict) -> str`, `_hms(t: float) -> str`.

- [ ] **Step 1: Write the failing tests**

```python
def _speechy_db(total, frame=0.05):
    import numpy as np

    db = np.full(int(round(total / frame)), -20.0)
    db[::5] = -65.0  # short gaps between words: sets the silence floor, never a 0.5 s pause
    return db


def test_part_minutes_for():
    from uadub.parts import part_minutes_for

    assert part_minutes_for(44 * 60, None) == 0 and part_minutes_for(46 * 60, None) == 15.0
    assert part_minutes_for(20 * 60, 5) == 5 and part_minutes_for(20 * 60, 15) == 0
    assert part_minutes_for(5 * 3600, 0) == 0


def test_find_cuts_prefers_longest_pause_and_falls_back():
    from uadub.parts import FRAME, find_cuts

    total = 3600.0
    db = _speechy_db(total)

    def quiet(a, b):
        db[int(round(a / FRAME)):int(round(b / FRAME))] = -70.0

    quiet(800, 800.6)  # inside the first window but shorter
    quiet(950, 952)  # the longest pause near 900 s
    quiet(1700, 1701)  # inside the second window (951 + 900 ± 180)
    cuts = find_cuts(db, total, 900.0)
    # a word gap right after a pause may extend it by one 50 ms frame
    assert abs(cuts[0]["time"] - 951.0) < 0.06 and abs(cuts[0]["pause"] - 2.0) < 0.06 and not cuts[0]["fallback"]
    assert abs(cuts[1]["time"] - 1700.5) < 0.06 and not cuts[1]["fallback"]
    assert len(cuts) == 3 and cuts[2]["fallback"]  # no pause near 2600 s: quietest window, flagged
    assert abs(cuts[2]["time"] - 2600.5) < 1.0  # equal loudness everywhere → nearest the target


def test_find_cuts_merges_a_short_tail():
    from uadub.parts import find_cuts

    assert len(find_cuts(_speechy_db(2000.0), 2000.0, 900.0)) == 1  # 900 + 1100, not 900 + 900 + 200
    assert find_cuts(_speechy_db(1100.0), 1100.0, 900.0) == []


def test_frame_db_pads_unknown_tail_with_nan(tmp_path):
    import subprocess

    import numpy as np

    from uadub.parts import frame_db

    src = tmp_path / "a.m4a"
    subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-y", "-f", "lavfi", "-i",
                    "sine=frequency=440:sample_rate=48000:duration=3",
                    "-af", "volume=enable='between(t,1,2)':volume=0", "-c:a", "aac", str(src)], check=True)
    db = frame_db(src, 3.0)
    assert len(db) == 60 and np.nanmax(db[24:36]) < np.nanmin(db[2:16]) - 30
    longer = frame_db(src, 4.0)  # e.g. the decoder died early on a damaged track
    assert len(longer) == 80 and np.isnan(longer[-15:]).all()


def test_plan_is_sample_aligned_and_reused(tmp_path, monkeypatch):
    from uadub import parts

    calls = []

    def fake_db(video, total, frame=parts.FRAME):
        calls.append(1)
        return _speechy_db(total, frame)

    monkeypatch.setattr(parts, "frame_db", fake_db)
    monkeypatch.setattr(parts.A, "duration", lambda p: 2000.123)
    plan = parts.load_or_make_plan(tmp_path, tmp_path / "v.mp4", 15.0, 44100, log=lambda m: None)
    assert len(plan["parts"]) == 2 and plan["parts"][-1]["end"] == 2000.123
    assert plan["parts"][0]["end"] == plan["parts"][1]["start"]
    assert all(abs(p["start"] * 44100 - round(p["start"] * 44100)) < 1e-6 for p in plan["parts"])
    assert plan["parts"][-1]["pause"] is None and "частин" in parts.format_plan(plan)
    parts.load_or_make_plan(tmp_path, tmp_path / "v.mp4", 15.0, 44100, log=lambda m: None)
    assert len(calls) == 1  # same input and settings: plan reused, part caches stay valid
    parts.load_or_make_plan(tmp_path, tmp_path / "v.mp4", 10.0, 44100, log=lambda m: None)
    assert len(calls) == 2
```

- [ ] **Step 2: Run them to see them fail**

Run: `.venv/bin/python -m pytest -q tests/test_logic.py -k "part_minutes_for or find_cuts or frame_db or plan_is"`
Expected: FAIL — `ModuleNotFoundError: No module named 'uadub.parts'`.

- [ ] **Step 3: Create `uadub/parts.py` with planning**

```python
"""Long videos: cut the audio at pauses, dub the parts one by one, assemble one result (ADR-027)."""

from __future__ import annotations

import json
import math
import subprocess
from pathlib import Path

import numpy as np

from . import audio as A

FRAME = 0.05  # loudness frame, s
AUTO_MINUTES = 15.0  # part length when splitting automatically
AUTO_ABOVE_MINUTES = 45.0  # inputs longer than this are split automatically
WINDOW = 180.0  # a cut is searched within ±WINDOW s of its target
MIN_PAUSE = 0.5  # shorter quiet runs are not pauses


def part_minutes_for(duration_s: float, part_minutes: float | None) -> float:
    """Target part length in minutes, or 0 when the input is processed whole."""
    if part_minutes is None:
        return AUTO_MINUTES if duration_s > AUTO_ABOVE_MINUTES * 60 else 0.0
    if part_minutes <= 0:
        return 0.0
    return float(part_minutes) if duration_s > 1.5 * part_minutes * 60 else 0.0


def frame_db(video: Path, total: float, frame: float = FRAME) -> np.ndarray:
    """RMS level (dB) per frame of the whole input, streamed: under 1 MB for 7 hours.

    Frames the decoder never delivered (a damaged track that ends early) are NaN."""
    sr = 16000
    hop = int(sr * frame)
    proc = subprocess.Popen(
        ["ffmpeg", "-nostdin", "-v", "error", "-max_error_rate", "1.0", "-i", str(video), "-vn",
         "-af", "aresample=async=1:first_pts=0", "-ac", "1", "-ar", str(sr), "-f", "s16le", "pipe:1"],
        stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    vals: list[np.ndarray] = []
    carry = np.zeros(0, np.float32)
    odd = b""
    while True:
        buf = proc.stdout.read(1 << 20)
        if not buf:
            break
        buf = odd + buf
        odd, buf = (buf[-1:], buf[:-1]) if len(buf) % 2 else (b"", buf)
        y = np.concatenate([carry, np.frombuffer(buf, "<i2").astype(np.float32) / 32768.0])
        m = len(y) // hop
        if m:
            f = y[: m * hop].reshape(m, hop)
            vals.append(10 * np.log10(np.mean(f * f, axis=1) + 1e-10))
        carry = y[m * hop:]
    proc.wait()
    db = np.concatenate(vals) if vals else np.zeros(0)
    n = math.ceil(round(total / frame, 6))
    if len(db) < n:
        db = np.concatenate([db, np.full(n - len(db), np.nan)])
    return db[:n]


def silence_threshold(db: np.ndarray) -> float:
    """Adaptive, as in segments.snap_to_speech: relative to this recording's floor and peaks."""
    v = db[np.isfinite(db)]
    if not len(v):
        return -np.inf
    return float(max(np.percentile(v, 10) + 12, np.percentile(v, 99) - 35))


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    d = np.diff(np.concatenate([[0], mask.astype(np.int8), [0]]))
    return list(zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1)))


def find_cuts(db: np.ndarray, total: float, part_s: float, *, frame: float = FRAME,
              window: float = WINDOW, min_pause: float = MIN_PAUSE) -> list[dict]:
    """Cut times between parts: the middle of the longest pause near each target, else the quietest second."""
    with np.errstate(invalid="ignore"):
        silent = db < silence_threshold(db)  # NaN → not silent
    runs = _runs(silent)
    level = np.where(np.isfinite(db), db, np.inf)
    w1 = max(1, int(round(1.0 / frame)))
    cuts: list[dict] = []
    prev = 0.0
    while total - prev > part_s * 4 / 3:  # otherwise the rest is the last part
        target = prev + part_s
        lo = max(target - window, prev + part_s / 3)
        hi = min(target + window, total - part_s / 3)
        best = None
        for a, b in runs:
            s, e = max(a * frame, lo), min(b * frame, hi)
            if e - s >= min_pause - 1e-9:
                key = (round(e - s, 6), -abs((s + e) / 2 - target))
                if best is None or key > best[0]:
                    best = (key, (s + e) / 2, e - s)
        if best:
            cut, pause, fallback = best[1], best[2], False
        else:
            i0, i1 = int(lo / frame), max(int(lo / frame) + w1, int(hi / frame))
            seg = level[i0:i1]
            avg = np.convolve(seg, np.ones(w1) / w1, mode="valid") if len(seg) >= w1 else seg
            avg = np.round(avg, 2)  # float noise must not beat "nearest the target" among equal windows
            centres = (i0 + np.arange(len(avg)) + w1 / 2) * frame
            i = int(np.lexsort((np.abs(centres - target), avg))[0])
            cut, pause, fallback = float(centres[i]), 0.0, True
        cuts.append({"time": float(cut), "pause": round(float(pause), 2), "fallback": fallback})
        prev = cut
    return cuts


def make_plan(db: np.ndarray, total: float, part_minutes: float, sr: int, *, frame: float = FRAME) -> dict:
    cuts = find_cuts(db, total, part_minutes * 60, frame=frame)
    bounds = [0.0] + [round(c["time"] * sr) / sr for c in cuts] + [total]
    parts = []
    for i in range(len(bounds) - 1):
        end_cut = cuts[i] if i < len(cuts) else {"pause": None, "fallback": False}
        parts.append({"n": i + 1, "start": bounds[i], "end": bounds[i + 1],
                      "pause": end_cut["pause"], "fallback": end_cut["fallback"]})
    return {"duration": total, "sample_rate": sr, "part_minutes": part_minutes, "parts": parts}


def load_or_make_plan(work: Path, video: Path, part_minutes: float, sr: int, *, log=print) -> dict:
    """The plan is fixed per input and settings, so a rerun keeps every part's cache."""
    path = Path(work) / "plan.json"
    total = A.duration(video)
    if path.exists():
        plan = json.loads(path.read_text(encoding="utf-8"))
        if (plan.get("input") == str(video) and abs(plan.get("duration", -1) - total) < 1e-3
                and plan.get("sample_rate") == sr and plan.get("part_minutes") == part_minutes):
            return plan
    log("   • шукаю паузи для розрізів…")
    plan = make_plan(frame_db(video, total), total, part_minutes, sr)
    plan["input"] = str(video)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(plan, ensure_ascii=False, indent=2), encoding="utf-8")
    return plan


def _hms(t: float) -> str:
    t = int(round(t))
    return f"{t // 3600}:{t % 3600 // 60:02d}:{t % 60:02d}"


def format_plan(plan: dict) -> str:
    parts = plan["parts"]
    lines = [f"Довге відео ({_hms(plan['duration'])}) → {len(parts)} частин по ~{plan['part_minutes']:g} хв, "
             "розрізи в паузах:"]
    for p in parts:
        if p["pause"] is None:
            note = ""
        elif p["fallback"]:
            note = "   ! пауз немає — найтихіше місце"
        else:
            note = f"   пауза {p['pause']:.1f} с".replace(".", ",")
        lines.append(f"  {p['n']:02d}  {_hms(p['start'])}–{_hms(p['end'])}{note}")
    return "\n".join(lines)
```

- [ ] **Step 4: Run the tests**

Run: `.venv/bin/python -m pytest -q tests/test_logic.py`
Expected: all pass.

- [ ] **Step 5: Checkpoint** — green; no commit.

---

### Task 4: Shared brief of the whole video and edge context (`uadub/translate.py`, `uadub/stages.py`)

**Files:**
- Modify: `uadub/translate.py` (`chunk_units`, `MERGE_BRIEF_SYSTEM`, `merge_briefs_fallback`, `long_brief`; `translate_units` gains `shared_brief`, `edge`; `_context_window`, `_translate_chunk` gain `edge`)
- Modify: `uadub/stages.py` (`stage_translate` loads `opt.shared_brief`, `opt.edge_context`)
- Test: `tests/test_logic.py`

**Interfaces:**
- Consumes: `Options.shared_brief`, `Options.edge_context` (Task 2).
- Produces:
  - `chunk_units(units: list[dict], limit: int = BRIEF_CHAR_LIMIT) -> list[list[dict]]`.
  - `merge_briefs_fallback(briefs: list[dict], max_glossary: int = 25) -> dict`.
  - `long_brief(llm, units, src_name="English", lang_rules="", domain=None, *, limit=BRIEF_CHAR_LIMIT, log=print) -> dict` — same fields as `make_brief`.
  - `translate_units(..., shared_brief: dict | None = None, edge: dict | None = None, log=print)`.
  - Edge file format: `{"before": str, "after": str}`.

- [ ] **Step 1: Write the failing tests**

```python
def test_long_brief_map_reduce_and_fallback():
    import json as _json

    from uadub import translate as T

    units = [{"text": "a" * 100} for _ in range(30)]
    chunks = T.chunk_units(units, limit=1000)
    assert [len(c) for c in chunks] == [9, 9, 9, 3]

    class FakeLLM:
        def __init__(self, merge_ok):
            self.merge_ok, self.briefs = merge_ok, 0

        def chat(self, system, user, *, max_tokens=4096, temperature=0.3):
            if system.startswith("You merge partial"):
                return _json.dumps({"summary": "Усе відео.", "domain": "IT",
                                    "glossary": [{"src": "deploy", "uk": "деплой"}]}) if self.merge_ok else "oops"
            self.briefs += 1
            return _json.dumps({
                "summary": f"Частина {self.briefs}.", "domain": "IT", "address": "ви",
                "speaker_gender": "female" if self.briefs == 2 else "male",
                "glossary": [{"src": "deploy", "uk": "деплой" if self.briefs == 1 else "розгортання"},
                             {"src": f"t{self.briefs}", "uk": "x"}],
                "characters": [{"name": "Bob", "uk": "Боб", "gender": "male"}], "idioms": [], "asr_fixes": []})

    llm = FakeLLM(True)
    brief = T.long_brief(llm, units, limit=1000, log=lambda m: None)
    assert llm.briefs == 4 and brief["summary"] == "Усе відео."
    brief = T.long_brief(FakeLLM(False), units, limit=1000, log=lambda m: None)  # merge reply is broken
    assert brief["speaker_gender"] == "male" and brief["address"] == "ви" and brief["domain"] == "IT"
    assert [g["uk"] for g in brief["glossary"] if g["src"] == "deploy"] == ["деплой"]
    assert len(brief["characters"]) == 1 and brief["summary"].startswith("Частина 1.")
    single = T.long_brief(FakeLLM(True), units[:3], limit=1000, log=lambda m: None)
    assert single["summary"] == "Частина 1."  # one chunk: no merge call
    forced = T.long_brief(FakeLLM(False), units, domain="медицина", limit=1000, log=lambda m: None)
    assert forced["domain"] == "медицина"


def test_context_window_adds_edges():
    from uadub.translate import _context_window

    units = [{"text": f"line {i}"} for i in range(3)]
    assert _context_window(units, 0, 3) == "» line 0\n» line 1\n» line 2"
    text = _context_window(units, 0, 3, {"before": "PREV", "after": "NEXT"})
    assert text.startswith("PREV\n") and text.endswith("\nNEXT")


def test_translate_units_uses_shared_brief_and_edges():
    import json as _json

    from uadub.translate import translate_units

    seen = []

    class FakeLLM:
        def chat(self, system, user, *, max_tokens=4096, temperature=0.3):
            seen.append((system, user))
            if system.startswith("You are an expert audiovisual translator"):
                ids = [l["id"] for l in _json.loads(user)["lines"]]
                return _json.dumps({"lines": [{"id": i, "uk": "Робимо деплой."} for i in ids]})
            return _json.dumps({"lines": []})

    units = [{"id": 1, "text": "Let's deploy.", "start": 0.0, "end": 2.0, "slot_end": 2.0}]
    translate_units(FakeLLM(), units, rate=6.0, gender="male", glossary_path=None, stress="off",
                    shared_brief={"summary": "S", "glossary": [{"src": "deploy", "uk": "деплой"}]},
                    edge={"before": "PREVIOUS PART", "after": ""}, log=lambda m: None)
    assert not any(s.startswith("You prepare a translation brief") for s, _ in seen)
    system, user = next(x for x in seen if x[0].startswith("You are an expert audiovisual translator"))
    assert "deploy → деплой" in system and "PREVIOUS PART" in user
```

- [ ] **Step 2: Run them to see them fail**

Run: `.venv/bin/python -m pytest -q tests/test_logic.py -k "long_brief or context_window_adds or shared_brief_and"`
Expected: FAIL — no `chunk_units` / unexpected keyword `shared_brief`.

- [ ] **Step 3: Implement the brief merge in `uadub/translate.py`**

After `make_brief`:

```python
MERGE_BRIEF_SYSTEM = """You merge partial translation briefs of consecutive parts of one long {src_name} video into one brief for dubbing the whole video into Ukrainian.
Return ONLY JSON with the same fields as the parts: "summary" (3-6 sentences in Ukrainian about the whole video), "domain", "speaker_gender", "address", "characters", "glossary", "idioms", "asr_fixes".
Keep one Ukrainian rendering per glossary term (the most fitting one) and at most {max_glossary} glossary entries: the terms that matter most for consistency across the whole video. Merge characters by name. Keep every idiom and speech-recognition fix."""


def chunk_units(units: list[dict], limit: int = BRIEF_CHAR_LIMIT) -> list[list[dict]]:
    """Consecutive groups whose joined text fits one brief request."""
    chunks: list[list[dict]] = [[]]
    size = 0
    for u in units:
        n = len(u["text"]) + 1
        if chunks[-1] and size + n > limit:
            chunks.append([])
            size = 0
        chunks[-1].append(u)
        size += n
    return [c for c in chunks if c]


def merge_briefs_fallback(briefs: list[dict], max_glossary: int = 25) -> dict:
    """Deterministic merge when the LLM merge fails: first rendering wins, majority votes."""
    from collections import Counter

    out: dict = {"summary": " ".join(str(b.get("summary", "")).strip() for b in briefs if b.get("summary")).strip()}
    for key in ("speaker_gender", "address", "domain"):
        votes = [b.get(key) for b in briefs if b.get(key) and b.get(key) != "unknown"]
        if votes:
            out[key] = Counter(votes).most_common(1)[0][0]

    def unique(field: str, key: str, cap: int | None = None) -> list[dict]:
        seen: dict[str, dict] = {}
        for b in briefs:
            for item in b.get(field) or []:
                if isinstance(item, dict) and item.get(key):
                    seen.setdefault(str(item[key]).lower(), item)
        items = list(seen.values())
        return items[:cap] if cap else items

    out["glossary"] = unique("glossary", "src", max_glossary)
    out["characters"] = unique("characters", "name")
    out["idioms"] = unique("idioms", "src")
    out["asr_fixes"] = unique("asr_fixes", "heard")
    return out


def long_brief(llm: LLM, units: list[dict], src_name: str = "English", lang_rules: str = "",
               domain: str | None = None, *, limit: int = BRIEF_CHAR_LIMIT, log=print) -> dict:
    """Brief of a whole long video: one brief per chunk of text, then one merge (ADR-027)."""
    chunks = chunk_units(units, limit)
    if len(chunks) <= 1:
        return make_brief(llm, units, src_name, lang_rules, domain)
    briefs = []
    for i, chunk in enumerate(chunks, 1):
        log(f"   • бриф: шматок {i}/{len(chunks)}")
        b = make_brief(llm, chunk, src_name, lang_rules, domain)
        if b:
            briefs.append(b)
    max_glossary = 40 if domain else 25
    merged: dict = {}
    if briefs:
        log("   • зводжу бриф у один")
        system = MERGE_BRIEF_SYSTEM.replace("{src_name}", src_name).replace("{max_glossary}", str(max_glossary))
        try:
            merged = _ask(llm, system, json.dumps({"parts": briefs}, ensure_ascii=False), max_tokens=3000)
        except RuntimeError:
            merged = {}
        if not isinstance(merged, dict) or not isinstance(merged.get("glossary"), list):
            log("   (зведення через LLM не вдалося — зводжу за правилами)")
            merged = merge_briefs_fallback(briefs, max_glossary)
    if domain and domain != "auto":
        merged["domain"] = domain
    return merged
```

In `translate_units`, change the signature and the brief line:

```python
def translate_units(llm: LLM, units: list[dict], *, rate: float, gender: str | None,
                    glossary_path: str | None, max_speed: float = 1.25, stress: str = "auto",
                    source_lang: str = "en", domain: str | None = None, shared_brief: dict | None = None,
                    edge: dict | None = None, log=print) -> dict:
```
```python
    if shared_brief is not None:  # a part of a long video: one brief for the whole video
        log("   • спільний бриф довгого відео")
        brief = shared_brief
    else:
        log("   • аналіз тексту (тема, тон, глосарій)…")
        brief = make_brief(llm, units, src_name, lang_rules, domain)
```
(remove the old `log("   • аналіз тексту …")` + `brief = make_brief(...)` lines) and pass `edge` to the chunk loop:
`_translate_chunk(llm, system, context, units, chunk, by_id, fixed_gender=gender, edge=edge)`.

Replace `_context_window` and the head of `_translate_chunk`:

```python
def _context_window(units: list[dict], first: int, n: int, edge: dict | None = None) -> str:
    """Original text around the chunk (roughly half before, half after), for sense-for-sense translation.
    Lines of the chunk itself are marked with » so the model sees where they sit in the story.
    edge: text of the neighbouring parts of a long video, added at this part's ends."""
    lo, hi = first, first + n
    text = lambda a, b: " ".join(u["text"] for u in units[a:b])
    while (lo > 0 or hi < len(units)) and len(text(lo, hi)) < CONTEXT_CHARS:
        if lo > 0:
            lo -= 1
        if hi < len(units) and len(text(lo, hi)) < CONTEXT_CHARS:
            hi += 1
    parts = []
    for i in range(lo, hi):
        mark = "» " if first <= i < first + n else ""
        parts.append(mark + units[i]["text"])
    out = "\n".join(parts)
    if edge and lo == 0 and edge.get("before"):
        out = edge["before"] + "\n" + out
    if edge and hi == len(units) and edge.get("after"):
        out = out + "\n" + edge["after"]
    return out


def _translate_chunk(llm, system, context, units, chunk, by_id, fixed_gender=None, edge=None) -> None:
    first = units.index(chunk[0])
    prev = [{"src": u["text"], "uk": u.get("uk", "")} for u in units[max(0, first - 6) : first]]
    nxt = [u["text"] for u in units[first + len(chunk) : first + len(chunk) + 4]]
    payload = {**context, "transcript_context": _context_window(units, first, len(chunk), edge), "previous": prev,
```
and in the recursive retry inside `_translate_chunk`:
`_translate_chunk(llm, system, context, units, part, by_id, fixed_gender, edge)`.

- [ ] **Step 4: `stage_translate` loads the files (`uadub/stages.py`)**

In the `else:` branch of `stage_translate`:

```python
            shared = json.loads(Path(opt.shared_brief).read_text(encoding="utf-8")) if opt.shared_brief else None
            edge = json.loads(Path(opt.edge_context).read_text(encoding="utf-8")) if opt.edge_context else None
            info = translate_units(llm, units, rate=SYLLABLE_RATE[opt.engine], gender=opt.speaker_gender,
                                   glossary_path=opt.glossary, max_speed=opt.max_speed,
                                   stress=opt.stress, source_lang=opt.text_lang, domain=opt.domain,
                                   shared_brief=shared, edge=edge, log=log)
```

- [ ] **Step 5: Run the tests**

Run: `.venv/bin/python -m pytest -q tests/test_logic.py`
Expected: all pass (the plain-mode prompt tests from ADR-026 still pass).

- [ ] **Step 6: Checkpoint** — green; no commit.

---

### Task 5: Assembly (`uadub/parts.py`, part 2)

**Files:**
- Modify: `uadub/parts.py` (`merged_units`, `assemble`)
- Test: `tests/test_logic.py`

**Interfaces:**
- Consumes: part folders with `dub.flac` and `units.json` (Task 2).
- Produces:
  - `merged_units(items: list[tuple[Path, float]]) -> list[dict]` — units of all parts, times shifted by each part's start (`start`, `end`, `slot_end`, `dub_start`, `dub_end`), ids renumbered from 1.
  - `assemble(video: Path, items: list[tuple[Path, float]], out: Path, master: Path, *, src_lang: str, keep_original: bool) -> None` — writes `master/units.json`, `master/uk.srt`, `master/<src_lang>.srt`, `out` and `out.with_suffix(".srt")`.

- [ ] **Step 1: Write the failing test**

```python
def test_assemble_concatenates_parts_in_sync(tmp_path):
    import json as _json
    import subprocess

    import numpy as np

    from uadub import audio as A
    from uadub.parts import assemble

    video = _test_video(tmp_path / "v.mp4", seconds=6)
    items = []
    for n, (start, end) in enumerate([(0.0, 2.5), (2.5, 6.0)], 1):
        w = tmp_path / f"p{n}"
        w.mkdir()
        A.write_flac(w / "dub.flac", np.full((round((end - start) * 44100), 2), 0.1, np.float32), 44100)
        (w / "units.json").write_text(_json.dumps(
            [{"id": 1, "start": 0.5, "end": 1.5, "slot_end": 2.0, "text": f"Hi {n}", "uk": f"Привіт {n}"}]))
        items.append((w, start))
    out = tmp_path / "v.uk.mp4"
    assemble(video, items, out, tmp_path / "master", src_lang="en", keep_original=True)
    assert abs(A.duration(out) - 6.0) < 0.1 and A.has_stream(out, "video")
    a = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "a:0", "-show_entries", "stream=duration",
                        "-of", "csv=p=0", str(out)], capture_output=True, text=True).stdout
    assert abs(float(a) - 6.0) < 0.06
    srt = out.with_suffix(".srt").read_text(encoding="utf-8")
    assert "00:00:03,000" in srt and "Привіт 2" in srt  # second part's line shifted by 2.5 s
    units = _json.loads((tmp_path / "master" / "units.json").read_text())
    assert [u["id"] for u in units] == [1, 2] and units[1]["start"] == 3.0
```

- [ ] **Step 2: Run it to see it fail**

Run: `.venv/bin/python -m pytest -q tests/test_logic.py -k assemble_concatenates`
Expected: FAIL — `ImportError: cannot import name 'assemble'`.

- [ ] **Step 3: Implement in `uadub/parts.py`**

```python
_TIME_KEYS = ("start", "end", "slot_end", "dub_start", "dub_end")


def merged_units(items: list[tuple[Path, float]]) -> list[dict]:
    """Units of all parts on the timeline of the whole input."""
    out: list[dict] = []
    for work, offset in items:
        for u in json.loads((Path(work) / "units.json").read_text(encoding="utf-8")):
            v = dict(u)
            for k in _TIME_KEYS:
                if isinstance(v.get(k), (int, float)):
                    v[k] = round(v[k] + offset, 3)
            v["id"] = len(out) + 1
            out.append(v)
    return out


def assemble(video: Path, items: list[tuple[Path, float]], out: Path, master: Path, *,
             src_lang: str, keep_original: bool) -> None:
    """One result: the parts' dub audio concatenated by ffmpeg (streamed, encoded once) over the original video."""
    import shutil

    from .srt import make_cues, write_srt

    master = Path(master)
    master.mkdir(parents=True, exist_ok=True)
    units = merged_units(items)
    (master / "units.json").write_text(json.dumps(units, ensure_ascii=False), encoding="utf-8")
    write_srt(make_cues(units, "uk"), master / "uk.srt")
    write_srt(make_cues(units, "text"), master / f"{src_lang}.srt")
    lst = master / "concat.txt"
    esc = lambda p: str(Path(p).resolve()).replace("'", "'\\''")
    lst.write_text("".join(f"file '{esc(Path(w) / 'dub.flac')}'\n" for w, _ in items), encoding="utf-8")
    whole = master / "dub_all.flac"
    try:
        A._run(["ffmpeg", "-nostdin", "-y", "-f", "concat", "-safe", "0", "-i", str(lst), "-c:a", "flac", str(whole)])
        A.mux(Path(video), whole, Path(out), srt=master / "uk.srt", keep_original=keep_original)
    finally:
        whole.unlink(missing_ok=True)
        lst.unlink(missing_ok=True)
    shutil.copyfile(master / "uk.srt", Path(out).with_suffix(".srt"))
```

- [ ] **Step 4: Run the tests**

Run: `.venv/bin/python -m pytest -q tests/test_logic.py`
Expected: all pass.

- [ ] **Step 5: Checkpoint** — green; no commit.

---

### Task 6: Orchestration and CLI (`uadub/parts.py` part 3, `uadub/cli.py`)

**Files:**
- Modify: `uadub/parts.py` (`part_dir`, `previews_dir`, `part_options`, `speech_loudness`, `loudness_target`, `brief_key`, `write_edges`, `build_shared_brief`, `_brief_child`, `assembly_key`, `run_long`)
- Modify: `uadub/cli.py` (`run_pipeline` gains `prefix`, `summary`; `--part-minutes`, `--keep-parts`; dispatch to `run_long`)
- Test: `tests/test_logic.py`

**Interfaces:**
- Consumes: Tasks 2–5.
- Produces:
  - `run_pipeline(opt, *, redo=None, stop_after=None, review=False, review_with=None, text=False, prefix: str = "", summary: bool = True)`.
  - `run_long(opt: Options, plan: dict, *, redo=None, stop_after=None, review=False, review_with=None, text=False) -> None` — raises `SystemExit` listing failed parts.
  - `build_shared_brief(master: Path, works: list[Path]) -> None` — writes `master/long_brief.json` in a spawn subprocess.
  - `speech_loudness(work: Path, sr: int) -> float | None` (cached in the part's `meta.json` as `speech_lufs`).
  - Shared `state.json` keys: `"brief"` (key string), `"assembled"` (list).

- [ ] **Step 1: Write the failing tests**

```python
def _long_fixture(tmp_path, monkeypatch, fail):
    import json as _json

    from uadub import cli, parts
    from uadub.config import Options

    opt = Options(input=str(tmp_path / "long.mp4"), output=str(tmp_path / "long.uk.mp4"),
                  workdir=str(tmp_path / "long.uadub"), voice="st")
    plan = {"duration": 30.0, "sample_rate": 44100, "part_minutes": 0.2, "parts": [
        {"n": 1, "start": 0.0, "end": 10.0, "pause": 1.0, "fallback": False},
        {"n": 2, "start": 10.0, "end": 20.0, "pause": 0.8, "fallback": False},
        {"n": 3, "start": 20.0, "end": 30.0, "pause": None, "fallback": False}]}
    calls, assembled = [], []

    def fake_run(po, *, redo=None, stop_after=None, review=False, review_with=None, text=False,
                 prefix="", summary=True):
        n = int(po.work.name)
        calls.append((n, stop_after, po.shared_brief, po.loudness_target))
        po.work.mkdir(parents=True, exist_ok=True)
        (po.work / "transcript.json").write_text(_json.dumps([{"start": 0, "end": 1, "text": f"part {n}"}]))
        (po.work / "meta.json").write_text(_json.dumps({"duration": 10.0}))
        if stop_after == "asr":
            return
        if n in fail:
            raise SystemExit("\nЕтап «tts» завершився з помилкою (код 1).")
        if not (po.work / "dub.flac").exists():
            (po.work / "dub.flac").write_bytes(b"x")
            (po.work / "units.json").write_text("[]")

    def fake_assemble(video, items, out, master, **kw):
        assembled.append([n for n, _ in enumerate(items, 1)])
        Path(out).write_bytes(b"video")

    monkeypatch.setattr(cli, "run_pipeline", fake_run)
    monkeypatch.setattr(parts, "build_shared_brief",
                        lambda master, works: (master / "long_brief.json").write_text("{}"))
    monkeypatch.setattr(parts, "speech_loudness", lambda w, sr: -20.0)
    monkeypatch.setattr(parts, "assemble", fake_assemble)
    return opt, plan, calls, assembled


def test_run_long_isolates_failures_and_resumes(tmp_path, monkeypatch):
    import pytest

    from uadub import parts

    fail = {2}
    opt, plan, calls, assembled = _long_fixture(tmp_path, monkeypatch, fail)
    with pytest.raises(SystemExit) as e:
        parts.run_long(opt, plan)
    assert "02" in str(e.value) and not assembled
    assert [c[0] for c in calls if c[1] == "asr"] == [1, 2, 3]  # phase 1 for all parts first
    phase3 = [c for c in calls if c[1] is None]
    assert [c[0] for c in phase3] == [1, 2, 3]  # part 3 still made after part 2 failed
    assert all(c[2] and c[2].endswith("long_brief.json") and c[3] == -20.0 for c in phase3)
    assert (tmp_path / "long.uadub" / "edge" / "02.json").exists()
    fail.clear()
    calls.clear()
    parts.run_long(opt, plan)
    assert assembled == [[1, 2, 3]]


def test_run_long_reassembles_when_a_part_changes(tmp_path, monkeypatch):
    import os

    from uadub import parts

    opt, plan, calls, assembled = _long_fixture(tmp_path, monkeypatch, set())
    parts.run_long(opt, plan)
    parts.run_long(opt, plan)
    assert len(assembled) == 1  # nothing changed: no second assembly
    dub = tmp_path / "long.uadub" / "parts" / "02" / "dub.flac"
    dub.write_bytes(b"re-dubbed after a review.md edit")
    os.utime(dub, ns=(dub.stat().st_atime_ns, dub.stat().st_mtime_ns + 10**9))
    parts.run_long(opt, plan)
    assert len(assembled) == 2
    assert not (tmp_path / "long.uk.parts").exists()  # previews removed without --keep-parts
```

(`Path` is used in `fake_assemble`; add `from pathlib import Path` at the top of the test module if it is not imported yet.)

- [ ] **Step 2: Run them to see them fail**

Run: `.venv/bin/python -m pytest -q tests/test_logic.py -k run_long`
Expected: FAIL — `AttributeError: module 'uadub.parts' has no attribute 'run_long'`.

- [ ] **Step 3: `run_pipeline` gets `prefix` and `summary` (`uadub/cli.py`)**

Signature:

```python
def run_pipeline(opt: Options, *, redo: str | None = None, stop_after: str | None = None,
                 review: bool = False, review_with: str | None = None, text: bool = False,
                 prefix: str = "", summary: bool = True) -> None:
```

Prefix both stage prints: `print(f"{prefix}[{i}/{len(STAGES)}] {titles[stage]} — вже готово, пропускаю", flush=True)` and `print(f"{prefix}[{i}/{len(STAGES)}] {titles[stage]}…", flush=True)`. Wrap the `stop_after` message + `_print_texts` and the final "Готово за …" block + `_print_texts` in `if summary:` (the `return`/`break` behaviour stays the same).

- [ ] **Step 4: Orchestration in `uadub/parts.py`**

```python
EDGE_CHARS = 2500
BRIEF_VERSION = 1  # bump when the long-brief prompts change


def part_dir(master: Path, n: int) -> Path:
    return Path(master) / "parts" / f"{n:02d}"


def previews_dir(opt) -> Path:
    out = Path(opt.output)
    return out.with_name(f"{out.stem}.parts")


def part_options(opt, part: dict, master: Path):
    from dataclasses import replace

    out = Path(opt.output)
    return replace(opt, workdir=str(part_dir(master, part["n"])), clip_start=part["start"], clip_end=part["end"],
                   output=str(previews_dir(opt) / f"{part['n']:02d}.uk{out.suffix}"),
                   shared_brief=None, edge_context=None, loudness_target=None)


def _transcript(work: Path) -> list[str]:
    return [s["text"] for s in json.loads((Path(work) / "transcript.json").read_text(encoding="utf-8"))]


def speech_loudness(work: Path, sr: int) -> float | None:
    """Loudness of the part's original speech, measured once and kept in meta.json."""
    meta_path = Path(work) / "meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    if "speech_lufs" not in meta:
        src = A.existing(Path(work) / "vocals.wav")
        if not src.exists():
            src = A.existing(Path(work) / "audio.wav")
        meta["speech_lufs"] = A.loudness(A.read(src, sr=sr)[0], sr)
        meta_path.write_text(json.dumps(meta), encoding="utf-8")
    return meta["speech_lufs"]


def loudness_target(values: list[float | None]) -> float | None:
    v = [x for x in values if x is not None]
    return float(np.clip(np.median(v), -26.0, -12.0)) if v else None


def brief_key(opt, works: list[Path]) -> str:
    import hashlib

    h = hashlib.sha1()
    for w in works:
        h.update((Path(w) / "transcript.json").read_bytes())
    h.update(json.dumps([BRIEF_VERSION, opt.llm, opt.domain, opt.text_lang]).encode())
    return h.hexdigest()[:16]


def _tail(lines: list[str], n: int = EDGE_CHARS) -> str:
    out: list[str] = []
    for line in reversed(lines):
        if out and sum(map(len, out)) + len(line) > n:
            break
        out.insert(0, line)
    return "\n".join(out)


def _head(lines: list[str], n: int = EDGE_CHARS) -> str:
    out: list[str] = []
    for line in lines:
        if out and sum(map(len, out)) + len(line) > n:
            break
        out.append(line)
    return "\n".join(out)


def edge_path(master: Path, n: int) -> Path:
    return Path(master) / "edge" / f"{n:02d}.json"


def write_edges(master: Path, works: list[Path]) -> None:
    texts = [_transcript(w) for w in works]
    for i in range(len(works)):
        edge = {"before": _tail(texts[i - 1]) if i else "", "after": _head(texts[i + 1]) if i + 1 < len(texts) else ""}
        p = edge_path(master, i + 1)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(edge, ensure_ascii=False), encoding="utf-8")


def _brief_child(master: str, works: list[str]) -> None:
    from .cli import _env_defaults
    from .config import LANGS, Options
    from .llm import make_llm
    from .translate import LANG_RULES, long_brief

    _env_defaults()
    opt = Options.load(Path(master) / "options.json")
    units = [{"text": t} for w in works for t in _transcript(Path(w))]
    llm = make_llm(opt.llm)
    try:
        brief = long_brief(llm, units, LANGS.get(opt.text_lang, (None, opt.text_lang))[1],
                           LANG_RULES.get(opt.text_lang, ""), opt.domain)
    finally:
        llm.close()
    (Path(master) / "long_brief.json").write_text(json.dumps(brief, ensure_ascii=False, indent=2), encoding="utf-8")


def build_shared_brief(master: Path, works: list[Path]) -> None:
    """In a spawn subprocess, like the heavy stages: the LLM leaves memory when it is done (ADR-002)."""
    import multiprocessing as mp

    proc = mp.get_context("spawn").Process(target=_brief_child, args=(str(master), [str(w) for w in works]))
    proc.start()
    try:
        proc.join()
    except KeyboardInterrupt:
        proc.terminate()
        proc.join()
        raise
    if proc.exitcode != 0:
        raise SystemExit(f"\nСпільний бриф не вдався (код {proc.exitcode}). Подробиці вище; "
                         "запустіть ту саму команду ще раз.")


def assembly_key(works: list[Path]) -> list:
    key = []
    for w in works:
        for name in ("dub.flac", "units.json"):
            st = (Path(w) / name).stat()
            key.append([f"{Path(w).name}/{name}", st.st_size, st.st_mtime_ns])
    return key


def _each_part(opt, plan: dict, master: Path, fn) -> dict[int, str]:
    """Run fn(part_options, n, prefix) for every part; a failed part does not stop the others."""
    failed: dict[int, str] = {}
    total = len(plan["parts"])
    for p in plan["parts"]:
        tag = f"Частина {p['n']}/{total} · "
        print(f"\n{tag}{_hms(p['start'])}–{_hms(p['end'])}", flush=True)
        try:
            fn(part_options(opt, p, master), p["n"], tag)
        except (Exception, SystemExit) as e:  # KeyboardInterrupt still stops everything
            msg = (str(e).strip().splitlines() or [type(e).__name__])[-1]
            failed[p["n"]] = msg
            print(f"   ! частина {p['n']} не вдалася: {msg}", flush=True)
    return failed


def _stop_if_failed(failed: dict[int, str]) -> None:
    if failed:
        lines = "\n".join(f"  {n:02d}: {msg}" for n, msg in sorted(failed.items()))
        raise SystemExit(f"\nНе вдалися частини ({len(failed)}):\n{lines}\n"
                         "Запустіть ту саму команду ще раз — готові частини буде пропущено.")


def run_long(opt, plan: dict, *, redo: str | None = None, stop_after: str | None = None,
             review: bool = False, review_with: str | None = None, text: bool = False) -> None:
    import shutil
    import time

    from . import cli
    from .term import link

    master = opt.work
    master.mkdir(parents=True, exist_ok=True)
    opt.save(master / "options.json")
    state_path = master / "state.json"
    state = json.loads(state_path.read_text(encoding="utf-8")) if state_path.exists() else {}
    save = lambda: state_path.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
    works = [part_dir(master, p["n"]) for p in plan["parts"]]
    early = redo in ("extract", "separate", "asr")
    t_all = time.time()
    print(format_plan(plan), flush=True)

    print("\nФаза 1/3: слухаю частини", flush=True)
    _stop_if_failed(_each_part(opt, plan, master, lambda po, n, tag: cli.run_pipeline(
        po, redo=redo if early else None, stop_after="asr", prefix=tag, summary=False)))
    if stop_after in ("extract", "separate", "asr"):
        print(f"\nЗупинено після етапу «{stop_after}» для всіх частин.")
        return
    target = loudness_target([speech_loudness(w, opt.sample_rate) for w in works])

    print("\nФаза 2/3: спільний бриф", flush=True)
    brief = master / "long_brief.json"
    key = brief_key(opt, works)
    if early or redo == "translate" or state.get("brief") != key or not brief.exists():
        build_shared_brief(master, works)
        state["brief"] = key
        save()
    else:
        print("   • вже готово, пропускаю", flush=True)
    write_edges(master, works)

    print("\nФаза 3/3: готую частини", flush=True)

    def make(po, n, tag):
        po.shared_brief, po.edge_context, po.loudness_target = str(brief), str(edge_path(master, n)), target
        cli.run_pipeline(po, redo=None if early else redo, stop_after=stop_after, review=review,
                         review_with=review_with, prefix=tag, summary=False)

    _stop_if_failed(_each_part(opt, plan, master, make))
    if stop_after:
        print(f"\nЗупинено після етапу «{stop_after}» для всіх частин.")
        return

    key = assembly_key(works)
    out = Path(opt.output)
    if state.get("assembled") != key or not out.exists():
        print(f"\nСклеюю {len(works)} частин → {out.name}", flush=True)
        assemble(Path(opt.input), [(w, p["start"]) for w, p in zip(works, plan["parts"])], out, master,
                 src_lang=opt.text_lang, keep_original=opt.keep_original)
        state["assembled"] = key
        save()
    if not opt.keep_parts:
        shutil.rmtree(previews_dir(opt), ignore_errors=True)
    print(f"\nГотово за {(time.time() - t_all) / 60:.1f} хв:\n"
          f"   Відео:          {link(out)}\n"
          f"   Субтитри:       {link(out.with_suffix('.srt'))}\n"
          f"   Робоча папка:   {link(master)}")
    cli._print_texts(opt, text)
```

- [ ] **Step 5: CLI flags and dispatch (`uadub/cli.py`)**

In `build_parser`, after `--domain`:

```python
    p.add_argument("--part-minutes", type=float, metavar="ХВ",
                   help="довге відео: різати звук у паузах на частини ~ХВ хвилин, обробляти по черзі й склеїти "
                        "(за замовч. автоматично для відео довших за 45 хв, частини по 15 хв; 0 — не різати)")
    p.add_argument("--keep-parts", action="store_true",
                   help="довге відео: не видаляти превʼю частин (<назва>.uk.parts/) після склеювання")
```

In `main`, add to `Options(...)`: `part_minutes=args.part_minutes, keep_parts=args.keep_parts,`.
Replace the final `run_pipeline(...)` call with:

```python
    from .parts import load_or_make_plan, part_minutes_for, run_long

    minutes = part_minutes_for(A.duration(inp), opt.part_minutes)
    if minutes:
        if opt.subs:
            raise SystemExit("--subs поки не працює з нарізкою довгого відео на частини. Додайте --part-minutes 0.")
        if args.review:
            print("Порада: --review зупинятиметься перед озвученням кожної частини; "
                  "для довгого відео зручніше --review-with.", flush=True)
        plan = load_or_make_plan(opt.work, inp, minutes, opt.sample_rate)
        run_long(opt, plan, redo=args.redo, stop_after=args.stop_after, review=args.review,
                 review_with=args.review_with, text=args.text)
        return
    run_pipeline(opt, redo=args.redo, stop_after=args.stop_after, review=args.review, review_with=args.review_with,
                 text=args.text)
```

- [ ] **Step 6: Run the tests**

Run: `.venv/bin/python -m pytest -q tests/test_logic.py`
Expected: all pass. Also `.venv/bin/uadub --help | grep -A2 part-minutes` shows the flag.

- [ ] **Step 7: Checkpoint** — green; no commit.

---

### Task 7: Documentation and the manual end-to-end check

**Files:**
- Modify: `spec.md` (ADR-027 after ADR-026; FR row; NFR-2 note; options table row)
- Modify: `README.md`, `README.uk.md` ("Long videos" / «Довгі відео» section after the `--domain` section; options table rows `--part-minutes N`, `--keep-parts`)
- Modify: `AGENTS.md` code map (add `uadub/parts.py`)

- [ ] **Step 1: ADR-027 in `spec.md`**

```markdown
### ADR-027: Long videos in parts
- **Context:** A 4–7 h input kept whole needs ~1.3 GB per hour per audio copy in separation and
  mixing, its brief sees only the first and last 12,000 characters, and one failure loses a whole
  stage. The user cut recordings by hand and got 21 files with drifting terminology.
- **Decision:**
  - Inputs over 45 min (or `--part-minutes N`, when longer than 1.5 × N) are planned into parts cut
    at the middle of the longest pause within ±3 min of each target (`uadub/parts.py`). The plan is
    stored in `plan.json` and reused.
  - Only the audio is cut. Each part is an ordinary work folder (`parts/NN`) with `clip_start/end`;
    extract and mix are sample-exact, so the parts add up to the input.
  - Phase 1 runs extract → asr for all parts; phase 2 builds one brief by map-reduce over the whole
    transcript (in a spawn subprocess); phase 3 translates (with neighbouring text), voices and
    mixes each part with one loudness target, writes `dub.flac` and a preview, and compresses stems.
  - The parts' dubs are concatenated by ffmpeg and muxed once over the original video. A failed part
    does not stop the others; a rerun resumes.
- **Consequences:** Peak RAM depends on the part length, not the video length. Previews appear after
  phase 1. Disk: ~10 GB for 7 h in the work folder. `--subs` is not supported in part mode yet.
```

Add an FR row `| FR-xx | Long videos (4–7 h): automatic parts cut at pauses, one shared brief, previews, one result | ✅ |` (use the next free number), append to NFR-2: "; long videos are processed in parts, so RAM does not grow with the length (ADR-027)", and add `| --part-minutes N, --keep-parts | long videos in parts (ADR-027) |` to the options table.

- [ ] **Step 2: README sections**

`README.md`, after the `--domain` section:

```markdown
### Long videos (`--part-minutes`)

Videos longer than 45 minutes are dubbed in parts automatically:

- the audio is cut at pauses into parts of about 15 minutes (the plan is printed first);
- every part is recognised first, then one brief is built for the whole video, so terms, address
  and `--domain` stay the same everywhere;
- each finished part appears as a preview in `<name>.uk.parts/NN.uk.mp4`;
- at the end the parts are joined into one `<name>.uk.mp4` and `<name>.uk.srt` over the original
  video (no re-encoding of the video).

Memory does not grow with the video length. If a part fails, the others continue; run the same
command again to finish. To fix one part, edit `<name>.uadub/parts/NN/review.md` and rerun: only
that part is re-voiced and the result is re-joined.

```bash
uadub lecture.mp4                       # automatic for videos over 45 min
uadub lecture.mp4 --part-minutes 8      # shorter parts for a Mac with less memory
uadub lecture.mp4 --part-minutes 0      # never split
uadub lecture.mp4 --keep-parts          # keep the part previews
```

The work folder needs about 10 GB for 7 hours. `--subs` does not work with parts yet.
```

Mirror it in `README.uk.md` («### Довгі відео (`--part-minutes`)», same content in Ukrainian), and add the two options-table rows to both READMEs:
`| --part-minutes N | long videos: parts of ~N min cut at pauses (automatic above 45 min; 0 = never) |`,
`| --keep-parts | keep the part previews after joining |` (Ukrainian: «довге відео: частини ~N хв, розрізи в паузах (автоматично понад 45 хв; 0 — не різати)», «не видаляти превʼю частин після склеювання»).

- [ ] **Step 3: AGENTS.md code map row**

`| uadub/parts.py | long videos: cut plan at pauses, per-part runs, shared brief, assembly (ADR-027) |`

- [ ] **Step 4: Manual end-to-end check (~1 h, in the background)**

```bash
cd /Users/kosmodev/Documents/pet_project/uadub
for i in 1 2 3 4 5 6 7 8; do echo "file '$PWD/examples/ab/clip.mp4'"; done > /tmp/long_list.txt
ffmpeg -nostdin -v error -y -f concat -safe 0 -i /tmp/long_list.txt -c copy /tmp/long.mp4
ffprobe -v error -show_entries format=duration -of csv=p=0 /tmp/long.mp4
nohup .venv/bin/uadub /tmp/long.mp4 --voice st --part-minutes 5 --keep-parts > /tmp/long.log 2>&1 &
```

Check, and write down the numbers:
- the plan table appears and cuts land in pauses (listen to 2 s around each cut in `/tmp/long.mp4`);
- `ffprobe` duration of `/tmp/long.uk.mp4` equals `/tmp/long.mp4` within 0.05 s, and the Ukrainian audio stream too;
- the last subtitle cue in `/tmp/long.uk.srt` matches speech in the last part (play the final minute);
- interrupt the run during phase 3 (`kill` the process), start the same command: finished parts print «вже готово, пропускаю»;
- edit one line in `/tmp/long.uadub/parts/02/review.md`, rerun: only part 2 re-voices, «Склеюю» runs again.

- [ ] **Step 5: Final checkpoint**

Run: `.venv/bin/python -m pytest -q tests/test_logic.py` — all pass. Report the E2E numbers to the user; no commit unless asked.
