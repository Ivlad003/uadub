# uadub on Linux and Windows with NVIDIA CUDA — design

Status: draft for review · Date: 2026-10-21 · Sub-project B of the "features for friends" plan
(A: `--only` deliverables, done in ADR-037; C: local web interface, later).

## 1. Intent

uadub today runs only on Apple Silicon: ASR and the LLM are MLX, the TTS engines run PyTorch on
MPS. The user wants friends and the community to run it on their own machines, which are Linux and
Windows PCs with NVIDIA cards of 8–24 GB VRAM, installed with one command, offline after the model
download. A browser interface (sub-project C) only makes sense once those machines are supported.

Success looks like this: a friend with an RTX card on Windows runs one install script, then
`uadub talk.mp4 --voice st` produces the same dubbed video the Mac produces, and when something is
wrong `uadub --doctor` tells them (and us) what.

What the user said: platforms Linux and Windows with NVIDIA; cards vary from 8 to 24 GB; one
command through uv is the install story; no CUDA machine is available to us for testing.

Assumptions (agreed in the design conversation): quality within the Mac's on the test clip; the
pipeline, the prompts and the voices stay the same; Intel Macs, AMD ROCm, Docker images and a
one-click launcher are out of scope.

## 2. Constraints

- Offline by default (NFR-1): all backends run locally; the cloud LLM stays opt-in.
- One heavy model resident at a time (NFR-2): stages already run in `spawn` subprocesses, so VRAM
  is freed between stages on CUDA too.
- Existing Mac caches survive: new options enter a fingerprint only when they are not the Mac
  default (AGENTS.md rule 1).
- No CUDA hardware in the development loop: every new code path must be exercisable on the Mac in
  CPU mode, and the real CUDA run is a rented GPU hour.
- Windows is the hardest platform for the dependency stack and will be verified only through
  friends' `--doctor` reports before the first release.

## 3. Approach chosen

"Backends per stage, device-agnostic core." Each stage keeps its logic and gets a backend chosen by
platform:

| Stage | Mac (today, unchanged) | Linux / Windows NVIDIA |
|---|---|---|
| separate | `audio-separator`, BS-RoFormer, PyTorch MPS | same model, PyTorch CUDA (`audio-separator[gpu]`) |
| asr, English | `parakeet-mlx` | Parakeet TDT 0.6B v3 via `onnx-asr` (onnxruntime CUDA) |
| asr, other languages | `mlx-whisper` large-v3-turbo | `faster-whisper` large-v3-turbo (CTranslate2 CUDA) |
| translate | `mlx-lm` Gemma 4 26B-A4B 4-bit | Ollama, Gemma model picked by VRAM (`ollama:<model>`) |
| tts | StyleTTS2 / OmniVoice / ukrainian-tts on MPS | the same engines on CUDA |
| mix, mux | numpy, ffmpeg | unchanged |

Rejected: an NVIDIA-native stack (NeMo + vLLM) has no Windows support; "Whisper everywhere +
llama-cpp-python in-process" avoids installing Ollama but its CUDA wheels are fragile on Windows and
English timing is worse than Parakeet's. The second stays as a fallback idea if Ollama proves
inconvenient.

Facts the design relies on (checked 2026-10-21): `onnx-asr` lists `nemo-parakeet-tdt-0.6b-v3`,
returns token-level timestamps, and supports CUDA, CoreML and CPU on Windows, Linux and macOS
(`pip install onnx-asr[cpu,hub]`). `faster-whisper` needs CUDA 12 and cuDNN 9; on Linux they come
from `nvidia-cublas-cu12` / `nvidia-cudnn-cu12` wheels, on Windows from the DLLs the PyTorch CUDA
wheels ship, which CTranslate2 finds when `torch` is imported first. `audio-separator[gpu]` pulls
`onnxruntime-gpu` and a CUDA PyTorch; the two must come from matching CUDA versions. uv pins a
PyTorch CUDA index per platform with `[[tool.uv.index]] explicit = true` and `[tool.uv.sources]`
markers.

## 4. Seams and modules

### 4.1 ASR seam: `uadub/asr.py`

One interface, three adapters. A caller passes a WAV path, the language and the model id and
receives the sentence list the pipeline already uses:

```python
def transcribe(wav: Path, *, lang: str, model: str, backend: str) -> list[dict]
# → [{"start": float, "end": float, "text": str, "words": [{"w": str, "start": float, "end": float}]}]
```

Adapters:

- `mlx`: the code that lives in `stages.py` today (`_parakeet_sentences`, `_whisper_sentences`),
  moved without behaviour change.
- `onnx`: `onnx_asr.load_model("nemo-parakeet-tdt-0.6b-v3")` with timestamps; its tokens become
  words through the existing `segments.merge_tokens_to_words`, then sentences through the same
  sentence builder the MLX path uses. Provider list: CUDA first, CoreML on macOS, CPU last.
- `ctranslate`: `faster_whisper.WhisperModel("large-v3-turbo", device=…, compute_type="float16"
  on CUDA, "int8" on CPU)`, `word_timestamps=True`, `condition_on_previous_text=False`, the same
  hallucination filter and `snap_to_speech` as today.

Selection is a pure function, unit-tested:

```python
def choose_asr_backend(lang: str, *, platform: str, machine: str, has_mlx: bool, wanted: str = "auto") -> str
# auto → "mlx" on darwin/arm64 with MLX installed; else "onnx" for "en", "ctranslate" otherwise
```

`Options.asr_backend = "auto"` (`--asr-backend mlx|onnx|ctranslate` forces one). The resolved
backend enters the asr fingerprint only when it is not `mlx`, so Mac work folders keep their cache.
The clone-reference transcription in `stage_tts` (today `parakeet_mlx` at `stages.py:371`) goes
through the same seam with `lang="en"` or the reference's language.

`Options.asr_model` keeps its meaning per backend: the MLX repo ids today; for `onnx` the
`onnx-asr` model name; for `ctranslate` the faster-whisper model name. `--prefetch` downloads the
models of the chosen backend.

### 4.2 LLM seam (exists): default and model choice

`llm.LLM` with `MLXLLM`, `OllamaLLM`, `AgentCLILLM` stays. `DEFAULT_LLM` becomes
`default_llm(platform, machine, vram_gb)`: the MLX repo on darwin/arm64, otherwise
`ollama:<pick_ollama_model(vram_gb)>`:

| VRAM | Model (candidate names, see below) |
|---|---|
| unknown or < 18 GB | `gemma3:12b` (Q4, ~8 GB: fits a 12 GB card; spills a little on 8 GB) |
| ≥ 18 GB | `gemma3:27b` (Q4, ~17 GB) |

The policy is the decision; the exact tags are constants in `config.py`, confirmed against the
Ollama library in step 2 of the work order (if the Mac's Gemma 4 26B-A4B is published there, it
becomes the ≥ 18 GB default, since it is the model the prompts were tuned on). A friend can override
with `--llm ollama:<anything>` or `UADUB_LLM`. VRAM comes from `nvidia-smi
--query-gpu=memory.total --format=csv,noheader,nounits` (parsed by a pure function); without
`nvidia-smi` the smaller model is chosen. `OllamaLLM` gets clear messages for "server not running" and "model not pulled"
(today they surface as raw `URLError`/`KeyError`, now caught by `_ask`'s retry and reported as a
RuntimeError that ends the stage).

### 4.3 Device: `uadub/device.py`

```python
def pick(prefer: str | None = None) -> str  # "cuda" | "mps" | "cpu"
```

The three TTS engines (and through them `_emotion_styles`) call it instead of testing MPS
themselves; `audio-separator` detects CUDA on its own and is left alone. `--device` (hidden) forces
one, mostly for the CPU runs on the Mac. The ASR subprocess
imports `torch` before `ctranslate2` so the cuDNN/cuBLAS DLLs in the PyTorch wheels are loaded on
Windows. Half precision stays as today: float16 on `cuda` and `mps`, float32 on `cpu`.

### 4.4 Unchanged

`translate.py` (prompts, budgets, passes), `fit.py`, `stages.stage_mix`, `audio.mux`,
`parts.py`, `review.py`, `texts.py`. `stages.py` shrinks: the ASR adapters move out.

## 5. Packaging and installation

### 5.1 `pyproject.toml`

- MLX packages (`parakeet-mlx`, `mlx-whisper`, `mlx-lm`) stay in the main dependencies with the
  marker `sys_platform == 'darwin' and platform_machine == 'arm64'`.
- Extra `cuda`: `onnx-asr[hub]`, `onnxruntime-gpu`, `faster-whisper`, `audio-separator[gpu]`.
- Extra `cpu`: `onnx-asr[cpu,hub]`, `faster-whisper`, `audio-separator[cpu]` (for the Mac test
  runs and machines without NVIDIA; slow but complete).
- One named index `pytorch-cu128` (`explicit = true`) and `[tool.uv.sources]` entries for `torch`
  and `torchaudio` with the marker `sys_platform == 'linux' or sys_platform == 'win32'`; macOS keeps
  PyPI wheels. The CUDA index version is one documented constant and changes deliberately.
- `onnxruntime-gpu` is pinned to a version built for the same CUDA major as the index.

### 5.2 One command

`install.sh` (Linux) and `install.ps1` (Windows), both in the repository root and linked from the
README:

1. install uv if missing (official one-liners);
2. `uv tool install "uadub[cuda]"` (from PyPI once published; from the Git URL until then);
3. check `nvidia-smi` and the driver, print the VRAM;
4. install Ollama with its official installer if `ollama` is not on PATH, start it, `ollama pull`
   the model `pick_ollama_model` chose;
5. check `ffmpeg`, otherwise print the `winget install ffmpeg` / `apt install ffmpeg` line;
6. `uadub --prefetch` (ASR, separation, the `st` voice);
7. `uadub --doctor`.

### 5.3 Windows details

Paths only through `pathlib`; `UkrTTSEngine` stops using `chdir` (absolute cache paths);
`term.py`: clipboard through `clip` on Windows (`pbcopy` on macOS, `xclip`/`wl-copy` on Linux
when present), `$EDITOR` falls back to `notepad`, OSC 8 links stay (Windows Terminal supports them);
`multiprocessing` `spawn` is already the context used. Console output is UTF-8 (`PYTHONUTF8=1`
set by the launcher scripts and by `cli.main` through `reconfigure`).

## 6. Diagnostics: `uadub --doctor`

One block of plain text meant to be pasted into a chat message:

- OS, architecture, Python, uadub version;
- torch version and `torch.cuda.is_available()`, onnxruntime version and providers, ctranslate2
  version, `nvidia-smi` driver and VRAM;
- chosen ASR backend per language, chosen LLM, whether Ollama answers and has the model;
- ffmpeg found; model cache location and size;
- for each problem a one-line fix ("Ollama не запущений: `ollama serve`").

Exit code 1 when something required is missing. Implemented on top of probe functions that take
their inputs as parameters, so the report is unit-tested with fake probes.

## 7. Verification without a GPU

1. Unit tests on the Mac: `choose_asr_backend`, `pick_ollama_model`, `device.pick`, the
   `nvidia-smi` parser, token/word → sentence conversion for `onnx` and `ctranslate`, the asr
   fingerprint with and without `asr_backend`, `--doctor` with fake probes, the Windows clipboard
   and editor command choice.
2. CPU runs on the Mac with the `cpu` extra: `examples/ab/clip.mp4 --asr-backend onnx` and
   `examples/korean/drama.ko.mp4 --asr-backend ctranslate`; CER/WER (`tests/roundtrip.py`) and
   pace (`tests/pace.py`) compared with the MLX path; the smaller Ollama model as the translator
   for one run, since Ollama runs on the Mac.
3. Linux without a GPU: a `python:3.12` Docker container on the Mac installs `uadub[cpu]` from the
   repository, runs `--doctor` and `--only transcript` on the clip: proves the dependency markers
   and the Linux code paths.
4. Real CUDA: one rented GPU hour (RunPod or Vast, Linux), once on an RTX 4090 and once on an
   8 GB card: `install.sh` from scratch, `--doctor`, a full dub of the clip; the report records the
   time per stage, peak VRAM and the metrics.
5. Windows: friends run `install.ps1` and send the `--doctor` output and the log. Known risk until
   then; recorded in the README as "tested by users".

Definition of done for B: a full run of the clip on the rented Linux GPU with CER within ±0.02 of
the Mac result; a from-scratch install with one command; `--doctor` green; all unit tests passing
on the Mac; both READMEs, `spec.md` (hardware table, ADR-038) and `AGENTS.md` updated.

## 8. Documentation

- README and README.uk: a section "Linux and Windows with NVIDIA" (install command, VRAM → model
  table, `--doctor`, known limitations), the options table (`--asr-backend`, `--doctor`), the
  hardware table.
- `spec.md`: §2.3 hardware table with three columns (Mac, Linux NVIDIA, Windows NVIDIA); ADR-038
  summarising sections 4–7.
- `AGENTS.md`: the environment section becomes a platform matrix; a rule that no module outside
  `asr.py` / `llm.py` imports `mlx`; the Linux-container and GPU-hour checks join the "Definition
  of done" for backend changes.

## 9. Risks

| Risk | Mitigation |
|---|---|
| cuDNN not found by CTranslate2 on Windows | `import torch` first in the ASR subprocess; `--doctor` names the missing DLL; later fallback: Whisper through `onnx-asr` |
| torch / onnxruntime-gpu CUDA mismatch (`audio-separator[gpu]`) | one CUDA index, pinned `onnxruntime-gpu`, `--doctor` prints both versions |
| Ollama not running or model not pulled | the install script pulls it; `OllamaLLM` messages name the exact command |
| 8 GB cards: layers spill to RAM, translation slow | documented; `gemma3:12b` default; `--llm claude:…` as the fast alternative |
| Windows never run by us before release | friends' `--doctor` reports; the README says so |

## 10. Order of work

1. ASR seam and adapters, `choose_asr_backend`, fingerprints, unit tests; CPU runs on the Mac.
2. `device.py`, LLM default and Ollama messages, Windows details, `--doctor`.
3. Packaging (extras, index, markers), `install.sh` / `install.ps1`, the Linux container check.
4. Rented GPU run, metrics, READMEs, `spec.md`, `AGENTS.md`.

Each step is one commit; the implementation plan (next document) breaks them into tasks.
