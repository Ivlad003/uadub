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


# Typical Whisper hallucinations on silence/music (dropped when they come out of nowhere).
HALLUCINATIONS = ("시청해주셔서 감사합니다", "구독과 좋아요", "MBC 뉴스", "자막 제공", "다음 영상에서",
                  "thanks for watching", "subtitles by", "ご視聴ありがとうございました")


def whisper_words(segments: list[dict]) -> list[dict]:
    """Whisper segments (dicts with text, no_speech_prob, avg_logprob, words[word,start,end]) →
    timed words, without silent segments and the usual hallucinated sign-offs."""
    words: list[dict] = []
    for seg in segments:
        text = str(seg.get("text", "")).strip()
        if seg.get("no_speech_prob", 0) > 0.6 and seg.get("avg_logprob", 0) < -1.0:
            continue
        if any(h in text.lower() for h in HALLUCINATIONS) and len(text) < 40:
            continue
        for wd in seg.get("words", []):
            words.append({"w": str(wd["word"]).strip(), "start": float(wd["start"]), "end": float(wd["end"])})
    return words


def transcribe(wav: Path, *, lang: str | None, model: str, backend: str, total: float,
               device: str | None = None) -> list[dict]:
    """Timed sentences of `wav` (mono 16 kHz) with the chosen backend."""
    if backend == "mlx":
        return _mlx_sentences(wav, lang, model, total)
    if backend == "onnx":
        return _onnx_sentences(wav, lang, model, device)
    if backend == "ctranslate":
        return _ct2_sentences(wav, lang, model, device)
    raise ValueError(f"невідомий ASR-бекенд «{backend}»")


def transcribe_text(path: str | Path, *, backend: str, device: str | None = None) -> str:
    """Plain text of a short clip (the clone reference), in whatever language it is."""
    if backend == "mlx":
        from parakeet_mlx import from_pretrained

        return from_pretrained(DEFAULT_ASR).transcribe(str(path)).text.strip()
    sentences = transcribe(Path(path), lang=None, model=default_asr_model(backend, "en"), backend=backend,
                           total=0.0, device=device)
    return " ".join(s["text"] for s in sentences).strip()


# ---- mlx (Apple Silicon) --------------------------------------------------------------------

def _mlx_sentences(wav: Path, lang: str, model: str, total: float) -> list[dict]:
    from .segments import merge_tokens_to_words, words_to_sentences

    if "parakeet" in model:
        from parakeet_mlx import DecodingConfig, SentenceConfig, from_pretrained

        m = from_pretrained(model)
        cfg = DecodingConfig(sentence=SentenceConfig(max_words=30, silence_gap=0.7, max_duration=10.0))
        res = m.transcribe(str(wav), decoding_config=cfg, chunk_duration=120.0 if total > 150 else None,
                           overlap_duration=15.0)
        out = []
        for s in res.sentences:
            text = s.text.strip()
            if not text:
                continue
            toks = [{"text": t.text, "start": float(t.start), "end": float(t.end)} for t in s.tokens]
            out.append({"start": round(float(s.start), 3), "end": round(float(s.end), 3),
                        "text": text, "words": merge_tokens_to_words(toks)})
        return out
    import mlx_whisper

    r = mlx_whisper.transcribe(str(wav), path_or_hf_repo=model, language=lang, word_timestamps=True,
                               condition_on_previous_text=False, hallucination_silence_threshold=2.0,
                               no_speech_threshold=0.6, verbose=None)
    return words_to_sentences(whisper_words(r.get("segments", [])))


class MissingBackend(RuntimeError):
    """A backend's package is not installed: the message names the extra to install."""


def _need(module: str, extra: str) -> None:
    if importlib.util.find_spec(module) is None:
        raise MissingBackend(f"Не встановлено {module}. Поставте uadub з додатком: "
                             f"uv tool install \"uadub[{extra}]\" (на машині з NVIDIA — uadub[cuda])")


def tokens_to_sentences(tokens: list[str], timestamps: list[float], *, tail: float = 0.08) -> list[dict]:
    """onnx-asr gives SentencePiece tokens (a leading space starts a word) and one start time per
    token. A token ends where the next one starts; the last one gets `tail` seconds."""
    from .segments import merge_tokens_to_words, words_to_sentences

    toks = []
    for i, (text, start) in enumerate(zip(tokens, timestamps)):
        if not text:
            continue
        nxt = next((timestamps[j] for j in range(i + 1, len(timestamps)) if tokens[j]), None)
        end = float(nxt) if nxt is not None else float(start) + tail
        toks.append({"text": text, "start": round(float(start), 3), "end": round(end, 3)})
    if toks and not toks[0]["text"].startswith(" "):
        toks[0]["text"] = " " + toks[0]["text"]  # the first token starts the first word
    return words_to_sentences(merge_tokens_to_words(toks))


ONNX_FILES = ("config.json", "vocab.txt", "encoder-model.onnx", "encoder-model.onnx.data", "decoder_joint-model.onnx")


def _onnx_model_dir(model: str, cache: Path | None = None) -> str | None:
    """A real (symlink-free) folder with the model, or None to let onnx-asr resolve it itself.
    onnxruntime 1.30 refuses symlinked external data, which is what the Hugging Face cache consists
    of. A folder that already has all the files is used without touching the Hub (offline runs)."""
    try:
        from onnx_asr.resolver import model_repos
    except ImportError:  # private module moved: fall back to onnx-asr's own resolving
        return None
    repo = model_repos.get(model)
    if repo is None:
        return None
    target = (cache or Path.home() / ".cache" / "uadub" / "onnx") / model
    if all((target / f).is_file() for f in ONNX_FILES):
        return str(target)
    try:
        from huggingface_hub import snapshot_download

        return snapshot_download(repo, local_dir=str(target), allow_patterns=list(ONNX_FILES))
    except Exception as e:
        raise RuntimeError(f"Не вдалося завантажити модель розпізнавання «{model}» ({type(e).__name__}). "
                           f"Завантажте її наперед (потрібен інтернет): uadub --prefetch") from e


def _onnx_sentences(wav: Path, lang: str | None, model: str, device: str | None) -> list[dict]:
    _need("onnx_asr", "cpu")
    import onnx_asr
    import soundfile as sf

    from .device import ort_providers, pick

    waveform, sr = sf.read(str(wav), dtype="float32")
    if waveform.ndim > 1:
        waveform = waveform.mean(axis=1)
    m = onnx_asr.load_model(model, _onnx_model_dir(model), providers=ort_providers(pick(device))).with_timestamps()
    # Parakeet v3 detects the language itself; only a non-English hint is passed on.
    kwargs = {"language": lang} if lang and lang != "en" else {}
    try:
        r = m.recognize(waveform, sample_rate=sr, **kwargs)
    except TypeError:
        if not kwargs:
            raise
        r = m.recognize(waveform, sample_rate=sr)  # this onnx-asr build takes no language hint
    return tokens_to_sentences(list(r.tokens or []), list(r.timestamps or []))


def ct2_segments_to_dicts(segments) -> list[dict]:
    """faster-whisper Segment objects → the plain dicts `whisper_words` understands."""
    out = []
    for s in segments:
        out.append({"text": s.text, "no_speech_prob": s.no_speech_prob, "avg_logprob": s.avg_logprob,
                    "words": [{"word": w.word, "start": float(w.start), "end": float(w.end)} for w in (s.words or [])]})
    return out


def _ct2_sentences(wav: Path, lang: str | None, model: str, device: str | None) -> list[dict]:
    _need("faster_whisper", "cpu")
    import torch  # noqa: F401  (loads the cuDNN/cuBLAS libraries the PyTorch wheels ship; CTranslate2 reuses them)
    from faster_whisper import WhisperModel

    from .device import pick
    from .segments import words_to_sentences

    dev = "cuda" if pick(device) == "cuda" else "cpu"
    m = WhisperModel(model, device=dev, compute_type="float16" if dev == "cuda" else "int8")
    segments, _info = m.transcribe(str(wav), language=lang, word_timestamps=True, condition_on_previous_text=False,
                                   no_speech_threshold=0.6, vad_filter=False)
    return words_to_sentences(whisper_words(ct2_segments_to_dicts(segments)))
