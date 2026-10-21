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


def transcribe(wav: Path, *, lang: str, model: str, backend: str, total: float,
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
    sentences = transcribe(Path(path), lang="en", model=default_asr_model(backend, "en"), backend=backend,
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


def _onnx_sentences(wav: Path, lang: str, model: str, device: str | None) -> list[dict]:
    raise NotImplementedError  # Task 3


def _ct2_sentences(wav: Path, lang: str, model: str, device: str | None) -> list[dict]:
    raise NotImplementedError  # Task 4
