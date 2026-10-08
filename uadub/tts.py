"""Ukrainian TTS engines.

* UkrTTSEngine  — robinhad/ukrainian-tts (ESPnet VITS, MIT): 5 preset voices, fast on CPU.
* OmniEngine    — k2-fsa/OmniVoice: zero-shot cloning / voice design, exact `duration=` control.
                  Code Apache-2.0, weights non-commercial. Ignores stress marks.
* St2Engine     — patriotyk StyleTTS2-ukrainian (MIT): 31 native Ukrainian voices, every word
                  stressed from the dictionary (+ context, + our homograph/user marks) via IPA.
                  No cloning.
"""

from __future__ import annotations

import contextlib
import io
import os
from pathlib import Path

import numpy as np

from .config import CACHE_DIR, OMNI_MODEL


class UkrTTSEngine:
    controls_duration = False

    def __init__(self, voice: str, device: str = "cpu"):
        from ukrainian_tts.tts import TTS

        cache = CACHE_DIR / "ukrainian-tts"
        cache.mkdir(parents=True, exist_ok=True)
        cwd = os.getcwd()
        os.chdir(cache)  # the ESPnet config refers to feats_stats.npz by relative path
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                self.tts = TTS(cache_folder=".", device=device)
        finally:
            os.chdir(cwd)
        self.voice = voice
        self.sr: int | None = None

    def synth(self, text: str) -> np.ndarray:
        with contextlib.redirect_stdout(io.StringIO()):
            audio, sr, _ = self.tts.tts_to_array(text, self.voice, "dictionary")
        self.sr = sr
        return np.asarray(audio, dtype=np.float32)


class OmniEngine:
    """OmniVoice wrapper. `reference(...)` builds a reusable clone prompt."""

    controls_duration = True

    def __init__(self, steps: int = 24, device: str | None = None):
        import torch
        from omnivoice import OmniVoice

        device = device or ("mps" if torch.backends.mps.is_available() else "cpu")
        dtype = torch.float16 if device in ("mps", "cuda") else torch.float32
        self.torch = torch
        self.model = OmniVoice.from_pretrained(OMNI_MODEL, device_map=device, dtype=dtype)
        self.sr: int = self.model.sampling_rate
        self.steps = steps

    def reference(self, audio: np.ndarray, sr: int, text: str):
        wav = self.torch.from_numpy(np.ascontiguousarray(audio, dtype=np.float32))
        return self.model.create_voice_clone_prompt(ref_audio=(wav, sr), ref_text=text)

    def design_reference(self, instruct: str, seed: int = 1234):
        """Voice design gives a new voice on every call, so render one sample and clone it."""
        self.torch.manual_seed(seed)
        sample = ("Добрий день! Сьогодні я розповім вам дещо цікаве. "
                  "Сідайте зручніше, буде захопливо.")
        audio = self.model.generate(text=sample, language="uk", instruct=instruct, num_step=32)[0]
        return self.reference(audio, self.sr, sample), audio

    def natural_duration(self, text: str, prompt=None) -> float:
        """Model's own estimate of how long `text` takes to say (seconds)."""
        try:
            n_ref = prompt.ref_audio_tokens.size(-1) if prompt is not None else None
            ref_text = prompt.ref_text if prompt is not None else None
            tokens = self.model._estimate_target_tokens(text, ref_text, n_ref)
            return tokens / self.model.audio_tokenizer.config.frame_rate
        except Exception:
            from .textnorm import syllables

            return syllables(text) / 6.2

    def synth_batch(self, texts: list[str], durations: list[float | None], prompts: list) -> list[np.ndarray]:
        out = self.model.generate(
            text=texts,
            language="uk",
            voice_clone_prompt=prompts,
            duration=durations,
            num_step=self.steps,
            postprocess_output=False,
        )
        return [np.asarray(a, dtype=np.float32).reshape(-1) for a in out]


class St2Engine:
    """StyleTTS2-ukrainian. Text → stress marks (dictionary, Stanza context) → IPA → speech."""

    controls_duration = True  # via `speed`
    sr = 24000

    def __init__(self, device: str | None = None):
        import torch
        from styletts2_inference.models import StyleTTS2
        from ukrainian_word_stress import Stressifier, StressSymbol

        from .config import ST_MODEL

        device = device or ("mps" if torch.backends.mps.is_available() else "cpu")
        self.torch = torch
        self.model = StyleTTS2(hf_path=ST_MODEL, device=device)
        self.model.eval()
        self.stressify = Stressifier(stress_symbol=StressSymbol.CombiningAcuteAccent)
        self._styles: dict = {}

    def style(self, name: str):
        if name not in self._styles:
            from huggingface_hub import hf_hub_download

            from .config import ST_SPACE

            path = hf_hub_download(ST_SPACE, f"voices/{name}.pt", repo_type="space")
            self._styles[name] = self.torch.load(path, map_location="cpu")
        return self._styles[name]

    def phonemes(self, text: str) -> str:
        from ipa_uk import ipa

        return ipa(prepare_st_text(text, self.stressify))

    def prosody(self, y: np.ndarray, sr: int):
        """Style vector of an original voice clip: [timbre 128 | prosody 128].

        StyleTTS2 keeps *how* a sentence is said (pitch movement, energy, pace) separate from *who*
        says it, so the prosodic half of a foreign-language line can drive a Ukrainian voice.
        """
        import librosa

        y = np.asarray(y, dtype=np.float32)
        if sr != self.sr:
            y = librosa.resample(y, orig_sr=sr, target_sr=self.sr)
        if len(y) == 0:
            return None
        y, _ = librosa.effects.trim(y, top_db=30)
        if len(y) < int(0.5 * self.sr):
            return None  # too short for a stable estimate
        mel = self.model.preprocess(y)
        with self.torch.no_grad():
            ref_s = self.model.style_encoder(mel)
            ref_p = self.model.predictor_encoder(mel)
        return self.torch.cat([ref_s, ref_p], dim=1).cpu()

    def blend(self, voice: str, prosody, strength: float):
        """The chosen voice's timbre with (part of) the original line's intonation."""
        base = self.style(voice).detach().cpu().reshape(1, -1)
        s = base.clone()
        k = float(min(max(strength, 0.0), 1.0))
        s[:, 128:] = (1 - k) * base[:, 128:] + k * prosody[:, 128:]
        return s

    def synth(self, text: str, voice: str, speed: float = 1.0, style=None) -> np.ndarray:
        s_prev = style if style is not None else self.style(voice)
        wavs = []
        for part in split_sentences(text):
            tokens = self.model.tokenizer.encode(self.phonemes(part))
            if len(tokens) == 0:
                continue
            with self.torch.no_grad():
                wav = self.model(tokens, speed=float(speed), s_prev=s_prev)
            wavs.append(np.asarray(wav.cpu().numpy(), dtype=np.float32).reshape(-1))
        return np.concatenate(wavs) if wavs else np.zeros(0, dtype=np.float32)


_STRESS_VOWEL = "аеєиіїоуюяАЕЄИІЇОУЮЯ"


def split_sentences(text: str) -> list[str]:
    """StyleTTS2 is trained on sentences; long lines are voiced sentence by sentence."""
    import re

    parts = [p.strip() for p in re.split(r"(?<=[.?!:…])\s+", text) if p.strip()]
    merged: list[str] = []
    for p in parts:  # keep very short fragments with their neighbour
        if merged and len(merged[-1]) < 20:
            merged[-1] = f"{merged[-1]} {p}"
        else:
            merged.append(p)
    return merged


_UK_WORD = r"[А-Яа-яЄєІіЇїҐґ'ʼ’\u0301-]+"


def quiet_stress_logs() -> None:
    """ukrainian-word-stress warns about every dictionary word without accents — noise in the console."""
    import logging

    logging.getLogger("ukrainian_word_stress").setLevel(logging.ERROR)


def stressify_text(text: str, stressify) -> str:
    """Stressify, working around capitalised sentence starts.

    ukrainian-word-stress looks capitalised words up as proper nouns, so a sentence-initial
    «Коли» comes out as «Ко́ли» instead of «Коли́». We stressify a copy with sentence-initial
    words lower-cased too and, for those words only, prefer the common-word stress.
    """
    import re

    quiet_stress_logs()
    marked = stressify(text)
    starts = [m.start(1) for m in re.finditer(r"(?:^|[.!?…:]\s+)[\"«(]?([А-ЯЄІЇҐ])", text)]
    if not starts:
        return marked
    lowered = list(text)
    for i in starts:
        lowered[i] = lowered[i].lower()
    lower_marked = stressify("".join(lowered))

    def words(s):
        return [(m.start(), m.end(), m.group(0)) for m in re.finditer(_UK_WORD, s)]

    def base(w):
        return w.replace("\u0301", "").lower()

    a, b = words(marked), words(lower_marked)
    if len(a) != len(b) or any(base(x[2]) != base(y[2]) for x, y in zip(a, b)):
        return marked  # tokenisation drifted; keep the plain result
    src = words(text)
    initial = {k for k, (st, _, _) in enumerate(src) if st in starts} if len(src) == len(a) else set()
    out, pos = [], 0
    for k, ((st, en, w), (_, _, lw)) in enumerate(zip(a, b)):
        if k in initial and "\u0301" in lw and lw.lower() != w.lower():
            out.append(marked[pos:st])
            out.append(w[0] + lw[1:])  # keep the capital letter
            pos = en
    out.append(marked[pos:])
    return "".join(out)


def stress_letter_names(text: str) -> str:
    """«ел-ем», «ем-ел-екс», «пі-сі»: an acronym read letter by letter is stressed on its last letter."""
    import re

    from .textnorm import _LETTER_NAMES

    names = sorted(set(_LETTER_NAMES.values()), key=len, reverse=True)
    alt = "|".join(re.escape(n) for n in names)
    pat = re.compile(rf"(?<![А-Яа-яЄєІіЇїҐґ\u0301])((?:(?:{alt})-)+)({alt})(?![А-Яа-яЄєІіЇїҐґ\u0301])", re.I)

    def repl(m: re.Match) -> str:
        last = m.group(2)
        for k, ch in enumerate(last):
            if ch in _STRESS_VOWEL:
                return m.group(1) + last[: k + 1] + "\u0301" + last[k + 1 :]
        return m.group(0)

    return pat.sub(repl, text)


def _stress_unknown_words(text: str, stressify) -> str:
    """Words the dictionary does not know («двобітну», «відфільтруєте») get the stress of their stem."""
    import re

    from .stress import stress_by_prefix, with_acute

    def stem_stress(stem: str) -> int | None:
        marked = stressify(stem)
        if "\u0301" not in marked:
            return None
        return sum(ch in _STRESS_VOWEL for ch in marked[: marked.index("\u0301")]) - 1

    def repl(m: re.Match) -> str:
        w = m.group(0)
        if "\u0301" in w or "-" in w or sum(ch in _STRESS_VOWEL for ch in w) < 2:
            return w
        idx = stress_by_prefix(w, stem_stress)
        return with_acute(w, idx) if idx is not None else w

    return re.sub(_UK_WORD, repl, text)


def prepare_st_text(text: str, stressify=None) -> str:
    """Our stress conventions → one combining acute after the stressed vowel of every word."""
    import re
    import unicodedata

    t = re.sub(rf"\+([{_STRESS_VOWEL}])", "\\1\u0301", text)  # «зам+ок» → «замо́к»
    t = unicodedata.normalize("NFKC", t)
    t = re.sub(r"[᠆‐‑‒–—―⁻₋−⸺⸻]", "-", t)
    t = re.sub(r"[\"«»„“”]", "", t)
    t = re.sub(r" - ", ": ", t).strip()
    t = re.sub(r"[\s,;\-]+$", "", t)  # a line that continues into the next one still ends cleanly
    if t and t[-1] not in ".?!:…":
        t += "."
    t = stress_letter_names(t)
    if stressify is not None:
        t = stressify_text(t, stressify)  # dictionary + grammatical context; marked words are kept
        t = _stress_unknown_words(t, stressify)

    def one_stress(m):  # words with two allowed stresses (ви́си́ть) → keep the first
        w = m.group(0)
        first = w.find("\u0301")
        return w if first < 0 else w[: first + 1] + w[first + 1 :].replace("\u0301", "")

    return re.sub(_UK_WORD, one_stress, t)


def parse_voice(voice: str) -> tuple[str, str | None]:
    """'clone' → ('clone', None); 'clone:ref.wav' → ('clone_file', path); 'omni:male' → ('design', 'male')."""
    if voice == "clone":
        return "clone", None
    if voice.startswith("clone:"):
        return "clone_file", str(Path(voice.split(":", 1)[1]).expanduser())
    if voice.startswith("omni:"):
        return "design", voice.split(":", 1)[1].strip() or "male, middle-aged"
    if voice == "duo":  # two OmniVoice voices, chosen per line by the speaker's pitch
        return "duo", None
    if voice == "st" or voice.startswith("st:"):  # StyleTTS2: correct stress, no cloning
        from .config import ST_DEFAULT_MALE

        return "st", (voice[3:] if voice.startswith("st:") else ST_DEFAULT_MALE)
    if voice == "duo:st" or voice.startswith("duo:st:"):
        from .config import ST_DEFAULT_FEMALE, ST_DEFAULT_MALE

        pair = voice[7:].split(",") if voice.startswith("duo:st:") else [ST_DEFAULT_MALE, ST_DEFAULT_FEMALE]
        return "st_duo", ",".join(pair)
    if voice.startswith("duo:"):  # two preset voices: duo:<male>,<female>
        return "duo_preset", voice.split(":", 1)[1]
    return "preset", voice
