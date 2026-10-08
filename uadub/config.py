"""Run options shared between the CLI and the per-stage worker processes."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, fields
from pathlib import Path

CACHE_DIR = Path.home() / ".cache" / "uadub"

DEFAULT_LLM = "mlx-community/gemma-4-26b-a4b-it-4bit"
DEFAULT_ASR = "mlx-community/parakeet-tdt-0.6b-v3"  # English (fast, precise timestamps)
WHISPER_ASR = "mlx-community/whisper-large-v3-turbo"  # every other source language (Korean, …)

# Source languages: code → (Ukrainian name in the accusative for messages, English name for prompts)
LANGS = {
    "en": ("англійську", "English"), "ko": ("корейську", "Korean"), "ja": ("японську", "Japanese"),
    "zh": ("китайську", "Chinese"), "es": ("іспанську", "Spanish"), "fr": ("французьку", "French"),
    "de": ("німецьку", "German"), "it": ("італійську", "Italian"), "pl": ("польську", "Polish"),
    "pt": ("португальську", "Portuguese"), "tr": ("турецьку", "Turkish"), "uk": ("українську", "Ukrainian"),
}
DEFAULT_SEP_MODEL = "model_bs_roformer_ep_317_sdr_12.9755.ckpt"  # best quality, ~1x real time
FAST_SEP_MODEL = "htdemucs.yaml"  # ~3x faster, a bit more bleed
OMNI_MODEL = "k2-fsa/OmniVoice"

# StyleTTS2-ukrainian (patriotyk, MIT): native Ukrainian TTS that takes explicit word stress
# (dictionary + context) → correct stresses, but no voice cloning. Voices = speakers of its dataset.
ST_MODEL = "patriotyk/styletts2_ukrainian_multispeaker"
ST_SPACE = "patriotyk/styletts2-ukrainian"  # the voice styles (voices/*.pt) live in the demo space
ST_VOICES = {  # name → gender (checked by voice pitch; 30/31 voices transcribed back perfectly)
    "Юрій Вихованець": "male", "Роман Куліш": "male", "Денис Денисенко": "male",
    "Матвій Ніколаєв": "male", "Артем Окороков": "male", "Кирило Татарченко": "male",
    "Павло Буковський": "male", "Олександр Ролдугін": "male", "Тарас Василюк": "male",
    "Вʼячеслав Дудко": "male", "Юрій Кудрявець": "male", "Петро Філяк": "male", "Михайло Тишин": "male",
    "Марта Мольфар": "female", "Марися Нікітюк": "female", "Тетяна Лукинюк": "female",
    "Людмила Чиркова": "female", "Слава Красовська": "female", "Інна Гелевера": "female",
    "Влада Муравець": "female", "Тетяна Гончарова": "female", "Поліна Еккерт(хлопчик)": "child",
    "Гаська Шиян": "female", "Вероніка Дорош": "female", "Катерина Потапенко": "female",
    "Марина Панас": "female", "Вікторія Левченко": "female", "Поліна Еккерт": "female",
    "Марічка Штирбулова": "female", "Анастасія Павленко": "female", "Олена Шверк": "female",
}
ST_DEFAULT_MALE = "Тарас Василюк"
ST_DEFAULT_FEMALE = "Влада Муравець"


def find_st_voice(query: str) -> str | None:
    """Exact name, or a unique case-insensitive prefix/substring ("тарас" → "Тарас Василюк")."""
    def key(x: str) -> str:
        return "".join(ch for ch in x.strip().lower() if ch not in "'’ʼ`")

    q = key(query)
    if not q:
        return None
    for name in ST_VOICES:
        if key(name) == q:
            return name
    hits = [n for n in ST_VOICES if key(n).startswith(q)] or [n for n in ST_VOICES if q in key(n)]
    return hits[0] if len(hits) == 1 else None

# Fast preset voices of robinhad/ukrainian-tts (MIT) and their grammatical gender.
UKR_VOICES = {
    "dmytro": "male",
    "oleksa": "male",
    "mykyta": "male",
    "tetiana": "female",
    "lada": "female",
}

# Natural speaking rate of each engine in Ukrainian syllables per second.
# Used to budget translation length before synthesis (calibrated on M1 Pro).
SYLLABLE_RATE = {"ukr": 5.3, "omni": 6.2, "st": 4.8}

STAGES = ["extract", "separate", "asr", "translate", "tts", "mix", "mux"]


def _file_hash(path: str | None) -> str:
    import hashlib

    try:
        return hashlib.sha1(Path(path).read_bytes()).hexdigest()[:12]
    except (OSError, TypeError):
        return ""


@dataclass
class Options:
    input: str
    output: str
    workdir: str
    voice: str = "dmytro"  # ukr-tts voice | "clone" | "clone:<ref.wav>" | "omni:<instruct>"
    gender: str | None = None  # speaker gender for Ukrainian grammar (male|female)
    ref_text: str | None = None  # transcript of a user-supplied clone reference
    llm: str = DEFAULT_LLM  # mlx-lm repo id or "ollama:<model>"
    asr_model: str = DEFAULT_ASR
    separate: bool = True
    sep_model: str = DEFAULT_SEP_MODEL
    glossary: str | None = None
    keep_original: bool = True
    max_speed: float = 1.25
    pace: float = 5.6  # target pace of the dub, syllables per second (every line, ±5 %)
    tail: float = 0.8  # how far (s) a dub line may run past the original line end
    omni_steps: int = 16  # 16 ≈ real time on M1 Pro; 32 = slower, slightly cleaner
    omni_batch: int = 1  # batching is slower on MPS (padding), keep 1
    duck_db: float | None = None
    sample_rate: int = 44100
    stress: str = "auto"  # off | dict | auto (dictionary + context-resolved homographs)
    stress_dict: str | None = None  # extra stress dictionary (besides ~/.config/uadub/stress.txt)
    source_lang: str = "en"  # language spoken in the video
    subs: str | None = None  # existing subtitles to use instead of speech recognition
    subs_lang: str | None = None  # language of those subtitles (default: source_lang; "uk" = no translation)
    emotion: float = 0.0  # StyleTTS2: share (0–1) of the original line's intonation in the chosen voice
    domain: str | None = None  # None: plain Ukrainian; "auto" or a field name: specialist jargon of that field
    part_minutes: float | None = None  # long videos: None = automatic, 0 = never split, N = parts of ~N min
    keep_parts: bool = False  # keep the part previews after the final assembly
    clip_start: float | None = None  # this run is one part of a long video: its range in the input (s)
    clip_end: float | None = None
    shared_brief: str | None = None  # brief of the whole long video (replaces the per-part brief)
    edge_context: str | None = None  # neighbouring parts' text for translation context
    loudness_target: float | None = None  # LUFS shared by all parts of a long video

    # ---- persistence -------------------------------------------------------
    def save(self, path: Path) -> None:
        Path(path).write_text(json.dumps(asdict(self), ensure_ascii=False, indent=2))

    @classmethod
    def load(cls, path: Path) -> "Options":
        data = json.loads(Path(path).read_text())
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in data.items() if k in known})

    # ---- derived -----------------------------------------------------------
    @property
    def engine(self) -> str:
        """'ukr' preset voices, 'st' StyleTTS2 (correct stress), 'omni' OmniVoice (clone/design)."""
        v = self.voice
        if v == "st" or v.startswith("st:") or v == "duo:st" or v.startswith("duo:st:"):
            return "st"
        return "ukr" if (v in UKR_VOICES or v.startswith("duo:")) else "omni"

    @property
    def text_lang(self) -> str:
        """Language of the text the pipeline translates from (subtitles may differ from speech)."""
        return (self.subs_lang or self.source_lang) if self.subs else self.source_lang

    @property
    def speaker_gender(self) -> str | None:
        if self.gender:
            return self.gender
        if self.source_lang != "en" or self.voice == "duo" or self.voice.startswith("duo:"):
            return None  # films/dramas: many characters → infer per line (single-voice voice-over style)
        if self.voice in UKR_VOICES:
            return UKR_VOICES[self.voice]
        if self.voice == "st" or self.voice.startswith("st:"):
            name = ST_DEFAULT_MALE if self.voice == "st" else self.voice[3:]
            g = ST_VOICES.get(name)
            return g if g in ("male", "female") else None
        if self.voice.startswith("omni:"):
            desc = self.voice.lower()
            if "female" in desc:
                return "female"
            if "male" in desc:
                return "male"
        return None  # clone mode: let the LLM infer from context

    @property
    def work(self) -> Path:
        return Path(self.workdir)

    def fingerprint(self, stage: str) -> dict:
        """Options that influence a stage; a change re-runs it and everything after."""
        keys = {
            "extract": ["input", "sample_rate"],
            "separate": ["separate", "sep_model"],
            "asr": ["asr_model", "source_lang", "subs", "subs_lang"],
            "translate": ["llm", "glossary", "gender", "max_speed", "tail", "stress", "source_lang", "subs_lang"],
            "tts": ["voice", "ref_text", "omni_steps", "max_speed", "tail", "stress", "stress_dict"],
            "mix": ["duck_db", "max_speed"],
            "mux": ["output", "keep_original"],
        }[stage]
        fp = {k: getattr(self, k) for k in keys}
        if stage == "asr":
            fp["version"] = 4  # snap-to-speech for every ASR (Parakeet ends hid the pauses, ADR-029)
        if stage == "asr" and self.subs and Path(self.subs).exists():
            import hashlib

            fp["subs_hash"] = hashlib.sha1(Path(self.subs).read_bytes()).hexdigest()[:12]
        if stage == "tts":
            from .stress import default_dict_paths, dict_fingerprint

            fp["version"] = 9 if self.engine == "st" else 4  # one pace, lines fitted to the time really left (ADR-033)
            if self.pace != 5.6:
                fp["pace"] = self.pace  # only when changed, so existing runs keep their cache
            fp["stress_dict_hash"] = dict_fingerprint(default_dict_paths(self.stress_dict, self.work))
            if self.emotion and self.engine == "st":
                fp["emotion"] = self.emotion  # only when on, so existing runs keep their cache
        if stage == "mix":
            fp["version"] = 4  # pauses mirror the speaker's (ADR-033)
        if stage == "translate":
            fp["version"] = 19  # bump when prompts/budgets change → old runs re-translate
            if self.pace != 5.6:
                fp["pace"] = self.pace  # budgets follow the pace (ADR-033)
            fp["engine"] = self.engine
            fp["speaker_gender"] = self.speaker_gender
            if self.domain:
                fp["domain"] = self.domain  # only when on, so existing runs keep their cache
        if self.clip_start is not None and stage in ("extract", "mix", "mux"):
            fp["clip"] = [self.clip_start, self.clip_end]
        if stage == "translate" and self.shared_brief:
            fp["shared_brief"] = _file_hash(self.shared_brief)
            fp["edge"] = _file_hash(self.edge_context)
        if stage == "mix" and self.loudness_target is not None:
            fp["loudness_target"] = round(self.loudness_target, 2)
        return fp
