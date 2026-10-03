"""Ukrainian text helpers: syllable counting and a deterministic 'make it speakable' pass.

The LLM is asked to produce a fully spelled-out `tts` string, so the code here is a
safety net for whatever slips through (digits, symbols, Latin words).
"""

from __future__ import annotations

import re

try:  # num2words is a runtime dependency; keep import soft for unit tests
    from num2words import num2words
except ImportError:  # pragma: no cover
    num2words = None

UK_VOWELS = frozenset("аеєиіїоуюяАЕЄИІЇОУЮЯ")
_LAT_VOWEL_RUN = re.compile(r"[aeiouy]+", re.I)
_APOSTROPHES = re.compile(r"[’ʼ`´‘]")
_SPACES = re.compile(r"\s+")


def normalize_apostrophes(text: str) -> str:
    return _APOSTROPHES.sub("'", text)


def syllables(text: str) -> int:
    """Approximate number of spoken syllables in (mostly) Ukrainian text.

    In Ukrainian every syllable has exactly one vowel letter, so counting vowels is
    exact. Latin words and digits that survived are estimated.
    """
    n = sum(ch in UK_VOWELS for ch in text)
    for word in re.findall(r"[A-Za-z]+", text):
        if word.isupper() and len(word) <= 5:
            n += len(word)  # spelled out letter by letter
        else:
            n += max(1, len(_LAT_VOWEL_RUN.findall(word)))
    for digits in re.findall(r"\d+", text):
        n += 2 * len(digits)
    return n


# ---------------------------------------------------------------------------
# Latin → Cyrillic fallback (rough, pronunciation-oriented)
# ---------------------------------------------------------------------------
_LETTER_NAMES = {
    "a": "ей", "b": "бі", "c": "сі", "d": "ді", "e": "і", "f": "еф", "g": "джі",
    "h": "ейч", "i": "ай", "j": "джей", "k": "кей", "l": "ел", "m": "ем", "n": "ен",
    "o": "оу", "p": "пі", "q": "к'ю", "r": "ар", "s": "ес", "t": "ті", "u": "ю",
    "v": "ві", "w": "дабл-ю", "x": "екс", "y": "вай", "z": "зед",
}

_TRANSLIT = [
    ("ing", "інґ"), ("sch", "ш"), ("tch", "ч"), ("ch", "ч"), ("sh", "ш"), ("th", "т"), ("ph", "ф"),
    ("gg", "ґ"), ("ll", "л"), ("ss", "с"), ("tt", "т"), ("pp", "п"), ("mm", "м"), ("nn", "н"),
    ("ck", "к"), ("qu", "кв"), ("ee", "і"), ("oo", "у"), ("ea", "і"), ("ou", "ау"),
    ("ow", "оу"), ("ai", "ей"), ("ay", "ей"), ("oa", "оу"), ("wh", "в"), ("x", "кс"),
    ("a", "а"), ("b", "б"), ("c", "к"), ("d", "д"), ("e", "е"), ("f", "ф"), ("g", "г"),
    ("h", "х"), ("i", "і"), ("j", "дж"), ("k", "к"), ("l", "л"), ("m", "м"), ("n", "н"),
    ("o", "о"), ("p", "п"), ("q", "к"), ("r", "р"), ("s", "с"), ("t", "т"), ("u", "а"),
    ("v", "в"), ("w", "в"), ("y", "і"), ("z", "з"),
]


def _spell_letters(word: str) -> str:
    return "-".join(_LETTER_NAMES.get(ch.lower(), ch) for ch in word)


def _translit_word(word: str) -> str:
    low = word.lower()
    out, i = [], 0
    while i < len(low):
        for src, dst in _TRANSLIT:
            if low.startswith(src, i):
                out.append(dst)
                i += len(src)
                break
        else:
            out.append(low[i])
            i += 1
    res = "".join(out)
    if res.endswith("е") and len(res) > 3 and low.endswith("e"):
        res = res[:-1]  # silent final e: "code" → "код"
    return res[:1].upper() + res[1:] if word[:1].isupper() else res


# How Ukrainian speakers usually say common tech names (used when the LLM gave no "tts").
PRONOUNCE = {
    "lm studio": "ел-ем студіо", "hugging face": "хаґінґ фейс", "github": "ґітхаб", "gitlab": "ґітлаб",
    "youtube": "ютуб", "google": "ґуґл", "apple": "епл", "macbook": "макбук", "iphone": "айфон",
    "windows": "віндовс", "linux": "лінукс", "macos": "мак-о-ес", "python": "пайтон",
    "javascript": "джаваскрипт", "docker": "докер", "kubernetes": "кубернетіс", "ollama": "оллама",
    "llama": "лама", "qwen": "квен", "gemma": "джемма", "mistral": "містраль", "openai": "оупен-ей-ай",
    "chatgpt": "чат-джі-пі-ті", "claude": "клод", "nvidia": "енвідіа", "cuda": "куда",
    "gguf": "джі-джі-ю-еф", "mlx": "ем-ел-екс", "gpu": "джі-пі-ю", "cpu": "сі-пі-ю", "ram": "рем",
    "vram": "ві-рем", "api": "ей-пі-ай", "ai": "ей-ай", "ui": "ю-ай", "url": "ю-ар-ел",
}


def _load_user_pronounce() -> dict[str, str]:
    from pathlib import Path

    path = Path.home() / ".config" / "uadub" / "pronounce.txt"
    out = {}
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.split("#", 1)[0]
            if "=" in line:
                k, v = line.split("=", 1)
                if k.strip() and v.strip():
                    out[k.strip().lower()] = v.strip()
    return out


def apply_pronunciations(text: str) -> str:
    table = {**PRONOUNCE, **_load_user_pronounce()}
    for name in sorted(table, key=len, reverse=True):  # longest first: "lm studio" before "studio"
        text = re.sub(rf"(?<![A-Za-z]){re.escape(name)}(?![A-Za-z])", table[name], text, flags=re.I)
    return text


def latin_to_cyrillic(text: str) -> str:
    def repl(m: re.Match) -> str:
        w = m.group(0)
        if w.isupper() and len(w) <= 5:
            return _spell_letters(w)
        return _translit_word(w)

    return re.sub(r"[A-Za-z]+", repl, text)


# ---------------------------------------------------------------------------
# Numbers and symbols
# ---------------------------------------------------------------------------
_THOUSANDS = re.compile(r"(?<=\d)[,   ](?=\d{3}(?!\d))")
_CURRENCY = [
    (re.compile(r"\$\s*(\d[\d.,]*)"), r"\1 доларів"),
    (re.compile(r"€\s*(\d[\d.,]*)"), r"\1 євро"),
    (re.compile(r"£\s*(\d[\d.,]*)"), r"\1 фунтів"),
    (re.compile(r"(\d[\d.,]*)\s*(?:грн|₴)"), r"\1 гривень"),
]
_NUMBER = re.compile(r"(?<![\w])(\d+(?:[.,]\d+)?)(\s*%)?")
_SYMBOLS = {
    "&": " і ", "=": " дорівнює ", "№": " номер ", "@": " ет ",
    "%": " відсотків ", "°": " градусів ",
}
_DROP = re.compile(r"[*_#<>\[\]{}|~^\\]")
_ABBREV = [
    (re.compile(r"\bмлрд\b\.?"), "мільярдів"), (re.compile(r"\bмлн\b\.?"), "мільйонів"),
    (re.compile(r"\bтис\.(?=\s|$)|\bтис\b"), "тисяч"), (re.compile(r"\bкм\b"), "кілометрів"),
    (re.compile(r"\bкг\b"), "кілограмів"), (re.compile(r"\bхв\b\.?"), "хвилин"),
    (re.compile(r"\bгод\b\.?"), "годин"), (re.compile(r"\bт\.\s?д\."), "так далі"),
    (re.compile(r"\bт\.\s?п\."), "тому подібне"), (re.compile(r"\bнапр\."), "наприклад"),
]


def _number_words(m: re.Match) -> str:
    raw = m.group(1).replace(",", ".")
    if num2words is None:
        return m.group(0)
    try:
        value = float(raw) if "." in raw else int(raw)
        words = num2words(value, lang="uk")
    except Exception:
        return m.group(0)
    if m.group(2):
        words += " відсотків"
    return words


def to_speech_text(text: str) -> str:
    """Return a version of `text` that a Ukrainian TTS can read aloud."""
    t = normalize_apostrophes(text)
    t = _THOUSANDS.sub("", t)
    for pattern, repl in _CURRENCY:
        t = pattern.sub(repl, t)
    t = _NUMBER.sub(_number_words, t)
    for sym, word in _SYMBOLS.items():
        t = t.replace(sym, word)
    t = re.sub(r"\+(?![аеєиіїоуюяАЕЄИІЇОУЮЯ])", " плюс ", t)  # '+' before a vowel = stress mark
    for pattern, word in _ABBREV:
        t = pattern.sub(word, t)
    t = _DROP.sub(" ", t)
    t = apply_pronunciations(t)
    t = latin_to_cyrillic(t)
    t = _SPACES.sub(" ", t).strip()
    t = re.sub(r"\s+([,.!?;:])", r"\1", t)
    return t


def needs_speech_form(text: str) -> bool:
    return bool(re.search(r"[0-9A-Za-z%$€£&+=№@°/]", text))
