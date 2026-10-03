"""Word-stress control.

OmniVoice ignores Cyrillic stress marks (´, +, capitals — see k2-fsa/OmniVoice#52, #65), but
it does follow bracketed ARPAbet transcriptions with stress digits, e.g. «замо́к» →
`[Z AA0 M AO1 K]`. English phonemes add a slight accent, so only selected words are
transcribed:

* words from the user's stress dictionary, and
* in `auto` mode also homographs (ру́ки/руки́, за́мок/замо́к…): grammatical ones are resolved by
  `ukrainian-word-stress`, semantic ones by the LLM during translation (see translate.py),
  which writes the choice into the `tts` text as an acute accent.

For the ukrainian-tts voices the same dictionary is applied as an acute accent, which that
engine honours natively.
"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path

from .config import CACHE_DIR

ACUTE = "́"
VOWELS = "аеєиіїоуюя"
_WORD = re.compile(r"[А-Яа-яЄєІіЇїҐґ'’ʼ́+]+")
DEFAULT_DICT = Path.home() / ".config" / "uadub" / "stress.txt"

# Very frequent words that the stress dictionary lists as homographs only because of a rare or
# archaic reading (вони́ vs во́ни). TTS models say them right on their own; transcribing them
# would only add risk and accent.
COMMON_SKIP = frozenset("""
вони нічого зараз мене тебе себе його неї них нам вам ними мною тобою собою кого чого нього
неї ній ньому тому цього того всього усього тоді доти поки отже також тільки лише щоб якщо
коли чому де куди звідки скільки потім завжди ніколи
""".split())

# Homographs among function words almost always have one everyday reading (коли́, вони́, за́раз);
# only content words (за́мок/замо́к, ру́ки/руки́) are worth resolving.
FUNCTION_POS = frozenset({"ADV", "SCONJ", "CCONJ", "PRON", "PART", "DET", "ADP", "INTJ", "AUX"})

_VOWEL_ARPA = {"а": ["AA"], "е": ["EH"], "є": ["Y", "EH"], "и": ["IH"], "і": ["IY"], "ї": ["Y", "IY"],
               "о": ["AO"], "у": ["UW"], "ю": ["Y", "UW"], "я": ["Y", "AA"]}
_CONS_ARPA = {"б": "B", "в": "V", "г": "HH", "ґ": "G", "д": "D", "ж": "ZH", "з": "Z", "й": "Y",
              "к": "K", "л": "L", "м": "M", "н": "N", "п": "P", "р": "R", "с": "S", "т": "T",
              "ф": "F", "х": "HH", "ц": "T S", "ч": "CH", "ш": "SH", "щ": "SH CH"}
_IOTATED_AFTER_SOFT = {"о": ["Y", "AO"]}  # «льон», «сьогодні»


def parse_marked(word: str) -> tuple[str, int] | None:
    """'замо́к' / 'зам+ок' / 'замОк' → ('замок', 1): plain word and index of the stressed vowel."""
    w = word.strip().replace("’", "'").replace("ʼ", "'")
    plain, idx, v = [], None, 0
    i = 0
    while i < len(w):
        ch = w[i]
        if ch == "+":
            if i + 1 < len(w) and w[i + 1].lower() in VOWELS:
                idx = v
            i += 1
            continue
        if ch in (ACUTE, "´"):
            idx = v - 1 if v > 0 else None
            i += 1
            continue
        if ch.lower() in VOWELS:
            if ch.isupper() and i > 0 and idx is None and not w.isupper():
                idx = v
            v += 1
        plain.append(ch.lower())
        i += 1
    if idx is None or idx < 0:
        return None
    return "".join(plain), idx


def load_dict(paths: list[Path]) -> dict[str, int]:
    out: dict[str, int] = {}
    for p in paths:
        if not p or not Path(p).exists():
            continue
        for line in Path(p).read_text(encoding="utf-8").splitlines():
            line = line.split("#", 1)[0].strip()
            if not line:
                continue
            entry = line.split("=", 1)[-1].strip()  # «замок = замо́к» or just «замо́к»
            for token in entry.split():
                parsed = parse_marked(token)
                if parsed:
                    out[parsed[0]] = parsed[1]
    return out


def dict_fingerprint(paths: list[Path]) -> str:
    h = hashlib.sha1()
    for p in paths:
        if p and Path(p).exists():
            h.update(Path(p).read_bytes())
    return h.hexdigest()[:12]


def to_arpa(word: str, stress_idx: int) -> str:
    w = word.lower().replace("’", "'").replace("ʼ", "'")
    phones: list[str] = []
    v = 0
    i = 0
    while i < len(w):
        ch = w[i]
        if w.startswith("дж", i):
            phones.append("JH")
            i += 2
            continue
        if ch in _VOWEL_ARPA:
            ph = list(_IOTATED_AFTER_SOFT[ch]) if (ch in _IOTATED_AFTER_SOFT and i > 0 and w[i - 1] == "ь") \
                else list(_VOWEL_ARPA[ch])
            ph[-1] += "1" if v == stress_idx else "0"
            phones += ph
            v += 1
        elif ch in _CONS_ARPA:
            phones += _CONS_ARPA[ch].split()
        i += 1
    return "[" + " ".join(phones) + "]"


def with_acute(word: str, stress_idx: int) -> str:
    v = 0
    for i, ch in enumerate(word):
        if ch.lower() in VOWELS:
            if v == stress_idx:
                return word[: i + 1] + ACUTE + word[i + 1 :]
            v += 1
    return word


def _vowel_count(word: str) -> int:
    return sum(ch.lower() in VOWELS for ch in word)


class StressFixer:
    """Turns stress information into something the TTS follows.

    Sources, in priority order: explicit marks already in the text (´ / U+0301 / + — the
    translate stage writes them for homographs), then the user's dictionary.
    target: 'arpa' (OmniVoice) | 'acute' (ukrainian-tts). mode 'off' strips all marks.
    """

    def __init__(self, mode: str = "auto", dict_paths: list[Path] | None = None, target: str = "arpa"):
        self.mode, self.target = mode, target
        self.user = load_dict(dict_paths or []) if mode != "off" else {}

    def apply(self, text: str) -> str:
        if not text:
            return text

        def repl(m: re.Match) -> str:
            word = m.group(0)
            plain = word.replace(ACUTE, "").replace("´", "").replace("+", "")
            if self.mode == "off":
                return plain
            explicit = parse_marked(word) if any(c in word for c in (ACUTE, "´", "+")) else None
            idx = explicit[1] if explicit else self.user.get(plain.lower().replace("’", "'"))
            if idx is None or _vowel_count(plain) < 2:
                return plain
            return to_arpa(plain, idx) if self.target == "arpa" else with_acute(plain, idx)

        return _WORD.sub(repl, text)


class HomographFinder:
    """Finds words whose stress depends on context (ру́ки/руки́, за́мок/замо́к).

    Grammatical homographs are resolved by `ukrainian-word-stress` (Stanza POS + morphology);
    semantic ones (за́мок/замо́к, а́тлас/атла́с, му́ка/мука́) are returned with their options so
    that the LLM can pick the right one from the meaning of the sentence.
    """

    def __init__(self):
        import logging

        logging.getLogger("stanza").setLevel(logging.ERROR)
        from ukrainian_word_stress import OnAmbiguity, Stressifier

        from .tts import quiet_stress_logs

        quiet_stress_logs()

        self._all = Stressifier(stress_symbol=ACUTE, on_ambiguity=OnAmbiguity.All)
        self._ctx = Stressifier(stress_symbol=ACUTE, on_ambiguity=OnAmbiguity.Skip)

    def find(self, text: str) -> list[dict]:
        """[{n, word, options: [vowel idx…], resolved: idx | None}] for every homograph in text."""
        if not text.strip():
            return []
        matches = list(_WORD.finditer(text))
        words = [m.group(0) for m in matches]
        pos = self._pos_by_word(text, matches)
        amb = [m.group(0) for m in _WORD.finditer(self._all(text))]
        ctx = [m.group(0) for m in _WORD.finditer(self._ctx(text))]
        if not (len(words) == len(amb) == len(ctx)):
            return []
        out = []
        for n, (w, a, c) in enumerate(zip(words, amb, ctx)):
            if any(ch in w for ch in (ACUTE, "´", "+")) or _vowel_count(w) < 2 or w.lower() in COMMON_SKIP \
                    or pos[n] in FUNCTION_POS:
                continue
            # Only words whose stress the dictionary cannot settle even with grammar (за́мок/замо́к).
            # Each transcription adds a little English accent, so common words are left alone.
            options = _mark_positions(a)
            if len(options) < 2:
                continue
            resolved = _mark_positions(c)
            out.append({"n": n, "word": w, "options": options,
                        "resolved": resolved[0] if len(resolved) == 1 else None})
        return out

    def _pos_by_word(self, text: str, matches: list) -> list[str]:
        """Universal POS tag (Stanza) for each regex word, matched by character offset."""
        spans = []
        try:
            for sent in self._ctx.nlp(text).sentences:
                for w in sent.words:
                    tok = w.parent
                    spans.append((tok.start_char, tok.end_char, w.upos))
        except Exception:
            return [""] * len(matches)
        out = []
        for m in matches:
            tag = ""
            for a, b, upos in spans:
                if a is not None and a <= m.start() < b:
                    tag = upos
                    break
            out.append(tag)
        return out

    def _dictionary_options(self, word: str) -> list[int]:
        """All stress positions the dictionary knows for this word form, across all grammatical readings."""
        from ukrainian_word_stress.stressify_ import _parse_dictionary_value

        trie = self._ctx.dict
        for form in (word, word.lower(), word.title()):
            if form in trie:
                entries = _parse_dictionary_value(trie[form][0])
                variants = {tuple(acc) for _, acc in entries if len(acc) == 1}
                if len(variants) < 2:
                    return []
                # character offsets → vowel indices
                pos_to_vowel, v = {}, 0
                for i, ch in enumerate(form):
                    if ch.lower() in VOWELS:
                        pos_to_vowel[i + 1] = v  # accent offsets point right after the vowel
                        v += 1
                return sorted({pos_to_vowel[a[0]] for a in variants if a[0] in pos_to_vowel})
        return []


def _mark_positions(marked: str) -> list[int]:
    """Vowel indices that carry an acute in a word like 'за́мо́к' → [0, 1]."""
    out, v = [], 0
    for i, ch in enumerate(marked):
        if ch.lower() in VOWELS:
            if i + 1 < len(marked) and marked[i + 1] == ACUTE:
                out.append(v)
            v += 1
    return out


def variant_caps(word: str, idx: int) -> str:
    """'замок', 1 → 'замОк' (how options are shown to the LLM)."""
    v = 0
    for i, ch in enumerate(word):
        if ch.lower() in VOWELS:
            if v == idx:
                return word[:i] + ch.upper() + word[i + 1 :]
            v += 1
    return word


def apply_marks(text: str, marks: dict[int, int]) -> str:
    """Put an acute on word number n (as counted by _WORD) at vowel marks[n]."""
    n = -1

    def repl(m: re.Match) -> str:
        nonlocal n
        n += 1
        return with_acute(m.group(0), marks[n]) if n in marks else m.group(0)

    return _WORD.sub(repl, text) if marks else text


def default_dict_paths(extra: str | None, work: Path | None = None) -> list[Path]:
    """Global dictionary, the per-video one (workdir/stress.txt) and an optional extra file."""
    paths = [DEFAULT_DICT]
    if work is not None:
        paths.append(Path(work) / "stress.txt")
    if extra:
        paths.append(Path(extra).expanduser())
    return paths


__all__ = ["StressFixer", "HomographFinder", "apply_marks", "variant_caps", "parse_marked", "to_arpa", "load_dict", "dict_fingerprint", "default_dict_paths",
           "DEFAULT_DICT", "CACHE_DIR"]


# ---------------------------------------------------------------------------
# `uadub --stress-lookup`: what the stress dictionary knows about a word (for people and agents)
_UPOS = {"NOUN": "іменник", "PROPN": "власна назва", "VERB": "дієслово", "AUX": "дієслово",
         "ADJ": "прикметник", "ADV": "прислівник", "PRON": "займенник", "DET": "займенник",
         "NUM": "числівник", "CCONJ": "сполучник", "SCONJ": "сполучник", "ADP": "прийменник",
         "PART": "частка", "INTJ": "вигук"}
_FEATS = {"Case": {"Nom": "наз.", "Gen": "род.", "Dat": "дав.", "Acc": "знах.", "Ins": "оруд.",
                   "Loc": "місц.", "Voc": "клич."},
          "Number": {"Sing": "одн.", "Plur": "мн."},
          "Gender": {"Masc": "ч. р.", "Fem": "ж. р.", "Neut": "с. р."}}


def _plus_at(word: str, offset: int) -> str:
    """Accent offset of ukrainian-word-stress (index right after the vowel) → «зам+ок»."""
    return word[: offset - 1] + "+" + word[offset - 1 :]


def _describe(tags: list[str]) -> str:
    kv = dict(t.split("=", 1) for t in tags if "=" in t)
    parts = [_UPOS.get(kv.get("upos", ""), kv.get("upos", ""))]
    for key in ("Gender", "Number", "Case"):
        if key in kv:
            parts.append(_FEATS[key].get(kv[key], kv[key]))
    return " ".join(p for p in parts if p)


def lookup(word: str, work: Path | None = None) -> str:
    """Human-readable stress options of one word form: dictionary readings + your stress.txt."""
    from ukrainian_word_stress import Stressifier
    from ukrainian_word_stress.stressify_ import _parse_dictionary_value

    from .tts import quiet_stress_logs

    quiet_stress_logs()
    w = word.strip().replace("’", "'").replace("ʼ", "'").replace("+", "").replace(ACUTE, "")
    lines = [w]
    own = load_dict(default_dict_paths(None, work))
    if w.lower() in own:
        idx, v = own[w.lower()], 0
        for i, ch in enumerate(w):
            if ch.lower() in VOWELS:
                if v == idx:
                    lines.append(f"  {w[:i]}+{w[i:]} — ваш stress.txt (має перевагу над словником)")
                v += 1
    trie = Stressifier().dict
    found = False
    for form in dict.fromkeys((w.lower(), w)):
        if form not in trie:
            continue
        found = True
        by_variant: dict[str, list[str]] = {}
        entries = _parse_dictionary_value(trie[form][0])
        both_ok = all(len(acc) > 1 for _, acc in entries)  # one reading with two stresses: по́милка / поми́лка
        for tags, acc in entries:
            for a in acc:  # several offsets = both stresses are allowed (по́милка / поми́лка)
                desc = _describe(tags) if tags else "усі форми"
                lst = by_variant.setdefault(_plus_at(form, a), [])
                if desc not in lst:
                    lst.append(desc)
        note = " (з великої літери: власна назва)" if form != w.lower() else ""
        for variant, descs in by_variant.items():
            lines.append(f"  {variant} — {'; '.join(descs)}{note}")
        if len(by_variant) > 1 and form == w.lower():
            lines.append("  → обидва наголоси допустимі нормою, нічого не виправляй" if both_ok else
                         "  → кілька варіантів: обери за змістом і граматикою речення")
    if not found:
        lines.append("  немає в словнику — синтезатор вгадуватиме; постав наголос сам (за орфоепічним словником)")
    return "\n".join(lines)
