"""Context-aware, length-aware EN→UK translation for dubbing with a local LLM.

Approach (borrowed from VideoLingo / duration-aware MT papers, simplified):
1. one "brief" pass over the whole transcript: topic, tone, ти/ви, speaker gender, glossary;
2. chunked translation with neighbouring context and a per-line syllable budget;
3. up to two "shorten" passes for lines that still will not fit their time slot.
"""

from __future__ import annotations

import json
import math
import re
from pathlib import Path

from .llm import LLM, extract_json
from .segments import slot
from .textnorm import syllables, to_speech_text

CHUNK = 20
BRIEF_CHAR_LIMIT = 24_000

SYSTEM = """You are an expert audiovisual translator who writes Ukrainian voice-over and dubbing scripts.
Translate {src_name} speech lines (field "src") into natural, idiomatic, modern spoken Ukrainian.

How to work:
A. Meaning first. Before translating, read "transcript_context" — the original transcript around these lines — and the summary, and understand what the speaker means and why: the argument, the logic between sentences, the jokes. Translate sense for sense, never word for word: restructure sentences, change word order, replace constructions that sound foreign in Ukrainian, make implicit links explicit when needed. A Ukrainian viewer must get the same meaning and the same feeling as an English viewer.
B. The transcript comes from automatic speech recognition and can contain misheard words, missing punctuation or broken sentences (e.g. "Quen 3.6" for the model "Qwen 3.6"). Infer what was actually said from the context and translate that.
C. Idioms, proverbs, set phrases and figures of speech: never translate them literally. Use an established Ukrainian equivalent with the same meaning and register (e.g. "a piece of cake" → «простіше простого», "kill two birds with one stone" → «убити двох зайців одним пострілом», "break the ice" → «розтопити кригу», "beat around the bush" → «ходити околясом», "once in a blue moon" → «раз на сто років», "costs an arm and a leg" → «коштує шалені гроші», "the ball is in your court" → «тепер слово за тобою», "hit the nail on the head" → «влучити в саму точку»). The same applies to set connective phrases: "when it comes to" → «коли йдеться про», "at the end of the day" → «зрештою», "that being said" → «утім», "as a matter of fact" → «насправді»; spoken fillers ("kind of", "you know", "like", "basically") are usually dropped. If there is no Ukrainian idiom, express the meaning naturally. Adapt jokes and wordplay so they work in Ukrainian.

Rules:
1. Output exactly one translation per input line with the same "id". Never merge, split, skip or reorder lines. If a sentence continues across lines, break the Ukrainian at a natural point so each line still matches its own timing.
2. Timing: every line has "max_syl" — the maximum number of Ukrainian syllables (vowel letters а е є и і ї о у ю я) that fits the original time slot. Translate the full meaning and use the available length: a good line is usually close to max_syl. Only if a faithful translation is longer, condense it just enough: drop filler words and repetitions, use shorter synonyms, simplify syntax — never drop facts, names, numbers or instructions. Never pad short lines.
3. Style: clear, spoken and pleasant to listen to; keep the tone (casual, formal, humorous). Use correct literary Ukrainian, no Russianisms, surzhyk or word-for-word calques. Keep a consistent form of address (ти/ви).
4. {gender_rule}
5. No anglicisms. Translate everything that has a Ukrainian word: UI labels, button and menu names, settings and ordinary technical terms (e.g. "Use this model" → «Використати цю модель», "browse" → «переглянути», "download" → «завантажити», "quantization" → «квантування», "checkbox" → «прапорець», "runtime" → «середовище виконання», "default" → «стандартний»). Never leave English phrases untranslated and never write English words in Cyrillic letters (no «юз зіс модел», «брауз», «квантайзейшн», «чекбокс», «дефолтний», «юзати»). Keep in Latin only proper names: products, companies, people, model names, file formats, code and commands (LM Studio, Hugging Face, Qwen, GGUF, MLX).
6. If "uk" contains digits, symbols (% $ € + / & @ °), Latin letters or abbreviations, also add "tts": the same line fully spelled out exactly as it should be pronounced in Ukrainian — numbers as words in the correct case and gender, symbols and units expanded, proper names in Cyrillic the way Ukrainian speakers say them (e.g. "GitHub" → "ґітхаб", "LM Studio" → "ел-ем студіо", "Hugging Face" → "хаґінґ фейс", "API" → "ей-пі-ай", "3.5%" → "три з половиною відсотка"). "tts" must contain only Cyrillic words and punctuation — no Latin letters, digits or stress marks. Otherwise omit "tts".
7. Do not add stress marks (´, +) anywhere — stress is handled separately.
{lang_rules}{glossary}
Return ONLY a JSON object: {{"lines": [{{"id": <int>, "uk": "<subtitle text>", "tts": "<optional speakable text>"}}]}}
Each line object has only "id", "uk" and optionally "tts" — never repeat "src", "max_syl", "speaker" or other input fields."""

BRIEF_SYSTEM = """You prepare a translation brief for dubbing a {src_name} video into Ukrainian.
Read the whole transcript (it comes from speech recognition and may contain misheard words), understand the speaker's line of thought, and return ONLY JSON:
{"summary": "<3-5 sentences in Ukrainian: topic, genre, audience, tone, and the main line of argument or story>",
 "speaker_gender": "male|female|unknown",
 "address": "ти|ви",
 "characters": [{"name": "<name as written in the transcript>", "uk": "<Ukrainian form>", "gender": "male|female|unknown"}],
 "glossary": [{"src": "<term/name/recurring phrase>", "uk": "<recommended Ukrainian rendering; keep brand names as is>"}],
 "idioms": [{"src": "<idiom, proverb, set phrase, joke or cultural reference as it appears>", "uk": "<Ukrainian equivalent with the same meaning and register, not a literal translation>"}],
 "asr_fixes": [{"heard": "<misrecognised word in the transcript>", "meant": "<what was actually said>"}]}
Include at most 25 glossary entries, only for terms that really matter for consistency. For technical terms give the established Ukrainian term (quantization → квантування), never a Cyrillic transliteration of the English word; keep proper names (products, companies) as they are. List named characters (people) in "characters"; empty list if none. List every idiom or figure of speech in "idioms" (empty list if none) and obvious speech-recognition errors in "asr_fixes".{lang_rules}"""

STRESS_SYSTEM = """You are a Ukrainian pronunciation expert. Each item is a word inside a sentence that will be read aloud. The word is a homograph: its stress depends on its meaning or grammatical form. Pick the variant whose stressed vowel (shown in UPPERCASE) is correct in this sentence. Examples: зАмок = castle, замОк = lock; Атлас = book of maps, атлАс = fabric; мУка = torment, мукА = flour; рУки = nominative plural (мої рУки), рукИ = genitive singular (немає рукИ); гОри = mountains, з горИ = from the mountain.
If both variants are acceptable in this sentence (free variation, e.g. нАтискати/натискАти), answer "both" — then nothing is marked.
Return ONLY JSON: {"items": [{"id": "<id>", "answer": "<the correct variant, copied exactly, or both>"}]}"""

ANGLICISM_SYSTEM = """You are a Ukrainian editor of dubbing scripts. In each line the listed "suspect" words look like English words written in Cyrillic letters, untranslated English, or non-standard slang. Rewrite "uk" in natural standard Ukrainian: replace such words with proper Ukrainian equivalents (button and menu labels are translated by meaning). Keep proper names of products, companies, people and file formats (LM Studio, Hugging Face, GGUF) unchanged. If a suspect word is in fact a correct Ukrainian word or a proper name, keep it. Keep the meaning and roughly the same length.
Follow the same "tts" rule: if the new "uk" has digits, symbols, Latin letters or abbreviations, add "tts" with everything spelled out as pronounced in Ukrainian; otherwise omit it.
Return ONLY JSON: {"lines": [{"id": <int>, "uk": "...", "tts": "..."}]}"""

SHORTEN_SYSTEM = """You edit Ukrainian dubbing lines that are too long for their time slots.
For each line rewrite "uk" so that it has at most "max_syl" syllables (vowel letters а е є и і ї о у ю я), keeping the meaning of "src" and a natural spoken style. Prefer cutting fillers, shorter synonyms and simpler syntax; a short Ukrainian idiom often says more than a long literal phrase. The result must stay a grammatical, complete Ukrainian sentence — never leave half of a set phrase (wrong: «Коли справа стає завантаження»; right: «Коли йдеться про завантаження»).
Follow the same "tts" rule: if the new "uk" has digits, symbols, Latin letters or abbreviations, add "tts" with everything spelled out as pronounced; otherwise omit it.
Return ONLY JSON: {"lines": [{"id": <int>, "uk": "...", "tts": "..."}]}"""


LANG_RULES = {
    "ko": """
8. Korean specifics: transliterate Korean personal and place names into Ukrainian by the Kontsevych system (система Концевича), keeping the Korean order (surname first) and exactly the same form every time (e.g. 김 → Кім, 이 → Лі, 박 → Пак, 서울 → Сеул). Render forms of address (오빠, 언니, 형, 누나, 선배, 씨, 님, 아저씨) naturally — by the person's name, a natural Ukrainian address, or a consistent transliteration such as «оппа» — never a clumsy literal «старший брат» each time. Show politeness levels with ти/ви. Drama dialogue is elliptical: restore implied subjects so the Ukrainian sounds natural.
""",
    "ja": """
8. Japanese specifics: transliterate names into Ukrainian by the Kovalenko system (система Коваленка), consistently. Render honorific suffixes (-san, -kun, -chan, -senpai) naturally or omit them; show politeness via ти/ви.
""",
}


def _gender_rule(gender: str | None) -> str:
    if gender == "female":
        return ("The voice is a woman's: unless the context clearly says otherwise, use feminine "
                "first-person forms (я зробила, я впевнена).")
    if gender == "male":
        return ("The voice is a man's: unless the context clearly says otherwise, use masculine "
                "first-person forms (я зробив, я впевнений).")
    return ("There are several speakers. A line may carry \"speaker\": the speaker's gender guessed from the voice "
            "pitch (usually right, but children and shouting can fool it). Use it together with the context and the "
            "character list for first-person forms (я зробив / я зробила). The gender of the person addressed "
            "(ти готовий / ти готова, ти снідав / ти снідала) must come from context — in a two-person dialogue it "
            "is usually the other speaker.")


def _load_glossary(path: str | None) -> list[dict]:
    if not path:
        return []
    items = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        for sep in ("=", "\t", " - ", ":"):
            if sep in line:
                en, uk = line.split(sep, 1)
                items.append({"src": en.strip(), "uk": uk.strip()})
                break
    return items


_LINE_RE = re.compile(
    r'"id"\s*:\s*(\d+)\s*,\s*"uk"\s*:\s*"((?:[^"\\]|\\.)*)"(?:\s*,\s*"tts"\s*:\s*"((?:[^"\\]|\\.)*)")?')


_STR_FIELD = r'"{}"\s*:\s*"((?:[^"\\]|\\.)*)"'


def _salvage_lines(reply: str) -> list[dict]:
    """Recover well-formed line objects from a reply whose JSON is broken elsewhere
    (e.g. cut off by the token limit), whatever order or extra fields they have."""
    out, seen = [], set()
    for obj in re.findall(r"\{[^{}]*\}", reply):
        mid = re.search(r'"id"\s*:\s*(\d+)', obj)
        muk = re.search(_STR_FIELD.format("uk"), obj)
        if not (mid and muk):
            continue
        try:
            item = {"id": int(mid.group(1)), "uk": json.loads(f'"{muk.group(1)}"')}
            mt = re.search(_STR_FIELD.format("tts"), obj)
            if mt:
                item["tts"] = json.loads(f'"{mt.group(1)}"')
        except json.JSONDecodeError:
            continue
        if item["id"] not in seen:
            seen.add(item["id"])
            out.append(item)
    if out:
        return out
    for m in _LINE_RE.finditer(reply):
        try:
            item = {"id": int(m.group(1)), "uk": json.loads(f'"{m.group(2)}"')}
            if m.group(3):
                item["tts"] = json.loads(f'"{m.group(3)}"')
            out.append(item)
        except json.JSONDecodeError:
            continue
    return out


def _ask(llm: LLM, system: str, user: str, *, max_tokens: int, retries: int = 2) -> dict:
    # Greedy decoding first: quantized MoE models produce the most reliable JSON that way.
    last_err: Exception | None = None
    for attempt in range(retries + 1):
        reply = llm.chat(system, user, max_tokens=max_tokens, temperature=0.0 if attempt == 0 else 0.2 * attempt)
        try:
            data = extract_json(reply)
            if isinstance(data, list):
                data = {"lines": data}
            if isinstance(data, dict):
                return data
        except ValueError as e:
            last_err = e
            salvaged = _salvage_lines(reply)
            if salvaged:
                return {"lines": salvaged}
    raise RuntimeError(f"LLM не дала валідний JSON після {retries + 1} спроб: {last_err}")


def _clean_tts(uk: str, tts: str) -> str:
    """Drop an LLM 'tts' line that broke its own rules (Latin or digits left) — the
    deterministic normalizer reading the subtitle text is safer than garbage like «кра komma»."""
    if not tts or tts == uk:
        return ""
    if re.search(r"[A-Za-z0-9]", tts):
        return ""
    return tts


def speech_text(u: dict) -> str:
    return to_speech_text(u.get("tts") or u.get("uk") or "")


def _syl(u: dict) -> int:
    return syllables(speech_text(u))


def make_brief(llm: LLM, units: list[dict], src_name: str = "English", lang_rules: str = "") -> dict:
    transcript = "\n".join(u["text"] for u in units)
    if len(transcript) > BRIEF_CHAR_LIMIT:
        half = BRIEF_CHAR_LIMIT // 2
        transcript = transcript[:half] + "\n…\n" + transcript[-half:]
    try:
        system = BRIEF_SYSTEM.replace("{src_name}", src_name).replace("{lang_rules}", lang_rules)
        brief = _ask(llm, system, transcript, max_tokens=2000)
        return brief if isinstance(brief, dict) else {}
    except Exception as e:  # the brief is helpful, not essential
        print(f"   (бриф не вдався: {e})")
        return {}


def set_budgets(units: list[dict], *, rate: float, max_speed: float = 1.25) -> None:
    # The dub may be sped up to `max_speed`, so budget for almost that pace; otherwise fast
    # speakers get heavily abridged translations.
    fill = max(1.0, min(max_speed, 1.3) * 0.96)
    for u in units:
        u["max_syl"] = max(3, math.floor(slot(u) * rate * fill))


def translate_units(llm: LLM, units: list[dict], *, rate: float, gender: str | None,
                    glossary_path: str | None, max_speed: float = 1.25, stress: str = "auto",
                    source_lang: str = "en", log=print) -> dict:
    from .config import LANGS

    src_name = LANGS.get(source_lang, (None, source_lang))[1]
    lang_rules = LANG_RULES.get(source_lang, "")
    set_budgets(units, rate=rate, max_speed=max_speed)

    log("   • аналіз тексту (тема, тон, глосарій)…")
    brief = make_brief(llm, units, src_name, lang_rules)
    if not gender and brief.get("speaker_gender") in ("male", "female"):
        gender = brief["speaker_gender"]
    glossary = (brief.get("glossary") or []) + _load_glossary(glossary_path)
    gl_text = ""
    if glossary:
        gl_text = "\nGlossary (use these renderings consistently):\n" + "\n".join(
            f'- {g.get("src") or g.get("en")} → {g.get("uk")}' for g in glossary
            if (g.get("src") or g.get("en")) and g.get("uk"))
    idioms = [i for i in (brief.get("idioms") or []) if isinstance(i, dict) and i.get("src") and i.get("uk")]
    if idioms:
        gl_text += "\nIdioms and figures of speech in this video (use these Ukrainian equivalents, not literal translations):\n" + \
            "\n".join(f'- {i["src"]} → {i["uk"]}' for i in idioms)
    fixes = [f for f in (brief.get("asr_fixes") or []) if isinstance(f, dict) and f.get("heard") and f.get("meant")]
    if fixes:
        gl_text += "\nSpeech-recognition errors in the transcript (translate what was meant):\n" + \
            "\n".join(f'- "{f["heard"]}" means "{f["meant"]}"' for f in fixes)
    chars = [c for c in (brief.get("characters") or []) if isinstance(c, dict) and c.get("name")]
    if chars:
        gl_text += "\nCharacters (name → Ukrainian form, gender):\n" + "\n".join(
            f'- {c["name"]} → {c.get("uk") or c["name"]} ({c.get("gender", "unknown")})' for c in chars)
    system = SYSTEM.format(gender_rule=_gender_rule(gender), glossary=gl_text, src_name=src_name,
                           lang_rules=lang_rules)
    context = {k: brief[k] for k in ("summary", "address") if brief.get(k)}  # + transcript window per chunk

    by_id = {u["id"]: u for u in units}
    n_chunks = math.ceil(len(units) / CHUNK)
    for ci in range(n_chunks):
        chunk = units[ci * CHUNK : (ci + 1) * CHUNK]
        log(f"   • переклад {ci + 1}/{n_chunks}")
        _translate_chunk(llm, system, context, units, chunk, by_id, fixed_gender=gender)

    for round_no in (1, 2):
        too_long = [u for u in units if u.get("uk") and _syl(u) > u["max_syl"] * 1.08]
        if not too_long:
            break
        log(f"   • скорочення задовгих реплік ({len(too_long)}), прохід {round_no}")
        for i in range(0, len(too_long), CHUNK):
            _shorten(llm, too_long[i : i + CHUNK])

    fixed = fix_anglicisms(llm, units, glossary=glossary, log=log)
    stats = resolve_homographs(llm, units, log=log) if stress == "auto" else {}
    return {"brief": brief, "gender": gender, "homographs": stats, "anglicism_fixes": fixed}


_LAT_WORD = re.compile(r"[A-Za-z][A-Za-z'-]*")
_CYR_WORD = re.compile(r"[А-Яа-яЄєІіЇїҐґ'’]+")


def find_suspects(u: dict, known, keep: set[str]) -> list[str]:
    """Words of a line that look like anglicisms: Cyrillic words unknown to the Ukrainian dictionary,
    and Latin words that are not proper names (lower-case English words, or phrases in quotes)."""
    out = []
    for w in _CYR_WORD.findall(u.get("uk") or ""):  # tts may legitimately spell brands in Cyrillic
        lw = w.lower().strip("'’")
        if len(lw) >= 4 and lw not in keep and not known(lw):
            out.append(w)
    for w in _LAT_WORD.findall(u.get("uk") or ""):
        if w.islower() and len(w) > 2 and w.lower() not in keep:
            out.append(w)
    return sorted(set(out))


def _dictionary_lookup():
    try:
        import importlib.resources as res

        import marisa_trie

        trie = marisa_trie.BytesTrie()
        trie.load(str(res.files("ukrainian_word_stress").joinpath("data/stress.trie")))
    except Exception:
        return None

    def known(word: str) -> bool:
        return any(f in trie for f in (word, word.title(), word.upper()))

    return known


def fix_anglicisms(llm: LLM, units: list[dict], *, glossary: list[dict] | None = None, log=print) -> int:
    """One editing pass over lines with transliterated/untranslated English words."""
    known = _dictionary_lookup()
    if known is None:
        return 0
    keep: set[str] = set()
    for g in glossary or []:  # proper names the brief decided to keep
        for w in _CYR_WORD.findall(str(g.get("uk", ""))) + _LAT_WORD.findall(str(g.get("uk", ""))):
            keep.add(w.lower())
    todo = [(u, s) for u in units if u.get("uk") for s in [find_suspects(u, known, keep)] if s]
    if not todo:
        return 0
    log(f"   • прибираю англіцизми: {len(todo)} реплік")
    changed = 0
    by_id = {u["id"]: u for u, _ in todo}
    for i in range(0, len(todo), CHUNK):
        batch = todo[i : i + CHUNK]
        payload = {"lines": [{"id": u["id"], "src": u["text"], "uk": u["uk"], "suspect": s, "max_syl": u["max_syl"]}
                             for u, s in batch]}
        try:
            data = _ask(llm, ANGLICISM_SYSTEM, json.dumps(payload, ensure_ascii=False),
                        max_tokens=200 + 160 * len(batch))
        except RuntimeError:
            continue
        for item in data.get("lines", []):
            try:
                u = by_id[int(item.get("id"))]
            except (KeyError, TypeError, ValueError):
                continue
            uk = str(item.get("uk", "")).strip()
            if uk and uk != u["uk"]:
                u["uk"] = uk
                tts = _clean_tts(uk, str(item.get("tts") or "").strip())
                if tts:
                    u["tts"] = tts
                else:
                    u.pop("tts", None)
                changed += 1
    return changed


def _pick_variant(item: dict, answer) -> int | None:
    """Map the LLM's answer (a variant like «замОк») back to a vowel index; None if unclear."""
    if not isinstance(answer, str) or not answer.strip():
        return None
    ans = answer.strip().strip(".,!?«»\"'")
    if ans.lower() in ("both", "обидва"):
        return None
    if ans in item["variants"]:
        return item["_options"][item["variants"].index(ans)]
    from .stress import VOWELS

    v, stressed = 0, []
    for ch in ans:
        if ch.lower() in VOWELS:
            if ch.isupper():
                stressed.append(v)
            v += 1
    if len(stressed) == 1 and stressed[0] in item["_options"]:
        return stressed[0]
    return None


def resolve_homographs(llm: LLM, units: list[dict], *, log=print) -> dict:
    """Mark the stress of homographs in each line's `tts` text with an acute accent.

    Grammatical homographs are resolved by ukrainian-word-stress; for semantic ones the LLM
    picks one of the dictionary's variants (a constrained choice, which LLMs do reliably,
    unlike free-form stress placement).
    """
    from .stress import HomographFinder, apply_marks, variant_caps

    try:
        finder = HomographFinder()
    except Exception as e:  # stanza model missing etc. — stress is a nice-to-have
        log(f"   • наголоси омографів пропущено: {e}")
        return {}
    marks: dict[int, dict[int, int]] = {}
    ask: list[dict] = []
    for ui, u in enumerate(units):
        for h in finder.find(speech_text(u)):
            # Stanza's morphology guess is wrong surprisingly often (e.g. «немає руки»), so the
            # LLM makes every call; it only has to choose between the dictionary's variants.
            ask.append({"id": f"{ui}:{h['n']}", "sentence": speech_text(u), "word": h["word"],
                        "variants": [variant_caps(h["word"].lower(), o) for o in h["options"]],
                        "_options": h["options"]})
    if ask:
        log(f"   • наголоси омографів: {len(ask)} слів, вибір за контекстом (LLM)")
        for i in range(0, len(ask), 30):
            batch = ask[i : i + 30]
            # Ask twice with the variants in opposite order; keep only answers that agree, so
            # the model's position bias cannot put a wrong stress on a word (then the TTS decides).
            votes: list[dict[str, int]] = []
            for reverse in (False, True):
                payload = {"items": [{"id": a["id"], "sentence": a["sentence"], "word": a["word"],
                                      "variants": a["variants"][::-1] if reverse else a["variants"]}
                                     for a in batch]}
                try:
                    data = _ask(llm, STRESS_SYSTEM, json.dumps(payload, ensure_ascii=False),
                                max_tokens=60 + 30 * len(batch))
                except RuntimeError:
                    data = {}
                by_id = {a["id"]: a for a in batch}
                got: dict[str, int] = {}
                for item in data.get("items", []):
                    a = by_id.get(str(item.get("id")))
                    choice = _pick_variant(a, item.get("answer")) if a else None
                    if choice is not None:
                        got[a["id"]] = choice
                votes.append(got)
            for aid, choice in votes[0].items():
                if votes[1].get(aid) == choice:
                    ui, n = map(int, aid.split(":"))
                    marks.setdefault(ui, {})[n] = choice
    for ui, m in marks.items():
        units[ui]["tts"] = apply_marks(speech_text(units[ui]), m)
    return {"marked": sum(map(len, marks.values())), "homographs": len(ask)}


CONTEXT_CHARS = 5000


def _context_window(units: list[dict], first: int, n: int) -> str:
    """Original text around the chunk (roughly half before, half after), for sense-for-sense translation.
    Lines of the chunk itself are marked with » so the model sees where they sit in the story."""
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
    return "\n".join(parts)


def _translate_chunk(llm, system, context, units, chunk, by_id, fixed_gender=None) -> None:
    first = units.index(chunk[0])
    prev = [{"src": u["text"], "uk": u.get("uk", "")} for u in units[max(0, first - 6) : first]]
    nxt = [u["text"] for u in units[first + len(chunk) : first + len(chunk) + 4]]
    payload = {**context, "transcript_context": _context_window(units, first, len(chunk)), "previous": prev,
               "lines": [{"id": u["id"], "src": u["text"], "max_syl": u["max_syl"],
                          **({"speaker": u["gender"]} if u.get("gender") and not fixed_gender else {})}
                         for u in chunk],
               "next": nxt}
    pending = {u["id"] for u in chunk}
    for _ in range(3):
        try:
            data = _ask(llm, system, json.dumps(payload, ensure_ascii=False),
                        max_tokens=400 + 260 * len(payload["lines"]))
        except RuntimeError:
            left = [u for u in chunk if u["id"] in pending]
            if len(left) <= 3:
                break
            half = len(left) // 2  # a smaller request is far less likely to break
            for part in (left[:half], left[half:]):
                _translate_chunk(llm, system, context, units, part, by_id, fixed_gender)
            return
        for item in data.get("lines", []):
            try:
                uid = int(item.get("id"))
            except (TypeError, ValueError):
                continue
            if uid in pending and str(item.get("uk", "")).strip():
                by_id[uid]["uk"] = str(item["uk"]).strip()
                tts = _clean_tts(by_id[uid]["uk"], str(item.get("tts") or "").strip())
                if tts:
                    by_id[uid]["tts"] = tts
                else:
                    by_id[uid].pop("tts", None)
                pending.discard(uid)
        if not pending:
            return
        payload["lines"] = [l for l in payload["lines"] if l["id"] in pending]
    for uid in pending:  # still missing: keep something speakable rather than crash
        by_id[uid]["uk"] = by_id[uid]["text"]
        by_id[uid]["untranslated"] = True


def _shorten(llm: LLM, items: list[dict]) -> None:
    payload = {"lines": [{"id": u["id"], "src": u["text"], "uk": u["uk"], "syl": _syl(u),
                          "max_syl": u["max_syl"]} for u in items]}
    try:
        data = _ask(llm, SHORTEN_SYSTEM, json.dumps(payload, ensure_ascii=False), max_tokens=400 + 260 * len(items))
    except RuntimeError:
        return
    by_id = {u["id"]: u for u in items}
    for item in data.get("lines", []):
        try:
            u = by_id[int(item.get("id"))]
        except (KeyError, TypeError, ValueError):
            continue
        uk_new = str(item.get("uk", "")).strip()
        cand = {"uk": uk_new, "tts": _clean_tts(uk_new, str(item.get("tts") or "").strip())}
        if cand["uk"] and syllables(to_speech_text(cand["tts"] or cand["uk"])) < _syl(u):
            u["uk"] = cand["uk"]
            if cand["tts"]:
                u["tts"] = cand["tts"]
            else:
                u.pop("tts", None)
