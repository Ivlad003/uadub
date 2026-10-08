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
A. Meaning first. Before translating, read "transcript_context" — the original transcript around these lines — and the summary, and understand what the speaker means and why: the argument, the logic between sentences, the jokes. Translate sense for sense, never word for word: restructure sentences, change word order, replace constructions that sound foreign in Ukrainian, make implicit links explicit when needed. A Ukrainian viewer must get the same meaning and the same feeling as the original viewer.
B. The transcript comes from automatic speech recognition and can contain misheard words, missing punctuation or broken sentences (e.g. "Quen 3.6" for the model "Qwen 3.6"). Infer what was actually said from the context and translate that.
C. Idioms, proverbs, set phrases, jokes and wordplay: never translate them literally — use an established Ukrainian equivalent with the same meaning and register, or express the meaning naturally if there is none. Spoken fillers are usually dropped.

Rules:
1. Output exactly one translation per input line with the same "id". Never merge, split, skip or reorder lines. If a sentence continues across lines, break the Ukrainian at a natural point so each line still matches its own timing.
2. Length: every line has "max_syl" (Ukrainian syllables = vowel letters а е є и і ї о у ю я) and "max_words" — what fits the time of the original line at a natural pace. Aim for 80–100 % of max_words: at most max_words, and not much shorter, otherwise the voice falls silent while the speaker is still talking. Never pad with words that say nothing. If the faithful translation does not fit, condense in this order: drop fillers and connectives, drop what the neighbouring line already says, use shorter synonyms, simplify the syntax. Never drop facts, names, numbers, on-screen labels or the object of an instruction (what to click, where to go).
3. Spoken style. These lines are spoken aloud, not read: short sentences; verbs instead of chains of nouns in the genitive («процесу завантаження моделей» → «як завантажити модель»); no participial constructions («натиснувши», «обраний користувачем») — use a clause with a verb; a natural spoken word order; the connectives people actually say (тож, отже, а ще, далі, тепер). Keep the tone of the original (casual, formal, humorous). Use correct literary Ukrainian, no Russianisms, surzhyk or word-for-word calques (e.g. concessive «як би ви не зробили» is a calque — write «хоч як ви зробите» / «хоч би як ви зробили»; «в кінці кінців» → «зрештою»; «приймати участь» → «брати участь»). Use the form of address given in "address" (ти/ви) in every line.
4. The lines are one continuous speech: read your own "previous" translations and do not start two neighbouring lines with the same word, do not repeat a phrase the previous line already used (unless the original repeats it), and vary the connectives.
5. {gender_rule}
6. {terms_rule}
7. If "uk" contains digits, symbols (% $ € + / & @ °), Latin letters or abbreviations, also add "tts": the same line fully spelled out exactly as it should be pronounced in Ukrainian — numbers as words in the correct case and gender, symbols and units expanded, proper names in Cyrillic the way Ukrainian speakers say them. Abbreviations are spelled letter by letter with hyphens: "API" → «ей-пі-ай», "LM Studio" → «ел-ем студіо», "MLX" → «ем-ел-ікс», "PC" → «пі-сі». Version numbers are read digit group by digit group: "Qwen 3.6" → «квен три шість», "Python 3.12" → «пайтон три дванадцять»; other numbers normally: "3.5%" → «три з половиною відсотка», "GitHub" → «ґітхаб», "Hugging Face" → «хаґінґ фейс». Use the pronunciations given in the glossary. "tts" must contain only Cyrillic words and punctuation — no Latin letters, digits or stress marks. Otherwise omit "tts".
8. Do not add stress marks (´, +) anywhere — stress is handled separately.
{lang_rules}{glossary}
Return ONLY a JSON object: {{"lines": [{{"id": <int>, "uk": "<subtitle text>", "tts": "<optional speakable text>"}}]}}
Each line object has only "id", "uk" and optionally "tts" — never repeat "src", "max_syl", "speaker" or other input fields."""

PLAIN_TERMS_RULE = """No anglicisms. Translate everything that has a Ukrainian word: UI labels, button and menu names, settings and ordinary technical terms (e.g. "Use this model" → «Використати цю модель», "browse" → «переглянути», "download" → «завантажити», "quantization" → «квантування», "checkbox" → «прапорець», "runtime" → «середовище виконання», "default" → «стандартний»). Never leave English phrases untranslated and never write English words in Cyrillic letters (no «юз зіс модел», «брауз», «квантайзейшн», «чекбокс», «дефолтний», «юзати»). Keep in Latin only proper names: products, companies, people, model names, file formats, code and commands (LM Studio, Hugging Face, Qwen, GGUF, MLX)."""

DOMAIN_TERMS_RULE = """Audience: specialists in {domain}. Use the terminology Ukrainian professionals in this field actually use when they talk, including established anglicisms (e.g. in software: фреймворк, деплой, коміт, пул-реквест, бекенд, реліз, промпт) — prefer them to artificial native coinages nobody says. Where the established term is native, use it (database → база даних, quantization → квантування). Still no slang or ad-hoc transliterations of ordinary words (no «юзати», «дефолтний», «юз зіс модел»), and never leave whole English phrases untranslated. Labels of buttons, menus and settings that the viewer sees on screen stay exactly as in the original, in Latin and in quotes (натисніть «Use this model»), so viewers can find them. Keep proper names in Latin (LM Studio, Hugging Face, Qwen, GGUF, MLX)."""

BRIEF_SYSTEM = """You prepare a translation brief for dubbing a {src_name} video into Ukrainian.
Read the whole transcript (it comes from speech recognition and may contain misheard words), understand the speaker's line of thought, and return ONLY JSON:
{"summary": "<3-5 sentences in Ukrainian: topic, genre, audience, and the main line of argument or story>",
 "style": "<1-3 sentences in Ukrainian: the speaker's register and manner — casual or formal, jokes, how they address the viewer, typical sentence length — so the translator can keep the same voice>",
 "domain": "<field of the video in Ukrainian, 1-4 words: «IT / розробка ПЗ», «медицина», «фінанси», «кулінарія», «загальна тема» …>",
 "speaker_gender": "male|female|unknown",
 "address": "ти|ви",
 "characters": [{"name": "<name as written in the transcript>", "uk": "<Ukrainian form>", "gender": "male|female|unknown"}],
 "glossary": [{"src": "<term/name/recurring phrase>", "uk": "<recommended Ukrainian rendering; keep brand names as is>", "say": "<for names and abbreviations kept in Latin: how a Ukrainian speaker pronounces them, in Cyrillic letters, abbreviations letter by letter with hyphens — ел-ем студіо, ей-пі-ай, хаґінґ фейс; omit for Ukrainian terms>"}],
 "idioms": [{"src": "<idiom, proverb, set phrase, joke or cultural reference as it appears>", "uk": "<Ukrainian equivalent with the same meaning and register, not a literal translation>"}],
 "asr_fixes": [{"heard": "<misrecognised word in the transcript>", "meant": "<what was actually said>"}]}
"address": a speaker talking to an audience (tutorials, lectures, presentations, reviews, news) addresses viewers as «ви»; «ти» only when the speech is a dialogue between people who are close (family, friends, lovers, children) or the speaker clearly talks to one close person.
{glossary_rule} List named characters (people) in "characters"; empty list if none. List every idiom or figure of speech in "idioms" (empty list if none) and obvious speech-recognition errors in "asr_fixes".{lang_rules}"""

STRESS_SYSTEM = """You are a Ukrainian pronunciation expert. Each item is a word inside a sentence that will be read aloud. The word is a homograph: its stress depends on its meaning or grammatical form. Pick the variant whose stressed vowel (shown in UPPERCASE) is correct in this sentence. Examples: зАмок = castle, замОк = lock; Атлас = book of maps, атлАс = fabric; мУка = torment, мукА = flour; рУки = nominative plural (мої рУки), рукИ = genitive singular (немає рукИ); гОри = mountains, з горИ = from the mountain.
Frequent pairs: сАмий / та сАма / ті сАмі = the same (той сАмий список, те сАме вікно); самИй / самА / самІ = by itself, the very (система самА підбере, ви самІ оберете, самИй низ); рОзмір = size (the usual noun); прАвильний is the standard form.
If both variants are acceptable in this sentence (free variation, e.g. нАтискати/натискАти), answer "both" — then nothing is marked.
Return ONLY JSON: {"items": [{"id": "<id>", "answer": "<the correct variant, copied exactly, or both>"}]}"""

ANGLICISM_SYSTEM = """You are a Ukrainian editor of dubbing scripts. In each line the listed "suspect" words look like English words written in Cyrillic letters, untranslated English, or non-standard slang. {fix_rule} Keep proper names of products, companies, people and file formats (LM Studio, Hugging Face, GGUF) unchanged. If a suspect word is in fact a correct Ukrainian word or a proper name, keep it. Keep the meaning and roughly the same length.
Follow the same "tts" rule: if the new "uk" has digits, symbols, Latin letters or abbreviations, add "tts" with everything spelled out as pronounced in Ukrainian; otherwise omit it.
Return ONLY JSON: {"lines": [{"id": <int>, "uk": "...", "tts": "..."}]}"""

PLAIN_GLOSSARY_RULE = "Include at most 25 glossary entries, only for terms that really matter for consistency. For technical terms give the established Ukrainian term (quantization → квантування), never a Cyrillic transliteration of the English word; keep proper names (products, companies) as they are."

DOMAIN_GLOSSARY_RULE = "The translation is for specialists in {domain}. Include at most 40 glossary entries: the terms that matter for consistency and the professional jargon of this field, rendered the way Ukrainian specialists actually say them in speech — an established anglicism when that is what they say (framework → фреймворк, deploy → деплой, pull request → пул-реквест), a native term when that is the established one (database → база даних); keep proper names (products, companies) as they are."

PLAIN_FIX_RULE = "Rewrite \"uk\" in natural standard Ukrainian: replace such words with proper Ukrainian equivalents (button and menu labels are translated by meaning)."

DOMAIN_FIX_RULE = "The audience are specialists in {domain}. Rewrite \"uk\" only where a suspect word is something these specialists would not say: slang, an ad-hoc transliteration of an ordinary word (юзати, дефолтний) or an untranslated English phrase. Keep established professional terms of this field (e.g. фреймворк, деплой, коміт) and on-screen button or menu labels quoted in Latin (натисніть «Use this model»)."

SPELL_SYSTEM = """You prepare Ukrainian dubbing lines for a speech synthesizer that can read only Cyrillic. For each line write "tts": the same "uk" text spelled out exactly as a Ukrainian speaker would say it aloud — Latin words, UI labels, product names and abbreviations in Cyrillic the way Ukrainian specialists pronounce them ("Use this model" → «юз зіс модел», "LM Studio" → «ел-ем студіо», "API" → «ей-пі-ай»), numbers and symbols as words in the correct case. Change nothing else in the line and keep its capital letters. "tts" must contain only Cyrillic words and punctuation — no Latin letters, digits or stress marks.
Return ONLY JSON: {"lines": [{"id": <int>, "tts": "..."}]}"""

SHORTEN_SYSTEM = """You edit Ukrainian dubbing lines that are too long for their time slots.
Each line gives "src" (the original), "uk" (the current translation), "over" (how many syllables too long it is; syllables = vowel letters а е є и і ї о у ю я), "max_words" (the word count that fits), and the neighbouring lines "previous" / "next" as they will be spoken. Rewrite "uk" so that it fits: at most "max_words" words, and cut only as much as needed — keep at least 80 % of "max_words", a line much shorter than its time leaves a hole in the dub.
Cut in this order: 1) fillers and connectives (тож, отже, просто, власне, насправді); 2) what the previous or next line already says; 3) shorter synonyms; 4) rebuild the sentence with a verb instead of noun chains; a short Ukrainian idiom often says more than a long literal phrase. Never drop facts, names, numbers, on-screen labels in quotes, or the object of an instruction (what to click, where to go) — if nothing else can go, keep those and cut description.
The result must stay a grammatical, complete, natural spoken Ukrainian sentence in the same style — never leave half of a set phrase (wrong: «Коли справа стає завантаження»; right: «Коли йдеться про завантаження»). Keep the form of address (ти/ви).{attempt}
Follow the same "tts" rule: if the new "uk" has digits, symbols, Latin letters or abbreviations, add "tts" with everything spelled out as pronounced (abbreviations letter by letter with hyphens, versions digit by digit: «квен три шість»); otherwise omit it.
Return ONLY JSON: {"lines": [{"id": <int>, "uk": "...", "tts": "..."}]}"""

EXPAND_SYSTEM = """You complete Ukrainian dubbing lines that came out much shorter than their time slot: the voice would fall silent while the speaker on screen is still talking.
Each line gives "src" (the original), "uk" (the current, condensed translation), "max_words" (the word count that fits the time) and the neighbouring lines "previous" / "next" as they will be spoken. Rewrite "uk" as a fuller translation of "src": restore the details, examples and clauses that were condensed away, in the same spoken style and form of address, aiming for 80–100 % of "max_words". Never pad with words that say nothing, never add facts that are not in "src", and keep every name, number and quoted label.
Follow the same "tts" rule: if the new "uk" has digits, symbols, Latin letters or abbreviations, add "tts" with everything spelled out as pronounced (abbreviations letter by letter with hyphens, versions digit by digit); otherwise omit it.
Return ONLY JSON: {"lines": [{"id": <int>, "uk": "...", "tts": "..."}]}"""

SHORTEN_RETRY = " This is the second attempt: the previous rewrite was still too long by the given number of syllables, so cut more decisively — rebuild the sentence rather than trimming words."


LANG_RULES = {
    "en": """
9. English specifics: common idioms and their Ukrainian equivalents — "a piece of cake" → «простіше простого», "kill two birds with one stone" → «убити двох зайців одним пострілом», "break the ice" → «розтопити кригу», "beat around the bush" → «ходити околясом», "once in a blue moon" → «раз на сто років», "costs an arm and a leg" → «коштує шалені гроші», "the ball is in your court" → «тепер слово за тобою», "hit the nail on the head" → «влучити в саму точку». Set connectives: "when it comes to" → «коли йдеться про», "at the end of the day" → «зрештою», "that being said" → «утім», "as a matter of fact" → «насправді». Fillers "kind of", "you know", "like", "basically", "so" at the start of a sentence are dropped.
""",
    "ko": """
9. Korean specifics: transliterate Korean personal and place names into Ukrainian by the Kontsevych system (система Концевича), keeping the Korean order (surname first) and exactly the same form every time (e.g. 김 → Кім, 이 → Лі, 박 → Пак, 서울 → Сеул). Render forms of address (오빠, 언니, 형, 누나, 선배, 씨, 님, 아저씨) naturally — by the person's name, a natural Ukrainian address, or a consistent transliteration such as «оппа» — never a clumsy literal «старший брат» each time. Show politeness levels with ти/ви. Drama dialogue is elliptical: restore implied subjects so the Ukrainian sounds natural.
""",
    "ja": """
9. Japanese specifics: transliterate names into Ukrainian by the Kovalenko system (система Коваленка), consistently. Render honorific suffixes (-san, -kun, -chan, -senpai) naturally or omit them; show politeness via ти/ви.
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
    if re.search(r"[^\W\d_]", re.sub(r"[А-Яа-яІіЇїЄєҐґ\u0301]", "", tts)):  # letters of another script: LLM garbage
        return ""
    return tts


def speech_text(u: dict) -> str:
    return to_speech_text(u.get("tts") or u.get("uk") or "")


def _syl(u: dict) -> int:
    return syllables(speech_text(u))


def make_brief(llm: LLM, units: list[dict], src_name: str = "English", lang_rules: str = "",
               domain: str | None = None) -> dict:
    """domain: None — plain mode; "auto" — specialist mode, field detected here; else the given field."""
    transcript = "\n".join(u["text"] for u in units)
    if len(transcript) > BRIEF_CHAR_LIMIT:
        half = BRIEF_CHAR_LIMIT // 2
        transcript = transcript[:half] + "\n…\n" + transcript[-half:]
    try:
        system = brief_system(src_name, lang_rules, domain)
        brief = _ask(llm, system, transcript, max_tokens=2000)
        return brief if isinstance(brief, dict) else {}
    except Exception as e:  # the brief is helpful, not essential
        print(f"   (бриф не вдався: {e})")
        return {}


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
        log("   • бриф: шматок 1/1")
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
        except Exception:  # any failure of the merge request: the rule-based merge below
            merged = {}
        if not isinstance(merged, dict) or not isinstance(merged.get("glossary"), list):
            log("   (зведення через LLM не вдалося — зводжу за правилами)")
            merged = merge_briefs_fallback(briefs, max_glossary)
        merged["glossary"] = merged["glossary"][:max_glossary]  # the model may ignore the limit
    if domain and domain != "auto":
        merged["domain"] = domain
    return merged


def _field(domain: str) -> str:
    return "the field of this video (see \"domain\")" if domain == "auto" else domain


def brief_system(src_name: str, lang_rules: str, domain: str | None) -> str:
    rule = PLAIN_GLOSSARY_RULE if not domain else DOMAIN_GLOSSARY_RULE.replace("{domain}", _field(domain))
    if domain and domain != "auto":
        rule += f" The field is given by the user: «{domain}» — put it in \"domain\"."
    return (BRIEF_SYSTEM.replace("{src_name}", src_name).replace("{glossary_rule}", rule)
            .replace("{lang_rules}", lang_rules))


def terms_rule(domain: str | None) -> str:
    """Rule 5 of the translation prompt: plain Ukrainian, or the jargon of the given field."""
    return DOMAIN_TERMS_RULE.replace("{domain}", domain) if domain else PLAIN_TERMS_RULE


def anglicism_system(domain: str | None) -> str:
    rule = DOMAIN_FIX_RULE.replace("{domain}", domain) if domain else PLAIN_FIX_RULE
    return ANGLICISM_SYSTEM.replace("{fix_rule}", rule)


_LATIN_TERM = re.compile(r"[A-Za-z](?:[A-Za-z0-9+-]|\.(?=[A-Za-z0-9]))*(?:\s+[A-Za-z](?:[A-Za-z0-9+-]|\.(?=[A-Za-z0-9]))*)*")
_ACRONYM = re.compile(r"(?<![A-Za-z])[A-Z][A-Z0-9]{1,3}(?![a-z])")  # LM, MLX, API, PC
_WORD = re.compile(r"[^\W\d_]+(?:['’][^\W\d_]+)*|\d+(?:[.,]\d+)?")


def _norm_word(w: str) -> str:
    return re.sub(r"[^\w']", "", w.replace("’", "'")).lower()


def _span_for_term(uk: str, tts: str, m: re.Match) -> tuple[int, int] | None:
    """Character span inside `tts` that pronounces the Latin term `m` of `uk`.

    Cyrillic words around the term in `uk` are looked up in `tts`; whatever lies between them is
    the pronunciation. Returns None when the neighbours cannot be found unambiguously.
    """
    before = [_norm_word(w) for w in _WORD.findall(uk[: m.start()])]
    after = [_norm_word(w) for w in _WORD.findall(uk[m.end():])]
    toks = [(t.start(), t.end(), _norm_word(t.group())) for t in re.finditer(r"\S+", tts)]
    lo = 0
    if before:
        idx = [i for i, t in enumerate(toks) if t[2] == before[-1]]
        if len(idx) != 1:
            return None
        lo = toks[idx[0]][1]
    hi = len(tts)
    if after:
        idx = [i for i, t in enumerate(toks) if t[2] == after[0] and t[0] >= lo]
        if not idx:
            return None
        hi = toks[idx[0]][0]
    start, end = lo, hi
    while start < end and not tts[start].isalpha():
        start += 1
    while end > start and not tts[end - 1].isalpha():
        end -= 1
    return (start, end) if end > start else None


_QUOTED = re.compile(r"[«\"„“”]([^«»\"„“”]+)[»\"“”]")
_NUMBER_TOKEN = re.compile(r"(?<![\w.])\d+(?:[.,]\d+)?(?![\w])")


def _plain_quotes(text: str) -> str:
    return re.sub(r"[\"„“”»«]", "\"", text).lower()


def lost_facts(old: str, new: str) -> list[str]:
    """Facts of `old` (a line) that `new` (its shortened rewrite) no longer contains:
    quoted on-screen labels, numbers and Latin names. Used to reject a rewrite that cut too much."""
    lost: list[str] = []
    hay = _plain_quotes(new)
    for m in _QUOTED.finditer(old):
        if f'"{m.group(1).lower()}"' not in hay:
            lost.append(f"«{m.group(1)}»")
    for m in _NUMBER_TOKEN.finditer(old):
        if m.group() not in new:
            lost.append(m.group())
    for m in _LATIN_TERM.finditer(old):
        if m.group().lower() not in new.lower():
            lost.append(m.group())
    return lost


def glossary_pronunciations(glossary: list[dict]) -> dict[str, str]:
    """Brief glossary entries with a Cyrillic "say": name → how to pronounce it (stress marks dropped)."""
    out: dict[str, str] = {}
    for g in glossary or []:
        src, say = str(g.get("src") or g.get("en") or "").strip(), str(g.get("say") or "").strip()
        say = say.replace("+", "").replace("\u0301", "")
        if src and say and not re.search(r"[A-Za-z0-9]", say) and re.search(r"[А-Яа-яІіЇїЄєҐґ]", say):
            out[src] = say
    return out


def unify_pronunciations(units: list[dict], pronunciations: dict[str, str] | None = None) -> int:
    """Make the LLM pronounce each Latin name the same way in every line.

    The model writes «ел ем студіо» in one line and «ель ем студіо» or «ел-ем студіо» in another,
    which is audible. A name the brief's glossary spelled out ("say") takes that form; a term with an
    acronym takes textnorm's hyphenated letter spelling; any other Latin term that occurs in several
    lines takes its most frequent spelling. Returns the number of lines changed.
    """
    found: dict[str, list[tuple[dict, int, int, str]]] = {}
    spelled: dict[str, str] = {}  # lowercased term → as written (case matters for acronyms)
    for u in units:
        uk, tts = u.get("uk") or "", u.get("tts")
        if not tts:
            continue
        for m in _LATIN_TERM.finditer(uk):
            span = _span_for_term(uk, tts, m)
            if span:
                term = re.sub(r"\s+", " ", m.group())
                spelled.setdefault(term.lower(), term)
                found.setdefault(term.lower(), []).append((u, *span, tts[span[0] : span[1]]))
    changed = 0
    edits: dict[int, tuple[dict, list[tuple[int, int, str]]]] = {}
    key = lambda text: re.sub(r"\s+", " ", text.lower())  # noqa: E731
    given = {k.lower(): v for k, v in (pronunciations or {}).items()}
    for term, occurrences in found.items():
        if term in given:  # the brief said how this name is pronounced
            canonical = given[term]
        elif _ACRONYM.search(spelled[term]):
            # letter names: textnorm's hyphenated spelling («ел-ем студіо») is what StyleTTS2 says
            # clearly; the LLM's loose «ел ем» / «ель ем» comes out slurred
            canonical = to_speech_text(spelled[term])
        elif len(occurrences) < 2:
            continue
        else:
            counts: dict[str, int] = {}
            for _, _, _, text in occurrences:
                counts[key(text)] = counts.get(key(text), 0) + 1
            best = max(counts, key=lambda k: (counts[k], -list(counts).index(k)))
            canonical = next(text for _, _, _, text in occurrences if key(text) == best)
        best = key(canonical)
        for u, start, end, text in occurrences:
            if key(text) == best:
                continue
            repl = canonical[0].upper() + canonical[1:] if text[0].isupper() else canonical.lower()
            edits.setdefault(id(u), (u, []))[1].append((start, end, repl))
    for u, spans in edits.values():  # right to left, so earlier spans keep their offsets
        for start, end, repl in sorted(spans, key=lambda e: -e[0]):
            u["tts"] = u["tts"][:start] + repl + u["tts"][end:]
        changed += 1
    return changed


PAUSE_RESERVE = 0.3  # seconds of the slot kept for the pause before the next line
SYL_PER_WORD = 2.4  # average spoken Ukrainian
SHORTEN_FLOOR = 0.7  # a rewrite shorter than this share of the budget threw away too much …
SHORTEN_FLOOR_UNLESS = 1.3  # … unless the line was over the budget by more than this factor


EXPAND_BELOW = 0.7  # a line under this share of its budget leaves the voice silent while the speaker talks


def needs_expansion(*, syl: int, max_syl: int, src: str) -> bool:
    """A translation far shorter than its time, of a source line with enough content to restore."""
    return syl < EXPAND_BELOW * max_syl and len(src.split()) >= 5


def expand_ok(*, old_syl: int, new_syl: int, max_syl: int) -> bool:
    """Accept a fuller rewrite only if it is longer and still within the budget (+8 %)."""
    return new_syl > old_syl and new_syl <= 1.08 * max_syl


def expand_outcome(*, old_syl: int, new_syl: int, max_syl: int) -> str:
    """What to do with a fuller rewrite: the model overshoots the budget by 30–50 % more often than
    not, so a rewrite up to 1.4× the budget is kept and sent through one shortening pass (reverted
    if it is still over 1.15×); beyond that a trim will not get there."""
    if new_syl <= old_syl:
        return "reject"
    if new_syl <= 1.08 * max_syl:
        return "accept"
    return "shorten" if new_syl <= 1.4 * max_syl else "reject"


def shorten_ok(*, old_syl: int, new_syl: int, max_syl: int) -> bool:
    """Accept a shortened line only if it is shorter and did not undershoot the budget badly:
    a 26-syllable line for a budget of 24 must not come back as 14 (a hole in the dub)."""
    if new_syl >= old_syl:
        return False
    return new_syl >= SHORTEN_FLOOR * max_syl or old_syl > SHORTEN_FLOOR_UNLESS * max_syl


def set_budgets(units: list[dict], *, rate: float, max_speed: float = 1.25, pause: float = PAUSE_RESERVE,
                fill: float | None = None) -> None:
    # Budget for a natural pace with a breath before the next line. A little of the allowed
    # speed-up is assumed (40 % of it), otherwise fast speakers get heavily abridged translations;
    # the rest stays in reserve, so lines are rarely sped up at all (ADR-029). Engines driven at the
    # dub's pace pass `rate=pace, fill=1.05` (the pace tolerance, ADR-033).
    if fill is None:
        fill = 1.0 + (min(max_speed, 1.3) - 1.0) * 0.4
    for u in units:
        spoken = max(u["end"] - u["start"], slot(u) - pause)
        u["max_syl"] = max(3, math.floor(spoken * rate * fill))
        u["max_words"] = max(2, round(u["max_syl"] / SYL_PER_WORD))  # LLMs count words, not syllables


def translate_units(llm: LLM, units: list[dict], *, rate: float, gender: str | None,
                    glossary_path: str | None, max_speed: float = 1.25, stress: str = "auto",
                    source_lang: str = "en", domain: str | None = None, shared_brief: dict | None = None,
                    edge: dict | None = None, engine: str = "omni", fill: float | None = None, log=print) -> dict:
    """domain: None — plain Ukrainian for everyone; "auto" or a field name — specialist jargon of that field."""
    from .config import LANGS

    src_name = LANGS.get(source_lang, (None, source_lang))[1]
    lang_rules = LANG_RULES.get(source_lang, "")
    set_budgets(units, rate=rate, max_speed=max_speed, fill=fill)

    if shared_brief is not None:  # a part of a long video: one brief for the whole video
        log("   • спільний бриф довгого відео")
        brief = shared_brief
    else:
        log("   • аналіз тексту (тема, тон, глосарій)…")
        brief = make_brief(llm, units, src_name, lang_rules, domain)
    field = str(brief.get("domain") or "").strip()
    if domain and domain != "auto":
        field = domain
    if field:
        log(f"   • сфера: {field}" + (" (фаховий переклад)" if domain else ""))
    expert = (field or "the field of this video") if domain else None
    if not gender and brief.get("speaker_gender") in ("male", "female"):
        gender = brief["speaker_gender"]
    glossary = (brief.get("glossary") or []) + _load_glossary(glossary_path)
    gl_text = ""
    if glossary:
        says = glossary_pronunciations(glossary)
        gl_text = "\nGlossary (use these renderings consistently; «say» = how to pronounce it in tts):\n" + "\n".join(
            f'- {g.get("src") or g.get("en")} → {g.get("uk")}'
            + (f' (say: {says[g.get("src") or g.get("en")]})' if (g.get("src") or g.get("en")) in says else "")
            for g in glossary if (g.get("src") or g.get("en")) and g.get("uk"))
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
                           lang_rules=lang_rules, terms_rule=terms_rule(expert))
    context = {k: brief[k] for k in ("summary", "style", "address") if brief.get(k)}
    context.setdefault("address", "ви")
    if expert:
        context["domain"] = expert  # + transcript window per chunk

    by_id = {u["id"]: u for u in units}
    n_chunks = math.ceil(len(units) / CHUNK)
    for ci in range(n_chunks):
        chunk = units[ci * CHUNK : (ci + 1) * CHUNK]
        log(f"   • переклад {ci + 1}/{n_chunks}")
        _translate_chunk(llm, system, context, units, chunk, by_id, fixed_gender=gender, edge=edge)

    for round_no in (1, 2, 3):  # the third pass only for lines still well over (they cause the drift)
        too_long = [u for u in units if u.get("uk") and _syl(u) > u["max_syl"] * (1.08 if round_no < 3 else 1.15)]
        if not too_long:
            break
        log(f"   • скорочення задовгих реплік ({len(too_long)}), прохід {round_no}")
        for i in range(0, len(too_long), CHUNK):
            _shorten(llm, too_long[i : i + CHUNK], units=units, context=context, attempt=round_no)

    short = [u for u in units if u.get("uk") and needs_expansion(syl=_syl(u), max_syl=u["max_syl"], src=u["text"])]
    if short:  # the opposite failure: a hole in the dub where the translation came out far too short
        log(f"   • доповнення закоротких реплік ({len(short)})")
        for i in range(0, len(short), CHUNK):
            over = _expand(llm, short[i : i + CHUNK], units=units, context=context)
            if over:  # the fuller line overshot: one trim, and back to the old line if still far over
                _shorten(llm, [u for u, _ in over], units=units, context=context, attempt=1)
                for u, (old_uk, old_tts) in over:
                    if _syl(u) > 1.15 * u["max_syl"]:
                        u["uk"] = old_uk
                        if old_tts:
                            u["tts"] = old_tts
                        else:
                            u.pop("tts", None)

    same_start = repeated_starts(units)
    if same_start:
        log(f"   • різні початки сусідніх реплік ({len(same_start)})")
        _vary_starts(llm, [units[i] for i in same_start], units=units, context=context)

    fixed = fix_anglicisms(llm, units, glossary=glossary, domain=expert, log=log)
    if expert:  # many on-screen labels stay in Latin: make sure each is spelled for the voice
        spell_latin(llm, units, log=log)
    same = unify_pronunciations(units, glossary_pronunciations(glossary))
    if same:
        log(f"   • однакова вимова назв: виправлено {same} реплік")
    # StyleTTS2 follows every mark at no cost, so it also gets the function-word homographs (са́мий/сами́й)
    stats = resolve_homographs(llm, units, broad=(engine == "st"), log=log) if stress == "auto" else {}
    return {"brief": brief, "gender": gender, "homographs": stats, "anglicism_fixes": fixed,
            "domain": field or None, "domain_mode": bool(domain)}


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


def spell_latin(llm: LLM, units: list[dict], *, log=print) -> int:
    """Ask for a Cyrillic "tts" spelling of lines that still have Latin letters and none yet;
    the letter-by-letter fallback would read «Use this model» as «асе тіс модел»."""
    todo = [u for u in units if u.get("uk") and not u.get("tts") and _LAT_WORD.search(u["uk"])]
    if not todo:
        return 0
    log(f"   • вимова латиниці: {len(todo)} реплік")
    by_id = {u["id"]: u for u in todo}
    done = 0
    for i in range(0, len(todo), CHUNK):
        batch = todo[i : i + CHUNK]
        payload = {"lines": [{"id": u["id"], "uk": u["uk"]} for u in batch]}
        try:
            data = _ask(llm, SPELL_SYSTEM, json.dumps(payload, ensure_ascii=False), max_tokens=200 + 120 * len(batch))
        except RuntimeError:
            continue
        for item in data.get("lines", []):
            try:
                u = by_id[int(item.get("id"))]
            except (KeyError, TypeError, ValueError):
                continue
            tts = _clean_tts(u["uk"], str(item.get("tts") or "").strip())
            if tts:
                u["tts"] = tts
                done += 1
    return done


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


def fix_anglicisms(llm: LLM, units: list[dict], *, glossary: list[dict] | None = None,
                   domain: str | None = None, log=print) -> int:
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
    log(f"   • {'перевіряю англіцизми' if domain else 'прибираю англіцизми'}: {len(todo)} реплік")
    system = anglicism_system(domain)
    changed = 0
    by_id = {u["id"]: u for u, _ in todo}
    for i in range(0, len(todo), CHUNK):
        batch = todo[i : i + CHUNK]
        payload = {"lines": [{"id": u["id"], "src": u["text"], "uk": u["uk"], "suspect": s, "max_syl": u["max_syl"]}
                             for u, s in batch]}
        try:
            data = _ask(llm, system, json.dumps(payload, ensure_ascii=False),
                        max_tokens=200 + 160 * len(batch))
        except RuntimeError:
            continue
        for item in data.get("lines", []):
            try:
                u = by_id[int(item.get("id"))]
            except (KeyError, TypeError, ValueError):
                continue
            uk = str(item.get("uk", "")).strip()
            if uk == u["uk"] and not u.get("tts"):  # kept as is (e.g. a quoted UI label): still take the spelling
                tts = _clean_tts(uk, str(item.get("tts") or "").strip())
                if tts:
                    u["tts"] = tts
            elif uk and uk != u["uk"]:
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


def _fallback_marks(ask: list[dict], marks: dict[int, dict[int, int]], *, broad: bool) -> dict[int, dict[int, int]]:
    """Homographs the two votes did not settle: for StyleTTS2 (`broad`) take the dictionary's first
    variant — a deterministic choice beats the engine's guess, and the mark costs nothing there."""
    out = {ui: dict(m) for ui, m in marks.items()}
    if broad:
        for a in ask:
            ui, n = map(int, a["id"].split(":"))
            out.setdefault(ui, {}).setdefault(n, a["_options"][0])
    return out


def resolve_homographs(llm: LLM, units: list[dict], *, broad: bool = False, log=print) -> dict:
    """Mark the stress of homographs in each line's `tts` text with an acute accent.

    Grammatical homographs are resolved by ukrainian-word-stress; for semantic ones the LLM
    picks one of the dictionary's variants (a constrained choice, which LLMs do reliably,
    unlike free-form stress placement).
    """
    from .stress import HomographFinder, apply_marks, variant_caps

    try:
        finder = HomographFinder(broad=broad)
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
    marks = _fallback_marks(ask, marks, broad=broad)
    for ui, m in marks.items():
        units[ui]["tts"] = apply_marks(speech_text(units[ui]), m)
    return {"marked": sum(map(len, marks.values())), "homographs": len(ask)}


CONTEXT_CHARS = 5000


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
               "lines": [{"id": u["id"], "src": u["text"], "max_syl": u["max_syl"], "max_words": u.get("max_words"),
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
                _translate_chunk(llm, system, context, units, part, by_id, fixed_gender, edge)
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


def _neighbour_payload(items: list[dict], units: list[dict], context: dict | None, extra) -> dict:
    pos = {id(u): i for i, u in enumerate(units)}

    def neighbour(u: dict, step: int) -> str:
        i = pos.get(id(u), -1) + step
        return str(units[i].get("uk") or units[i]["text"]) if 0 <= i < len(units) else ""

    return {**{k: v for k, v in (context or {}).items() if k in ("summary", "style", "address")},
            "lines": [{"id": u["id"], "src": u["text"], "uk": u["uk"],
                       "max_words": u.get("max_words") or max(2, round(u["max_syl"] / SYL_PER_WORD)),
                       "previous": neighbour(u, -1), "next": neighbour(u, 1), **extra(u)} for u in items]}


def _first_word(text: str) -> str:
    m = re.search(r"[^\W\d_]+(?:['’-][^\W\d_]+)*", text or "")
    return m.group(0).lower() if m else ""


def repeated_starts(units: list[dict]) -> list[int]:
    """Indices of lines that begin with the same word as the line before them (the ear notices)."""
    out = []
    for i in range(1, len(units)):
        a, b = _first_word(units[i - 1].get("uk", "")), _first_word(units[i].get("uk", ""))
        if a and a == b:
            out.append(i)
    return out


VARY_SYSTEM = """You polish Ukrainian dubbing lines. Each line begins with the same word as the line spoken just before it ("previous"), which sounds mechanical. Rewrite "uk" so that it begins differently — same meaning, same spoken style and form of address, about the same length (±10 % of the words), keeping every name, number and quoted label. Do not change "previous".
Follow the same "tts" rule: if the new "uk" has digits, symbols, Latin letters or abbreviations, add "tts" with everything spelled out as pronounced; otherwise omit it.
Return ONLY JSON: {"lines": [{"id": <int>, "uk": "...", "tts": "..."}]}"""


def _vary_starts(llm: LLM, items: list[dict], *, units: list[dict], context: dict | None = None) -> None:
    payload = _neighbour_payload(items, units, context, lambda u: {})
    try:
        data = _ask(llm, VARY_SYSTEM, json.dumps(payload, ensure_ascii=False), max_tokens=400 + 260 * len(items))
    except RuntimeError:
        return
    by_id = {u["id"]: u for u in items}
    pos = {id(u): i for i, u in enumerate(units)}
    for item in data.get("lines", []):
        try:
            u = by_id[int(item.get("id"))]
        except (KeyError, TypeError, ValueError):
            continue
        uk_new = str(item.get("uk", "")).strip()
        tts_new = _clean_tts(uk_new, str(item.get("tts") or "").strip())
        prev = units[pos[id(u)] - 1] if pos.get(id(u), 0) > 0 else None
        new_syl = syllables(to_speech_text(tts_new or uk_new)) if uk_new else 0
        if (uk_new and prev is not None and _first_word(uk_new) != _first_word(prev.get("uk", ""))
                and 0.85 * _syl(u) <= new_syl <= 1.1 * _syl(u) and not lost_facts(u["uk"], uk_new)):
            u["uk"] = uk_new
            if tts_new:
                u["tts"] = tts_new
            else:
                u.pop("tts", None)


def _expand(llm: LLM, items: list[dict], *, units: list[dict], context: dict | None = None) -> list[tuple[dict, tuple]]:
    """Fuller rewrites of lines far under their time. Returns the lines that came back over the
    budget (with their previous version), for one shortening pass and a possible revert."""
    payload = _neighbour_payload(items, units, context, lambda u: {})
    try:
        data = _ask(llm, EXPAND_SYSTEM, json.dumps(payload, ensure_ascii=False), max_tokens=400 + 300 * len(items))
    except RuntimeError:
        return []
    by_id = {u["id"]: u for u in items}
    over: list[tuple[dict, tuple]] = []
    for item in data.get("lines", []):
        try:
            u = by_id[int(item.get("id"))]
        except (KeyError, TypeError, ValueError):
            continue
        uk_new = str(item.get("uk", "")).strip()
        tts_new = _clean_tts(uk_new, str(item.get("tts") or "").strip())
        new_syl = syllables(to_speech_text(tts_new or uk_new)) if uk_new else 0
        outcome = expand_outcome(old_syl=_syl(u), new_syl=new_syl, max_syl=u["max_syl"]) if uk_new else "reject"
        if outcome == "reject" or lost_facts(u["uk"], uk_new):
            continue
        if outcome == "shorten":
            over.append((u, (u["uk"], u.get("tts"))))
        u["uk"] = uk_new
        if tts_new:
            u["tts"] = tts_new
        else:
            u.pop("tts", None)
    return over


def _shorten(llm: LLM, items: list[dict], *, units: list[dict] | None = None, context: dict | None = None,
             attempt: int = 1) -> None:
    units = units or items
    pos = {id(u): i for i, u in enumerate(units)}

    def neighbour(u: dict, step: int) -> str:
        i = pos.get(id(u), -1) + step
        return str(units[i].get("uk") or units[i]["text"]) if 0 <= i < len(units) else ""

    payload = {**{k: v for k, v in (context or {}).items() if k in ("summary", "style", "address")},
               "lines": [{"id": u["id"], "src": u["text"], "uk": u["uk"], "over": _syl(u) - u["max_syl"],
                          "max_words": u.get("max_words") or max(2, round(u["max_syl"] / SYL_PER_WORD)),
                          "previous": neighbour(u, -1), "next": neighbour(u, 1)} for u in items]}
    system = SHORTEN_SYSTEM.replace("{attempt}", SHORTEN_RETRY if attempt > 1 else "")
    try:
        data = _ask(llm, system, json.dumps(payload, ensure_ascii=False), max_tokens=400 + 260 * len(items))
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
        new_syl = syllables(to_speech_text(cand["tts"] or cand["uk"])) if cand["uk"] else 0
        if cand["uk"] and shorten_ok(old_syl=_syl(u), new_syl=new_syl, max_syl=u["max_syl"]) and not lost_facts(u["uk"], cand["uk"]):
            u["uk"] = cand["uk"]
            if cand["tts"]:
                u["tts"] = cand["tts"]
            else:
                u.pop("tts", None)
