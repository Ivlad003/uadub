"""Pause between translation and voicing: a human (or Claude Code / opencode) edits the script.

`review.md` holds one block per line:

    ## 0007  00:12.3–00:15.8 · слот 3.5 с · складів 17/19
    EN:  So let's go back to the base model.
    UK:  Повернімося до базової моделі.
    TTS:

`UK` is the subtitle/translation, `TTS` (optional) is how it should be read aloud, with `+`
before a stressed vowel. With StyleTTS2 a read-only `НАГОЛОСИ:` line shows the line exactly as it
will be stressed. Edits are merged back into units.json before voicing; a per-video `stress.txt`
next to it extends the stress dictionary. Header and AGENTS.md instructions depend on the engine.
"""

from __future__ import annotations

import hashlib
import json
import re
import shlex
import subprocess
from pathlib import Path

from .segments import slot
from .stress import ACUTE, VOWELS
from .textnorm import syllables, to_speech_text

REVIEW = "review.md"
SNAPSHOT = ".review-export.json"
AGENTS = "AGENTS.md"
LOCAL_STRESS = "stress.txt"

# ---------------------------------------------------------------------------
# Texts. Stress handling differs by engine:
#   st   (StyleTTS2)  — «+» is a real stress mark, safe and precise → encourage fixing stress;
#   omni (OmniVoice)  — «+» turns the word into English phonemes (accent) → last resort only;
#   ukr  (ukrainian-tts) — «+» works natively.
# ---------------------------------------------------------------------------
_STRESS_HELP = {
    "st": ("- `НАГОЛОСИ:` (лише для читання) — як синтезатор прочитає рядок; `+` стоїть перед наголошеною\n"
           "  голосною. Побачили неправильний наголос — скопіюйте цей рядок у `TTS:` і пересуньте `+`.\n"
           "  Слово, що повторюється з помилкою, допишіть у `stress.txt` (`кілом+етр`).\n"
           "  Сумніваєтеся — `uadub --stress-lookup слово` покаже всі варіанти зі словника."),
    "ukr": ("- Наголос виправляється `+` перед наголошеною голосною в `TTS:` (`зам+ок`) або словом\n"
            "  у `stress.txt`. Сумніваєтеся — `uadub --stress-lookup слово` покаже варіанти зі словника."),
    "omni": ("- Наголос через `+` і словник `stress.txt` — крайній засіб: цей синтезатор читає позначене\n"
             "  слово англійськими фонемами, тобто з легким акцентом. Лише для слів, які ви почули неправильно.\n"
             "  Для точних наголосів скористайтеся голосом `--voice st`."),
}


def review_domain(work: Path) -> str | None:
    """Field of the video when the specialist mode (--domain) is on, else None."""
    try:
        from .config import Options

        opt = Options.load(work / "options.json")
    except Exception:
        return None
    if not opt.domain:
        return None
    if opt.domain != "auto":
        return opt.domain
    try:
        found = json.loads((work / "brief.json").read_text(encoding="utf-8")).get("domain")
    except Exception:
        found = None
    return found or "сфера відео"


def _domain_terms(work: Path) -> set[str]:
    """Lower-case words of the glossary renderings: in the specialist mode they are wanted jargon."""
    try:
        info = json.loads((work / "brief.json").read_text(encoding="utf-8"))
    except Exception:
        return set()
    words: set[str] = set()
    for g in (info.get("brief") or {}).get("glossary") or []:
        if isinstance(g, dict):
            words.update(w.lower() for w in re.findall(r"[А-Яа-яЄєІіЇїҐґ'’]+", str(g.get("uk", ""))))
    return words


def terms_note(domain: str | None) -> str:
    """One-line terminology rule for the review header and the agent prompt."""
    if domain:
        return (f"фаховий переклад для сфери «{domain}»: усталені терміни фахівців, зокрема англіцизми "
                "(фреймворк, деплой), лишаються; кнопки й меню — як на екрані, латиницею в лапках; "
                "без сленгу на кшталт «юзати», «дефолтний»")
    return "без англіцизмів"


def header(engine: str, domain: str | None = None) -> str:
    return f"""# uadub — сценарій дубляжу на перевірку

Редагуйте лише рядки `UK:` і `TTS:`. Заголовки `## …`, рядок оригіналу (`EN:`, `KO:` …) і `НАГОЛОСИ:` не змінюйте.

- `UK:` — переклад, він же субтитри. Перекладаємо за змістом, а не дослівно; ідіоми й крилаті
  вирази — українськими відповідниками; {terms_note(domain)}.
- `TTS:` — як читати вголос, якщо це відрізняється від UK: числа й дати словами, назви кирилицею
  так, як їх вимовляють українською. Порожній рядок означає «читати як UK».
- `складів X/Y`: Y — скільки складів уміщається в таймінг. Трохи більше можна: озвучка прискориться.
{_STRESS_HELP.get(engine, _STRESS_HELP["omni"])}

Після правок поверніться в термінал і натисніть Enter (або запустіть ту саму команду uadub ще раз).
"""


_STRESS_HOWTO = """### Як позначати наголос
- `+` ставиться ПЕРЕД наголошеною голосною (а е є и і ї о у ю я): `зам+ок`, `кілом+етр`. Один `+` на слово.
  Слова з однією голосною не позначай.
- Наголоси — лише в рядку `TTS:` (і в `stress.txt`), ніколи в `UK:`: `UK:` іде в субтитри. Якщо `TTS:`
  порожній — скопіюй туди текст з `UK:` і додай `+`.
- У `stress.txt` один рядок — одне слово, у потрібній формі: `кілом+етр` (або `кіломе́тр`). Словник
  діє на всі репліки з цим словом у цій формі й має перевагу над вбудованим словником.

### Перевіряй словником, а не з пам'яті
- Наголос в українській вільний і рухомий, загальних правил для нього немає — норму задає орфоепічний
  словник. Для будь-якого сумнівного слова виконай у цій папці:
  `uadub --stress-lookup слово [слово …]` — покаже всі варіанти наголосу зі словника (2,9 млн словоформ
  «Словників України») з граматичними ознаками й запис із твого `stress.txt`.
- Один варіант → так і має бути. Кілька варіантів → обери за змістом і граматикою речення
  (за́мок — споруда, замо́к — на дверях; ру́ки — наз. мн., руки́ — род. одн.). Якщо команда пише
  «обидва наголоси допустимі» (по́милка / поми́лка, ма́буть / мабу́ть) — не виправляй.
  «Немає в словнику» → постав наголос сам: це нові терміни, назви, рідкісні форми.

### Типові пастки (перевір їх у першу чергу)
- Доконані дієслова з префіксом ви-: наголос на ви- — ви́конати, ви́користати, ви́брати, ви́рішити.
- Числівники: одина́дцять, чотирна́дцять, сімдеся́т, вісімдеся́т.
- Запозичення: кіломе́тр, сантиме́тр, катало́г, діало́г.
- Рухомий наголос в іменниках: рука́ — ру́ки, нога́ — но́ги, вода́ — во́ди (наз. мн.).
- Слова, які часто наголошують неправильно: ви́падок, на́голос, одноча́сно, ненави́сть,
  фарту́х, кропива́, черго́вий, озна́ка, запита́ння.
- Слова з великої літери на початку речення: словник інколи сприймає їх як власні назви.
"""

_AGENT_STRESS = {
    "st": """## Наголоси (синтезатор StyleTTS2 читає `+` точно, тож їх варто виправляти)
- Рядок `НАГОЛОСИ:` показує, як буде прочитано текст: `+` стоїть ПЕРЕД наголошеною голосною.
  Його згенеровано автоматично зі словника; він не змінюється, правки йдуть у `TTS:`.
- Перевір наголоси в кожному рядку `НАГОЛОСИ:` за нормами сучасної української літературної мови.
- Якщо в рядку є помилка: скопіюй увесь рядок `НАГОЛОСИ:` у `TTS:` і пересунь `+` у неправильних словах
  (або постав, де його немає). В інших словах `+` можна прибрати. Нічого іншого в тексті не змінюй.
- Якщо те саме слово в тій самій формі помилково наголошено кілька разів — додай його в `stress.txt`
  замість правок у кожному рядку.
- Не чіпай рядки, де наголоси правильні.

""" + _STRESS_HOWTO,
    "ukr": """## Наголоси
- Синтезатор розуміє `+` перед наголошеною голосною. Виправляй наголос у `TTS:` лише там, де він
  очевидно неправильний (омографи, числівники, рідкісні слова); повторювані слова — у `stress.txt`.

""" + _STRESS_HOWTO,
    "omni": """## Наголоси — НЕ виправляй
- Цей синтезатор (OmniVoice) читає позначене `+` слово англійськими фонемами, з акцентом.
  Тому не став `+` і не змінюй `stress.txt` — це робить лише людина, яка почула помилку.
""",
}


_PLAIN_TERMS = """- Без англіцизмів: назви кнопок, меню й звичайні терміни — українською («Use this model» →
  «Використати цю модель», browse → переглянути, quantization → квантування, checkbox → прапорець).
  Латиницею лишаються тільки власні назви (LM Studio, Hugging Face, GGUF). Англійських слів,
  записаних кирилицею («юз зіс модел», «брауз», «квантайзейшн», «рантайм»), бути не повинно."""

_DOMAIN_TERMS = """- Фаховий переклад: глядачі — фахівці сфери «{domain}». Терміни — так, як їх кажуть українські
  фахівці, зокрема усталені англіцизми (у розробці ПЗ: фреймворк, деплой, коміт, пул-реквест, бекенд);
  не заміняй їх штучними відповідниками, яких ніхто не вживає. Якщо усталений термін питомо
  український — він (база даних, квантування). Назви кнопок, меню й налаштувань, які глядач бачить
  на екрані, — як в оригіналі, латиницею в лапках (натисніть «Use this model»). Сленгу й випадкових
  транслітерацій звичайних слів («юзати», «дефолтний», «юз зіс модел») бути не повинно."""

_PLAIN_UNKNOWN = "англіцизми (заміни в UK українським словом) або назви (перевір, що TTS вимовляє їх правильно)."
_DOMAIN_UNKNOWN = ("фахові терміни (усталені — лишай), сленг (заміни) або назви (перевір, що TTS вимовляє їх "
                   "правильно).")


def agent_task(engine: str, src_srt: str = "en.srt", domain: str | None = None) -> str:
    terms = _DOMAIN_TERMS.replace("{domain}", domain) if domain else _PLAIN_TERMS
    unknown = _DOMAIN_UNKNOWN if domain else _PLAIN_UNKNOWN
    return f"""# Завдання: перевірити сценарій українського дубляжу

У цій папці `review.md` — сценарій дубляжу іноземного відео українською (одна репліка — один блок
`## NNNN`), а `stress.txt` — словник наголосів для цього відео. Редагуй лише ці два файли, на місці.
Ти не чуєш звук, тож працюєш із текстом: переклад, вимова, наголоси.

## Що не можна змінювати
- Заголовки `## NNNN …` (номери, таймінги, стать мовця, склади), рядки оригіналу (`EN:`, `KO:` …)
  і рядки `НАГОЛОСИ:`. Не додавай і не видаляй блоки, не змінюй їхній порядок.

## Спершу зрозумій зміст
- Перш ніж правити, прочитай усі рядки оригіналу (`EN:`/`KO:` …) підряд, від початку до кінця,
  ніби це суцільний текст: про що відео, яка в автора логіка, де жарти. Повний текст оригіналу
  з таймінгами є також у файлі `{src_srt}`.
- Оригінал — це автоматичне розпізнавання мови: в ньому трапляються неправильно почуті слова
  й обірвані речення (наприклад, «Quen 3.6» замість назви моделі Qwen 3.6). Перекладай те, що
  людина насправді сказала, а не помилку розпізнавання.

## UK — переклад і субтитри
- Перекладай за змістом, а не дослівно: так, як це сказав би носій української. Можна
  перебудовувати речення, змінювати порядок слів, прибирати чужі для української конструкції
  («є такий спосіб, який…» → «можна так…»). Глядач має зрозуміти те саме й відчути те саме.
- Крилаті вирази, ідіоми, приказки й образні звороти — заміни на усталені українські відповідники
  з тим самим змістом і стилем, а не перекладай слово в слово: a piece of cake → простіше простого;
  kill two birds with one stone → убити двох зайців одним пострілом; break the ice → розтопити кригу;
  beat around the bush → ходити околясом; once in a blue moon → раз на сто років; costs an arm and
  a leg → коштує шалені гроші; hit the nail on the head → влучити в саму точку; the ball is in
  your court → тепер слово за тобою. Так само зі сполучними зворотами: when it comes to → коли
  йдеться про; at the end of the day → зрештою. Якщо українського відповідника немає — передай
  зміст природно. Жарти й гру слів адаптуй так, щоб вони працювали українською.
- Слова-паразити усного мовлення (kind of, you know, like, basically) зазвичай пропускай.
- Кожна репліка — граматично правильне, завершене речення; уривків сталих зворотів не лишай
  («Коли справа стає завантаження» ✗ → «Коли йдеться про завантаження» ✓).
- Виправ помилки змісту, пропущені думки, русизми, суржик, кальки й неприродні звороти.
  Тримай одну термінологію і одне звертання (ти/ви) в усьому тексті.
{terms}
- Рід у минулому часі (зробив/зробила) — за статтею мовця з заголовка («чол. голос» / «жін. голос»)
  і змістом; рід співрозмовника (ти готовий/готова) — зі змісту.
- Корейські/японські імена — за системою Концевича/Коваленка, однаково в усьому тексті.
- Довжина: у заголовку `складів X/Y`, Y — ліміт складів (голосних) для таймінгу. Не перевищуй Y
  більш ніж на 10 %; рядки з позначкою `⚠ задовго` скороти без втрати змісту.

## TTS — як читати вголос
- Заповнюй, лише коли читання відрізняється від UK: числа, дати, одиниці, символи — словами в
  правильному відмінку («о 8:30» → «о восьмій тридцять»); власні назви й абревіатури — кирилицею
  так, як їх вимовляють українською (GitHub → ґітхаб, LM Studio → ел-ем студіо, API → ей-пі-ай).
- Порожній TTS означає «читати як UK». Якщо змінюєш UK, а TTS був заповнений — онови й TTS.
- Список «Немає в словнику» наприкінці файлу — слова, яких не знає словник: зазвичай це
  {unknown}

{_AGENT_STRESS.get(engine, _AGENT_STRESS["omni"])}
## Наприкінці
Коротко перелічи, що змінено (номери блоків і суть правок).
"""


def agent_prompt(engine: str, domain: str | None = None) -> str:
    stress = {"st": "та наголоси (рядки НАГОЛОСИ:, правки через TTS: і stress.txt)",
              "ukr": "і явні помилки наголосу",
              "omni": "(наголоси не чіпай)"}.get(engine, "(наголоси не чіпай)")
    return ("Прочитай AGENTS.md у поточній папці й виконай його. Спершу прочитай увесь оригінал і зрозумій "
            "логіку відео, потім перевір і виправ review.md: переклад за змістом, а не дослівно; ідіоми й "
            f"крилаті вирази — українськими відповідниками; {terms_note(domain)}; помилки розпізнавання в оригіналі; "
            f"вимова чисел і назв {stress}. Змінюй лише review.md і stress.txt. "
            "Наприкінці коротко перелічи зміни.")


LOOKUP = "uadub --stress-lookup"  # the only command an agent may run on its own (read-only dictionary)

# harness → (command with {prompt}, how to pass a model). The agent may only edit files in the work folder.
AGENT_COMMANDS = {
    "claude": (["claude", "-p", "{prompt}", "--permission-mode", "acceptEdits", "{model}",
                "--allowedTools", f"Bash({LOOKUP}:*)"], ["--model", "{m}"]),
    "opencode": (["opencode", "run", "{model}", "{prompt}"], ["-m", "{m}"]),  # model = provider/model
    "codex": (["codex", "exec", "--skip-git-repo-check", "-s", "workspace-write", "{model}", "{prompt}"],
              ["-m", "{m}"]),
    "gemini": (["gemini", "--skip-trust", "--approval-mode", "auto_edit", "--allowed-tools", f"run_shell_command({LOOKUP})",
                "{model}", "-p", "{prompt}"], ["-m", "{m}"]),
}


def agent_command(spec: str, prompt: str) -> list[str]:
    """«claude», «claude:sonnet», «opencode:anthropic/claude-sonnet-4-5», «codex:gpt-5», «gemini:gemini-2.5-pro»,
    or any custom command line with {prompt} (otherwise the prompt is appended)."""
    harness, _, model = spec.partition(":")
    if harness in AGENT_COMMANDS and " " not in harness:
        template, model_args = AGENT_COMMANDS[harness]
        cmd: list[str] = []
        for part in template:
            if part == "{model}":
                cmd += [a.replace("{m}", model) for a in model_args] if model else []
            else:
                cmd.append(part.replace("{prompt}", prompt))
        return cmd
    template = shlex.split(spec)
    if "{prompt}" not in " ".join(template):
        template = template + ["{prompt}"]
    return [part.replace("{prompt}", prompt) for part in template]


def _ts(t: float) -> str:
    m, s = divmod(max(0.0, t), 60)
    return f"{int(m):02d}:{s:04.1f}"


def to_plus(text: str) -> str:
    """'замо́к' → 'зам+ок' (easier to type and read than a combining accent)."""
    out = []
    for ch in text:
        if ch in (ACUTE, "´") and out and out[-1].lower() in VOWELS:
            out.insert(len(out) - 1, "+")
        else:
            out.append(ch)
    return "".join(out)


def _unknown_words(units: list[dict]) -> list[str]:
    """Words the stress dictionary does not know — the likeliest mispronunciations."""
    try:
        import importlib.resources as res

        import marisa_trie

        trie = marisa_trie.BytesTrie()
        trie.load(str(res.files("ukrainian_word_stress").joinpath("data/stress.trie")))
    except Exception:
        return []
    seen: set[str] = set()
    for u in units:
        for w in re.findall(r"[А-Яа-яЄєІіЇїҐґ'’]+", (u.get("tts") or u.get("uk") or "").replace(ACUTE, "")):
            if sum(c.lower() in VOWELS for c in w) < 2:
                continue
            if not any(f in trie for f in (w, w.lower(), w.title())):
                seen.add(w.lower())
    return sorted(seen)


def _engine(work: Path) -> str:
    try:
        from .config import Options

        return Options.load(work / "options.json").engine
    except Exception:
        return "omni"


def _stress_preview(work: Path, engine: str, *, any_engine: bool = False, acute: bool = False):
    """For StyleTTS2: a function giving each line exactly as it will be stressed («+» notation).

    any_engine: also for other voices (dictionary stress, for the --text stress file);
    acute: keep the combining acute (на́голос) instead of «+».
    """
    if engine != "st" and not any_engine:
        return None
    try:
        import logging

        logging.getLogger("stanza").setLevel(logging.ERROR)
        from ukrainian_word_stress import Stressifier, StressSymbol

        from .config import Options
        from .stress import StressFixer, default_dict_paths
        from .translate import speech_text
        from .tts import prepare_st_text

        opt = Options.load(work / "options.json")
        fixer = StressFixer("off" if opt.stress == "off" else "dict",
                            default_dict_paths(opt.stress_dict, work), target="acute")
        stressify = Stressifier(stress_symbol=StressSymbol.CombiningAcuteAccent)
    except Exception:
        return None

    def preview(u: dict) -> str:
        marked = prepare_st_text(fixer.apply(speech_text(u)), stressify)
        return marked if acute else to_plus(marked)

    return preview


def refresh_agents(work: Path) -> None:
    """Rewrite AGENTS.md (not user-edited) without touching review.md."""
    try:
        from .config import Options

        opt = Options.load(work / "options.json")
        (work / AGENTS).write_text(agent_task(_engine(work), f"{opt.text_lang}.srt", review_domain(work)),
                                   encoding="utf-8")
    except Exception:
        pass


def export_review(work: Path) -> Path:
    units = json.loads((work / "units.json").read_text(encoding="utf-8"))
    try:
        opts = json.loads((work / "options.json").read_text(encoding="utf-8"))
        src = ((opts.get("subs_lang") or opts.get("source_lang")) if opts.get("subs") else opts.get("source_lang")) or "en"
    except Exception:
        src = "en"
    label = f"{src.upper()}:".ljust(4)
    engine = _engine(work)
    preview = _stress_preview(work, engine)
    path = work / REVIEW
    if path.exists():
        path.replace(work / "review.old.md")  # translation changed → keep the previous edits around
    domain = review_domain(work)
    out = [header(engine, domain)]
    snapshot = {}
    for u in units:
        uk = u.get("uk", "")
        tts = to_plus(u.get("tts", "")) if u.get("tts") else ""
        if tts == uk:
            tts = ""  # identical to UK → leave the line empty, less noise to review
        budget = u.get("max_syl") or 0
        syl = syllables(to_speech_text(u.get("tts") or uk))
        warn = "  ⚠ задовго" if budget and syl > budget * 1.1 else ""
        who = {"male": " · чол. голос", "female": " · жін. голос"}.get(u.get("gender") or "", "")
        out.append(f"## {u['id']:04d}  {_ts(u['start'])}–{_ts(u['end'])}{who} · слот {slot(u):.1f} с · складів {syl}/{budget}{warn}")
        out.append(f"{label} {u['text']}")
        out.append(f"UK:  {uk}")
        out.append(f"TTS: {tts}")
        if preview:
            out.append(f"НАГОЛОСИ: {preview(u)}")
        out.append("")
        snapshot[str(u["id"])] = {"uk": uk, "tts": tts}
    unknown = _unknown_words(units)
    if domain:  # glossary jargon is wanted in the specialist mode
        terms = _domain_terms(work)
        unknown = [w for w in unknown if w not in terms]
    if unknown:
        what = "сленг, неусталені терміни чи помилки" if domain else "можливі англіцизми чи помилки"
        out.append(f"---\n\n**Немає в словнику ({what}, перевірте):** " + ", ".join(unknown) + "\n")
    path.write_text("\n".join(out), encoding="utf-8")
    (work / SNAPSHOT).write_text(json.dumps(snapshot, ensure_ascii=False), encoding="utf-8")
    (work / AGENTS).write_text(agent_task(engine, f"{src}.srt", domain), encoding="utf-8")
    local = work / LOCAL_STRESS
    if not local.exists():
        local.write_text("# Наголоси для цього відео: слово з + перед наголошеною голосною, напр. зам+ок\n",
                         encoding="utf-8")
    return path


_BLOCK = re.compile(r"^##\s+(\d+)\b", re.M)


def parse_review(text: str) -> dict[int, dict]:
    items: dict[int, dict] = {}
    marks = list(_BLOCK.finditer(text))
    for k, m in enumerate(marks):
        body = text[m.end() : marks[k + 1].start() if k + 1 < len(marks) else len(text)]
        item = {}
        for line in body.splitlines():
            for key in ("UK", "TTS"):
                if line.startswith(f"{key}:"):
                    item[key.lower()] = line[len(key) + 1 :].strip()
        if "uk" in item:
            items[int(m.group(1))] = item
    return items


def import_review(work: Path) -> int:
    """Merge review.md edits into units.json. Returns the number of changed lines."""
    path = work / REVIEW
    if not path.exists():
        return 0
    edits = parse_review(path.read_text(encoding="utf-8"))
    snap = json.loads((work / SNAPSHOT).read_text(encoding="utf-8")) if (work / SNAPSHOT).exists() else {}
    units = json.loads((work / "units.json").read_text(encoding="utf-8"))
    changed = 0
    for u in units:
        e = edits.get(u["id"])
        if not e:
            continue
        before = snap.get(str(u["id"]), {})
        uk, tts = e["uk"], e.get("tts", "")
        if uk != before.get("uk", u.get("uk", "")) and tts == before.get("tts", "") and tts:
            tts = ""  # UK was rewritten but the old TTS left as is → read the new UK
        new_tts = tts or None
        old_tts = to_plus(u["tts"]) if u.get("tts") else None
        if uk != u.get("uk") or new_tts != old_tts:
            changed += 1
        u["uk"] = uk
        if new_tts:
            u["tts"] = new_tts
        else:
            u.pop("tts", None)
    (work / "units.json").write_text(json.dumps(units, ensure_ascii=False, indent=2), encoding="utf-8")
    from .srt import make_cues, write_srt

    write_srt(make_cues(units, "uk"), work / "uk.srt")
    return changed


def review_hash(work: Path) -> str:
    h = hashlib.sha1()
    for name in (REVIEW, LOCAL_STRESS):
        p = work / name
        if p.exists():
            h.update(p.read_bytes())
    return h.hexdigest()[:12]


def run_agent(spec: str, work: Path) -> None:
    prompt = agent_prompt(_engine(work), review_domain(work))
    cmd = agent_command(spec, prompt)
    model = spec.partition(":")[2] if spec.partition(":")[0] in AGENT_COMMANDS else ""
    print(f"   • перевірка сценарію агентом: {cmd[0]}{f' (модель {model})' if model else ''} — у папці {work}",
          flush=True)
    try:
        subprocess.run(cmd, cwd=work, check=False)
    except FileNotFoundError:
        print(f"   ! не знайдено команду «{cmd[0]}» — пропускаю автоматичну перевірку", flush=True)
