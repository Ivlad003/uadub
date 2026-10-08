"""Command line entry point: `uadub video.mp4` → video.uk.mp4 (+ .srt)."""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path

from .config import (DEFAULT_ASR, DEFAULT_LLM, DEFAULT_SEP_MODEL, FAST_SEP_MODEL, LANGS, OMNI_MODEL, ST_MODEL,
                     ST_SPACE, ST_VOICES, STAGES, UKR_VOICES, WHISPER_ASR, Options, find_st_voice)

HEAVY = {"separate", "asr", "translate", "tts"}  # each gets a fresh process → memory fully freed
STAGE_TITLES = {
    "extract": "Витягую аудіо",
    "separate": "Відокремлюю голос від фону",
    "asr": "Розпізнаю мову оригіналу",
    "translate": "Перекладаю українською (локальна LLM)",
    "tts": "Озвучую українською",
    "mix": "Зводжу звук",
    "mux": "Збираю відео",
}

EMOTION_DEFAULT = 0.6

VOICE_HELP = (
    "Два типи голосів. Точні наголоси, без клонування (StyleTTS2): st (чоловічий), st:<ім'я> (31 голос, "
    "див. --list-voices), duo:st (чоловік/жінка за висотою голосу мовця). "
    "Клонування, але наголоси бувають неточні (OmniVoice): "
    "clone — голос оригінального спікера; "
    "clone:/шлях/до/зразка.wav — клонувати голос із вашого зразка (наприклад, українського диктора); "
    "omni:'female, young adult' — згенерувати голос за описом (OmniVoice); "
    "duo — два голоси OmniVoice, чоловічий/жіночий за висотою голосу мовця (для фільмів і дорам); "
    "duo:dmytro,tetiana — те саме швидкими готовими голосами. "
    "Швидкі прості голоси (роботизовані): dmytro (за замовч.), oleksa, mykyta, tetiana, lada"
)


def _env_defaults() -> None:
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")


def _child(stage: str, workdir: str) -> None:
    _env_defaults()
    from .stages import STAGE_FUNCS

    STAGE_FUNCS[stage](Options.load(Path(workdir) / "options.json"))


def _run_stage(stage: str, opt: Options) -> None:
    if stage not in HEAVY:
        from .stages import STAGE_FUNCS

        try:
            STAGE_FUNCS[stage](opt)
        except RuntimeError as e:  # ffmpeg failures: the message already holds the useful part
            raise SystemExit(f"\nЕтап «{stage}» завершився з помилкою:\n{e}") from None
        return
    proc = mp.get_context("spawn").Process(target=_child, args=(stage, str(opt.work)))
    proc.start()
    try:
        proc.join()
    except KeyboardInterrupt:
        proc.terminate()
        proc.join()
        raise
    if proc.exitcode != 0:
        raise SystemExit(f"\nЕтап «{stage}» завершився з помилкою (код {proc.exitcode}). Подробиці вище.\n"
                         f"Після виправлення просто запустіть ту саму команду — готові етапи буде пропущено.")


def _save_state(path: Path, state: dict) -> None:
    path.write_text(json.dumps(state, ensure_ascii=False, indent=2))


def _terms_hint(domain: str | None) -> str:
    if domain:
        return (f"Фаховий переклад («{domain}»): терміни як у фахівців, усталені англіцизми лишаються\n"
                "      (фреймворк, деплой); кнопки й меню — як на екрані («Use this model»). Без «юзати», «дефолтний».")
    return ("Без англіцизмів: кнопки, меню й терміни — українською («Use this model» → «Використати цю модель»);\n"
            "      латиницею лише власні назви (LM Studio, Hugging Face). Ніяких «юз зіс модел», «брауз», «рантайм».")


_STRESS_RULE = {
    "st": "+ перед наголошеною голосною (зам+ок), лише в TTS: і stress.txt, один на слово; рядок НАГОЛОСИ:\n"
          "      показує, як буде прочитано. Перевірити слово: uadub --stress-lookup замок",
    "ukr": "+ перед наголошеною голосною (зам+ок), лише в TTS: і stress.txt — синтезатор його розуміє.\n"
           "      Перевірити слово: uadub --stress-lookup замок",
    "omni": "у цього голосу + дає англійський акцент — лише для слів, які ви почули неправильно "
            "(для точних наголосів: --voice st)",
}


class UserStop(SystemExit):
    """The user deliberately stopped the run (review pause): a long run must not go on to the next part."""


def _review_pause(opt: Options, state: dict, state_path: Path, review: bool, review_with: str | None,
                  force_agent: bool = False) -> None:
    import shlex

    from .review import AGENTS, LOCAL_STRESS, REVIEW, agent_prompt, export_review, review_domain, run_agent
    from .stress import DEFAULT_DICT
    from .term import copy_to_clipboard, link, open_in_editor, shell_cd

    w = opt.work
    if not (w / REVIEW).exists() or state.get("review_export") != state.get("translate"):
        export_review(w)
        state["review_export"] = state.get("translate")
        _save_state(state_path, state)
    else:
        from .review import refresh_agents

        refresh_agents(w)  # instructions follow the current uadub version; review.md keeps your edits
    if review_with:
        done = state.get("agent_review") or {}
        if done.get("spec") == review_with and done.get("translate") == state.get("translate") and not force_agent:
            print(f"   • сценарій уже перевірено агентом ({review_with}) — пропускаю "
                  f"(ще раз: --redo review)", flush=True)
        else:
            run_agent(review_with, w)
            state["agent_review"] = {"spec": review_with, "translate": state.get("translate")}
            _save_state(state_path, state)
    if not review:
        print(f"   • сценарій: {link(w / REVIEW)}", flush=True)
        return
    domain = review_domain(w)
    prompt = shlex.quote(agent_prompt(opt.engine, domain))
    claude_cmd = f"{shell_cd(w)} && claude {prompt}"
    opencode_cmd = f"{shell_cd(w)} && opencode run {prompt}"
    src = opt.text_lang
    pronounce = Path.home() / ".config" / "uadub" / "pronounce.txt"
    print(f"""
⏸  Пауза перед озвученням: перевірте переклад і вимову, потім натисніть Enter.

ФАЙЛИ (клікніть або скопіюйте шлях)
   Сценарій для правок:       {link(w / REVIEW)}
   Оригінал із таймінгами:    {link(w / f"{src}.srt")}
   Наголоси цього відео:      {link(w / LOCAL_STRESS)}
   Наголоси для всіх відео:   {link(DEFAULT_DICT)}
   Вимова назв (усі відео):   {link(pronounce)}   (рядки «LM Studio = ел-ем студіо»)
   Інструкція для агента:     {link(w / AGENTS)}

ЯК ЧИТАТИ review.md (один блок — одна репліка)
   ## 0007  00:12.3–00:15.8 · чол. голос · слот 3.5 с · складів 17/19   ← не змінювати
   {src.upper()}:  оригінал (розпізнаний автоматично, можливі помилки)     ← не змінювати
   UK:  переклад, він же субтитри                                      ← правити
   TTS: як читати вголос; порожньо = читати як UK                      ← правити
   НАГОЛОСИ: як синтезатор поставить наголоси (лише голос st)          ← не змінювати

ПРАВИЛА
   1. Спершу прочитайте оригінал цілком: перекладаємо за змістом і логікою мовця, а не дослівно.
      Речення можна перебудовувати; помилки розпізнавання в оригіналі — перекладати те, що мали на увазі.
   2. Ідіоми й крилаті вирази — українськими відповідниками, не дослівно:
      a piece of cake → простіше простого · break the ice → розтопити кригу · once in a blue moon → раз на сто років
      · when it comes to → коли йдеться про. Слова-паразити (kind of, you know, like) зазвичай пропускаємо.
   3. {_terms_hint(domain)}
   4. Довжина: «складів X/Y» — Y уміщається в таймінг; до +10 % можна, «⚠ задовго» — скоротіть.
   5. Рід: «чол./жін. голос» у заголовку — хто говорить (я зробив / я зробила); співрозмовник — зі змісту.
   6. TTS: числа й дати словами в правильному відмінку («о 8:30» → «о восьмій тридцять»),
      назви кирилицею так, як їх вимовляють (GitHub → ґітхаб, API → ей-пі-ай).
   7. Наголос: {_STRESS_RULE.get(opt.engine, _STRESS_RULE["omni"])}.
      Слово, що повторюється з помилкою, — у stress.txt (один рядок: кілом+етр).

АГЕНТ (Claude Code або opencode в іншому вікні терміналу; правила для нього — в AGENTS.md)
   {claude_cmd}
   {opencode_cmd}
""", flush=True)
    if not sys.stdin.isatty():
        raise UserStop("Запустіть ту саму команду без --review, щоб озвучити з правками.")
    while True:
        answer = input("Enter — озвучити з правками · o — відкрити сценарій · s — наголоси відео · "
                       "c — скопіювати команду для Claude Code · q — вийти: ").strip().lower()
        if answer == "":
            return
        if answer in ("q", "й", "quit", "exit"):
            raise UserStop("Зупинено. Щоб озвучити з правками, запустіть ту саму команду без --review.")
        if answer in ("o", "щ"):
            open_in_editor(w / REVIEW)
        elif answer in ("s", "і", "ы"):
            open_in_editor(w / LOCAL_STRESS)
        elif answer in ("c", "с"):
            print("   ✓ скопійовано в буфер обміну" if copy_to_clipboard(claude_cmd) else f"   {claude_cmd}")


def run_pipeline(opt: Options, *, redo: str | None = None, stop_after: str | None = None,
                 review: bool = False, review_with: str | None = None, text: bool = False,
                 prefix: str = "", summary: bool = True) -> None:
    w = opt.work
    w.mkdir(parents=True, exist_ok=True)
    opt.save(w / "options.json")
    state_path = w / "state.json"
    state = json.loads(state_path.read_text()) if state_path.exists() else {}
    dirty = False
    t_all = time.time()
    titles = dict(STAGE_TITLES)
    from .llm import is_agent_llm

    if is_agent_llm(opt.llm):
        titles["translate"] = f"Перекладаю українською ({opt.llm.partition(':')[0]})"
    elif opt.llm.startswith("ollama:"):
        titles["translate"] = "Перекладаю українською (локальна LLM, Ollama)"
    titles["asr"] = (f"Читаю субтитри" if opt.subs
                     else f"Розпізнаю {LANGS.get(opt.source_lang, (opt.source_lang,))[0]} мову")
    if opt.text_lang == "uk":
        titles["translate"] = "Готую український текст (субтитри вже українською)"
    for i, stage in enumerate(STAGES, 1):
        fp = opt.fingerprint(stage)
        if stage == "tts" and state.get("translate"):
            from .review import REVIEW, import_review, review_hash

            if review or review_with:
                _review_pause(opt, state, state_path, review, review_with, force_agent=redo == "review")
            if (w / REVIEW).exists():
                if state.get("review_export") == state.get("translate"):
                    n = import_review(w)
                    if n:
                        print(f"   • застосовано правки з {REVIEW}: {n} реплік", flush=True)
                else:
                    print(f"   • {REVIEW} застарів (переклад змінився) — не застосовую", flush=True)
            fp = opt.fingerprint(stage)  # again: the review may have added words to stress.txt
            fp["review"] = review_hash(w)
        if redo == stage:
            dirty = True
        if not dirty and state.get(stage) == fp:
            print(f"{prefix}[{i}/{len(STAGES)}] {titles[stage]} — вже готово, пропускаю", flush=True)
            if stage == stop_after:
                break
            continue
        dirty = True
        for s in STAGES[STAGES.index(stage):]:
            state.pop(s, None)
        state_path.write_text(json.dumps(state, ensure_ascii=False, indent=2))
        print(f"{prefix}[{i}/{len(STAGES)}] {titles[stage]}…", flush=True)
        t0 = time.time()
        _run_stage(stage, opt)
        print(f"   ✓ {time.time() - t0:.1f} с", flush=True)
        state[stage] = fp
        state_path.write_text(json.dumps(state, ensure_ascii=False, indent=2))
        if stage == stop_after:
            if summary:
                print(f"\nЗупинено після етапу «{stage}». Для зручного редагування перекладу "
                      f"запустіть з --review.")
                _print_texts(opt, text)
            return
    if not summary:
        return
    from .term import link

    print(f"\nГотово за {(time.time() - t_all) / 60:.1f} хв:\n"
          f"   Відео:          {link(opt.output)}\n"
          f"   Субтитри:       {link(Path(opt.output).with_suffix('.srt'))}\n"
          f"   Робоча папка:   {link(w)}")
    _print_texts(opt, text)


def _print_texts(opt: Options, enabled: bool) -> None:
    """--text: the transcript and the translation as plain .txt files, with links."""
    if not enabled:
        return
    from .term import link
    from .texts import write_texts

    files = write_texts(opt.work, opt.input, opt.output, opt.text_lang)
    if not files:
        print("   (тексти з'являться після етапу перекладу)")
        return
    print("   Тексти:")
    for label, path in files:
        print(f"     {label + ':':<19}{link(path)}")


def _hf_cached(repo: str) -> bool:
    try:
        from huggingface_hub import try_to_load_from_cache

        return isinstance(try_to_load_from_cache(repo, "config.json"), str)
    except Exception:
        return False


def prefetch(opt: Options, all_engines: bool) -> None:
    """Download every model the chosen setup needs, so later runs work fully offline."""
    from huggingface_hub import snapshot_download

    from .llm import is_local_mlx

    repos = [opt.asr_model] + ([opt.llm] if is_local_mlx(opt.llm) else [])
    if all_engines and WHISPER_ASR not in repos:
        repos.append(WHISPER_ASR)  # other source languages (Korean, …)
    if all_engines or opt.engine == "omni":
        repos.append(OMNI_MODEL)
    for repo in repos:
        print(f"• {repo}", flush=True)
        snapshot_download(repo)
    print("• модель сепарації", opt.sep_model, flush=True)
    from audio_separator.separator import Separator

    from .config import CACHE_DIR

    Separator(model_file_dir=str(CACHE_DIR / "separator")).download_model_and_data(opt.sep_model)
    if all_engines or opt.engine == "ukr":
        print("• ukrainian-tts (+ словник наголосів)", flush=True)
        from .tts import UkrTTSEngine

        UkrTTSEngine("dmytro").synth("Перевірка наголосів.")
    if all_engines or opt.engine == "st":
        print("• StyleTTS2-ukrainian + голоси", flush=True)
        from huggingface_hub import hf_hub_download

        snapshot_download(ST_MODEL)
        for name in ST_VOICES:
            hf_hub_download(ST_SPACE, f"voices/{name}.pt", repo_type="space")
    if all_engines or opt.engine == "omni":
        print("• OmniVoice: перший прогін", flush=True)
        from .tts import OmniEngine

        OmniEngine(steps=8).model.generate(text="Перевірка.", language="uk", num_step=4)
        print("• модель наголосів (stanza uk)", flush=True)
        from .stress import StressFixer

        StressFixer("auto").apply("Старий замок.")
    print("\nУсе завантажено — далі uadub працює офлайн.")


def print_voices() -> None:
    from .config import ST_DEFAULT_FEMALE, ST_DEFAULT_MALE

    label = {"male": "чол.", "female": "жін.", "child": "дит."}
    print("Точні наголоси, без клонування — StyleTTS2-ukrainian (--voice st:<ім'я>):")
    for g in ("male", "female", "child"):
        names = [n for n, x in ST_VOICES.items() if x == g]
        if names:
            print(f"  {label[g]}: " + ", ".join(names))
    print(f"  за замовчуванням: st = {ST_DEFAULT_MALE}; duo:st = {ST_DEFAULT_MALE} + {ST_DEFAULT_FEMALE}")
    print("  приклад: --voice \"st:Марта Мольфар\"  (можна коротко: st:марта)")
    print("\nКлонування / голос за описом — OmniVoice (наголоси бувають неточні):")
    print("  clone · clone:зразок.wav · \"omni:male, middle-aged\" · \"omni:female, young adult\" · duo")
    print("  опис: male/female · child, teenager, young adult, middle-aged, elderly · low/high pitch")
    print("\nШвидкі прості (роботизовані) — ukrainian-tts:")
    print("  " + ", ".join(UKR_VOICES) + " · duo:dmytro,tetiana")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="uadub",
        description="Локальний AI-дубляж відео українською: з англійської, корейської та інших мов (Apple Silicon, офлайн).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Приклади:\n"
            "  uadub lecture.mp4 --voice st            # точні наголоси (StyleTTS2), без клонування\n"
            "  uadub talk.mp4 --voice clone            # голос оригінального спікера (наголоси бувають неточні)\n"
            "  uadub --list-voices                     # усі голоси\n"
            "  uadub talk.mp4 --voice clone:diktor.wav # голос із вашого зразка\n"
            "  uadub vlog.mp4 --voice tetiana --no-separate   # закадровий переклад поверх оригіналу\n"
            "  uadub talk.mp4 --review                 # пауза: правите сценарій перед озвученням\n"
            "  uadub talk.mp4 --review-with claude:sonnet   # агент перевіряє сценарій і одразу озвучує (+ --review — пауза)\n"
            "  uadub talk.mp4 --voice clone            # повторний запуск: перероблюються лише потрібні етапи\n"
            "  uadub drama.mkv --from ko --voice duo   # дорама: чоловічі/жіночі репліки різними голосами\n"
            "  uadub drama.mkv --from ko --subs drama.ko.srt   # з готових субтитрів (точніше за розпізнавання)\n"
            "  uadub drama.mkv --from ko --subs drama.uk.srt --subs-lang uk   # українські субтитри → лише озвучити\n"
            "  uadub --prefetch                        # завантажити всі моделі наперед\n"
        ),
    )
    p.add_argument("input", nargs="?", help="відео або аудіо (мова оригіналу — --from, за замовч. англійська)")
    p.add_argument("-o", "--output", help="куди зберегти результат (за замовчуванням <назва>.uk.mp4)")
    p.add_argument("--voice", default="dmytro", help=VOICE_HELP)
    p.add_argument("--gender", choices=["male", "female"], help="рід мовця для граматики (зробив/зробила)")
    p.add_argument("--ref-text", help="текст, який звучить у зразку для clone:<файл> (інакше розпізнається)")
    p.add_argument("--llm", default=os.environ.get("UADUB_LLM") or DEFAULT_LLM, metavar="MODEL",
                   help=f"модель перекладу: локальна MLX (за замовч. {DEFAULT_LLM}), ollama:<модель> або агент "
                        "claude[:модель], opencode[:провайдер/модель], codex[:модель], gemini[:модель] — через вашу "
                        "підписку, не офлайн. За замовч. — змінна UADUB_LLM")
    p.add_argument("--from", dest="source_lang", default="en", metavar="LANG",
                   help="мова відео: en (за замовч.), ko — корейська, ja, zh, es, fr, de… (не en → Whisper)")
    p.add_argument("--subs", metavar="FILE", help="взяти готові субтитри .srt/.vtt замість розпізнавання мови")
    p.add_argument("--subs-lang", metavar="LANG",
                   help="мова цих субтитрів (за замовч. = --from); uk — субтитри вже українською, лише озвучити")
    p.add_argument("--asr", default=None, help=argparse.SUPPRESS)
    p.add_argument("--glossary", help="файл термінів: рядки «English = Українська»")
    p.add_argument("--no-separate", action="store_true",
                   help="не відокремлювати фон: класичний закадровий переклад поверх приглушеного оригіналу")
    p.add_argument("--text", action="store_true",
                   help="зберегти транскрипцію оригіналу й переклад окремими .txt поруч із відео "
                        "(<назва>.en.txt, <назва>.uk.txt, <назва>.en-uk.txt, наголоси — <назва>.uk.stress.txt) і показати посилання")
    p.add_argument("--drop-original", action="store_true", help="не додавати оригінальну звукову доріжку")
    p.add_argument("--max-speed", type=float, default=1.25, help="максимальне прискорення мовлення (1.25)")
    p.add_argument("--pace", type=float, default=5.6, help="темп дубляжу, складів за секунду, однаковий для всіх реплік (5.6)")
    p.add_argument("--duck", type=float, help="приглушення фону під час мовлення, дБ (-4; без сепарації -13)")
    p.add_argument("--emotion", nargs="?", type=float, const=EMOTION_DEFAULT, default=0.0, metavar="K",
                   help=f"голоси st: брати інтонацію кожної репліки з оригіналу (тембр лишається українським). "
                        f"K від 0 до 1 — наскільки сильно (без числа — {EMOTION_DEFAULT}). "
                        "Для clone інтонація й так копіюється")
    p.add_argument("--domain", nargs="?", const="auto", default=None, metavar="СФЕРА",
                   help="фаховий переклад: термінологія сфери відео, зокрема усталені англіцизми (фреймворк, деплой); "
                        "кнопки інтерфейсу — як на екрані. Без СФЕРИ сфера визначається автоматично, або вкажіть її: "
                        "--domain \"медицина\". Без прапорця — проста літературна мова для всіх")
    p.add_argument("--steps", type=int, default=16, help="кроки OmniVoice: 16 ≈ реальний час, 32 — повільніше й трохи чистіше")
    p.add_argument("--fast", action="store_true", help="швидша сепарація фону (htdemucs, ~3× швидше, трохи гірше)")
    p.add_argument("--sep-model", help=argparse.SUPPRESS)
    p.add_argument("--stress", choices=["auto", "dict", "off"], default="auto",
                   help="наголоси: auto — словник + омографи за контекстом (за замовч.), dict — лише словник, off")
    p.add_argument("--stress-dict", help="додатковий словник наголосів (крім ~/.config/uadub/stress.txt): рядки «замо́к» або «зам+ок»")
    p.add_argument("--part-minutes", type=float, metavar="ХВ",
                   help="довге відео: різати звук у паузах на частини ~ХВ хвилин, обробляти по черзі й склеїти "
                        "(за замовч. автоматично для відео довших за 45 хв, частини по 15 хв; 0 — не різати)")
    p.add_argument("--keep-parts", action="store_true",
                   help="довге відео: не видаляти превʼю частин (<назва>.uk.parts/) після склеювання")
    p.add_argument("--review", action="store_true",
                   help="пауза перед озвученням: редагуєте сценарій review.md (вручну, Claude Code чи opencode), потім Enter")
    p.add_argument("--review-with", metavar="HARNESS[:MODEL]", default=os.environ.get("UADUB_REVIEW_WITH") or None,
                   help="автоматична перевірка сценарію агентом перед озвученням: claude, opencode, codex, gemini, "
                        "з моделлю через двокрапку (claude:sonnet, opencode:anthropic/claude-sonnet-4-5, codex:gpt-5, "
                        "gemini:gemini-2.5-pro) або власна команда з {prompt}. Без --review — одразу озвучує. "
                        "За замовч. — змінна UADUB_REVIEW_WITH")
    p.add_argument("--redo", choices=STAGES + ["review"],
                   help="примусово переробити етап і всі наступні (review — ще раз перевірити агентом)")
    p.add_argument("--stop-after", choices=STAGES, help="зупинитися після етапу (напр. translate)")
    p.add_argument("--workdir", help="папка проміжних файлів (за замовч. <назва>.uadub поруч із відео)")
    p.add_argument("--stress-lookup", nargs="+", metavar="СЛОВО",
                   help="показати наголоси слова за словником (усі варіанти з граматичними ознаками) і ваш stress.txt")
    p.add_argument("--list-voices", action="store_true", help="показати всі голоси")
    p.add_argument("--prefetch", action="store_true", help="лише завантажити моделі (з --all — для всіх голосів)")
    p.add_argument("--all", action="store_true", help=argparse.SUPPRESS)
    return p


def main(argv: list[str] | None = None) -> None:
    _env_defaults()
    args = build_parser().parse_args(argv)

    voice = args.voice.strip()
    if voice.lower() in UKR_VOICES:
        voice = voice.lower()
    elif voice.lower() in ("st", "duo:st"):
        voice = voice.lower()
    elif voice.startswith("st:"):
        name = find_st_voice(voice[3:])
        if not name:
            raise SystemExit(f"Невідомий голос StyleTTS2 «{voice[3:]}». Список: uadub --list-voices")
        voice = "st:" + name
    elif voice.startswith("duo:st:"):
        pair = [find_st_voice(v) for v in voice[7:].split(",")]
        if len(pair) != 2 or not all(pair):
            raise SystemExit("duo:st:<чоловічий>,<жіночий> — імена з uadub --list-voices")
        voice = "duo:st:" + ",".join(pair)
    elif voice.startswith("duo:"):
        pair = [v.strip().lower() for v in voice[4:].split(",")]
        if len(pair) != 2 or any(v not in UKR_VOICES for v in pair):
            raise SystemExit(f"duo:<чоловічий>,<жіночий> з готових голосів: {', '.join(UKR_VOICES)}")
        voice = "duo:" + ",".join(pair)
    elif not (voice in ("clone", "duo") or voice.startswith("clone:") or voice.startswith("omni:")):
        raise SystemExit(f"Невідомий голос «{voice}».\n{VOICE_HELP}")
    if voice.startswith("clone:") and not Path(voice[6:]).expanduser().exists():
        raise SystemExit(f"Файл зразка голосу не знайдено: {voice[6:]}")

    if args.list_voices:
        print_voices()
        return
    if args.stress_lookup:
        from .stress import lookup

        work = Path.cwd() if (Path.cwd() / "stress.txt").exists() else None  # run inside a .uadub folder
        print("\n\n".join(lookup(wd, work) for wd in args.stress_lookup))
        return
    if args.subs and not Path(args.subs).expanduser().exists():
        raise SystemExit(f"Файл субтитрів не знайдено: {args.subs}")
    if args.source_lang not in LANGS:
        print(f"Увага: мова «{args.source_lang}» не в списку перевірених ({', '.join(LANGS)}), пробую через Whisper.")
    if args.prefetch:
        opt = Options(input="", output="", workdir="", voice=voice, llm=args.llm,
                      asr_model=args.asr or (DEFAULT_ASR if args.source_lang == "en" else WHISPER_ASR),
                      sep_model=args.sep_model or (FAST_SEP_MODEL if args.fast else DEFAULT_SEP_MODEL))
        prefetch(opt, all_engines=args.all)
        return
    if not args.input:
        build_parser().print_help()
        sys.exit(1)

    inp = Path(args.input).expanduser().resolve()
    if not inp.exists():
        raise SystemExit(f"Файл не знайдено: {inp}")
    from . import audio as A

    A.require_ffmpeg()
    if args.output:
        out = Path(args.output).expanduser().resolve()
    else:
        has_video = A.has_stream(inp, "video")
        ext = inp.suffix.lower() if inp.suffix.lower() in {".mp4", ".mov", ".m4v", ".mkv"} else (".mkv" if has_video else ".m4a")
        out = inp.with_name(f"{inp.stem}.uk{ext}")
    work = Path(args.workdir).expanduser().resolve() if args.workdir else inp.with_name(f"{inp.stem}.uadub")

    opt = Options(
        input=str(inp), output=str(out), workdir=str(work), voice=voice, gender=args.gender,
        ref_text=args.ref_text, llm=args.llm, separate=not args.no_separate,
        asr_model=args.asr or (DEFAULT_ASR if args.source_lang == "en" else WHISPER_ASR),
        source_lang=args.source_lang,
        subs=str(Path(args.subs).expanduser().resolve()) if args.subs else None,
        subs_lang=args.subs_lang,
        glossary=str(Path(args.glossary).expanduser().resolve()) if args.glossary else None,
        keep_original=not args.drop_original, max_speed=args.max_speed, pace=args.pace, duck_db=args.duck,
        omni_steps=args.steps,
        sep_model=args.sep_model or (FAST_SEP_MODEL if args.fast else DEFAULT_SEP_MODEL),
        stress=args.stress,
        stress_dict=str(Path(args.stress_dict).expanduser().resolve()) if args.stress_dict else None,
        emotion=min(max(args.emotion or 0.0, 0.0), 1.0),
        domain=(args.domain or "").strip() or None,
        part_minutes=args.part_minutes, keep_parts=args.keep_parts,
    )
    if opt.emotion and opt.engine != "st":
        print("Увага: --emotion працює лише з голосами StyleTTS2 (st, st:<ім'я>, duo:st)"
              + ("; clone і так копіює інтонацію оригіналу." if opt.engine == "omni" else "."), flush=True)
        opt.emotion = 0.0
    from .llm import is_agent_llm, is_local_mlx

    needed = [opt.asr_model] + ([opt.llm] if is_local_mlx(opt.llm) else [])
    if opt.engine == "omni":
        needed.append(OMNI_MODEL)
    if opt.engine == "st":
        needed.append(ST_MODEL)
    if all(_hf_cached(r) for r in needed):
        os.environ["HF_HUB_OFFLINE"] = "1"  # everything is local → never touch the network

    if is_agent_llm(opt.llm):
        print(f"Увага: переклад через {opt.llm.partition(':')[0]} — текст відео надсилається в хмару "
              "(розпізнавання й озвучення лишаються локальними).", flush=True)
    extra = f", інтонація оригіналу {opt.emotion:g}" if opt.emotion else ""
    if opt.domain:
        extra += ", фаховий переклад" + ("" if opt.domain == "auto" else f" ({opt.domain})")
    print(f"uadub: {inp.name} → {out.name}  (голос: {voice}{extra}, LLM: {opt.llm})\n", flush=True)
    from .parts import choose_part_minutes, load_or_make_plan, run_long

    minutes = choose_part_minutes(opt, A.duration(inp))
    if minutes:
        if args.review:
            print("Порада: --review зупинятиметься перед озвученням кожної частини; "
                  "для довгого відео зручніше --review-with.", flush=True)
        plan = load_or_make_plan(opt.work, inp, minutes, opt.sample_rate)
        run_long(opt, plan, redo=args.redo, stop_after=args.stop_after, review=args.review,
                 review_with=args.review_with, text=args.text)
        return
    run_pipeline(opt, redo=args.redo, stop_after=args.stop_after, review=args.review, review_with=args.review_with,
                 text=args.text)


if __name__ == "__main__":
    main()
