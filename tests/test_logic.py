from pathlib import Path

from uadub.fit import place_clips, st_speed, target_duration
from uadub.llm import extract_json
from uadub.segments import assign_slots, build_units, merge_tokens_to_words
from uadub.srt import make_cues, wrap
from uadub.textnorm import latin_to_cyrillic, syllables, to_speech_text


def test_syllables_ukrainian_exact():
    assert syllables("Привіт, як справи?") == 5  # при-віт як спра-ви
    assert syllables("їжак") == 2
    assert syllables("API") == 3


def test_speech_text_numbers_and_latin():
    s = to_speech_text("Ціна $5 і 10% знижки на GitHub")
    assert "5" not in s and "10" not in s and "%" not in s and "$" not in s
    assert "доларів" in s and "відсотків" in s
    assert not any("a" <= c.lower() <= "z" for c in s)


def test_latin_acronyms_spelled():
    assert latin_to_cyrillic("AI") == "ей-ай"
    assert latin_to_cyrillic("Python") == "Пітон"


def test_merge_tokens():
    toks = [{"text": " Hel", "start": 0.0, "end": 0.2}, {"text": "lo", "start": 0.2, "end": 0.4},
            {"text": " world", "start": 0.5, "end": 0.9}]
    assert [w["w"] for w in merge_tokens_to_words(toks)] == ["Hello", "world"]


def test_build_units_merges_short_and_slots():
    sents = [
        {"start": 0.0, "end": 0.6, "text": "Okay."},
        {"start": 0.8, "end": 3.0, "text": "Today we talk about dubbing."},
        {"start": 5.0, "end": 8.0, "text": "It is a long and complicated topic."},
    ]
    units = build_units(sents)
    assert len(units) == 2 and units[0]["text"].startswith("Okay. Today")
    assign_slots(units, 10.0, tail=0.8)
    assert units[0]["slot_end"] == 3.8  # tail-limited
    assert units[1]["slot_end"] == 8.8
    assign_slots(units, 8.2, tail=0.8)
    assert units[1]["slot_end"] == 8.2  # video end


def test_place_clips_speedup_and_spill():
    units = [{"start": 0.0, "end": 2.0, "slot_end": 2.5}, {"start": 2.6, "end": 4.0, "slot_end": 5.0}]
    plan = place_clips(units, [2.3, 3.6], max_speed=1.25, spill=0.0)
    assert plan[0]["speed"] == 1.0  # fits, and leaves a 0.3 s gap before the next line
    assert abs(plan[1]["speed"] - 1.25) < 1e-6  # needs 1.5x, capped
    assert plan[1]["start"] == 2.6
    plan = place_clips(units, [3.5, 1.0], max_speed=1.25, spill=0.0)
    assert plan[0]["speed"] == 1.25 and plan[1]["start"] > 2.6  # spill pushes next line


def test_place_clips_small_overrun_is_not_stretched():
    # A line 0.3 s over its slot keeps its pace: the overrun spills and the next line starts late.
    units = [{"start": 0.0, "end": 2.0, "slot_end": 2.5}, {"start": 2.6, "end": 4.0, "slot_end": 5.0}]
    plan = place_clips(units, [2.8, 1.0], max_speed=1.25)
    assert plan[0]["speed"] == 1.0
    assert plan[1]["start"] > 2.6
    # Beyond the allowed spill it is stretched to fit `avail + spill`, not the bare slot.
    plan = place_clips(units, [3.5, 1.0], max_speed=1.25, spill=0.5)
    assert abs(plan[0]["speed"] - 3.5 / 3.0) < 1e-3
    # Tiny stretches (< 8 %) are skipped: 3.2 s into 3.0 s would be 1.067.
    plan = place_clips(units, [3.2, 1.0], max_speed=1.25, spill=0.5)
    assert plan[0]["speed"] == 1.0


def test_st_speed_targets_a_pace_band():
    # Drawn-out line (12 syllables in 2.88 s = 4.2 syl/s) is brought up to the band floor, 5 syl/s.
    assert abs(st_speed(2.88, 12, 3.12) - 5.0 * 2.88 / 12) < 1e-6
    # In the band and fits its slot: untouched.
    assert st_speed(2.82, 15, 2.96) == 1.0
    # Slightly too long for the slot: sped up just enough.
    assert abs(st_speed(3.16, 17, 2.96) - 3.16 / 2.96) < 1e-6
    # Far too long: the speed-up stops where the pace would exceed 7.5 syl/s (never 1.35x here).
    assert abs(st_speed(2.75, 20, 1.84) - 7.5 * 2.75 / 20) < 1e-6
    # Already faster than the ceiling: never sped up, even when it does not fit.
    assert st_speed(5.02, 41, 4.0) == 1.0
    # No syllable count (digits only, etc.): fit the slot, capped at max_speed.
    assert abs(st_speed(3.0, 0, 2.0) - 1.35) < 1e-6
    assert st_speed(3.0, 0, 2.95) == 1.0  # 1.017: below the 2 % threshold


def test_target_duration():
    assert target_duration(2.0, 3.0, 1.25) is None
    assert target_duration(3.0, 2.0, 1.25) == 2.4
    assert target_duration(2.2, 2.0, 1.25) == 2.0


def test_extract_json_variants():
    assert extract_json('```json\n{"lines": []}\n```') == {"lines": []}
    assert extract_json('Sure! {"lines": [{"id": 1}]} hope it helps') == {"lines": [{"id": 1}]}
    assert extract_json('<think>hmm</think>[1, 2]') == [1, 2]


def test_srt_cues_split():
    assert wrap("one two three", 7) == ["one two", "three"]
    units = [{"start": 0.0, "end": 6.0, "uk": "слово " * 30}, {"start": 7.0, "end": 8.0, "uk": "ні"}]
    cues = make_cues(units, "uk")
    assert len(cues) >= 3 and cues[-1][2] == "ні"
    assert all(b > a for a, b, _ in cues)


def test_plus_before_vowel_is_stress_not_plus():
    assert to_speech_text("зам+ок і C+") .startswith("зам+ок")
    assert "плюс" in to_speech_text("2 + 2")


def test_stress_parsing_and_arpa(tmp_path):
    from uadub.stress import StressFixer, load_dict, parse_marked, to_arpa

    assert parse_marked("замо\u0301к") == ("замок", 1)
    assert parse_marked("зам+ок") == ("замок", 1)
    assert parse_marked("за´мок") == ("замок", 0)
    assert parse_marked("замОк") == ("замок", 1)
    assert parse_marked("Замок") is None
    assert to_arpa("замок", 1) == "[Z AA0 M AO1 K]"
    assert to_arpa("джерело", 2) == "[JH EH0 R EH0 L AO1]"
    assert to_arpa("сьогодні", 1) == "[S Y AO0 HH AO1 D N IY0]"
    d = tmp_path / "stress.txt"
    d.write_text("# коментар\nзамок = замо\u0301к\nвип+адок\n", encoding="utf-8")
    assert load_dict([d]) == {"замок": 1, "випадок": 1}
    fx = StressFixer("dict", [d])
    assert fx.apply("Новий замок, а не випадок.") == "Новий [Z AA0 M AO1 K], а не [V IH0 P AA1 D AO0 K]."
    assert StressFixer("dict", [d], target="acute").apply("Замок") == "Замо\u0301к"
    assert fx.apply("рук+а") == "[R UW0 K AA1]"  # explicit marks in text always win
    assert StressFixer("off", [d]).apply("замо\u0301к") == "замок"
    from uadub.stress import apply_marks, variant_caps
    assert apply_marks("Новий замок, старий замок.", {1: 1, 3: 0}) == "Новий замо\u0301к, старий за\u0301мок."
    assert variant_caps("замок", 0) == "зАмок"


def test_review_roundtrip(tmp_path):
    import json

    from uadub.review import export_review, import_review, parse_review, to_plus

    assert to_plus("замо́к і ру́ки") == "зам+ок і р+уки"
    units = [
        {"id": 1, "start": 0.0, "end": 2.0, "slot_end": 2.5, "text": "The castle.", "uk": "Замок.", "max_syl": 10},
        {"id": 2, "start": 3.0, "end": 5.0, "slot_end": 5.5, "text": "Five apples.", "uk": "5 яблук.",
         "tts": "П'ять яблук.", "max_syl": 10},
        {"id": 3, "start": 6.0, "end": 8.0, "slot_end": 8.5, "text": "Hi.", "uk": "Привіт.", "max_syl": 10},
    ]
    (tmp_path / "units.json").write_text(json.dumps(units, ensure_ascii=False))
    path = export_review(tmp_path)
    text = path.read_text()
    assert "## 0001" in text and "EN:  The castle." in text and (tmp_path / "AGENTS.md").exists()
    text = text.replace("UK:  Замок.", "UK:  Замок.").replace("TTS: \n\n## 0002", "TTS: Зам+ок.\n\n## 0002", 1)
    text = text.replace("UK:  5 яблук.", "UK:  П'ять яблук!")  # UK edited, old TTS untouched → TTS dropped
    text = text.replace("UK:  Привіт.", "UK:  Вітаю.")
    path.write_text(text)
    assert set(parse_review(text)) == {1, 2, 3}
    assert import_review(tmp_path) == 3
    out = {u["id"]: u for u in json.loads((tmp_path / "units.json").read_text())}
    assert out[1]["tts"] == "Зам+ок." and out[1]["uk"] == "Замок."
    assert out[2]["uk"] == "П'ять яблук!" and "tts" not in out[2]
    assert out[3]["uk"] == "Вітаю."
    assert (tmp_path / "uk.srt").exists()


def test_read_subs_and_word_sentences(tmp_path):
    from uadub.segments import words_to_sentences
    from uadub.subs import read_subs

    srt = tmp_path / "a.srt"
    srt.write_text("1\n00:00:01,000 --> 00:00:02,500\n<i>- 오빠, 어디 가요?</i>\n\n"
                   "2\n00:00:03,000 --> 00:00:05,000\n- 회사에 가야 돼.\n- 왜?\n\n", encoding="utf-8")
    subs = read_subs(srt)
    assert [s["text"] for s in subs] == ["오빠, 어디 가요?", "회사에 가야 돼. 왜?"]
    assert subs[0]["start"] == 1.0 and subs[1]["end"] == 5.0
    vtt = tmp_path / "b.vtt"
    vtt.write_text("WEBVTT\n\n00:01.000 --> 00:02.000\n안녕\n", encoding="utf-8")
    assert read_subs(vtt)[0]["text"] == "안녕"
    words = [{"w": "오빠,", "start": 0.0, "end": 0.4}, {"w": "어디", "start": 0.5, "end": 0.8},
             {"w": "가요?", "start": 0.8, "end": 1.2}, {"w": "회사에", "start": 2.5, "end": 3.0},
             {"w": "가야", "start": 3.0, "end": 3.3}, {"w": "돼", "start": 3.3, "end": 3.6}]
    sents = words_to_sentences(words)
    assert [s["text"] for s in sents] == ["오빠, 어디 가요?", "회사에 가야 돼"]


def test_clone_window_respects_voice_pitch():
    from uadub.stages import _clone_window

    units = [{"start": 0.0, "end": 1.0}, {"start": 1.2, "end": 2.0}, {"start": 2.2, "end": 5.0}]
    assert _clone_window(units, 1) == (0, 2)  # no pitch info → borrow neighbours
    assert _clone_window(units, 1, [110.0, 220.0, 115.0]) == (1, 1)  # neighbours are another voice
    assert _clone_window(units, 1, [210.0, 220.0, 115.0]) == (0, 1)


def test_snap_to_speech():
    import numpy as np

    from uadub.segments import snap_to_speech

    sr = 1000
    y = np.zeros(6 * sr)
    y[1000:2500] = 0.5 * np.sin(np.arange(1500))  # speech 1.0–2.5 s
    y[3200:4000] = 0.5 * np.sin(np.arange(800))  # speech 3.2–4.0 s
    s = [{"start": 0.0, "end": 2.2, "text": "a"}, {"start": 3.4, "end": 4.3, "text": "b"}]
    assert snap_to_speech(s, y, sr) == 2
    assert abs(s[0]["start"] - 0.95) < 0.02 and abs(s[0]["end"] - 2.55) < 0.06
    assert abs(s[1]["start"] - 3.15) < 0.02 and abs(s[1]["end"] - 4.05) < 0.02


def test_find_suspect_anglicisms():
    from uadub.translate import find_suspects

    known = {"натисніть", "кнопку", "модель", "відкрийте", "налаштування", "використати", "цю"}.__contains__
    u = {"uk": "Натисніть «use this model» і відкрийте рантайм у LM Studio, чекбокс у налаштування"}
    assert find_suspects(u, known, set()) == ["model", "this", "use", "рантайм", "чекбокс"]
    assert find_suspects({"uk": "Натисніть «Використати цю модель»"}, known, set()) == []


def test_omni_duration_never_drawn_out():
    from uadub.stages import _omni_duration

    class Eng:
        def __init__(self, est):
            self.est = est

        def natural_duration(self, text, prompt):
            return self.est

    text = "Ви натискаєте на квантування, і з'явиться вікно."  # 17 syllables
    assert _omni_duration(Eng(9.0), text, None, 10.0, 1.25) == round(17 / 5.0, 3)  # inflated → capped at 5 syl/s
    assert _omni_duration(Eng(2.8), text, None, 10.0, 1.25) == 2.8  # natural pace kept
    assert _omni_duration(Eng(2.8), text, None, 2.0, 1.25) == 2.24  # too long for slot → faster


def test_styletts_text_preparation():
    from uadub.tts import prepare_st_text, split_sentences

    assert prepare_st_text("Новий зам+ок") == "Новий замо́к."
    assert prepare_st_text("ви́си́ть") == "ви́сить."  # two allowed stresses → first
    assert prepare_st_text("«Так» — ні!") == "Так: ні!"
    assert split_sentences("Так. Ну. Сьогодні ми перевіримо модель! А потім ще одну?") == [
        "Так. Ну. Сьогодні ми перевіримо модель!", "А потім ще одну?"]


def test_engine_aware_review_texts():
    from uadub.review import agent_prompt, agent_task, header, parse_review

    assert "НАГОЛОСИ:" in agent_task("st") and "stress.txt" in agent_task("st")
    assert "НЕ виправляй" in agent_task("omni")
    assert "наголоси не чіпай" in agent_prompt("omni") and "НАГОЛОСИ" in agent_prompt("st")
    assert "--voice st" in header("omni") and "НАГОЛОСИ:" in header("st")
    text = "## 0001  00:00.0–00:02.0\nEN:  Hi.\nUK:  Привіт.\nTTS: \nНАГОЛОСИ: Прив+іт.\n"
    assert parse_review(text) == {1: {"uk": "Привіт.", "tts": ""}}


def test_sentence_start_stress_prefers_common_word():
    import re

    from uadub.tts import prepare_st_text

    def fake(text):  # mimics ukrainian-word-stress: capitalised word looked up as a proper noun
        text = re.sub(r"\bКоли\b", "Ко\u0301ли", text)
        text = re.sub(r"\bколи\b", "коли\u0301", text)
        return text.replace("модель", "моде\u0301ль")

    out = prepare_st_text("Коли модель готова. Я знаю, коли.", fake)
    assert out.startswith("Коли\u0301 моде\u0301ль")
    assert "знаю, коли\u0301." in out
    assert prepare_st_text("Київ", lambda x: x.replace("Київ", "Ки\u0301їв")).startswith("Ки\u0301їв")


def test_emotion_fingerprint_only_when_on(tmp_path):
    from uadub.config import Options

    base = dict(input="a.mp4", output="a.uk.mp4", workdir=str(tmp_path))
    assert "emotion" not in Options(voice="st", **base).fingerprint("tts")
    assert Options(voice="st", emotion=0.8, **base).fingerprint("tts")["emotion"] == 0.8
    assert "emotion" not in Options(voice="clone", emotion=0.8, **base).fingerprint("tts")


def test_salvage_lines_with_echoed_fields_and_cutoff():
    from uadub.translate import _salvage_lines

    reply = ('{"lines": [{"id": 1, "src": "오빠", "max_syl": 8, "uk": "Оппа, вже йдеш?", "tts": "Оппа"},'
             '{"id": 2, "speaker": "male", "uk": "Так."}, {"id": 3, "src": "응", "uk": "Обрі')
    got = _salvage_lines(reply)
    assert [x["id"] for x in got] == [1, 2] and got[0]["tts"] == "Оппа" and got[1]["uk"] == "Так."


def test_write_texts(tmp_path):
    import json

    from uadub.texts import write_texts

    units = [{"start": 0.0, "end": 1.0, "text": "Hello there.", "uk": "Привіт."},
             {"start": 1.2, "end": 2.0, "text": "How are you?", "uk": "Як справи?"},
             {"start": 65.0, "end": 66.0, "text": "Bye.", "uk": "Бувай."}]
    (tmp_path / "units.json").write_text(json.dumps(units))
    files = dict(write_texts(tmp_path, "/x/video.mp4", str(tmp_path / "video.uk.mp4"), "en"))
    assert (tmp_path / "video.en.txt").read_text() == "[00:00] Hello there. How are you?\n\n[01:05] Bye.\n"
    assert (tmp_path / "video.uk.txt").read_text().startswith("[00:00] Привіт. Як справи?")
    assert "EN: Bye.\nUK: Бувай." in (tmp_path / "video.en-uk.txt").read_text()
    assert len(files) == 3


def test_stress_lookup_formatting():
    from uadub.stress import _describe, _plus_at

    assert _plus_at("замок", 4) == "зам+ок"
    assert _describe(["Case=Nom", "Gender=Masc", "Number=Sing", "upos=NOUN"]) == "іменник ч. р. одн. наз."


def test_agent_command_specs():
    from uadub.review import agent_command

    c = agent_command("claude:sonnet", "P")
    assert c[:3] == ["claude", "-p", "P"] and c[c.index("--model") + 1] == "sonnet"
    assert "--model" not in agent_command("claude", "P")
    assert agent_command("opencode:ollama/qwen3:8b", "P") == ["opencode", "run", "-m", "ollama/qwen3:8b", "P"]
    assert agent_command("codex:gpt-5", "P")[-3:] == ["-m", "gpt-5", "P"]
    assert agent_command("gemini", "P")[-2:] == ["-p", "P"]
    assert agent_command("my-agent --fast", "P") == ["my-agent", "--fast", "P"]


def test_llm_spec_kinds():
    from uadub.llm import is_agent_llm, is_local_mlx

    assert is_agent_llm("claude") and is_agent_llm("claude:sonnet") and is_agent_llm("opencode:ollama/qwen3:8b")
    assert not is_agent_llm("mlx-community/gemma-4-26b-a4b-it-4bit") and not is_agent_llm("ollama:qwen3")
    assert is_local_mlx("mlx-community/gemma-4-26b-a4b-it-4bit") and not is_local_mlx("ollama:qwen3")


def test_salvage_timeline_keeps_pieces_at_their_time():
    import numpy as np

    from uadub.audio import salvage_timeline, silent_fraction

    sr, total = 100, 10.0
    calls = []

    def decode(pos):  # the "decoder" dies at 3 s and at 6 s of the source
        calls.append(pos)
        end = 3.0 if pos < 3 else 6.0 if pos < 6 else total
        y = np.ones((round((end - pos) * sr), 2), np.float32) * (pos + 1)
        return y, end == total

    y, runs = salvage_timeline(decode, total, sr, skip=0.5)
    assert len(y) == 1000 and runs == 3
    assert calls == [0.0, 3.5, 6.5]
    assert y[0, 0] == 1 and y[299, 0] == 1  # first run: 0–3 s
    assert not y[300:350].any()  # skipped past the failure → silence
    assert y[350, 0] == 4.5 and y[599, 0] == 4.5  # second run starts exactly at 3.5 s
    assert y[650, 0] == 7.5 and y[-1, 0] == 7.5
    assert abs(silent_fraction(y, sr) - 0.1) < 1e-9


def test_salvage_timeline_gives_up_on_a_dead_decoder():
    import numpy as np

    from uadub.audio import salvage_timeline, silent_fraction

    y, runs = salvage_timeline(lambda pos: (np.zeros((0, 2), np.float32), False), 5.0, 100, skip=0.5)
    assert len(y) == 500 and runs == 10
    assert silent_fraction(y, 100) == 1.0


def test_domain_fingerprint_only_when_on(tmp_path):
    from uadub.config import Options

    base = dict(input="a.mp4", output="a.uk.mp4", workdir=str(tmp_path), voice="st")
    assert "domain" not in Options(**base).fingerprint("translate")
    assert Options(domain="auto", **base).fingerprint("translate")["domain"] == "auto"
    assert Options(domain="медицина", **base).fingerprint("translate")["domain"] == "медицина"


def test_domain_prompts():
    from uadub.translate import SYSTEM, anglicism_system, brief_system, terms_rule

    kw = dict(gender_rule="g", glossary="", src_name="English", lang_rules="")
    plain = SYSTEM.format(**kw, terms_rule=terms_rule(None))
    expert = SYSTEM.format(**kw, terms_rule=terms_rule("IT / розробка ПЗ"))
    assert "No anglicisms" in plain and "фреймворк" not in plain
    assert "specialists in IT / розробка ПЗ" in expert and "фреймворк" in expert and "«Use this model»" in expert
    assert '"domain"' in brief_system("English", "", None) and "квантування" in brief_system("English", "", None)
    assert "деплой" in brief_system("English", "", "auto")
    assert "«медицина»" in brief_system("English", "", "медицина")
    assert "proper Ukrainian equivalents" in anglicism_system(None)
    assert "specialists in IT" in anglicism_system("IT") and "{fix_rule}" not in anglicism_system("IT")


def test_translate_units_specialist_mode():
    import json as _json

    from uadub.translate import translate_units

    seen, logs = [], []

    class FakeLLM:
        def chat(self, system, user, *, max_tokens=4096, temperature=0.3):
            seen.append(system)
            if system.startswith("You prepare a translation brief"):
                return _json.dumps({"domain": "IT / розробка ПЗ", "summary": "Про деплой.",
                                    "glossary": [{"src": "deploy", "uk": "деплой"}]})
            if system.startswith("You are an expert audiovisual translator"):
                ids = [l["id"] for l in _json.loads(user)["lines"]]
                return _json.dumps({"lines": [{"id": i, "uk": "Робимо деплой."} for i in ids]})
            return _json.dumps({"lines": []})

    units = [{"id": 1, "text": "Let's deploy.", "start": 0.0, "end": 2.0, "slot_end": 2.0}]
    info = translate_units(FakeLLM(), units, rate=6.0, gender="male", glossary_path=None, stress="off",
                           domain="auto", log=logs.append)
    assert info["domain"] == "IT / розробка ПЗ" and info["domain_mode"]
    assert any("сфера: IT / розробка ПЗ (фаховий переклад)" in l for l in logs)
    translate_prompt = next(s for s in seen if s.startswith("You are an expert audiovisual translator"))
    assert "specialists in IT / розробка ПЗ" in translate_prompt and "деплой" in translate_prompt
    assert units[0]["uk"] == "Робимо деплой."  # glossary jargon is not "fixed" away
    assert not any("прибираю" in l for l in logs)

    seen.clear(), logs.clear()
    units = [{"id": 1, "text": "Let's deploy.", "start": 0.0, "end": 2.0, "slot_end": 2.0}]
    info = translate_units(FakeLLM(), units, rate=6.0, gender="male", glossary_path=None, stress="off",
                           log=logs.append)
    assert not info["domain_mode"] and any(l.endswith("сфера: IT / розробка ПЗ") for l in logs)
    assert "No anglicisms" in next(s for s in seen if s.startswith("You are an expert audiovisual translator"))


def test_review_texts_follow_domain_mode(tmp_path):
    import json as _json

    from uadub.config import Options
    from uadub.review import agent_prompt, agent_task, header, review_domain

    assert "без англіцизмів" in header("st") and "Без англіцизмів" in agent_task("st")
    assert "фреймворк" in header("st", "IT") and "без англіцизмів" not in header("st", "IT")
    task = agent_task("st", "en.srt", "IT")
    assert "Без англіцизмів" not in task and "фахівці сфери «IT»" in task and "усталені — лишай" in task
    assert "без англіцизмів" in agent_prompt("st") and "«IT»" in agent_prompt("st", "IT")

    base = dict(input="a.mp4", output="a.uk.mp4", workdir=str(tmp_path), voice="st")
    Options(**base).save(tmp_path / "options.json")
    assert review_domain(tmp_path) is None
    Options(domain="auto", **base).save(tmp_path / "options.json")
    assert review_domain(tmp_path) == "сфера відео"
    (tmp_path / "brief.json").write_text(_json.dumps({"domain": "IT / розробка ПЗ"}), encoding="utf-8")
    assert review_domain(tmp_path) == "IT / розробка ПЗ"
    Options(domain="медицина", **base).save(tmp_path / "options.json")
    assert review_domain(tmp_path) == "медицина"


def test_anglicism_pass_takes_spelling_of_kept_ui_label():
    import json as _json

    from uadub.translate import _dictionary_lookup, fix_anglicisms

    if _dictionary_lookup() is None:
        return  # dictionary not installed

    class FakeLLM:
        def chat(self, system, user, *, max_tokens=4096, temperature=0.3):
            assert "specialists in IT" in system
            ids = [l["id"] for l in _json.loads(user)["lines"]]
            return _json.dumps({"lines": [{"id": i, "uk": "Натисніть «use this model».",
                                           "tts": "Натисніть юз зіс модел."} for i in ids]})

    units = [{"id": 1, "text": "Click use this model.", "uk": "Натисніть «use this model».", "max_syl": 12}]
    assert fix_anglicisms(FakeLLM(), units, domain="IT", log=lambda m: None) == 0
    assert units[0]["uk"] == "Натисніть «use this model»." and units[0]["tts"] == "Натисніть юз зіс модел."


def test_spell_latin_fills_missing_tts_only():
    import json as _json

    from uadub.translate import spell_latin

    asked = []

    class FakeLLM:
        def chat(self, system, user, *, max_tokens=4096, temperature=0.3):
            lines = _json.loads(user)["lines"]
            asked.extend(l["id"] for l in lines)
            return _json.dumps({"lines": [{"id": l["id"], "tts": "Натисніть юз зіс модел."} for l in lines]})

    units = [{"id": 1, "uk": "Натисніть «Use this model»."},
             {"id": 2, "uk": "Відкрийте LM Studio.", "tts": "Відкрийте ел-ем студіо."},
             {"id": 3, "uk": "Усе готово."}]
    assert spell_latin(FakeLLM(), units, log=lambda m: None) == 1
    assert asked == [1] and units[0]["tts"] == "Натисніть юз зіс модел."
    assert units[1]["tts"] == "Відкрийте ел-ем студіо." and "tts" not in units[2]


def test_existing_and_fit_length(tmp_path):
    import numpy as np

    from uadub import audio as A

    wav = tmp_path / "vocals.wav"
    assert A.existing(wav) == wav  # nothing there: the name itself
    A.write_flac(tmp_path / "vocals.flac", np.zeros((10, 2), np.float32), 44100)
    assert A.existing(wav) == tmp_path / "vocals.flac"
    A.write(wav, np.zeros((10, 2), np.float32), 44100)
    assert A.existing(wav) == wav  # WAV wins
    y = np.ones((5, 2), np.float32)
    assert A.fit_length(y, 3).shape == (3, 2)
    padded = A.fit_length(y, 8)
    assert padded.shape == (8, 2) and not padded[5:].any()
    assert A.fit_length(np.ones(4, np.float32), 6).shape == (6,)


def test_to_flac_replaces_wav(tmp_path):
    import numpy as np
    import soundfile as sf

    from uadub import audio as A

    wav = tmp_path / "background.wav"
    A.write(wav, np.full((441, 2), 0.25, np.float32), 44100)
    out = A.to_flac(wav)
    assert out == tmp_path / "background.flac" and not wav.exists()
    y, sr = sf.read(str(out), dtype="float32")
    assert sr == 44100 and y.shape == (441, 2) and abs(float(y[0, 0]) - 0.25) < 1e-4


def _sine_m4a(path, seconds=4, rate=48000):
    import subprocess

    subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-y", "-f", "lavfi", "-i",
                    f"sine=frequency=440:sample_rate={rate}:duration={seconds}", "-c:a", "aac", str(path)],
                   check=True)
    return path


def test_extract_audio_clip_is_sample_exact(tmp_path):
    import soundfile as sf

    from uadub import audio as A

    src = _sine_m4a(tmp_path / "src.m4a")
    A.extract_audio(src, tmp_path / "a.wav", 44100, start=1.0, length=1.5)
    assert sf.info(str(tmp_path / "a.wav")).frames == 66150


def test_salvage_audio_clip_is_sample_exact(tmp_path):
    from uadub import audio as A

    src = _sine_m4a(tmp_path / "src.m4a")
    y, runs = A._salvage_audio(src, 44100, start=1.0, length=1.5)
    assert y.shape == (66150, 2) and runs == 1 and abs(y).max() > 0.05


def test_part_fingerprints_only_when_set(tmp_path):
    from uadub.config import Options

    base = dict(input="a.mp4", output="a.uk.mp4", workdir=str(tmp_path), voice="st")
    plain = Options(**base)
    for stage in ("extract", "translate", "mix", "mux"):
        assert not {"clip", "shared_brief", "edge", "loudness_target"} & set(plain.fingerprint(stage))
    (tmp_path / "b.json").write_text("{}")
    (tmp_path / "e.json").write_text("{}")
    part = Options(**base, clip_start=10.0, clip_end=20.0, shared_brief=str(tmp_path / "b.json"),
                   edge_context=str(tmp_path / "e.json"), loudness_target=-19.5)
    assert part.fingerprint("extract")["clip"] == [10.0, 20.0]
    before = part.fingerprint("translate")
    (tmp_path / "b.json").write_text('{"a": 1}')
    after = part.fingerprint("translate")
    assert after["shared_brief"] != before["shared_brief"] and after["edge"] == before["edge"]
    assert part.fingerprint("mix")["loudness_target"] == -19.5
    assert part.fingerprint("mix")["clip"] == [10.0, 20.0] and part.fingerprint("mux")["clip"] == [10.0, 20.0]


def _test_video(path, seconds=6):
    import subprocess

    subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-y",
                    "-f", "lavfi", "-i", f"testsrc=size=160x120:rate=25:duration={seconds}",
                    "-f", "lavfi", "-i", f"sine=frequency=300:sample_rate=44100:duration={seconds}",
                    "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", "-shortest", str(path)], check=True)
    return path


def test_mux_clip_preview_and_audio_only(tmp_path):
    import numpy as np

    from uadub import audio as A

    dub = tmp_path / "dub.flac"
    A.write_flac(dub, np.zeros((44100 * 2, 2), np.float32), 44100)
    video = _test_video(tmp_path / "v.mp4")
    out = tmp_path / "p.uk.mp4"
    A.mux(video, dub, out, srt=None, keep_original=True, clip=(2.0, 2.0))
    assert abs(A.duration(out) - 2.0) < 0.15 and A.has_stream(out, "video")
    import subprocess

    grainy = tmp_path / "grainy.mp4"  # a real-looking ~400 kbps source (testsrc alone compresses to nothing)
    subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-y", "-f", "lavfi", "-i",
                    "testsrc=size=320x240:rate=25:duration=6", "-vf", "noise=alls=40:allf=t",
                    "-c:v", "libx264", "-b:v", "300k", "-pix_fmt", "yuv420p", str(grainy)], check=True)
    A.mux(grainy, dub, out, srt=None, keep_original=False, clip=(2.0, 2.0))
    source_rate = A.video_bitrate(A.probe(grainy))
    assert source_rate and A.video_bitrate(A.probe(out)) <= 1.5 * source_rate  # previews never outgrow the source
    podcast = _sine_m4a(tmp_path / "pod.m4a", seconds=6)
    out = tmp_path / "p.uk.m4a"
    A.mux(podcast, dub, out, srt=None, keep_original=True, clip=(2.0, 2.0))
    assert abs(A.duration(out) - 2.0) < 0.15 and not A.has_stream(out, "video")


def test_cleanup_part_compresses_stems(tmp_path):
    import numpy as np

    from uadub import audio as A
    from uadub.stages import _cleanup_part

    for name in ("audio.wav", "vocals.wav", "background.wav", "mix.wav"):
        A.write(tmp_path / name, np.zeros((100, 2), np.float32), 44100)
    _cleanup_part(tmp_path)
    assert sorted(p.name for p in tmp_path.iterdir()) == ["background.flac", "vocals.flac"]
    A.write(tmp_path / "audio.wav", np.zeros((100, 2), np.float32), 44100)  # --no-separate part
    for name in ("vocals.flac", "background.flac"):
        (tmp_path / name).unlink()
    _cleanup_part(tmp_path)
    assert sorted(p.name for p in tmp_path.iterdir()) == ["audio.flac"]


def _speechy_db(total, frame=0.05):
    import numpy as np

    db = np.full(int(round(total / frame)), -20.0)
    db[::5] = -65.0  # short gaps between words: sets the silence floor, never a 0.5 s pause
    return db


def test_part_minutes_for():
    from uadub.parts import part_minutes_for

    assert part_minutes_for(44 * 60, None) == 0 and part_minutes_for(46 * 60, None) == 15.0
    assert part_minutes_for(20 * 60, 5) == 5 and part_minutes_for(20 * 60, 15) == 0
    assert part_minutes_for(5 * 3600, 0) == 0


def test_find_cuts_prefers_longest_pause_and_falls_back():
    from uadub.parts import FRAME, find_cuts

    total = 3600.0
    db = _speechy_db(total)

    def quiet(a, b):
        db[int(round(a / FRAME)):int(round(b / FRAME))] = -70.0

    quiet(800, 800.6)  # inside the first window but shorter
    quiet(950, 952)  # the longest pause near 900 s
    quiet(1700, 1701)  # inside the second window (951 + 900 ± 180)
    cuts = find_cuts(db, total, 900.0)
    # a word gap right after a pause may extend it by one 50 ms frame
    assert abs(cuts[0]["time"] - 951.0) < 0.06 and abs(cuts[0]["pause"] - 2.0) < 0.06 and not cuts[0]["fallback"]
    assert abs(cuts[1]["time"] - 1700.5) < 0.06 and not cuts[1]["fallback"]
    assert len(cuts) == 3 and cuts[2]["fallback"]  # no pause near 2600 s: quietest window, flagged
    assert abs(cuts[2]["time"] - 2600.5) < 1.0  # equal loudness everywhere → nearest the target


def test_find_cuts_merges_a_short_tail():
    from uadub.parts import find_cuts

    assert len(find_cuts(_speechy_db(2000.0), 2000.0, 900.0)) == 1  # 900 + 1100, not 900 + 900 + 200
    assert find_cuts(_speechy_db(1100.0), 1100.0, 900.0) == []


def test_frame_db_pads_unknown_tail_with_nan(tmp_path):
    import subprocess

    import numpy as np

    from uadub.parts import frame_db

    src = tmp_path / "a.m4a"
    subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-y", "-f", "lavfi", "-i",
                    "sine=frequency=440:sample_rate=48000:duration=3",
                    "-af", "volume=enable='between(t,1,2)':volume=0", "-c:a", "aac", str(src)], check=True)
    db = frame_db(src, 3.0)
    assert len(db) == 60 and np.nanmax(db[24:36]) < np.nanmin(db[2:16]) - 30
    longer = frame_db(src, 4.0)  # e.g. the decoder died early on a damaged track
    assert len(longer) == 80 and np.isnan(longer[-15:]).all()


def test_plan_is_sample_aligned_and_reused(tmp_path, monkeypatch):
    from uadub import parts

    calls = []

    def fake_db(video, total, frame=parts.FRAME):
        calls.append(1)
        return _speechy_db(total, frame)

    monkeypatch.setattr(parts, "frame_db", fake_db)
    monkeypatch.setattr(parts.A, "duration", lambda p: 2000.123)
    plan = parts.load_or_make_plan(tmp_path, tmp_path / "v.mp4", 15.0, 44100, log=lambda m: None)
    assert len(plan["parts"]) == 2 and plan["parts"][-1]["end"] == 2000.123
    assert plan["parts"][0]["end"] == plan["parts"][1]["start"]
    assert all(abs(p["start"] * 44100 - round(p["start"] * 44100)) < 1e-6 for p in plan["parts"])
    assert plan["parts"][-1]["pause"] is None and "частин" in parts.format_plan(plan)
    parts.load_or_make_plan(tmp_path, tmp_path / "v.mp4", 15.0, 44100, log=lambda m: None)
    assert len(calls) == 1  # same input and settings: plan reused, part caches stay valid
    parts.load_or_make_plan(tmp_path, tmp_path / "v.mp4", 10.0, 44100, log=lambda m: None)
    assert len(calls) == 2


def test_frame_db_fails_when_nothing_decodes(tmp_path):
    import pytest

    from uadub.parts import frame_db

    junk = tmp_path / "x.mp4"
    junk.write_text("not media")
    with pytest.raises(SystemExit) as e:
        frame_db(junk, 3.0)
    assert "x.mp4" in str(e.value)


def test_corrupt_plan_json_is_replanned(tmp_path, monkeypatch):
    from uadub import parts

    monkeypatch.setattr(parts, "frame_db", lambda v, t, frame=parts.FRAME: _speechy_db(t, frame))
    monkeypatch.setattr(parts.A, "duration", lambda p: 2000.0)
    (tmp_path / "plan.json").write_text('{"input": "tru')
    plan = parts.load_or_make_plan(tmp_path, tmp_path / "v.mp4", 15.0, 44100, log=lambda m: None)
    assert len(plan["parts"]) == 2


def test_long_brief_map_reduce_and_fallback():
    import json as _json

    from uadub import translate as T

    units = [{"text": "a" * 100} for _ in range(30)]
    chunks = T.chunk_units(units, limit=1000)
    assert [len(c) for c in chunks] == [9, 9, 9, 3]

    class FakeLLM:
        def __init__(self, merge_ok):
            self.merge_ok, self.briefs = merge_ok, 0

        def chat(self, system, user, *, max_tokens=4096, temperature=0.3):
            if system.startswith("You merge partial"):
                return _json.dumps({"summary": "Усе відео.", "domain": "IT",
                                    "glossary": [{"src": "deploy", "uk": "деплой"}]}) if self.merge_ok else "oops"
            self.briefs += 1
            return _json.dumps({
                "summary": f"Частина {self.briefs}.", "domain": "IT", "address": "ви",
                "speaker_gender": "female" if self.briefs == 2 else "male",
                "glossary": [{"src": "deploy", "uk": "деплой" if self.briefs == 1 else "розгортання"},
                             {"src": f"t{self.briefs}", "uk": "x"}],
                "characters": [{"name": "Bob", "uk": "Боб", "gender": "male"}], "idioms": [], "asr_fixes": []})

    llm = FakeLLM(True)
    brief = T.long_brief(llm, units, limit=1000, log=lambda m: None)
    assert llm.briefs == 4 and brief["summary"] == "Усе відео."
    brief = T.long_brief(FakeLLM(False), units, limit=1000, log=lambda m: None)  # merge reply is broken
    assert brief["speaker_gender"] == "male" and brief["address"] == "ви" and brief["domain"] == "IT"
    assert [g["uk"] for g in brief["glossary"] if g["src"] == "deploy"] == ["деплой"]
    assert len(brief["characters"]) == 1 and brief["summary"].startswith("Частина 1.")
    single = T.long_brief(FakeLLM(True), units[:3], limit=1000, log=lambda m: None)
    assert single["summary"] == "Частина 1."  # one chunk: no merge call
    forced = T.long_brief(FakeLLM(False), units, domain="медицина", limit=1000, log=lambda m: None)
    assert forced["domain"] == "медицина"

    class CrashingMerge(FakeLLM):  # e.g. a ValueError or a timeout from the backend, not a RuntimeError
        def chat(self, system, user, **kw):
            if system.startswith("You merge partial"):
                raise TimeoutError("backend stalled")
            return super().chat(system, user, **kw)

    logs = []
    brief = T.long_brief(CrashingMerge(True), units, limit=1000, log=logs.append)
    assert brief["address"] == "ви" and any("за правилами" in m for m in logs)
    assert [m for m in logs if "шматок" in m] == [f"   • бриф: шматок {i}/4" for i in range(1, 5)]

    class GreedyMerge(FakeLLM):  # the merge ignores the glossary limit
        def chat(self, system, user, **kw):
            if system.startswith("You merge partial"):
                return _json.dumps({"summary": "S", "glossary": [{"src": f"t{i}", "uk": "x"} for i in range(60)]})
            return super().chat(system, user, **kw)

    assert len(T.long_brief(GreedyMerge(True), units, limit=1000, log=lambda m: None)["glossary"]) == 25
    assert len(T.long_brief(GreedyMerge(True), units, domain="IT", limit=1000, log=lambda m: None)["glossary"]) == 40
    from uadub.parts import brief_summary

    assert brief_summary({"domain": "IT", "glossary": [{}, {}]}) == "   • бриф готовий: сфера «IT», термінів у глосарії: 2"


def test_context_window_adds_edges():
    from uadub.translate import _context_window

    units = [{"text": f"line {i}"} for i in range(3)]
    assert _context_window(units, 0, 3) == "» line 0\n» line 1\n» line 2"
    text = _context_window(units, 0, 3, {"before": "PREV", "after": "NEXT"})
    assert text.startswith("PREV\n") and text.endswith("\nNEXT")


def test_translate_units_uses_shared_brief_and_edges():
    import json as _json

    from uadub.translate import translate_units

    seen = []

    class FakeLLM:
        def chat(self, system, user, *, max_tokens=4096, temperature=0.3):
            seen.append((system, user))
            if system.startswith("You are an expert audiovisual translator"):
                ids = [l["id"] for l in _json.loads(user)["lines"]]
                return _json.dumps({"lines": [{"id": i, "uk": "Робимо деплой."} for i in ids]})
            return _json.dumps({"lines": []})

    units = [{"id": 1, "text": "Let's deploy.", "start": 0.0, "end": 2.0, "slot_end": 2.0}]
    translate_units(FakeLLM(), units, rate=6.0, gender="male", glossary_path=None, stress="off",
                    shared_brief={"summary": "S", "glossary": [{"src": "deploy", "uk": "деплой"}]},
                    edge={"before": "PREVIOUS PART", "after": ""}, log=lambda m: None)
    assert not any(s.startswith("You prepare a translation brief") for s, _ in seen)
    system, user = next(x for x in seen if x[0].startswith("You are an expert audiovisual translator"))
    assert "deploy → деплой" in system and "PREVIOUS PART" in user


def test_assemble_concatenates_parts_in_sync(tmp_path):
    import json as _json
    import subprocess

    import numpy as np

    from uadub import audio as A
    from uadub.parts import assemble

    video = _test_video(tmp_path / "v.mp4", seconds=6)
    items = []
    for n, (start, end) in enumerate([(0.0, 2.5), (2.5, 6.0)], 1):
        w = tmp_path / f"p{n}"
        w.mkdir()
        A.write_flac(w / "dub.flac", np.full((round((end - start) * 44100), 2), 0.1, np.float32), 44100)
        (w / "units.json").write_text(_json.dumps(
            [{"id": 1, "start": 0.5, "end": 1.5, "slot_end": 2.0, "text": f"Hi {n}", "uk": f"Привіт {n}"}]))
        items.append((w, start))
    out = tmp_path / "v.uk.mp4"
    assemble(video, items, out, tmp_path / "master", src_lang="en", keep_original=True)
    assert abs(A.duration(out) - 6.0) < 0.1 and A.has_stream(out, "video")
    a = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "a:0", "-show_entries", "stream=duration",
                        "-of", "csv=p=0", str(out)], capture_output=True, text=True).stdout
    assert abs(float(a) - 6.0) < 0.06
    srt = out.with_suffix(".srt").read_text(encoding="utf-8")
    assert "00:00:03,000" in srt and "Привіт 2" in srt  # second part's line shifted by 2.5 s
    units = _json.loads((tmp_path / "master" / "units.json").read_text())
    assert [u["id"] for u in units] == [1, 2] and units[1]["start"] == 3.0


def _long_fixture(tmp_path, monkeypatch, fail):
    import json as _json

    from uadub import cli, parts
    from uadub.config import Options

    opt = Options(input=str(tmp_path / "long.mp4"), output=str(tmp_path / "long.uk.mp4"),
                  workdir=str(tmp_path / "long.uadub"), voice="st")
    plan = {"duration": 30.0, "sample_rate": 44100, "part_minutes": 0.2, "parts": [
        {"n": 1, "start": 0.0, "end": 10.0, "pause": 1.0, "fallback": False},
        {"n": 2, "start": 10.0, "end": 20.0, "pause": 0.8, "fallback": False},
        {"n": 3, "start": 20.0, "end": 30.0, "pause": None, "fallback": False}]}
    calls, assembled = [], []

    def fake_run(po, *, redo=None, stop_after=None, review=False, review_with=None, text=False,
                 prefix="", summary=True):
        n = int(po.work.name)
        calls.append((n, stop_after, po.shared_brief, po.loudness_target))
        po.work.mkdir(parents=True, exist_ok=True)
        (po.work / "transcript.json").write_text(_json.dumps([{"start": 0, "end": 1, "text": f"part {n}"}]))
        (po.work / "meta.json").write_text(_json.dumps({"duration": 10.0}))
        if stop_after == "asr":
            return
        if n in fail:
            raise SystemExit("\nЕтап «tts» завершився з помилкою (код 1).")
        if not (po.work / "dub.flac").exists():
            (po.work / "dub.flac").write_bytes(b"x")
            (po.work / "units.json").write_text("[]")

    def fake_assemble(video, items, out, master, **kw):
        assembled.append([n for n, _ in enumerate(items, 1)])
        Path(out).write_bytes(b"video")

    monkeypatch.setattr(cli, "run_pipeline", fake_run)
    monkeypatch.setattr(parts, "build_shared_brief",
                        lambda master, works: (master / "long_brief.json").write_text("{}"))
    monkeypatch.setattr(parts, "speech_loudness", lambda w, sr: -20.0)
    monkeypatch.setattr(parts, "assemble", fake_assemble)
    return opt, plan, calls, assembled


def test_run_long_isolates_failures_and_resumes(tmp_path, monkeypatch):
    import pytest

    from uadub import parts

    fail = {2}
    opt, plan, calls, assembled = _long_fixture(tmp_path, monkeypatch, fail)
    with pytest.raises(SystemExit) as e:
        parts.run_long(opt, plan)
    assert "02" in str(e.value) and not assembled
    assert [c[0] for c in calls if c[1] == "asr"] == [1, 2, 3]  # phase 1 for all parts first
    phase3 = [c for c in calls if c[1] is None]
    assert [c[0] for c in phase3] == [1, 2, 3]  # part 3 still made after part 2 failed
    assert all(c[2] and c[2].endswith("long_brief.json") and c[3] == -20.0 for c in phase3)
    assert (tmp_path / "long.uadub" / "edge" / "02.json").exists()
    import json as _json

    state = _json.loads((tmp_path / "long.uadub" / "state.json").read_text())
    assert list(state["failed"]) == ["2"] and "tts" in state["failed"]["2"]
    fail.clear()
    calls.clear()
    parts.run_long(opt, plan)
    assert assembled == [[1, 2, 3]]
    assert not _json.loads((tmp_path / "long.uadub" / "state.json").read_text())["failed"]


def test_run_long_reassembles_when_a_part_changes(tmp_path, monkeypatch):
    import os

    from uadub import parts

    opt, plan, calls, assembled = _long_fixture(tmp_path, monkeypatch, set())
    parts.run_long(opt, plan)
    parts.run_long(opt, plan)
    assert len(assembled) == 1  # nothing changed: no second assembly
    dub = tmp_path / "long.uadub" / "parts" / "02" / "dub.flac"
    dub.write_bytes(b"re-dubbed after a review.md edit")
    os.utime(dub, ns=(dub.stat().st_atime_ns, dub.stat().st_mtime_ns + 10**9))
    parts.run_long(opt, plan)
    assert len(assembled) == 2
    assert not (tmp_path / "long.uk.parts").exists()  # previews removed without --keep-parts


def test_run_long_reassembles_when_keep_original_changes(tmp_path, monkeypatch):
    from dataclasses import replace

    from uadub import parts

    opt, plan, calls, assembled = _long_fixture(tmp_path, monkeypatch, set())
    parts.run_long(opt, plan)
    parts.run_long(replace(opt, keep_original=not opt.keep_original), plan)
    assert len(assembled) == 2


def test_run_long_user_stop_ends_the_run(tmp_path, monkeypatch):
    import pytest

    from uadub import cli, parts

    opt, plan, calls, assembled = _long_fixture(tmp_path, monkeypatch, set())

    def stopper(po, **kw):
        calls.append(int(po.work.name))
        raise cli.UserStop("Зупинено.")

    monkeypatch.setattr(cli, "run_pipeline", stopper)
    with pytest.raises(cli.UserStop):
        parts.run_long(opt, plan)
    assert calls == [1]


def test_run_long_loudness_failure_is_a_part_failure_and_early_stop(tmp_path, monkeypatch):
    import pytest

    from uadub import parts

    opt, plan, calls, assembled = _long_fixture(tmp_path, monkeypatch, set())

    def bad(w, sr):
        if w.name == "02":
            raise RuntimeError("broken stems")
        return -20.0

    monkeypatch.setattr(parts, "speech_loudness", bad)
    with pytest.raises(SystemExit) as e:
        parts.run_long(opt, plan)
    assert "02" in str(e.value) and "broken stems" in str(e.value)
    calls.clear()
    monkeypatch.setattr(parts, "speech_loudness", lambda w, sr: -20.0)
    parts.run_long(opt, plan, stop_after="extract")
    assert {c[1] for c in calls} == {"extract"}


def test_no_separate_after_part_cleanup_extracts_audio_again(tmp_path):
    import numpy as np

    from uadub import audio as A
    from uadub.config import Options
    from uadub.stages import _cleanup_part, stage_separate

    src = _sine_m4a(tmp_path / "src.m4a")
    w = tmp_path / "part"
    w.mkdir()
    for name in ("audio.wav", "vocals.wav", "background.wav", "mix.wav"):
        A.write(w / name, np.zeros((100, 2), np.float32), 44100)
    _cleanup_part(w)  # separated part: audio.wav is gone, stems are FLAC
    assert not A.existing(w / "audio.wav").exists()
    opt = Options(input=str(src), output=str(tmp_path / "o.m4a"), workdir=str(w), separate=False,
                  clip_start=1.0, clip_end=2.5)
    stage_separate(opt)  # the user reruns the part with --no-separate
    assert (w / "audio.wav").exists() and not A.existing(w / "vocals.wav").exists()
    assert A.read(w / "audio.wav")[0].shape == (66150, 2)


def test_damaged_part_goes_on_but_whole_file_stops(tmp_path, monkeypatch):
    import subprocess

    import numpy as np
    import pytest
    import soundfile as sf

    from uadub import audio as A

    def fake_run(cmd, **kw):
        return subprocess.CompletedProcess(cmd, 1, "", f"{A._DECODE_ERROR}\n" * 3)

    monkeypatch.setattr(A.subprocess, "run", fake_run)
    monkeypatch.setattr(A, "_salvage_audio",
                        lambda v, sr, start=0.0, length=None: (np.zeros((round((length or 3.0) * sr), 2), np.float32), 7))
    A.extract_audio(tmp_path / "v.mp4", tmp_path / "part.wav", 1000, start=5.0, length=2.0)
    assert sf.info(str(tmp_path / "part.wav")).frames == 2000  # a silent part, the long run goes on
    with pytest.raises(SystemExit):
        A.extract_audio(tmp_path / "v.mp4", tmp_path / "whole.wav", 1000)


def test_choose_part_minutes_keeps_whole_runs_and_subs(tmp_path):
    import json as _json

    import pytest

    from uadub.config import Options
    from uadub.parts import choose_part_minutes

    notes = []
    base = dict(input="v.mp4", output="v.uk.mp4", workdir=str(tmp_path / "w"))
    long = 60 * 60
    assert choose_part_minutes(Options(**base), long, log=notes.append) == 15.0 and not notes
    assert choose_part_minutes(Options(**base), 30 * 60, log=notes.append) == 0
    # --subs: automatic split falls back to the whole pipeline; an explicit split is an error
    assert choose_part_minutes(Options(**base, subs="v.srt"), long, log=notes.append) == 0
    assert len(notes) == 1 and "--subs" in notes[0]
    with pytest.raises(SystemExit):
        choose_part_minutes(Options(**base, subs="v.srt", part_minutes=10), long, log=notes.append)
    # a work folder of an earlier whole run stays whole (its caches are kept)
    (tmp_path / "w").mkdir()
    (tmp_path / "w" / "state.json").write_text(_json.dumps({"extract": {}, "separate": {}}))
    notes.clear()
    assert choose_part_minutes(Options(**base), long, log=notes.append) == 0
    assert len(notes) == 1 and "--part-minutes 15" in notes[0]
    assert choose_part_minutes(Options(**base, part_minutes=15), long, log=notes.append) == 15.0
    (tmp_path / "w" / "plan.json").write_text("{}")  # an earlier long run: stays in part mode
    assert choose_part_minutes(Options(**base), long, log=notes.append) == 15.0


def test_parts_word_plural():
    from uadub.parts import parts_word

    assert [parts_word(n) for n in (1, 2, 4, 5, 11, 12, 14, 21, 22, 25, 111)] == [
        "1 частина", "2 частини", "4 частини", "5 частин", "11 частин", "12 частин", "14 частин",
        "21 частина", "22 частини", "25 частин", "111 частин"]


def test_non_object_plan_json_is_replanned(tmp_path, monkeypatch):
    from uadub import parts

    monkeypatch.setattr(parts, "frame_db", lambda v, t, frame=parts.FRAME: _speechy_db(t, frame))
    monkeypatch.setattr(parts.A, "duration", lambda p: 2000.0)
    (tmp_path / "plan.json").write_text("[1, 2]")
    plan = parts.load_or_make_plan(tmp_path, tmp_path / "v.mp4", 15.0, 44100, log=lambda m: None)
    assert len(plan["parts"]) == 2 and "2 частини" in parts.format_plan(plan)


def test_run_long_stop_after_mux_assembles(tmp_path, monkeypatch):
    from uadub import parts

    opt, plan, calls, assembled = _long_fixture(tmp_path, monkeypatch, set())
    parts.run_long(opt, plan, stop_after="mux")
    assert assembled == [[1, 2, 3]]


def test_run_long_remakes_a_missing_part_dub(tmp_path, monkeypatch):
    from uadub import parts

    opt, plan, calls, assembled = _long_fixture(tmp_path, monkeypatch, set())
    parts.run_long(opt, plan)
    from uadub import cli

    redos = []
    fake = cli.run_pipeline
    real_dub = tmp_path / "long.uadub" / "parts" / "02" / "dub.flac"

    def no_dub(po, **kw):  # phase 3 sees the state "done" and does not recreate the file
        redos.append((int(po.work.name), kw.get("redo")))
        if kw.get("redo") == "mix":
            fake(po, **kw)

    monkeypatch.setattr(cli, "run_pipeline", no_dub)
    real_dub.unlink()
    parts.run_long(opt, plan)
    assert (2, "mix") in redos and real_dub.exists() and len(assembled) == 2


def test_brief_child_reports_progress(tmp_path, monkeypatch, capsys):
    import json as _json

    from uadub import llm as L
    from uadub import parts
    from uadub.config import Options

    class FakeLLM:
        def chat(self, system, user, **kw):
            return _json.dumps({"summary": "S", "domain": "IT", "glossary": [{"src": "a", "uk": "б"}]})

        def close(self):
            pass

    monkeypatch.setattr(L, "make_llm", lambda spec: FakeLLM())
    Options(input="v.mp4", output="v.uk.mp4", workdir=str(tmp_path)).save(tmp_path / "options.json")
    w = tmp_path / "parts" / "01"
    w.mkdir(parents=True)
    (w / "transcript.json").write_text(_json.dumps([{"text": "Hello."}]))
    parts._brief_child(str(tmp_path), [str(w)])
    out = [line for line in capsys.readouterr().out.splitlines()]
    assert all(line.strip() for line in out)  # no stray blank lines
    assert "   • бриф: шматок 1/1" in out and out[-1] == "   • бриф готовий: сфера «IT», термінів у глосарії: 1"
    assert _json.loads((tmp_path / "long_brief.json").read_text())["domain"] == "IT"


def test_place_clips_per_line_caps():
    # A line already at the pace ceiling gets its own, lower cap; below `min_stretch` it is left alone.
    units = [{"start": 0.0, "end": 2.0, "slot_end": 2.5}, {"start": 2.6, "end": 4.0, "slot_end": 5.0}]
    plan = place_clips(units, [3.5, 1.0], max_speed=1.25, spill=0.5, caps=[1.04, 1.25])
    assert plan[0]["speed"] == 1.0 and plan[1]["start"] > 2.6
    plan = place_clips(units, [3.5, 1.0], max_speed=1.25, spill=0.5, caps=[1.1, 1.25])
    assert abs(plan[0]["speed"] - 1.1) < 1e-6


def test_pace_cap():
    from uadub.fit import pace_cap

    assert pace_cap(2.0, 10, 1.15) == 1.15  # 5 syl/s: room up to 7.5
    assert abs(pace_cap(2.0, 14, 1.15) - 7.5 * 2.0 / 14) < 1e-6  # 7 syl/s: only ~1.07 left
    assert pace_cap(2.0, 20, 1.15) == 1.0  # already past the ceiling
    assert pace_cap(2.0, 0, 1.15) == 1.15  # no syllables: engine cap only


def test_place_clips_keeps_a_breath_between_lines():
    # When a line overruns, the next one starts after a natural pause, not glued at 50 ms.
    units = [{"start": 0.0, "end": 2.0, "slot_end": 2.5}, {"start": 2.6, "end": 4.0, "slot_end": 5.0}]
    plan = place_clips(units, [2.8, 1.0], max_speed=1.25)
    assert abs(plan[1]["start"] - (2.8 + 0.25)) < 1e-6


def test_budgets_reserve_a_pause_and_assume_a_natural_pace():
    from uadub.translate import set_budgets

    units = [{"start": 0.0, "end": 2.0, "slot_end": 2.8}, {"start": 3.0, "end": 4.0, "slot_end": 4.1}]
    set_budgets(units, rate=4.8, max_speed=1.25)
    # (2.8 − 0.3 s pause) × 4.8 syl/s × 1.1 = 13.2 → 13; the old formula gave 2.8 × 4.8 × 1.2 = 16
    assert units[0]["max_syl"] == 13
    # the reserve never eats into the spoken part of the line: max(1.0, 1.1 − 0.3) × 4.8 × 1.1 = 5
    assert units[1]["max_syl"] == 5


def test_version_numbers_after_a_latin_name_are_read_digit_by_digit():
    assert to_speech_text("модель Qwen 3.6.") == "модель квен три шість."
    assert to_speech_text("Python 3.12 вийшов") == "пайтон три дванадцять вийшов"
    assert "один кома п'ять" in to_speech_text("версія 1.5")  # a plain decimal keeps the comma


def test_match_spectrum_moves_the_dub_towards_the_reference():
    import numpy as np
    from uadub.audio import match_spectrum

    rng = np.random.default_rng(0)
    sr = 24000
    white = rng.standard_normal(sr * 4).astype(np.float32)
    bright = np.diff(white, prepend=0.0).astype(np.float32)  # reference: tilted up
    dull = np.convolve(white, np.ones(8) / 8, mode="same").astype(np.float32)  # dub: tilted down

    def tilt(y):
        spec = np.abs(np.fft.rfft(y)) ** 2
        f = np.fft.rfftfreq(len(y), 1 / sr)
        lo, hi = spec[(f > 200) & (f < 800)].mean(), spec[(f > 3000) & (f < 6000)].mean()
        return 10 * np.log10(hi / lo)

    before = abs(tilt(dull) - tilt(bright))  # ~33 dB apart
    out = match_spectrum(dull, bright, sr, max_db=40.0)
    assert len(out) == len(dull)
    assert abs(tilt(out) - tilt(bright)) < 3.0  # unclamped: the tilt is matched
    # the default ±6 dB clamp moves the tilt by about 12 dB and no more
    after = abs(tilt(match_spectrum(dull, bright, sr)) - tilt(bright))
    assert before - 14 < after < before - 8


def test_st_engine_speed_compensates_the_nonlinear_response():
    # Measured: StyleTTS2 `speed` 1.1 shortens a line by ~6–8 %, 1.2 by ~17 %, 1.3 by ~25 %; 1.4 jumps.
    from uadub.fit import st_engine_speed

    assert st_engine_speed(1.0) == 1.0
    assert abs(st_engine_speed(1.08) - 1.1) < 1e-6
    assert abs(st_engine_speed(1.16) - 1.2) < 1e-6
    assert st_engine_speed(1.5) == 1.35  # never into the region where the engine garbles


def test_unify_pronunciations_makes_a_name_sound_the_same_in_every_line():
    from uadub.translate import unify_pronunciations

    units = [
        {"id": 1, "uk": "Для LM Studio опцій немає.", "tts": "Для ел ем студіо опцій немає."},
        {"id": 2, "uk": "Відкриється LM Studio з посиланням.", "tts": "Відкриється ел ем студіо з посиланням."},
        {"id": 3, "uk": "В LM Studio ми оберемо варіант.", "tts": "В ель ем студіо ми оберемо варіант."},
        {"id": 4, "uk": "Прокрутіть до LM Studio, і це знову відкриє.", "tts": "Прокрутіть до ел-ем студіо, і це знову відкриє."},
        {"id": 5, "uk": "Той самий список із Hugging Face.", "tts": "Той самий список із хаґінґ фейс."},
        {"id": 6, "uk": "Без латинки.", "tts": None},
    ]
    changed = unify_pronunciations(units)
    # «LM» is an acronym: StyleTTS2 says our hyphenated letter spelling clearly, the LLM's loose
    # «ел ем» / «ель ем» is slurred. So the term takes textnorm's form in every line.
    assert changed == 3
    assert units[0]["tts"] == "Для ел-ем студіо опцій немає."
    assert units[2]["tts"] == "В ел-ем студіо ми оберемо варіант."
    assert units[3]["tts"] == "Прокрутіть до ел-ем студіо, і це знову відкриє."
    assert units[4]["tts"] == "Той самий список із хаґінґ фейс."  # a plain name seen once is left alone


def test_unify_pronunciations_majority_for_plain_names():
    from uadub.translate import unify_pronunciations

    units = [
        {"id": 1, "uk": "Відкрийте Hugging Face.", "tts": "Відкрийте хаґінґ фейс."},
        {"id": 2, "uk": "На Hugging Face є список.", "tts": "На хагінг фейс є список."},
        {"id": 3, "uk": "Знову Hugging Face.", "tts": "Знову хаґінґ фейс."},
    ]
    assert unify_pronunciations(units) == 1
    assert units[1]["tts"] == "На хаґінґ фейс є список."


def test_budgets_also_give_a_word_count():
    # LLMs count words far better than syllables: ~2.4 syllables per Ukrainian word.
    from uadub.translate import set_budgets

    units = [{"start": 0.0, "end": 5.0, "slot_end": 5.8}]
    set_budgets(units, rate=4.8, max_speed=1.25)
    assert units[0]["max_syl"] == 29
    assert units[0]["max_words"] == 12


def test_lost_facts_flags_dropped_names_numbers_and_labels():
    from uadub.translate import lost_facts

    old = "Натисніть «Використати цю модель», щоб завантажити Qwen 3.6 з Hugging Face."
    assert lost_facts(old, "Натисніть кнопку, щоб завантажити Qwen 3.6 з Hugging Face.") == ["«Використати цю модель»"]
    assert lost_facts(old, "Натисніть «Використати цю модель» і завантажте Qwen з Hugging Face.") == ["3.6"]
    assert lost_facts(old, "Натисніть «Використати цю модель», щоб завантажити Qwen 3.6.") == ["Hugging Face"]
    assert lost_facts(old, "Натисніть «Використати цю модель» — і Qwen 3.6 завантажиться з Hugging Face.") == []
    # case and quote style do not matter
    assert lost_facts("Кнопка «Завантажити» тут.", "Кнопка „завантажити“ — тут.") == []


def test_unify_pronunciations_prefers_the_brief():
    from uadub.translate import unify_pronunciations

    units = [
        {"id": 1, "uk": "Відкрийте Hugging Face.", "tts": "Відкрийте хагінг фейс."},
        {"id": 2, "uk": "На Hugging Face є список.", "tts": "На хагінг фейс є список."},
    ]
    assert unify_pronunciations(units, {"Hugging Face": "хаґінґ фейс"}) == 2
    assert units[0]["tts"] == "Відкрийте хаґінґ фейс."
    assert units[1]["tts"] == "На хаґінґ фейс є список."


def test_glossary_pronunciations_from_the_brief():
    from uadub.translate import glossary_pronunciations

    gl = [{"src": "LM Studio", "uk": "LM Studio", "say": "ел-ем студіо"},
          {"src": "quantization", "uk": "квантування"},
          {"src": "Hugging Face", "uk": "Hugging Face", "say": "Hugging Face"},  # not Cyrillic: ignored
          {"src": "GitHub", "uk": "GitHub", "say": "ґіт+хаб"}]  # stress marks are stripped
    assert glossary_pronunciations(gl) == {"LM Studio": "ел-ем студіо", "GitHub": "ґітхаб"}


def test_unify_pronunciations_two_names_in_one_line():
    # Replacing the first name must not shift the span of the second one («еем-ел-екс» bug).
    from uadub.translate import unify_pronunciations

    units = [
        {"id": 1, "uk": "Оберіть LM Studio чи MLX.", "tts": "Оберіть ель ем студіо чи ем-ел-ікс."},
        {"id": 2, "uk": "Знову LM Studio.", "tts": "Знову ел-ем студіо."},
    ]
    unify_pronunciations(units)
    assert units[0]["tts"] == "Оберіть ел-ем студіо чи ем-ел-екс."


def test_clean_tts_rejects_garbage_letters():
    from uadub.translate import _clean_tts

    assert _clean_tts("файл завеликий", "файл завеликий через відсутність відеокар粹карти.") == ""
    assert _clean_tts("LM Studio тут", "ел-ем студіо тут") == "ел-ем студіо тут"
