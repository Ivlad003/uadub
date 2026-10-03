from uadub.fit import place_clips, target_duration
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
    plan = place_clips(units, [2.4, 3.6], max_speed=1.25)
    assert plan[0]["speed"] == 1.0  # fits
    assert abs(plan[1]["speed"] - 1.25) < 1e-6  # needs 1.5x, capped
    assert plan[1]["start"] == 2.6
    plan = place_clips(units, [3.5, 1.0], max_speed=1.25)
    assert plan[0]["speed"] == 1.25 and plan[1]["start"] > 2.6  # spill pushes next line


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
