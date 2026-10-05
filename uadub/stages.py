"""Pipeline stages. Each reads/writes files in the work directory, so runs are resumable
and any intermediate (e.g. units.json with the translation) can be edited by hand."""

from __future__ import annotations

import json
import logging
import shutil
from pathlib import Path

import numpy as np

from . import audio as A
from .config import CACHE_DIR, SYLLABLE_RATE, Options
from .fit import MAX_RATE, MIN_RATE, SPILL, pace_cap, place_clips, st_speed, target_duration
from .segments import assign_slots, build_units, merge_tokens_to_words, slot
from .srt import make_cues, write_srt


def log(msg: str) -> None:
    print(msg, flush=True)


def load_json(work: Path, name: str):
    return json.loads((work / name).read_text(encoding="utf-8"))


def _voice_src(w: Path) -> Path:
    """Separated speech if there is one (WAV, or FLAC after part cleanup), else the full audio."""
    v = A.existing(w / "vocals.wav")
    return v if v.exists() else A.existing(w / "audio.wav")


def save_json(work: Path, name: str, data) -> None:
    (work / name).write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


# ---------------------------------------------------------------------------
def stage_extract(opt: Options) -> None:
    w = opt.work
    if not A.has_stream(opt.input, "audio"):
        raise SystemExit("У файлі немає аудіодоріжки — нічого перекладати.")
    if opt.clip_start is not None:
        A.extract_audio(opt.input, w / "audio.wav", opt.sample_rate, start=opt.clip_start,
                        length=opt.clip_end - opt.clip_start)
    else:
        A.extract_audio(opt.input, w / "audio.wav", opt.sample_rate)
    save_json(w, "meta.json", {"duration": A.duration(w / "audio.wav")})


# ---------------------------------------------------------------------------
def stage_separate(opt: Options) -> None:
    w = opt.work
    for name in ("vocals.wav", "background.wav", "vocals.flac", "background.flac"):
        (w / name).unlink(missing_ok=True)
    # Part cleanup removes audio.wav once stems exist: decode the part again. The separator needs the
    # WAV; --no-separate only needs some copy (mix and asr also read the FLAC).
    if not (w / "audio.wav").exists() and (opt.separate or not (w / "audio.flac").exists()):
        stage_extract(opt)
    if not opt.separate:
        log("   • пропущено (режим закадрового перекладу поверх оригіналу)")
        return
    from audio_separator.separator import Separator

    out_dir = w / "sep"
    shutil.rmtree(out_dir, ignore_errors=True)
    out_dir.mkdir()
    sep = Separator(output_dir=str(out_dir), model_file_dir=str(CACHE_DIR / "separator"),
                    output_format="WAV", log_level=logging.WARNING)
    sep.load_model(model_filename=opt.sep_model)
    produced = [Path(p) if Path(p).is_absolute() else out_dir / Path(p).name for p in sep.separate(str(w / "audio.wav"))]

    def pick(*keys: str) -> Path | None:
        for p in produced:
            name = p.name.lower()
            if any(k in name for k in keys):
                return p
        return None

    vocals = pick("(vocals)", "_vocals")
    if vocals is None:
        raise RuntimeError(f"Модель сепарації не повернула вокал: {[p.name for p in produced]}")
    instrumental = pick("(instrumental)", "(no vocals)", "no_vocals")  # 4-stem models → mix − vocals
    mix, sr = A.read(w / "audio.wav")
    voc, _ = A.read(vocals, sr=sr)
    n = min(len(mix), len(voc))
    if instrumental is not None:
        bg, _ = A.read(instrumental, sr=sr)
        bg = np.pad(bg[:n], ((0, max(0, n - len(bg))), (0, 0)))
    else:
        bg = mix[:n] - voc[:n]
    A.write(w / "vocals.wav", voc[:n], sr)
    A.write(w / "background.wav", bg[:n], sr)
    shutil.rmtree(out_dir, ignore_errors=True)


# ---------------------------------------------------------------------------
# Typical Whisper hallucinations on silence/music (dropped when they come out of nowhere).
_HALLUCINATIONS = ("시청해주셔서 감사합니다", "구독과 좋아요", "MBC 뉴스", "자막 제공", "다음 영상에서",
                   "thanks for watching", "subtitles by", "ご視聴ありがとうございました")


def _whisper_sentences(opt: Options, wav: Path) -> list[dict]:
    import mlx_whisper

    from .segments import words_to_sentences

    r = mlx_whisper.transcribe(str(wav), path_or_hf_repo=opt.asr_model, language=opt.source_lang,
                               word_timestamps=True, condition_on_previous_text=False,
                               hallucination_silence_threshold=2.0, no_speech_threshold=0.6, verbose=None)
    words = []
    for seg in r.get("segments", []):
        text = seg.get("text", "").strip()
        if seg.get("no_speech_prob", 0) > 0.6 and seg.get("avg_logprob", 0) < -1.0:
            continue
        if any(h in text.lower() for h in _HALLUCINATIONS) and len(text) < 40:
            continue
        for wd in seg.get("words", []):
            words.append({"w": wd["word"].strip(), "start": float(wd["start"]), "end": float(wd["end"])})
    return words_to_sentences(words)


def _parakeet_sentences(opt: Options, wav: Path, total: float) -> list[dict]:
    from parakeet_mlx import DecodingConfig, SentenceConfig, from_pretrained

    model = from_pretrained(opt.asr_model)
    cfg = DecodingConfig(sentence=SentenceConfig(max_words=30, silence_gap=0.7, max_duration=10.0))
    res = model.transcribe(str(wav), decoding_config=cfg,
                           chunk_duration=120.0 if total > 150 else None, overlap_duration=15.0)
    sentences = []
    for s in res.sentences:
        text = s.text.strip()
        if not text:
            continue
        toks = [{"text": t.text, "start": float(t.start), "end": float(t.end)} for t in s.tokens]
        sentences.append({"start": round(float(s.start), 3), "end": round(float(s.end), 3),
                          "text": text, "words": merge_tokens_to_words(toks)})
    return sentences


FEMALE_F0 = 165.0  # Hz: typical adult male voices sit below, female above


def _tag_speaker_gender(w: Path, sentences: list[dict]) -> None:
    """Guess each line's speaker gender from voice pitch (used for grammar and for --voice duo)."""
    voice = _voice_src(w)
    y, sr = A.read(voice, sr=16000, mono=True)
    for s in sentences:
        f0 = _median_pitch(y[int(s["start"] * sr) : int(s["end"] * sr)], sr)
        s["pitch"] = round(f0, 1) if f0 else None
        s["gender"] = None if f0 is None else ("female" if f0 >= FEMALE_F0 else "male")


def stage_asr(opt: Options) -> None:
    w = opt.work
    total = load_json(w, "meta.json")["duration"]
    if opt.subs:
        from .subs import read_subs

        sentences = [s for s in read_subs(opt.subs) if s["start"] < total]
        log(f"   • субтитри: {len(sentences)} реплік з {Path(opt.subs).name}")
    else:
        src = _voice_src(w)
        A.to_mono_16k(src, w / "asr16k.wav")
        if "parakeet" in opt.asr_model:
            sentences = _parakeet_sentences(opt, w / "asr16k.wav", total)
        else:
            sentences = _whisper_sentences(opt, w / "asr16k.wav")
        (w / "asr16k.wav").unlink(missing_ok=True)
        log(f"   • розпізнано {len(sentences)} речень, {sum(len(s['words']) for s in sentences)} слів")
    if opt.subs or "parakeet" not in opt.asr_model:
        # Whisper word times and subtitle cues are approximate → snap them to the actual speech
        from .segments import snap_to_speech

        voice = _voice_src(w)
        y, sr = A.read(voice, sr=16000, mono=True)
        log(f"   • уточнено межі {snap_to_speech(sentences, y, sr)} реплік за звуком")
    _tag_speaker_gender(w, sentences)
    save_json(w, "transcript.json", sentences)


# ---------------------------------------------------------------------------
def stage_translate(opt: Options) -> None:
    from .llm import make_llm
    from .translate import translate_units

    w = opt.work
    total = load_json(w, "meta.json")["duration"]
    # Dialogue (films, dramas, subtitles): never glue short lines together — they may belong to
    # different speakers. English monologue-style videos: merge fragments into whole phrases.
    dialogue = opt.source_lang != "en" or bool(opt.subs)
    units = build_units(load_json(w, "transcript.json"), merge_gap=0.0 if dialogue else 0.35)
    assign_slots(units, total, tail=opt.tail)
    if not units:
        log("   • мовлення не знайдено")
        save_json(w, "units.json", [])
        return
    log(f"   • модель: {opt.llm}")
    llm = make_llm(opt.llm)
    try:
        if opt.text_lang == "uk":  # Ukrainian subtitles: nothing to translate, just voice them
            from .translate import resolve_homographs, set_budgets

            set_budgets(units, rate=SYLLABLE_RATE[opt.engine], max_speed=opt.max_speed)
            for u in units:
                u["uk"] = u["text"]
            info = {"homographs": resolve_homographs(llm, units, log=log) if opt.stress == "auto" else {}}
        else:
            shared = json.loads(Path(opt.shared_brief).read_text(encoding="utf-8")) if opt.shared_brief else None
            edge = json.loads(Path(opt.edge_context).read_text(encoding="utf-8")) if opt.edge_context else None
            info = translate_units(llm, units, rate=SYLLABLE_RATE[opt.engine], gender=opt.speaker_gender,
                                   glossary_path=opt.glossary, max_speed=opt.max_speed,
                                   stress=opt.stress, source_lang=opt.text_lang, domain=opt.domain,
                                   shared_brief=shared, edge=edge, log=log)
    finally:
        llm.close()
    save_json(w, "brief.json", info)
    save_json(w, "units.json", units)
    write_srt(make_cues(units, "text"), w / f"{opt.text_lang}.srt")
    write_srt(make_cues(units, "uk"), w / "uk.srt")


# ---------------------------------------------------------------------------
def _median_pitch(y: np.ndarray, sr: int) -> float | None:
    """Median F0 of a voice clip (Hz) — a cheap way to tell e.g. a man from a woman."""
    if len(y) < int(0.3 * sr):
        return None
    try:
        import librosa

        f0 = librosa.yin(y.astype(np.float32), fmin=60, fmax=400, sr=sr, frame_length=2048)
        rms = librosa.feature.rms(y=y, frame_length=2048, hop_length=512)[0][: len(f0)]
        voiced = f0[: len(rms)][rms > rms.max() * 0.2]
        return float(np.median(voiced)) if voiced.size else None
    except Exception:
        return None


def _same_voice(p1: float | None, p2: float | None) -> bool:
    if p1 is None or p2 is None:
        return True
    return abs(np.log2(p1 / p2)) < 0.3  # within ~3.5 semitones


def _clone_window(units: list[dict], i: int, pitch: list | None = None, min_len=4.0, max_len=12.0,
                  max_gap=0.8) -> tuple[int, int]:
    """Neighbouring lines of the same speaker used as the voice reference for line i.

    A neighbour is only borrowed if it is close in time and has a similar voice pitch, so in a
    dialogue the man's line never gets the woman's reference and vice versa.
    """
    pitch = pitch or [None] * len(units)
    a = b = i
    while units[b]["end"] - units[a]["start"] < min_len:
        if b + 1 < len(units) and units[b + 1]["start"] - units[b]["end"] <= max_gap \
                and units[b + 1]["end"] - units[a]["start"] <= max_len and _same_voice(pitch[i], pitch[b + 1]):
            b += 1
        elif a > 0 and units[a]["start"] - units[a - 1]["end"] <= max_gap \
                and units[b]["end"] - units[a - 1]["start"] <= max_len and _same_voice(pitch[i], pitch[a - 1]):
            a -= 1
        else:
            break
    return a, b


def _speakable(text: str) -> bool:
    """A line with at least one letter (ASR sometimes yields lines like «.» or «♪»)."""
    import re

    return bool(re.search(r"[^\W\d_]", text or ""))


def _omni_duration(eng, text: str, prompt, slot_s: float, max_speed: float) -> float:
    """Always give OmniVoice an explicit length.

    Left to itself it estimates the length from the text it receives — including bracketed
    stress transcriptions like «[K V AA0 N T …]», which inflates the estimate and makes the
    speech drawn-out. The estimate is made on the plain text and clamped to a natural pace.
    """
    from .textnorm import syllables

    natural = eng.natural_duration(text, prompt)
    syl = syllables(text)
    if syl:
        natural = min(max(natural, syl / MAX_RATE), syl / MIN_RATE)
    forced = target_duration(natural, slot_s, max_speed)
    return round(forced if forced is not None else natural, 3)


ST_MAX_SPEED = 1.35  # StyleTTS2 still sounds natural up to ~1.3×
MIX_RESTRETCH = 1.15  # engines that already fitted their lines may be stretched only this much more


def _st_fit(eng, text: str, voice: str, plain: str, slot_s: float, max_speed: float, style=None) -> np.ndarray:
    """Voice a line with StyleTTS2, re-synthesising faster (its own `speed`, no time-stretch
    artefacts) when it is drawn-out or too long for the slot; see `st_speed` for the pace band."""
    from .textnorm import syllables

    y = A.trim_silence(eng.synth(text, voice, 1.0, style=style), eng.sr)
    length = len(y) / eng.sr
    if length <= 0:
        return y
    cap = min(ST_MAX_SPEED, max(max_speed, 1.0) + 0.1)
    speed = st_speed(length, syllables(plain), slot_s, max_speed=cap)
    if speed > 1.0:
        y = eng.synth(text, voice, speed, style=style)
    return y


def _emotion_styles(eng, opt: Options, units: list[dict], voices: list[str]) -> list:
    """Per-line StyleTTS2 styles: the chosen voice + the intonation of the original line (--emotion).

    Lines spoken by the opposite sex to the voice keep the plain voice style — a man's voice with
    a woman's pitch movement sounds wrong. Very short lines borrow a neighbour of the same speaker.
    """
    from .config import ST_VOICES

    w = opt.work
    src = _voice_src(w)
    orig, _ = A.read(src, sr=eng.sr, mono=True)
    pitch = [_median_pitch(orig[int(u["start"] * eng.sr) : int(u["end"] * eng.sr)], eng.sr) for u in units]
    styles: list = []
    used = 0
    for i, (u, voice) in enumerate(zip(units, voices)):
        vg, lg = ST_VOICES.get(voice), u.get("gender")
        if vg in ("male", "female") and lg in ("male", "female") and vg != lg:
            styles.append(None)
            continue
        a, b = _clone_window(units, i, pitch, min_len=1.5, max_len=6.0)
        p = eng.prosody(orig[int(units[a]["start"] * eng.sr) : int(units[b]["end"] * eng.sr)], eng.sr)
        styles.append(eng.blend(voice, p, opt.emotion) if p is not None else None)
        used += p is not None
    log(f"   • інтонація оригіналу (--emotion {opt.emotion:g}): {used} з {len(units)} реплік")
    return styles


def _transcribe_reference(path: str) -> str:
    """Transcript of a user-supplied reference voice (parakeet v3 understands Ukrainian too)."""
    from parakeet_mlx import from_pretrained

    from .config import DEFAULT_ASR

    return from_pretrained(DEFAULT_ASR).transcribe(path).text.strip()


def stage_tts(opt: Options) -> None:
    from tqdm import tqdm

    from .stress import StressFixer, default_dict_paths
    from .translate import speech_text
    from .tts import OmniEngine, St2Engine, UkrTTSEngine, parse_voice

    w = opt.work
    units = load_json(w, "units.json")
    clips = w / "tts"
    shutil.rmtree(clips, ignore_errors=True)
    clips.mkdir()
    kind, arg = parse_voice(opt.voice)
    dict_paths = default_dict_paths(opt.stress_dict, opt.work)

    def say(u: dict, fixer: StressFixer) -> str:
        text = fixer.apply(speech_text(u))
        if text != speech_text(u):
            u["tts_text"] = text  # what the TTS actually got (stress transcriptions)
        else:
            u.pop("tts_text", None)
        return text

    def store(u: dict, y: np.ndarray, sr: int) -> None:
        y = A.trim_silence(y, sr)
        u["tts_len"] = round(len(y) / sr, 3)
        if len(y):
            A.write(clips / f"{u['id']:04d}.wav", y, sr)

    if kind in ("st", "st_duo"):
        # StyleTTS2: our marks (homographs from the LLM, the user's dictionary, «+» from review) are
        # written as acute accents; every other word gets its stress from the dictionary inside the engine.
        eng = St2Engine()
        fixer = StressFixer("off" if opt.stress == "off" else "dict", dict_paths, target="acute")
        male, female = (arg.split(",") + [arg])[:2] if kind == "st_duo" else (arg, arg)
        voices = [female if (kind == "st_duo" and u.get("gender") == "female") else male for u in units]
        styles = _emotion_styles(eng, opt, units, voices) if opt.emotion > 0 else [None] * len(units)
        for u, voice, style in tqdm(list(zip(units, voices, styles)), desc="   озвучення", unit="реп"):
            text = speech_text(u)
            if not _speakable(text):
                u["tts_len"] = 0.0
                continue
            store(u, _st_fit(eng, say(u, fixer), voice, text, slot(u), opt.max_speed, style), eng.sr)
    elif kind in ("preset", "duo_preset"):
        male, female = (arg.split(",") + ["tetiana"])[:2] if kind == "duo_preset" else (arg, arg)
        eng = UkrTTSEngine(male)
        # ukrainian-tts stresses words itself; the user dictionary is passed as acute accents
        fixer = StressFixer("off" if opt.stress == "off" else "dict", dict_paths, target="acute")
        for u in tqdm(units, desc="   озвучення", unit="реп"):
            if not _speakable(speech_text(u)):
                u["tts_len"] = 0.0
                continue
            eng.voice = female if u.get("gender") == "female" else male
            store(u, eng.synth(say(u, fixer)), eng.sr)
    else:
        eng = OmniEngine(steps=opt.omni_steps)
        prompts: list = []
        if kind == "design":
            prompt, sample = eng.design_reference(arg)
            A.write(w / "voice_ref.wav", sample, eng.sr)
            prompts = [prompt] * len(units)
        elif kind == "duo":
            p_m, s_m = eng.design_reference("male, middle-aged", seed=1234)
            p_f, s_f = eng.design_reference("female, young adult", seed=4321)
            A.write(w / "voice_ref_male.wav", s_m, eng.sr)
            A.write(w / "voice_ref_female.wav", s_f, eng.sr)
            prompts = [p_f if u.get("gender") == "female" else p_m for u in units]
        elif kind == "clone_file":
            ref, _ = A.read(Path(arg), sr=eng.sr, mono=True)
            ref = ref[: int(15 * eng.sr)]
            ref_text = opt.ref_text or _transcribe_reference(arg)
            log(f"   • еталонний голос: «{ref_text[:80]}»")
            prompts = [eng.reference(ref, eng.sr, ref_text)] * len(units)
        else:  # clone each line from the original speaker's own voice
            src = _voice_src(w)
            orig, _ = A.read(src, sr=eng.sr, mono=True)
            cache: dict[tuple[int, int], object] = {}
            pitch = [_median_pitch(orig[int(u["start"] * eng.sr) : int(u["end"] * eng.sr)], eng.sr) for u in units]
            for i in range(len(units)):
                a, b = _clone_window(units, i, pitch)
                if (a, b) not in cache:
                    seg = orig[int(units[a]["start"] * eng.sr) : int(units[b]["end"] * eng.sr)]
                    cache[(a, b)] = eng.reference(seg, eng.sr, " ".join(u["text"] for u in units[a : b + 1]))
                prompts.append(cache[(a, b)])

        fixer = StressFixer(opt.stress, dict_paths, target="arpa")
        todo = [i for i, u in enumerate(units) if _speakable(speech_text(u))]
        for u in units:
            u.setdefault("tts_len", 0.0)
        bs = max(1, opt.omni_batch)
        for k in tqdm(range(0, len(todo), bs), desc="   озвучення", unit="пакет"):
            idx = todo[k : k + bs]
            plain = [speech_text(units[i]) for i in idx]
            texts = [say(units[i], fixer) for i in idx]
            durs = [_omni_duration(eng, t, prompts[i], slot(units[i]), opt.max_speed) for t, i in zip(plain, idx)]
            for i, y in zip(idx, eng.synth_batch(texts, durs, [prompts[i] for i in idx])):
                store(units[i], y, eng.sr)
    save_json(w, "units.json", units)


# ---------------------------------------------------------------------------
def stage_mix(opt: Options) -> None:
    from .textnorm import syllables
    from .translate import speech_text

    w = opt.work
    sr = opt.sample_rate
    units = load_json(w, "units.json")
    total = load_json(w, "meta.json")["duration"]
    separated = A.existing(w / "background.wav").exists()
    bg, _ = A.read(A.existing(w / ("background.wav" if separated else "audio.wav")), sr=sr)
    if opt.clip_end is not None:  # a part of a long video: exactly its length, so the parts add up
        bg = A.fit_length(bg, round((opt.clip_end - opt.clip_start) * sr))
    n = len(bg)
    voice = np.zeros(n, dtype=np.float32)

    # st/omni already fitted their lines at synthesis time; here only a long overrun is stretched
    # again, and gently, so the pace does not jump from line to line.
    restretch = opt.max_speed if opt.engine == "ukr" else min(opt.max_speed, MIX_RESTRETCH)
    lengths = [u.get("tts_len", 0.0) for u in units]
    caps = [pace_cap(n, syllables(speech_text(u)), restretch) for u, n in zip(units, lengths)]
    plan = place_clips(units, lengths, max_speed=restretch, caps=caps)
    sped = overflow = 0
    for u, p in zip(units, plan):
        u.update(dub_start=p["start"], speed=p["speed"])
        f = w / "tts" / f"{u['id']:04d}.wav"
        if not u.get("tts_len") or not f.exists():
            u["dub_end"] = p["start"]
            continue
        y, _ = A.read(f, sr=sr, mono=True)
        y = A.fade(A.time_stretch(y, sr, p["speed"]), sr)
        s = int(p["start"] * sr)
        e = min(n, s + len(y))
        if e > s:
            voice[s:e] += y[: e - s]
        u["dub_end"] = round(p["start"] + len(y) / sr, 3)
        sped += p["speed"] > 1.0
        overflow += u["dub_end"] > u["slot_end"] + SPILL + 0.05

    # loudness: dub speaks as loud as the original speech did (one target for all parts of a long video)
    if opt.loudness_target is not None:
        target = opt.loudness_target
    else:
        ref = A.read(A.existing(w / "vocals.wav"), sr=sr)[0] if separated else bg
        target = A.loudness(ref, sr)
        target = float(np.clip(target if target is not None else -18.0, -26.0, -12.0))
    measured = A.loudness(voice, sr)
    voice *= 10 ** ((target - measured) / 20) if measured is not None else 1.0

    duck_db = opt.duck_db if opt.duck_db is not None else (-4.0 if separated else -13.0)
    out = bg * A.duck_gain(voice, sr, duck_db)[:, None] + voice[:, None]
    A.write(w / "mix.wav", A.limit(out, sr), sr)
    save_json(w, "units.json", units)
    write_srt(make_cues(units, "uk"), w / "uk.srt")
    log(f"   • реплік: {len(units)}, прискорено: {sped}, вийшли за слот: {overflow}, тривалість {total:.0f} с")


# ---------------------------------------------------------------------------
def stage_mux(opt: Options) -> None:
    if opt.clip_start is not None:
        _part_output(opt)
        return
    w = opt.work
    out = Path(opt.output)
    srt = w / "uk.srt"
    A.mux(Path(opt.input), w / "mix.wav", out, srt=srt if srt.exists() else None, keep_original=opt.keep_original)
    if srt.exists():
        shutil.copyfile(srt, out.with_suffix(".srt"))


def _part_output(opt: Options) -> None:
    """One part of a long video: keep its dub as FLAC for the final assembly and write a preview."""
    w = opt.work
    out = Path(opt.output)
    srt = w / "uk.srt"
    length = opt.clip_end - opt.clip_start
    src = w / "mix.wav" if (w / "mix.wav").exists() else w / "dub.flac"
    y, sr = A.read(src)
    A.write_flac(w / "dub.flac", A.fit_length(y, round(length * sr)), sr)
    out.parent.mkdir(parents=True, exist_ok=True)
    A.mux(Path(opt.input), w / "dub.flac", out, srt=srt if srt.exists() else None,
          keep_original=opt.keep_original, clip=(opt.clip_start, length))
    if srt.exists():
        shutil.copyfile(srt, out.with_suffix(".srt"))
    log(f"   • превʼю: {out}")
    _cleanup_part(w)


def _cleanup_part(w: Path) -> None:
    """Free disk after a part is done: the stems a later re-dub needs stay as FLAC, the rest goes."""
    (w / "mix.wav").unlink(missing_ok=True)
    separated = A.existing(w / "vocals.wav").exists()
    for name in ("vocals.wav", "background.wav"):
        if (w / name).exists():
            A.to_flac(w / name)
    if (w / "audio.wav").exists():
        if separated:
            (w / "audio.wav").unlink()  # separate re-extracts it if ever needed
        else:
            A.to_flac(w / "audio.wav")  # --no-separate: mix and asr read it


STAGE_FUNCS = {
    "extract": stage_extract,
    "separate": stage_separate,
    "asr": stage_asr,
    "translate": stage_translate,
    "tts": stage_tts,
    "mix": stage_mix,
    "mux": stage_mux,
}
