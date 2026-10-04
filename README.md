# uadub — local AI dubbing into Ukrainian

**English** · [Українська](README.uk.md)

One command: an English (or Korean, Japanese, …) video goes in, the same video comes out with a
Ukrainian voice track, the original music and background preserved, and Ukrainian subtitles.
Everything runs **offline on an Apple Silicon Mac** (tested on a MacBook Pro M1 Pro, 32 GB).

```bash
uadub lecture.mp4 --voice st          # → lecture.uk.mp4 + lecture.uk.srt
```

## Contents

- [Install](#install)
- [Quick start](#quick-start)
- [Choosing a voice](#choosing-a-voice)
- [Reviewing the script before voicing (`--review`)](#reviewing-the-script-before-voicing---review)
- [Translation quality](#translation-quality)
- [Word stress](#word-stress)
- [Korean, other languages and subtitles](#korean-other-languages-and-subtitles)
- [All options](#all-options)
- [How it works](#how-it-works)
- [Performance and quality](#performance-and-quality)
- [Limitations](#limitations)
- [Project layout](#project-layout)
- [Credits and licences](#credits-and-licences)

## Install

```bash
brew install ffmpeg
git clone <this repo> ~/Documents/pet_project/uadub && cd ~/Documents/pet_project/uadub
uv venv --python 3.12 && uv pip install -e .
ln -sf "$PWD/.venv/bin/uadub" ~/.local/bin/uadub     # make `uadub` available everywhere
uadub --prefetch --all                               # download every model once (~25 GB)
```

After the prefetch nothing touches the network: when all models are cached uadub sets
`HF_HUB_OFFLINE=1` by itself. Python 3.12, macOS 14+, about 25 GB of disk for models.

## Quick start

```bash
uadub video.mp4 --voice st                     # accurate Ukrainian stress, no cloning
uadub video.mp4 --voice clone                  # the original speaker's voice (stress may be off)
uadub video.mp4 --voice st --emotion           # …with the intonation of the original lines
uadub video.mp4 --voice st --review            # pause before voicing to check the translation
uadub drama.mkv --from ko --voice duo:st       # Korean drama: male and female lines, two voices
uadub --list-voices                            # every available voice
```

The result is `video.uk.mp4` next to the input: the video stream is copied without re-encoding,
audio track 1 is Ukrainian (default), track 2 is the original (`--drop-original` removes it), plus
embedded Ukrainian subtitles and a separate `video.uk.srt`. Audio-only files work too (→ `.m4a`).

All intermediate files live in `video.uadub/` next to the video. Running the same command again
skips finished stages and redoes only what your changed options affect — e.g. switching `--voice`
re-voices the lines without recognising or translating them again.

## Choosing a voice

There are two kinds of voices. Pick by what matters more for the video:

| | **Accurate stress** — StyleTTS2-ukrainian | **Cloning** — OmniVoice |
|---|---|---|
| Option | `st`, `st:<name>`, `duo:st` | `clone`, `clone:sample.wav`, `"omni:<description>"`, `duo` |
| Word stress | every word stressed from a 2.9-million-form dictionary with grammatical context; homographs chosen by the LLM; your own fixes with `+` | sometimes wrong; the model ignores stress marks |
| Voices | 31 native Ukrainian voices (13 male, 17 female, 1 child) | the original speaker, your sample, or a voice designed from a description |
| Speed on M1 Pro | ~0.2 s per second of speech | ~1–1.5 s per second of speech |
| Licence | MIT | code Apache-2.0, weights non-commercial |

```bash
uadub video.mp4 --voice st                        # default male StyleTTS2 voice
uadub video.mp4 --voice "st:Марта Мольфар"         # any of the 31 voices (prefix works: st:марта)
uadub video.mp4 --voice duo:st                    # male/female per line, detected from voice pitch
uadub video.mp4 --voice clone                     # each line in its own speaker's voice
uadub video.mp4 --voice clone:announcer.wav       # your 5–15 s sample (a Ukrainian voice → no accent)
uadub video.mp4 --voice "omni:female, young adult" --steps 32
uadub video.mp4 --voice duo                       # two OmniVoice voices, male/female per line
```

### Intonation of the original (`--emotion`)

StyleTTS2 voices read evenly, like a newsreader. Add `--emotion` and each line keeps the chosen Ukrainian
voice's timbre but takes its intonation from the original line: rises and falls, energy, pace.
It works because StyleTTS2 stores *who speaks* (timbre) and *how* (prosody) in separate halves of
its style vector. uadub takes the prosody half from the original line's voice, cut out of the
separated vocal track.

```bash
uadub video.mp4 --voice st --emotion          # strength 0.8
uadub video.mp4 --voice st --emotion 1.0      # 0…1; more = closer to the original's manner
uadub drama.mkv --from ko --voice duo:st --emotion
```

On the test clip the pitch range of the dub grew from 4.6 to 6.7 semitones (the original speaker has
6.9) with the same intelligibility. Lines spoken by the opposite sex to the voice keep the plain
style, because a man's voice with a woman's pitch contour sounds wrong. OmniVoice `clone` copies
intonation anyway, so the flag is only for `st` voices.

There are also fast but robotic preset voices from `ukrainian-tts`: `dmytro` (the default),
`oleksa`, `mykyta`, `tetiana`, `lada`, and `duo:dmytro,tetiana`.

## Reviewing the script before voicing (`--review`)

`uadub video.mp4 --voice st --review` translates, places stress and then **pauses** before voicing.
The terminal shows clickable paths to every file, the editing rules, and ready-made commands for
an AI agent. The work folder now contains:

| File | What it is |
|---|---|
| `review.md` | the script: one block per line |
| `en.srt` (or `ko.srt` …) | the original transcript with timings |
| `stress.txt` | stress dictionary for this video |
| `AGENTS.md` | instructions for Claude Code / opencode |

A block in `review.md`:

```
## 0013  00:41.8–00:44.2 · чол. голос · слот 2.4 с · складів 12/13   ← do not edit
EN:  There's two ways to download a model from this page.            ← do not edit
UK:  Є два способи завантажити модель.                               ← translation = subtitles
TTS:                                                                ← how to read it; empty = as UK
НАГОЛОСИ: Є два сп+особи завант+ажити мод+ель.                      ← how StyleTTS2 will stress it
```

Edit `UK:` and `TTS:` by hand, or let an agent do it. Then press Enter in the uadub window and voicing
continues with your edits. Keys at the pause:

| Key | Action |
|---|---|
| Enter | voice the script with your edits |
| `o` | open `review.md` in `$EDITOR` (or the default text editor) |
| `s` | open this video's `stress.txt` |
| `c` | copy the Claude Code command to the clipboard |
| `q` | quit; run the same command later to continue |

### Automatic review by an agent (`--review-with`)

An AI agent can review the script on its own before voicing. The format is `harness[:model]`:

```bash
uadub video.mp4 --voice st --review-with claude                     # Claude Code, default model
uadub video.mp4 --voice st --review-with claude:sonnet              # Claude Code with a chosen model
uadub video.mp4 --voice st --review-with opencode:anthropic/claude-sonnet-4-5   # opencode: provider/model
uadub video.mp4 --voice st --review-with opencode:ollama/qwen3:8b   # fully local via Ollama
uadub video.mp4 --voice st --review-with codex:gpt-5                # Codex CLI
uadub video.mp4 --voice st --review-with gemini:gemini-2.5-pro      # Gemini CLI
uadub video.mp4 --voice st --review-with claude --review            # agent first, then pause for you
```

- Without `--review` the agent edits the script and voicing starts right away. With `--review` you get the
  usual pause afterwards to check its edits.
- The agent may only edit `review.md` and `stress.txt` in the work folder and run `uadub --stress-lookup`.
- Running the same command again does not repeat the review. A different harness or model, a new
  translation, or `--redo review` runs it again.
- Any other tool works as a custom command: `--review-with "my-agent --flag {prompt}"`.
- To always use an agent, set it once: `export UADUB_REVIEW_WITH=claude:sonnet`.

`AGENTS.md` adapts to the voice. For StyleTTS2 the agent checks every `НАГОЛОСИ:` line and fixes wrong
stress with `+`, checking doubtful words against the dictionary with `uadub --stress-lookup`. For OmniVoice it is told not to touch stress, because marks there add an accent.
Edits persist: change `review.md` or `stress.txt` later, run the same command, and only the voice is redone.

## Translation quality

The translation runs on a local LLM (Gemma 4 26B-A4B by default) with these rules:

- **Meaning first.** Before translating, the model reads a brief of the whole video (topic, tone,
  line of argument) and a window of about 5,000 characters of the original around each block.
  It translates sense for sense: sentences are rebuilt, word order changes, and implicit links
  are made explicit.
- **Speech-recognition errors** in the original (e.g. "Quen 3.6" for "Qwen 3.6") are detected and
  translated as what was meant.
- **Idioms and set phrases** get Ukrainian equivalents instead of literal translations
  (*a piece of cake* → «простіше простого», *break the ice* → «розтопити кригу», *once in a blue moon* →
  «раз на сто років»). The brief collects every idiom in the video first, so they are handled consistently.
- **No anglicisms.** UI labels and ordinary terms are translated («Використати цю модель», «квантування»).
  Only proper names stay in Latin script. A separate pass finds words that look like English written
  in Cyrillic and rewrites them. For an audience of specialists, see `--domain` below.
- **Timing.** Every line gets a syllable budget, because Ukrainian syllables are exactly its vowels.
  Lines that would not fit are condensed without dropping facts.
- **Grammar.** Speaker gender (detected from voice pitch) drives я зробив/я зробила. A glossary keeps
  terms consistent; add your own with `--glossary terms.txt` (`English = Українська` per line).

### Translating for specialists (`--domain`)

The brief always detects the field of the video (IT, medicine, finance, cooking…) and prints it as
`• сфера: …`. By default the translation is still plain literary Ukrainian for everyone. With
`--domain` it is written for specialists in that field:

- the jargon Ukrainian professionals actually say, including established anglicisms
  (*deploy* → «деплой», *pull request* → «пул-реквест», *framework* → «фреймворк»), and native
  terms where those are the established ones («база даних»);
- button, menu and setting names stay as they appear on screen, in Latin script and in quotes
  (натисніть «Use this model»), and are spelled out in Cyrillic for the voice;
- slang and ad-hoc transliterations of ordinary words («юзати», «дефолтний») are still removed;
- the review rules (`review.md`, the agent's `AGENTS.md`) switch to the same style.

```bash
uadub talk.mp4 --domain                 # detect the field automatically
uadub talk.mp4 --domain "медицина"      # or name it yourself if the guess is wrong
```

Put `--domain` without a value after the file name, not right before it, or the file name is
taken as the field. Turning the flag on or off re-runs translation (the cache keeps the rest).
Your `--glossary` entries still win, which helps in narrow fields.

### Long videos (`--part-minutes`)

Videos longer than 45 minutes are dubbed in parts automatically:

- the audio is cut at pauses into parts of about 15 minutes (the plan is printed first);
- every part is recognised first, then one brief is built for the whole video, so terms, address
  and `--domain` stay the same everywhere;
- each finished part appears as a preview in `<name>.uk.parts/NN.uk.mp4`;
- at the end the parts are joined into one `<name>.uk.mp4` and `<name>.uk.srt` over the original
  video (no re-encoding of the video).

Memory does not grow with the video length. If a part fails, the others continue; run the same
command again to finish. If a part fails while it is still being recognised, the run stops before the
shared brief (it needs every transcript); the error is saved in `state.json` in the work folder, and
a rerun continues. To fix one part, edit `<name>.uadub/parts/NN/review.md` (it exists after a run
with `--review` or `--review-with`) and rerun: only that part is re-voiced and the result is re-joined. With `--review` the pause happens for each part;
`q` stops the whole run.

```bash
uadub lecture.mp4                       # automatic for videos over 45 min
uadub lecture.mp4 --part-minutes 8      # shorter parts for a Mac with less memory
uadub lecture.mp4 --part-minutes 0      # never split
uadub lecture.mp4 --keep-parts          # keep the part previews
```

The work folder needs about 10 GB for 7 hours. `--subs` does not work with parts yet: with `--subs` a
long video is processed whole (`--subs` with an explicit `--part-minutes N` is an error). A work folder
of an earlier whole run (for example, started before this feature) is continued whole; add
`--part-minutes 15` to split it.

### Translating with Claude, opencode, Codex or Gemini (`--llm`)

The local Gemma model is the default. Instead, translation can go through an agent CLI you already use,
with the same prompts and rules. The format is `harness[:model]`:

```bash
uadub video.mp4 --voice st --llm claude:sonnet
uadub video.mp4 --voice st --llm opencode:anthropic/claude-sonnet-4-5
uadub video.mp4 --voice st --llm codex:<model>       # a model your Codex account allows
uadub video.mp4 --voice st --llm gemini:gemini-2.5-pro
uadub video.mp4 --voice st --llm ollama:qwen3:30b    # local Ollama server, still offline
```

- The text of the video goes to that provider, so the run is no longer offline. Speech recognition
  and voicing stay local. uadub prints a warning.
- It uses the CLI's own login or subscription. The CLI runs headless in an empty temporary folder
  without tools and is used only as a text model.
- Each request starts the CLI again, which adds a few seconds per call. A 10-minute video makes
  about 10–20 calls.
- Switching `--llm` re-translates. Recognition is not redone.
- To make it the default: `export UADUB_LLM=claude:sonnet`.
- It combines with `--review-with`: for example, translate with `--llm claude:sonnet` and review
  with `--review-with opencode:<model>`.

## Word stress

- **StyleTTS2 (`--voice st…`)** stresses every word from the
  [ukrainian-word-stress](https://github.com/lang-uk/ukrainian-word-stress) dictionary (2.9 M word
  forms from *Словники України*), using Stanza grammar for context. Homographs the dictionary cannot
  settle (за́мок/замо́к) are resolved by the LLM from the meaning of the sentence. A `+` before a vowel
  (`зам+ок`) is a real stress mark here.
- **OmniVoice** ignores stress marks ([#52](https://github.com/k2-fsa/OmniVoice/issues/52),
  [#65](https://github.com/k2-fsa/OmniVoice/issues/65)). uadub can force a stress by rewriting the word as
  an ARPAbet transcription (`[Z AA0 M AO1 K]`), but that adds an English accent. So it is used only for
  true homographs and for words you add yourself.
- **Your dictionaries:** `~/.config/uadub/stress.txt` (all videos) and `video.uadub/stress.txt` (one
  video), one word per line (`кілом+етр` or `кіломе́тр`). `~/.config/uadub/pronounce.txt` sets how names
  are read (`LM Studio = ел-ем студіо`).
- `--stress dict` turns off automatic homograph marking, and `--stress off` turns off all marks.

## Korean, other languages and subtitles

`--from ko` (also `ja`, `zh`, `es`, `fr`, `de`, `it`, `pl`, `pt`, `tr`) switches speech recognition to
Whisper large-v3-turbo and translates from that language. For Korean:

- names follow the Kontsevych system (системa Концевича), consistently;
- forms of address (오빠, 선배, 씨 …) are rendered naturally;
- dialogue lines of different characters are never merged.

| Voice for dramas | Result on the test dialogue |
|---|---|
| `--voice duo:st` | accurate stress, male/female voices |
| `--voice duo` | natural OmniVoice voices, male/female |
| `--voice clone` | the actors' timbre, but a strong Korean accent — not recommended |
| one voice + `--no-separate` | classic single-voice voice-over on top of the quieted original |

Existing subtitles are more accurate than speech recognition: `--subs drama.ko.srt` (or `.vtt`).
Ukrainian subtitles can simply be voiced with `--subs drama.uk.srt --subs-lang uk`.

## All options

| Option | Meaning |
|---|---|
| `-o FILE` | output file (default `<name>.uk.mp4`) |
| `--voice V` | voice, see [Choosing a voice](#choosing-a-voice) and `--list-voices` |
| `--emotion [K]` | StyleTTS2 voices take the intonation of each original line, K = 0…1 (0.8 without a number) |
| `--steps N` | OmniVoice quality steps: 16 (default, ~real time) or 32 (slower, cleaner) |
| `--from LANG` | language of the video (`en` default, `ko`, `ja`, …) |
| `--subs FILE`, `--subs-lang LANG` | use existing subtitles instead of recognition |
| `--review` | pause before voicing to edit the script |
| `--review-with HARNESS[:MODEL]` | automatic review by `claude`, `opencode`, `codex` or `gemini` (with an optional model) or a custom command |
| `--redo review` | run the agent review again |
| `--glossary FILE` | your term list, `English = Українська` |
| `--domain [FIELD]` | translate for specialists: field jargon and established anglicisms, on-screen UI labels kept; the field is detected unless given |
| `--part-minutes N` | long videos: parts of ~N min cut at pauses (automatic above 45 min; 0 = never) |
| `--keep-parts` | keep the part previews after joining |
| `--gender male/female` | speaker gender for grammar (otherwise from the voice or detected) |
| `--stress auto/dict/off`, `--stress-dict FILE` | stress handling, extra stress dictionary |
| `--no-separate` | keep the original audio quieted under the voice (voice-over style) |
| `--fast` | faster background separation (htdemucs) |
| `--max-speed X` | maximum speed-up of a line (1.25) |
| `--duck DB` | background level while the voice speaks (−4 dB; −13 dB with `--no-separate`) |
| `--text` | also save the original transcript and the translation as `.txt` files next to the video (`<name>.en.txt`, `<name>.uk.txt`, side by side in `<name>.en-uk.txt`, with every stress marked in `<name>.uk.stress.txt`); links are printed at the end |
| `--drop-original` | do not keep the original audio track |
| `--llm M` | translation model: another MLX model, `ollama:<model>`, or an agent `claude[:model]`, `opencode[:provider/model]`, `codex[:model]`, `gemini[:model]` (not offline); default from `UADUB_LLM` |
| `--redo STAGE`, `--stop-after STAGE` | force a stage again / stop after it |
| `--workdir DIR` | where intermediate files go (default `<name>.uadub/` next to the video) |
| `--stress-lookup WORD…` | show a word's stress variants from the dictionary, with grammar, and your `stress.txt` entry |
| `--prefetch [--all]`, `--list-voices` | download models; list voices |

## How it works

```
video ─ffmpeg→ audio ─BS-RoFormer→ voice + background
   voice ─Parakeet v3 / Whisper (MLX)→ timed transcript ─snap to speech, pitch → speaker gender
      ─Gemma 4 (MLX): brief → sense-for-sense translation with context → anglicisms → homograph stress
      ─[--review: you / Claude Code / opencode edit review.md]
      ─StyleTTS2 / OmniVoice / ukrainian-tts → lines fitted to the original timing
background + voice (ducking, loudness match, limiter) ─ffmpeg→ video.uk.mp4 + .srt
```

| Stage | Model / tool |
|---|---|
| Background separation | `audio-separator`: BS-RoFormer (htdemucs with `--fast`) |
| Speech recognition | `parakeet-mlx` (English), `mlx-whisper` large-v3-turbo (other languages) |
| Translation | `mlx-lm` with `gemma-4-26b-a4b-it-4bit` (or any MLX / Ollama model) |
| Voice | StyleTTS2-ukrainian · OmniVoice · ukrainian-tts |
| Stress | `ukrainian-word-stress` + Stanza, LLM for homographs, `ipa-uk` for StyleTTS2 |
| Timing | syllable budget, the TTS speed / duration controls, Rubber Band (pedalboard) as a fallback |

Every heavy stage runs in its own process, so only one model sits in memory at a time (peak ~15 GB
for the LLM).

Ideas borrowed from other projects: VideoLingo (brief, chunked translation with context,
shortening pass), open-dubbing (lines may spill into the following pause, capped speed-up),
pyVideoTrans / VoiceStudio (length budget before synthesis), and OmniVoice (generating the exact
duration instead of stretching).

## Performance and quality

On an M1 Pro, 32 GB:

| Stage | Time |
|---|---|
| Background separation, BS-RoFormer / htdemucs | ~1× / ~0.4× real time |
| Speech recognition, Parakeet / Whisper | ~0.1× / ~0.3× real time |
| Translation (Gemma 4 26B, all passes) | ~1.5–2 min per minute of video |
| Voice, StyleTTS2 / OmniVoice (16 steps) / ukrainian-tts | ~0.2× / ~1× / ~0.4× real time |

A 10-minute video takes about 30–40 minutes end to end.

Quality was checked by transcribing the dubbed track back and comparing it with the script. The
word error rate is 0.07–0.14, and most "errors" are numbers that the recogniser wrote as digits.
All 31 StyleTTS2 voices transcribed back perfectly on a test sentence.

## Limitations

- OmniVoice cloning carries the original accent (English, Korean) into Ukrainian. For clean
  Ukrainian use `--voice st` or clone a Ukrainian sample.
- OmniVoice weights are non-commercial. Check every model's licence before monetising content.
- Speaker gender comes from voice pitch, which children, shouting or whispering can fool. It is
  shown in `review.md`.
- Overlapping speech and singing are handled poorly. There is no lip-sync and no multi-speaker
  diarization (yet).
- A damaged audio track (corrupt AAC packets) is salvaged rather than rejected. The broken parts
  become silence, stay in sync with the video, and are not translated. The console prints how many
  seconds were lost.

## Project layout

```
uadub/cli.py        CLI, stages in separate processes, state cache, the --review pause
uadub/stages.py     extract → separate → asr → translate → tts → mix → mux
uadub/translate.py  prompts: brief, sense-for-sense translation, shortening, anglicisms, homographs
uadub/tts.py        StyleTTS2, OmniVoice and ukrainian-tts engines
uadub/stress.py     stress dictionaries, homographs, ARPAbet for OmniVoice
uadub/review.py     review.md / AGENTS.md export and import, agent launcher
uadub/textnorm.py   syllables, numbers → words, name pronunciation, Latin → Cyrillic
uadub/segments.py   sentence units, timing slots, snapping to speech
uadub/fit.py        placing lines on the timeline
uadub/subs.py       reading .srt / .vtt
uadub/term.py       clickable terminal links, clipboard, editor
tests/              unit tests, sample generators (EN/KO), quality checks (roundtrip, pace)
```

## Credits and licences

Built on [Parakeet](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3) /
[parakeet-mlx](https://github.com/senstella/parakeet-mlx),
[Whisper](https://github.com/openai/whisper) / [mlx-whisper](https://github.com/ml-explore/mlx-examples),
[Gemma](https://ai.google.dev/gemma) via [mlx-lm](https://github.com/ml-explore/mlx-lm),
[audio-separator](https://github.com/nomadkaraoke/python-audio-separator),
[StyleTTS2-ukrainian](https://huggingface.co/patriotyk/styletts2_ukrainian_multispeaker) (patriotyk),
[OmniVoice](https://github.com/k2-fsa/OmniVoice), [ukrainian-tts](https://github.com/robinhad/ukrainian-tts),
[ukrainian-word-stress](https://github.com/lang-uk/ukrainian-word-stress), [ipa-uk](https://github.com/patriotyk/ipa-uk),
and [pedalboard](https://github.com/spotify/pedalboard).
Each keeps its own licence. Note that OmniVoice weights are non-commercial and pedalboard is GPL-3.0.
