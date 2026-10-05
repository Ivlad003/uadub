# uadub — Specification, Requirements and Architecture Decisions

Status: living document · Last updated: 2026-10-03 · Owner: Kosmodev

This document collects the requirements, design and architecture decisions (ADRs) behind **uadub**,
a local AI dubbing tool that turns a foreign-language video into a Ukrainian-dubbed one. It was
reconstructed from the design conversation in which the tool was built, so it also records why each
decision was taken and which alternatives were rejected.

---

## 1. Goal

Pass a video, get it back dubbed into Ukrainian, as simply as possible:

```bash
uadub video.mp4 --voice st      # → video.uk.mp4 + video.uk.srt
```

The original music and background stay. The original audio is kept as a second track, and Ukrainian
subtitles are embedded. By default everything runs offline on the user's MacBook.

### 1.1 Context

- User: Ukrainian-speaking senior engineer; personal (pet) project.
- Primary machine: MacBook Pro M1 Pro, 32 GB RAM, 512 GB disk, macOS 26. A desktop with an RTX 4060
  also exists, but it is not a target platform.
- Project location: `~/Documents/pet_project/uadub` (uv-managed Python 3.12 venv; `uadub` symlinked into `~/.local/bin`).
- Inspiration: overcrash66/video-translator and Huanshere/VideoLingo, plus research into open-dubbing,
  pyVideoTrans, VoiceStudio and others.

### 1.2 Initial choices made by the user

| Question | Choice |
|---|---|
| Deliverable | Research plus a working CLI |
| Voice | Both modes (preset voices and voice cloning) behind a switch |
| Connectivity | 100 % offline |
| Install target | Installed and tested directly on the user's Mac |

---

## 2. Requirements

### 2.1 Functional requirements

| ID | Requirement | Status |
|---|---|---|
| FR-1 | One command: video (or audio) in, Ukrainian-dubbed video out | ✅ |
| FR-2 | Preserve background music/effects; replace only the voice | ✅ |
| FR-3 | Keep the original audio as a second track (optional `--drop-original`) | ✅ |
| FR-4 | Ukrainian subtitles: embedded and as a separate `.srt` | ✅ |
| FR-5 | Audio-only input supported (→ `.m4a`) | ✅ |
| FR-6 | Selectable voices: fast presets, voice cloning, voice design, accurate-stress voices | ✅ |
| FR-7 | Correct Ukrainian word stress | ✅ (StyleTTS2 path) |
| FR-8 | Natural, less robotic voice | ✅ (OmniVoice / StyleTTS2) |
| FR-9 | Pause before voicing to edit the script by hand or with Claude Code / opencode (`--review`) | ✅ |
| FR-10 | Clickable and copyable file links in the terminal | ✅ |
| FR-11 | Korean source (dramas) plus other languages; use existing subtitles as input | ✅ |
| FR-12 | Male/female voices chosen per line for dialogue (`duo`) | ✅ |
| FR-13 | No anglicisms in the translation (default; `--domain` allows the field's established jargon, ADR-026); no drawn-out speech | ✅ |
| FR-14 | Translation by meaning, not literally; idioms replaced with Ukrainian equivalents; use the whole transcript as context | ✅ |
| FR-15 | Reproduce the original intonation (`--emotion`) | ✅ (StyleTTS2, experimental) |
| FR-16 | Save the original transcript, the translation, the stressed text and a bilingual text as `.txt`, with links (`--text`) | ✅ |
| FR-17 | Dictionary lookup for stress (`--stress-lookup`) usable by humans and agents | ✅ |
| FR-18 | Fully automatic review by an agent, with model choice (`--review-with harness:model`) | ✅ |
| FR-19 | Translation through an agent CLI (Claude, opencode, Codex, Gemini) as an option (`--llm harness:model`) | ✅ |
| FR-20 | Documentation: English README (primary) plus Ukrainian README | ✅ |
| FR-21 | Long videos (4–7 h): automatic parts cut at pauses, one shared brief, previews, one result | ✅ |

### 2.2 Non-functional requirements

| ID | Requirement |
|---|---|
| NFR-1 | Runs offline after a one-time `uadub --prefetch --all` (cloud LLM and agent options are opt-in exceptions) |
| NFR-2 | Fits in 32 GB unified memory: only one heavy model is resident at a time; long videos are processed in parts, so RAM does not grow with the length (ADR-027) |
| NFR-3 | Re-running a command is cheap: finished stages are cached and only affected stages are redone |
| NFR-4 | Quality is measurable: round-trip ASR intelligibility (CER/WER), speaking pace, pitch range |
| NFR-5 | Console UX in Ukrainian; code, comments and the primary docs in English |
| NFR-6 | Licences are respected and documented (OmniVoice weights are non-commercial; pedalboard is GPL-3.0) |

### 2.3 Hardware requirements

| | Everything local (default) | Translation in the cloud (`--llm claude:…`) |
|---|---|---|
| Chip | Apple Silicon (M1+), because MLX is required | Apple Silicon (M1+), because ASR stays on MLX |
| RAM | 32 GB comfortable; 16 GB with a smaller LLM | 16 GB comfortable; 8 GB likely tight (untested) |
| Disk for models | ~25 GB | ~10 GB |
| Network | First download only | Every translation run |

Intel Macs, Windows and Linux (including CUDA) are not supported in the current version.

---

## 3. Architecture

### 3.1 Pipeline

```
video ─ffmpeg→ audio ─BS-RoFormer→ vocals + background
  vocals ─Parakeet v3 (EN) / Whisper large-v3-turbo (other)→ timed sentences
     ─snap to speech, pitch → speaker gender per line
     ─LLM: brief → sense-for-sense translation with context → shorten → anglicism fix → homograph stress
     ─[--review / --review-with: human or agent edits review.md + stress.txt]
     ─TTS: StyleTTS2 / OmniVoice / ukrainian-tts, fitted to the original timing (+ --emotion prosody)
background + voice (ducking, loudness match, limiter) ─ffmpeg→ video.uk.mp4 + .srt (+ --text files)
```

Stages: `extract → separate → asr → translate → tts → mix → mux`.

### 3.2 Modules

| Module | Responsibility |
|---|---|
| `cli.py` | Argument parsing, stage orchestration in subprocesses, state cache, `--review` pause, final links |
| `config.py` | `Options` dataclass, models, voices, per-stage fingerprints |
| `stages.py` | Implementation of all seven stages, clip fitting, clone windows, `--emotion` styles |
| `translate.py` | Prompts (brief, translate, shorten, anglicisms, homographs), JSON salvage, syllable budgets |
| `llm.py` | LLM backends: MLX (default), Ollama, agent CLIs |
| `tts.py` | `St2Engine`, `OmniEngine`, `UkrTTSEngine`; stress-text preparation |
| `stress.py` | Stress dictionaries, homograph finder, ARPAbet, `--stress-lookup` |
| `review.py` | `review.md` / `AGENTS.md` export and import, agent launcher (`harness:model`) |
| `textnorm.py` | Syllables, numbers → words, pronunciations, Latin → Cyrillic |
| `segments.py`, `fit.py`, `srt.py`, `subs.py` | Units and slots, placement, subtitles, subtitle input |
| `texts.py` | `--text` output files |
| `term.py` | OSC 8 links, clipboard, editor |

### 3.3 Work folder

`<video>.uadub/` sits next to the input and holds:

- intermediates: `audio.wav`, `vocals.wav`, `background.wav`, `transcript.json`, `units.json`, `brief.json`, `tts/`, `mix.wav`;
- `state.json` (stage fingerprints) and `options.json`;
- review files: `review.md`, `.review-export.json`, `AGENTS.md`, `stress.txt`, `<src>.srt`, `uk.srt`.

User-wide files:

- `~/.config/uadub/stress.txt` — stress dictionary for all videos;
- `~/.config/uadub/pronounce.txt` — how names are read.

---

## 4. Architecture Decision Records

Format: Context → Decision → Consequences. Status is *Accepted* unless noted.

### ADR-001: Apple Silicon and MLX as the runtime
- **Context:** The target is an M1 Pro, and the work must be local and reasonably fast.
- **Decision:** Use MLX for ASR (parakeet-mlx, mlx-whisper) and the LLM (mlx-lm). PyTorch on MPS runs
  the TTS engines.
- **Consequences:** Good speed on Mac. No Intel Mac, Windows or Linux/CUDA support; the RTX desktop is not a target.

### ADR-002: One process per heavy stage, with a fingerprint cache
- **Context:** The LLM (~15 GB), separator, ASR and TTS cannot all stay resident in 32 GB.
- **Decision:** Each stage runs in a `spawn` subprocess. `state.json` stores a fingerprint of the
  options that affect each stage, and a mismatch reruns that stage and the ones after it. Fingerprint
  versions are bumped when prompts change. New fields are added only when non-default, so old caches survive.
- **Consequences:** Peak memory equals one model. Changing `--voice` revoices without re-translating.
  Killing a run can leave orphaned children, which must be killed by hand.

### ADR-003: Source separation with BS-RoFormer (htdemucs with `--fast`)
- **Decision:** `audio-separator[cpu]` with `model_bs_roformer_ep_317_sdr_12.9755.ckpt` by default
  (best SDR, ~1× real time); `htdemucs` with `--fast` (~3× faster). For 4-stem models the background
  is computed as mix − vocals.
- **Consequences:** Clean background. `audioread` and the `[cpu]` extra are required dependencies.

### ADR-004: ASR — Parakeet for English, Whisper for other languages
- **Decision:** `parakeet-tdt-0.6b-v3` (precise timestamps) for English, `whisper-large-v3-turbo` for other
  languages. Whisper and subtitle timings are snapped to real speech, and a hallucination filter is applied.
- **Consequences:** Accurate English timing; Korean and others are supported. Existing subtitles
  (`--subs`) can replace ASR, and Ukrainian subtitles (`--subs-lang uk`) skip translation.

### ADR-005: Speaker gender from pitch
- **Decision:** The per-line median F0 is computed with librosa yin; above 165 Hz the line is female.
  Gender drives Ukrainian grammar (зробив/зробила) and `duo` voice selection.
- **Consequences:** Cheap, and 16/16 correct on the Korean test. Children, shouting or whispering can
  fool it, so gender is shown in `review.md`.

### ADR-006: Local LLM translation — Gemma 4 26B-A4B (4-bit) via mlx-lm
- **Decision:** Default `mlx-community/gemma-4-26b-a4b-it-4bit` (~45 tok/s), with thinking disabled.
  The first attempt is greedy, retries use temperature 0.2 and 0.4, and well-formed lines are salvaged
  from broken JSON. `ollama:<model>` is an alternative.
- **Consequences:** Offline, good quality, peak ~15 GB.

### ADR-007: VideoLingo-style translation with a brief, chunks and context
- **Decision:**
  1. A brief of the whole transcript: summary, characters, glossary, idioms, ASR fixes.
  2. Chunks of 20 lines, each with previous and next lines plus a ~5,000-character
     `transcript_context` window (current lines marked `»`).
  3. One output per input id; lines are never merged or split.
- **Consequences:** Consistent terminology and coherent sense-for-sense translation.

### ADR-008: Translation style — meaning first, idioms, no anglicisms
- **Context:** The user reported literal translations, anglicisms and over-abridged lines.
- **Decision:** The prompt has three parts:
  - "How to work":
    - A — meaning first, using the transcript context;
    - B — infer ASR errors ("Quen" → Qwen);
    - C — idioms and set phrases get established Ukrainian equivalents, with examples
      ("when it comes to" → «коли йдеться про»); fillers are dropped.
  - Numbered rules: no anglicisms (UI labels translated, Latin only for proper names), a Cyrillic `tts` field, no stress marks.
  - A dedicated anglicism-fix pass on suspect words (unknown Cyrillic words, lowercase Latin).
  - Korean and Japanese get extra rules: Kontsevych/Kovalenko transliteration, forms of address, ти/ви.
- **Consequences:** Natural Ukrainian. The shorten pass must keep sentences grammatical; a rule with a
  wrong/right example was added after «Коли справа стає завантаження».

### ADR-009: Timing by syllable budgets, not time-stretch
- **Decision:** Ukrainian syllables are counted as vowels. Each line gets
  `max_syl = slot × rate × fill`, where rate is engine-specific (`ukr` 5.3, `omni` 6.2, `st` 4.8) and
  `fill = min(max_speed, 1.3) × 0.96`. Lines over budget are condensed by the LLM (up to two passes).
  A line may spill up to 0.8 s into the next pause. Speed is capped by `--max-speed` (1.25), and
  Rubber Band (pedalboard) stretching is a fallback only (how much of a fallback: ADR-028).
- **Consequences:** Few artefacts and consistent pacing.

### ADR-010: Three TTS engines behind one `--voice` switch
| Engine | Option | Strength | Weakness |
|---|---|---|---|
| ukrainian-tts (ESPnet) | `dmytro`, `tetiana`, … , `duo:a,b` | fast, MIT | robotic |
| OmniVoice | `clone`, `clone:file`, `omni:<desc>`, `duo` | natural; clones the speaker; copies intonation | ignores stress marks; carries the source accent; weights non-commercial |
| StyleTTS2-ukrainian | `st`, `st:<name>`, `duo:st` | accurate stress; 31 native voices; RTF ~0.2; MIT | no cloning; flat newsreader prosody without `--emotion` |

- **Decision:** All three are kept. Recommended: `st` for accuracy, `clone` for the speaker's timbre,
  and `duo:st` for dramas.
- **Consequences:** The user picks the trade-off: "cloning with stress problems" vs "no cloning,
  correct stress".

### ADR-011: OmniVoice pacing — explicit duration
- **Context:** Speech was drawn out (median 5.2 vs 6.0 syl/s). OmniVoice estimated duration from text
  that included ARPAbet brackets.
- **Decision:** Estimate duration on the plain text, clamp the rate to 5.0–7.5 syl/s, and always pass
  an explicit duration. Unspeakable lines (".", "♪") are skipped.
- **Consequences:** 0 drawn-out lines on the test clip.

### ADR-012: Clone references — pitch-aware windows
- **Decision:** Each line is cloned from its own audio, extended with neighbours of the same speaker
  (gap ≤ 0.8 s, pitch within ~3.5 semitones, 4–12 s long).
- **Consequences:** In dialogue a man's line never gets a woman's reference. Cross-language cloning
  (Korean) still has an accent, so `duo` or `duo:st` is recommended for dramas (CER 0.023 vs 0.33).

### ADR-013: Word stress strategy
- **Context:** Stress errors were the most audible issue.
- **Decision:**
  - The dictionary is lang-uk `ukrainian-word-stress` (2.9 M forms from «Словники України») with
    Stanza POS context.
  - True homographs (unresolvable by POS) are resolved by the LLM from sentence meaning: two passes
    in reversed order, answering with the variant text or "both".
  - Common function words are skipped.
  - User notation is `+` before the stressed vowel; internally a combining acute is used.
  - StyleTTS2 and ukrainian-tts take acute accents directly.
  - OmniVoice ignores marks, so marked words are rewritten as ARPAbet. Because that adds an English
    accent, it is used only for homographs and user dictionary words.
  - Dictionaries: global, per-video and `--stress-dict`; `--stress auto|dict|off`.
- **Consequences:** Accurate stress on StyleTTS2 (verified by ear and by round-trip ASR).
  OmniVoice stays imperfect by design.
- **Fix:** ukrainian-word-stress looks capitalised words up as proper nouns («Ко́ли»), so
  sentence-initial words are stressified lower-cased as well, and the common-word stress wins.

### ADR-014: Human/agent review loop (`--review`)
- **Decision:** After translation:
  - **`review.md`:**
    - one block per line: header with timing, gender, slot and syllables X/Y with ⚠;
    - read-only source line;
    - editable `UK:` (subtitles) and `TTS:` (speech);
    - read-only `НАГОЛОСИ:` preview (st only).
  - **Edit tracking:** a `.review-export.json` snapshot detects edits.
  - **`AGENTS.md`:** engine-aware instructions — understand the content first, idioms, anglicisms,
    gender, numbers/names, and stress (fix for st, never for omni).
  - **Console pause:** shows files, how to read `review.md`, rules 1–7 and ready-made agent commands.
    Keys: Enter / o / s / c / q (Ukrainian layout too).
  - **Import:** edits are imported only if the export matches the current translation. `AGENTS.md`
    is refreshed on every pause.
- **Consequences:** The human stays in control; edits survive reruns, and only the voice is redone.

### ADR-015: Agent review — `--review-with harness[:model]`
- **Decision:**
  - **Harnesses:**
    - `claude` (`-p`, acceptEdits, `--model`);
    - `opencode` (`run -m provider/model`);
    - `codex` (`exec`, workspace-write sandbox, `-m`);
    - `gemini` (`--approval-mode auto_edit`, `--skip-trust`, `-m`);
    - any custom command with `{prompt}`.
  - **Permissions:** the agent may edit files in the work folder and run only `uadub --stress-lookup`.
  - **Flow:** without `--review` voicing starts right away; with it, the pause follows.
  - **Repeat runs:** the review does not repeat for the same harness and translation; `--redo review`
    forces it.
  - **Default:** `UADUB_REVIEW_WITH` env.
- **Consequences:** Fully automatic quality pass. Speed depends on the model; a free opencode model
  took over 15 minutes.

### ADR-016: Stress verification by dictionary, not by rules
- **Context:** The user asked to put "modern Ukrainian stress rules" into the prompt.
- **Decision:** Ukrainian stress is free and mobile, and the norm lives in dictionaries, not rules.
  So instead:
  1. `uadub --stress-lookup WORD…` prints every dictionary variant with grammar and the user's `stress.txt`
     entry, and flags "both allowed" (по́милка/поми́лка).
  2. `AGENTS.md` gets a "check with the dictionary, not from memory" rule plus a pitfalls cheat sheet
     (ви- perfectives, numerals, loanwords, mobile stress, frequent errors), with every example
     verified against the dictionary.
  3. Online dictionaries (goroh, r2u) are not bundled, because of offline and licensing concerns.
- **Consequences:** Agents check with the same dictionary the synthesizer uses.

### ADR-017: Prosody transfer — `--emotion [K]`
- **Context:** StyleTTS2 sounds flat. The question was whether it can voice with intonation.
- **Decision:**
  - StyleTTS2's 256-dimensional style vector is timbre (first 128) plus prosody (last 128).
  - Per line, the prosody half is computed from the original vocal clip with the model's own
    `predictor_encoder` and blended as `(1−K)·voice + K·original`. Default K = 1.0 (was 0.8: at 1.0 the
    pitch range is 6.9 vs 6.6 semitones against the speaker's 7.6, with a slightly better CER).
  - Opposite-sex lines keep the plain style. Short lines borrow same-speaker neighbours (1.5–6 s).
- **Measured on the test clip:** pitch range (10–90 %) 4.6 → 6.7 semitones (original speaker 6.9);
  25/28 lines wider; CER 0.073 → 0.075; drawn-out lines 2 → 0. Korean `duo:st --emotion`: CER 0.011.
- **Consequences:** More expressive speech with the same intelligibility. The voice is slightly lower.
  Experimental; only for `st` voices.

### ADR-018: Translation via agent CLIs — `--llm harness[:model]`
- **Decision:** `AgentCLILLM` runs `claude`, `opencode`, `codex` or `gemini` headless in an empty
  temporary folder, without tools, as a plain text model. It uses the same prompts and the CLI's own
  login. Thinking is disabled (`MAX_THINKING_TOKENS=0`, overridable with `UADUB_AGENT_THINKING`),
  because it made Claude calls 3–6× slower. Codex uses low reasoning effort. A cloud warning is printed.
  Default from `UADUB_LLM`.
- **Measured:** `claude:haiku` translated the whole clip in 92 s, vs ~136 s for Gemma.
- **Consequences:** Not offline when used. Codex and Gemini depend on the user's account: on the test
  Mac the ChatGPT plan rejected the models, and the free Gemini Code Assist tier is discontinued.

### ADR-019: Text outputs — `--text`
- **Decision:** Next to the output video, write:
  - `<name>.<src>.txt` (original) and `<name>.uk.txt` (translation): paragraphs split at pauses of
    2 s or more, each with a timestamp;
  - `<name>.uk.stress.txt` (every stress marked with an acute; exact for st, dictionary-based for other voices);
  - `<name>.<src>-uk.txt` (bilingual, line by line).

  Links are printed at the end, and also on `--stop-after`.
- **Consequences:** Reading and checking without opening the video. The stress file is output only.

### ADR-020: Robust JSON handling
- **Context:** The Korean translation once failed because the model echoed input fields and the output was truncated.
- **Decision:**
  - The prompt forbids echoing fields.
  - Token budget is 400 + 260 per line.
  - Salvage parses each `{…}` object regardless of field order or extra fields.
  - On repeated failure a chunk is split in halves recursively; as a last resort the source text is kept.
- **Consequences:** A single bad reply no longer kills the stage.

### ADR-021: Mixing and muxing
- **Decision:**
  - Mixing: pedalboard and pyloudnorm loudness match, ducking (−4 dB separated, −13 dB voice-over style), limiter.
  - Muxing: video stream copied; Ukrainian audio is the default track and the original is second.
  - Subtitles: `mov_text` for mp4/m4a/mov (the srt codec failed in the ipod muxer).
- **Consequences:** Fast mux, compatible files.

### ADR-022: Terminal UX
- **Decision:**
  - OSC 8 clickable links, `pbcopy` copy, `$EDITOR` / `open` for editing.
  - Ukrainian stage titles that show which LLM is translating.
  - `HF_HUB_OFFLINE=1` is set automatically when all models are cached.

### ADR-023: Documentation language
- **Decision:** `README.md` in English (primary) and `README.uk.md` in Ukrainian, cross-linked. This
  spec is in English.

### ADR-024: Deployment workflow (development-time)
- **Context:** The code was developed in a cloud workspace and deployed to the Mac through a file bridge.
- **Decision:**
  - Uniquely timestamped tarballs, because re-committing the same path served stale content.
  - Extracted in the project, and `pytest` is run there.
  - Long jobs run in the background with log polling (the remote shell has a 60 s limit).
- **Consequences:** Reproducible deploys. Not relevant to end users.

### ADR-025: Damaged audio tracks
- **Context:** Recordings with corrupt AAC packets (cut from a damaged OBS file) made the extract
  stage fail. ffmpeg either drops bad packets, which shifts all later audio, or dies with
  `Error reinitializing filters` when a garbage packet "changes" the sample rate. A garbage packet
  can also leave the decoder in a broken state, so later good packets fail too. Apple's `aac_at`
  decoder stalls the same way.
- **Decision:**
  - `extract_audio` runs one ffmpeg call with `-max_error_rate 1.0`. If its stderr has no decoder
    errors, the result is used as before.
  - Otherwise `salvage_timeline` assembles a buffer of the input's length. It restarts ffmpeg with a
    fresh decoder (`-ss pos`, `-reinit_filter 0`, `aresample=async=1`) 0.5 s past the last decoded
    sample each time a run dies, and writes every piece at its own timestamp. Gaps stay silent.
  - The console reports how much is silent. More than 90% silent aborts with a plain message.
  - Non-heavy stages turn `RuntimeError` into a one-message `SystemExit` without a traceback.
- **Consequences:** The dub stays in sync with the video, and the damaged parts become silence. A
  badly damaged 23-minute file takes about 25 s to salvage. The extract fingerprint is unchanged,
  because clean files decode exactly as before.


### ADR-026: Field detection and the specialist mode (`--domain`)
- **Context:** The plain style of ADR-008 (no anglicisms, every term translated) suits a general
  audience. Talks for specialists, such as software developers, sound unnatural without the jargon
  they actually use («деплой», «пул-реквест»).
- **Decision:**
  - The brief always returns `domain`, the field of the video. It is printed in both modes but
    changes nothing in the plain mode, whose prompts are unchanged.
  - `--domain` (`Options.domain = "auto"`) or `--domain FIELD` switches to the specialist mode. Rule
    5 of the translation prompt asks for the field's professional terminology, including
    established anglicisms. On-screen UI labels stay in Latin in quotes. Slang and ad-hoc
    transliterations of ordinary words are still banned.
  - The brief's glossary collects the field's jargon (up to 40 entries). Its words are exempt from
    the anglicism pass, whose prompt changes to "replace only what specialists would not say".
  - `review.md`, the generated `AGENTS.md`, the agent prompt and the `--review` hint use the same
    rule. The "not in the dictionary" list drops glossary jargon.
  - A spelling pass (`spell_latin`) asks for a Cyrillic `tts` for every line that still has Latin
    text and no `tts`, because the letter-by-letter fallback reads «Use this model» as «асе тіс
    модел». The anglicism pass also keeps a `tts` it returns for a line it left unchanged.
  - `domain` enters the translate fingerprint only when set, so existing caches survive.
- **Consequences:** One flag gives a plain or a specialist dub of the same video. Quality in
  narrow fields depends on how well the LLM knows the jargon; `--glossary` overrides it.

### ADR-027: Long videos in parts
- **Context:** A 4–7 h input kept whole needs ~1.3 GB per hour per audio copy in separation and
  mixing, its brief sees only the first and last 12,000 characters, and one failure loses a whole
  stage. The user cut recordings by hand and got 21 files with drifting terminology.
- **Decision:**
  - Inputs over 45 min (or `--part-minutes N`, when longer than 1.5 × N) are planned into parts cut
    at the middle of the longest pause within ±3 min of each target (`uadub/parts.py`). The plan is
    stored in `plan.json` and reused. The automatic split is skipped (with a one-line note) when the
    work folder already holds a whole run without a plan, or with `--subs`.
  - Only the audio is cut. Each part is an ordinary work folder (`parts/NN`) with `clip_start/end`;
    extract and mix are sample-exact, so the parts add up to the input. A part whose audio is almost
    all lost to damage is dubbed as near-silence instead of stopping the run (a whole input still stops).
  - Phase 1 runs extract → asr for all parts; phase 2 builds one brief by map-reduce over the whole
    transcript (in a spawn subprocess); phase 3 translates (with neighbouring text), voices and
    mixes each part with one loudness target, writes `dub.flac` and a preview (video bitrate capped
    at the source's), and compresses stems. A part whose `dub.flac` went missing is mixed again before
    assembly. `--stop-after mux` means the whole job, assembly included.
  - The parts' dubs are concatenated by ffmpeg and muxed once over the original video. A failed part
    does not stop the others; a rerun resumes.
  - A failed part is recorded in the shared work folder's `state.json` (`failed`) with its last error
    line. If a part fails in phase 1 the run stops before phases 2 and 3, because the brief needs
    every transcript; rerun to continue.
  - With `--review` the pause happens per part in phase 3; `q` at a pause stops the whole run.
- **Consequences:** Peak RAM depends on the part length, not the video length. Previews appear as
  each part finishes (phase 3). Disk: ~10 GB for 7 h in the work folder. `--subs` is not supported in
  part mode yet: with an automatic split the input is processed whole (more RAM), and an explicit
  `--part-minutes N` with `--subs` is an error.

### ADR-028: One pace band, no double speed-up
- **Context:** The `st` dub sounded uneven: lines sped up and slowed down from one to the next. On
  `examples/ab/clip.mp4` the pace ran from 4.2 to 11.1 syl/s (p10–p90 4.8–8.2). Two causes:
  `_st_fit` re-synthesised a long line faster (up to 1.35×) and then the mix stretched it *again*
  with Rubber Band for any overrun above 0.1 s, each line by its own factor (16 of 28 lines,
  1.03–1.14×); and short lines StyleTTS2 spoke slowly (4.2–4.9 syl/s) were left alone, next to
  neighbours pushed to 7–8.
- **Decision:**
  - `fit.st_speed` picks the StyleTTS2 re-synthesis speed from a pace band (`MIN_RATE` 5.0,
    `MAX_RATE` 7.5 syl/s, shared with OmniVoice): a line is sped up to reach the floor and to fit its
    slot, but never past the ceiling or `min(1.35, --max-speed + 0.1)`. A line already faster than
    the ceiling is not sped up at all: a rushed line is less intelligible than a late one.
    StyleTTS2's `speed` is not linear (1.1 shortens a line by ~7 %, 1.2 by ~17 %, 1.4 garbles), so
    `fit.st_engine_speed` maps the wanted ratio to the engine value, `1 + 1.25 × (ratio − 1)`, capped
    at 1.35 (tts version 6).
  - `fit.place_clips` lets a line run `spill` = 0.5 s past its slot untouched (the next line starts
    late and the timeline catches up at the next gap); only a longer overrun is stretched, to fit
    `avail + spill`, and stretches under 8 % are skipped (`min_stretch` 1.03 → 1.08).
  - For engines that already fit their lines (`st`, `omni`) the mix may re-stretch by at most 1.15
    (`MIX_RESTRETCH`); `ukr` keeps the full `--max-speed`, since the mix is its only fitting. The
    ceiling holds in the mix too: `fit.pace_cap` lowers each line's cap to what keeps it under
    `MAX_RATE`, so a line the engine already voiced at 7.5 syl/s is not stretched again.
  - The mix summary counts a line as "over its slot" only past `slot_end + SPILL`.
  - Fingerprints: tts (st) version 5, mix version 2. Translate is untouched.
- **Consequences:** On the test clip (`st`, `--emotion 0.8`): p10–p90 4.8–8.2 → 4.9–7.4 syl/s,
  rushed lines 3 → 0, lines stretched in the mix 16 → 9 (all ≤ 1.15), slowest line 4.2 → 4.7. The
  price is drift: in continuous speech the dub starts up to ~1 s after the original line and catches
  up at the next pause. Lines far over their syllable budget (the LLM failed to shorten them) now
  spill instead of being rushed; a third shortening pass in translate is the deferred follow-up.

### ADR-029: Pauses between lines and budgets for a natural pace
- **Context:** The `st` dub sounded breathless. The speaker on the test clip pauses a median 0.34 s
  between sentences; the dub had 0.05 s at 25 of 27 joins. Three causes: Parakeet sentence ends run
  into the next sentence, so the pause was inside the slot and budgeted as speaking time
  (`snap_to_speech` ran only for Whisper and subtitles); `set_budgets` assumed every line would be
  sped up 1.2× (`fill = max_speed × 0.96`) while the prompt asks for lines close to `max_syl`, so
  translations were systematically long and the dub drifted ~1 s late; and `place_clips` glued a
  late line to the previous one with `min_gap` 0.05.
- **Decision:**
  - `snap_to_speech` runs for every ASR source (asr version 4).
  - `set_budgets` keeps `PAUSE_RESERVE` 0.3 s of the slot for the pause (never cutting into the
    spoken part of the line) and assumes only 40 % of the allowed speed-up:
    `fill = 1 + (min(max_speed, 1.3) − 1) × 0.4` (1.1 by default). Translate version 14.
  - `place_clips` keeps `BREATH` 0.25 s between a line that overran and the next.
  - The style rule of the prompt names the concessive calque «як би ви не …» → «хоч як ви …».
- **Consequences:** Measured in `tests/pace.py` (pauses, late start). Shorter translations also mean
  fewer stretched lines and less drift; the cost is a one-time re-run of asr → mux for old folders.

### ADR-030: Spectrum match of the dub to the original speaker
- **Context:** Against the separated vocals the StyleTTS2 voice measured +6 dB at 80–160 Hz and
  −3 … −9 dB from 315 Hz to 11.5 kHz: boomy and dull, sitting apart from the background.
- **Decision:** `audio.match_spectrum` compares the long-term spectra of the louder frames of the
  dub and of `vocals.wav` in 1/3-octave bands (80 Hz–10 kHz), clamps the difference to ±6 dB and
  applies it as a zero-phase FIR before loudness matching. Only when the source was separated
  (without separation the reference would contain music). Mix version 3.
- **Consequences:** The dub takes the microphone's tonal balance, not its reverb or noise. The clamp
  keeps the TTS from being pushed into hiss above its own bandwidth.

### ADR-031: One pronunciation per name
- **Context:** The LLM fills the `tts` field of each line on its own, so the same name came out
  differently from line to line: «ел ем студіо», «ель ем студіо», «ел-ем студіо». The roundtrip check
  showed the loose forms slurred (the ASR heard «лмстіо» five times) while textnorm's hyphenated
  letter spelling was heard as «LM Studio» six times out of seven.
- **Decision:** `translate.unify_pronunciations` runs after all translation passes. For every Latin
  term in `uk` it finds the pronunciation inside `tts` (the span between the neighbouring Cyrillic
  words). A term that contains an acronym (`LM`, `MLX`, `API`, `PC`) takes textnorm's spelling in
  every line; any other name that occurs in several lines takes the most frequent LLM spelling.
- **Consequences:** Names sound the same throughout a video. On the test clip 6 lines were changed.
  Replacements within one line are applied right to left, so two names in a line keep their offsets.
  `_clean_tts` also drops an LLM `tts` that contains letters of another script (Gemma occasionally
  emits a CJK character mid-word); the deterministic normalizer then reads the subtitle text.

### ADR-032: Prompt revision for length, spoken style and consistency
- **Context:** On the test clip 23 of 28 lines came back over budget and went through two shortening
  passes that cut facts («Це те саме, що й нижче» lost what to click). Four lines in a row began with
  «Тож», two neighbours began with the same phrase, the brief chose «ти» for a tutorial, and version
  numbers were read «три кома шість». LLMs do not count syllables; they count words reasonably well.
- **Decision:**
  - Every line carries `max_words` (= `max_syl` / 2.4) next to `max_syl`; the length rule says "at
    most max_words, shorter is fine", not "close to max_syl", and gives the order in which to condense.
  - New rules: spoken style (short sentences, verbs instead of genitive chains, no participial
    constructions, spoken connectives) and continuity (no two neighbouring lines starting with the
    same word, no phrase repeated from the previous line, varied connectives). English idiom examples
    moved to `LANG_RULES["en"]`.
  - The brief gets a `style` field (register, humour, how the speaker addresses the viewer), a rule
    for `address` («ви» for an audience, «ти» only between close people; default «ви»), and a `say`
    pronunciation per Latin glossary name, which `unify_pronunciations` uses first (ADR-031).
  - `SHORTEN_SYSTEM` sees the neighbouring lines, the summary and style, the syllable surplus and
    `max_words`, cuts in a fixed order (fillers → repetition → synonyms → rebuild) and never drops
    facts, labels or the object of an instruction; the second pass is told it is the second pass.
    A rewrite is rejected by `translate.lost_facts` when a quoted label, a number or a Latin name of
    the previous version is missing.
  - The `tts` rule spells abbreviations with hyphens and versions digit by digit. Translate version 15.
- **Consequences:** Fewer and safer shortenings; a consistent voice across lines. Measured by the number
  of lines sent to shortening, repeated line starts and a manual read of the 28 lines.

---

## 5. Rejected or deferred alternatives

| Alternative | Why not (yet) |
|---|---|
| Stress via "rules" in the prompt | Ukrainian stress is lexical; the dictionary plus lookup is more reliable (ADR-016) |
| ARPAbet for every OmniVoice word | Adds an English accent; restricted to homographs and user words (ADR-013) |
| Batch OmniVoice synthesis | Slower on MPS because of padding; batch = 1 |
| Cross-language `clone` for dramas | Strong accent (CER 0.33); `duo` / `duo:st` recommended |
| Contextual ByT5 stress model (lang-uk ukrainian-tts-preprocessing, 92.5 % word accuracy) | Researched as a reference; dictionary + Stanza + LLM homographs chosen instead |
| Higgs TTS with emotion tags | Heavy; better suited to the RTX desktop; not integrated |
| Third shortening pass for lines far over budget | Would bump translate (minutes per video); ADR-028 lets such lines spill instead of rushing them, measure first |
| Online stress dictionaries (goroh, r2u) | Breaks offline; licensing |
| Lip-sync, multi-speaker diarization | Out of scope for now |
| CUDA / RTX desktop support | MLX-based pipeline; would need a port |

---

## 6. Quality evidence

| Check | Result |
|---|---|
| English sample, round trip | WER ~0.10–0.12 |
| OmniVoice clip after the pacing fix | CER 0.067 / WER 0.125; 0 drawn-out lines |
| StyleTTS2 clip | CER 0.069–0.075; stresses verified |
| StyleTTS2 voice scan | 30/31 voices CER 0.000 on a test sentence |
| Korean `duo` / `duo:st --emotion` | CER 0.023 / 0.011; pitch gender 16/16 |
| `--emotion` pitch range | 4.6 → 6.7 semitones (original 6.9) |
| Unit tests | 26 passing (`tests/test_logic.py`) |

Tools: `tests/roundtrip.py` (intelligibility), `tests/pace.py` (syl/s), and sample generators for EN/KO.

---

## 7. CLI reference (summary)

| Option | Meaning |
|---|---|
| `--voice` | `st`, `st:<name>`, `duo:st[:m,f]`, `clone`, `clone:file`, `omni:<desc>`, `duo`, presets, `duo:a,b` |
| `--emotion [K]` | StyleTTS2 takes the original intonation (0–1, default 1.0) |
| `--from LANG`, `--subs FILE`, `--subs-lang LANG` | source language / existing subtitles |
| `--llm M` | MLX model, `ollama:<m>`, or `claude|opencode|codex|gemini[:model]` (env `UADUB_LLM`) |
| `--review`, `--review-with H[:M]`, `--redo review` | human pause / agent review (env `UADUB_REVIEW_WITH`) |
| `--text` | `.txt` transcript, translation, stressed text, bilingual |
| `--stress auto|dict|off`, `--stress-dict F`, `--stress-lookup W…` | stress control and dictionary lookup |
| `--glossary F`, `--gender m/f` | terminology, grammar gender |
| `--domain [FIELD]` | specialist translation for the detected or given field (ADR-026) |
| `--part-minutes N`, `--keep-parts` | long videos in parts (ADR-027) |
| `--no-separate`, `--fast`, `--duck DB`, `--max-speed X`, `--steps N` | audio and pacing |
| `--drop-original`, `-o`, `--workdir`, `--redo STAGE`, `--stop-after STAGE` | output and pipeline control |
| `--prefetch [--all]`, `--list-voices` | setup |

---

## 8. Open items and ideas

- Measure the actual peak RAM per stage and publish the numbers in the README (currently estimates).
- Some StyleTTS2 lines with brand names run at ~4.2–4.4 syl/s.
- Decimal numbers in versions ("3.6" → «три шість») could be spoken as «три крапка шість».
- An agent may edit read-only `НАГОЛОСИ:` lines; this is harmless (ignored) but could be warned about.
- Possible: speaker diarization for more than two voices, a CUDA port for the RTX desktop, and Higgs
  TTS emotion control.
