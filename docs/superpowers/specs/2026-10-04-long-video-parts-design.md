# Long videos in parts — design

Date: 2026-10-04. Status: approved in conversation, awaiting spec review.

## 1. Goal

Dub recordings of 4–7 hours with one command. Today the user cuts a 4.5-hour OBS recording into 21
pieces in LosslessCut, dubs each with `transcribe.sh`, and ends up with 21 separate files whose
terminology drifts between pieces.

Success means all four of:

1. **No manual cutting.** One `uadub long.mp4` call.
2. **One result.** One `long.uk.mp4` and one `long.uk.srt`, in sync over the full length.
3. **Bounded resources.** Peak RAM does not grow with the video length, and a failure in one part
   does not throw away the work on the others. It must also run on smaller Apple Silicon Macs
   (16 GB with a smaller or cloud `--llm`).
4. **Early preview.** Finished parts can be watched while the rest is still processing.

Non-goals: making the pipeline faster (≈3.2× the video length on the user's Mac, so ~20–23 h for
7 h); platforms other than macOS on Apple Silicon; `--subs` in part mode (see §8).

## 2. Approach

Only the **audio** is cut, never the video. The audio is split at pauses into parts of about
15 minutes. Each part goes through the existing stages in its own work folder. The finished part
audio files are concatenated, and the result is muxed once over the untouched original video.

Rejected alternatives:

- **Cut video files and concatenate finished `.uk.mp4`.** Stream copy can only cut at keyframes,
  which rarely fall in pauses. AAC priming can click at the joins. Each piece gets its own brief,
  so terminology drifts.
- **Streaming stages without parts.** This means rewriting separation, mixing, clone and emotion to
  process in windows. That is the same splitting, done inside every stage. It gives neither early
  preview nor failure isolation, because the cache works per stage.

Facts this design relies on:

- Separation, mixing, `clone` and `--emotion` load the whole track as 44.1 kHz stereo float32. That
  is about 1.27 GB per hour per copy, with up to 4 copies at once.
- `make_brief` keeps only the first and last 12,000 characters of the transcript
  (`BRIEF_CHAR_LIMIT`), so the middle of a long video never reaches the brief.
- The LLM (~15 GB for the default Gemma 26B 4-bit) is the largest memory consumer. The audio of a
  15-minute part is not.

## 3. Activation and CLI

- Without the flag, part mode turns on automatically when the input is longer than **45 minutes**,
  with parts of **15** minutes.
- `--part-minutes N` sets the target part length and turns part mode on whenever the input is longer
  than 1.5 × N. A 20-minute video with `--part-minutes 5` is split; with `--part-minutes 15` it is
  not.
- `--part-minutes 0` disables part mode.
- `Options.part_minutes` is `None` when the flag is not given, which means automatic.
- `--keep-parts` keeps the preview videos after the final assembly.
- If part mode is active, the startup line says so. If `--review` (a human pause) is combined with
  part mode, it suggests `--review-with`, because there will be one pause per part.
- With `--subs`, part mode exits with a plain message that suggests `--part-minutes 0`.

New `Options` fields:

- `part_minutes: float | None = None`: `None` is automatic, `0` is off;
- `keep_parts: bool = False`;
- `clip_start: float | None = None` and `clip_end: float | None = None`: the part's range in the
  input;
- `shared_brief: str | None = None`: path to the long-video brief;
- `edge_context: str | None = None`: path to the neighbouring parts' text;
- `loudness_target: float | None = None`.

Each new field joins a stage fingerprint only when it is set, so the caches of normal runs survive.

## 4. Cut points

The new module `uadub/parts.py` owns planning, orchestration and assembly.

1. **Loudness.** ffmpeg decodes the input to 16 kHz mono s16 and pipes it to Python. Python reads it
   in blocks and keeps one RMS value in dB per 50 ms frame. For 7 h that is about 504,000 floats,
   under 1 MB. The pipe uses the same `-max_error_rate 1.0` tolerance as `extract_audio`.
2. **Threshold.** The threshold adapts to the recording, as in `segments.snap_to_speech`:
   `max(p10 + 12, p99 - 35)` dB. A frame is silent below it.
3. **Cuts.** The plan looks near each target `k × part_minutes`:
   - It picks the **longest run of silent frames** within ±3 minutes of the target and cuts at the
     middle of that run.
   - If no run reaches 0.5 s, it cuts at the quietest 1-second window in the range and records a
     warning.
   - If the last part would be shorter than a third of the target, it merges into the previous part.
4. **Sample alignment.** Cut times are rounded to whole samples at `sample_rate` (44,100).
5. **The plan.** It is written to `<stem>.uadub/plan.json`:
   ```json
   {"input": "...", "duration": 16619.487, "sample_rate": 44100, "part_minutes": 15,
    "parts": [{"n": 1, "start": 0.0, "end": 892.4, "pause": 1.8, "fallback": false}, ...]}
   ```
   The plan is reused while `input`, `duration`, `sample_rate` and `part_minutes` match, so a rerun
   keeps every part's cache valid. The console prints it as a table, as in the design discussion.

## 5. Folders

```
long.uadub/                 shared folder: plan.json, long_brief.json, edge/, state.json, options.json
  parts/01/ … parts/NN/     ordinary uadub work folders, one per part, each with its own state.json
long.uk.parts/              previews NN.uk.mp4 (+ .srt); deleted after assembly unless --keep-parts
long.uk.mp4, long.uk.srt    final result
```

## 6. Processing

`run_pipeline` gets a branch: if part mode is active, `parts.run_long(opt, …)` takes over. Parts
reuse `run_pipeline` with a per-part `Options` (`input` is the original file, plus `clip_start`,
`clip_end` and `workdir=parts/NN`).

### Phase 1 — listen (all parts)

- For each part, run `extract → separate → asr` (`stop_after="asr"`).
- `stage_extract` honours `clip_start`/`clip_end`. In the normal ffmpeg call and in
  `_salvage_audio` it uses input `-ss start` and `-t length`. It then pads or trims the WAV to
  exactly `round(length × sr)` samples.
- After `separate`, measure the part's speech loudness from `vocals.wav` and store it in the part's
  `meta.json`.
- Parts are processed one after another, so only one heavy model is resident (ADR-002 holds).

### Phase 2 — shared brief

- Concatenate the transcripts of all parts in order, then split them into chunks of at most
  `BRIEF_CHAR_LIMIT` characters at line boundaries.
- **Map:** run `make_brief` on each chunk with the same `src_name`, `lang_rules` and `domain`.
- **Reduce:** a new `MERGE_BRIEF_SYSTEM` prompt asks the LLM to merge the partial briefs into one:
  - one summary of the whole video;
  - glossary entries deduplicated with one rendering each (25 at most, or 40 with `--domain`);
  - characters, idioms and asr_fixes merged;
  - `speaker_gender` and `address` decided by majority;
  - one `domain`.
- **Fallback:** if the reduce call fails, merge deterministically: concatenate the summaries, keep
  the first glossary entry for each `src`, and take a majority vote for gender and address.
- **Short videos:** with a single chunk, its brief is used directly.
- Save the result to `long.uadub/long_brief.json`. The name differs from the parts' own `brief.json`, which has another format. The shared folder's `state.json` stores its
  fingerprint: the hash of all part transcripts plus the translate options. `--redo translate`
  rebuilds it.

### Phase 3 — make each part

- **Edge context.** Write `edge/NN.json` with the last ~2,500 characters of the previous part's
  transcript and the first ~2,500 of the next.
- **Translate.**
  - `translate_units` gets `shared_brief`. When it is given, the brief is used as is and
    `make_brief` is skipped.
  - `_context_window` adds the edge text before the first line and after the last one.
  - The part's translate fingerprint includes the hash of `long_brief.json` and of its edge file.
- **Review.** `--review-with` runs per part as today. `--review` pauses per part.
- **TTS.** Unchanged.
- **Mix.**
  - With `loudness_target` set, the mix uses it instead of the part's own measurement. The target
    is the median of the per-part measurements from phase 1, clipped to −26…−12 LUFS.
  - The mix output has exactly `round(length × sr)` samples.
- **Part output.** `mux` is replaced by `stage_part_out`:
  - Write `dub.flac` (24-bit) from `mix.wav`.
  - Build the preview `long.uk.parts/NN.uk.mp4`: `-ss start -t length` on the original video,
    re-encoded with `h264_videotoolbox` (falling back to `libx264 -preset veryfast`). The dub audio
    is the default track, the original audio of the range is the second track, and the part's
    `uk.srt` is embedded.
  - Copy the part's `uk.srt` next to the preview.
- **Disk cleanup.**
  - Delete `audio.wav` and `mix.wav`.
  - Convert `vocals.wav` and `background.wav` to 24-bit FLAC and delete the WAVs.
  - `A.read` and every stage that opens these files go through a helper `A.existing(path)`, which
    returns the `.wav` or the `.flac` that exists. A later redo of `tts`/`mix` after a review edit
    still works.
- **Failure.** An exception in one part is caught and recorded in the shared `state.json` with the
  last error line. Processing continues with the next part.

### Assembly

- Assembly runs when every part has `dub.flac` and no part failed.
- Otherwise print the list of failed parts and how to resume (run the same command), and exit with a
  non-zero code.
- **Audio.** Use the ffmpeg concat demuxer over the `dub.flac` files, encode to AAC 192k once, and
  mux with the original video (`-c:v copy`), with the original audio as the second track when
  `keep_original` is set. The same `A.mux` path is used, so the subtitles go in as `mov_text` for mp4.
- **Subtitles.** Offset the cues of each part's `uk.srt` by the part's start and write them to
  `long.uadub/uk.srt`, renumbered. The source-language SRT is merged the same way for `--text`.
- **`--text`.** `write_texts` runs on the merged units: each part's `units.json` with times shifted
  by the part's start and ids renumbered.
- **Cleanup.** Delete `long.uk.parts/` unless `--keep-parts` is given.
- **State.** The shared `state.json` stores the assembly fingerprint: the hashes of all `dub.flac`
  files and `uk.srt` files. A rerun after one part changes re-assembles.

### Resuming and redo

- Rerunning the same command resumes. Parts whose fingerprints match are skipped by their own
  `state.json`. Failed parts are retried.
- After a manual edit of `parts/NN/review.md`, that part's review hash changes, so it redoes
  `tts → mix → part output`, and the assembly reruns.
- `--redo STAGE` is applied to every part. `--redo translate` (or an earlier stage) also rebuilds
  the shared brief.
- `--stop-after asr` stops after phase 1. `--stop-after translate` stops after every part is
  translated.

## 7. Console

```
Довге відео (4:36:59) → 19 частин по ~15 хв, розрізи в паузах:
  01  0:00:00–0:14:52   пауза 1,8 с
  …
Фаза 1/3: слухаю частини
Частина 1/19 · [1/7] Витягую аудіо…
…
Фаза 2/3: спільний бриф (12 шматків тексту)
Фаза 3/3: готую частини
Частина 1/19 · [4/7] Перекладаю українською…
   • превʼю: long.uk.parts/01.uk.mp4
…
Склеюю 19 частин → long.uk.mp4
```

## 8. Limits and later work

- `--subs` in part mode: cues would need filtering per part with offsets. Deferred.
- The first preview appears only after phase 1. That is the price of a shared brief.
- Disk: about 10 GB in `long.uadub/` for 7 hours (FLAC stems and dub), plus previews of roughly the
  original's size until assembly.
- Background music during a pause can hide it. The fallback (the quietest window) keeps cuts
  reasonable.

## 9. Testing

Unit tests in `tests/test_logic.py` (no models):

- Cut finder on synthetic dB arrays:
  - it picks the longest pause in the window and cuts at its middle;
  - it falls back to the quietest window;
  - it merges a short tail;
  - cuts are sample-aligned and the part lengths sum exactly to the total.
- The plan is reused when its inputs match and rebuilt when `part_minutes` changes.
- SRT merge applies offsets and renumbers.
- Brief merge with a fake LLM covers chunking, reduce, deduplication, the majority vote and the
  deterministic fallback when the reduce call fails.
- Part fingerprints:
  - `clip_start`/`clip_end` affect extract;
  - the shared brief and edge hashes affect translate;
  - `loudness_target` affects mix;
  - none of these appear in a normal run's fingerprints.
- `A.existing` prefers `.wav`, then `.flac`.

Manual end-to-end check:

- Build a ~20-minute video by concatenating `examples/ab/clip.mp4` and run it with
  `--part-minutes 5`.
- Check that:
  - the output duration equals the input to within one frame;
  - the joins sound clean;
  - the subtitles stay in sync at the last part;
  - an interrupted run resumes;
  - a `review.md` edit in one part redoes only that part.

The 4.5-hour recording (~15 h) is a run the user starts, not part of this work.

## 10. Documentation

- `spec.md`: ADR-027 (long videos in parts), a new FR row, and an NFR note that memory is bounded by
  the part length.
- `README.md` and `README.uk.md`: a "Long videos" section and new rows in the options table.
