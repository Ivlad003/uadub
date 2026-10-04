"""Long videos: cut the audio at pauses, dub the parts one by one, assemble one result (ADR-027)."""

from __future__ import annotations

import json
import math
import subprocess
import tempfile
from pathlib import Path

import numpy as np

from . import audio as A

FRAME = 0.05  # loudness frame, s
AUTO_MINUTES = 15.0  # part length when splitting automatically
AUTO_ABOVE_MINUTES = 45.0  # inputs longer than this are split automatically
WINDOW = 180.0  # a cut is searched within ±WINDOW s of its target
MIN_PAUSE = 0.5  # shorter quiet runs are not pauses


def part_minutes_for(duration_s: float, part_minutes: float | None) -> float:
    """Target part length in minutes, or 0 when the input is processed whole."""
    if part_minutes is None:
        return AUTO_MINUTES if duration_s > AUTO_ABOVE_MINUTES * 60 else 0.0
    if part_minutes <= 0:
        return 0.0
    return float(part_minutes) if duration_s > 1.5 * part_minutes * 60 else 0.0


def choose_part_minutes(opt, duration_s: float, *, log=print) -> float:
    """part_minutes_for, plus the cases where an automatic split would hurt: then the input stays whole."""
    minutes = part_minutes_for(duration_s, opt.part_minutes)
    if not minutes:
        return 0.0
    if opt.part_minutes is not None:  # asked for explicitly
        if opt.subs:
            raise SystemExit("--subs поки не працює з нарізкою довгого відео на частини. Додайте --part-minutes 0.")
        return minutes
    if opt.subs:
        log("Примітка: з --subs довге відео обробляється цілим (нарізка на частини з --subs поки не працює).")
        return 0.0
    from .config import STAGES

    work = Path(opt.workdir)
    try:
        state = json.loads((work / "state.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        state = {}
    if not (work / "plan.json").exists() and isinstance(state, dict) and any(s in state for s in STAGES):
        log(f"Примітка: продовжую попередній запуск цілим файлом (нарізати на частини: "
            f"--part-minutes {AUTO_MINUTES:g}).")
        return 0.0
    return minutes


def frame_db(video: Path, total: float, frame: float = FRAME) -> np.ndarray:
    """RMS level (dB) per frame of the whole input, streamed: under 1 MB for 7 hours.

    Frames the decoder never delivered (a damaged track that ends early) are NaN."""
    sr = 16000
    hop = int(sr * frame)
    vals: list[np.ndarray] = []
    carry = np.zeros(0, np.float32)
    odd = b""
    with tempfile.TemporaryFile() as err:  # a file, not a PIPE: nothing to drain, no deadlock
        proc = subprocess.Popen(
            ["ffmpeg", "-nostdin", "-v", "error", "-max_error_rate", "1.0", "-i", str(video), "-vn",
             "-af", "aresample=async=1:first_pts=0", "-ac", "1", "-ar", str(sr), "-f", "s16le", "pipe:1"],
            stdout=subprocess.PIPE, stderr=err)
        try:
            while True:
                buf = proc.stdout.read(1 << 20)
                if not buf:
                    break
                buf = odd + buf
                odd, buf = (buf[-1:], buf[:-1]) if len(buf) % 2 else (b"", buf)
                y = np.concatenate([carry, np.frombuffer(buf, "<i2").astype(np.float32) / 32768.0])
                m = len(y) // hop
                if m:
                    f = y[: m * hop].reshape(m, hop)
                    vals.append(10 * np.log10(np.mean(f * f, axis=1) + 1e-10))
                carry = y[m * hop:]
        finally:
            if proc.poll() is None:
                proc.kill()
            proc.stdout.close()
            proc.wait()
        err.seek(0)
        tail = err.read().decode("utf-8", "replace").strip().splitlines()[-3:]
    db = np.concatenate(vals) if vals else np.zeros(0)
    if not len(db):  # a nonzero exit alone is fine (damaged track, ADR-025); nothing decoded is not
        raise SystemExit(f"Не вдалося прочитати аудіо з {Path(video).name}"
                         + (": " + " | ".join(tail) if tail else ""))
    n = math.ceil(round(total / frame, 6))
    if len(db) < n:
        db = np.concatenate([db, np.full(n - len(db), np.nan)])
    return db[:n]


def silence_threshold(db: np.ndarray) -> float:
    """Adaptive, as in segments.snap_to_speech: relative to this recording's floor and peaks."""
    v = db[np.isfinite(db)]
    if not len(v):
        return -np.inf
    return float(max(np.percentile(v, 10) + 12, np.percentile(v, 99) - 35))


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    d = np.diff(np.concatenate([[0], mask.astype(np.int8), [0]]))
    return list(zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1)))


def find_cuts(db: np.ndarray, total: float, part_s: float, *, frame: float = FRAME,
              window: float = WINDOW, min_pause: float = MIN_PAUSE) -> list[dict]:
    """Cut times between parts: the middle of the longest pause near each target, else the quietest second."""
    with np.errstate(invalid="ignore"):
        silent = db < silence_threshold(db)  # NaN → not silent
    runs = _runs(silent)
    level = np.where(np.isfinite(db), db, np.inf)
    w1 = max(1, int(round(1.0 / frame)))
    cuts: list[dict] = []
    prev = 0.0
    while total - prev > part_s * 4 / 3:  # otherwise the rest is the last part
        target = prev + part_s
        lo = max(target - window, prev + part_s / 3)
        hi = min(target + window, total - part_s / 3)
        best = None
        for a, b in runs:
            s, e = max(a * frame, lo), min(b * frame, hi)
            if e - s >= min_pause - 1e-9:
                key = (round(e - s, 6), -abs((s + e) / 2 - target))
                if best is None or key > best[0]:
                    best = (key, (s + e) / 2, e - s)
        if best:
            cut, pause, fallback = best[1], best[2], False
        else:
            i0, i1 = int(lo / frame), max(int(lo / frame) + w1, int(hi / frame))
            seg = level[i0:i1]
            avg = np.convolve(seg, np.ones(w1) / w1, mode="valid") if len(seg) >= w1 else seg
            avg = np.round(avg, 2)  # float noise must not beat "nearest the target" among equal windows
            centres = (i0 + np.arange(len(avg)) + w1 / 2) * frame
            i = int(np.lexsort((np.abs(centres - target), avg))[0])
            cut, pause, fallback = float(centres[i]), 0.0, True
        cuts.append({"time": float(cut), "pause": round(float(pause), 2), "fallback": fallback})
        prev = cut
    return cuts


def make_plan(db: np.ndarray, total: float, part_minutes: float, sr: int, *, frame: float = FRAME) -> dict:
    cuts = find_cuts(db, total, part_minutes * 60, frame=frame)
    bounds = [0.0] + [round(c["time"] * sr) / sr for c in cuts] + [total]
    parts = []
    for i in range(len(bounds) - 1):
        end_cut = cuts[i] if i < len(cuts) else {"pause": None, "fallback": False}
        parts.append({"n": i + 1, "start": bounds[i], "end": bounds[i + 1],
                      "pause": end_cut["pause"], "fallback": end_cut["fallback"]})
    return {"duration": total, "sample_rate": sr, "part_minutes": part_minutes, "parts": parts}


def load_or_make_plan(work: Path, video: Path, part_minutes: float, sr: int, *, log=print) -> dict:
    """The plan is fixed per input and settings, so a rerun keeps every part's cache."""
    path = Path(work) / "plan.json"
    total = A.duration(video)
    if path.exists():
        try:
            plan = json.loads(path.read_text(encoding="utf-8"))
        except ValueError:  # truncated or corrupt: plan again
            plan = {}
        if not isinstance(plan, dict):  # valid JSON, but not a plan
            plan = {}
        if (plan.get("input") == str(video) and abs(plan.get("duration", -1) - total) < 1e-3
                and plan.get("sample_rate") == sr and plan.get("part_minutes") == part_minutes):
            return plan
    log("   • шукаю паузи для розрізів…")
    plan = make_plan(frame_db(video, total), total, part_minutes, sr)
    plan["input"] = str(video)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(plan, ensure_ascii=False, indent=2), encoding="utf-8")
    return plan


def _hms(t: float) -> str:
    t = int(round(t))
    return f"{t // 3600}:{t % 3600 // 60:02d}:{t % 60:02d}"


def parts_word(n: int) -> str:
    """«N частина/частини/частин» with the Ukrainian plural."""
    if n % 10 == 1 and n % 100 != 11:
        word = "частина"
    elif n % 10 in (2, 3, 4) and n % 100 not in (12, 13, 14):
        word = "частини"
    else:
        word = "частин"
    return f"{n} {word}"


def format_plan(plan: dict) -> str:
    parts = plan["parts"]
    lines = [f"Довге відео ({_hms(plan['duration'])}) → {parts_word(len(parts))} по ~{plan['part_minutes']:g} хв, "
             "розрізи в паузах:"]
    for p in parts:
        if p["pause"] is None:
            note = ""
        elif p["fallback"]:
            note = "   ! пауз немає — найтихіше місце"
        else:
            note = f"   пауза {p['pause']:.1f} с".replace(".", ",")
        lines.append(f"  {p['n']:02d}  {_hms(p['start'])}–{_hms(p['end'])}{note}")
    return "\n".join(lines)


_TIME_KEYS = ("start", "end", "slot_end", "dub_start", "dub_end")


def merged_units(items: list[tuple[Path, float]]) -> list[dict]:
    """Units of all parts on the timeline of the whole input."""
    out: list[dict] = []
    for work, offset in items:
        for u in json.loads((Path(work) / "units.json").read_text(encoding="utf-8")):
            v = dict(u)
            for k in _TIME_KEYS:
                if isinstance(v.get(k), (int, float)):
                    v[k] = round(v[k] + offset, 3)
            v["id"] = len(out) + 1
            out.append(v)
    return out


def assemble(video: Path, items: list[tuple[Path, float]], out: Path, master: Path, *,
             src_lang: str, keep_original: bool) -> None:
    """One result: the parts' dub audio concatenated by ffmpeg (streamed, encoded once) over the original video."""
    import shutil

    from .srt import make_cues, write_srt

    master = Path(master)
    master.mkdir(parents=True, exist_ok=True)
    units = merged_units(items)
    (master / "units.json").write_text(json.dumps(units, ensure_ascii=False), encoding="utf-8")
    write_srt(make_cues(units, "uk"), master / "uk.srt")
    write_srt(make_cues(units, "text"), master / f"{src_lang}.srt")
    lst = master / "concat.txt"
    esc = lambda p: str(Path(p).resolve()).replace("'", "'\\''")
    lst.write_text("".join(f"file '{esc(Path(w) / 'dub.flac')}'\n" for w, _ in items), encoding="utf-8")
    whole = master / "dub_all.flac"
    try:
        A._run(["ffmpeg", "-nostdin", "-y", "-f", "concat", "-safe", "0", "-i", str(lst), "-c:a", "flac", str(whole)])
        A.mux(Path(video), whole, Path(out), srt=master / "uk.srt", keep_original=keep_original)
    finally:
        whole.unlink(missing_ok=True)
        lst.unlink(missing_ok=True)
    shutil.copyfile(master / "uk.srt", Path(out).with_suffix(".srt"))


EDGE_CHARS = 2500
BRIEF_VERSION = 1  # bump when the long-brief prompts change


def part_dir(master: Path, n: int) -> Path:
    return Path(master) / "parts" / f"{n:02d}"


def previews_dir(opt) -> Path:
    out = Path(opt.output)
    return out.with_name(f"{out.stem}.parts")


def part_options(opt, part: dict, master: Path):
    from dataclasses import replace

    out = Path(opt.output)
    return replace(opt, workdir=str(part_dir(master, part["n"])), clip_start=part["start"], clip_end=part["end"],
                   output=str(previews_dir(opt) / f"{part['n']:02d}.uk{out.suffix}"),
                   shared_brief=None, edge_context=None, loudness_target=None)


def _transcript(work: Path) -> list[str]:
    return [s["text"] for s in json.loads((Path(work) / "transcript.json").read_text(encoding="utf-8"))]


def speech_loudness(work: Path, sr: int) -> float | None:
    """Loudness of the part's original speech, measured once and kept in meta.json."""
    meta_path = Path(work) / "meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    if "speech_lufs" not in meta:
        src = A.existing(Path(work) / "vocals.wav")
        if not src.exists():
            src = A.existing(Path(work) / "audio.wav")
        meta["speech_lufs"] = A.loudness(A.read(src, sr=sr)[0], sr)
        meta_path.write_text(json.dumps(meta), encoding="utf-8")
    return meta["speech_lufs"]


def loudness_target(values: list[float | None]) -> float | None:
    v = [x for x in values if x is not None]
    return float(np.clip(np.median(v), -26.0, -12.0)) if v else None


def brief_key(opt, works: list[Path]) -> str:
    import hashlib

    h = hashlib.sha1()
    for w in works:
        h.update((Path(w) / "transcript.json").read_bytes())
    h.update(json.dumps([BRIEF_VERSION, opt.llm, opt.domain, opt.text_lang]).encode())
    return h.hexdigest()[:16]


def _tail(lines: list[str], n: int = EDGE_CHARS) -> str:
    out: list[str] = []
    for line in reversed(lines):
        if out and sum(map(len, out)) + len(line) > n:
            break
        out.insert(0, line)
    return "\n".join(out)


def _head(lines: list[str], n: int = EDGE_CHARS) -> str:
    out: list[str] = []
    for line in lines:
        if out and sum(map(len, out)) + len(line) > n:
            break
        out.append(line)
    return "\n".join(out)


def edge_path(master: Path, n: int) -> Path:
    return Path(master) / "edge" / f"{n:02d}.json"


def write_edges(master: Path, works: list[Path]) -> None:
    texts = [_transcript(w) for w in works]
    for i in range(len(works)):
        edge = {"before": _tail(texts[i - 1]) if i else "", "after": _head(texts[i + 1]) if i + 1 < len(texts) else ""}
        p = edge_path(master, i + 1)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(edge, ensure_ascii=False), encoding="utf-8")


def _brief_child(master: str, works: list[str]) -> None:
    from .cli import _env_defaults
    from .config import LANGS, Options
    from .llm import make_llm
    from .translate import LANG_RULES, long_brief

    _env_defaults()
    try:  # the model's "Fetching N files" bar is noise here (it showed up as stray lines)
        from huggingface_hub.utils import disable_progress_bars

        disable_progress_bars()
    except Exception:
        pass
    log = lambda msg: print(msg, flush=True)
    opt = Options.load(Path(master) / "options.json")
    units = [{"text": t} for w in works for t in _transcript(Path(w))]
    log(f"   • модель: {opt.llm}")
    llm = make_llm(opt.llm)
    try:
        brief = long_brief(llm, units, LANGS.get(opt.text_lang, (None, opt.text_lang))[1],
                           LANG_RULES.get(opt.text_lang, ""), opt.domain, log=log)
    finally:
        llm.close()
    (Path(master) / "long_brief.json").write_text(json.dumps(brief, ensure_ascii=False, indent=2), encoding="utf-8")
    log(brief_summary(brief))


def brief_summary(brief: dict) -> str:
    glossary = brief.get("glossary") if isinstance(brief.get("glossary"), list) else []
    domain = str(brief.get("domain") or "").strip()
    return ("   • бриф готовий" + (f": сфера «{domain}»" if domain else "")
            + f", термінів у глосарії: {len(glossary)}")


def build_shared_brief(master: Path, works: list[Path]) -> None:
    """In a spawn subprocess, like the heavy stages: the LLM leaves memory when it is done (ADR-002)."""
    import multiprocessing as mp

    proc = mp.get_context("spawn").Process(target=_brief_child, args=(str(master), [str(w) for w in works]))
    proc.start()
    try:
        proc.join()
    except KeyboardInterrupt:
        proc.terminate()
        proc.join()
        raise
    if proc.exitcode != 0:
        raise SystemExit(f"\nСпільний бриф не вдався (код {proc.exitcode}). Подробиці вище; "
                         "запустіть ту саму команду ще раз.")


def assembly_key(works: list[Path], keep_original: bool = True, out: str | Path = "") -> list:
    key: list = [["keep_original", bool(keep_original)], ["out", str(out)]]
    for w in works:
        for name in ("dub.flac", "units.json"):
            st = (Path(w) / name).stat()
            key.append([f"{Path(w).name}/{name}", st.st_size, st.st_mtime_ns])
    return key


def _each_part(opt, plan: dict, master: Path, fn, record=None, total: int | None = None) -> dict[int, str]:
    """Run fn(part_options, n, prefix) for every part; a failed part does not stop the others."""
    failed: dict[int, str] = {}
    total = total or len(plan["parts"])
    for p in plan["parts"]:
        tag = f"Частина {p['n']}/{total} · "
        print(f"\n{tag}{_hms(p['start'])}–{_hms(p['end'])}", flush=True)
        try:
            fn(part_options(opt, p, master), p["n"], tag)
        except (Exception, SystemExit) as e:  # KeyboardInterrupt still stops everything
            from .cli import UserStop

            if isinstance(e, UserStop):
                raise
            msg = (str(e).strip().splitlines() or [type(e).__name__])[-1]
            failed[p["n"]] = msg
            print(f"   ! частина {p['n']} не вдалася: {msg}", flush=True)
        if record:
            record(p["n"], failed.get(p["n"]))
    return failed


def _stop_if_failed(failed: dict[int, str]) -> None:
    if failed:
        lines = "\n".join(f"  {n:02d}: {msg}" for n, msg in sorted(failed.items()))
        raise SystemExit(f"\nНе вдалися частини ({len(failed)}):\n{lines}\n"
                         "Запустіть ту саму команду ще раз — готові частини буде пропущено.")


def run_long(opt, plan: dict, *, redo: str | None = None, stop_after: str | None = None,
             review: bool = False, review_with: str | None = None, text: bool = False) -> None:
    import shutil
    import time

    from . import cli
    from .term import link

    master = opt.work
    master.mkdir(parents=True, exist_ok=True)
    opt.save(master / "options.json")
    state_path = master / "state.json"
    state = json.loads(state_path.read_text(encoding="utf-8")) if state_path.exists() else {}
    save = lambda: state_path.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
    works = [part_dir(master, p["n"]) for p in plan["parts"]]

    def record(n, msg):
        failures = state.setdefault("failed", {})
        if msg is None:
            failures.pop(str(n), None)
        else:
            failures[str(n)] = msg
        save()

    early = redo in ("extract", "separate", "asr")
    if stop_after == "mux":  # the last stage: a part's mux is only its preview, so finish the whole job
        stop_after = None
    t_all = time.time()
    print(format_plan(plan), flush=True)

    print("\nФаза 1/3: слухаю частини", flush=True)
    _stop_if_failed(_each_part(opt, plan, master, lambda po, n, tag: cli.run_pipeline(
        po, redo=redo if early else None, stop_after=stop_after if stop_after in ("extract", "separate") else "asr",
        prefix=tag, summary=False), record))
    if stop_after in ("extract", "separate", "asr"):
        print(f"\nЗупинено після етапу «{stop_after}» для всіх частин.")
        return
    levels, failed = [], {}
    for p, w in zip(plan["parts"], works):
        try:
            levels.append(speech_loudness(w, opt.sample_rate))
        except Exception as e:
            failed[p["n"]] = (str(e).strip().splitlines() or [type(e).__name__])[-1]
            print(f"   ! частина {p['n']}: не вдалося виміряти гучність: {failed[p['n']]}", flush=True)
            record(p["n"], failed[p["n"]])
    _stop_if_failed(failed)
    target = loudness_target(levels)

    print("\nФаза 2/3: спільний бриф", flush=True)
    brief = master / "long_brief.json"
    key = brief_key(opt, works)
    if early or redo == "translate" or state.get("brief") != key or not brief.exists():
        build_shared_brief(master, works)
        state["brief"] = key
        save()
    else:
        print("   • вже готово, пропускаю", flush=True)
    write_edges(master, works)

    print("\nФаза 3/3: готую частини", flush=True)

    def make(po, n, tag):
        po.shared_brief, po.edge_context, po.loudness_target = str(brief), str(edge_path(master, n)), target
        cli.run_pipeline(po, redo=None if early else redo, stop_after=stop_after, review=review,
                         review_with=review_with, prefix=tag, summary=False)

    _stop_if_failed(_each_part(opt, plan, master, make, record))
    if stop_after:
        print(f"\nЗупинено після етапу «{stop_after}» для всіх частин.")
        return

    missing = [p for p, w in zip(plan["parts"], works) if not (w / "dub.flac").exists()]
    if missing:  # deleted by hand while the state says done: mix and mux those parts again

        def remake(po, n, tag):
            po.shared_brief, po.edge_context, po.loudness_target = str(brief), str(edge_path(master, n)), target
            cli.run_pipeline(po, redo="mix", prefix=tag, summary=False)

        print(f"\nНемає готового звуку для {len(missing)} з {len(works)} частин — зводжу їх ще раз", flush=True)
        _stop_if_failed(_each_part(opt, {"parts": missing}, master, remake, record, total=len(works)))

    key = assembly_key(works, opt.keep_original, opt.output)
    out = Path(opt.output)
    if state.get("assembled") != key or not out.exists():
        print(f"\nСклеюю {parts_word(len(works))} → {out.name}", flush=True)
        assemble(Path(opt.input), [(w, p["start"]) for w, p in zip(works, plan["parts"])], out, master,
                 src_lang=opt.text_lang, keep_original=opt.keep_original)
        state["assembled"] = key
        save()
    if not opt.keep_parts:
        shutil.rmtree(previews_dir(opt), ignore_errors=True)
    print(f"\nГотово за {(time.time() - t_all) / 60:.1f} хв:\n"
          f"   Відео:          {link(out)}\n"
          f"   Субтитри:       {link(out.with_suffix('.srt'))}\n"
          f"   Робоча папка:   {link(master)}")
    cli._print_texts(opt, text)
