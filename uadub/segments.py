"""Turning ASR sentences into dubbing units with time slots."""

from __future__ import annotations


def merge_tokens_to_words(tokens: list[dict]) -> list[dict]:
    """Join BPE tokens (leading space = new word) into words with timings."""
    words: list[dict] = []
    for t in tokens:
        text = t["text"]
        if not words or text.startswith(" "):
            words.append({"w": text.strip(), "start": t["start"], "end": t["end"]})
        else:
            words[-1]["w"] += text
            words[-1]["end"] = t["end"]
    return [w for w in words if w["w"]]


def build_units(
    sentences: list[dict],
    *,
    max_dur: float = 12.0,
    merge_gap: float = 0.35,
    min_dur: float = 1.2,
    min_words: int = 3,
) -> list[dict]:
    """Merge very short sentences into a neighbour so TTS gets whole phrases.

    Two sentences are merged only if they are separated by a short pause, one of
    them is short, and the result stays under `max_dur`.
    """
    units: list[dict] = []
    for s in sentences:
        text = s["text"].strip()
        if not text:
            continue
        cur = {"start": float(s["start"]), "end": float(s["end"]), "text": text, "gender": s.get("gender")}
        if units:
            prev = units[-1]
            gap = cur["start"] - prev["end"]
            prev_short = prev["end"] - prev["start"] < min_dur or len(prev["text"].split()) < min_words
            cur_short = cur["end"] - cur["start"] < min_dur or len(text.split()) < min_words
            if gap <= merge_gap and (prev_short or cur_short) and cur["end"] - prev["start"] <= max_dur:
                prev["end"] = cur["end"]
                prev["text"] = f'{prev["text"]} {text}'
                if prev.get("gender") != cur.get("gender"):
                    prev["gender"] = None
                continue
        units.append(cur)
    for i, u in enumerate(units):
        u["id"] = i + 1
    return units


def assign_slots(units: list[dict], total_dur: float, *, tail: float = 0.8, guard: float = 0.08) -> None:
    """Set `slot_end`: how long the dubbed line may last without stepping on the next one.

    A line may run up to `tail` seconds past the original end, but never into the
    next line (minus a small `guard`) or past the end of the video.
    """
    for i, u in enumerate(units):
        nxt = units[i + 1]["start"] if i + 1 < len(units) else total_dur + guard
        limit = min(nxt - guard, u["end"] + tail, total_dur)
        u["slot_end"] = round(max(u["end"], limit) if limit >= u["start"] else u["end"], 3)


def slot(u: dict) -> float:
    return max(0.05, u["slot_end"] - u["start"])


_SENT_END = ("." , "?", "!", "…", "。", "？", "！", "~")


def words_to_sentences(words: list[dict], *, max_dur: float = 10.0, gap: float = 0.7,
                       max_words: int = 30) -> list[dict]:
    """Group timed words (Whisper) into sentence-like segments."""
    out: list[dict] = []
    cur: list[dict] = []

    def flush():
        if cur:
            out.append({"start": round(cur[0]["start"], 3), "end": round(cur[-1]["end"], 3),
                        "text": " ".join(w["w"] for w in cur).strip(), "words": list(cur)})
            cur.clear()

    for w in words:
        if not w["w"]:
            continue
        if cur and (w["start"] - cur[-1]["end"] >= gap or w["end"] - cur[0]["start"] > max_dur
                    or len(cur) >= max_words):
            flush()
        cur.append(w)
        if w["w"].endswith(_SENT_END):
            flush()
    flush()
    return out


def snap_to_speech(sentences: list[dict], y, sr: int, *, pad: float = 0.4, margin: float = 0.05) -> int:
    """Tighten approximate timings (Whisper words, subtitle cues) to where speech really is.

    Looks for voiced frames (energy) in [start − pad, end + pad], never crossing the neighbours,
    and moves start/end onto them. Accurate boundaries matter twice: the dub starts on time and
    voice-clone references contain exactly the words of their transcript. Returns #adjusted.
    """
    import numpy as np

    hop = max(1, int(sr * 0.01))
    n = len(y) // hop
    if n == 0:
        return 0
    rms = np.sqrt((y[: n * hop].reshape(n, hop) ** 2).mean(axis=1) + 1e-12)
    db = 20 * np.log10(rms)
    floor, peak = np.percentile(db, 10), np.percentile(db, 99)
    thresh = max(floor + 12, peak - 35)
    voiced = db > thresh
    changed = 0
    prev_end = 0.0
    for i, s in enumerate(sentences):
        nxt = sentences[i + 1]["start"] if i + 1 < len(sentences) else len(y) / sr
        lo = max(prev_end, s["start"] - pad)
        hi = min(max(nxt, s["end"]), s["end"] + pad)
        a, b = int(lo * 100), min(n, int(hi * 100))
        idx = np.where(voiced[a:b])[0]
        if idx.size:
            new_start = max(lo, (a + idx[0]) / 100 - margin)
            new_end = min(hi, (a + idx[-1] + 1) / 100 + margin)
            if new_end - new_start > 0.2 and (abs(new_start - s["start"]) > 0.03 or abs(new_end - s["end"]) > 0.03):
                s["start"], s["end"] = round(new_start, 3), round(new_end, 3)
                changed += 1
        prev_end = s["end"]
    return changed
