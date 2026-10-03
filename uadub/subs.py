"""Reading existing subtitles (.srt / .vtt) — e.g. dramas that already ship with subtitles."""

from __future__ import annotations

import re
from pathlib import Path

_TIME = re.compile(
    r"(?:(\d+):)?(\d{1,2}):(\d{2})[,.](\d{1,3})\s*-->\s*(?:(\d+):)?(\d{1,2}):(\d{2})[,.](\d{1,3})")
_TAGS = re.compile(r"<[^>]+>|\{\\[^}]*\}")
_NOISE = re.compile(r"[♪♫]+")


def _sec(h, m, s, ms) -> float:
    return int(h or 0) * 3600 + int(m) * 60 + int(s) + int(ms.ljust(3, "0")) / 1000


def read_subs(path: str | Path) -> list[dict]:
    raw = Path(path).read_text(encoding="utf-8-sig", errors="replace").replace("\r\n", "\n")
    out = []
    for block in re.split(r"\n\s*\n", raw):
        lines = [l.strip() for l in block.strip().split("\n") if l.strip()]
        for i, line in enumerate(lines):
            m = _TIME.search(line)
            if not m:
                continue
            start, end = _sec(*m.group(1, 2, 3, 4)), _sec(*m.group(5, 6, 7, 8))
            parts = []
            for t in lines[i + 1 :]:
                t = _NOISE.sub("", _TAGS.sub("", t)).strip()
                t = re.sub(r"^[-–—]\s*", "", t)  # dialogue dashes
                if t:
                    parts.append(t)
            text = " ".join(parts).strip()
            if text and end > start:
                out.append({"start": round(start, 3), "end": round(end, 3), "text": text, "words": []})
            break
    out.sort(key=lambda s: s["start"])
    return out
