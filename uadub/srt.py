"""SRT subtitle writer with simple line wrapping and long-cue splitting."""

from __future__ import annotations

from pathlib import Path


def _ts(t: float) -> str:
    ms = max(0, int(round(t * 1000)))
    h, ms = divmod(ms, 3_600_000)
    m, ms = divmod(ms, 60_000)
    s, ms = divmod(ms, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def wrap(text: str, max_chars: int = 42) -> list[str]:
    lines: list[str] = []
    cur = ""
    for word in text.split():
        if cur and len(cur) + 1 + len(word) > max_chars:
            lines.append(cur)
            cur = word
        else:
            cur = f"{cur} {word}" if cur else word
    if cur:
        lines.append(cur)
    return lines


def make_cues(units: list[dict], key: str, *, max_chars: int = 42, max_lines: int = 2) -> list[tuple[float, float, str]]:
    cues: list[tuple[float, float, str]] = []
    for idx, u in enumerate(units):
        text = (u.get(key) or "").strip()
        if not text:
            continue
        lines = wrap(text, max_chars)
        groups = [lines[i : i + max_lines] for i in range(0, len(lines), max_lines)]
        start, end = u["start"], max(u["end"], u.get("dub_end", u["end"]))
        if idx + 1 < len(units):
            nxt = units[idx + 1]["start"]
            end = min(end, max(u["end"], nxt - 0.02))
        total = sum(len(" ".join(g)) for g in groups) or 1
        t = start
        for g in groups:
            d = (end - start) * len(" ".join(g)) / total
            cues.append((t, t + d, "\n".join(g)))
            t += d
    return cues


def write_srt(cues: list[tuple[float, float, str]], path: Path) -> None:
    out = []
    for i, (a, b, text) in enumerate(cues, 1):
        out.append(f"{i}\n{_ts(a)} --> {_ts(b)}\n{text}\n")
    Path(path).write_text("\n".join(out), encoding="utf-8")
