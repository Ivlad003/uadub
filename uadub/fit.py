"""Deciding where each synthesized line goes on the timeline and how much to speed it up."""

from __future__ import annotations


def place_clips(
    units: list[dict],
    lengths: list[float],
    *,
    max_speed: float = 1.25,
    min_gap: float = 0.05,
    min_stretch: float = 1.03,
) -> list[dict]:
    """Return per-line {start, speed, length}.

    Each clip starts at its original time (or right after the previous clip if that
    one overran). If it does not fit its slot, it is sped up by at most `max_speed`;
    anything left over spills into the following pause and the timeline catches up
    at the next gap. Tiny speed-ups (< `min_stretch`) are skipped to avoid artefacts.
    """
    plan: list[dict] = []
    cursor = 0.0
    for u, length in zip(units, lengths):
        start = max(u["start"], cursor)
        avail = u["slot_end"] - start
        speed = 1.0
        if length > 0:
            if avail <= 0:
                speed = max_speed
            elif length > avail:
                speed = min(max_speed, length / avail)
            if speed < min_stretch:
                speed = 1.0
        out_len = length / speed
        plan.append({"start": round(start, 4), "speed": round(speed, 4), "length": round(out_len, 4)})
        if length > 0:
            cursor = start + out_len + min_gap
    return plan


def target_duration(natural: float, avail: float, max_speed: float) -> float | None:
    """For engines with duration control: None = speak naturally, else forced seconds."""
    if natural <= avail:
        return None
    return max(avail, natural / max_speed)
