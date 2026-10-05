"""Deciding where each synthesized line goes on the timeline and how much to speed it up."""

from __future__ import annotations

# Ukrainian speaking-rate band for the dub (syllables per second). Below it speech sounds
# drawn-out, above it rushed. Lines are brought up to the floor and never pushed past the ceiling.
MIN_RATE, MAX_RATE = 5.0, 7.5
SPILL = 0.5  # seconds a line may run past its slot before the mix stretches it


def pace_cap(length: float, syl: int, max_speed: float, *, max_rate: float = MAX_RATE) -> float:
    """How much a clip may still be sped up before its pace passes `max_rate` syl/s (≤ `max_speed`)."""
    if length <= 0 or not syl:
        return max_speed
    return float(min(max_speed, max(1.0, max_rate * length / syl)))


def place_clips(
    units: list[dict],
    lengths: list[float],
    *,
    max_speed: float = 1.25,
    min_gap: float = 0.05,
    min_stretch: float = 1.08,
    spill: float = SPILL,
    caps: list[float] | None = None,
) -> list[dict]:
    """Return per-line {start, speed, length}.

    Each clip starts at its original time (or right after the previous clip if that
    one overran). A clip may run up to `spill` seconds past its slot without being
    touched: the next line simply starts a little late and the timeline catches up at
    the next gap. Only a longer overrun is time-stretched, by at most `max_speed`, to
    fit `avail + spill`. Small stretches (< `min_stretch`) are skipped: they would make
    the pace jump from line to line, which is more audible than a late start.
    `caps` optionally lowers `max_speed` per line (see `pace_cap`).
    """
    plan: list[dict] = []
    cursor = 0.0
    for i, (u, length) in enumerate(zip(units, lengths)):
        start = max(u["start"], cursor)
        avail = u["slot_end"] - start + spill
        cap = min(max_speed, caps[i]) if caps else max_speed
        speed = 1.0
        if length > 0:
            if avail <= 0:
                speed = cap
            elif length > avail:
                speed = min(cap, length / avail)
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


def st_speed(
    length: float,
    syl: int,
    slot_s: float,
    *,
    min_rate: float = MIN_RATE,
    max_rate: float = MAX_RATE,
    max_speed: float = 1.35,
    min_change: float = 1.02,
) -> float:
    """Re-synthesis speed for an engine with its own `speed` control (StyleTTS2).

    `length` is the natural (speed 1.0) duration of the line, `syl` its syllable count.
    The line is sped up to reach the pace floor and to fit its slot, but never past the
    pace ceiling or `max_speed`: a rushed line is less intelligible than one that spills
    into the next pause. A line already faster than the ceiling is left alone.
    """
    if length <= 0:
        return 1.0
    speed = 1.0
    if length > slot_s > 0:
        speed = length / slot_s
    cap = max_speed
    if syl:
        rate = syl / length
        if rate >= max_rate:
            return 1.0
        speed = max(speed, min_rate / rate)
        cap = min(cap, max_rate / rate)
    speed = min(speed, cap)
    return speed if speed >= min_change else 1.0
