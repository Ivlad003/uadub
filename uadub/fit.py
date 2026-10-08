"""Deciding where each synthesized line goes on the timeline and how fast it is spoken."""

from __future__ import annotations

# Ukrainian speaking-rate band for the dub (syllables per second). Below it speech sounds
# drawn-out, above it rushed. A line is never pushed past the ceiling.
MIN_RATE, MAX_RATE = 5.0, 7.5
PACE = 5.6  # target pace of the whole dub (syl/s): every line is spoken at this rate, ±PACE_TOL
PACE_TOL = 0.05
SPILL = 0.5  # seconds a line may run past its slot before the mix stretches it
BREATH_MIN, BREATH_MAX = 0.2, 0.5  # pause after a line that ran late: the speaker's own pause, clamped


def pace_cap(length: float, syl: int, max_speed: float, *, max_rate: float = MAX_RATE) -> float:
    """How much a clip may still be sped up before its pace passes `max_rate` syl/s (≤ `max_speed`)."""
    if length <= 0 or not syl:
        return max_speed
    return float(min(max_speed, max(1.0, max_rate * length / syl)))


def line_start(u: dict, prev_dub_end: float | None, prev_orig_end: float | None, *,
               breath_min: float = BREATH_MIN, breath_max: float = BREATH_MAX) -> float:
    """Where a line starts in the dub: on time, unless the previous line ran late — then after it,
    with the pause the speaker made at that point (clamped to `breath_min`–`breath_max`)."""
    start = u["start"]
    if prev_dub_end is not None and prev_dub_end + breath_min > start:
        gap = min(max(start - prev_orig_end, breath_min), breath_max)
        start = prev_dub_end + gap
    return start


def place_clips(
    units: list[dict],
    lengths: list[float],
    *,
    max_speed: float = 1.25,
    min_stretch: float = 1.08,
    spill: float = SPILL,
    caps: list[float] | None = None,
    breath_min: float = BREATH_MIN,
    breath_max: float = BREATH_MAX,
) -> list[dict]:
    """Return per-line {start, speed, length}.

    Each clip starts at its original time. When the previous clip ran late, this one starts after
    it with the pause the speaker made at that point (clamped to `breath_min`–`breath_max`), so the
    dub keeps the original rhythm instead of a metronome. A clip may run up to `spill` seconds past
    its slot without being touched; only a longer overrun is time-stretched, by at most `max_speed`,
    to fit `avail + spill`. Small stretches (< `min_stretch`) are skipped: they would make the pace
    jump from line to line, which is more audible than a late start. `caps` optionally lowers
    `max_speed` per line (see `pace_cap`).
    """
    plan: list[dict] = []
    prev_dub_end = prev_orig_end = None
    for i, (u, length) in enumerate(zip(units, lengths)):
        start = line_start(u, prev_dub_end, prev_orig_end, breath_min=breath_min, breath_max=breath_max)
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
            prev_dub_end, prev_orig_end = start + out_len, u["end"]
    return plan


def target_duration(natural: float, avail: float, max_speed: float) -> float | None:
    """For engines with duration control: None = speak naturally, else forced seconds."""
    if natural <= avail:
        return None
    return max(avail, natural / max_speed)


ST_MIN_SPEED = 0.89  # slowing a line more than this sounds drawn-out (engine value 0.85)


def st_speed(
    length: float,
    syl: int,
    slot_s: float,
    *,
    pace: float = PACE,
    tol: float = PACE_TOL,
    min_speed: float = ST_MIN_SPEED,
    max_speed: float = 1.35,
    max_rate: float = MAX_RATE,
    min_change: float = 1.02,
    spoken_s: float | None = None,
) -> float:
    """Re-synthesis ratio for an engine with its own `speed` control (StyleTTS2): old / new length.

    `length` is the natural (speed 1.0) duration of the line, `syl` its syllable count. Every line
    is brought to the same pace (`pace` ± `tol`), slow ones sped up and fast ones slowed down, so the
    dub does not lurch from line to line. Fitting the slot comes second: a line is sped up further
    when it must, but never past the pace ceiling or `max_speed`: a rushed line is less intelligible
    than one that spills into the next pause. Slowing down never makes a line miss its slot.
    `spoken_s`, the time the speaker took for this line: a translation that would fill less than
    80 % of it is spoken at the slow edge of the band (pace − 10 %) rather than hurried to the pace,
    so the hole after it is smaller.
    """
    if length <= 0:
        return 1.0
    if not syl:
        ratio = min(max_speed, length / slot_s) if length > slot_s > 0 else 1.0
        return ratio if ratio >= min_change else 1.0
    rate = syl / length
    target = pace
    if spoken_s and syl / pace < 0.8 * spoken_s:
        target = max(pace * 0.9, syl / (0.8 * spoken_s))
    ratio = target / rate
    if abs(ratio - 1.0) <= tol:
        ratio = 1.0
    if slot_s > 0 and length / ratio > slot_s:
        ratio = length / slot_s
    ceiling = max(min_speed, min(max_speed, max_rate / rate))
    ratio = min(max(ratio, min_speed), ceiling)
    return ratio if abs(ratio - 1.0) >= min_change - 1.0 else 1.0


ST_SPEED_GAIN = 1.25  # StyleTTS2 shortens a line by only ~80 % of (speed − 1); 1.4 and above garbles
ST_SLOW_GAIN = 1.4  # and lengthens it by ~70 % of (1 − speed)
ST_ENGINE_MAX, ST_ENGINE_MIN = 1.35, 0.85


def st_engine_speed(ratio: float, *, gain: float = ST_SPEED_GAIN, cap: float = ST_ENGINE_MAX) -> float:
    """The `speed` to pass to StyleTTS2 so that the line really comes out `ratio` times shorter."""
    if ratio == 1.0:
        return 1.0
    if ratio > 1.0:
        return float(min(cap, 1.0 + (ratio - 1.0) * gain))
    return float(max(ST_ENGINE_MIN, 1.0 - (1.0 - ratio) * ST_SLOW_GAIN))


def emotion_strength(k: float, syl: int) -> float:
    """`--emotion K` for one line: full on lines of 14+ syllables, fading to 30 % on very short ones.

    StyleTTS2 draws one intonation arc per sentence; with the original line's prosody that arc
    gets steep, and on a short sentence it sounds like a slammed door (up to −6 semitones at the end
    on the test clip). Longer lines carry the expressiveness without that effect.
    """
    return k * min(1.0, max(0.3, (syl - 4) / 10.0))
