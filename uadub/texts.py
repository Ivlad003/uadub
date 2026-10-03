"""Plain-text copies of the original transcript and the translation (--text)."""

from __future__ import annotations

import json
from pathlib import Path

PARAGRAPH_GAP = 2.0  # seconds of silence that start a new paragraph


def _stamp(t: float) -> str:
    t = int(t)
    return f"{t // 3600}:{t % 3600 // 60:02d}:{t % 60:02d}" if t >= 3600 else f"{t // 60:02d}:{t % 60:02d}"


def _paragraphs(units: list[dict]) -> list[list[dict]]:
    paras: list[list[dict]] = []
    for u in units:
        if paras and u["start"] - paras[-1][-1]["end"] < PARAGRAPH_GAP:
            paras[-1].append(u)
        else:
            paras.append([u])
    return paras


def render(units: list[dict], key: str) -> str:
    """Readable text: paragraphs split at pauses, each with its start time."""
    out = []
    for p in _paragraphs(units):
        text = " ".join(str(u.get(key) or "").strip() for u in p if str(u.get(key) or "").strip())
        if text:
            out.append(f"[{_stamp(p[0]['start'])}] {text}")
    return "\n\n".join(out) + "\n"


def render_bilingual(units: list[dict], src_label: str) -> str:
    out = []
    for u in units:
        src, uk = str(u.get("text") or "").strip(), str(u.get("uk") or "").strip()
        if src or uk:
            out.append(f"[{_stamp(u['start'])}]\n{src_label}: {src}\nUK: {uk}")
    return "\n\n".join(out) + "\n"


def write_texts(work: Path, input_path: str, output_path: str, src_lang: str) -> list[tuple[str, Path]]:
    """Write <name>.<src>.txt, <name>.uk.txt and <name>.<src>-uk.txt next to the output video.

    Returns (label, path) pairs for the final summary; nothing if there is no translation yet.
    """
    f = Path(work) / "units.json"
    if not f.exists():
        return []
    units = json.loads(f.read_text())
    if not units or not any(u.get("uk") for u in units):
        return []
    base = Path(output_path).parent / Path(input_path).stem
    files: list[tuple[str, Path]] = []
    if src_lang != "uk":
        p = base.with_name(f"{base.name}.{src_lang}.txt")
        p.write_text(render(units, "text"), encoding="utf-8")
        files.append(("Оригінал (текст)", p))
    p = base.with_name(f"{base.name}.uk.txt")
    p.write_text(render(units, "uk"), encoding="utf-8")
    files.append(("Переклад (текст)", p))
    stressed = _stressed_text(Path(work), units)
    if stressed:
        p = base.with_name(f"{base.name}.uk.stress.txt")
        p.write_text(stressed, encoding="utf-8")
        files.append(("З наголосами", p))
    if src_lang != "uk":
        p = base.with_name(f"{base.name}.{src_lang}-uk.txt")
        p.write_text(render_bilingual(units, src_lang.upper()), encoding="utf-8")
        files.append(("Обидва поруч", p))
    return files


def _stressed_text(work: Path, units: list[dict]) -> str | None:
    """The spoken text with every stress marked (на́голос): for StyleTTS2 exactly what was voiced,
    for other voices the dictionary stress plus your stress.txt."""
    try:
        from .config import Options
        from .review import _stress_preview

        engine = Options.load(work / "options.json").engine
        preview = _stress_preview(work, engine, any_engine=True, acute=True)
    except Exception:
        return None
    if preview is None:
        return None
    head = ("# Текст для озвучення з наголосами (так їх поставив StyleTTS2)\n\n" if engine == "st" else
            "# Текст для озвучення з наголосами за словником (голос OmniVoice/ukrainian-tts може читати інакше)\n\n")
    marked = [dict(u, uk=preview(u)) for u in units if str(u.get("uk") or "").strip()]
    return head + render(marked, "uk")
