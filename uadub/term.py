"""Terminal helpers: clickable file links (OSC 8), clipboard, opening files in an editor."""

from __future__ import annotations

import os
import shlex
import shutil
import subprocess
import sys
from pathlib import Path
from urllib.parse import quote


def _links_supported() -> bool:
    if os.environ.get("UADUB_NO_LINKS") or os.environ.get("TERM") == "dumb":
        return False
    return sys.stdout.isatty()


def link(path: str | Path, text: str | None = None) -> str:
    """A clickable file link (iTerm2, Terminal.app, VS Code, Warp, Ghostty…).

    The visible text is the full path itself, so it can also be selected and copied as is;
    terminals without OSC 8 support simply show the plain path.
    """
    p = Path(path).expanduser().resolve()
    label = text or str(p)
    if not _links_supported():
        return label
    url = "file://" + quote(str(p))
    return f"\033]8;;{url}\033\\{label}\033]8;;\033\\"


def shell_cd(path: str | Path) -> str:
    return f"cd {shlex.quote(str(Path(path).expanduser().resolve()))}"


def copy_to_clipboard(text: str) -> bool:
    tool = shutil.which("pbcopy") or shutil.which("wl-copy") or shutil.which("xclip")
    if not tool:
        return False
    cmd = [tool] + (["-selection", "clipboard"] if tool.endswith("xclip") else [])
    try:
        subprocess.run(cmd, input=text.encode(), check=True)
        return True
    except Exception:
        return False


def open_in_editor(path: str | Path) -> None:
    """$VISUAL/$EDITOR if set (blocking, e.g. vim), otherwise the macOS default text editor."""
    editor = os.environ.get("VISUAL") or os.environ.get("EDITOR")
    p = str(Path(path).expanduser())
    if editor:
        subprocess.run(shlex.split(editor) + [p])
    elif shutil.which("open"):
        subprocess.run(["open", "-t", p])
    else:
        print(f"   Відкрийте файл вручну: {p}")
