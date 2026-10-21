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


def clipboard_command(platform: str, which=shutil.which) -> list[str] | None:
    """The command that reads stdin into the clipboard on this platform, or None."""
    if platform == "darwin" and which("pbcopy"):
        return ["pbcopy"]
    if platform == "win32" and which("clip"):
        return ["clip"]
    if which("wl-copy"):
        return ["wl-copy"]
    if which("xclip"):
        return ["xclip", "-selection", "clipboard"]
    return None


def copy_to_clipboard(text: str) -> bool:
    cmd = clipboard_command(sys.platform)
    if not cmd:
        return False
    try:
        subprocess.run(cmd, input=text.encode(), check=True)
        return True
    except Exception:
        return False


def editor_command(path: str, *, platform: str, env: dict, which=shutil.which) -> list[str] | None:
    """$VISUAL/$EDITOR (blocking, e.g. vim), else the OS text editor: open -t, notepad."""
    editor = env.get("VISUAL") or env.get("EDITOR")
    if editor:
        return shlex.split(editor, posix=platform != "win32") + [path]
    if platform == "darwin" and which("open"):
        return ["open", "-t", path]
    if platform == "win32" and which("notepad"):
        return ["notepad", path]
    return None


def open_in_editor(path: str | Path) -> None:
    p = str(Path(path).expanduser())
    cmd = editor_command(p, platform=sys.platform, env=dict(os.environ))
    if cmd:
        subprocess.run(cmd)
    else:
        print(f"   Відкрийте файл вручну: {p}")
