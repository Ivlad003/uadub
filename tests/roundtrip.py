"""Intelligibility check: re-transcribe the Ukrainian dub (parakeet v3 knows Ukrainian)
and compare with the text we asked the TTS to say. Usage: python tests/roundtrip.py video.uk.mp4 workdir"""
import json, re, subprocess, sys, tempfile
from pathlib import Path

from parakeet_mlx import from_pretrained

from uadub.translate import speech_text

video, work = Path(sys.argv[1]), Path(sys.argv[2])
wav = Path(tempfile.mkdtemp()) / "uk.wav"
subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", str(video), "-map", "0:a:0", "-ac", "1", "-ar", "16000", str(wav)], check=True)
heard = from_pretrained("mlx-community/parakeet-tdt-0.6b-v3").transcribe(str(wav)).text
said = " ".join(speech_text(u) for u in json.loads((work / "units.json").read_text()))


def norm(s: str) -> str:
    s = s.lower().replace("’", "'").replace("ʼ", "'")
    return re.sub(r"\s+", " ", re.sub(r"[^\w' ]", " ", s)).strip()


def dist(a: str, b: str) -> int:
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


a, b = norm(said), norm(heard)
wa, wb = a.split(), b.split()
print("HEARD:", heard)
print(f"CER = {dist(a, b) / max(1, len(a)):.3f}   WER = {dist(wa, wb) / max(1, len(wa)):.3f}")
