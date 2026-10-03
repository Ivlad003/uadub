"""Speaking pace of a dub, per line: syllables / seconds. Below ~4.5 syl/s sounds drawn-out.
Usage: python tests/pace.py video.uadub"""
import json, sys
from pathlib import Path

from uadub.textnorm import syllables
from uadub.translate import speech_text
from uadub.stress import StressFixer

units = json.loads((Path(sys.argv[1]) / "units.json").read_text())
plain = StressFixer("off")
rates = []
for u in units:
    if not u.get("tts_len"):
        continue
    rate = syllables(plain.apply(speech_text(u))) / (u["tts_len"] / u.get("speed", 1.0))
    rates.append((rate, u))
rates.sort(key=lambda x: x[0])
slow = [r for r in rates if r[0] < 4.5]
print(f"lines: {len(rates)}, median {rates[len(rates) // 2][0]:.1f} syl/s, drawn-out (<4.5): {len(slow)}")
for r, u in slow[:8]:
    print(f"  {r:.1f} syl/s  #{u['id']}  {(u.get('tts_text') or speech_text(u))[:90]}")
