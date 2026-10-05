"""Speaking pace of a dub, per line: syllables / seconds. Below ~5 syl/s sounds drawn-out,
above ~7.5 rushed; a wide spread between neighbouring lines sounds unnatural.
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
n = len(rates)
slow = [r for r in rates if r[0] < 5.0]
fast = [r for r in rates if r[0] > 7.5]
stretched = [u for _, u in rates if u.get("speed", 1.0) > 1.0]
p10, p90 = rates[int(n * 0.1)][0], rates[min(n - 1, int(n * 0.9))][0]
print(f"lines: {n}, median {rates[n // 2][0]:.1f} syl/s, p10–p90 {p10:.1f}–{p90:.1f}, "
      f"drawn-out (<5): {len(slow)}, rushed (>7.5): {len(fast)}, stretched in mix: {len(stretched)}")
voiced = [u for u in units if u.get("tts_len") and "dub_end" in u]
gaps = sorted(b["dub_start"] - a["dub_end"] for a, b in zip(voiced, voiced[1:]))
late = [u["dub_start"] - u["start"] for u in voiced]
if gaps:
    print(f"pauses between lines: median {gaps[len(gaps) // 2]:.2f} s, under 0.15 s: {sum(g < 0.15 for g in gaps)}/{len(gaps)}; "
          f"dub starts late by up to {max(late):.2f} s")
for r, u in slow[:8] + fast[-8:]:
    print(f"  {r:.1f} syl/s  #{u['id']}  {(u.get('tts_text') or speech_text(u))[:90]}")
