"""Synthetic two-speaker Korean 'drama' clip (macOS voices + music bed) and a matching .ko.srt.
Usage: python tests/make_sample_ko.py /tmp/uadub-ko"""
import subprocess, sys
from pathlib import Path

OUT = Path(sys.argv[1] if len(sys.argv) > 1 else "/tmp/uadub-ko")
OUT.mkdir(parents=True, exist_ok=True)
F, M = "Yuna", "Reed (Korean (South Korea))"
LINES = [  # original dialogue written for this test
    (F, "오빠, 벌써 나가요? 아직 일곱 시도 안 됐잖아요."),
    (M, "응, 오늘 회사에 중요한 회의가 있거든. 일찍 가야 돼."),
    (F, "아침도 안 먹었잖아요. 이 김밥이라도 가져가요."),
    (M, "고마워. 너는 항상 나를 챙겨 주는구나."),
    (F, "당연하죠. 그런데 오늘 저녁에 시간 있어요?"),
    (M, "아마 여덟 시쯤 끝날 것 같아. 같이 저녁 먹을까?"),
    (F, "좋아요! 우리가 처음 만났던 그 식당에서 만나요."),
    (M, "알았어. 이번에는 절대 늦지 않을게. 약속해."),
]


def dur(p):
    return float(subprocess.check_output(["ffprobe", "-v", "error", "-show_entries", "format=duration",
                                          "-of", "csv=p=0", str(p)]).decode())


t, parts, srt = 1.0, [], []
for i, (voice, text) in enumerate(LINES):
    f = OUT / f"l{i}.aiff"
    subprocess.run(["say", "-v", voice, "-r", "190", "-o", str(f), text], check=True)
    d = dur(f)
    parts.append((f, t))
    ts = lambda x: f"{int(x // 3600):02d}:{int(x % 3600 // 60):02d}:{int(x % 60):02d},{int(x * 1000 % 1000):03d}"
    srt.append(f"{i + 1}\n{ts(t)} --> {ts(t + d)}\n{text}\n")
    t += d + 0.6
total = t + 1.0
(OUT / "drama.ko.srt").write_text("\n".join(srt), encoding="utf-8")
inputs, filt = [], []
for k, (f, start) in enumerate(parts):
    inputs += ["-i", str(f)]
    ms = int(start * 1000)
    filt.append(f"[{k + 2}:a]aresample=44100,adelay={ms}|{ms},pan=stereo|c0=c0|c1=c0[v{k}]")
music = ("aevalsrc='0.04*sin(2*PI*196*t)*(0.6+0.4*sin(2*PI*0.25*t))+0.03*sin(2*PI*246.9*t)"
         f"+0.03*sin(2*PI*293.7*t)':s=44100:d={total}")
mix = "".join(f"[v{k}]" for k in range(len(parts)))
filt.append(f"[1:a]pan=stereo|c0=c0|c1=c0[m];[m]{mix}amix=inputs={len(parts) + 1}:normalize=0:duration=first[a]")
subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "lavfi", "-i",
                f"testsrc2=size=1280x720:rate=25:duration={total}", "-f", "lavfi", "-i", music, *inputs,
                "-filter_complex", ";".join(filt), "-map", "0:v", "-map", "[a]", "-c:v", "libx264",
                "-preset", "veryfast", "-crf", "26", "-c:a", "aac", "-b:a", "160k", str(OUT / "drama.mp4")],
               check=True)
for f, _ in parts:
    f.unlink()
print(OUT / "drama.mp4", f"{total:.1f}s")
