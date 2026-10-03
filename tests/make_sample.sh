#!/bin/zsh
# Builds a synthetic English test video (macOS `say` voice + a synthetic music bed).
set -e
OUT=${1:-/tmp/uadub-sample}
mkdir -p "$OUT"
TEXT="Hi everyone, and welcome back to the channel. Today I want to show you how I built a small tool that translates videos right on my laptop. It runs completely offline, so none of your files ever leave your computer. First, we separate the voice from the background music. Then a speech recognition model writes down every word, with timestamps. Next, a local language model translates the text into Ukrainian, keeping each line short enough to fit the original timing. Finally, a text to speech model reads the translation, and we mix it back with the music. On my MacBook with 32 gigabytes of memory, a ten minute video takes about fifteen minutes. I was honestly surprised by how well it works. The whole project is about 1,500 lines of Python, and it costs exactly 0 dollars to run. Thanks for watching, and see you next time!"
VOICE=$(say -v '?' | grep -E '^(Samantha|Alex|Daniel|Karen) ' | head -1 | awk '{print $1}')
say ${VOICE:+-v $VOICE} -r 180 -o "$OUT/speech.aiff" "$TEXT"
DUR=$(ffprobe -v error -show_entries format=duration -of csv=p=0 "$OUT/speech.aiff")
TOTAL=$(python3 -c "print(round($DUR + 2.5, 2))")
ffmpeg -y -loglevel error \
  -f lavfi -i "testsrc2=size=1280x720:rate=25:duration=$TOTAL" \
  -f lavfi -i "aevalsrc='0.05*sin(2*PI*220*t)*(0.6+0.4*sin(2*PI*0.5*t))+0.04*sin(2*PI*277.2*t)+0.035*sin(2*PI*329.6*t)+0.02*sin(2*PI*110*t)':s=44100:d=$TOTAL" \
  -i "$OUT/speech.aiff" \
  -filter_complex "[2:a]adelay=1000|1000,aresample=44100,pan=stereo|c0=c0|c1=c0[s];[1:a]pan=stereo|c0=c0|c1=c0[m];[m][s]amix=inputs=2:normalize=0:duration=first[a]" \
  -map 0:v -map "[a]" -c:v libx264 -preset veryfast -crf 26 -c:a aac -b:a 160k -shortest "$OUT/sample.mp4"
echo "$OUT/sample.mp4 ($TOTAL s, voice: ${VOICE:-default})"
