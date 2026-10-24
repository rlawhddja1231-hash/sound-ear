---
name: sound-ear
description: "Lets Claude 'hear' audio files. Turns songs, vocals, sound effects or video soundtracks into numbers and pictures (waveform, spectrogram, pitch curve) with a bundled script, then finds problems with timestamps: clicks, clipping, abrupt cuts, pitch glitches, off-pitch notes, wobbly notes, harsh sibilance, loud breaths, cut-off endings, loudness too high or low. Also estimates BPM, key and vocal range. Made for checking songs from vocal synths (Synthesizer V, VOCALOID, UTAU, CeVIO) before release. Use whenever the user shares an audio or video file (wav, mp3, flac, m4a, ogg, opus, webm, mp4) or asks to listen, check a mix, find glitches, measure BPM/key/loudness, or compare two versions. 소리 듣기: 노래 들어 봐, 검수, 음정, 음 튐, 딸깍, 잡음, 깨짐, 끊김, BPM, 키, 음역, 믹싱, 두 판 비교."
license: MIT
---

# sound-ear — listen to audio and find problems

Claude has no ears. This skill turns an audio file into **numbers and pictures** with `scripts/listen.py`, so Claude can inspect it, point at exact timestamps, and explain how to fix each problem. Reply to the user in their own language (the report and chart labels are in Korean; translate them when the user writes in another language).

What it cannot do: listen live through a microphone, transcribe lyrics, or judge taste. Pitch checks need a **vocal-only** file; with a full mix the accompaniment confuses pitch tracking.

## 1. Setup (once per session)

```bash
pip install -q librosa soundfile pyloudnorm matplotlib koreanize-matplotlib scipy 2>/dev/null \
  || pip install -q --break-system-packages librosa soundfile pyloudnorm matplotlib koreanize-matplotlib scipy
which ffmpeg ffprobe   # required; install ffmpeg if missing
L="${CLAUDE_SKILL_DIR:-}/scripts/listen.py"
[ -f "$L" ] || L=$(find / -name listen.py -path '*sound-ear*' 2>/dev/null | head -1)
echo "$L"
```

Self-test (optional, ~20 s): `python3 "$(dirname "$L")/../tests/selftest.py"` → should print `PASS` (7 planted problems found, none extra).

## 2. Workflow

1. **Get the file.** Work on a copy; never edit the user's original.
2. **Pick the mode.** For vocal-synth songs, ask for both files if available:
   - vocal-only export → `--mode vocal` (pitch, glitches, sibilance, breaths)
   - full mix → `--mode mix` (clipping, loudness, clicks, cut ends)
   - `--mode auto` (default) picks vocal when the filename contains vocal/vox/voice/보컬.
3. **Run the check.**
   ```bash
   python3 "$L" check vocal.wav --mode vocal --out vocal_check
   python3 "$L" check mix.wav   --mode mix   --out mix_check
   ```
   About 30 s for a 3-minute vocal, 15 s for a 3-minute mix. Output: `보고서.md` (report), `result.json`, `00_전체.png` (overview), 30-second pages, and `확대/` close-ups for each problem.
4. **Look at the pictures yourself** (open the PNGs) — don't trust the report alone. Red dashed line = must fix, orange = listen and decide, gray = info. In the pitch row, dark green dots = smoothed pitch, light green = fine-grained pitch; a single light-green dot far from the rest is a pitch glitch. Drop findings that are clearly intentional (grace notes, deliberate breaths, vibrato, stutter/gate effects, hi-hats in a mix).
5. **Report** as a table: time | what is wrong | urgency | how to fix. Then share a few close-ups and short audio snippets so the user can hear each spot:
   ```bash
   python3 "$L" cut vocal.wav 1:23.4 --len 2 --mode vocal --out snippets
   ```
6. **After the user fixes it,** re-run `check` and compare versions:
   ```bash
   python3 "$L" compare old.wav new.wav --out compare
   ```
   Say "before: N issues, now: M issues".

## 3. What each finding means and how to fix it

Menu names differ between vocal synths and versions; treat these as general directions.

| Finding (Korean label) | Meaning | Typical fix |
|---|---|---|
| 클리핑 (clipping) | Samples hit full scale; audible crackle | Lower track/master gain; limiter ceiling −1 dB |
| 소리가 0dB까지 꽉 참 | Lossy file (mp3/opus) touching 0 dB; can't prove real clipping | Check the original wav; master is very hot |
| 트루 피크 넘침 (true peak over) | Fine now, may distort after mp3/AAC encoding | Limiter ceiling −1 dBTP |
| 딸깍·틱 소리 (click) | Very short tick | Inspect note joins; re-render that part; add short fades at edit points |
| 뚝 끊김 (abrupt cut) | Sound stops instantly instead of decaying | Extend note tail / fade; check region ends (ignore if it's a gate effect) |
| 끝이 잘림 / 시작이 잘림 | Export range shorter than the sound | Extend export range; add fade-out |
| 음이 순간 튐 (pitch glitch) | Pitch jumps for tens of ms and returns | Remove the spike in the pitch curve; adjust note transition; check overlapping notes |
| 음정 어긋남 (off-pitch) | Note centre more than 30 cents off | Reset pitch deviation/offset near 0; reduce over-strong auto expression |
| 음 흔들림 큼 (wobbly) | Long note swings widely | Reduce vibrato depth or delay its onset |
| 조에 없는 음 (out of key) | Note not in the estimated key | Ignore if intentional |
| 전체 튜닝 치우침 | Whole vocal off A=440 | Match the reference pitch of vocal and backing |
| ㅅ·ㅊ 소리 셈 (sibilance) | "s/ts" as loud as vowels | Lower consonant volume/length; de-esser |
| 숨소리 큼 (loud breath) | Loud breath/noise between phrases | Fine if intentional; otherwise lower it |
| 너무 큼 / 너무 작음 | Integrated loudness far from −14 LUFS | Streaming services normalize; very loud masters just get turned down and sound squashed |
| 위상 반대 (phase) | L/R cancel on mono phone speakers | Flip polarity of one channel |

## 4. Detection thresholds (inside the script)

- Clipping: ≥0.999 for more than 3 samples in a row (lossy files: summary only)
- Click: >3 kHz energy in a 1 ms bin over 8× the 150 ms local median, with 3–8 ms neighbours under a third of it (in mix mode downgraded to "listen and decide")
- Abrupt cut: from above −25 dB to below −60 dB within 10 ms
- Pitch glitch: fine pitch (yin, 5.8 ms hop) more than 3 semitones from the local median and back within 80 ms, ignoring fade-ins/outs
- Off-pitch: notes ≥0.25 s whose median is >30 cents off after removing the global tuning offset
- Sibilance: 5–10 kHz energy within 6 dB of vowel level (75th percentile), >50 % of the frame, ≥30 ms

Tuned on synthetic tests and one real song. If you see false positives or misses on real material, note them in `references/log.md` with the date and adjust.
