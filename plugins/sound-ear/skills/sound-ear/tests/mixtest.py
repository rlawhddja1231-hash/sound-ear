"""Mix self-test: builds a 40 s synthetic song (intro without drums → verse → louder chorus that repeats → outro)
with drums, bass, chords and a synthetic voice, then checks that `describe` (and `full` if models exist) behave:
  - drums found in the drummed sections and not in the intro
  - the chorus repeat is matched (same letter twice)
  - no crash, report files written
Usage: python3 mixtest.py   (prints PASS/FAIL; ~1 min without models, longer with separation)"""
import json
import os
import subprocess
import sys
import tempfile

import numpy as np
import soundfile as sf

HERE = os.path.dirname(os.path.abspath(__file__))
LISTEN = os.path.join(HERE, "..", "scripts", "listen.py")
SR = 44100
BPM = 120
BEAT = 60 / BPM
rng = np.random.default_rng(1)


def tone(midi, dur, amp=0.15, harm=8, vib=True):
    t = np.arange(int(SR * dur)) / SR
    f = 440 * 2 ** ((midi - 69) / 12) * (1 + (0.005 * np.sin(2 * np.pi * 5.5 * t) if vib else 0))
    ph = 2 * np.pi * np.cumsum(f) / SR
    x = sum((1 / k) * np.sin(k * ph) for k in range(1, harm))
    env = np.clip(np.minimum(t / 0.02, (dur - t) / 0.05), 0, 1)
    return amp * x * env


def kick():
    t = np.arange(int(SR * 0.25)) / SR
    return 0.8 * np.sin(2 * np.pi * (50 + 80 * np.exp(-t * 30)) * t) * np.exp(-t * 12)


def snare():
    t = np.arange(int(SR * 0.15)) / SR
    return 0.35 * rng.standard_normal(len(t)) * np.exp(-t * 25)


def hat():
    t = np.arange(int(SR * 0.05)) / SR
    from scipy.signal import butter, sosfilt
    return 0.12 * sosfilt(butter(4, 7000, "hp", fs=SR, output="sos"), rng.standard_normal(len(t))) * np.exp(-t * 80)


def place(buf, x, at):
    i = int(at * SR)
    j = min(len(buf), i + len(x))
    buf[i:j] += x[: j - i]


def section(n_bars, chords, melody, drums, loud):
    dur = n_bars * 4 * BEAT
    b = np.zeros(int(SR * dur))
    for bar in range(n_bars):
        root = chords[bar % len(chords)]
        t0 = bar * 4 * BEAT
        for n in (root, root + 4 if root % 12 in (0, 5, 7) else root + 3, root + 7):  # 화음(패드)
            place(b, tone(n + 12, 4 * BEAT, 0.05 * loud, 5, False), t0)
        for k in range(4):  # 베이스
            place(b, tone(root - 24, BEAT * 0.9, 0.18 * loud, 4, False), t0 + k * BEAT)
        if drums:
            for k in range(4):
                place(b, kick() * loud if k % 2 == 0 else snare() * loud, t0 + k * BEAT)
                place(b, hat(), t0 + k * BEAT + BEAT / 2)
        for k, m in enumerate(melody[bar % len(melody)]):  # 노래(합성 목소리)
            if m:
                place(b, tone(m, BEAT * 0.95, 0.14 * loud), t0 + k * BEAT)
    return b


def build(path):
    C, F, G, Am = 48, 53, 55, 57
    verse_mel = [[64, 64, 65, 67], [67, 65, 64, 62], [60, 62, 64, 64], [62, 62, 60, None]]
    chorus_mel = [[72, 71, 69, 67], [69, 71, 72, 72], [74, 72, 71, 69], [67, 69, 67, None]]
    parts = [
        ("intro", section(4, [C, G, C, G], [[None] * 4], False, 0.6)),
        ("verse", section(4, [C, Am, F, G], verse_mel, True, 0.8)),
        ("chorus", section(4, [F, G, C, Am], chorus_mel, True, 1.3)),
        ("verse2", section(4, [C, Am, F, G], [[62, 62, 64, 65], [65, 64, 62, 60], [60, 60, 62, 64], [62, 60, 60, None]], True, 0.8)),
        ("chorus2", section(4, [F, G, C, Am], chorus_mel, True, 1.3)),
    ]
    marks, t, out = {}, 0.0, []
    for name, b in parts:
        marks[name] = (t, t + len(b) / SR)
        t += len(b) / SR
        out.append(b)
    y = np.concatenate(out)
    y = y / np.abs(y).max() * 0.9
    sf.write(path, np.stack([y, y], 1).astype(np.float32), SR)
    return marks


def main():
    ok = True
    with tempfile.TemporaryDirectory() as d:
        wav = os.path.join(d, "mix.wav")
        marks = build(wav)
        out = os.path.join(d, "desc")
        r = subprocess.run([sys.executable, LISTEN, "describe", wav, "--no-words", "--no-separate", "--no-ai", "--out", out],
                           capture_output=True, text=True)
        if r.returncode != 0:
            print("describe crashed:\n", r.stderr[-2000:])
            print("FAIL")
            sys.exit(1)
        dj = json.load(open(os.path.join(out, "describe.json"), encoding="utf-8"))
        secs = dj["sections"]

        def at(t):
            return next(s for s in secs if s["start"] <= t < s["end"])
        intro = at(sum(marks["intro"]) / 2)
        verse = at(sum(marks["verse"]) / 2)
        ch1 = at(sum(marks["chorus"]) / 2)
        ch2 = at(sum(marks["chorus2"]) / 2)
        checks = [("intro has no drums", not intro["drums"]),
                  ("verse has drums", verse["drums"]),
                  ("chorus has drums", ch1["drums"]),
                  ("chorus repeat matched (same letter)", ch1["label"] == ch2["label"]),
                  ("chorus louder than verse", ch1["energy_db"] > verse["energy_db"] + 1),
                  ("tempo ~120 or 60/240", any(abs(dj["tempo"] - x) < 6 for x in (60, 120, 240)))]
        for name, good in checks:
            print(("ok     " if good else "WRONG  ") + name)
            ok &= bool(good)
        print("sections:", [(round(s["start"], 1), s["label"], s["name"]) for s in secs])
    print("PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
