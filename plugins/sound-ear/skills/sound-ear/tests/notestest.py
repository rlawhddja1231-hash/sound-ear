#!/usr/bin/env python3
"""노트 검사 자체 시험: 작은 브이신스(.svp) 프로젝트를 만들고, 그 노트대로 소리를 합성하되 일부러 흠을 심은 뒤
notes 명령이 그 노트들을 짚는지 본다.  심은 것: 노트 4 +45센트, 노트 7 쓴 길이 50ms, 노트 9 소리 없음, 노트 10 실제 소리 60ms, 노트 11 가운데 틱, 앞에 0.4초 빈 소리."""
import json
import os
import subprocess
import sys
import tempfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
LISTEN = os.path.join(HERE, "..", "scripts", "listen.py")
Q = 705600000
BPM = 120.0
LYR = ["나", "는", "밤", "하", "늘", "의", "별", "을", "세", "다", "가", "너"]
PITCH = [60, 62, 64, 65, 67, 69, 71, 72, 71, 69, 67, 65]


def make(tmp):
    notes, pos = [], 0
    for i, (l, p) in enumerate(zip(LYR, PITCH)):
        dur = Q // 2 if i != 6 else Q // 20  # 8분음표 = 0.25초, 노트 7만 50ms
        notes.append(dict(onset=pos, duration=dur, lyrics=l, phonemes="", pitch=p))
        pos += dur
    svp = dict(version=1, time=dict(meter=[dict(index=0, numerator=4, denominator=4)], tempo=[dict(position=0, bpm=BPM)]),
               library=[], tracks=[dict(name="보컬", mainGroup=dict(name="main", notes=notes), mainRef=dict(blickOffset=0))])
    sp = os.path.join(tmp, "test.svp")
    json.dump(svp, open(sp, "w", encoding="utf-8"), ensure_ascii=False)
    sr = 44100
    lead = 0.4
    total = lead + len(notes) * 0.25 + 0.5
    y = np.zeros(int(total * sr))
    for i, n in enumerate(notes, 1):
        a = lead + n["onset"] / Q * 60 / BPM
        d = n["duration"] / Q * 60 / BPM
        midi = n["pitch"] + (0.45 if i == 4 else 0.0)
        if i == 10:
            d = 0.06
        if i == 9:
            continue
        f = 440 * 2 ** ((midi - 69) / 12)
        t = np.arange(int(d * sr)) / sr
        env = np.minimum(1, t / 0.02) * np.minimum(1, (d - t) / 0.08)
        seg = sum((0.5 ** k) * np.sin(2 * np.pi * f * (k + 1) * t) for k in range(5)) * env * 0.25
        s0 = int(a * sr)
        y[s0: s0 + len(seg)] += seg[: len(y) - s0]
        if i == 11:
            e = s0 + len(seg) // 2
            y[e: e + 40] += 0.9 * np.sign(np.random.randn(40))
    import soundfile as sf
    wp = os.path.join(tmp, "test_vocal.wav")
    sf.write(wp, (y / max(1e-9, np.abs(y).max()) * 0.8).astype(np.float32), sr)
    return sp, wp


def main():
    tmp = tempfile.mkdtemp()
    sp, wp = make(tmp)
    out = os.path.join(tmp, "out")
    subprocess.run([sys.executable, LISTEN, "notes", wp, sp, "--out", out], check=True, capture_output=True)
    j = json.load(open(os.path.join(out, "notes.json"), encoding="utf-8"))
    flagged = {n["i"]: [x["msg"] for x in n["issues"]] for n in j["notes"] if n["issues"]}
    ok = True
    def want(i, word):
        nonlocal ok
        hit = any(word in m for m in flagged.get(i, []))
        print(("found " if hit else "MISS  ") + f"노트 {i}: {word}")
        ok &= hit
    want(4, "음정")
    want(7, "짧아")
    want(9, "소리가 없음")
    want(10, "실제 소리")
    want(11, "틱")
    extra = [i for i in flagged if i not in (4, 7, 9, 10, 11)]
    print(f"offset {j['offset']:+.2f}s (심은 값 +0.40)  extra flagged: {extra}")
    ok &= abs(j["offset"] - 0.4) < 0.06 and len(extra) <= 1
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
