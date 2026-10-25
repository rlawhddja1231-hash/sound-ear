#!/usr/bin/env python3
"""listen.py — 소리 듣기 도구 (Claude가 소리 파일을 '듣는' 도구)

쓰는 법
  python3 listen.py check  노래.wav [--mode vocal|mix|auto] [--out 폴더]
  python3 listen.py cut    노래.wav 1:23.5 [--len 2] [--out 폴더]
  python3 listen.py compare 앞판.wav 새판.wav [--out 폴더]
  python3 listen.py describe 노래.wav [--words 가사폴더/words.json] [--out 폴더]
  python3 listen.py separate 노래.wav [--out 폴더]   (보컬.wav, 반주.wav)
  python3 listen.py words  노래.wav [--lang ko] [--model ggml-small-q5_1.bin] [--out 폴더]

check   : 소리 전체를 검사해 보고서(보고서.md, result.json)와 그림을 만든다.
cut     : 정한 시각 앞뒤를 잘라 소리 조각과 확대 그림을 만든다.
compare : 두 판의 크기·길이·음높이를 견준다.

준비물: pip install --break-system-packages librosa soundfile pyloudnorm matplotlib koreanize-matplotlib scipy
        ffmpeg (작업 공간에 보통 있음)
"""
import argparse
import json
import math
import os
import subprocess
import sys
import warnings

import numpy as np

warnings.filterwarnings("ignore")

NOTE_EN = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
NOTE_KO = ["도", "도#", "레", "레#", "미", "파", "파#", "솔", "솔#", "라", "라#", "시"]
MAJOR_KO = ["다", "올림다", "라", "내림마", "마", "바", "올림바", "사", "내림가", "가", "내림나", "나"]

LEVEL_MUST = "꼭 고칠 것"
LEVEL_CHECK = "들어 보고 판단"
LEVEL_INFO = "참고"


# ---------------------------------------------------------------- 공통
def fmt_t(t):
    t = round(max(0.0, float(t)), 1)
    m = int(t // 60)
    s = t - m * 60
    return f"{m}:{s:04.1f}"


def parse_t(s):
    s = str(s).strip()
    if ":" in s:
        parts = s.split(":")
        v = 0.0
        for p in parts:
            v = v * 60 + float(p)
        return v
    return float(s)


def db(x, floor=-120.0):
    x = np.asarray(x, dtype=np.float64)
    with np.errstate(divide="ignore"):
        v = 20 * np.log10(np.maximum(x, 1e-12))
    return np.maximum(v, floor)


def note_name(midi):
    n = int(round(midi))
    pc = n % 12
    octv = n // 12 - 1
    return f"{NOTE_EN[pc]}{octv}({NOTE_KO[pc]})"


def probe(path):
    info = {}
    try:
        out = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries",
             "stream=codec_name,sample_rate,channels,bits_per_sample,bits_per_raw_sample,bit_rate:format=duration,bit_rate",
             "-select_streams", "a:0", "-of", "json", path],
            capture_output=True, text=True, check=True).stdout
        j = json.loads(out)
        st = (j.get("streams") or [{}])[0]
        info = {
            "형식": st.get("codec_name"),
            "샘플레이트": int(st.get("sample_rate") or 0),
            "채널": int(st.get("channels") or 0),
            "비트": st.get("bits_per_raw_sample") or st.get("bits_per_sample"),
            "비트레이트": st.get("bit_rate") or j.get("format", {}).get("bit_rate"),
        }
    except Exception as e:  # noqa
        info = {"오류": str(e)}
    return info


def load(path, sr=None):
    """ffmpeg로 어떤 파일이든(영상 포함) 풀어서 (스테레오 배열, 샘플레이트) 를 준다."""
    if sr is None:
        sr = probe(path).get("샘플레이트") or 44100
    ch = probe(path).get("채널") or 2
    ch = 1 if ch == 1 else 2  # 5.1 같은 것은 2채널로 내림
    cmd = ["ffmpeg", "-v", "error", "-i", path, "-vn", "-f", "f32le", "-ac", str(ch), "-ar", str(sr), "-"]
    r = subprocess.run(cmd, capture_output=True)
    if r.returncode != 0 or not r.stdout:
        sys.exit(f"소리를 못 열었습니다: {path}\n{r.stderr.decode(errors='ignore')[:500]}")
    st = np.frombuffer(r.stdout, np.float32).reshape(-1, ch).astype(np.float64)
    if ch == 1:
        st = np.repeat(st, 2, axis=1)  # 모노는 양쪽에 그대로 복사(크기 안 바뀜)
    return st, sr


def group_times(times, gap):
    """가까운 시각끼리 묶어 [(시작, 끝, 개수)]"""
    out = []
    for t in sorted(times):
        if out and t - out[-1][1] <= gap:
            out[-1][1] = t
            out[-1][2] += 1
        else:
            out.append([t, t, 1])
    return [tuple(x) for x in out]


def runs(mask):
    """True가 이어진 구간 [(시작칸, 끝칸+1)]"""
    mask = np.asarray(mask, bool)
    if not mask.any():
        return []
    d = np.diff(np.concatenate([[0], mask.astype(np.int8), [0]]))
    starts = np.where(d == 1)[0]
    ends = np.where(d == -1)[0]
    return list(zip(starts, ends))


# ---------------------------------------------------------------- 검사들
def check_levels(st, sr, probs, stats):
    import pyloudnorm as pyln
    from scipy.signal import resample_poly

    mono = st.mean(axis=1)
    peak = float(np.abs(st).max())
    stats["최고 크기(dBFS)"] = round(float(db(peak)), 2)
    stats["평균 크기 RMS(dBFS)"] = round(float(db(np.sqrt(np.mean(mono ** 2)))), 2)

    # 통합 라우드니스
    try:
        meter = pyln.Meter(sr)
        lufs = float(meter.integrated_loudness(st))
        stats["라우드니스(LUFS)"] = round(lufs, 1) if np.isfinite(lufs) else None
    except Exception:
        lufs = None
        stats["라우드니스(LUFS)"] = None

    # 트루 피크(4배로 늘려서 봄) — 앞 3분만(속도)
    seg = st[: sr * 180]
    tp = float(np.abs(resample_poly(seg, 4, 1, axis=0)).max()) if len(seg) > 100 else peak
    stats["트루 피크(dBTP)"] = round(float(db(tp)), 2)

    # 클리핑: 0.999 이상이 3샘플 넘게 이어짐
    clip_times = []
    for ch in range(2):
        m = np.abs(st[:, ch]) >= 0.999
        for a, b in runs(m):
            if b - a >= 3:
                clip_times.append(a / sr)
    groups = group_times(clip_times, 0.05)
    stats["클리핑 횟수"] = len(groups)
    lossy = str(stats.get("형식", "")).lower() in ("opus", "vorbis", "mp3", "aac", "mp2", "wmav2", "ac3")
    if lossy and groups:
        # 손실 압축(유튜브 녹음, mp3 등)은 풀 때 파형이 0dB를 살짝 넘게 출렁여 '깨짐'을 정확히 가릴 수 없다
        secs = sorted({int(a // 10) * 10 for a, b, n in groups})
        spans = ", ".join(f"{fmt_t(x)[:-2]}~" for x in secs[:12])
        stats["클리핑 횟수"] = f"{len(groups)} (손실 압축이라 참고만)"
        probs.append(dict(t=0, end=None, kind="소리가 0dB까지 꽉 참", level=LEVEL_CHECK,
                          msg=f"{len(groups)}곳에서 최대치에 닿습니다(많은 구간: {spans}). 손실 압축 파일이라 진짜로 깨졌는지는 "
                              "원본 wav로 봐야 정확합니다. 마스터가 아주 세게 눌린 것은 맞습니다."))
        groups = []
    for a, b, n in groups[:40]:
        probs.append(dict(t=a, end=b, kind="클리핑(소리 깨짐)", level=LEVEL_MUST,
                          msg=f"소리가 최대치를 넘어 잘렸습니다({n}곳 묶음). 볼륨이나 마스터를 낮추세요."))
    if len(groups) > 40:
        probs.append(dict(t=groups[40][0], end=groups[-1][1], kind="클리핑(소리 깨짐)", level=LEVEL_MUST,
                          msg=f"이 뒤로도 {len(groups) - 40}곳 더 있습니다."))
    if not groups and tp > 1.0:
        probs.append(dict(t=0, end=None, kind="트루 피크 넘침", level=LEVEL_CHECK,
                          msg=f"샘플은 안 넘지만 실제 재생 때 {db(tp):.1f}dBTP까지 올라가 mp3로 바꾸면 깨질 수 있습니다. 끝을 -1dB로 맞추세요."))

    if lufs is not None and np.isfinite(lufs):
        if lufs > -9:
            probs.append(dict(t=0, end=None, kind="너무 큼", level=LEVEL_CHECK,
                              msg=f"전체 크기 {lufs:.1f} LUFS — 유튜브는 -14 근처로 줄여 버립니다. 이렇게 크면 소리만 납작해집니다."))
        elif lufs < -20:
            probs.append(dict(t=0, end=None, kind="너무 작음", level=LEVEL_INFO,
                              msg=f"전체 크기 {lufs:.1f} LUFS — 완성곡이면 -14 근처가 보통입니다(보컬만 따로 뽑은 파일이면 괜찮음)."))

    dc = float(np.abs(mono.mean()))
    if dc > 0.01:
        probs.append(dict(t=0, end=None, kind="DC 치우침", level=LEVEL_CHECK,
                          msg=f"파형 가운데가 0에서 {dc:.3f}만큼 치우쳤습니다. 하이패스(20Hz)를 거세요."))

    # 스테레오
    L, R = st[:, 0], st[:, 1]
    if np.allclose(L, R, atol=1e-4):
        stats["스테레오"] = "모노(양쪽 같음)"
    else:
        c = float(np.corrcoef(L, R)[0, 1]) if L.std() > 0 and R.std() > 0 else 1.0
        stats["스테레오"] = f"스테레오(좌우 닮음 {c:.2f})"
        if c < 0:
            probs.append(dict(t=0, end=None, kind="위상 반대", level=LEVEL_MUST,
                              msg=f"좌우가 서로 반대로 움직입니다(닮음 {c:.2f}). 휴대폰 스피커(모노)에서 소리가 사라질 수 있습니다."))


def frame_rms(x, sr, hop_s=0.01):
    hop = max(1, int(sr * hop_s))
    n = len(x) // hop
    if n == 0:
        return np.array([0.0]), hop
    fr = x[: n * hop].reshape(n, hop)
    return np.sqrt((fr ** 2).mean(axis=1)), hop


def check_silence(st, sr, probs, stats, mode):
    mono = st.mean(axis=1)
    rms, hop = frame_rms(mono, sr, 0.01)
    lv = db(rms)
    loud = lv > -50
    if loud.any():
        first = int(np.argmax(loud))
        last = len(loud) - 1 - int(np.argmax(loud[::-1]))
        stats["앞 무음(초)"] = round(first * 0.01, 2)
        stats["뒤 무음(초)"] = round((len(loud) - 1 - last) * 0.01, 2)
        if first * 0.01 > 3:
            probs.append(dict(t=0, end=first * 0.01, kind="앞 무음 김", level=LEVEL_INFO,
                              msg=f"처음 {first * 0.01:.1f}초 동안 소리가 없습니다."))
    else:
        probs.append(dict(t=0, end=None, kind="소리 없음", level=LEVEL_MUST, msg="파일 전체가 거의 무음입니다."))
        return lv

    # 뚝 끊김: 10ms 안에 -25dB 이상에서 -60dB 아래로
    cut_times = []
    for i in range(len(lv) - 2):
        if lv[i] > -25 and min(lv[i + 1], lv[i + 2]) < -60:
            cut_times.append((i + 1) * 0.01)
    for a, b, n in group_times(cut_times, 0.3)[:20]:
        probs.append(dict(t=a, end=None, kind="뚝 끊김", level=LEVEL_CHECK,
                          msg="소리가 자연스럽게 줄지 않고 한순간에 끊깁니다. 노트 끝이나 리전 끝을 확인하세요(일부러 넣은 끊어치기 효과면 괜찮음)."))

    # 파일 끝이 소리 나는 중에 잘림
    if lv[-1] > -40 and len(lv) > 5:
        probs.append(dict(t=(len(lv) - 1) * 0.01, end=None, kind="끝이 잘림", level=LEVEL_MUST,
                          msg="파일 끝에서 소리가 아직 나고 있습니다. 내보내기 범위를 늘리거나 페이드아웃을 넣으세요."))
    if lv[0] > -30:
        probs.append(dict(t=0, end=0.0, kind="시작이 잘림", level=LEVEL_CHECK,
                          msg="파일 첫 순간부터 큰 소리가 납니다. 앞이 잘렸는지 확인하세요."))

    # 소음 바닥
    quiet = lv[(lv > -110) & (lv < -40)]
    if len(quiet) > 50:
        nf = float(np.percentile(quiet, 10))
        stats["바탕 잡음(dBFS)"] = round(nf, 1)
        if mode == "vocal" and nf > -60:
            probs.append(dict(t=0, end=None, kind="바탕 잡음", level=LEVEL_CHECK,
                              msg=f"조용해야 할 곳에도 {nf:.0f}dB 정도 잡음이 깔려 있습니다."))
    return lv


def check_clicks(st, sr, probs, stats, mode="vocal"):
    from scipy.ndimage import median_filter, maximum_filter1d
    from scipy.signal import butter, sosfilt

    mono = st.mean(axis=1)
    sos = butter(4, 3000, btype="highpass", fs=sr, output="sos")
    hp = np.abs(sosfilt(sos, mono))
    f = max(1, int(sr * 0.001))  # 1ms 칸
    n = len(hp) // f
    if n < 50:
        stats["딸깍 소리"] = 0
        return
    fm = hp[: n * f].reshape(n, f).max(axis=1)
    med = median_filter(fm, size=151, mode="nearest") + 1e-6
    # 주변(±3~8ms)의 최댓값
    left = np.concatenate([np.zeros(8), fm[:-8]]) if n > 8 else fm * 0
    near = np.zeros(n)
    for k in list(range(-8, -2)) + list(range(3, 9)):
        sh = np.roll(fm, k)
        near = np.maximum(near, sh)
    cand = (fm > 8 * med) & (fm > 0.02) & (near < fm / 3)
    # 소리가 새로 시작되는 곳(북·건반·자음의 첫소리)은 빼고 본다: 진짜 딸깍은 앞뒤 크기가 비슷하다
    low = sosfilt(butter(4, 150, btype="lowpass", fs=sr, output="sos"), mono)
    keep = []
    cut_end = set()
    w = int(sr * 0.02)
    for i in np.where(cand)[0]:
        c = i * f
        before = mono[max(0, c - w - f * 2): max(1, c - f * 2)]
        after = mono[c + f * 3: c + f * 3 + w]
        if len(before) < w // 2 or len(after) < w // 2:
            continue
        rb = np.sqrt(np.mean(before ** 2)) + 1e-9
        ra = np.sqrt(np.mean(after ** 2)) + 1e-9
        if ra / rb > 1.5:  # 3.5dB 넘게 커짐 = 새 소리의 시작
            continue
        # 킥 드럼: 튐과 동시에 낮은 소리(150Hz 아래)가 확 커지면 북 소리지 딸깍이 아니다
        # [확인: 10-25 새 시유 「Alone」 1:00.4 — 확대해 보니 앞뒤 킥과 똑같은 모양]
        lb = low[max(0, c - w - f * 2): max(1, c - f * 2)]
        la = low[c: c + 2 * w]
        if len(lb) and len(la):
            rl = np.sqrt(np.mean(la ** 2)) / (np.sqrt(np.mean(lb ** 2)) + 1e-9)
            # 완성곡은 베이스가 계속 깔려 있어 킥이 와도 1.3배쯤만 커진다 → 완성곡은 1.25배, 그 밖은 2배
            if rl > (1.25 if mode == "mix" else 2.0) and np.sqrt(np.mean(la ** 2)) > 0.02:
                continue
        keep.append(i * 0.001)
        if ra / rb < 0.3:  # 큰 소리가 갑자기 끊기면서 난 틱 [확인: 10-24 테토 보컬 노트 끝]
            cut_end.add(round(i * 0.001, 3))
    times = keep
    groups = group_times(times, 0.03)
    stats["딸깍 소리"] = len(groups)
    # 완성곡에서 10초 안에 틱이 4개 넘게 몰리면 한 줄로 묶는다(끊어 자르기·글리치 같은 일부러 넣은 효과일 때가 많음)
    # [확인: 10-24 Gumi 완성곡 1:08~1:18에 몰린 6곳]
    if mode == "mix" and len(groups) >= 4:
        starts = [g[0] for g in groups]
        merged, used = [], set()
        for i, a0 in enumerate(starts):
            if i in used:
                continue
            idx = [j for j, a1 in enumerate(starts) if a0 <= a1 <= a0 + 10 and j not in used]
            if len(idx) >= 4:
                used.update(idx)
                merged.append((starts[idx[0]], starts[idx[-1]], len(idx)))
        for a0, a1, n in merged:
            probs.append(dict(t=a0, end=a1, kind="틱이 몰린 구간", level=LEVEL_CHECK,
                              msg=f"{n}곳에서 짧은 틱이 몰려 있습니다. 일부러 넣은 끊어 자르기·글리치 효과면 괜찮고, 아니면 이 구간의 편집 이음새를 보세요."))
        groups = [g for i, g in enumerate(groups) if i not in used]
    for a, b, k in groups[:30]:
        if any(abs(a - c) < 0.03 for c in cut_end):
            probs.append(dict(t=a, end=None, kind="노트 끝 틱", level=LEVEL_CHECK,
                              msg="큰 소리가 서서히 줄지 않고 한순간에 끊기면서 '틱' 소리가 납니다. 노트 끝에 짧은 페이드를 넣거나 노트 길이를 확인하세요."))
            continue
        if mode == "mix":
            probs.append(dict(t=a, end=None, kind="딸깍·틱 소리", level=LEVEL_CHECK,
                              msg="아주 짧게 튀는 소리가 있습니다. 반주가 섞인 곡에서는 하이햇·자음 같은 짧은 소리도 잡히니 들어 보고 판단하세요."))
            continue
        probs.append(dict(t=a, end=None, kind="딸깍·틱 소리", level=LEVEL_MUST,
                          msg="아주 짧게 튀는 소리가 있습니다(노트 이음새, 잘린 파형, 플러그인 잡음일 수 있음)."))
    if len(groups) > 30:
        probs.append(dict(t=groups[30][0], end=groups[-1][0], kind="딸깍·틱 소리", level=LEVEL_MUST,
                          msg=f"이 뒤로도 {len(groups) - 30}곳 더 있습니다."))


def estimate_key_tempo(y, sr, stats, probs):
    import librosa

    try:
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        tempo = float(np.atleast_1d(tempo)[0])
        if tempo > 170:
            stats["빠르기(BPM, 추정)"] = f"{tempo / 2:.1f} (또는 {tempo:.1f})"
        elif tempo < 70:
            stats["빠르기(BPM, 추정)"] = f"{tempo:.1f} (또는 {tempo * 2:.1f})"
        else:
            stats["빠르기(BPM, 추정)"] = round(tempo, 1)
    except Exception:
        stats["빠르기(BPM, 추정)"] = None

    chroma = librosa.feature.chroma_cqt(y=y, sr=sr).mean(axis=1)
    maj = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
    mnr = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])
    best = []
    for i in range(12):
        best.append((np.corrcoef(chroma, np.roll(maj, i))[0, 1], i, "major"))
        best.append((np.corrcoef(chroma, np.roll(mnr, i))[0, 1], i, "minor"))
    best.sort(reverse=True)
    c, i, kind = best[0]
    if kind == "major":
        name = f"{NOTE_EN[i]} 메이저({MAJOR_KO[i]}장조)"
        scale = [(i + s) % 12 for s in [0, 2, 4, 5, 7, 9, 11]]
    else:
        name = f"{NOTE_EN[i]} 마이너({MAJOR_KO[i]}단조)"
        scale = [(i + s) % 12 for s in [0, 2, 3, 5, 7, 8, 10, 11]]  # 화성단음계 이끔음 포함
    sure = "높음" if c - best[1][0] > 0.05 and c > 0.6 else "낮음"
    stats["조(키, 추정)"] = f"{name} — 확신 {sure}"
    stats["_scale"] = scale
    return scale


def check_pitch(y, sr, probs, stats, scale):
    """보컬 전용: 음정, 음 튐, 흔들림, 음역"""
    import librosa
    from scipy.ndimage import median_filter

    hop = 256
    f0, vflag, vprob = librosa.pyin(y, fmin=65.0, fmax=1400.0, sr=sr, frame_length=2048, hop_length=hop)
    t = librosa.times_like(f0, sr=sr, hop_length=hop)
    dt = hop / sr
    # 거의 무음인 곳에서 음높이를 지어내지 않게 함 [확인: 10-24 곡 끝 무음에서 E6를 1.4초 잡음]
    fr_rms = librosa.feature.rms(y=y, frame_length=2048, hop_length=hop)[0][: len(f0)]
    if vflag.any():
        vflag = vflag & (fr_rms > max(1e-4, 0.05 * float(np.median(fr_rms[vflag]))))
    midi = np.where(vflag, 69 + 12 * np.log2(np.where(np.isfinite(f0), f0, 1.0) / 440.0), np.nan)
    voiced_frac = float(np.mean(vflag))
    stats["목소리 있는 시간 비율"] = round(voiced_frac, 2)

    vr = runs(vflag)
    # 음 튐: 짧은 창(yin, 5.8ms 간격)으로 다시 보고, 앞뒤 중앙값에서 3반음 넘게 벗어났다가 80ms 안에 돌아오는 곳
    glitch_t = []
    h2 = 128
    f2 = librosa.yin(y, fmin=65.0, fmax=1400.0, sr=sr, frame_length=1024, hop_length=h2)
    t2 = librosa.times_like(f2, sr=sr, hop_length=h2)
    m2 = 69 + 12 * np.log2(np.maximum(f2, 1.0) / 440.0)
    r2 = librosa.feature.rms(y=y, frame_length=1024, hop_length=h2)[0][: len(f2)]
    # 확인용 스펙트럼: 튐이 진짜인지(배음이 정말 그 높이로 옮겨 갔는지) 본다
    S2 = np.abs(librosa.stft(y, n_fft=2048, hop_length=h2, win_length=1024))
    fr2 = librosa.fft_frequencies(sr=sr, n_fft=2048)

    def salience(frame, hz):
        if frame >= S2.shape[1] or hz <= 0:
            return 0.0
        col = S2[:, frame]
        tot = 0.0
        for k in range(1, 6):
            f = hz * k
            if f >= fr2[-1]:
                break
            b = int(round(f / (fr2[1] - fr2[0])))
            tot += col[max(0, b - 1): b + 2].max()
        return tot

    def peak(frame, hz):
        if frame >= S2.shape[1] or hz <= 20:
            return 0.0
        b = int(round(hz / (fr2[1] - fr2[0])))
        return float(S2[max(0, b - 1): b + 2, frame].max())
    v2 = np.interp(t2, t, vflag.astype(float)) > 0.5
    gmed = float(np.median(r2[v2])) if v2.any() else 0.0
    raw_t = []
    for a, b in runs(v2):
        if b - a < 20:
            continue
        if np.median(r2[a:b]) < 0.15 * gmed:  # 숨·잡음 같은 아주 작은 소리 덩어리는 건너뜀
            continue
        seg = m2[a:b]
        sm = median_filter(seg, size=21, mode="nearest")
        dev = np.abs(seg - sm)
        rr = r2[a:b]
        # 소리가 커지는 중·작아지는 중(노트 시작·끝의 미끄러짐)은 빼고 봄.
        # 0.3이던 기준을 0.6으로 올림 [확인: 10-24 신스V 보컬에서 노트 끝 내려가는 소리를 튐으로 헛짚음]
        strong = rr > 0.6 * np.median(rr)
        medr = np.median(rr)
        for ga0, gb0 in runs(dev > 3.0):
            raw_t.append(t2[a + ga0])
        for ga, gb in runs((dev > 3.0) & strong):
            jump = abs(float(np.median(seg[ga:gb] - sm[ga:gb])))
            # 한 옥타브(12)·두 옥타브(24) 차이는 음높이 찾기가 흔히 헷갈리는 것(노트 끝이 작아질 때)이라,
            # 소리가 충분히 클 때만 진짜 튐으로 본다 [확인: 10-24 신스V 보컬 녹음에서 헛짚은 39곳 대부분]
            octave = min(abs(jump - 12), abs(jump - 24), abs(jump - 19)) < 1.0
            if octave and np.min(rr[ga:gb]) < 0.8 * medr:
                continue
            # 갑자기 꺾여야 튐이다: 부드럽게 내려갔다 올라오는 V자(이음새의 자연스러운 꺾임)는 뺌
            lo_i, hi_i = max(1, ga - 2), min(len(seg) - 1, gb + 2)
            if np.max(np.abs(np.diff(seg[lo_i - 1: hi_i + 1]))) < 2.0:
                continue
            if gb - ga <= int(0.08 / (h2 / sr)) and ga > 4 and gb < len(seg) - 4:
                # 배음 확인: 튄 높이의 배음이 원래 높이의 배음보다 뚜렷해야 진짜 튐
                # [확인: 10-24 신스V 보컬에서 yin이 옥타브·자음을 헷갈린 곳을 걸러냄]
                k = ga + int(np.argmax(dev[ga:gb]))
                hz_new = 440.0 * 2 ** ((seg[k] - 69) / 12)
                hz_base = 440.0 * 2 ** ((sm[k] - 69) / 12)
                if salience(a + k, hz_new) < 1.2 * salience(a + k, hz_base):
                    continue
                if peak(a + k, hz_new / 2) > (0.25 if octave else 0.5) * peak(a + k, hz_new):  # 한 옥타브 아래가 진짜 바탕음 = 헷갈린 것
                    continue
                # 원래 음의 바탕음이 튄 순간에도 그대로 남아 있으면, 음이 바뀐 게 아니라 찾기가 헷갈린 것
                # [확인: 10-24 노트 끝에서 2배음이 더 커지자 한 옥타브 위로 잘못 잡음]
                ref = peak(a + max(0, ga - 3), hz_base)
                if ref > 0 and peak(a + k, hz_base) >= 0.5 * ref:
                    continue
                # 튄 높이에 실제 바탕음 소리가 있어야 함(없으면 음 바뀜 사이의 잡소리를 잘못 읽은 것)
                if ref > 0 and peak(a + k, hz_new) < 0.3 * ref:
                    continue
                glitch_t.append((t2[a + ga], octave))
    # 1초 안에 흔들림 후보가 4개 넘게 몰린 곳은 화음(두 목소리 이상)·잡음이라 음높이를 믿을 수 없다
    # [확인: 10-24 테토 보컬의 화음 구간에서 두 줄 사이를 오가며 튐으로 헛짚음]
    raw_arr = np.array(raw_t)
    busy = []
    kept = []
    for g in glitch_t:
        n_near = int(np.sum(np.abs(raw_arr - g[0]) <= 0.5)) if len(raw_arr) else 0
        (busy if n_near >= 4 else kept).append(g)
    glitch_t = kept
    stats["음높이를 믿기 어려운 곳(화음·잡음)"] = len(group_times([b[0] for b in busy], 1.0))
    m2v = np.where(v2 & (r2 > 0.3 * (np.median(r2[v2]) if v2.any() else 0)), m2, np.nan)
    gl = group_times([g[0] for g in glitch_t], 0.15)
    stats["음 튐"] = len(gl)
    for g0, g1, n in gl[:25]:
        probs.append(dict(t=g0, end=None, kind="음이 순간 튐", level=LEVEL_CHECK,
                          msg="음높이가 아주 잠깐 크게 튀었다 돌아옵니다(피치 곡선의 뾰족한 점, 노트 이음새를 확인)."))

    # 노트 나누기: 반올림 음이 같은 칸이 150ms 이상 이어지면 한 노트
    sm_all = np.full_like(midi, np.nan)
    for a, b in vr:
        if b - a >= 3:
            sm_all[a:b] = median_filter(midi[a:b], size=5, mode="nearest")
    rnd = np.where(np.isfinite(sm_all), np.round(sm_all), -1)
    notes = []
    i = 0
    N = len(rnd)
    while i < N:
        if rnd[i] < 0:
            i += 1
            continue
        j = i
        while j < N and rnd[j] == rnd[i]:
            j += 1
        if (j - i) * dt >= 0.15:
            seg = midi[i:j]
            notes.append(dict(a=i, b=j, t=t[i], dur=(j - i) * dt, note=int(rnd[i]),
                              med=float(np.nanmedian(seg)), std=float(np.nanstd(seg) * 100)))
        i = j
    stats["찾은 노트 수"] = len(notes)
    if not notes:
        return dict(t=t, midi=midi, notes=notes, vflag=vflag, t2=t2, m2=m2v)

    # 전체 튜닝 치우침(A=440 기준)
    devs = np.array([(n["med"] - n["note"]) * 100 for n in notes])
    w = np.array([n["dur"] for n in notes])
    glob = float(np.average(devs, weights=w))
    stats["전체 튜닝 치우침(센트)"] = round(glob, 1)
    if abs(glob) > 15:
        probs.append(dict(t=0, end=None, kind="전체 튜닝", level=LEVEL_CHECK,
                          msg=f"목소리 전체가 A=440 기준에서 {glob:+.0f}센트 어긋나 있습니다. 반주와 기준음이 같은지 보세요."))

    lo = min(n["note"] for n in notes)
    hi = max(n["note"] for n in notes)
    stats["음역"] = f"{note_name(lo)} ~ {note_name(hi)}"

    flagged = 0
    for n in notes:
        d = (n["med"] - n["note"]) * 100 - glob
        if n["dur"] >= 0.25 and abs(d) > 30 and flagged < 40:
            flagged += 1
            way = "높음" if d > 0 else "낮음"
            probs.append(dict(t=n["t"], end=n["t"] + n["dur"], kind="음정 어긋남", level=LEVEL_CHECK,
                              msg=f"{note_name(n['note'])} 근처인데 {abs(d):.0f}센트 {way}. 반음 사이에 걸쳐 어중간하게 들릴 수 있습니다."))
        if n["dur"] >= 0.4 and n["std"] > 85:
            probs.append(dict(t=n["t"], end=n["t"] + n["dur"], kind="음 흔들림 큼", level=LEVEL_CHECK,
                              msg=f"{note_name(n['note'])} 노트가 많이 흔들립니다(흔들림 {n['std']:.0f}센트). 비브라토가 너무 깊거나 피치 곡선이 울퉁불퉁할 수 있습니다."))
        if scale and n["dur"] >= 0.3 and (n["note"] % 12) not in scale:
            probs.append(dict(t=n["t"], end=n["t"] + n["dur"], kind="조에 없는 음", level=LEVEL_INFO,
                              msg=f"{note_name(n['note'])} — 추정한 조에 없는 음입니다. 일부러 넣은 게 아니면 노트를 확인하세요."))
    return dict(t=t, midi=midi, notes=notes, vflag=vflag, t2=t2, m2=m2v)


def check_sibilance_breath(y, sr, probs, stats, pitch):
    import librosa

    S = np.abs(librosa.stft(y, n_fft=1024, hop_length=256)) ** 2
    freqs = librosa.fft_frequencies(sr=sr, n_fft=1024)
    t = librosa.times_like(S[0], sr=sr, hop_length=256)
    tot = S.sum(axis=0) + 1e-12
    sib = S[(freqs >= 5000) & (freqs <= 10000)].sum(axis=0)
    tot_db = 10 * np.log10(tot)
    sib_db = 10 * np.log10(sib + 1e-12)
    loud = tot_db > tot_db.max() - 40
    if not loud.any():
        return
    ref = float(np.percentile(tot_db[loud], 75))
    m = (sib_db > ref - 6) & (sib / tot > 0.5)
    ev = []
    for a, b in runs(m):
        if (b - a) * 256 / sr >= 0.03:
            ev.append(t[a])
    groups = group_times(ev, 0.2)
    stats["센 ㅅ·ㅊ 소리"] = len(groups)
    for a, b, n in groups[:20]:
        probs.append(dict(t=a, end=None, kind="ㅅ·ㅊ 소리 셈", level=LEVEL_CHECK,
                          msg="'스·츠' 같은 쇳소리가 모음만큼 큽니다. 디에서(De-esser)를 걸거나 그 노트의 자음 크기를 줄이세요."))

    # 숨소리: 목소리 없는데 크고 잡음 같은 구간
    if pitch is None or "vflag" not in pitch:
        return
    flat = librosa.feature.spectral_flatness(S=np.sqrt(S))[0]
    vt = pitch["t"]
    vflag = np.interp(t, vt, pitch["vflag"].astype(float)) > 0.5
    cent = librosa.feature.spectral_centroid(S=np.sqrt(S), sr=sr)[0]
    m2 = (~vflag) & (tot_db > ref - 12) & (flat > 0.15) & (cent < 5000)
    bt = []
    for a, b in runs(m2):
        if (b - a) * 256 / sr >= 0.15:
            bt.append((t[a], t[min(b, len(t) - 1)]))
    stats["큰 숨소리"] = len(bt)
    for a, b in bt[:15]:
        probs.append(dict(t=a, end=b, kind="숨소리 큼", level=LEVEL_INFO,
                          msg="목소리 사이 숨소리(또는 잡음)가 꽤 큽니다. 일부러 넣은 숨이면 괜찮습니다."))


# ---------------------------------------------------------------- 그림
def setup_plt():
    import matplotlib
    matplotlib.use("Agg")
    try:
        import koreanize_matplotlib  # noqa
    except Exception:
        pass
    import matplotlib.pyplot as plt
    plt.rcParams["axes.unicode_minus"] = False
    return plt


KIND_COLOR = {LEVEL_MUST: "red", LEVEL_CHECK: "orange", LEVEL_INFO: "gray"}


def draw_view_fixed(y, sr, t0, t1, pitch, probs, path, title):
    """specshow의 시간축을 실제 시각에 맞추기 위해 extent로 직접 그린다."""
    import librosa
    plt = setup_plt()
    a, b = int(t0 * sr), int(min(t1, len(y) / sr) * sr)
    seg = y[a:b]
    if len(seg) < 2048:
        seg = np.pad(seg, (0, 2048 - len(seg)))
    rows = 3 if pitch is not None else 2
    fig, ax = plt.subplots(rows, 1, figsize=(16, 3.0 * rows), sharex=True)
    tt = np.arange(len(seg)) / sr + t0
    step = max(1, len(seg) // 20000)
    ax[0].plot(tt[::step], seg[::step], lw=0.5, color="steelblue")
    ax[0].set_ylim(-1.05, 1.05)
    ax[0].axhline(1, color="red", lw=0.5, ls=":")
    ax[0].axhline(-1, color="red", lw=0.5, ls=":")
    ax[0].set_ylabel("파형")
    ax[0].set_title(title)
    short = (t1 - t0) < 6
    nfft = 1024 if short else 2048
    D = librosa.amplitude_to_db(np.abs(librosa.stft(seg, n_fft=nfft, hop_length=64 if short else 256)), ref=np.max)
    freqs = librosa.fft_frequencies(sr=sr, n_fft=nfft)
    ax[1].imshow(D[1:], origin="lower", aspect="auto", cmap="magma", vmin=-80, vmax=0,
                 extent=[t0, t0 + len(seg) / sr, freqs[1], freqs[-1]])
    ax[1].set_yscale("symlog", linthresh=500)
    ax[1].set_ylim(50, freqs[-1])
    ax[1].set_ylabel("주파수(Hz)")
    if pitch is not None:
        pt, pm = pitch["t"], pitch["midi"]
        mk = (pt >= t0) & (pt <= t1)
        if short and "t2" in pitch:  # 확대 그림에는 촘촘한 음높이(연두 점)도 같이
            q = (pitch["t2"] >= t0) & (pitch["t2"] <= t1)
            ax[2].plot(pitch["t2"][q], pitch["m2"][q], "o", ms=3, color="limegreen", zorder=5)
        ax[2].plot(pt[mk], pm[mk], ".", ms=2.5, color="darkgreen", zorder=6)
        vals = pm[mk][np.isfinite(pm[mk])]
        if short and "t2" in pitch:
            v2 = pitch["m2"][q]
            vals = np.concatenate([vals, v2[np.isfinite(v2)]])
        if len(vals):
            lo, hi = int(np.floor(vals.min())) - 1, int(np.ceil(vals.max())) + 1
            ax[2].set_ylim(lo, hi)
            ticks = list(range(lo, hi + 1))
            ax[2].set_yticks(ticks)
            every = 1 if (hi - lo) < 26 else (2 if (hi - lo) < 40 else 12)
            ax[2].set_yticklabels([note_name(k) if k % every == 0 else "" for k in ticks], fontsize=7)
            ax[2].grid(axis="y", lw=0.3)
        ax[2].set_ylabel("음높이")
    for p in probs:
        if p.get("end") is None and p["t"] == 0:
            continue
        if t0 <= p["t"] <= t1:
            for x in ax:
                x.axvline(p["t"], color=KIND_COLOR.get(p["level"], "gray"), lw=1.0, alpha=0.5, zorder=1, ls="--")
            ax[0].text(p["t"], 1.0, p["kind"], fontsize=7, color=KIND_COLOR.get(p["level"], "gray"),
                       rotation=90, va="top")
    ax[-1].set_xlim(t0, t0 + len(seg) / sr)
    ax[-1].set_xlabel("시간(초)")
    fig.tight_layout()
    fig.savefig(path, dpi=80)
    plt.close(fig)


# ---------------------------------------------------------------- 명령
def detect_mode(path, mode):
    if mode != "auto":
        return mode
    low = os.path.basename(path).lower()
    if any(k in low for k in ["vocal", "vox", "voice", "보컬", "목소리", "acapella", "acappella", "sv_", "synthv"]):
        return "vocal"
    return "mix"


def cmd_check(args):
    import librosa
    path = args.file
    mode = detect_mode(path, args.mode)
    out = args.out or (os.path.splitext(os.path.basename(path))[0] + "_소리검사")
    os.makedirs(out, exist_ok=True)

    info = probe(path)
    st, sr = load(path)
    dur = len(st) / sr
    stats = {"파일": os.path.basename(path), "검사 방식": "보컬만" if mode == "vocal" else "완성곡(반주 포함)",
             "길이": fmt_t(dur) + f" ({dur:.1f}초)"}
    stats.update({k: v for k, v in info.items() if v})
    probs = []

    check_levels(st, sr, probs, stats)
    check_silence(st, sr, probs, stats, mode)
    check_clicks(st, sr, probs, stats, mode)

    y = librosa.resample(st.mean(axis=1), orig_sr=sr, target_sr=22050)
    ysr = 22050
    # 목소리만 있는 파일은 멜로디만 보고 조를 잘못 짚기 쉽다 → 반주 파일이 있으면 조는 반주에서 잰다
    # [확인: 10-25 「Alone」 분리 보컬은 내림나장조, 원곡·반주는 내림마장조 — 조에 없는 음 13곳이 헛짚음이었음]
    key_src = args.key_from
    if not key_src and mode == "vocal":
        sib = os.path.join(os.path.dirname(os.path.abspath(path)), "반주.wav")
        if os.path.isfile(sib):
            key_src = sib
    if key_src and os.path.isfile(key_src):
        kst, ksr = load(key_src)
        ky = librosa.resample(kst.mean(axis=1), orig_sr=ksr, target_sr=22050)
        scale = estimate_key_tempo(ky, 22050, stats, probs)
        stats["조(키, 추정)"] += f" (조는 {os.path.basename(key_src)}에서 잼)"
    else:
        scale = estimate_key_tempo(y, ysr, stats, probs)
    pitch = None
    if mode == "vocal":
        pitch = check_pitch(y, ysr, probs, stats, scale)
        check_sibilance_breath(y, ysr, probs, stats, pitch)
    stats.pop("_scale", None)

    order = {LEVEL_MUST: 0, LEVEL_CHECK: 1, LEVEL_INFO: 2}
    probs.sort(key=lambda p: (p["t"] if p.get("end") is not None or p["t"] > 0 else -1, order[p["level"]]))

    # 그림: 전체 + 30초씩 + 문제 확대
    images = []
    p0 = os.path.join(out, "00_전체.png")
    draw_view_fixed(y, ysr, 0, dur, pitch, probs, p0, f"전체 보기 — {os.path.basename(path)}")
    images.append(p0)
    page = 30.0
    if dur > page * 1.2:
        k = 1
        s = 0.0
        while s < dur:
            pp = os.path.join(out, f"{k:02d}_{fmt_t(s).replace(':', '분')}부터.png")
            draw_view_fixed(y, ysr, s, min(dur, s + page), pitch, probs, pp, f"{fmt_t(s)} ~ {fmt_t(min(dur, s + page))}")
            images.append(pp)
            s += page
            k += 1
    zooms = [p for p in probs if p["level"] != LEVEL_INFO and not (p.get("end") is None and p["t"] == 0)]
    zdir = os.path.join(out, "확대")
    os.makedirs(zdir, exist_ok=True)
    seen = []
    for p in zooms:
        if any(abs(p["t"] - s) < 0.8 for s in seen):
            continue
        if len(seen) >= args.max_zoom:
            break
        seen.append(p["t"])
        z0, z1 = max(0, p["t"] - 0.6), min(dur, p["t"] + 0.9)
        zp = os.path.join(zdir, f"{fmt_t(p['t']).replace(':', '분')}_{p['kind'].split('(')[0].replace('·', '')}.png")
        draw_view_fixed(y, ysr, z0, z1, pitch, probs, zp, f"{fmt_t(p['t'])} {p['kind']} 확대")
        images.append(zp)

    # 보고서
    for p in probs:
        p["시각"] = "전체" if (p.get("end") is None and p["t"] == 0) else fmt_t(p["t"])
    must = [p for p in probs if p["level"] == LEVEL_MUST]
    chk = [p for p in probs if p["level"] == LEVEL_CHECK]
    inf = [p for p in probs if p["level"] == LEVEL_INFO]
    lines = [f"# 소리 검사 보고서 — {os.path.basename(path)}", ""]
    lines.append(f"- 꼭 고칠 것 **{len(must)}**곳 / 들어 보고 판단 **{len(chk)}**곳 / 참고 **{len(inf)}**곳")
    if mode == "mix":
        lines.append("- 완성곡으로 검사했습니다. 음정·ㅅ소리·숨소리는 **보컬만 따로 뽑은 파일**로 검사해야 정확합니다.")
    lines += ["", "## 파일 정보", "", "| 항목 | 값 |", "|---|---|"]
    for k, v in stats.items():
        lines.append(f"| {k} | {v} |")
    for title, lst in [("꼭 고칠 것", must), ("들어 보고 판단", chk), ("참고", inf)]:
        lines += ["", f"## {title} ({len(lst)})", ""]
        if not lst:
            lines.append("- 없음")
        for p in lst:
            rng = f" ~ {fmt_t(p['end'])}" if p.get("end") not in (None,) and p["end"] != p["t"] else ""
            lines.append(f"- **{p['시각']}{rng}** {p['kind']} — {p['msg']}")
    lines += ["", "## 그림", ""] + [f"- {os.path.relpath(i, out)}" for i in images]
    lines += ["", "※ 자동 검사는 '의심 가는 곳'을 찾는 것입니다. 시각을 들어 보고 최종 판단하세요."]
    rp = os.path.join(out, "보고서.md")
    with open(rp, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    with open(os.path.join(out, "result.json"), "w", encoding="utf-8") as f:
        json.dump(dict(stats=stats, problems=[{k: v for k, v in p.items()} for p in probs],
                       images=images), f, ensure_ascii=False, indent=1, default=float)
    print(f"[완료] 보고서: {rp}")
    print(f"꼭 고칠 것 {len(must)} / 들어 보고 판단 {len(chk)} / 참고 {len(inf)} / 그림 {len(images)}장")
    for p in (must + chk)[:15]:
        print(f"  {p['시각']} [{p['level']}] {p['kind']}")


def cmd_cut(args):
    import librosa
    st, sr = load(args.file)
    t = parse_t(args.time)
    L = args.len
    t0 = max(0.0, t - L / 2)
    t1 = min(len(st) / sr, t + L / 2)
    out = args.out or "소리조각"
    os.makedirs(out, exist_ok=True)
    tag = fmt_t(t).replace(":", "분")
    import soundfile as sf
    wp = os.path.join(out, f"조각_{tag}.wav")
    sf.write(wp, st[int(t0 * sr): int(t1 * sr)].astype(np.float32), sr)
    y = librosa.resample(st.mean(axis=1), orig_sr=sr, target_sr=22050)
    pitch = None
    if detect_mode(args.file, args.mode) == "vocal":
        seg = y[int(t0 * 22050): int(t1 * 22050)]
        f0, vflag, _ = librosa.pyin(seg, fmin=65.0, fmax=1400.0, sr=22050, frame_length=2048, hop_length=128)
        pt = librosa.times_like(f0, sr=22050, hop_length=128) + t0
        pm = np.where(vflag, 69 + 12 * np.log2(np.where(np.isfinite(f0), f0, 1.0) / 440.0), np.nan)
        pitch = dict(t=pt, midi=pm)
    ip = os.path.join(out, f"확대_{tag}.png")
    draw_view_fixed(y, 22050, t0, t1, pitch, [], ip, f"{fmt_t(t0)} ~ {fmt_t(t1)} 확대")
    print(f"[완료] 소리 조각: {wp}\n       확대 그림: {ip}")


def summarize(path):
    import librosa
    import pyloudnorm as pyln
    st, sr = load(path)
    y = librosa.resample(st.mean(axis=1), orig_sr=sr, target_sr=22050)
    lufs = pyln.Meter(sr).integrated_loudness(st)
    rms, _ = frame_rms(y, 22050, 1.0)
    return dict(dur=len(st) / sr, peak=float(db(np.abs(st).max())), lufs=float(lufs), sec_db=db(rms), y=y)


def cmd_compare(args):
    a, b = summarize(args.a), summarize(args.b)
    out = args.out or "소리비교"
    os.makedirs(out, exist_ok=True)
    lines = [f"# 두 판 비교", "", f"| 항목 | 앞판 ({os.path.basename(args.a)}) | 새판 ({os.path.basename(args.b)}) |", "|---|---|---|"]
    lines.append(f"| 길이 | {fmt_t(a['dur'])} | {fmt_t(b['dur'])} |")
    lines.append(f"| 최고 크기 | {a['peak']:.1f} dBFS | {b['peak']:.1f} dBFS |")
    lines.append(f"| 라우드니스 | {a['lufs']:.1f} LUFS | {b['lufs']:.1f} LUFS |")
    n = min(len(a["sec_db"]), len(b["sec_db"]))
    diff = b["sec_db"][:n] - a["sec_db"][:n]
    big = [(i, d) for i, d in enumerate(diff) if abs(d) > 3]
    lines += ["", "## 1초 단위로 크기가 3dB 넘게 달라진 곳", ""]
    lines += [f"- {fmt_t(i)}: {d:+.1f} dB" for i, d in big[:40]] or ["- 없음"]
    plt = setup_plt()
    fig, ax = plt.subplots(1, 1, figsize=(16, 4))
    ax.plot(a["sec_db"][:n], label="앞판")
    ax.plot(b["sec_db"][:n], label="새판")
    ax.set_xlabel("시간(초)")
    ax.set_ylabel("크기(dB)")
    ax.legend()
    ip = os.path.join(out, "크기비교.png")
    fig.tight_layout()
    fig.savefig(ip, dpi=80)
    plt.close(fig)
    lines += ["", f"그림: 크기비교.png"]
    rp = os.path.join(out, "비교.md")
    open(rp, "w", encoding="utf-8").write("\n".join(lines))
    print(f"[완료] {rp}")
    print("\n".join(lines[:8]))


# ---------------------------------------------------------------- 받아쓰기 (말·가사)
def find_whisper_model(explicit=None):
    """ggml whisper 모델 파일을 찾는다: --model, 환경변수 SOUND_EAR_MODEL, 흔한 폴더 순서."""
    import glob
    cands = []
    if explicit:
        cands.append(explicit)
    if os.environ.get("SOUND_EAR_MODEL"):
        cands.append(os.environ["SOUND_EAR_MODEL"])
    pats = ["ggml-*.bin"]
    dirs = [os.getcwd(), os.path.expanduser("~/.cache/sound-ear"), os.path.expanduser("~/.cache/whisper"),
            "/mnt/user-data/uploads", "/mnt/user-data/uploads/Downloads", os.path.expanduser("~/Downloads")]
    for d in dirs:
        for pat in pats:
            cands += sorted(glob.glob(os.path.join(d, pat)), key=lambda f: -os.path.getsize(f))
    for c in cands:
        if c and os.path.isfile(c):
            return c
    return None


def transcribe(path, model_path, lang="auto", threads=None, prompt=None):
    """whisper.cpp로 받아쓰기. 16kHz 모노로 풀어서 넣는다. [(시작초, 끝초, 글)]"""
    from pywhispercpp.model import Model
    r = subprocess.run(["ffmpeg", "-v", "error", "-i", path, "-vn", "-f", "f32le", "-ac", "1", "-ar", "16000", "-"],
                       capture_output=True)
    y = np.frombuffer(r.stdout, np.float32)
    params = dict(language=lang if lang != "auto" else "", print_progress=False, print_realtime=False,
                  n_threads=threads or max(1, os.cpu_count() or 1), suppress_blank=True, no_context=False)
    if prompt:
        params["initial_prompt"] = prompt
    m = Model(model_path, redirect_whispercpp_logs_to=None, **params)
    segs = m.transcribe(y)
    out = []
    for sg in segs:
        txt = sg.text.strip()
        if not txt:
            continue
        out.append((sg.t0 / 100.0, sg.t1 / 100.0, txt))
    return out


def clean_segments(segs):
    """whisper가 음악에서 자주 지어내는 것(같은 줄 반복, 자막 인사, 음표만)을 걸러 표시한다."""
    import re
    junk = re.compile(r"(구독|좋아요|시청해 주셔서|자막|MBC|KBS|SBS|thank you for watching|subscribe|ご視聴|字幕)", re.I)
    res = []
    prev = None
    rep = 0
    for a, b, t in segs:
        flag = ""
        if junk.search(t):
            flag = "지어낸 말일 수 있음"
        if re.fullmatch(r"[\s♪♫~.\-…()\[\]]*", t) or re.fullmatch(r"\s*[\[(（【].{0,12}[\])）】]\s*", t):
            continue  # 음표만, 또는 [끝] (음악) 같은 괄호 표시
        if prev is not None and t == prev:
            rep += 1
            if rep >= 2:
                flag = flag or "같은 줄 반복(지어낸 말일 수 있음)"
        else:
            rep = 0
        prev = t
        res.append((a, b, t, flag))
    return res


def cmd_words(args):
    model = find_whisper_model(args.model)
    if not model:
        sys.exit("받아쓰기 모델(ggml-*.bin)을 못 찾았습니다. --model 로 경로를 주거나 SOUND_EAR_MODEL 환경변수를 정하세요.\n"
                 "받는 곳: https://huggingface.co/ggerganov/whisper.cpp (예: ggml-small-q5_1.bin)")
    src = args.file
    tmp = None
    if args.start is not None or args.end is not None:
        t0 = parse_t(args.start) if args.start else 0.0
        tmp = os.path.join(os.path.dirname(os.path.abspath(args.out or ".")) or ".", "_cut_for_words.wav")
        cmd = ["ffmpeg", "-v", "error", "-y", "-ss", str(t0)]
        if args.end:
            cmd += ["-to", str(parse_t(args.end))]
        subprocess.run(cmd + ["-i", src, "-ac", "1", "-ar", "16000", tmp], check=True)
        src = tmp
    else:
        t0 = 0.0
    import time
    st = time.time()
    segs = transcribe(src, model, lang=args.lang, prompt=args.prompt)
    took = time.time() - st
    if tmp:
        os.remove(tmp)
    segs = clean_segments([(a + t0, b + t0, t) for a, b, t in segs])
    out = args.out or (os.path.splitext(os.path.basename(args.file))[0] + "_가사")
    os.makedirs(out, exist_ok=True)
    lines = [f"# 받아쓰기 — {os.path.basename(args.file)}", "",
             f"- 모델: {os.path.basename(model)} / 언어: {args.lang} / 걸린 시간: {took:.0f}초",
             "- 노래 가사는 반주·효과 때문에 틀리기 쉽습니다. **보컬만 있는 파일**일수록 정확합니다.",
             "- '지어낸 말일 수 있음' 표시는 음악 구간에서 AI가 흔히 지어내는 말입니다.", ""]
    for a, b, t, flag in segs:
        lines.append(f"- **{fmt_t(a)}~{fmt_t(b)}** {t}" + (f"  _({flag})_" if flag else ""))
    rp = os.path.join(out, "가사.md")
    open(rp, "w", encoding="utf-8").write("\n".join(lines))
    json.dump([dict(start=a, end=b, text=t, flag=f) for a, b, t, f in segs],
              open(os.path.join(out, "words.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    srt = []
    for i, (a, b, t, f) in enumerate(segs, 1):
        def ts(x):
            h = int(x // 3600); m_ = int(x % 3600 // 60); s_ = x % 60
            return f"{h:02d}:{m_:02d}:{int(s_):02d},{int((s_ % 1) * 1000):03d}"
        srt += [str(i), f"{ts(a)} --> {ts(b)}", t, ""]
    open(os.path.join(out, "자막.srt"), "w", encoding="utf-8").write("\n".join(srt))
    print(f"[완료] {rp} (줄 {len(segs)}개, {took:.0f}초)")
    for a, b, t, flag in segs[:40]:
        print(f"  {fmt_t(a)} {t}" + (f"  ({flag})" if flag else ""))


# ---------------------------------------------------------------- 보컬 분리 (목소리 / 반주)
SEP_URL = "https://github.com/TRvlvr/model_repo/releases/download/all_public_uvr_models/Kim_Vocal_2.onnx"


def find_sep_model(explicit=None):
    for c in [explicit, os.environ.get("SOUND_EAR_SEP_MODEL"), os.path.expanduser("~/.cache/sound-ear/Kim_Vocal_2.onnx"),
              os.path.join(os.getcwd(), "Kim_Vocal_2.onnx")]:
        if c and os.path.isfile(c):
            return c
    return None


def separate_vocals(path, model, out_dir, clean=False):
    """UVR MDX-Net(Kim_Vocal_2)으로 목소리와 반주를 가른다. 44.1kHz 스테레오 wav 두 개를 만든다.
    [확인: 10-25 2코어에서 노래 길이와 비슷한 시간(30초 곡 32초), clean은 두 배]"""
    import librosa
    import onnxruntime as ort
    import soundfile as sf
    N_FFT, HOP, DIM_F, DIM_T = 7680, 1024, 3072, 256
    CHUNK = HOP * (DIM_T - 1)
    TRIM = N_FFT // 2
    GEN = CHUNK - 2 * TRIM
    r = subprocess.run(["ffmpeg", "-v", "error", "-i", path, "-vn", "-f", "f32le", "-ac", "2", "-ar", "44100", "-"], capture_output=True)
    mix = np.frombuffer(r.stdout, np.float32).reshape(-1, 2).T.astype(np.float32)
    n = mix.shape[1]
    pad = GEN - n % GEN
    mp = np.concatenate([np.zeros((2, TRIM), np.float32), mix, np.zeros((2, pad), np.float32), np.zeros((2, TRIM), np.float32)], 1)
    so = ort.SessionOptions()
    so.intra_op_num_threads = max(1, os.cpu_count() or 1)
    sess = ort.InferenceSession(model, sess_options=so, providers=["CPUExecutionProvider"])
    nb = N_FFT // 2 + 1

    def stft2(w):
        S = [librosa.stft(w[c], n_fft=N_FFT, hop_length=HOP, window="hann", center=True, pad_mode="reflect") for c in range(2)]
        return np.stack([S[0].real, S[0].imag, S[1].real, S[1].imag])[:, :DIM_F, :][None].astype(np.float32)

    def istft2(x):
        full = np.zeros((4, nb, DIM_T), np.float32)
        full[:, :DIM_F] = x[0]
        return np.stack([librosa.istft(full[2 * c] + 1j * full[2 * c + 1], n_fft=N_FFT, hop_length=HOP, window="hann",
                                       center=True, length=CHUNK) for c in range(2)])
    outs = []
    total = (n + pad) // GEN
    for k, i in enumerate(range(0, n + pad, GEN)):
        spec = stft2(mp[:, i:i + CHUNK])
        o = sess.run(None, {"input": spec})[0]
        if clean:  # 부호를 뒤집어 한 번 더 돌려 평균 — 잡음이 조금 줄고 두 배 느림
            o = (o - sess.run(None, {"input": -spec})[0]) / 2
        outs.append(istft2(o)[:, TRIM:-TRIM])
        if (k + 1) % 10 == 0:
            print(f"  보컬 분리 {k + 1}/{total}")
    voc = np.concatenate(outs, 1)[:, :n] * 1.009
    inst = mix - voc
    os.makedirs(out_dir, exist_ok=True)
    vp, ip = os.path.join(out_dir, "보컬.wav"), os.path.join(out_dir, "반주.wav")
    sf.write(vp, voc.T, 44100, subtype="PCM_16")
    sf.write(ip, inst.T, 44100, subtype="PCM_16")
    return vp, ip


def cmd_separate(args):
    model = find_sep_model(args.model)
    if not model:
        sys.exit(f"보컬 분리 모델(Kim_Vocal_2.onnx)이 없습니다. ~/.cache/sound-ear/ 에 두거나 --model 로 주세요.\n받는 곳: {SEP_URL}")
    out = args.out or (os.path.splitext(os.path.basename(args.file))[0] + "_분리")
    import time
    t0 = time.time()
    vp, ip = separate_vocals(args.file, model, out, clean=args.clean)
    print(f"[완료] {vp}\n       {ip}\n       걸린 시간 {time.time() - t0:.0f}초")

# ---------------------------------------------------------------- 음악 AI (Essentia: 악기·장르·분위기)
INST_KO = {"accordion": "아코디언", "acousticbassguitar": "어쿠스틱 베이스", "acousticguitar": "어쿠스틱 기타", "bass": "베이스",
           "beat": "비트", "bell": "종", "bongo": "봉고", "brass": "금관악기", "cello": "첼로", "clarinet": "클라리넷",
           "classicalguitar": "클래식 기타", "computer": "전자음", "doublebass": "콘트라베이스", "drummachine": "드럼 머신",
           "drums": "드럼", "electricguitar": "일렉 기타", "electricpiano": "전자 피아노", "flute": "플루트", "guitar": "기타",
           "harmonica": "하모니카", "harp": "하프", "horn": "호른", "keyboard": "키보드", "oboe": "오보에", "orchestra": "오케스트라",
           "organ": "오르간", "pad": "신스 패드", "percussion": "타악기", "piano": "피아노", "pipeorgan": "파이프 오르간",
           "rhodes": "로즈 피아노", "sampler": "샘플러", "saxophone": "색소폰", "strings": "현악기", "synthesizer": "신시사이저",
           "trombone": "트롬본", "trumpet": "트럼펫", "viola": "비올라", "violin": "바이올린", "voice": "목소리"}
MOOD_KO = {"action": "액션", "adventure": "모험", "advertising": "광고 같은", "background": "배경음악 같은", "ballad": "발라드",
           "calm": "차분한", "children": "아이 같은", "christmas": "크리스마스", "commercial": "상업적인", "cool": "멋진",
           "corporate": "기업 홍보 같은", "dark": "어두운", "deep": "깊은", "documentary": "다큐 같은", "drama": "드라마",
           "dramatic": "극적인", "dream": "꿈같은", "emotional": "감성적인", "energetic": "에너지 넘치는", "epic": "웅장한",
           "fast": "빠른", "film": "영화 같은", "fun": "재미있는", "funny": "익살스러운", "game": "게임 음악 같은",
           "groovy": "그루브 있는", "happy": "행복한", "heavy": "묵직한", "holiday": "휴일 같은", "hopeful": "희망찬",
           "inspiring": "영감을 주는", "love": "사랑 노래", "meditative": "명상적인", "melancholic": "쓸쓸한", "melodic": "멜로디가 좋은",
           "motivational": "힘을 주는", "movie": "영화 음악", "nature": "자연 같은", "party": "파티", "positive": "긍정적인",
           "powerful": "힘찬", "relaxing": "편안한", "retro": "레트로", "romantic": "로맨틱", "sad": "슬픈", "sexy": "섹시한",
           "slow": "느린", "soft": "부드러운", "soundscape": "소리 풍경", "space": "우주 같은", "sport": "스포츠",
           "summer": "여름", "trailer": "예고편 같은", "travel": "여행", "upbeat": "경쾌한", "uplifting": "기분이 올라가는"}
GENRE_TOP_KO = {"Pop": "팝", "Electronic": "일렉트로닉", "Rock": "록", "Hip Hop": "힙합", "Jazz": "재즈", "Classical": "클래식",
                "Folk, World, & Country": "포크·월드·컨트리", "Funk / Soul": "펑크·소울", "Reggae": "레게", "Latin": "라틴",
                "Blues": "블루스", "Stage & Screen": "영화·뮤지컬", "Children's": "어린이", "Non-Music": "음악 아님", "Brass & Military": "관악·군악"}


def find_essentia_dir(explicit=None):
    import glob
    for d in [explicit, os.environ.get("SOUND_EAR_ESSENTIA"), os.path.expanduser("~/.cache/sound-ear/essentia"), os.getcwd()]:
        if d and glob.glob(os.path.join(d, "discogs-effnet-bs64-1.pb")):
            return d
    return None


def music_ai(path, mdir):
    """Essentia 음악 AI로 1초쯤마다 악기·분위기·장르 확률을 낸다. [10-24 4분 곡 6초]"""
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
    import essentia
    essentia.log.infoActive = False
    essentia.log.warningActive = False
    import essentia.standard as es
    r = subprocess.run(["ffmpeg", "-v", "error", "-i", path, "-vn", "-f", "f32le", "-ac", "1", "-ar", "16000", "-"], capture_output=True)
    a = np.frombuffer(r.stdout, np.float32).copy()
    emb = es.TensorflowPredictEffnetDiscogs(graphFilename=os.path.join(mdir, "discogs-effnet-bs64-1.pb"), output="PartitionedCall:1")(a)
    out = dict(n=emb.shape[0], dur=len(a) / 16000.0)
    for key, name, inp, outp in [("inst", "mtg_jamendo_instrument", "model/Placeholder", "model/Sigmoid"),
                                 ("mood", "mtg_jamendo_moodtheme", "model/Placeholder", "model/Sigmoid"),
                                 ("genre", "genre_discogs400", "serving_default_model_Placeholder", "PartitionedCall:0")]:
        pb = os.path.join(mdir, f"{name}-discogs-effnet-1.pb")
        js = os.path.join(mdir, f"{name}-discogs-effnet-1.json")
        if not (os.path.isfile(pb) and os.path.isfile(js)):
            continue
        cls = json.load(open(js, encoding="utf-8"))["classes"]
        out[key] = (es.TensorflowPredict2D(graphFilename=pb, input=inp, output=outp)(emb), cls)
    # 임베딩 하나는 약 3초 창을 1초쯤 간격으로 본다 → 시각
    out["t"] = np.linspace(0, max(0.0, out["dur"] - 3.0), out["n"]) + 1.5
    return out


def ai_top(ai, key, mask=None, k=3, floor=0.12, drop=()):
    if key not in ai:
        return []
    P, cls = ai[key]
    if mask is not None:
        if not mask.any():
            return []
        P = P[mask]
    m = P.mean(axis=0)
    idx = [i for i in np.argsort(m)[::-1] if cls[i] not in drop][:k]
    return [(cls[i], float(m[i])) for i in idx if m[i] >= floor]


def genre_ko(g):
    top, _, sub = g.partition("---")
    return f"{GENRE_TOP_KO.get(top, top)}·{sub}" if sub else GENRE_TOP_KO.get(top, top)

# ---------------------------------------------------------------- 곡 설명 (구성·악기·분위기)
def _feat_frames(y, sr, hop):
    import librosa
    S = np.abs(librosa.stft(y, n_fft=2048, hop_length=hop))
    freqs = librosa.fft_frequencies(sr=sr, n_fft=2048)
    H, P = librosa.decompose.hpss(S, margin=2.0)
    tot = (S ** 2).sum(axis=0) + 1e-12
    perc = (P ** 2).sum(axis=0) / tot
    low = (S[freqs < 150] ** 2).sum(axis=0) / tot
    mid = (S[(freqs >= 200) & (freqs < 3000)] ** 2).sum(axis=0) / tot
    cent = librosa.feature.spectral_centroid(S=S, sr=sr)[0]
    rms = librosa.feature.rms(S=S, frame_length=2048)[0]
    onset = librosa.onset.onset_strength(S=librosa.amplitude_to_db(P, ref=np.max), sr=sr)
    return dict(S=S, H=H, perc=perc, low=low, mid=mid, cent=cent, rms=rms, onset=onset[: len(rms)])


def _segments(y, sr, hop, dur):
    """곡을 덩어리로 나누고(소리가 바뀌는 곳 찾기), 닮은 덩어리끼리 같은 글자(A, B, C…)를 붙인다."""
    import librosa
    from scipy.ndimage import median_filter
    from scipy.signal import find_peaks
    tempo, beats = librosa.beat.beat_track(y=y, sr=sr, hop_length=hop)
    tempo = float(np.atleast_1d(tempo)[0])
    if len(beats) < 32:
        beats = np.arange(0, int(dur * sr / hop), int(0.5 * sr / hop))
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr, hop_length=hop)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20, hop_length=hop)[1:]
    rms = librosa.amplitude_to_db(librosa.feature.rms(y=y, hop_length=hop), ref=np.max)
    C = librosa.util.sync(chroma, beats, aggregate=np.median)
    M = librosa.util.sync(mfcc, beats, aggregate=np.mean)
    E = librosa.util.sync(rms, beats, aggregate=np.mean)
    bt = librosa.frames_to_time(np.concatenate([[0], beats]), sr=sr, hop_length=hop)
    n = C.shape[1]
    beat_s = dur / max(n, 1)

    def zs(X):
        return (X - X.mean(axis=1, keepdims=True)) / (X.std(axis=1, keepdims=True) + 1e-9)
    F = np.vstack([zs(C), zs(M) * 0.8, zs(E) * 1.5])
    Fn = F / (np.linalg.norm(F, axis=0, keepdims=True) + 1e-9)
    SSM = Fn.T @ Fn
    # Foote 체커보드 커널로 '여기서 소리가 바뀐다' 곡선
    L = int(np.clip(round(6.0 / beat_s), 4, 24))
    g = np.outer(np.hanning(2 * L), np.hanning(2 * L))
    K = g * np.kron(np.array([[1, -1], [-1, 1]]), np.ones((L, L)))
    pad = np.pad(SSM, L, mode="edge")
    nov = np.array([np.sum(pad[i:i + 2 * L, i:i + 2 * L] * K) for i in range(n)])
    nov = np.maximum(nov, 0)
    nov = nov / (nov.max() + 1e-9)
    want = int(np.clip(round(dur / 16.0), 4, 20))
    dist = max(1, int(round(7.0 / beat_s)))
    pk, pr = find_peaks(nov, distance=dist, prominence=0.04)
    if len(pk) > want - 1:
        pk = np.sort(pk[np.argsort(pr["prominences"])[::-1][: want - 1]])
    bnd = [0] + [int(p) for p in pk] + [n]
    segs = []
    for i in range(len(bnd) - 1):
        a, b = bnd[i], bnd[i + 1]
        if b - a < 1:
            continue
        ta = 0.0 if a == 0 else bt[min(a, len(bt) - 1)]
        tb = dur if b >= n else bt[min(b, len(bt) - 1)]
        segs.append([ta, tb, a, b])
    # 닮은꼴: 반복 찾기 행렬(시간 지연을 넣은 chroma의 이웃 관계)에서 두 덩어리 사이 '대각선 줄'이 길게 이어지면 같은 반복
    Cs = librosa.feature.stack_memory(C, n_steps=4)
    R = librosa.segment.recurrence_matrix(Cs, width=3, mode="affinity", sym=True)
    R = librosa.segment.path_enhance(R, 9) if n > 20 else R
    m = len(segs)
    sim = np.eye(m)
    for i in range(m):
        for j in range(i + 1, m):
            a1, b1, a2, b2 = segs[i][2], segs[i][3], segs[j][2], segs[j][3]
            B = R[a1:b1, a2:b2]
            if B.size == 0:
                continue
            need = max(2, int(0.5 * min(b1 - a1, b2 - a2)))
            best = 0.0
            for k in range(-(B.shape[0] - 1), B.shape[1]):
                d = np.diagonal(B, offset=k)
                if len(d) >= need:
                    best = max(best, float(np.mean(d)))
            sim[i, j] = sim[j, i] = best
    segs_sim = sim
    off = sim[np.triu_indices(m, 1)] if m > 1 else np.array([0.0])
    thr = max(0.25, float(np.percentile(off, 85))) if m > 2 else 1.0
    labels = [-1] * m
    nxt = 0
    for i in range(m):
        if labels[i] >= 0:
            continue
        labels[i] = nxt
        for j in range(i + 1, m):
            if labels[j] < 0 and sim[i, j] >= thr:
                labels[j] = nxt
        nxt += 1
    return [(sg[0], sg[1], chr(ord("A") + min(l, 25))) for sg, l in zip(segs, labels)], tempo, segs_sim


def _mood(stats_key, tempo, energy_db, cent_hz):
    minor = "마이너" in (stats_key or "")
    color = "어둡고 진지한 느낌" if minor else "밝은 느낌"
    if tempo >= 140:
        sp = "빠르게 몰아치는"
    elif tempo >= 110:
        sp = "경쾌하게 달리는"
    elif tempo >= 85:
        sp = "보통 빠르기의"
    else:
        sp = "느리고 차분한"
    pw = "세고 꽉 찬" if energy_db > -8 else ("힘 있는" if energy_db > -14 else "잔잔한")
    tone = ""
    if cent_hz > 3000:
        tone = ", 높은 소리가 많아 반짝임"
    elif cent_hz < 1500:
        tone = ", 낮은 소리가 많아 묵직함"
    return f"{color}의 {sp} 곡, 소리는 {pw} 편{tone}"


def cmd_describe(args):
    """곡을 '사람이 듣듯이' 설명한다: 구성(전주·절·후렴…), 악기 있는지, 세기 변화, 분위기, 가사."""
    import librosa
    path = args.file
    out = args.out or (os.path.splitext(os.path.basename(path))[0] + "_곡설명")
    os.makedirs(out, exist_ok=True)
    st, sr0 = load(path)
    sr = 22050
    y = librosa.resample(st.mean(axis=1), orig_sr=sr0, target_sr=sr)
    dur = len(y) / sr
    hop = 512
    stats = {}
    estimate_key_tempo(y, sr, stats, [])
    stats.pop("_scale", None)
    f = _feat_frames(y, sr, hop)
    ft = librosa.frames_to_time(np.arange(len(f["rms"])), sr=sr, hop_length=hop)
    rms_db = db(f["rms"])
    song_db = float(np.percentile(rms_db, 90))

    # 보컬 분리(있으면): 반주가 빽빽해도 가사·노래 위치를 정확하게 [확인: 10-25 「Alone」 30초에서 가사·시작 시각이 더 정확해짐]
    vocal_path = None
    if not args.no_separate:
        sm_ = find_sep_model(args.sep_model)
        if sm_:
            cached = os.path.join(out, "보컬.wav")
            if os.path.isfile(cached) and os.path.getmtime(cached) >= os.path.getmtime(path):
                vocal_path = cached
            else:
                print("목소리와 반주를 가르는 중… (노래 길이만큼 걸림)")
                try:
                    vocal_path, _ = separate_vocals(path, sm_, out)
                except Exception as e:  # noqa
                    print("보컬 분리를 못 했습니다:", e)
    # 가사(말) — 모델이 있으면 받아쓰기, 이미 한 결과(words.json)가 있으면 그걸 씀
    words = []
    wj = args.words
    if wj and os.path.isfile(wj):
        words = [(w["start"], w["end"], w["text"], w.get("flag", "")) for w in json.load(open(wj, encoding="utf-8"))]
    elif not args.no_words:
        model = find_whisper_model(args.model)
        if model:
            print("가사 받아쓰는 중… (4분 곡에 약 2분)")
            if vocal_path:
                # 분리한 목소리로 받아쓰고, 빈 곳은 원곡 받아쓰기로 메운다
                # [확인: 10-25 「Alone」 — 목소리만 넣으면 대부분 더 정확했지만 첫 절(0:14~0:45)을 통째로 '[끝]'으로 놓침]
                ws = clean_segments(transcribe(vocal_path, model, lang=args.lang))
                print("빈 곳을 메우려고 원곡도 받아쓰는 중…")
                wm = clean_segments(transcribe(path, model, lang=args.lang))
                words = list(ws)
                for a, b, t, fl in wm:
                    if not any(min(b, b2) - max(a, a2) > 0.3 * (b - a) for a2, b2, _, _ in ws):
                        words.append((a, b, t, fl or "원곡에서 받아씀"))
                words.sort(key=lambda w: w[0])
            else:
                words = clean_segments(transcribe(path, model, lang=args.lang))
            json.dump([dict(start=a, end=b, text=t, flag=fl) for a, b, t, fl in words],
                      open(os.path.join(out, "words.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    voc = np.zeros(len(ft), bool)
    def ok(fl):
        return (not fl) or fl == "원곡에서 받아씀"
    for a, b, t, fl in words:
        if ok(fl):
            voc |= (ft >= a) & (ft < b)
    have_words = bool(words)
    # 분리된 보컬이 있으면 '노래 있는 곳'을 목소리 세기로 다시 정한다(받아쓰기 빈틈 보완)
    if vocal_path:
        vy, _ = librosa.load(vocal_path, sr=sr, mono=True)
        vr = librosa.feature.rms(y=vy, hop_length=hop)[0]
        vr = np.pad(vr, (0, max(0, len(ft) - len(vr))))[: len(ft)]
        vdb = db(np.convolve(vr, np.ones(20) / 20, mode="same"))
        vstem = vdb > float(np.percentile(vdb, 90)) - 14  # 분리가 덜 된 반주(새는 소리)를 노래로 보지 않게
        voc = (voc | vstem) if have_words else vstem
        have_words = True if not have_words else have_words

    segs, tempo, ssim = _segments(y, sr, hop, dur)
    mg = []
    for sgm in segs:  # 이웃한 같은 글자 덩어리 중 짧은 것(10초 미만)만 앞에 붙임 — 긴 반복은 따로 보여 준다
        if mg and mg[-1][2] == sgm[2] and (sgm[1] - sgm[0] < 10 or mg[-1][1] - mg[-1][0] < 10):
            mg[-1] = (mg[-1][0], sgm[1], sgm[2])
        else:
            mg.append(sgm)
    segs = mg
    rows = []
    for (a, b, lab) in segs:
        m = (ft >= a) & (ft < b)
        if not m.any():
            continue
        e = float(np.percentile(rms_db[m], 75))  # 숨·쉼표 같은 빈틈에 끌려 내려가지 않게 위쪽 값으로
        perc = float(np.mean(f["perc"][m]))
        low = float(np.mean(f["low"][m]))
        ons = float(np.mean(f["onset"][m])) if len(f["onset"]) == len(m) else 0.0
        vfrac = float(np.mean(voc[m])) if have_words else None
        txt = [t for (wa, wb, t, fl) in words if ok(fl) and a <= (wa + wb) / 2 < b]
        rows.append(dict(a=a, b=b, lab=lab, e=e, perc=perc, low=low, ons=ons, v=vfrac, words=" / ".join(txt)))
    if not rows:
        sys.exit("구간을 나누지 못했습니다(소리가 너무 짧음).")

    # 악기 있는지(상대 비교)
    # 드럼: 때리는 소리만 뽑아(HPSS) 그 '때림'이 박자마다 규칙적으로 반복되는지(자기상관)로 본다.
    # [확인: 10-24 세기 비율만 쓰니 전부 '없음' → 규칙성으로 바꾸자 보컬만 있는 곡 0.25~0.28, 드럼 있는 곡 0.5~0.9로 갈림]
    yh, yp = librosa.effects.hpss(y, margin=3.0)
    env = librosa.onset.onset_strength(y=yp, sr=sr, hop_length=hop)
    lag = int(round(60.0 / max(tempo, 40) * sr / hop))
    prms = librosa.feature.rms(y=yp, hop_length=hop)[0]
    hrms = librosa.feature.rms(y=yh, hop_length=hop)[0] + 1e-9
    et = librosa.frames_to_time(np.arange(len(env)), sr=sr, hop_length=hop)
    low_top = float(np.percentile(f["low"], 90))
    for r in rows:
        mm = (et >= r["a"]) & (et < r["b"])
        e = env[mm] - env[mm].mean() if mm.sum() > 4 * lag else None
        reg = 0.0
        if e is not None and np.any(e):
            ac = np.correlate(e, e, "full")[len(e) - 1:]
            ac = ac / (ac[0] + 1e-9)
            cands = [ac[max(1, lag // 2 - 2): lag // 2 + 3].max() if lag // 2 + 3 < len(ac) else 0,
                     ac[max(1, lag - 2): lag + 3].max() if lag + 3 < len(ac) else 0]
            reg = float(max(cands))
        ph = float(20 * np.log10(prms[mm].mean() / hrms[mm].mean())) if mm.any() else -99
        r["reg"] = reg
        r["drums"] = reg >= 0.45 and ph > -26
        r["bass"] = r["low"] > max(0.05, 0.45 * low_top)

    # 음악 AI(있으면): 곡 전체와 구간별 악기, 분위기, 장르
    ai = None
    if not args.no_ai:
        mdir = find_essentia_dir(args.essentia)
        if mdir:
            try:
                ai = music_ai(path, mdir)
            except Exception as e:  # noqa
                print("음악 AI를 못 돌렸습니다:", e)
    any_drums = any(r["drums"] for r in rows)
    # 우리 드럼 검사(박자 규칙성)가 곡 전체에 드럼이 없다고 하면 AI의 드럼류는 뺀다
    # [확인: 10-24 보컬만 있는 Bad Apple에서 AI가 드럼 0.40이라고 함 — 반주 있는 곡으로 배운 AI라 그럼]
    drumish = ("drums", "drummachine", "beat", "percussion", "bongo")
    drop = drumish if not any_drums else ()
    vocal_only = ai is not None and not any_drums and all(not r["bass"] for r in rows)
    if ai is not None:
        for r in rows:
            mk = (ai["t"] >= r["a"]) & (ai["t"] < r["b"])
            r["inst"] = [] if vocal_only else ai_top(ai, "inst", mk, k=4, floor=0.15, drop=drop + ("voice",))
    # 이름 붙이기 — 노래 없는 곳은 전주·간주·후주, 노래 있는 곳은 글자+몇 번째 반복.
    # '후렴'은 반복되면서 다른 노래 부분보다 확실히(1.5dB 넘게) 셀 때만 붙인다
    # [확인: 10-24 「마법이 사라진 세상에서」는 같은 멜로디를 여러 번 부르는 곡이라 '가장 많이 반복'을 후렴이라 부르면 틀림]
    from collections import defaultdict
    def sings(r):
        return (r["v"] > 0.25) if r["v"] is not None else (r["e"] > song_db - 10)
    by = defaultdict(list)
    for r in rows:
        if sings(r):
            by[r["lab"]].append(r)
    chorus = None

    def bigr(t):
        t = "".join(ch for ch in t.lower() if ch.isalnum())
        return {t[i:i + 2] for i in range(len(t) - 1)}
    # 1순위: 같은 멜로디(같은 글자)가 '같은 가사'로 반복되면 후렴. 멜로디는 같고 가사가 다르면 절(벌스).
    # [확인: 10-24 Gumi 「Pray To Me」의 'Now pray to me' 부분은 세기만으로는 못 찾았는데 가사 반복으로 찾힘]
    if have_words:
        best = None
        for lab, rs in by.items():
            if len(rs) < 2:
                continue
            sims = []
            for i in range(len(rs)):
                for j in range(i + 1, len(rs)):
                    A, B = bigr(rs[i]["words"]), bigr(rs[j]["words"])
                    if A and B:
                        sims.append(len(A & B) / len(A | B))
            if sims and max(sims) >= 0.35:
                sc = np.mean([x["e"] for x in rs]) + 3 * max(sims)
                if best is None or sc > best[0]:
                    best = (sc, lab)
        if best:
            chorus = best[1]
    if chorus is None and len(by) >= 2:
        means = {lab: np.mean([x["e"] for x in rs]) for lab, rs in by.items()}
        rep = [lab for lab, rs in by.items() if len(rs) >= 2]
        if rep:
            top = max(rep, key=lambda l: means[l])
            others = [means[l] for l in means if l != top]
            if others and means[top] >= max(others) - 0.5 and means[top] >= np.mean(others) + 1.5:
                chorus = top
    count = defaultdict(int)
    total = {lab: sum(1 for r in rows if r["lab"] == lab) for lab in set(r["lab"] for r in rows)}
    names = []
    for i, r in enumerate(rows):
        count[r["lab"]] += 1
        k = count[r["lab"]]
        if not sings(r):
            before_first = all(not sings(x) for x in rows[:i])
            nm = "전주" if before_first else ("후주(끝)" if i == len(rows) - 1 else "간주(노래 없음)")
        elif r["lab"] == chorus:
            nm = f"후렴 {k}번째" if total[r["lab"]] > 1 else "후렴"
        elif r["e"] < song_db - 9:
            nm = f"조용한 노래 부분 {r['lab']}"
        else:
            nm = f"노래 {r['lab']}" + (f" ({k}번째 반복)" if total[r["lab"]] > 1 and k > 1 else "")
        names.append(nm)

    try:
        import pyloudnorm as pyln
        lufs = float(pyln.Meter(sr0).integrated_loudness(st))
    except Exception:
        lufs = None
    # 세기는 사람이 듣는 크기(LUFS)로 판단 [확인: 10-25 꽉 찬 팝 곡이 '잔잔한 편'으로 나옴 — RMS dB 기준이 틀렸음]
    pw_db = (lufs + 2.0) if lufs is not None and np.isfinite(lufs) else song_db
    mood = _mood(stats.get("조(키, 추정)"), tempo, pw_db, float(np.median(f["cent"])))
    lines = [f"# 곡 설명 — {os.path.basename(path)}", ""]
    lines.append(f"**한 줄 느낌(추정):** {mood}")
    lines.append(f"- 길이 {fmt_t(dur)} / 조 {stats.get('조(키, 추정)', '?')} / 빠르기 약 {tempo:.0f} BPM (반이나 두 배일 수 있음)")
    if not have_words:
        lines.append("- 가사 받아쓰기 모델이 없어 노래가 있는지는 소리 세기로만 짐작했습니다.")
    if ai is not None:
        gi = ai_top(ai, "genre", k=3, floor=0.05)
        mi = ai_top(ai, "mood", k=4, floor=0.05)
        ii = [] if vocal_only else ai_top(ai, "inst", k=6, floor=0.12, drop=drop + ("voice",))
        lines.append(f"- **음악 AI가 들은 장르:** " + (", ".join(f"{genre_ko(g)} ({p:.0%})" for g, p in gi) or "잘 모르겠음"))
        lines.append(f"- **음악 AI가 들은 분위기:** " + (", ".join(f"{MOOD_KO.get(m_, m_)} ({p:.0%})" for m_, p in mi) or "뚜렷하지 않음"))
        if vocal_only:
            lines.append("- **악기:** 반주 없이 목소리만 있는 파일로 보여 악기 이름은 적지 않았습니다.")
        else:
            lines.append(f"- **음악 AI가 들은 악기(곡 전체):** " + (", ".join(f"{INST_KO.get(i_, i_)} ({p:.0%})" for i_, p in ii) or "뚜렷하지 않음"))
        lines.append("- 퍼센트는 AI가 '들린다'고 믿는 정도입니다. 30% 넘으면 꽤 확실, 15% 안팎은 '아마'.")
    show_inst = ai is not None and not vocal_only
    if show_inst:
        lines += ["", "## 흐름", "", "| 시각 | 구간(추정) | 닮은꼴 | 세기 | 드럼 | 저음 | 노래 | 들리는 악기(AI) | 가사 |", "|---|---|---|---|---|---|---|---|---|"]
    else:
        lines += ["", "## 흐름", "", "| 시각 | 구간(추정) | 닮은꼴 | 세기 | 드럼 | 저음 | 노래 | 가사 |", "|---|---|---|---|---|---|---|---|"]
    prev_inst = set()
    prev_e = None
    story = []
    for r, nm in zip(rows, names):
        bar = "█" * int(np.clip((r["e"] - song_db + 24) / 3, 1, 8))
        v = "-" if r["v"] is None else ("있음" if r["v"] > 0.25 else "없음")
        w = (r["words"][:60] + "…") if len(r["words"]) > 60 else r["words"]
        insts = [INST_KO.get(i_, i_) for i_, p in r.get("inst", [])]
        cells = f"| {fmt_t(r['a'])}~{fmt_t(r['b'])} | {nm} | {r['lab']} | {bar} | {'O' if r['drums'] else '·'} | {'O' if r['bass'] else '·'} | {v} |"
        if show_inst:
            cells += f" {', '.join(insts) or '-'} |"
        lines.append(cells + f" {w} |")
        change = ""
        if prev_e is not None:
            d = r["e"] - prev_e
            change = "확 커지며 " if d > 4 else ("살짝 커지며 " if d > 2 else ("확 작아지며 " if d < -4 else ("살짝 작아지며 " if d < -2 else "")))
        parts = []
        if r["drums"]:
            parts.append("드럼")
        if r["bass"]:
            parts.append("묵직한 저음")
        inst = (" 과 ".join(parts).replace(" 과 ", "과 ") + "이 깔리고") if parts else "드럼 없이 가볍게"
        if show_inst:
            cur = {i_ for i_, p in r.get("inst", [])}
            newin = [INST_KO.get(i_, i_) for i_ in cur - prev_inst if i_ not in drumish and i_ != "bass"]
            if newin and prev_e is not None:
                inst += f", {'·'.join(newin)} 소리가 새로 들림"
            elif insts and prev_e is None:
                inst += f", {'·'.join(insts[:3])} 소리"
            prev_inst = cur
        story.append(f"- **{fmt_t(r['a'])}** {change}{nm} — {inst}" + (f", 노래: “{r['words'][:40]}”" if r["words"] else ""))
        prev_e = r["e"]
    lines += ["", "## 들은 대로 말하면", ""] + story
    reps = sorted({r["lab"] for r in rows if total[r["lab"]] > 1})
    if reps:
        lines.insert(next(i for i, l in enumerate(lines) if l.startswith("- 길이")) + 1, "- 반복되는 부분: " + ", ".join(f"{l}({total[l]}번)" for l in reps) + (f" — 그중 {chorus}가 같은 가사·멜로디로 반복되고 세서 후렴으로 봤습니다" if chorus else " — 다른 부분보다 확 센 반복이 없어 후렴은 정하지 않았습니다"))
    lines += ["", "※ 구간 이름·분위기·악기는 소리 특징으로 짐작한 것입니다. 닮은꼴 글자가 같으면 비슷한 멜로디·화음의 반복입니다.",
              ("※ 드럼·저음 칸은 소리 특징으로, '들리는 악기' 칸은 음악 AI(Essentia, 비상업용 모델)가 들은 것입니다. 악기 AI는 반주가 있는 곡으로 배워서 목소리만 있는 파일에서는 믿기 어렵습니다."
               if ai is not None else "※ 악기는 '드럼(때리는 소리)'과 '저음(베이스·킥)'만 가려냅니다. 악기 이름·장르·분위기 AI(Essentia 모델 파일)가 있으면 더 자세히 들립니다.")]
    if have_words:
        # 반복되는 가사(훅·후렴구): 비슷한 줄이 다른 시각에 또 나오면 묶어서 보여 준다
        good = [(a, b, t) for a, b, t, fl in words if ok(fl) and len(t) >= 6]
        used = set()
        hooks = []
        for i in range(len(good)):
            if i in used:
                continue
            grp = [i]
            for j in range(i + 1, len(good)):
                if j in used:
                    continue
                A, B = bigr(good[i][2]), bigr(good[j][2])
                if A and B and len(A & B) / len(A | B) >= 0.45 and abs(good[j][0] - good[i][0]) > 5:
                    grp.append(j)
            if len(grp) >= 2:
                used.update(grp)
                hooks.append(grp)
        if hooks:
            lines += ["", "## 반복되는 가사 (훅·후렴구 후보)", ""]
            for grp in hooks[:8]:
                t = good[grp[0]][2]
                lines.append(f"- “{t[:50]}” — " + ", ".join(fmt_t(good[k][0]) for k in grp))
        lines += ["", "## 가사(받아쓰기)", ""] + [f"- {fmt_t(a)} {t}" + (f" _({fl})_" if fl else "") for a, b, t, fl in words]
    rp = os.path.join(out, "곡설명.md")
    open(rp, "w", encoding="utf-8").write("\n".join(lines))
    json.dump(dict(mood=mood, tempo=tempo, key=stats.get("조(키, 추정)"),
                   ai=None if ai is None else dict(genre=ai_top(ai, "genre", k=5, floor=0.0), mood=ai_top(ai, "mood", k=6, floor=0.0),
                                                    instruments=[] if vocal_only else ai_top(ai, "inst", k=8, floor=0.0, drop=drop)),
                   sections=[dict(start=r["a"], end=r["b"], name=nm, label=r["lab"], energy_db=r["e"], drums=bool(r["drums"]),
                                  bass=bool(r["bass"]), vocals=r["v"], words=r["words"], instruments=r.get("inst", [])) for r, nm in zip(rows, names)]),
              open(os.path.join(out, "describe.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)

    # 그림: 세기 곡선 + 구간 색 띠 + 노래 있는 곳
    plt = setup_plt()
    fig, ax = plt.subplots(2, 1, figsize=(16, 6), sharex=True, gridspec_kw=dict(height_ratios=[3, 1]))
    sm = np.convolve(rms_db, np.ones(20) / 20, mode="same")
    ax[0].plot(ft, sm, color="black", lw=1)
    cols = plt.get_cmap("tab10")
    for r, nm in zip(rows, names):
        ci = ord(r["lab"]) - ord("A")
        ax[0].axvspan(r["a"], r["b"], color=cols(ci % 10), alpha=0.18)
        ax[0].text((r["a"] + r["b"]) / 2, sm.max() + 1, f"{r['lab']}\n{nm.split('(')[0]}", ha="center", va="bottom", fontsize=8)
        ax[1].axvspan(r["a"], r["b"], ymin=0.55, ymax=0.95, color="tab:red" if r["drums"] else "white", alpha=0.5)
        ax[1].axvspan(r["a"], r["b"], ymin=0.05, ymax=0.45, color="tab:blue" if r["bass"] else "white", alpha=0.5)
    if have_words:
        ax[0].fill_between(ft, sm.min() - 1, sm.min() + 1, where=voc, color="green", alpha=0.6, label="노래")
        ax[0].legend(loc="lower right")
    ax[0].set_ylim(sm.min() - 2, sm.max() + 6)
    ax[0].set_ylabel("세기(dB)")
    ax[0].set_title(f"곡 흐름 — {os.path.basename(path)}  |  {mood}")
    ax[1].set_yticks([0.25, 0.75])
    ax[1].set_yticklabels(["저음", "드럼"])
    ax[1].set_xlabel("시간(초)")
    fig.tight_layout()
    ip = os.path.join(out, "곡흐름.png")
    fig.savefig(ip, dpi=80)
    plt.close(fig)
    print(f"[완료] {rp}\n       그림: {ip}")
    print("\n".join(lines[:4 + 4 + len(rows)]))


def main():
    ap = argparse.ArgumentParser(description="소리 듣기 도구")
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("check")
    c.add_argument("file")
    c.add_argument("--mode", default="auto", choices=["auto", "vocal", "mix"])
    c.add_argument("--out")
    c.add_argument("--max-zoom", type=int, default=12)
    c.add_argument("--key-from", help="조(키)를 잴 다른 파일(반주·원곡). 보컬.wav 옆에 반주.wav가 있으면 자동")
    c.set_defaults(fn=cmd_check)
    k = sub.add_parser("cut")
    k.add_argument("file")
    k.add_argument("time")
    k.add_argument("--len", type=float, default=2.0)
    k.add_argument("--mode", default="auto", choices=["auto", "vocal", "mix"])
    k.add_argument("--out")
    k.set_defaults(fn=cmd_cut)
    w = sub.add_parser("words", help="말·가사 받아쓰기 (whisper.cpp)")
    w.add_argument("file")
    w.add_argument("--model", help="ggml-*.bin 경로")
    w.add_argument("--lang", default="auto", help="ko, ja, en ... (기본: 자동)")
    w.add_argument("--prompt", help="힌트 글(곡 제목, 자주 나오는 단어)")
    w.add_argument("--start")
    w.add_argument("--end")
    w.add_argument("--out")
    w.set_defaults(fn=cmd_words)
    d = sub.add_parser("describe", help="곡 설명: 구성·드럼·저음·세기·분위기·가사")
    d.add_argument("file")
    d.add_argument("--model")
    d.add_argument("--lang", default="auto")
    d.add_argument("--words", help="이미 받아쓴 words.json")
    d.add_argument("--no-words", action="store_true")
    d.add_argument("--essentia", help="Essentia 모델 폴더")
    d.add_argument("--no-ai", action="store_true", help="음악 AI(악기·장르·분위기) 끄기")
    d.add_argument("--no-separate", action="store_true", help="보컬 분리 끄기")
    d.add_argument("--sep-model", help="Kim_Vocal_2.onnx 경로")
    sp = sub.add_parser("separate", help="목소리와 반주 가르기 (보컬.wav, 반주.wav)")
    sp.add_argument("file")
    sp.add_argument("--model")
    sp.add_argument("--clean", action="store_true", help="두 번 돌려 잡음을 조금 더 줄임(두 배 느림)")
    sp.add_argument("--out")
    sp.set_defaults(fn=cmd_separate)
    d.add_argument("--out")
    d.set_defaults(fn=cmd_describe)
    m = sub.add_parser("compare")
    m.add_argument("a")
    m.add_argument("b")
    m.add_argument("--out")
    m.set_defaults(fn=cmd_compare)
    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
