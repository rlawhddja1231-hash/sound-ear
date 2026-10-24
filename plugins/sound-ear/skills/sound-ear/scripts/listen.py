#!/usr/bin/env python3
"""listen.py — 소리 듣기 도구 (Claude가 소리 파일을 '듣는' 도구)

쓰는 법
  python3 listen.py check  노래.wav [--mode vocal|mix|auto] [--out 폴더]
  python3 listen.py cut    노래.wav 1:23.5 [--len 2] [--out 폴더]
  python3 listen.py compare 앞판.wav 새판.wav [--out 폴더]

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
    keep = []
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
        keep.append(i * 0.001)
    times = keep
    groups = group_times(times, 0.03)
    stats["딸깍 소리"] = len(groups)
    for a, b, k in groups[:30]:
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
    v2 = np.interp(t2, t, vflag.astype(float)) > 0.5
    for a, b in runs(v2):
        if b - a < 20:
            continue
        seg = m2[a:b]
        sm = median_filter(seg, size=21, mode="nearest")
        dev = np.abs(seg - sm)
        rr = r2[a:b]
        strong = rr > 0.3 * np.median(rr)  # 소리가 커지는 중·작아지는 중은 빼고 봄
        for ga, gb in runs((dev > 3.0) & strong):
            octave = abs(abs(np.median(seg[ga:gb] - sm[ga:gb])) - 12) < 0.6
            if gb - ga <= int(0.08 / (h2 / sr)) and ga > 4 and gb < len(seg) - 4:
                glitch_t.append((t2[a + ga], octave))
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


def main():
    ap = argparse.ArgumentParser(description="소리 듣기 도구")
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("check")
    c.add_argument("file")
    c.add_argument("--mode", default="auto", choices=["auto", "vocal", "mix"])
    c.add_argument("--out")
    c.add_argument("--max-zoom", type=int, default=12)
    c.set_defaults(fn=cmd_check)
    k = sub.add_parser("cut")
    k.add_argument("file")
    k.add_argument("time")
    k.add_argument("--len", type=float, default=2.0)
    k.add_argument("--mode", default="auto", choices=["auto", "vocal", "mix"])
    k.add_argument("--out")
    k.set_defaults(fn=cmd_cut)
    m = sub.add_parser("compare")
    m.add_argument("a")
    m.add_argument("b")
    m.add_argument("--out")
    m.set_defaults(fn=cmd_compare)
    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
