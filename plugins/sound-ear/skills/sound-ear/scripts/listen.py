#!/usr/bin/env python3
"""listen.py — 소리 듣기 도구 (Claude가 소리 파일을 '듣는' 도구)

쓰는 법
  python3 listen.py check  노래.wav [--mode vocal|mix|auto] [--out 폴더]
  python3 listen.py cut    노래.wav 1:23.5 [--len 2] [--out 폴더]
  python3 listen.py compare 앞판.wav 새판.wav [--out 폴더]
  python3 listen.py describe 노래.wav [--words 가사폴더/words.json] [--out 폴더]
  python3 listen.py full   노래.wav [--lang ko] [--out 폴더]   ← 처음엔 이것 하나면 됨 (종합보고서.md)
  python3 listen.py separate 노래.wav [--out 폴더]   (보컬.wav, 반주.wav)
  python3 listen.py words  노래.wav [--lang ko] [--model ggml-small-q5_1.bin] [--out 폴더]
  python3 listen.py lyrics 노래.wav 가사.txt [--out 폴더]   (내가 쓴 가사와 들린 것 맞춰 보기)
  python3 listen.py origin 노래.mp3 [--out 폴더]            (꼬리표·출처 서명으로 어디서 만든 파일인지)
  python3 listen.py notes  보컬.wav 노래.svp [--out 폴더]    (노트마다 음정·길이·흠 — 몇 번째 노트를 고칠지)
  python3 listen.py full   노래.wav --lyrics 가사.txt      (종합보고서에 가사 대조까지, 보고서.html도 같이)

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

# ---------------------------------------------------------------- ffmpeg 찾기·자동 준비 [v1.8.2]
FF_DIR = os.path.join(os.path.expanduser("~"), ".cache", "sound-ear", "ffmpeg")
FF_URLS = {  # 윈도우·리눅스는 BtbN 정적 빌드(깃허브), 맥은 evermeet
    "win": "https://github.com/BtbN/FFmpeg-Builds/releases/latest/download/ffmpeg-master-latest-win64-gpl.zip",
    "linux-x86_64": "https://github.com/BtbN/FFmpeg-Builds/releases/latest/download/ffmpeg-master-latest-linux64-gpl.tar.xz",
    "linux-aarch64": "https://github.com/BtbN/FFmpeg-Builds/releases/latest/download/ffmpeg-master-latest-linuxarm64-gpl.tar.xz",
    "mac-ffmpeg": "https://evermeet.cx/ffmpeg/getrelease/zip",
    "mac-ffprobe": "https://evermeet.cx/ffmpeg/getrelease/ffprobe/zip",
}


def _use_local_ffmpeg():
    """시스템에 ffmpeg가 없으면 ~/.cache/sound-ear/ffmpeg 에 받아 둔 것을 PATH 앞에 넣는다."""
    import shutil
    if shutil.which("ffmpeg") and shutil.which("ffprobe"):
        return True
    if os.path.isdir(FF_DIR):
        os.environ["PATH"] = FF_DIR + os.pathsep + os.environ.get("PATH", "")
    return bool(shutil.which("ffmpeg") and shutil.which("ffprobe"))


def ensure_ffmpeg(download=True):
    """ffmpeg·ffprobe를 쓸 수 있게 한다. 없으면 운영체제에 맞는 정적 빌드를 받아 ~/.cache/sound-ear/ffmpeg 에 푼다.
    돌려주는 것: (됨, 설명)"""
    import platform
    import shutil
    import stat
    import tarfile
    import zipfile
    if _use_local_ffmpeg():
        return True, "있음 (" + (shutil.which("ffmpeg") or "") + ")"
    if not download:
        return False, "없음"
    os.makedirs(FF_DIR, exist_ok=True)
    sysname = platform.system().lower()
    arch = platform.machine().lower()
    try:
        if sysname.startswith("win"):
            z = os.path.join(FF_DIR, "ffmpeg.zip")
            if not _download(FF_URLS["win"], z, timeout=1200):
                return False, "내려받기 실패 — 인터넷이 막혔으면 " + FF_URLS["win"] + " 를 받아 bin 안의 ffmpeg.exe·ffprobe.exe 를 " + FF_DIR + " 에 두세요"
            with zipfile.ZipFile(z) as zf:
                for m in zf.namelist():
                    base = os.path.basename(m)
                    if base in ("ffmpeg.exe", "ffprobe.exe"):
                        with zf.open(m) as src, open(os.path.join(FF_DIR, base), "wb") as dst:
                            shutil.copyfileobj(src, dst)
            os.remove(z)
        elif sysname == "darwin":
            for name in ("ffmpeg", "ffprobe"):
                z = os.path.join(FF_DIR, name + ".zip")
                if not _download(FF_URLS["mac-" + name], z, timeout=1200):
                    return False, "내려받기 실패 — brew install ffmpeg 를 하거나 evermeet.cx/ffmpeg 에서 받아 " + FF_DIR + " 에 두세요"
                with zipfile.ZipFile(z) as zf:
                    for m in zf.namelist():
                        if os.path.basename(m) == name:
                            with zf.open(m) as src, open(os.path.join(FF_DIR, name), "wb") as dst:
                                shutil.copyfileobj(src, dst)
                os.remove(z)
        else:
            key = "linux-aarch64" if ("arm" in arch or "aarch64" in arch) else "linux-x86_64"
            t = os.path.join(FF_DIR, "ffmpeg.tar.xz")
            if not _download(FF_URLS[key], t, timeout=1200):
                return False, "내려받기 실패 — sudo apt install ffmpeg 를 하거나 " + FF_URLS[key] + " 를 받아 bin 안의 ffmpeg·ffprobe 를 " + FF_DIR + " 에 두세요"
            with tarfile.open(t) as tf:
                for m in tf.getmembers():
                    base = os.path.basename(m.name)
                    if m.isfile() and base in ("ffmpeg", "ffprobe"):
                        with tf.extractfile(m) as src, open(os.path.join(FF_DIR, base), "wb") as dst:
                            shutil.copyfileobj(src, dst)
            os.remove(t)
        for name in ("ffmpeg", "ffprobe"):
            pth = os.path.join(FF_DIR, name)
            if os.path.exists(pth):
                os.chmod(pth, os.stat(pth).st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    except Exception as e:  # noqa
        return False, f"풀다가 실패: {e}"
    ok = _use_local_ffmpeg()
    return ok, ("받아서 준비함 → " + FF_DIR) if ok else "받았는데 실행 파일을 못 찾음: " + FF_DIR


_use_local_ffmpeg()

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


# ---------------------------------------------------------------- 진행 표시
# 긴 작업 중에 "지금 몇 번째 단계, 몇 % , 몇 분 남음"을 화면과 out/진행.txt 에 적는다.
# Claude가 뒤에서 돌려 둔 작업은 진행.txt 를 읽으면 어디까지 왔는지 알 수 있다. [v1.8.0]
_PROG = {"out": None, "t0": None, "steps": [], "i": 0, "est": 0.0, "tick_t": 0.0, "name": ""}


def _fmt_dur(s):
    s = max(0, int(round(s)))
    return f"{s // 60}분 {s % 60:02d}초" if s >= 60 else f"{s}초"


def progress_start(out, steps):
    """steps: [(이름, 예상 초)]"""
    import time
    _PROG.update(out=out, t0=time.time(), steps=list(steps), i=0, est=sum(e for _, e in steps), tick_t=0.0, name="")
    _progress_write("시작")


def _progress_write(line):
    import time
    if not _PROG["out"]:
        return
    try:
        el = time.time() - _PROG["t0"]
        with open(os.path.join(_PROG["out"], "진행.txt"), "w", encoding="utf-8") as f:
            f.write(f"{time.strftime('%H:%M:%S')} 지금까지 {_fmt_dur(el)}\n{line}\n")
    except Exception:
        pass


def progress_step(name, est=None):
    """단계 하나를 시작한다. 지금까지 걸린 시간과 남은 예상 시간을 같이 찍는다."""
    import time
    _PROG["i"] += 1
    _PROG["name"] = name
    _PROG["tick_t"] = 0.0
    n = len(_PROG["steps"]) or 1
    i = _PROG["i"]
    el = time.time() - (_PROG["t0"] or time.time())
    done_est = sum(e for _, e in _PROG["steps"][: i - 1])
    remain = sum(e for _, e in _PROG["steps"][i - 1:]) if _PROG["steps"] else 0
    # 실제 속도가 예상과 다르면 남은 시간도 그 비율로 늘리거나 줄인다
    if done_est > 20 and el > 0:
        remain *= min(4.0, max(0.25, el / done_est))
    msg = f"[{i}/{n}] {name} — 지금까지 {_fmt_dur(el)}" + (f", 남은 시간 약 {_fmt_dur(remain)}" if remain else "")
    print("\n=== " + msg + " ===", flush=True)
    _progress_write(msg)


def progress_tick(frac, label="", every=5.0):
    """단계 안에서 몇 %인지. 같은 줄을 너무 자주 찍지 않게 every초마다만."""
    import time
    now = time.time()
    if now - _PROG["tick_t"] < every and frac < 0.999:
        return
    _PROG["tick_t"] = now
    i, n = _PROG["i"], len(_PROG["steps"]) or 1
    cur_est = _PROG["steps"][i - 1][1] if _PROG["steps"] and 0 < i <= len(_PROG["steps"]) else 0
    left = cur_est * (1 - frac) + sum(e for _, e in _PROG["steps"][i:])
    msg = f"  {_PROG['name']} {frac:.0%}" + (f" ({label})" if label else "") + (f" — 남은 시간 약 {_fmt_dur(left)}" if left else "")
    print(msg, flush=True)
    _progress_write(f"[{i}/{n}] " + msg.strip())


def progress_done():
    import time
    if _PROG["t0"]:
        _progress_write(f"끝 — 모두 {_fmt_dur(time.time() - _PROG['t0'])} 걸림")


def est_times(dur, deep=False, model_big=False):
    """2코어 기준 대략의 예상 시간(초). [확인: 10-25 4분 곡 빠른 검사 3~4분, --deep 10분쯤]"""
    return dict(check_mix=12 + 0.08 * dur, check_vocal=12 + 0.2 * dur, separate=10 + 1.05 * dur,
                words=(1.7 if model_big else 0.5) * dur + 10, describe=8 + 0.17 * dur, lyrics=3, html=3)


# ---------------------------------------------------------------- 가사 대조 (내가 쓴 가사 vs 받아쓴 것)
def _jamo(ch):
    """한글 한 글자를 초성·중성·종성으로 푼다. 한글이 아니면 그대로."""
    o = ord(ch)
    if 0xAC00 <= o <= 0xD7A3:
        o -= 0xAC00
        cho, jung, jong = o // 588, (o % 588) // 28, o % 28
        CHO = "ㄱㄲㄴㄷㄸㄹㅁㅂㅃㅅㅆㅇㅈㅉㅊㅋㅌㅍㅎ"
        JUNG = "ㅏㅐㅑㅒㅓㅔㅕㅖㅗㅘㅙㅚㅛㅜㅝㅞㅟㅠㅡㅢㅣ"
        JONG = " ㄱㄲㄳㄴㄵㄶㄷㄹㄺㄻㄼㄽㄾㄿㅀㅁㅂㅄㅅㅆㅇㅈㅊㅋㅌㅍㅎ"
        return CHO[cho] + JUNG[jung] + (JONG[jong].strip())
    return ch


_KAKASI = None


def _ja_hira(text):
    """일본어 한자·가타카나를 히라가나로(pykakasi가 있을 때만). 글자 수가 달라지므로 [(히라가나 글자, 원래 글자 번호)]로 돌려준다."""
    global _KAKASI
    if _KAKASI is None:
        try:
            import pykakasi
            _KAKASI = pykakasi.kakasi()
        except Exception:
            _KAKASI = False
    if not _KAKASI:
        return None
    out = []
    pos = 0
    for item in _KAKASI.convert(text):
        orig, hira = item["orig"], item["hira"]
        k = text.find(orig, pos)
        k = pos if k < 0 else k
        for ch in hira:
            out.append((ch, k))
        pos = k + len(orig)
    return out


def _norm_chars(text):
    """비교용 글자 목록: 띄어쓰기·문장부호를 빼고, 한글은 자모로 풀고, 영어는 소문자로.
    일본어는 pykakasi가 있으면 한자·가타카나를 히라가나로 바꿔 소리로 견준다. [(비교 글자, 원래 글자 번호)]"""
    import unicodedata
    out = []
    src = list(enumerate(text))
    if any(0x3040 <= ord(c) <= 0x30FF or 0x4E00 <= ord(c) <= 0x9FFF for c in text):
        conv = _ja_hira(text)
        if conv:
            src = [(k, ch) for ch, k in conv]
    for i, ch in src:
        if ch.isspace() or unicodedata.category(ch)[0] in ("P", "S", "Z"):
            continue
        ch2 = ch.lower()
        ch2 = unicodedata.normalize("NFKC", ch2)
        # 일본어 가타카나는 히라가나로 (받아쓰기가 둘을 섞어 씀)
        if 0x30A1 <= ord(ch2) <= 0x30F6:
            ch2 = chr(ord(ch2) - 0x60)
        for j in _jamo(ch2):
            out.append((j, i))
    return out


def _read_lyrics(path):
    """가사 파일을 줄 목록으로. [Verse] 같은 대괄호 줄, 빈 줄, 번호만 있는 줄은 뺀다."""
    import re
    lines = []
    for raw in open(path, encoding="utf-8", errors="ignore").read().splitlines():
        s = raw.strip()
        if not s or re.fullmatch(r"[\[(（【].*[\])）】]", s) or re.fullmatch(r"[\d\s.\-:]+", s):
            continue
        lines.append(s)
    return lines


def compare_lyrics(words, ref_lines, min_line=2):
    """받아쓴 조각들(words: (시작, 끝, 글, 표시))과 내가 쓴 가사 줄을 맞춰 보고, 줄마다 얼마나 맞는지와 틀린 글자를 찾는다.
    돌려주는 것: dict(lines=[{line, score, heard, t0, t1, wrong(원래 가사의 글자 번호들), level}], extra=[(t0,t1,text)], overall)"""
    import difflib
    good = [(a, b, t) for a, b, t, fl in words if not fl or fl == "원곡에서 받아씀"]
    heard_chars = []   # (비교 글자, 조각 번호, 조각 안 글자 번호)
    for k, (a, b, t) in enumerate(good):
        for j, i in _norm_chars(t):
            heard_chars.append((j, k, i))
    ref_chars = []     # (비교 글자, 줄 번호, 줄 안 글자 번호)
    for li, line in enumerate(ref_lines):
        for j, i in _norm_chars(line):
            ref_chars.append((j, li, i))
    A = [c[0] for c in ref_chars]
    B = [c[0] for c in heard_chars]
    sm = difflib.SequenceMatcher(None, A, B, autojunk=False)
    matched_ref = set()          # 맞은 가사 글자 자리
    ref_to_heard = {}            # 가사 글자 자리 → 받아쓴 글자 자리
    matched_heard = set()
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":
            for d in range(i2 - i1):
                matched_ref.add(i1 + d)
                ref_to_heard[i1 + d] = j1 + d
                matched_heard.add(j1 + d)
        elif tag == "replace":
            # 바뀐 덩어리끼리는 시각만 이어 둔다(틀린 자리)
            for d in range(i2 - i1):
                ref_to_heard[i1 + d] = j1 + min(d, max(0, j2 - j1 - 1))
    rows = []
    for li, line in enumerate(ref_lines):
        idx = [n for n, c in enumerate(ref_chars) if c[1] == li]
        if len(idx) < min_line:
            continue
        # 틀린 글자(원래 줄의 글자 번호) — 자모 하나라도 틀리면 그 글자. 점수는 글자(음절) 단위로 센다
        wrong = sorted({ref_chars[n][2] for n in idx if n not in matched_ref})
        n_syl = len({ref_chars[n][2] for n in idx})
        score = 1.0 - len(wrong) / max(1, n_syl)
        # 이 줄이 들린 시각: 맞춰진 받아쓴 글자들의 조각 시각
        hk = [heard_chars[ref_to_heard[n]][1] for n in idx if n in ref_to_heard and ref_to_heard[n] < len(heard_chars)]
        if hk:
            k0, k1 = min(hk), max(hk)
            t0, t1 = good[k0][0], good[k1][1]
            heard = " ".join(good[k][2] for k in range(k0, k1 + 1))
        else:
            t0 = t1 = None
            heard = ""
        level = "ok" if not wrong else ("check" if score >= 0.6 else "bad")
        rows.append(dict(line=line, score=score, heard=heard, t0=t0, t1=t1, wrong=wrong, level=level))
    # 가사에 없는데 들린 말(받아쓴 조각 중 글자가 거의 안 맞은 것)
    extra = []
    for k, (a, b, t) in enumerate(good):
        idx = [n for n, c in enumerate(heard_chars) if c[1] == k]
        if len(idx) >= 4 and sum(1 for n in idx if n in matched_heard) / len(idx) < 0.3:
            extra.append((a, b, t))
    overall = (sum(r["score"] * len(r["line"]) for r in rows) / max(1, sum(len(r["line"]) for r in rows))) if rows else 0.0
    return dict(lines=rows, extra=extra, overall=overall, n_heard=len(good))


def _mark_wrong(line, wrong):
    """틀린 글자를 【】로 감싼다(붙은 글자는 한 묶음으로)."""
    if not wrong:
        return line
    out, i = [], 0
    ws = set(wrong)
    while i < len(line):
        if i in ws:
            j = i
            while j < len(line) and (j in ws or line[j] == " " and j + 1 in ws):
                j += 1
            out.append("【" + line[i:j] + "】")
            i = j
        else:
            out.append(line[i])
            i += 1
    return "".join(out)


def lyrics_report(res, src_name, model_name, audio_path=None, out=None, snippets=True):
    """가사 대조 결과를 md 줄 목록으로. 틀린 줄은 소리 조각(wav)도 잘라 둔다."""
    rows = res["lines"]
    bad = [r for r in rows if r["level"] == "bad"]
    chk = [r for r in rows if r["level"] == "check"]
    L = [f"# 가사 대조 — {src_name}", "",
         f"- 내가 쓴 가사 {len(rows)}줄 ↔ 받아쓴 조각 {res['n_heard']}개 (모델 {model_name})",
         f"- 전체 일치 **{res['overall']:.0%}** / 많이 다른 줄 **{len(bad)}** / 조금 다른 줄 **{len(chk)}** / 잘 맞는 줄 {len(rows) - len(bad) - len(chk)}", ""]
    if res["overall"] < 0.45:
        L.append("> ⚠️ 전체가 거의 안 맞습니다. 가사 파일이 이 노래 것이 맞는지, 언어(--lang)가 맞는지, 받아쓰기 모델이 너무 작지 않은지 먼저 보세요. 반주가 센 완성곡이면 `--deep`(목소리 분리)으로 다시 해 보세요.")
        L.append("")
    L += ["> 【】 안이 받아쓰기와 다르게 들린 글자입니다. **같은 글자가 다른 줄에서는 잘 들렸다면** 그 자리의 발음(또는 음 길이·세기)을 보세요. "
           "**곡 전체에서 늘 틀리는 글자**(예: ㅅ→ㅆ, 받침 ㄹ)는 받아쓰기 모델의 버릇일 수 있으니 소리 조각을 직접 들어 보고 판단하세요.", ""]
    L += ["| 정도 | 시각 | 내가 쓴 가사 | 들린 것 | 일치 |", "|---|---|---|---|---|"]
    for r in rows:
        if r["level"] == "ok":
            continue
        mark = "🔴" if r["level"] == "bad" else "🟠"
        ts = f"{fmt_t(r['t0'])}~{fmt_t(r['t1'])}" if r["t0"] is not None else "(못 찾음)"
        L.append(f"| {mark} | {ts} | {_mark_wrong(r['line'], r['wrong'])} | {r['heard'] or '-'} | {r['score']:.0%} |")
    if not bad and not chk:
        L.append("| ✅ | | 모든 줄이 잘 들렸습니다 | | |")
    if res["extra"]:
        L += ["", "## 가사에 없는데 들린 말", ""]
        L += [f"- {fmt_t(a)}~{fmt_t(b)} “{t}” — 애드리브·코러스이거나 받아쓰기가 지어낸 말일 수 있음" for a, b, t in res["extra"][:12]]
    L += ["", "## 잘 맞는 줄", ""]
    L += [f"- ✅ {fmt_t(r['t0']) if r['t0'] is not None else ''} {r['line']}" for r in rows if r["level"] == "ok"] or ["- 없음"]
    # 틀린 줄 소리 조각
    cuts = []
    if snippets and audio_path and out:
        import soundfile as sf
        sd = os.path.join(out, "가사조각")
        os.makedirs(sd, exist_ok=True)
        for r in (bad + chk)[:12]:
            if r["t0"] is None:
                continue
            a, b = max(0.0, r["t0"] - 0.3), r["t1"] + 0.3
            wp = os.path.join(sd, f"{fmt_t(a).replace(':', '분')}_{r['line'][:12].replace('/', ' ')}.wav")
            rr = subprocess.run(["ffmpeg", "-v", "error", "-y", "-ss", f"{a:.2f}", "-to", f"{b:.2f}", "-i", audio_path, "-ac", "2", wp], capture_output=True)
            if rr.returncode == 0:
                cuts.append(wp)
                r["snippet"] = wp
        if cuts:
            L += ["", "## 들어 볼 조각", ""] + [f"- {os.path.relpath(c, out)}" for c in cuts]
    return L


def cmd_lyrics(args):
    """내가 쓴 가사(txt)와 노래를 맞춰 본다. 받아쓴 words.json이 있으면 그걸 쓰고, 없으면 받아쓴다."""
    ref = _read_lyrics(args.lyrics)
    if not ref:
        sys.exit("가사 파일에서 줄을 못 읽었습니다: " + args.lyrics)
    out = args.out or (os.path.splitext(os.path.basename(args.file))[0] + "_가사대조")
    os.makedirs(out, exist_ok=True)
    words, model_name = [], "-"
    if args.words and os.path.isfile(args.words):
        words = [(w["start"], w["end"], w["text"], w.get("flag", "")) for w in json.load(open(args.words, encoding="utf-8"))]
        model_name = "이미 받아쓴 것"
    else:
        model = find_whisper_model(args.model)
        if not model:
            sys.exit("받아쓰기 모델(ggml-*.bin)이 없습니다. `setup`을 먼저 돌리거나 --words 로 받아쓴 words.json을 주세요.")
        model_name = os.path.basename(model)
        progress_start(out, [("받아쓰기", est_times(probe_dur(args.file), model_big="medium" in model or "large" in model)["words"]), ("가사 대조", 3)])
        progress_step("받아쓰기")
        hint = " ".join(ref[:6])[:200]  # 가사 앞부분을 힌트로 주면 고유명사·조어를 덜 틀린다
        words = clean_segments(transcribe(args.file, model, lang=args.lang, prompt=hint, chunks=looks_vocal_only(args.file)))
        json.dump([dict(start=a, end=b, text=t, flag=f) for a, b, t, f in words],
                  open(os.path.join(out, "words.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
        progress_step("가사 대조")
    res = compare_lyrics(words, ref)
    L = lyrics_report(res, os.path.basename(args.file), model_name, audio_path=args.file, out=out)
    rp = os.path.join(out, "가사대조.md")
    open(rp, "w", encoding="utf-8").write("\n".join(L))
    json.dump(dict(overall=res["overall"], lines=[{k: v for k, v in r.items() if k != "snippet"} for r in res["lines"]],
                   extra=[dict(start=a, end=b, text=t) for a, b, t in res["extra"]]),
              open(os.path.join(out, "lyrics.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1, default=float)
    progress_done()
    print(f"[완료] {rp}")
    print("\n".join(L[:12]))


def probe_dur(path):
    try:
        o = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", path],
                           capture_output=True, text=True).stdout.strip()
        return float(o)
    except Exception:
        return 0.0


# ---------------------------------------------------------------- HTML 한 장 보고서 [v1.8.0]
def _b64_file(p):
    import base64
    with open(p, "rb") as f:
        return base64.b64encode(f.read()).decode("ascii")


def _snippet_b64(audio, t, before=1.0, after=1.4):
    """시각 앞뒤를 잘라 mp3(base64)로. 못 하면 None."""
    import tempfile
    a = max(0.0, float(t) - before)
    fd, tmp = tempfile.mkstemp(suffix=".mp3")
    os.close(fd)
    try:
        r = subprocess.run(["ffmpeg", "-v", "error", "-y", "-ss", f"{a:.2f}", "-t", f"{before + after:.2f}", "-i", audio,
                            "-vn", "-ac", "1", "-ar", "32000", "-c:a", "libmp3lame", "-b:a", "80k", tmp], capture_output=True)
        if r.returncode != 0 or not os.path.getsize(tmp):
            return None
        return "data:audio/mpeg;base64," + _b64_file(tmp)
    except Exception:
        return None
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)


def build_html(out, title, verdict, summary_rows, items, images, lyrics=None, sections=None, links=None,
               fname="보고서.html", max_snip=24, max_img=20):
    """그림과 소리 조각을 안에 넣은 HTML 한 장. 브라우저에서 열어 눌러서 듣고 본다.
    items: [{level, 시각, src, kind, msg, t, end, _audio}]  images: [(설명, 경로)]
    lyrics: 가사 대조 dict(lines=[...], extra=[...], overall)  sections: describe의 sections"""
    import html as H
    esc = lambda s: H.escape(str(s if s is not None else ""))
    css = """
    body{font-family:'Noto Sans KR','Malgun Gothic','Apple SD Gothic Neo',sans-serif;max-width:1100px;margin:24px auto;padding:0 16px;color:#222;line-height:1.5}
    h1{font-size:1.5em;margin-bottom:4px} h2{font-size:1.15em;margin-top:28px;border-bottom:1px solid #ddd;padding-bottom:4px}
    .verdict{background:#f3f7ff;border-left:5px solid #3b6fd6;padding:10px 14px;margin:12px 0;font-size:1.05em}
    table{border-collapse:collapse;width:100%;font-size:.93em} th,td{border:1px solid #e3e3e3;padding:6px 8px;vertical-align:top;text-align:left}
    th{background:#fafafa} .must{background:#fff1f0} .chk{background:#fff8e6} .info{color:#777}
    audio{height:30px;width:220px;vertical-align:middle} details summary{cursor:pointer;color:#3b6fd6}
    img{max-width:100%;border:1px solid #ddd;margin:6px 0} .img-cap{font-size:.85em;color:#666}
    .wrong{background:#ffd7d7;border-radius:3px;padding:0 2px} .ok{color:#2a8a2a} .small{font-size:.85em;color:#666}
    .bar{display:inline-block;height:10px;background:#3b6fd6;vertical-align:middle}
    """
    P = [f"<!doctype html><html lang='ko'><head><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'>"
         f"<title>{esc(title)}</title><style>{css}</style></head><body>", f"<h1>🎧 {esc(title)}</h1>",
         "<div class='small'>sound-ear 보고서 — 그림·소리 조각이 이 파일 안에 들어 있어 혼자서도 열립니다. ▶를 누르면 그 자리를 들을 수 있습니다.</div>",
         f"<div class='verdict'>{esc(verdict)}</div>"]
    # 한눈에
    if summary_rows:
        P.append("<h2>한눈에</h2><table><tbody>")
        for k, v in summary_rows:
            P.append(f"<tr><th style='width:220px'>{esc(k)}</th><td>{esc(v)}</td></tr>")
        P.append("</tbody></table>")
    # 고칠 것 + 소리 조각
    P.append("<h2>고칠 것 · 들어 볼 곳</h2>")
    show = [p for p in items if p.get("level") != LEVEL_INFO]
    info = [p for p in items if p.get("level") == LEVEL_INFO]
    if not show:
        P.append("<p class='ok'>🎉 크게 고칠 곳이 없습니다.</p>")
    else:
        P.append("<table><thead><tr><th>급함</th><th>시각</th><th>어디서</th><th>무엇</th><th>설명</th><th>듣기</th></tr></thead><tbody>")
        n_snip = 0
        for p in show:
            cls = "must" if p["level"] == LEVEL_MUST else "chk"
            mark = "🔴" if p["level"] == LEVEL_MUST else "🟠"
            snip = ""
            t = p.get("t", 0) or 0
            is_whole = p.get("end") is None and t == 0
            if not is_whole and p.get("_audio") and n_snip < max_snip and os.path.isfile(p["_audio"]):
                b = _snippet_b64(p["_audio"], t)
                if b:
                    snip = f"<audio controls preload='none' src='{b}'></audio>"
                    n_snip += 1
            P.append(f"<tr class='{cls}'><td>{mark}</td><td>{esc(p.get('시각', fmt_t(t)))}</td><td>{esc(p.get('src', ''))}</td>"
                     f"<td>{esc(p.get('kind', ''))}</td><td>{esc(p.get('msg', ''))}</td><td>{snip}</td></tr>")
        P.append("</tbody></table>")
    if info:
        P.append(f"<details><summary>참고 {len(info)}곳 보기</summary><ul class='info'>")
        P += [f"<li>{esc(p.get('시각', ''))} {esc(p.get('kind', ''))} — {esc(p.get('msg', ''))}</li>" for p in info]
        P.append("</ul></details>")
    # 가사 대조
    if lyrics and lyrics.get("lines"):
        rows = lyrics["lines"]
        bad = sum(1 for r in rows if r["level"] == "bad")
        chk = sum(1 for r in rows if r["level"] == "check")
        P.append(f"<h2>가사 대조</h2><p>전체 일치 <b>{lyrics['overall']:.0%}</b> — 많이 다른 줄 {bad}, 조금 다른 줄 {chk}, 잘 맞는 줄 {len(rows) - bad - chk}. "
                 "<span class='small'>빨간 글자가 받아쓰기와 다르게 들린 곳. 같은 글자가 다른 줄에서는 잘 들렸다면 그 자리의 발음을 보세요.</span></p>")
        P.append("<table><thead><tr><th>정도</th><th>시각</th><th>내가 쓴 가사</th><th>들린 것</th><th>일치</th><th>듣기</th></tr></thead><tbody>")
        for r in rows:
            if r["level"] == "ok":
                continue
            marked = "".join(f"<span class='wrong'>{esc(ch)}</span>" if i in set(r["wrong"]) else esc(ch) for i, ch in enumerate(r["line"]))
            ts = f"{fmt_t(r['t0'])}~{fmt_t(r['t1'])}" if r.get("t0") is not None else "(못 찾음)"
            snip = ""
            if r.get("snippet") and os.path.isfile(r["snippet"]):
                try:
                    snip = f"<audio controls preload='none' src='data:audio/wav;base64,{_b64_file(r['snippet'])}'></audio>"
                except Exception:
                    snip = ""
            P.append(f"<tr class='{'must' if r['level'] == 'bad' else 'chk'}'><td>{'🔴' if r['level'] == 'bad' else '🟠'}</td><td>{ts}</td>"
                     f"<td>{marked}</td><td>{esc(r['heard'] or '-')}</td><td>{r['score']:.0%}</td><td>{snip}</td></tr>")
        if not bad and not chk:
            P.append("<tr><td>✅</td><td></td><td colspan='4'>모든 줄이 잘 들렸습니다</td></tr>")
        P.append("</tbody></table>")
        if lyrics.get("extra"):
            P.append("<p class='small'>가사에 없는데 들린 말: " + " / ".join(f"{fmt_t(a)} “{esc(t)}”" for a, b, t in lyrics["extra"][:8]) + "</p>")
    # 구성
    if sections:
        P.append("<h2>곡 흐름</h2><table><thead><tr><th>시각</th><th>구간</th><th>세기</th><th>드럼</th><th>저음</th><th>가사</th></tr></thead><tbody>")
        es = [s.get("energy_db", -30) for s in sections]
        top = max(es) if es else 0
        for s in sections:
            w = int(max(4, 120 + 4 * (s.get("energy_db", -30) - top)))
            P.append(f"<tr><td>{fmt_t(s['start'])}~{fmt_t(s['end'])}</td><td>{esc(s['name'])}</td><td><span class='bar' style='width:{w}px'></span></td>"
                     f"<td>{'O' if s.get('drums') else '·'}</td><td>{'O' if s.get('bass') else '·'}</td><td class='small'>{esc((s.get('words') or '')[:70])}</td></tr>")
        P.append("</tbody></table>")
    # 그림
    if images:
        P.append("<h2>그림</h2>")
        for cap, pth in images[:max_img]:
            if os.path.isfile(pth):
                P.append(f"<div class='img-cap'>{esc(cap)}</div><img src='data:image/png;base64,{_b64_file(pth)}' alt='{esc(cap)}'>")
        if len(images) > max_img:
            P.append(f"<p class='small'>그림 {len(images) - max_img}장은 폴더에만 있습니다.</p>")
    if links:
        P.append("<h2>자세한 결과(폴더 안 파일)</h2><ul>" + "".join(f"<li>{esc(l)}</li>" for l in links) + "</ul>")
    P.append("<p class='small'>※ 자동 검사는 '의심 가는 곳'을 찾는 것입니다. 들어 보고 최종 판단하세요.</p></body></html>")
    hp = os.path.join(out, fname)
    open(hp, "w", encoding="utf-8").write("\n".join(P))
    return hp



# ---------------------------------------------------------------- 출처 검사 (어디서 만든 파일인지 — 꼬리표·출처 서명) [v1.8.1]
AI_MAKERS = [("suno", "Suno"), ("udio", "Udio"), ("mureka", "Mureka"), ("riffusion", "Riffusion"), ("stable audio", "Stable Audio"),
             ("stableaudio", "Stable Audio"), ("elevenlabs", "ElevenLabs"), ("musicgen", "MusicGen"), ("audiocraft", "AudioCraft"),
             ("boomy", "Boomy"), ("soundraw", "Soundraw"), ("aiva", "AIVA"), ("ai generated", "AI generated"), ("ai-generated", "AI generated"),
             ("generated by ai", "AI generated"), ("synthesizer v", "Synthesizer V(보컬 합성)"), ("vocaloid", "VOCALOID(보컬 합성)")]


def _scan_bytes(path, limit=64 << 20):
    """파일 앞·뒤를 훑어 C2PA 출처 서명(JUMBF)과 만든 곳 이름이 글자로 박혀 있는지 본다."""
    import re
    size = os.path.getsize(path)
    chunks = []
    with open(path, "rb") as f:
        chunks.append(f.read(min(size, 4 << 20)))          # 앞 4MB (ID3·moov·JUMBF는 보통 앞)
        if size > 8 << 20:
            f.seek(size - (4 << 20))
            chunks.append(f.read())                         # 뒤 4MB (ID3v1·꼬리 atom)
    blob = b"".join(chunks)
    low = blob.lower()
    out = {"c2pa": False, "c2pa_hint": [], "makers": []}
    for key in (b"urn:c2pa", b"c2pa.", b"jumb", b"contentauth", b"claim_generator"):
        if key in low:
            out["c2pa"] = True
    if out["c2pa"]:
        # claim_generator 근처 글자(어떤 프로그램이 서명했는지)
        for m in re.finditer(rb"claim_generator[^\x00]{0,8}([\x20-\x7e]{3,80})", blob):
            out["c2pa_hint"].append(m.group(1).decode("ascii", "ignore"))
            if len(out["c2pa_hint"]) >= 3:
                break
    for key, name in AI_MAKERS:
        k = key.encode()
        i = low.find(k)
        if i >= 0 and name not in [m[0] for m in out["makers"]]:
            ctx = blob[max(0, i - 40): i + 60]
            ctx = "".join(chr(c) if 32 <= c < 127 else "·" for c in ctx)
            out["makers"].append((name, ctx))
    # 유튜브·다운로더 흔적
    for key, name in ((b"lavf", "ffmpeg(Lavf)"), (b"lame", "LAME"), (b"youtube", "YouTube"), (b"yt-dlp", "yt-dlp")):
        if key in low[: 1 << 20]:
            out.setdefault("tools", []).append(name)
    return out


def cmd_origin(args):
    """꼬리표(메타데이터)·출처 서명으로 어디서 만든 파일인지 본다. 맞으면 확실하고, 없으면 '모른다'고 말한다."""
    import re
    path = args.file
    out = args.out or (os.path.splitext(os.path.basename(path))[0] + "_출처")
    os.makedirs(out, exist_ok=True)
    tags = {}
    try:
        j = json.loads(subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format_tags:stream_tags:format=format_name",
                                       "-of", "json", path], capture_output=True, text=True, check=True).stdout)
        tags.update(j.get("format", {}).get("tags", {}) or {})
        for st in j.get("streams", []):
            for k, v in (st.get("tags") or {}).items():
                tags.setdefault("stream:" + k, v)
        fmt = j.get("format", {}).get("format_name", "")
    except Exception as e:  # noqa
        fmt = "?"
        tags["(오류)"] = str(e)
    found = []     # (근거, 설명, 확실함)
    for k, v in tags.items():
        lv = str(v).lower()
        for key, name in AI_MAKERS:
            if key in lv or key in k.lower():
                found.append((f"꼬리표 {k} = {str(v)[:120]}", f"{name}에서 만든 파일로 보임", True))
                break
    comm = next((v for k, v in tags.items() if k.lower() in ("comment", "description", "stream:comment")), None)
    if comm and len(str(comm)) > 80 and re.search(r"(bpm|vocal|male|female|verse|chorus|style|genre|upbeat|lo-fi|synth)", str(comm), re.I):
        found.append((f"꼬리표 comment = {str(comm)[:120]}…", "AI 생성 프롬프트처럼 보이는 긴 설명이 들어 있음", False))
    b = _scan_bytes(path)
    if b["c2pa"]:
        hint = ", ".join(b["c2pa_hint"]) or "서명한 프로그램 이름은 못 읽음"
        found.append(("파일 안에 C2PA 출처 서명(JUMBF)", f"만든 프로그램이 서명해 둔 출처 기록이 있음 — {hint}", True))
    for name, ctx in b["makers"]:
        if not any(name in f[1] for f in found):
            found.append((f"파일 바이트에 '{name}' 글자: {ctx}", f"{name} 이름이 파일 안에 남아 있음", False))
    uuid = re.search(r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}", os.path.basename(path), re.I)
    if uuid:
        found.append((f"파일 이름의 ID {uuid.group(0)}", "수노·우디오가 내려받기 이름에 쓰는 꼴의 ID — 브라우저로 suno.com/song/ID 를 열어 보면 확인됨", False))
    sure = [f for f in found if f[2]]
    verdict = ("출처 확인됨: " + " / ".join(sorted({f[1].split('에서')[0].split(' 이름')[0] for f in sure}))) if sure else (
        "출처를 가리키는 흔적은 있지만 확실하진 않음" if found else "출처 흔적 없음 — 꼬리표가 지워졌거나 원래 없는 파일. AI가 만든 게 아니라는 뜻은 아님")
    L = [f"# 출처 검사 — {os.path.basename(path)}", "", f"> **{verdict}**", "",
         f"- 형식: {fmt} / 꼬리표 {len(tags)}개 / 파일 안 도구 흔적: {', '.join(b.get('tools', [])) or '없음'}", ""]
    L += ["## 근거", ""] + ([f"- {'✅' if c else '🟠'} {g} → {d}" for g, d, c in found] or ["- 없음"])
    L += ["", "## 꼬리표 전부", ""] + ([f"- {k}: {str(v)[:200]}" for k, v in tags.items()] or ["- 없음"])
    L += ["", "## 더 확인하는 법", "",
          "- 수노 공개 곡인지: 파일 이름이나 꼬리표에 ID가 있으면 `suno.com/song/ID`를 열어 본다.",
          "- 가사로 찾기: `words`로 받아쓴 가사 중 특이한 한 줄을 따옴표로 묶어 구글에 검색한다(`\"가사 한 줄\" suno`). 수노 자체 검색은 가사를 안 찾아 주고, 새 곡은 구글에도 아직 없을 때가 많다 [확인: 10-26].",
          "- 소리만으로 AI 여부를 가리는 검사는 넣지 않았다 — 평균 스펙트럼으로는 수노 곡과 사람 곡이 갈리지 않았다 [확인: 10-26 수노 4곡·사람 3곡].",
          "", "※ 꼬리표와 서명은 지울 수 있다. '흔적 없음'은 '사람이 만들었다'가 아니라 '모른다'이다."]
    rp = os.path.join(out, "출처.md")
    open(rp, "w", encoding="utf-8").write("\n".join(L))
    json.dump(dict(verdict=verdict, found=[dict(evidence=g, meaning=d, sure=c) for g, d, c in found], tags=tags, c2pa=b["c2pa"]),
              open(os.path.join(out, "origin.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    if not getattr(args, "quiet", False):
        print(f"[완료] {rp}")
        print("\n".join(L[:12]))
    return 0



# ---------------------------------------------------------------- 노트 단위 검사 (브이신스 .svp / 보컬로이드 .vpr / UTAU .ust / MIDI) [v1.9.0]
SVP_Q = 705600000          # 브이신스: 4분음표 하나 = 705,600,000 blick
MIDI_Q = 480               # 보컬로이드 vpr·UTAU ust: 4분음표 하나 = 480 틱


def _tempo_map(tempos, q):
    """[(위치틱, bpm)] → 틱을 초로 바꾸는 함수. 템포가 중간에 바뀌어도 맞음."""
    tempos = sorted((int(p), float(b)) for p, b in tempos) or [(0, 120.0)]
    if tempos[0][0] > 0:
        tempos.insert(0, (0, tempos[0][1]))
    segs = []   # (시작틱, 시작초, bpm)
    sec = 0.0
    for i, (p, b) in enumerate(tempos):
        if i > 0:
            pp, bb = tempos[i - 1]
            sec += (p - pp) / q * 60.0 / bb
        segs.append((p, sec, b))

    def to_sec(tick):
        s = segs[0]
        for sg in segs:
            if sg[0] <= tick:
                s = sg
        return s[1] + (tick - s[0]) / q * 60.0 / s[2]
    return to_sec


def read_project(path, track=None):
    """노래 프로젝트에서 노트 목록을 읽는다. [{i, start, end, pitch(midi), lyric, phonemes, track}]
    되는 형식: 브이신스 .svp, 보컬로이드5/6 .vpr, UTAU .ust, 표준 MIDI .mid"""
    ext = os.path.splitext(path)[1].lower()
    notes, tracks = [], []
    if ext == ".svp":
        j = json.load(open(path, encoding="utf-8-sig"))
        to_sec = _tempo_map([(t["position"], t["bpm"]) for t in j.get("time", {}).get("tempo", [])], SVP_Q)
        lib = {g.get("uuid"): g for g in j.get("library", [])}
        for tr in j.get("tracks", []):
            name = tr.get("name", "")
            tracks.append(name)
            if track and name != track:
                continue
            groups = [(tr.get("mainGroup", {}), int((tr.get("mainRef") or {}).get("blickOffset", 0)))]
            for ref in tr.get("groups", []):
                g = lib.get(ref.get("groupID"))
                if g:
                    groups.append((g, int(ref.get("blickOffset", 0))))
            for g, off in groups:
                for n in g.get("notes", []):
                    a, d = int(n["onset"]) + off, int(n["duration"])
                    notes.append(dict(start=to_sec(a), end=to_sec(a + d), pitch=int(n["pitch"]), lyric=str(n.get("lyrics", "")),
                                      phonemes=str(n.get("phonemes", "")), track=name))
    elif ext == ".vpr":
        import zipfile
        with zipfile.ZipFile(path) as z:
            seq = next(n for n in z.namelist() if n.endswith("sequence.json"))
            j = json.loads(z.read(seq).decode("utf-8"))
        ev = (j.get("masterTrack", {}).get("tempo", {}) or {}).get("events", [])
        to_sec = _tempo_map([(e["pos"], e["value"] / 100.0) for e in ev], MIDI_Q)
        for tr in j.get("tracks", []):
            name = tr.get("name", "")
            tracks.append(name)
            if track and name != track:
                continue
            for part in tr.get("parts", []):
                off = int(part.get("pos", 0))
                for n in part.get("notes", []):
                    a, d = off + int(n["pos"]), int(n["duration"])
                    notes.append(dict(start=to_sec(a), end=to_sec(a + d), pitch=int(n["number"]), lyric=str(n.get("lyric", "")),
                                      phonemes=str(n.get("phoneme", "")), track=name))
    elif ext == ".ust":
        raw = open(path, "rb").read()
        for enc in ("utf-8-sig", "cp932", "cp949", "latin-1"):
            try:
                txt = raw.decode(enc)
                break
            except Exception:
                continue
        tempo, tick, cur = 120.0, 0, None
        items = []
        for line in txt.splitlines():
            line = line.strip()
            if line.startswith("[#"):
                if cur is not None:
                    items.append(cur)
                cur = {"_sec": line} if line not in ("[#SETTING]", "[#VERSION]", "[#TRACKEND]") else None
                continue
            if "=" not in line:
                continue
            k, v = line.split("=", 1)
            if k == "Tempo":
                try:
                    tempo = float(v)
                except ValueError:
                    pass
                if cur is not None:
                    cur["Tempo"] = tempo
            elif cur is not None:
                cur[k] = v
        if cur is not None:
            items.append(cur)
        tracks.append("UTAU")
        sec = 0.0
        bpm = tempo
        for it in items:
            bpm = float(it.get("Tempo", bpm))
            L = int(float(it.get("Length", 0)))
            dur = L / MIDI_Q * 60.0 / bpm
            lyric = it.get("Lyric", "")
            if lyric not in ("R", "r", "", "休"):
                notes.append(dict(start=sec, end=sec + dur, pitch=int(it.get("NoteNum", 60)), lyric=lyric, phonemes="", track="UTAU"))
            sec += dur
    elif ext in (".mid", ".midi"):
        notes, tracks = _read_midi(path, track)
    else:
        sys.exit("모르는 프로젝트 형식입니다: " + ext + " (.svp .vpr .ust .mid 가 됩니다)")
    notes.sort(key=lambda n: n["start"])
    for i, n in enumerate(notes, 1):
        n["i"] = i
    return notes, tracks


def _read_midi(path, track=None):
    """표준 MIDI를 아주 단순하게 읽는다(노트·템포·가사 메타만)."""
    data = open(path, "rb").read()
    if data[:4] != b"MThd":
        sys.exit("MIDI 파일이 아닙니다")
    hlen = int.from_bytes(data[4:8], "big")
    fmt, ntr, div = int.from_bytes(data[8:10], "big"), int.from_bytes(data[10:12], "big"), int.from_bytes(data[12:14], "big")
    pos = 8 + hlen
    tempos, raw_tracks = [], []
    for ti in range(ntr):
        if data[pos:pos + 4] != b"MTrk":
            break
        tl = int.from_bytes(data[pos + 4:pos + 8], "big")
        chunk = data[pos + 8: pos + 8 + tl]
        pos += 8 + tl
        p, tick, status, name = 0, 0, 0, f"track{ti}"
        on, notes, lyrics = {}, [], []

        def vlq():
            nonlocal p
            v = 0
            while True:
                b = chunk[p]
                p += 1
                v = (v << 7) | (b & 0x7F)
                if not b & 0x80:
                    return v
        while p < len(chunk):
            tick += vlq()
            b = chunk[p]
            if b == 0xFF:
                typ = chunk[p + 1]
                p += 2
                ln = vlq()
                body = chunk[p:p + ln]
                p += ln
                if typ == 0x51:
                    tempos.append((tick, 60000000.0 / int.from_bytes(body, "big")))
                elif typ == 0x03:
                    name = body.decode("utf-8", "ignore")
                elif typ == 0x05:
                    lyrics.append((tick, body.decode("utf-8", "ignore")))
                continue
            if b in (0xF0, 0xF7):
                p += 1
                ln = vlq()
                p += ln
                continue
            if b & 0x80:
                status = b
                p += 1
            ev = status & 0xF0
            if ev in (0x80, 0x90, 0xA0, 0xB0, 0xE0):
                d1, d2 = chunk[p], chunk[p + 1]
                p += 2
            else:
                d1, d2 = chunk[p], 0
                p += 1
            if ev == 0x90 and d2 > 0:
                on[d1] = tick
            elif ev == 0x80 or (ev == 0x90 and d2 == 0):
                if d1 in on:
                    notes.append((on.pop(d1), tick, d1))
        raw_tracks.append((name, notes, lyrics))
    to_sec = _tempo_map(tempos, div)
    out, names = [], []
    for name, notes, lyrics in raw_tracks:
        names.append(name)
        if not notes or (track and name != track):
            continue
        for a, b, pch in notes:
            lyr = next((t for tk, t in lyrics if tk == a), "")
            out.append(dict(start=to_sec(a), end=to_sec(b), pitch=pch, lyric=lyr, phonemes="", track=name))
    return out, names


def _best_offset(notes, pt, pm, search=1.5, step=0.01):
    """소리의 음높이 곡선이 노트에 가장 잘 맞는 시간차(초)를 찾는다(렌더 앞에 빈 소리가 붙었을 때). 양수면 소리가 늦음."""
    best, boff = -1, 0.0
    good = np.isfinite(pm)
    for off in np.arange(-search, search + 1e-9, step):
        score = 0
        for n in notes:
            m = good & (pt >= n["start"] + off) & (pt < n["end"] + off)
            if m.any():
                score += int(np.sum(np.abs(pm[m] - n["pitch"]) < 0.8))
        if score > best:
            best, boff = score, float(off)
    return boff


def cmd_notes(args):
    """노래 소리 + 프로젝트 파일 → 노트마다 음정·길이·흠을 짚는 표. 몇 번째 노트를 고치면 되는지 바로 보이게."""
    import librosa
    from argparse import Namespace
    notes, tracks = read_project(args.project, args.track)
    if not notes:
        sys.exit("노트를 못 읽었습니다. 트랙 이름: " + ", ".join(tracks) + " → --track 으로 고르세요")
    out = args.out or (os.path.splitext(os.path.basename(args.file))[0] + "_노트검사")
    os.makedirs(out, exist_ok=True)
    dur_p = probe_dur(args.file)
    progress_start(out, [("음높이 재기", 10 + 0.25 * dur_p), ("흠 찾기", est_times(dur_p)["check_vocal"]), ("노트 맞추기", 3)])
    progress_step("음높이 재기")
    st, sr0 = load(args.file)
    sr = 22050
    y = librosa.resample(st.mean(axis=1), orig_sr=sr0, target_sr=sr)
    hop = 256
    f0, vflag, _ = librosa.pyin(y, fmin=65.0, fmax=1400.0, sr=sr, frame_length=2048, hop_length=hop)
    pt = librosa.times_like(f0, sr=sr, hop_length=hop)
    pm = np.where(vflag & np.isfinite(f0), 69 + 12 * np.log2(np.where(np.isfinite(f0), f0, 1.0) / 440.0), np.nan)
    rms = librosa.feature.rms(y=y, frame_length=2048, hop_length=hop)[0][: len(pt)]
    rdb = db(rms)
    loud = float(np.percentile(rdb, 90))
    off = args.offset if args.offset is not None else _best_offset(notes, pt, pm)
    # 흠 찾기(보컬 모드) — 이미 한 결과가 있으면 재사용
    progress_step("흠 찾기")
    cdir = os.path.join(out, "소리검사")
    rj = os.path.join(cdir, "result.json")
    if not (os.path.isfile(rj) and os.path.getmtime(rj) >= os.path.getmtime(args.file)):
        _check_one(Namespace(file=args.file, mode="vocal", out=cdir, max_zoom=8, key_from=None), quiet=True)
    probs = [p for p in json.load(open(rj, encoding="utf-8")).get("problems", []) if not (p.get("end") is None and p["t"] == 0)]
    progress_step("노트 맞추기")
    rows = []
    n_bad = n_chk = 0
    for n in notes:
        a, b = n["start"] + off, n["end"] + off
        length = n["end"] - n["start"]
        pad = min(0.03, length * 0.2)
        m = (pt >= a + pad) & (pt < b - pad)
        core = pm[m]
        voiced = core[np.isfinite(core)]
        issues = []
        cents = None
        vfrac = float(len(voiced) / max(1, len(core)))
        mm_all = (pt >= a) & (pt < b)
        thr = loud - 25
        sounding = float(np.sum(rdb[mm_all] > thr)) * hop / sr if mm_all.any() else 0.0
        # 앞 노트의 꼬리(릴리스)가 넘어온 것과 가르기: 노트 시작에서 소리가 '새로 커지는지', 뒤쪽 구간에 소리가 있는지
        late = (pt >= a + min(0.08, 0.3 * length)) & (pt < b)
        sounding_late = float(np.sum(rdb[late] > thr)) * hop / sr if late.any() else 0.0
        e_m, p_m = (pt >= a) & (pt < a + 0.05), (pt >= a - 0.05) & (pt < a)
        early_lv = float(np.median(rdb[e_m])) if e_m.any() else -120.0
        pre_lv = float(np.median(rdb[p_m])) if p_m.any() else -120.0
        onset = early_lv > thr and early_lv > pre_lv + 3
        if length > 0.12 and 0.02 <= sounding < 0.5 * length and onset:
            # 실제로 난 소리만으로 음정을 잰다
            sm = mm_all & (rdb > thr)
            core = pm[sm]
            voiced = core[np.isfinite(core)]
            vfrac = float(len(voiced) / max(1, len(core)))
        if sounding_late < 0.05 and not onset:
            issues.append(("🔴", "소리가 없음 — 노트가 렌더에 없거나 음소거"))
        elif vfrac < 0.3:
            issues.append(("🟠", f"목소리로 안 들림(유성 {vfrac:.0%}) — 숨·자음만 남았거나 너무 짧음"))
        else:
            cents = float(np.median(voiced) - n["pitch"]) * 100
            if abs(cents) > 60:
                issues.append(("🔴", f"음정 {cents:+.0f}센트 — {note_name(n['pitch'] + round(cents / 100))} 쪽으로 들림"))
            elif abs(cents) > 30:
                issues.append(("🟠", f"음정 {cents:+.0f}센트 치우침"))
            sp = float(np.nanstd(core)) * 100 if len(voiced) > 4 else 0.0
            if sp > 80 and length > 0.25:
                issues.append(("🟠", f"노트 안에서 음이 {sp:.0f}센트나 흔들림(비브라토·꺾임이 과한지)"))
        # 쓴 길이보다 실제로 난 소리가 훨씬 짧으면(끝이 일찍 죽음·음소가 짧음)
        if length > 0.12 and 0.02 <= sounding < 0.5 * length and onset:
            issues.append(("🟠", f"쓴 길이 {length * 1000:.0f}ms인데 실제 소리는 {sounding * 1000:.0f}ms — 끝이 일찍 끊기거나 음소가 짧음"))
        if length < 0.08 and n["lyric"] and n["lyric"] not in ("-", "+", "a", "あ"):
            issues.append(("🟠", f"길이 {length * 1000:.0f}ms — 너무 짧아 '{n['lyric']}' 발음이 뭉개질 수 있음"))
        # 이 노트 자리의 흠(끝·시작 50ms 안도 포함)
        for p in probs:
            t = p["t"]
            if a - 0.05 <= t <= b + 0.05 and p["kind"] not in ("음정 어긋남",):
                where = "끝" if t > b - 0.06 else ("시작" if t < a + 0.06 else "중간")
                issues.append(("🔴" if p["level"] == LEVEL_MUST else "🟠", f"{p['kind']} ({where}, {fmt_t(t)})"))
        if any(x[0] == "🔴" for x in issues):
            n_bad += 1
        elif issues:
            n_chk += 1
        rows.append(dict(i=n["i"], start=a, end=b, length=length, lyric=n["lyric"], pitch=n["pitch"], cents=cents, voiced=vfrac,
                         issues=issues, track=n["track"]))
    verdict = ("노트 전부 깨끗함 👍" if not n_bad and not n_chk else
               f"꼭 볼 노트 {n_bad}개 🔧, 들어 볼 노트 {n_chk}개" if n_bad else f"꼭 고칠 노트는 없음 👍 — 들어 볼 노트 {n_chk}개")
    tuned = [r["cents"] for r in rows if r["cents"] is not None]
    glob = float(np.median(tuned)) if tuned else 0.0
    L = [f"# 노트 검사 — {os.path.basename(args.file)} ↔ {os.path.basename(args.project)}", "", f"> **한마디: {verdict}**", "",
         f"- 노트 {len(rows)}개 (트랙: {', '.join(sorted(set(r['track'] for r in rows)))}) / 소리가 프로젝트보다 {off:+.2f}초 밀림(자동 맞춤) / 전체 튜닝 치우침 {glob:+.0f}센트",
         "- 센트: 100센트 = 반음. ±30 안이면 보통 괜찮고, 60 넘으면 다른 음으로 들림. 일부러 넣은 꺾임·비브라토면 넘어가세요.", ""]
    if abs(glob) > 20:
        L.append(f"> 노트 대부분이 같은 쪽으로 {glob:+.0f}센트 치우쳐 있어요. 보컬 전체 튜닝(A=440)이나 반주와의 기준음을 먼저 보세요.\n")
    L += ["## 고칠 노트", "", "| 노트 | 시각 | 가사 | 음 | 길이 | 측정 | 문제 |", "|---|---|---|---|---|---|---|"]
    shown = 0
    for r in rows:
        if not r["issues"]:
            continue
        shown += 1
        mk = "🔴" if any(x[0] == "🔴" for x in r["issues"]) else "🟠"
        meas = f"{r['cents']:+.0f}c" if r["cents"] is not None else "-"
        L.append(f"| {mk} **{r['i']}** | {fmt_t(r['start'])} | {r['lyric'] or '(없음)'} | {note_name(r['pitch'])} | {r['length'] * 1000:.0f}ms | {meas} | "
                 + "; ".join(x[1] for x in r["issues"]) + " |")
    if not shown:
        L.append("| ✅ | | 모든 노트가 괜찮습니다 | | | | |")
    L += ["", "## 노트 전부", "", "| 노트 | 시각 | 가사 | 음 | 길이 | 측정 |", "|---|---|---|---|---|---|"]
    for r in rows:
        meas = f"{r['cents']:+.0f}c" if r["cents"] is not None else "-"
        L.append(f"| {r['i']} | {fmt_t(r['start'])} | {r['lyric']} | {note_name(r['pitch'])} | {r['length'] * 1000:.0f}ms | {meas} |")
    L += ["", "- 소리 검사(시각 기준 흠·그림): `소리검사/보고서.md`", "", "※ 노트 번호는 프로젝트의 시간 순서(1부터)입니다. 트랙이 여러 개면 `--track 이름`으로 하나만 보세요."]
    rp = os.path.join(out, "노트표.md")
    open(rp, "w", encoding="utf-8").write("\n".join(L))
    json.dump(dict(verdict=verdict, offset=off, global_cents=glob,
                   notes=[dict(r, issues=[dict(level=l, msg=m) for l, m in r["issues"]]) for r in rows]),
              open(os.path.join(out, "notes.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1, default=float)
    # 그림: 노트(상자) 위에 측정 음높이(점)
    try:
        plt = setup_plt()
        t0, t1 = max(0.0, rows[0]["start"] - 1), rows[-1]["end"] + 1
        fig, ax = plt.subplots(1, 1, figsize=(16, 5))
        mm = (pt >= t0) & (pt <= t1)
        ax.plot(pt[mm], pm[mm], ".", ms=2, color="tab:green", label="들린 음높이")
        for r in rows:
            bad = bool(r["issues"])
            ax.plot([r["start"], r["end"]], [r["pitch"], r["pitch"]], lw=4, color="tab:red" if bad else "gray", alpha=0.7)
            if bad:
                ax.text(r["start"], r["pitch"] + 0.6, str(r["i"]), fontsize=8, color="tab:red")
        ax.set_xlim(t0, t1)
        ax.set_xlabel("시간(초)")
        ax.set_ylabel("음(MIDI)")
        ax.set_title("노트(회색/빨강=문제) 위에 들린 음높이(초록)")
        ax.legend(loc="upper right")
        fig.tight_layout()
        fig.savefig(os.path.join(out, "노트그림.png"), dpi=80)
        plt.close(fig)
    except Exception as e:  # noqa
        print("그림을 못 그렸습니다:", e)
    progress_done()
    print(f"[완료] {rp}")
    print("\n".join(L[:14]))
    return 0


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
                          msg=(f"{len(groups)}곳에서 최대치에 닿습니다(주로 {spans}). " if len(groups) >= 5 else
                               f"{len(groups)}곳({', '.join(fmt_t(a) for a, b, n in groups)})에서 최대치에 닿습니다. ")
                          + "손실 압축 파일이라 진짜로 깨졌는지는 원본 wav로 봐야 정확합니다."
                          + (" 마스터가 아주 세게 눌린 것은 맞습니다." if len(groups) >= 20 else "")))
        groups = []
    for a, b, n in groups[:40]:
        probs.append(dict(t=a, end=b, kind="클리핑(소리 깨짐)", level=LEVEL_MUST,
                          msg=f"소리가 최대치를 넘어 잘렸습니다({n}곳 묶음). 볼륨이나 마스터를 낮추세요."))
    if len(groups) > 40:
        probs.append(dict(t=groups[40][0], end=groups[-1][1], kind="클리핑(소리 깨짐)", level=LEVEL_MUST,
                          msg=f"이 뒤로도 {len(groups) - 40}곳 더 있습니다."))
    if not groups and tp > 1.0:
        probs.append(dict(t=0, end=None, kind="트루 피크 넘침", level=LEVEL_CHECK,
                          msg=f"샘플은 안 넘지만 실제 재생 때 +{db(tp):.2f}dBTP까지 올라가 mp3로 바꾸면 깨질 수 있습니다. 끝을 -1dB로 맞추세요."))

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
    # 기준은 곡 크기에 맞춤: 조용히 만든 곡(-19 LUFS쯤)도 잡히게 [확인: 10-25 예시곡 0.25초 구멍을 -25dB 고정 기준으로 놓침]
    thr = min(-25.0, float(np.median(lv[loud])) - 8)
    for i in range(len(lv) - 2):
        if lv[i] > thr and min(lv[i + 1], lv[i + 2]) < -60:
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
                # 노트가 끝나거나 시작하는 곳(앞뒤 20ms 세기가 2배 넘게 다름)은 이음새의 자연스러운 흔들림
                # [확인: 10-25 「구마」 한국어 커버 1:40.9 — 큰 노트가 끝나는 순간 화음 목소리가 드러나며 튐으로 보임]
                kk = a + ga
                n20 = max(1, int(0.02 * sr / h2))
                rb_ = float(np.mean(r2[max(0, kk - n20):kk])) if kk > 0 else 0.0
                ra_ = float(np.mean(r2[kk + 1:kk + 1 + n20])) if kk + 1 < len(r2) else 0.0
                if min(rb_, ra_) > 0 and max(rb_, ra_) / min(rb_, ra_) > 2.0:
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
        # 음이 미끄러지며 움직이는 노트(흔들림 25센트 넘음)는 표현이지 음정 틀림이 아니다 [확인: 10-25 「Alone」 3:36 레#→레 미끄러짐]
        if n["dur"] >= 0.25 and abs(d) > 32 and n["std"] <= 25 and flagged < 40:
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


def check_sibilance_breath(y, sr, probs, stats, pitch, backing=None):
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
    # 분리한 목소리라면: 같은 순간 반주에도 쇳소리(하이햇·심벌)가 있으면 반주에서 샌 소리로 보고 뺀다
    # [확인: 10-25 「구마」 원곡 분리 목소리에서 쇳소리 62곳 — 대부분 하이햇이 샌 것]
    leak = 0
    if backing is not None:
        import librosa as _lb
        by_, _ = _lb.load(backing, sr=sr, mono=True)
        Sb = np.abs(_lb.stft(by_, n_fft=1024, hop_length=256)) ** 2
        hb = 10 * np.log10(Sb[(freqs >= 5000) & (freqs <= 10000)].sum(axis=0) + 1e-12)
        on = _lb.onset.onset_detect(onset_envelope=np.maximum(0, np.diff(hb, prepend=hb[0])), sr=sr, hop_length=256, units="time")
        if len(on):
            keep = [e for e in ev if np.min(np.abs(on - e)) > 0.04]
            leak = len(ev) - len(keep)
            ev = keep
    groups = group_times(ev, 0.2)
    if leak:
        stats["반주에서 샌 쇳소리(뺌)"] = leak
    stats.setdefault("센 ㅅ·ㅊ 소리", len(groups))
    if len(groups) > 8:
        # 많으면 하나하나보다 '전체적으로 셈'이 맞는 진단 — 디에서 하나로 고칠 일
        ex = ", ".join(fmt_t(g[0]) for g in groups[:6])
        probs.append(dict(t=groups[0][0], end=groups[-1][0], kind="ㅅ·ㅊ 소리 셈(곡 전체)", level=LEVEL_CHECK,
                          msg=f"쇳소리가 모음만큼 센 곳이 {len(groups)}곳으로 곡 전체에 퍼져 있습니다(예: {ex} …). "
                              "노트마다 고치기보다 보컬 트랙에 디에서(De-esser)를 한 번 거는 게 좋습니다."))
        groups = []
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
    return "vocal" if looks_vocal_only(path) else "mix"


def looks_vocal_only(path):
    """70Hz 아래(킥·베이스) 소리가 거의 없으면 보컬만 있는 파일로 본다.
    [확인: 10-25 보컬만 6개 0.0000~0.0001, 완성곡 6개 0.09~0.48 — 기준 0.01]"""
    import librosa
    r = subprocess.run(["ffmpeg", "-v", "error", "-i", path, "-t", "240", "-vn", "-f", "f32le", "-ac", "1", "-ar", "16000", "-"],
                       capture_output=True)
    y = np.frombuffer(r.stdout, np.float32)
    if len(y) < 16000:
        return False
    S = np.abs(librosa.stft(y, n_fft=1024, hop_length=512)) ** 2
    f = librosa.fft_frequencies(sr=16000, n_fft=1024)
    tot = S.sum(0) + 1e-12
    lv = 10 * np.log10(tot)
    act = lv > np.percentile(lv, 95) - 30
    return bool(act.any() and float((S[f < 70].sum(0) / tot)[act].mean()) < 0.01)


LONG_SEC = 20 * 60     # 이보다 길면 10분 조각으로 나눠 검사한다 (한 시간짜리도 메모리가 안 터지게) [v1.8.0]
PIECE_SEC = 10 * 60


def _pieces(path, out, piece=PIECE_SEC):
    """긴 파일을 piece초짜리 wav 조각으로 잘라 둔다. [(조각 경로, 시작 초, 길이 초)]"""
    dur = probe_dur(path)
    d = os.path.join(out, "_조각")
    os.makedirs(d, exist_ok=True)
    res, k, s = [], 0, 0.0
    while s < dur - 1.0:
        p = os.path.join(d, f"{k + 1:02d}.wav")
        if not (os.path.isfile(p) and os.path.getmtime(p) >= os.path.getmtime(path)):
            subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", path, "-ss", f"{s:.3f}", "-t", str(piece), "-vn", "-ac", "2",
                            "-c:a", "pcm_f32le", p], check=True)
        res.append((p, s, min(piece, dur - s)))
        s += piece
        k += 1
    return res


def _merge_piece_stats(piece_stats, path, dur):
    """조각마다 잰 값을 하나로. 크기는 최대, 라우드니스는 조각 길이로 가중한 평균(파워), 개수는 합."""
    stats = dict(piece_stats[0][0])
    stats["파일"] = os.path.basename(path)
    stats["길이"] = fmt_t(dur) + f" ({dur:.1f}초)"
    stats["긴 파일"] = f"{len(piece_stats)}조각(10분씩)으로 나눠 검사함 — 조각 폴더 안 그림의 시각은 그 조각의 시작부터 센 것"
    for k in ("최고 크기(dBFS)", "트루 피크(dBTP)"):
        vals = [s.get(k) for s, _ in piece_stats if isinstance(s.get(k), (int, float))]
        if vals:
            stats[k] = max(vals)
    lu = [(s.get("라우드니스(LUFS)"), L) for s, L in piece_stats if isinstance(s.get("라우드니스(LUFS)"), (int, float))]
    if lu:
        tot = sum(L for _, L in lu)
        stats["라우드니스(LUFS)"] = round(10 * math.log10(sum(10 ** (v / 10) * L / tot for v, L in lu)), 1)
    for k in ("클리핑 횟수", "딸깍 소리", "큰 숨소리", "음 튐", "찾은 노트 수", "음높이를 믿기 어려운 곳(화음·잡음)"):
        vals = [s.get(k) for s, _ in piece_stats if isinstance(s.get(k), int)]
        if vals:
            stats[k] = sum(vals)
    for k in ("조(키, 추정)", "빠르기(BPM, 추정)", "음역", "전체 튜닝 치우침(센트)"):
        if k in stats and stats[k] is not None:
            stats[k] = f"{stats[k]} (첫 조각에서 잼)"
    return stats


def _check_long(args, dur):
    """긴 파일: 10분 조각마다 검사하고 결과를 합친다."""
    from argparse import Namespace
    path = args.file
    mode = detect_mode(path, args.mode)
    out = args.out or (os.path.splitext(os.path.basename(path))[0] + "_소리검사")
    os.makedirs(out, exist_ok=True)
    pieces = _pieces(path, out)
    n = len(pieces)
    own = _PROG["out"] is None
    if own:
        est = est_times(PIECE_SEC)["check_vocal" if mode == "vocal" else "check_mix"]
        progress_start(out, [(f"조각 {k + 1}/{n} 검사", est) for k in range(n)])
    all_probs, images, piece_stats, whole = [], [], [], {}
    for k, (pp, off, plen) in enumerate(pieces):
        if own:
            progress_step(f"조각 {k + 1}/{n} ({fmt_t(off)}~{fmt_t(off + plen)}) 검사")
        else:
            progress_tick(k / n, f"조각 {k + 1}/{n}", every=0)
        sub = os.path.join(out, f"조각{k + 1:02d}_{fmt_t(off).replace(':', '분')}부터")
        st, probs, imgs, _, _ = _check_one(Namespace(file=pp, mode=mode, out=sub, max_zoom=args.max_zoom, key_from=None), pages=False, quiet=True)
        drop = set()
        if k > 0:
            drop |= {"앞 무음 김", "시작이 잘림"}
        if k < n - 1:
            drop |= {"끝이 잘림"}
        for p in probs:
            if p["kind"] in drop:
                continue
            if p["kind"] == "뚝 끊김" and (p["t"] < 0.3 or p["t"] > plen - 0.3):
                continue  # 조각 경계
            q = dict(p)
            is_whole = p.get("end") is None and p["t"] == 0
            if is_whole:
                key = (p["kind"], p["level"])
                if key in whole:
                    whole[key]["_pieces"].append(k + 1)
                    continue
                q["_pieces"] = [k + 1]
                whole[key] = q
            else:
                q["t"] = p["t"] + off
                if q.get("end") is not None:
                    q["end"] = p["end"] + off
            q["조각"] = k + 1
            all_probs.append(q)
        images += imgs
        piece_stats.append((st, plen))
    for q in all_probs:
        if q.get("_pieces"):
            if len(q["_pieces"]) < n:
                q["msg"] += f" (조각 {', '.join(map(str, q['_pieces']))}에서)"
            q.pop("_pieces")
    stats = _merge_piece_stats(piece_stats, path, dur)
    stats["검사 방식"] = "보컬만" if mode == "vocal" else "완성곡(반주 포함)"
    order = {LEVEL_MUST: 0, LEVEL_CHECK: 1, LEVEL_INFO: 2}
    all_probs.sort(key=lambda p: (p["t"] if p.get("end") is not None or p["t"] > 0 else -1, order[p["level"]]))
    res = _write_check_report(out, path, mode, stats, all_probs, images)
    if own:
        progress_done()
    return res


def cmd_check(args):
    dur = probe_dur(args.file)
    if dur > LONG_SEC and not getattr(args, "no_split", False):
        _check_long(args, dur)
    else:
        _check_one(args)
    return 0


def _check_one(args, pages=True, quiet=False):
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
        check_sibilance_breath(y, ysr, probs, stats, pitch, backing=key_src if key_src and os.path.basename(key_src) == "반주.wav" else None)
    stats.pop("_scale", None)

    order = {LEVEL_MUST: 0, LEVEL_CHECK: 1, LEVEL_INFO: 2}
    probs.sort(key=lambda p: (p["t"] if p.get("end") is not None or p["t"] > 0 else -1, order[p["level"]]))

    # 그림: 전체 + 30초씩 + 문제 확대
    images = []
    p0 = os.path.join(out, "00_전체.png")
    draw_view_fixed(y, ysr, 0, dur, pitch, probs, p0, f"전체 보기 — {os.path.basename(path)}")
    images.append(p0)
    page = 30.0
    if pages and dur > page * 1.2:
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

    res = _write_check_report(out, path, mode, stats, probs, images, quiet=quiet)
    return stats, probs, images, mode, dur


def _write_check_report(out, path, mode, stats, probs, images, quiet=False):
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
                       images=images, source=os.path.abspath(path), mode=mode), f, ensure_ascii=False, indent=1, default=float)
    if not quiet:
        try:  # 한 장짜리 HTML (그림·소리 조각이 안에 들어 있어 눌러서 바로 들을 수 있음)
            build_html(out, f"소리 검사 — {os.path.basename(path)}",
                       f"꼭 고칠 것 {len(must)}곳 / 들어 보고 판단 {len(chk)}곳 / 참고 {len(inf)}곳",
                       [(k, v) for k, v in stats.items()],
                       [dict(p, src=os.path.basename(path), _audio=path) for p in probs],
                       [(os.path.basename(i), i) for i in images])
        except Exception as e:  # noqa
            print("HTML 보고서를 못 만들었습니다:", e)
        print(f"[완료] 보고서: {rp}")
        print(f"꼭 고칠 것 {len(must)} / 들어 보고 판단 {len(chk)} / 참고 {len(inf)} / 그림 {len(images)}장")
        for p in (must + chk)[:15]:
            print(f"  {p['시각']} [{p['level']}] {p['kind']}")
    return rp


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


def _check_results(path, mode, out):
    """check 결과(result.json)를 읽는다. 없거나 오래됐으면 새로 돌린다."""
    from argparse import Namespace
    rj = os.path.join(out, "result.json")
    if not (os.path.isfile(rj) and os.path.getmtime(rj) >= os.path.getmtime(path)):
        _check_one(Namespace(file=path, mode=mode, out=out, max_zoom=6, key_from=None), quiet=True)
    return json.load(open(rj, encoding="utf-8"))


def _estimate_offset(a_db, b_db, max_lag=30):
    """새판이 앞판보다 몇 초 밀렸는지(앞을 잘랐거나 붙였을 때) 1초 크기 곡선을 맞춰 본다. 양수면 새판이 늦게 시작."""
    n = min(len(a_db), len(b_db))
    if n < 10:
        return 0.0
    x = np.asarray(a_db[:n], float)
    y = np.asarray(b_db[:n], float)
    x = x - x.mean()
    y = y - y.mean()
    best, lag = -1e9, 0
    for L in range(-max_lag, max_lag + 1):
        if L >= 0:
            s = float(np.dot(x[: n - L], y[L:])) if n - L > 5 else -1e9
        else:
            s = float(np.dot(x[-L:], y[: n + L])) if n + L > 5 else -1e9
        if s > best:
            best, lag = s, L
    return float(lag)


def cmd_compare(args):
    """두 판 비교: 크기·라우드니스뿐 아니라 흠 목록까지 맞춰 '없어진 흠 / 새로 생긴 흠 / 그대로인 흠'을 표로. [v1.8.2]"""
    a, b = summarize(args.a), summarize(args.b)
    out = args.out or "소리비교"
    os.makedirs(out, exist_ok=True)
    lines = [f"# 두 판 비교", "", "VERDICT", "", f"| 항목 | 앞판 ({os.path.basename(args.a)}) | 새판 ({os.path.basename(args.b)}) |", "|---|---|---|"]
    lines.append(f"| 길이 | {fmt_t(a['dur'])} | {fmt_t(b['dur'])} |")
    lines.append(f"| 최고 크기 | {a['peak']:.1f} dBFS | {b['peak']:.1f} dBFS |")
    lines.append(f"| 라우드니스 | {a['lufs']:.1f} LUFS | {b['lufs']:.1f} LUFS |")
    # 흠 목록 맞추기
    mode = detect_mode(args.b, getattr(args, "mode", "auto"))
    off = _estimate_offset(a["sec_db"], b["sec_db"]) if not getattr(args, "no_align", False) else 0.0
    ra = _check_results(args.a, mode, os.path.join(out, "앞판검사"))
    rb = _check_results(args.b, mode, os.path.join(out, "새판검사"))
    pa = [p for p in ra.get("problems", []) if p["level"] != LEVEL_INFO]
    pb = [p for p in rb.get("problems", []) if p["level"] != LEVEL_INFO]
    whole = lambda p: p.get("end") is None and p["t"] == 0
    used = set()
    same, gone, new = [], [], []
    for p in pa:
        hit = None
        for j, q in enumerate(pb):
            if j in used or q["kind"] != p["kind"]:
                continue
            if whole(p) and whole(q):
                hit = j
            elif not whole(p) and not whole(q) and abs((q["t"] - off) - p["t"]) <= 0.6:
                hit = j
            if hit is not None:
                break
        if hit is None:
            gone.append(p)
        else:
            used.add(hit)
            same.append((p, pb[hit]))
    new = [q for j, q in enumerate(pb) if j not in used]
    n_must_a = sum(1 for p in pa if p["level"] == LEVEL_MUST)
    n_must_b = sum(1 for p in pb if p["level"] == LEVEL_MUST)
    if n_must_b == 0 and not new:
        verdict = f"좋아짐 👍 — 꼭 고칠 것 {n_must_a}곳 → 0곳, 새로 생긴 흠 없음"
    elif len(gone) > len(new):
        verdict = f"좋아짐 — 없어진 흠 {len(gone)}곳, 새로 생긴 흠 {len(new)}곳 (꼭 고칠 것 {n_must_a} → {n_must_b})"
    elif len(gone) == len(new) == 0:
        verdict = "흠 목록은 그대로 — 소리 크기만 달라졌는지 아래를 보세요"
    else:
        verdict = f"주의 ⚠️ — 새로 생긴 흠 {len(new)}곳이 없어진 흠 {len(gone)}곳보다 많음 (꼭 고칠 것 {n_must_a} → {n_must_b})"
    lines[lines.index("VERDICT")] = f"> **한마디: {verdict}**" + (f"\n>\n> 새판이 앞판보다 {off:+.0f}초 밀려 있어 시각을 맞춰 비교했습니다." if abs(off) >= 1 else "")
    mk = lambda p: "🔴" if p["level"] == LEVEL_MUST else "🟠"
    tv = lambda p: "전체" if whole(p) else fmt_t(p["t"])
    lines += ["", f"## 없어진 흠 ({len(gone)}) — 앞판에 있었는데 새판엔 없음", ""]
    lines += [f"- {mk(p)} {tv(p)} {p['kind']}" for p in gone] or ["- 없음"]
    lines += ["", f"## 새로 생긴 흠 ({len(new)}) — 새판에서 처음 보임", ""]
    lines += [f"- {mk(q)} {tv(q)} {q['kind']} — {q['msg']}" for q in new] or ["- 없음"]
    lines += ["", f"## 그대로인 흠 ({len(same)})", ""]
    lines += [f"- {mk(q)} {tv(p)} → {tv(q)} {p['kind']}" for p, q in same] or ["- 없음"]
    sa, sb = ra.get("stats", {}), rb.get("stats", {})
    rows = [(k, sa.get(k), sb.get(k)) for k in ("클리핑 횟수", "딸깍 소리", "음 튐", "큰 숨소리", "센 ㅅ·ㅊ 소리", "전체 튜닝 치우침(센트)", "음역", "조(키, 추정)") if k in sa or k in sb]
    if rows:
        lines += ["", "## 숫자로 보면", "", "| 항목 | 앞판 | 새판 |", "|---|---|---|"] + [f"| {k} | {x} | {y} |" for k, x, y in rows]
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
    lines += ["", f"그림: 크기비교.png / 각 판의 검사: `앞판검사/보고서.md`, `새판검사/보고서.md`"]
    rp = os.path.join(out, "비교.md")
    open(rp, "w", encoding="utf-8").write("\n".join(lines))
    json.dump(dict(verdict=verdict, offset=off, gone=gone, new=new, same=[dict(before=p, after=q) for p, q in same]),
              open(os.path.join(out, "compare.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1, default=float)
    print(f"[완료] {rp}")
    print("\n".join(lines[:12]))


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
    if os.environ.get("SOUND_EAR_FAST"):
        # 빠르게: 가장 작은 모델 (small은 medium보다 약 3배 빠름) [확인: 10-25]
        files = [c for c in cands if c and os.path.isfile(c)]
        if explicit and os.path.isfile(explicit):
            return explicit
        return min(files, key=os.path.getsize) if files else None
    for c in cands:
        if c and os.path.isfile(c):
            return c
    return None


def _speech_chunks(y, sr=16000, max_len=25.0):
    """조용한 틈(0.4초 넘게)에서 잘라 25초 안쪽 덩어리로 나눈다. [(시작샘플, 끝샘플)]"""
    import librosa
    hop = sr // 20
    r = db(librosa.feature.rms(y=y, frame_length=hop * 2, hop_length=hop)[0])
    if not len(r):
        return [(0, len(y))]
    quiet = r < float(np.percentile(r, 90)) - 30
    cuts = [0]
    for i, j in runs(quiet):
        if (j - i) >= 8:
            cuts.append((i + j) // 2 * hop)
    cuts.append(len(y))
    out, a, lim = [], 0, int(max_len * sr)
    while a < len(y):
        nxt = [c for c in cuts if a + sr * 2 < c <= a + lim]
        b = max(nxt) if nxt else min(len(y), a + lim)
        out.append((a, b))
        a = b
    # 거의 무음인 덩어리는 버림
    keep = []
    for a, b in out:
        seg = y[a:b]
        if len(seg) > sr * 0.5 and db(np.sqrt(np.mean(seg ** 2))) > float(np.percentile(r, 90)) - 35:
            keep.append((a, b))
    return keep


def pick_language(y, model, sr=16000):
    """노래가 가장 빽빽한 30초 구간 세 곳으로 언어를 정한다.
    whisper.cpp는 처음 30초로만 언어를 정해서, 전주가 길면 엉뚱한 언어로 받아쓴다."""
    hop = sr
    e = np.array([np.sqrt(np.mean(y[i:i + hop] ** 2)) for i in range(0, max(1, len(y) - hop), hop)])
    if len(e) < 5:
        return ""
    w = min(30, len(e))
    score = np.convolve(e, np.ones(w), "valid")
    picks = []
    for _ in range(3):
        k = int(np.argmax(score))
        if score[k] <= 0:
            break
        picks.append(k)
        score[max(0, k - w):k + w] = -1
    tot = {}
    for k in picks:
        try:
            (_, _), probs = model.auto_detect_language(np.ascontiguousarray(y[k * hop:(k + w) * hop], dtype=np.float32))
        except Exception:
            continue
        for lg, pr in probs.items():
            tot[lg] = tot.get(lg, 0) + float(pr)
    return max(tot, key=tot.get) if tot else ""


def transcribe(path, model_path, lang="auto", threads=None, prompt=None, chunks=False, info=None):
    """whisper.cpp로 받아쓰기. 16kHz 모노로 풀어서 넣는다. [(시작초, 끝초, 글)]
    chunks=True: 조용한 틈에서 25초 안쪽으로 잘라 따로 받아쓴다(분리한 목소리용 — 긴 빈 곳 뒤를 통째로 놓치던 것을 막음)."""
    from pywhispercpp.model import Model
    r = subprocess.run(["ffmpeg", "-v", "error", "-i", path, "-vn", "-f", "f32le", "-ac", "1", "-ar", "16000", "-"],
                       capture_output=True)
    y = np.frombuffer(r.stdout, np.float32).copy()
    # suppress_nst(말이 아닌 표시 막기)+no_context(앞 줄에 끌려가지 않기): 노래에서 "(♪~♪)", "[블랙핑크 - …]"만 내던 것을 막음
    # [확인: 10-25 medium으로 일본어 「구마」 앞 45초가 '(♪~♪)' 두 줄 → 이 설정으로 가사 7줄, 「Alone」 후렴도 더 정확]
    params = dict(print_progress=False, print_realtime=False,
                  n_threads=threads or max(1, os.cpu_count() or 1), suppress_blank=True, no_context=True, suppress_nst=True)
    if prompt:
        params["initial_prompt"] = prompt
    m = Model(model_path, redirect_whispercpp_logs_to=None, **params)
    if lang in (None, "", "auto"):
        lang = pick_language(y, m) or ""
    if info is not None:
        info["lang"] = lang or "auto"
    pieces = _speech_chunks(y) if chunks else [(0, len(y))]
    out = []
    total = max(1.0, len(y) / 16000.0)
    for a, b in pieces:
        def _cb(sg, _a=a):  # 진행 표시: 받아쓴 데까지 / 전체 길이
            try:
                done = (_a / 16000.0 + sg.t1 / 100.0)
                progress_tick(min(0.99, done / total), f"{fmt_t(done)}/{fmt_t(total)}")
            except Exception:
                pass
        try:
            segs_ = m.transcribe(np.ascontiguousarray(y[a:b]), language=lang, new_segment_callback=_cb)
        except TypeError:
            segs_ = m.transcribe(np.ascontiguousarray(y[a:b]), language=lang)
        for sg in segs_:
            txt = sg.text.strip()
            if not txt:
                continue
            out.append((a / 16000 + sg.t0 / 100.0, a / 16000 + sg.t1 / 100.0, txt))
    return out


def clean_segments(segs):
    """whisper가 음악에서 자주 지어내는 것(같은 줄 반복, 자막 인사, 음표만)을 걸러 표시한다."""
    import re
    junk = re.compile(r"(구독|좋아요|시청해 주셔서|시청해주셔서|자막|MBC|KBS|SBS|thank you for watching|subscribe|ご視聴|字幕|노래가 정말 잘 어울려|한국어 노래|제가 제일 좋아하는|이 노래는|^감사합니다\.?$|^\[?음악\]?$|【音楽】)", re.I)
    res = []
    prev = None
    rep = 0
    for a, b, t in segs:
        flag = ""
        if junk.search(t):
            flag = "지어낸 말일 수 있음"
        elif re.search(r"(.)\1{5,}", t.replace(" ", "")):  # 후우우우우우 같은 늘어진 소리
            flag = "늘어진 소리(가사 아님)"
        if re.fullmatch(r"[\s♪♫~.\-…()\[\]]*", t) or re.fullmatch(r"\s*[\[(（【][^\])）】]{0,40}[\])）】]\s*", t):
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


def write_srt(path, segs):
    def ts(x):
        h = int(x // 3600)
        m_ = int(x % 3600 // 60)
        s_ = x % 60
        return f"{h:02d}:{m_:02d}:{int(s_):02d},{int((s_ % 1) * 1000):03d}"
    rows = []
    for i, (a, b, t) in enumerate(segs, 1):
        rows += [str(i), f"{ts(a)} --> {ts(b)}", t, ""]
    open(path, "w", encoding="utf-8").write("\n".join(rows))


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
    winfo = {}
    out = args.out or (os.path.splitext(os.path.basename(args.file))[0] + "_가사")
    os.makedirs(out, exist_ok=True)
    if _PROG["out"] is None:
        progress_start(out, [("받아쓰기", est_times(probe_dur(src), model_big="medium" in model or "large" in model)["words"])])
        progress_step("받아쓰기")
    segs = transcribe(src, model, lang=args.lang, prompt=args.prompt, chunks=looks_vocal_only(src), info=winfo)
    progress_done()
    took = time.time() - st
    if tmp:
        os.remove(tmp)
    segs = clean_segments([(a + t0, b + t0, t) for a, b, t in segs])
    out = args.out or (os.path.splitext(os.path.basename(args.file))[0] + "_가사")
    os.makedirs(out, exist_ok=True)
    lines = [f"# 받아쓰기 — {os.path.basename(args.file)}", "",
             f"- 모델: {os.path.basename(model)} / 언어: {winfo.get('lang', args.lang)} / 걸린 시간: {took:.0f}초",
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
        progress_tick((k + 1) / total, f"보컬 분리 {k + 1}/{total}", every=8.0)
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
    from scipy.ndimage import median_filter
    D = librosa.stft(y.astype(np.float32), n_fft=2048, hop_length=hop)  # float32: 10분 조각에서 메모리 3.6GB → 절반쯤 [v1.8.0]
    S = np.abs(D).astype(np.float32)
    freqs = librosa.fft_frequencies(sr=sr, n_fft=2048)
    # HPSS는 느려서(4분 곡 한 번에 약 15초) 중앙값 거르기를 한 번만 하고 두 가지 세기(margin 2, 3)로 나눠 쓴다
    # [확인: 10-25 곡 설명에서 HPSS를 세 번 돌려 46초 걸리던 것]
    hm = median_filter(S, size=(1, 31), mode="reflect")
    pm = median_filter(S, size=(31, 1), mode="reflect")
    sm = librosa.util.softmask
    H = S * sm(hm, pm * 2.0, power=2)
    P = S * sm(pm, hm * 2.0, power=2)
    yh = librosa.istft(D * sm(hm, pm * 3.0, power=2), hop_length=hop, length=len(y))
    yp = librosa.istft(D * sm(pm, hm * 3.0, power=2), hop_length=hop, length=len(y))
    tot = (S ** 2).sum(axis=0) + 1e-12
    perc = (P ** 2).sum(axis=0) / tot
    low = (S[freqs < 150] ** 2).sum(axis=0) / tot
    mid = (S[(freqs >= 200) & (freqs < 3000)] ** 2).sum(axis=0) / tot
    cent = librosa.feature.spectral_centroid(S=S, sr=sr)[0]
    rms = librosa.feature.rms(S=S, frame_length=2048)[0]
    onset = librosa.onset.onset_strength(S=librosa.amplitude_to_db(P, ref=np.max), sr=sr)
    return dict(S=S, H=H, perc=perc, low=low, mid=mid, cent=cent, rms=rms, onset=onset[: len(rms)], yh=yh, yp=yp)


def _segments(y, sr, hop, dur, yp=None):
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
    # 드럼이 들어오고 빠지는 곳도 '소리가 바뀌는 곳'으로 본다 [확인: 10-25 합성 시험곡에서 드럼 없는 전주가 첫 절에 묻힘]
    if yp is None:
        yp = librosa.effects.percussive(y, margin=3.0)
    prm = librosa.amplitude_to_db(librosa.feature.rms(y=yp, hop_length=hop), ref=np.max)
    PR = librosa.util.sync(prm, beats, aggregate=np.mean)
    bt = librosa.frames_to_time(np.concatenate([[0], beats]), sr=sr, hop_length=hop)
    n = C.shape[1]
    beat_s = dur / max(n, 1)

    def zs(X):
        return (X - X.mean(axis=1, keepdims=True)) / (X.std(axis=1, keepdims=True) + 1e-9)
    F = np.vstack([zs(C), zs(M) * 0.8, zs(E) * 1.5, zs(PR) * 2.0])
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
    want = int(np.clip(round(dur / 14.0), 5, 22))  # 짧은 곡도 전주·절·후렴을 다 나눌 수 있게 최소 5덩어리
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
    own = _PROG["out"] is None
    if own:
        dd = probe_dur(path)
        mm = None if args.no_words else find_whisper_model(args.model)
        E = est_times(dd, model_big=bool(mm and ("medium" in mm or "large" in mm)))
        progress_start(out, [("곡 설명" + (" (가사 받아쓰기 포함)" if mm and not args.words else ""), E["describe"] + (E["words"] if mm and not args.words else 0))])
        progress_step("곡 설명")
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
    vocal_path = getattr(args, "vocal", None) if getattr(args, "vocal", None) and os.path.isfile(args.vocal) else None
    if vocal_path is None and not args.no_separate:
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
                # 조용한 틈에서 25초 안쪽으로 잘라 따로 받아쓴다 — 통째로 넣으면 첫 빈 곳 뒤를 다 놓침
                # [확인: 10-25 일본어 「구마」 분리 목소리 medium: 통째로 1줄(지어낸 말) → 잘라서 37줄, 345초 → 317초]
                linfo = {}
                ws = clean_segments(transcribe(vocal_path, model, lang=args.lang, chunks=True, info=linfo, prompt=getattr(args, 'prompt', None)))
                glang = linfo.get("lang", args.lang)
                words = list(ws)
                # 목소리는 나는데 받아쓴 말이 없는 빈 곳(3초 넘게)만 원곡에서 잘라 다시 받아쓴다
                vy_, vsr_ = librosa.load(vocal_path, sr=16000, mono=True)
                vr_ = db(librosa.feature.rms(y=vy_, frame_length=1600, hop_length=800)[0])
                live = vr_ > float(np.percentile(vr_, 90)) - 14
                tt_ = np.arange(len(vr_)) * 0.05
                covered = np.zeros(len(vr_), bool)
                for a, b, t, fl in ws:
                    if not fl:
                        covered |= (tt_ >= a - 0.5) & (tt_ <= b + 0.5)
                gaps = [(tt_[i], tt_[min(j, len(tt_) - 1)]) for i, j in runs(live & ~covered) if (j - i) * 0.05 >= 3.0]
                if gaps:
                    print(f"빈 곳 {len(gaps)}군데를 원곡에서 다시 받아쓰는 중…")
                for ga, gb in gaps:
                    clip = os.path.join(out, "_gap.wav")
                    subprocess.run(["ffmpeg", "-v", "error", "-y", "-ss", f"{max(0, ga - 0.5):.2f}", "-to", f"{gb + 0.5:.2f}", "-i", path,
                                    "-ac", "1", "-ar", "16000", clip], check=True)
                    for a, b, t, fl in clean_segments(transcribe(clip, model, lang=glang)):
                        a += max(0, ga - 0.5)
                        b += max(0, ga - 0.5)
                        if not any(min(b, b2) - max(a, a2) > 0.3 * (b - a) for a2, b2, _, _ in ws):
                            words.append((a, b, t, fl or "원곡에서 받아씀"))
                    os.remove(clip)
                words.sort(key=lambda w: w[0])
            else:
                words = clean_segments(transcribe(path, model, lang=args.lang, prompt=getattr(args, 'prompt', None)))
            json.dump([dict(start=a, end=b, text=t, flag=fl) for a, b, t, fl in words],
                      open(os.path.join(out, "words.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
            write_srt(os.path.join(out, "자막.srt"), [(a, b, t) for a, b, t, fl in words if not fl or fl == "원곡에서 받아씀"])
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

    segs, tempo, ssim = _segments(y, sr, hop, dur, yp=f["yp"])
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
    yh, yp = f["yh"], f["yp"]
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
                    # 구간 전체는 달라도 '같은 줄'(훅)이 양쪽에 있으면 후렴으로 본다
                    # [확인: 10-25 「Alone」 G 부분 — 끝 두 줄 '수천 개의 해와 달이 돌고', '흩어져 가는 감정'이 1:44·3:45에 반복]
                    for la in rs[i]["words"].split(" / "):
                        for lb in rs[j]["words"].split(" / "):
                            A2, B2 = bigr(la), bigr(lb)
                            if len(la) >= 8 and A2 and B2 and len(A2 & B2) / len(A2 | B2) >= 0.5:
                                sims.append(0.4)
            # 후렴은 '다른 부분을 지나서 돌아오는' 반복이다. 바로 붙어 있는 반복(같은 절 안 두 줄)은 한 번으로 센다
            # [확인: 10-25 「Alone」 0:45~1:14 '울지 말라고'가 연달아 두 번 → 후렴으로 잘못 고름]
            returns = 1
            for i in range(1, len(rs)):
                if rs[i]["a"] - rs[i - 1]["b"] > 15:
                    returns += 1
            if sims and max(sims) >= 0.35 and returns >= 2:
                sc = np.mean([x["e"] for x in rs]) + 3 * max(sims) + 1.5 * returns
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
    # 한 글자가 5번 넘게 나오면(피아노 발라드처럼 절과 후렴이 같은 글자로 묶임) 훅 가사가 겹치고 세기도 안 작은 것만 후렴으로 본다
    # [확인: 10-25 UNI 「목성은 사랑을 한다」 B가 7번 → 전부 '후렴'으로 나옴]
    chorus_rows = None
    if chorus and len(by.get(chorus, [])) >= 5:
        rs = by[chorus]
        med_e = float(np.median([x["e"] for x in rs]))

        def lines_of(r):
            return [la for la in r["words"].split(" / ") if len(la) >= 8]
        hook = set()
        for i in range(len(rs)):
            for j in range(len(rs)):
                if i == j or abs(rs[i]["a"] - rs[j]["a"]) < 15:
                    continue
                if any(bigr(la) and bigr(lb) and len(bigr(la) & bigr(lb)) / len(bigr(la) | bigr(lb)) >= 0.5
                       for la in lines_of(rs[i]) for lb in lines_of(rs[j])):
                    hook.add(id(rs[i]))
        chorus_rows = {id(r) for r in rs if id(r) in hook and r["e"] >= med_e - 1.0}
        if len(chorus_rows) < 2:
            chorus_rows = None
    count = defaultdict(int)
    total = {lab: sum(1 for r in rows if r["lab"] == lab) for lab in set(r["lab"] for r in rows)}
    ch_total = len(chorus_rows) if chorus_rows else total.get(chorus, 0)
    ch_k = 0
    names = []
    for i, r in enumerate(rows):
        count[r["lab"]] += 1
        k = count[r["lab"]]
        if not sings(r):
            before_first = all(not sings(x) for x in rows[:i])
            nm = "전주" if before_first else ("후주(끝)" if i == len(rows) - 1 else "간주(노래 없음)")
        elif r["lab"] == chorus and (chorus_rows is None or id(r) in chorus_rows):
            ch_k += 1
            nm = f"후렴 {ch_k}번째" if ch_total > 1 else "후렴"
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
    if own:
        progress_done()
    print(f"[완료] {rp}\n       그림: {ip}")
    print("\n".join(lines[:4 + 4 + len(rows)]))


# ---------------------------------------------------------------- 한 번에 다 듣기
def _shift_probs(probs, off, plen, k, n, src, audio, whole_seen):
    """조각 하나의 문제 목록을 전체 시각으로 옮긴다. 조각 경계에서 생기는 헛짚음은 뺀다."""
    out = []
    drop = set()
    if k > 0:
        drop |= {"앞 무음 김", "시작이 잘림"}
    if k < n - 1:
        drop |= {"끝이 잘림"}
    for p in probs:
        if n > 1 and p["kind"] in drop:
            continue
        if n > 1 and p["kind"] == "뚝 끊김" and (p["t"] < 0.3 or p["t"] > plen - 0.3):
            continue
        q = dict(p)
        is_whole = p.get("end") is None and p["t"] == 0
        q["src"] = src
        q["_audio"] = audio
        q["_at"] = p["t"]
        if is_whole:
            key = (src, p["kind"], p["level"])
            if key in whole_seen:
                whole_seen[key]["_pieces"].append(k + 1)
                continue
            q["_pieces"] = [k + 1]
            whole_seen[key] = q
        else:
            q["t"] = p["t"] + off
            if q.get("end") is not None:
                q["end"] = p["end"] + off
            q["시각"] = fmt_t(q["t"])
        out.append(q)
    return out


def _full_one(args, out, path, sep_model, model, top=True):
    """한 파일(또는 한 조각)에 대해: 분리 → 흠 찾기 → 목소리 음정 → 곡 설명. 결과 json들을 읽어 dict로 돌려준다."""
    from argparse import Namespace
    os.makedirs(out, exist_ok=True)
    step = progress_step if top else (lambda name: progress_tick(0.0, name, every=0))
    stems = os.path.join(out, "분리")
    vocal = inst = None
    vocal_only = False
    if args.deep and not sep_model and top:
        print("※ 보컬 분리 모델이 없어 목소리 검사는 건너뜁니다. `setup`으로 준비하세요.")
    if sep_model and not args.no_separate:
        step("목소리와 반주 가르기")
        vp, ip = os.path.join(stems, "보컬.wav"), os.path.join(stems, "반주.wav")
        if not (os.path.isfile(vp) and os.path.getmtime(vp) >= os.path.getmtime(path)):
            vp, ip = separate_vocals(path, sep_model, stems)
        vocal, inst = vp, ip
        import soundfile as sf
        a, _ = sf.read(ip, dtype="float32")
        b, _ = sf.read(vp, dtype="float32")
        ra, rb = float(np.sqrt(np.mean(a ** 2))), float(np.sqrt(np.mean(b ** 2)))
        vocal_only = rb > 0 and 20 * np.log10((ra + 1e-9) / rb) < -18
    step("흠 찾기 (원곡)")
    mode1 = "vocal" if vocal_only else detect_mode(path, args.mode)
    _check_one(Namespace(file=path, mode=mode1, out=os.path.join(out, "1_원곡검사"), max_zoom=8, key_from=None), quiet=not top)
    if vocal and not vocal_only:
        step("음정·음 튐 검사 (분리한 목소리)")
        _check_one(Namespace(file=vocal, mode="vocal", out=os.path.join(out, "2_목소리검사"), max_zoom=8, key_from=inst), quiet=not top)
    step("곡 설명 (구성·가사·악기·분위기)")
    wj = os.path.join(out, "3_곡설명", "words.json")
    reuse = wj if os.path.isfile(wj) and os.path.getmtime(wj) >= os.path.getmtime(path) and not args.redo_words else None
    hint = None
    if getattr(args, "lyrics", None) and os.path.isfile(args.lyrics):
        hint = " ".join(_read_lyrics(args.lyrics)[:6])[:200]  # 내 가사 앞부분을 힌트로 → 고유명사를 덜 틀림
    cmd_describe(Namespace(file=path, model=args.model, lang=args.lang, words=reuse, no_words=args.no_words, out=os.path.join(out, "3_곡설명"),
                           essentia=None, no_ai=args.no_ai, no_separate=True, sep_model=None, vocal=vocal, prompt=hint))

    def rj(p):
        try:
            return json.load(open(p, encoding="utf-8"))
        except Exception:
            return None
    words = []
    if os.path.isfile(wj):
        words = [(w["start"], w["end"], w["text"], w.get("flag", "")) for w in rj(wj) or []]
    return dict(c1=rj(os.path.join(out, "1_원곡검사", "result.json")) or {},
                c2=rj(os.path.join(out, "2_목소리검사", "result.json")) or {},
                d=rj(os.path.join(out, "3_곡설명", "describe.json")) or {},
                words=words, vocal=vocal, vocal_only=vocal_only, out=out, path=path)


def cmd_full(args):
    """흠 찾기(완성곡) → 목소리 분리 → 음정 검사(목소리) → 곡 설명 → (가사 대조) → 종합 보고서 한 장 + HTML."""
    import time
    path = args.file
    out = args.out or (os.path.splitext(os.path.basename(path))[0] + "_전체듣기")
    os.makedirs(out, exist_ok=True)
    t0 = time.time()
    dur = probe_dur(path)
    model = None if args.no_words else find_whisper_model(args.model)
    big = bool(model and ("medium" in model or "large" in model))
    sep_model = find_sep_model(args.sep_model) if args.deep else None
    E = est_times(dur, model_big=big)
    long_mode = dur > LONG_SEC and not getattr(args, "no_split", False)
    if long_mode:
        pieces = _pieces(path, out)
        n = len(pieces)
        Ep = est_times(PIECE_SEC, model_big=big)
        per = Ep["check_mix"] + Ep["describe"] + (Ep["words"] if model else 0) + ((Ep["separate"] + Ep["check_vocal"]) if sep_model else 0)
        steps = [(f"조각 {k + 1}/{n}", per) for k in range(n)]
    else:
        pieces = [(path, 0.0, dur)]
        steps = []
        if sep_model and not args.no_separate:
            steps += [("목소리와 반주 가르기", E["separate"])]
        steps += [("흠 찾기 (원곡)", E["check_mix"])]
        if sep_model and not args.no_separate:
            steps += [("음정·음 튐 검사 (분리한 목소리)", E["check_vocal"])]
        steps += [("곡 설명 (구성·가사·악기·분위기)", E["describe"] + (E["words"] if model else 0))]
    if getattr(args, "lyrics", None):
        steps += [("가사 대조", E["lyrics"])]
    steps += [("종합 보고서", E["html"])]
    progress_start(out, steps)
    print(f"예상 시간: 약 {_fmt_dur(sum(e for _, e in steps))} (2코어 기준, 파일 {fmt_t(dur)})", flush=True)

    from argparse import Namespace
    try:
        cmd_origin(Namespace(file=path, out=os.path.join(out, "0_출처"), quiet=True))
    except Exception as e:  # noqa
        print("출처 검사를 못 했습니다:", e)
    parts = []
    for k, (pp, off, plen) in enumerate(pieces):
        if long_mode:
            progress_step(f"조각 {k + 1}/{n} ({fmt_t(off)}~{fmt_t(off + plen)})")
            sub = os.path.join(out, f"조각{k + 1:02d}_{fmt_t(off).replace(':', '분')}부터")
            r = _full_one(args, sub, pp, sep_model, model, top=False)
        else:
            r = _full_one(args, out, pp, sep_model, model, top=True)
        parts.append((off, plen, r))

    # 가사 대조 (내가 쓴 가사가 있으면)
    lyr = None
    if getattr(args, "lyrics", None):
        progress_step("가사 대조")
        if not os.path.isfile(args.lyrics):
            print("※ 가사 파일이 없습니다:", args.lyrics)
        else:
            ref = _read_lyrics(args.lyrics)
            words_all = []
            for off, plen, r in parts:
                words_all += [(a + off, b + off, t, fl) for a, b, t, fl in r["words"]]
            if not words_all:
                print("※ 받아쓴 가사가 없어 가사 대조를 못 합니다(받아쓰기 모델이 필요).")
            else:
                lyr = compare_lyrics(words_all, ref)
                ld = os.path.join(out, "4_가사대조")
                os.makedirs(ld, exist_ok=True)
                mn = os.path.basename(model) if model else "-"
                LL = lyrics_report(lyr, os.path.basename(path), mn, audio_path=path, out=ld)
                open(os.path.join(ld, "가사대조.md"), "w", encoding="utf-8").write("\n".join(LL))
                json.dump(dict(overall=lyr["overall"], lines=[{k: v for k, v in r.items() if k != "snippet"} for r in lyr["lines"]],
                               extra=[dict(start=a, end=b, text=t) for a, b, t in lyr["extra"]]),
                          open(os.path.join(ld, "lyrics.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1, default=float)

    progress_step("종합 보고서")
    _write_full_report(out, path, parts, args, dur, lyr, t0, long_mode, sep_model is not None)
    progress_done()


def _write_full_report(out, path, parts, args, dur, lyr, t0, long_mode, deep_ready):
    import time
    n = len(parts)
    # 원곡·목소리 문제 모으기(조각이면 시각을 옮김)
    items, whole = [], {}
    piece_stats1, piece_stats2 = [], []
    sections, imgs = [], []
    for k, (off, plen, r) in enumerate(parts):
        c1, c2, d = r["c1"], r["c2"], r["d"]
        if c1.get("stats"):
            piece_stats1.append((c1["stats"], plen))
        if c2.get("stats"):
            piece_stats2.append((c2["stats"], plen))
        items += _shift_probs(c1.get("problems", []), off, plen, k, n, "원곡", r["path"] if n == 1 else path, whole)
        if n > 1:  # 조각이면 원곡 시각은 전체 파일 기준이라 원본에서 자른다
            for q in items:
                if q["src"] == "원곡" and q.get("_audio") == path:
                    q["_at"] = q["t"]
        voc_items = []
        for p in c2.get("problems", []):
            if p["kind"] in ("앞 무음 김", "끝이 잘림", "시작이 잘림", "트루 피크 넘침", "너무 큼", "너무 작음", "소리가 0dB까지 꽉 참", "클리핑(소리 깨짐)"):
                continue  # 분리한 목소리의 크기·앞뒤는 원곡에서 본다
            if p["kind"] in ("딸깍·틱 소리", "노트 끝 틱", "틱이 몰린 구간") and p["level"] == LEVEL_MUST:
                # 분리한 목소리의 틱은 분리하다 남은 소리일 때가 많다 [확인: 10-25 구마 커버 1:30.3 — 그림에 아무것도 안 보임]
                p = dict(p, level=LEVEL_CHECK, msg=p["msg"] + " (분리한 목소리에서만 보임 — 분리하다 남은 소리일 수 있으니 원곡에서 들어 보세요)")
            voc_items.append(p)
        items += _shift_probs(voc_items, off, plen, k, n, "목소리", r.get("vocal"), whole)
        for s in d.get("sections", []):
            s2 = dict(s)
            s2["start"] += off
            s2["end"] += off
            if n > 1:
                s2["name"] = f"[{k + 1}] " + s2["name"]
            sections.append(s2)
        base = r["out"]
        ov = os.path.join(base, "1_원곡검사", "00_전체.png")
        if os.path.isfile(ov):
            imgs.append((f"전체 보기{'' if n == 1 else f' — 조각 {k + 1} ({fmt_t(off)}~)'}", ov))
    for q in items:
        if q.get("_pieces"):
            if len(q["_pieces"]) < n:
                q["msg"] += f" (조각 {', '.join(map(str, q['_pieces']))}에서)"
            q.pop("_pieces")
    st1 = piece_stats1[0][0] if n == 1 and piece_stats1 else (_merge_piece_stats(piece_stats1, path, dur) if piece_stats1 else {})
    st2 = piece_stats2[0][0] if n == 1 and piece_stats2 else (_merge_piece_stats(piece_stats2, path, dur) if piece_stats2 else {})
    d0 = parts[0][2]["d"]
    vocal_only = any(r["vocal_only"] for _, _, r in parts)
    # 같은 종류의 🟠가 많으면 한 줄로 묶는다(완성곡의 딸깍은 하이햇·자음일 때가 많아 표만 길어짐) [확인: 10-25 「마법」 빠른 검사 딸깍 8줄]
    grouped = []
    for src in ("원곡", "목소리"):
        for kind in ("딸깍·틱 소리", "노트 끝 틱", "뚝 끊김", "음이 순간 튐"):
            same = [p for p in items if p["src"] == src and p["kind"] == kind and p["level"] == LEVEL_CHECK]
            if len(same) > 3:
                ts_ = ", ".join(p.get("시각", fmt_t(p.get("t", 0))) for p in same[:12]) + (" …" if len(same) > 12 else "")
                for p in same:
                    items.remove(p)
                grouped.append(dict(src=src, kind=f"{kind} {len(same)}곳", level=LEVEL_CHECK, t=same[0].get("t", 0), 시각="여러 곳",
                                    msg=same[0]["msg"] + f" 시각: {ts_}", _audio=same[0].get("_audio"), _at=same[0].get("_at", 0)))
    items += grouped
    order = {LEVEL_MUST: 0, LEVEL_CHECK: 1, LEVEL_INFO: 2}
    items.sort(key=lambda p: (order.get(p["level"], 3), p.get("t", 0)))
    n_must = sum(1 for p in items if p["level"] == LEVEL_MUST)
    n_chk = sum(1 for p in items if p["level"] == LEVEL_CHECK)

    L = [f"# 종합 보고서 — {os.path.basename(path)}", "", "VERDICT", ""]
    L.append(f"- 걸린 시간: {time.time() - t0:.0f}초 / " + ("자세히 듣기(--deep)" if args.deep else "빠른 검사 — 목소리만 갈라 음정까지 보려면 `--deep`")
             + (" / 목소리만 있는 파일로 보고 분리 없이 검사함" if vocal_only else "")
             + (f" / 긴 파일이라 {n}조각(10분씩)으로 나눠 검사함" if n > 1 else ""))
    L += ["", "## 한눈에", ""]
    L.append(f"- **느낌**: {d0.get('mood', '-')}" + (" (첫 조각 기준)" if n > 1 else ""))
    L.append(f"- **길이** {st1.get('길이', '-')} / **조** {d0.get('key') or st1.get('조(키, 추정)', '-')} / **빠르기** 약 {d0.get('tempo', 0):.0f} BPM / **크기** {st1.get('라우드니스(LUFS)', '-')} LUFS")
    ai = d0.get("ai") or {}
    summary = [("느낌", d0.get("mood", "-")), ("길이", st1.get("길이", "-")), ("조", d0.get("key") or st1.get("조(키, 추정)", "-")),
               ("빠르기", f"약 {d0.get('tempo', 0):.0f} BPM"), ("크기", f"{st1.get('라우드니스(LUFS)', '-')} LUFS")]
    if ai:
        g = ", ".join(genre_ko(x[0]) for x in (ai.get("genre") or [])[:3])
        m_ = ", ".join(MOOD_KO.get(x[0], x[0]) for x in (ai.get("mood") or [])[:3])
        i_ = ", ".join(INST_KO.get(x[0], x[0]) for x in (ai.get("instruments") or [])[:5])
        L.append(f"- **장르** {g or '-'} / **분위기** {m_ or '-'} / **악기** {i_ or '(목소리만)'}")
        summary += [("장르(AI)", g or "-"), ("분위기(AI)", m_ or "-"), ("악기(AI)", i_ or "(목소리만)")]
    if st2:
        L.append(f"- **노래 음역** {st2.get('음역', '-')} / 튜닝 치우침 {st2.get('전체 튜닝 치우침(센트)', '-')}센트")
        summary += [("노래 음역", st2.get("음역", "-")), ("튜닝 치우침", f"{st2.get('전체 튜닝 치우침(센트)', '-')}센트")]
    try:
        og = json.load(open(os.path.join(out, "0_출처", "origin.json"), encoding="utf-8"))
        L.append(f"- **출처**: {og['verdict']}" + (" (자세히: `0_출처/출처.md`)" if og.get("found") else ""))
        summary += [("출처", og["verdict"])]
    except Exception:
        pass
    if sections:
        parts_, prev = [], None
        for sx in sections:
            nm_ = sx["name"].split(" (")[0]
            if nm_ != prev:
                parts_.append(f"{fmt_t(sx['start'])} {nm_}")
            prev = nm_
        if len(parts_) > 14:
            parts_ = parts_[:14] + [f"… (모두 {len(sections)}구간 — 보고서.html의 곡 흐름 표에)"]
        L.append("- **구성**: " + " → ".join(parts_))
    if lyr:
        bad = sum(1 for r in lyr["lines"] if r["level"] == "bad")
        chk = sum(1 for r in lyr["lines"] if r["level"] == "check")
        L.append(f"- **가사 대조**: 전체 일치 {lyr['overall']:.0%} — 많이 다른 줄 {bad}, 조금 다른 줄 {chk} (자세히: `4_가사대조/가사대조.md`)")
        summary += [("가사 대조", f"일치 {lyr['overall']:.0%} — 많이 다른 줄 {bad}, 조금 다른 줄 {chk}")]
    L += ["", f"## 고칠 것 — 꼭 {n_must}곳 / 들어 보고 판단 {n_chk}곳", ""]
    if n_must == 0 and n_chk == 0:
        verdict = "크게 고칠 곳 없음 👍"
    elif n_must == 0:
        verdict = f"꼭 고칠 곳은 없음 👍 — 들어 보고 판단할 곳 {n_chk}곳만 귀로 확인해 보세요"
    elif n_must <= 3:
        verdict = f"꼭 고칠 곳 {n_must}곳 🔧 — 아래 표 맨 위부터 고치면 됩니다"
    else:
        verdict = f"고칠 곳이 많음 ⚠️ (꼭 {n_must}곳) — 아래 표 맨 위부터 하나씩 보세요"
    if lyr:
        bad = sum(1 for r in lyr["lines"] if r["level"] == "bad")
        if bad:
            verdict += f" / 가사가 다르게 들린 줄 {bad}줄 🎤"
    L[L.index("VERDICT")] = f"> **한마디: {verdict}**" + (
        "\n>\n> 🟠 '들어 보고 판단'은 잘 만든 곡에서도 흔히 나옵니다(일부러 넣은 효과·악기 소리일 때가 많음). 귀로 듣고 괜찮으면 넘어가세요." if n_chk else "")
    if not [p for p in items if p["level"] != LEVEL_INFO]:
        L.append("- 크게 고칠 곳이 없습니다 🎉")
    L += ["| 급함 | 시각 | 어디서 | 무엇 | 설명 |", "|---|---|---|---|---|"]
    for p in items:
        if p["level"] == LEVEL_INFO and not args.all:
            continue
        mark = "🔴" if p["level"] == LEVEL_MUST else ("🟠" if p["level"] == LEVEL_CHECK else "⚪")
        L.append(f"| {mark} | {p.get('시각', '')} | {p['src']} | {p['kind']} | {p['msg']} |")
    n_info = sum(1 for p in items if p["level"] == LEVEL_INFO)
    if n_info and not args.all:
        L.append(f"\n참고 {n_info}곳은 각 검사 보고서에 있습니다.")
    if lyr:
        L += ["", "## 가사 대조 (내가 쓴 가사 ↔ 들린 것)", ""]
        rows = [r for r in lyr["lines"] if r["level"] != "ok"]
        if not rows:
            L.append("- ✅ 모든 줄이 잘 들렸습니다")
        else:
            L += ["| 정도 | 시각 | 내가 쓴 가사 | 들린 것 |", "|---|---|---|---|"]
            for r in rows[:20]:
                ts = f"{fmt_t(r['t0'])}~{fmt_t(r['t1'])}" if r["t0"] is not None else "(못 찾음)"
                L.append(f"| {'🔴' if r['level'] == 'bad' else '🟠'} | {ts} | {_mark_wrong(r['line'], r['wrong'])} | {r['heard'] or '-'} |")
            if len(rows) > 20:
                L.append(f"\n나머지 {len(rows) - 20}줄은 `4_가사대조/가사대조.md`에 있습니다.")
    sub = "" if n == 1 else "조각NN_…/"
    L += ["", "## 자세한 결과", "",
          f"- 보고서 HTML 한 장(그림·소리 조각 포함, 브라우저로 열기): `보고서.html`",
          f"- 원곡 흠 찾기: `{sub}1_원곡검사/보고서.md` (그림: `{sub}1_원곡검사/00_전체.png`, `확대/`)"]
    if st2:
        L.append(f"- 목소리 음정 검사: `{sub}2_목소리검사/보고서.md` (음높이 그림 포함)")
    L.append(f"- 곡 설명·가사: `{sub}3_곡설명/곡설명.md`, 흐름 그림 `{sub}3_곡설명/곡흐름.png`, 자막 파일 `{sub}3_곡설명/자막.srt`")
    if lyr:
        L.append("- 가사 대조: `4_가사대조/가사대조.md`, 틀린 줄 소리 조각 `4_가사대조/가사조각/`")
    if deep_ready:
        L.append(f"- 갈라낸 소리: `{sub}분리/보컬.wav`, `{sub}분리/반주.wav`")
    if n > 1:
        L.append("- 조각 폴더 안 그림의 시각은 그 조각의 시작부터 센 것입니다(폴더 이름에 시작 시각이 있음).")
    L.append("- 출처 검사(꼬리표·C2PA 서명): `0_출처/출처.md`")
    L.append("- 진행 기록: `진행.txt`")
    rp = os.path.join(out, "종합보고서.md")
    open(rp, "w", encoding="utf-8").write("\n".join(L))
    # HTML 한 장
    for _, _, r in parts:
        for folder in ("1_원곡검사", "2_목소리검사"):
            zd = os.path.join(r["out"], folder, "확대")
            if os.path.isdir(zd):
                imgs += [(f"{'원곡' if folder.startswith('1') else '목소리'} 확대 — {f[:-4]}", os.path.join(zd, f)) for f in sorted(os.listdir(zd))[:8]]
        fl = os.path.join(r["out"], "3_곡설명", "곡흐름.png")
        if os.path.isfile(fl):
            imgs.append(("곡 흐름", fl))
    try:
        build_html(out, f"종합 보고서 — {os.path.basename(path)}", "한마디: " + verdict, summary, items, imgs, lyrics=lyr, sections=sections,
                   links=[l[2:] for l in L[L.index("## 자세한 결과") + 2:] if l.startswith("- ")])
    except Exception as e:  # noqa
        print("HTML 보고서를 못 만들었습니다:", e)
    print(f"\n[완료] {rp} ({time.time() - t0:.0f}초)")
    print("\n".join(L[:14]))


# ---------------------------------------------------------------- 준비 한 번에
MODEL_DIR = os.path.expanduser("~/.cache/sound-ear")
ESS_BASE = "https://essentia.upf.edu/models/"
ESS_FILES = ["feature-extractors/discogs-effnet/discogs-effnet-bs64-1.pb"] + [
    f"classification-heads/{h}/{h}-discogs-effnet-1.{ext}"
    for h in ("mtg_jamendo_instrument", "genre_discogs400", "mtg_jamendo_moodtheme") for ext in ("pb", "json")]
WHISPER_URL = "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/{}"
PKGS = [("librosa", "librosa"), ("soundfile", "soundfile"), ("pyloudnorm", "pyloudnorm"), ("matplotlib", "matplotlib"),
        ("koreanize_matplotlib", "koreanize-matplotlib"), ("scipy", "scipy"), ("pywhispercpp", "pywhispercpp"),
        ("onnxruntime", "onnxruntime"), ("essentia", "essentia-tensorflow"), ("pykakasi", "pykakasi")]


def _pip(name):
    for extra in ([], ["--break-system-packages"]):
        r = subprocess.run([sys.executable, "-m", "pip", "install", "-q", *extra, name], capture_output=True, text=True)
        if r.returncode == 0:
            return True
    return False


def _download(url, dest, timeout=600):
    import urllib.request
    tmp = dest + ".part"
    try:
        with urllib.request.urlopen(url, timeout=30) as r, open(tmp, "wb") as f:
            while True:
                b = r.read(1 << 20)
                if not b:
                    break
                f.write(b)
        os.replace(tmp, dest)
        return True
    except Exception:
        if os.path.exists(tmp):
            os.remove(tmp)
        return False


def _sha(p):
    import hashlib
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def _collect_uploads(search_dirs):
    """사용자가 올리거나 가져온 모델 파일을 찾아 제자리에 둔다: ggml-*.bin, part1/part2, essentia tar, Kim_Vocal_2.onnx"""
    import glob
    import tarfile
    found = []
    os.makedirs(os.path.join(MODEL_DIR, "essentia"), exist_ok=True)
    for d in search_dirs:
        if not d or not os.path.isdir(d):
            continue
        deep = d != os.getcwd()  # 지금 폴더는 바로 안만 본다(홈 전체를 뒤지지 않게)
        for p in glob.glob(os.path.join(d, "**", "*") if deep else os.path.join(d, "*"), recursive=deep):
            b = os.path.basename(p)
            if not os.path.isfile(p):
                continue
            if b.startswith("ggml-") and b.endswith(".bin"):
                dst = os.path.join(MODEL_DIR, b)
                if not os.path.exists(dst) or os.path.getsize(dst) != os.path.getsize(p):
                    subprocess.run(["cp", p, dst])
                    found.append(b)
            elif b.endswith(".bin.part1"):
                p2 = p[:-1] + "2"
                name = b[:-len(".part1")]
                dst = os.path.join(MODEL_DIR, name)
                if os.path.exists(p2) and not os.path.exists(dst):
                    with open(dst, "wb") as out:
                        for q in (p, p2):
                            with open(q, "rb") as f:
                                for chunk in iter(lambda: f.read(1 << 20), b""):
                                    out.write(chunk)
                    found.append(name + " (두 조각을 합침)")
            elif b == "Kim_Vocal_2.onnx":
                dst = os.path.join(MODEL_DIR, b)
                if not os.path.exists(dst):
                    subprocess.run(["cp", p, dst])
                    found.append(b)
            elif b.endswith(".tar") and "essentia" in b.lower():
                if all(os.path.exists(os.path.join(MODEL_DIR, "essentia", os.path.basename(f))) for f in ESS_FILES):
                    continue
                try:
                    with tarfile.open(p) as t:
                        for m in t.getmembers():
                            if m.isfile() and "/" not in m.name and ".." not in m.name:
                                t.extract(m, os.path.join(MODEL_DIR, "essentia"))
                    found.append(b + " (풀어 둠)")
                except Exception:
                    pass
            elif "discogs-effnet" in b and b.endswith((".pb", ".json")):
                dst = os.path.join(MODEL_DIR, "essentia", b)
                if not os.path.exists(dst):
                    subprocess.run(["cp", p, dst])
                    found.append(b)
    return found


def cmd_setup(args):
    """필요한 것을 확인하고 깔 수 있는 건 깔고, 못 받는 건 받는 법을 알려 준다."""
    rows = []
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

    def row(name, ok, note=""):
        rows.append((name, ok, note))
        print(f"  {'✅' if ok else '❌'} {name}" + (f" — {note}" if note else ""))
    print("1) 프로그램")
    import shutil
    ok_ff, note_ff = ensure_ffmpeg(download=not args.check)
    row("ffmpeg", ok_ff, note_ff)
    import importlib
    for mod, pipname in PKGS:
        try:
            importlib.import_module(mod)
            ok = True
        except Exception:
            ok = _pip(pipname) if not args.check else False
        row(pipname, ok, "" if ok else "pip로 못 깔았습니다")
    print("2) 올려 둔 모델 파일 찾기")
    dirs = [os.getcwd(), "/mnt/user-data/uploads", os.path.expanduser("~/Downloads")] + (args.from_dir or [])
    got = _collect_uploads(dirs)
    for g in got:
        print("   가져옴:", g)
    print("3) 모델")
    os.makedirs(MODEL_DIR, exist_ok=True)
    need = []
    # 받아쓰기
    small = os.path.join(MODEL_DIR, "ggml-small-q5_1.bin")
    med = os.path.join(MODEL_DIR, "ggml-medium-q5_0.bin")
    if not (os.path.exists(small) or os.path.exists(med)) and not args.check:
        _download(WHISPER_URL.format("ggml-small-q5_1.bin"), small)
    if args.accurate and not os.path.exists(med) and not args.check:
        _download(WHISPER_URL.format("ggml-medium-q5_0.bin"), med)
    for p_, mn in ((med, 500e6), (small, 150e6)):
        if os.path.exists(p_) and os.path.getsize(p_) < mn:
            os.remove(p_)
            print(f"   {os.path.basename(p_)} 이 덜 받아져(크기가 작음) 지웠습니다")
    has_w = os.path.exists(small) or os.path.exists(med)
    row("가사 받아쓰기 (whisper)", has_w, ("큰 모델" if os.path.exists(med) else "작은 모델") if has_w else "")
    if not has_w:
        need.append(("ggml-small-q5_1.bin (190MB) — 가사 받아쓰기", WHISPER_URL.format("ggml-small-q5_1.bin")))
    if args.accurate and not os.path.exists(med):
        need.append(("ggml-medium-q5_0.bin (539MB) — 한국어 가사가 훨씬 정확", WHISPER_URL.format("ggml-medium-q5_0.bin")))
    # 보컬 분리
    sep = os.path.join(MODEL_DIR, "Kim_Vocal_2.onnx")
    if not os.path.exists(sep) and not args.check:
        _download(SEP_URL, sep)
    row("보컬 분리 (UVR)", os.path.exists(sep))
    if not os.path.exists(sep):
        need.append(("Kim_Vocal_2.onnx (67MB) — 보컬 분리", SEP_URL))
    # 음악 AI
    ed = os.path.join(MODEL_DIR, "essentia")
    os.makedirs(ed, exist_ok=True)
    miss = [f for f in ESS_FILES if not os.path.exists(os.path.join(ed, os.path.basename(f)))]
    if miss and not args.check:
        for f in miss:
            _download(ESS_BASE + f, os.path.join(ed, os.path.basename(f)))
        miss = [f for f in ESS_FILES if not os.path.exists(os.path.join(ed, os.path.basename(f)))]
    row("악기·장르·분위기 AI (Essentia, 비상업용)", not miss)
    for f in miss:
        need.append((os.path.basename(f) + " — 악기·장르·분위기", ESS_BASE + f))
    print()
    if need:
        print("❗ 아래 파일은 이 환경에서 직접 받지 못했습니다(인터넷이 막혀 있을 수 있음).")
        print("   브라우저로 받아서 채팅에 올리거나 폴더에 두고 `setup`을 다시 돌리면 알아서 제자리에 둡니다.")
        print("   (400MB가 넘는 파일을 못 올리면 반씩 .part1/.part2로 나눠 올려도 합쳐 줍니다)")
        for name, url in need:
            print(f"   - {name}\n     {url}")
    else:
        print("🎉 전부 준비됐습니다. `full 노래파일`로 시작하세요.")
    json.dump(dict(rows=rows, missing=need), open(os.path.join(MODEL_DIR, "setup_status.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    return 0 if not need else 1


def main():
    ap = argparse.ArgumentParser(description="소리 듣기 도구")
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("check")
    c.add_argument("file")
    c.add_argument("--mode", default="auto", choices=["auto", "vocal", "mix"])
    c.add_argument("--out")
    c.add_argument("--max-zoom", type=int, default=12)
    c.add_argument("--key-from", help="조(키)를 잴 다른 파일(반주·원곡). 보컬.wav 옆에 반주.wav가 있으면 자동")
    c.add_argument("--no-split", action="store_true", help="20분 넘는 파일도 조각내지 않고 한 번에(메모리가 넉넉할 때)")
    c.set_defaults(fn=cmd_check)
    nt = sub.add_parser("notes", help="노트 단위 검사: 노래 소리 + 프로젝트(.svp/.vpr/.ust/.mid) → 몇 번째 노트가 문제인지")
    nt.add_argument("file", help="렌더한 보컬 소리 파일")
    nt.add_argument("project", help="브이신스 .svp / 보컬로이드 .vpr / UTAU .ust / MIDI .mid")
    nt.add_argument("--track", help="트랙 이름(여러 트랙이면)")
    nt.add_argument("--offset", type=float, help="소리가 프로젝트보다 몇 초 늦는지(기본: 자동)")
    nt.add_argument("--out")
    nt.set_defaults(fn=cmd_notes)
    og = sub.add_parser("origin", help="출처 검사: 꼬리표·C2PA 서명으로 수노·우디오 등 어디서 만든 파일인지")
    og.add_argument("file")
    og.add_argument("--out")
    og.set_defaults(fn=cmd_origin)
    ly = sub.add_parser("lyrics", help="가사 대조: 내가 쓴 가사(txt)와 들린 것을 맞춰 발음이 뭉개진 곳 찾기")
    ly.add_argument("file")
    ly.add_argument("lyrics", help="내가 쓴 가사 txt (한 줄에 한 소절)")
    ly.add_argument("--words", help="이미 받아쓴 words.json (있으면 다시 받아쓰지 않음)")
    ly.add_argument("--model")
    ly.add_argument("--lang", default="auto")
    ly.add_argument("--fast", action="store_true")
    ly.add_argument("--out")
    ly.set_defaults(fn=cmd_lyrics)
    k = sub.add_parser("cut")
    k.add_argument("file")
    k.add_argument("time")
    k.add_argument("--len", type=float, default=2.0)
    k.add_argument("--mode", default="auto", choices=["auto", "vocal", "mix"])
    k.add_argument("--out")
    k.set_defaults(fn=cmd_cut)
    w = sub.add_parser("words", help="말·가사 받아쓰기 (whisper.cpp)")
    w.add_argument("file")
    w.add_argument("--fast", action="store_true", help="받아쓰기를 작은 모델로 빠르게")
    w.add_argument("--model", help="ggml-*.bin 경로")
    w.add_argument("--lang", default="auto", help="ko, ja, en ... (기본: 자동)")
    w.add_argument("--prompt", help="힌트 글(곡 제목, 자주 나오는 단어)")
    w.add_argument("--start")
    w.add_argument("--end")
    w.add_argument("--out")
    w.set_defaults(fn=cmd_words)
    d = sub.add_parser("describe", help="곡 설명: 구성·드럼·저음·세기·분위기·가사")
    d.add_argument("file")
    d.add_argument("--fast", action="store_true", help="받아쓰기를 작은 모델로 빠르게")
    d.add_argument("--model")
    d.add_argument("--lang", default="auto")
    d.add_argument("--words", help="이미 받아쓴 words.json")
    d.add_argument("--no-words", action="store_true")
    d.add_argument("--essentia", help="Essentia 모델 폴더")
    d.add_argument("--no-ai", action="store_true", help="음악 AI(악기·장르·분위기) 끄기")
    d.add_argument("--no-separate", action="store_true", help="보컬 분리 끄기")
    d.add_argument("--sep-model", help="Kim_Vocal_2.onnx 경로")
    d.add_argument("--vocal", help="이미 갈라 둔 보컬.wav (분리를 건너뜀)")
    d.add_argument("--prompt", help="받아쓰기 힌트(곡 제목·자주 나오는 말)")
    sp = sub.add_parser("separate", help="목소리와 반주 가르기 (보컬.wav, 반주.wav)")
    sp.add_argument("file")
    sp.add_argument("--model")
    sp.add_argument("--clean", action="store_true", help="두 번 돌려 잡음을 조금 더 줄임(두 배 느림)")
    sp.add_argument("--out")
    sp.set_defaults(fn=cmd_separate)
    d.add_argument("--out")
    d.set_defaults(fn=cmd_describe)
    fu = sub.add_parser("full", help="한 번에 듣기: 흠 찾기 → 곡 설명 → 종합보고서.md (기본 빠른 검사, --deep은 보컬 분리·큰 모델·목소리 음정)")
    fu.add_argument("file")
    fu.add_argument("--deep", action="store_true", help="자세히: 보컬 분리 + 큰 받아쓰기 모델 + 목소리 음정 검사 (몇 배 느림)")
    fu.add_argument("--mode", default="auto", choices=["auto", "vocal", "mix"], help="보컬만 있는 파일이면 vocal (이름에 vocal·보컬이 있으면 자동)")
    fu.add_argument("--lang", default="auto")
    fu.add_argument("--fast", action="store_true", help="받아쓰기를 작은 모델로 빠르게")
    fu.add_argument("--model")
    fu.add_argument("--sep-model")
    fu.add_argument("--no-separate", action="store_true")
    fu.add_argument("--no-words", action="store_true")
    fu.add_argument("--no-ai", action="store_true")
    fu.add_argument("--all", action="store_true", help="참고 항목까지 종합 보고서에 넣기")
    fu.add_argument("--redo-words", action="store_true", help="이미 받아쓴 가사가 있어도 다시 받아쓰기")
    fu.add_argument("--lyrics", help="내가 쓴 가사 txt — 들린 것과 맞춰 발음이 뭉개진 줄을 찾아 종합보고서에 넣음")
    fu.add_argument("--no-split", action="store_true", help="20분 넘는 파일도 조각내지 않고 한 번에")
    fu.add_argument("--out")
    fu.set_defaults(fn=cmd_full)
    se = sub.add_parser("setup", help="필요한 프로그램·모델을 한 번에 확인하고 준비")
    se.add_argument("--check", action="store_true", help="설치·다운로드 없이 상태만 보기")
    se.add_argument("--accurate", action="store_true", help="한국어 가사가 더 정확한 큰 받아쓰기 모델(539MB)도 받기")
    se.add_argument("--from", dest="from_dir", action="append", help="모델 파일을 둔 폴더(여러 번 가능)")
    se.set_defaults(fn=cmd_setup)
    m = sub.add_parser("compare", help="두 판 비교: 없어진 흠·새로 생긴 흠·그대로인 흠 표 + 크기 비교")
    m.add_argument("a")
    m.add_argument("b")
    m.add_argument("--mode", default="auto", choices=["auto", "vocal", "mix"])
    m.add_argument("--no-align", action="store_true", help="두 판의 시각 어긋남을 맞추지 않음")
    m.add_argument("--out")
    m.set_defaults(fn=cmd_compare)
    args = ap.parse_args()
    if getattr(args, "fast", False) or (args.cmd == "full" and not args.deep):
        os.environ["SOUND_EAR_FAST"] = "1"
    sys.exit(args.fn(args) or 0)


if __name__ == "__main__":
    main()
