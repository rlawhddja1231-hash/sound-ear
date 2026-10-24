"""Self-test: plant 7 known problems in a synthetic vocal and check listen.py finds all of them.
Usage: python3 selftest.py   (prints PASS/FAIL)"""
import json, os, subprocess, sys, tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
LISTEN = os.path.join(HERE, "..", "scripts", "listen.py")
EXPECT = {  # kind keyword -> time (s)
    "음정 어긋남": 3.6, "딸깍": 7.3, "클리핑": 10.0, "음이 순간 튐": 12.6,
    "뚝 끊김": 15.0, "ㅅ·ㅊ": 17.6, "끝이 잘림": 21.5,
}
with tempfile.TemporaryDirectory() as d:
    wav = os.path.join(d, "test_vocal.wav")
    subprocess.run([sys.executable, os.path.join(HERE, "make_test.py"), wav], check=True, capture_output=True)
    out = os.path.join(d, "out")
    subprocess.run([sys.executable, LISTEN, "check", wav, "--mode", "vocal", "--out", out], check=True, capture_output=True)
    probs = json.load(open(os.path.join(out, "result.json"), encoding="utf-8"))["problems"]
    ok = True
    for kw, t in EXPECT.items():
        hit = any(kw in p["kind"] and abs(p["t"] - t) < 0.4 for p in probs)
        print(("found  " if hit else "MISSED ") + f"{kw} @ {t}s")
        ok &= hit
    extra = [p for p in probs if not any(kw in p["kind"] and abs(p["t"] - t) < 0.4 for kw, t in EXPECT.items())
             and not (p.get("end") is None and p["t"] == 0)]
    for p in extra:
        print(f"extra  {p['kind']} @ {p['t']:.1f}s")
    print("PASS" if ok and not extra else "FAIL")
    sys.exit(0 if ok and not extra else 1)
