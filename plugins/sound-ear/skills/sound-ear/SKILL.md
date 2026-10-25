---
name: sound-ear
description: "Lets Claude 'hear' audio files. Turns songs, vocals or video soundtracks into numbers and pictures with a bundled script, then finds problems with timestamps: clicks, clipping, abrupt cuts, pitch glitches, off-pitch notes, wobbly notes, harsh sibilance, loud breaths, cut-off endings, loudness too high or low. Also estimates BPM, key and vocal range, transcribes lyrics and speech with whisper.cpp, and describes a song the way a listener would: sections (intro, verses, chorus, breaks), repeats, drums, bass and named instruments, genre, loudness changes, mood and repeated lyric hooks. Built for checking vocal-synth songs (SynthV, VOCALOID, UTAU). Use whenever the user shares an audio or video file (wav, mp3, flac, m4a, ogg, opus, webm, mp4) or asks to listen, describe a song, get lyrics or subtitles, check a mix, find glitches, measure BPM/key/loudness, or compare two versions. 소리 듣기: 노래 들어 봐, 검수, 가사, 받아쓰기, 자막, 곡 구성, 후렴, 분위기, 음정, 음 튐, 딸깍, 잡음, 깨짐, 끊김, BPM, 키, 음역, 믹싱, 두 판 비교."
license: MIT
---

# sound-ear — listen to audio and find problems

Claude has no ears. This skill turns an audio file into **numbers and pictures** with `scripts/listen.py`, so Claude can inspect it, point at exact timestamps, and explain how to fix each problem. Reply to the user in their own language (the report and chart labels are in Korean; translate them when the user writes in another language).

What it cannot do: listen live through a microphone or judge taste. Lyrics need a whisper model file; instrument, genre and mood names need Essentia model files (see setup). Pitch checks need a **vocal-only** file; for a full mix, separate it first (see 2b).

## 1. Setup (once per session)

```bash
L="${CLAUDE_SKILL_DIR:-}/scripts/listen.py"
[ -f "$L" ] || L=$(find / -name listen.py -path '*sound-ear*' 2>/dev/null | head -1)
pip install -q librosa soundfile pyloudnorm matplotlib koreanize-matplotlib scipy 2>/dev/null \
  || pip install -q --break-system-packages librosa soundfile pyloudnorm matplotlib koreanize-matplotlib scipy
python3 "$L" setup          # checks/installs everything, fetches reachable models; --check = status only
```

`setup` installs the optional packages, gathers model files the user attached (from the working folder, `/mnt/user-data/uploads`, `~/Downloads`, or `--from DIR`; `X.bin.part1`+`.part2` are joined, an `essentia*.tar` is unpacked), downloads what it can reach, and prints a ✅/❌ table plus the URL of every model it could not fetch. If something is missing, show the user that list in plain words and ask them to attach the file, then run `setup` again. Add `--accurate` to also fetch the 539 MB medium lyrics model. ffmpeg must already be installed. Manual details follow.

**Lyrics / speech (optional):** `pip install pywhispercpp` and a whisper.cpp model file `ggml-*.bin` from https://huggingface.co/ggerganov/whisper.cpp (`ggml-small-q5_1.bin`, 190 MB, fast; or `ggml-medium-q5_0.bin`, 539 MB, much better for Korean sung lyrics, ~1.6× real time on 2 cores — the largest model found is used unless `--fast`). Put it in `~/.cache/sound-ear/` or pass `--model`, or set `SOUND_EAR_MODEL`. If the sandbox cannot reach Hugging Face, ask the user to download the file and attach it. Without a model, `describe` still works but guesses where the singing is from loudness only.

**Instruments, genre and mood (optional):** `pip install essentia-tensorflow` and these files from https://essentia.upf.edu/models/ (about 26 MB; **CC BY-NC-ND 4.0, non-commercial** — never bundle or redistribute them, the user downloads them): `feature-extractors/discogs-effnet/discogs-effnet-bs64-1.pb`, and from `classification-heads/`: `mtg_jamendo_instrument-discogs-effnet-1.pb/.json` (40 instruments), `genre_discogs400-discogs-effnet-1.pb/.json` (400 genres), `mtg_jamendo_moodtheme-discogs-effnet-1.pb/.json` (56 moods). Put them in `~/.cache/sound-ear/essentia/` or pass `--essentia DIR` / set `SOUND_EAR_ESSENTIA`. About 6 s for a 4-minute song.

**Vocal separation (optional, recommended for full mixes):** `pip install onnxruntime` and the UVR MDX-Net model `Kim_Vocal_2.onnx` (67 MB) from the Ultimate Vocal Remover model repo: https://github.com/TRvlvr/model_repo/releases/download/all_public_uvr_models/Kim_Vocal_2.onnx → `~/.cache/sound-ear/` (or `--sep-model`, `SOUND_EAR_SEP_MODEL`). Not bundled; download it. Takes about as long as the song on 2 CPU cores.

Self-tests (optional): `python3 "$(dirname "$L")/../tests/selftest.py"` (~20 s; 7 planted problems found, none extra) and `python3 "$(dirname "$L")/../tests/mixtest.py"` (~10 s; synthetic song with drumless intro, verses and a repeated louder chorus — checks sections, drums and chorus matching). Both should print `PASS`.

## 1b. Fastest path: one command

```bash
python3 "$L" full song.mp3 --lang ko --out song_full          # quick (default): ~2–3 min for a 4-min song
python3 "$L" full song.mp3 --lang ko --out song_full --deep   # detailed: separation + medium model + vocal pitch
```

Default `full` is **quick**: check the mix → describe with the small lyrics model, no separation. Use it unless the user asks for a detailed listen, lyrics accuracy or vocal pitch on a full mix; then run `--deep` (= separate → check the mix → pitch-check the separated vocal, key taken from the backing → describe with the largest model). **Report the result to the user as soon as it is ready; do not start extra runs they did not ask for.** Both write **`종합보고서.md`**, whose first line is a one-sentence verdict (e.g. "꼭 고칠 곳은 없음 👍 — 들어 보고 판단할 곳 3곳만…"); lead with it. 🟠 "들어 보고 판단" items show up even in professional songs (intentional effects, hi-hats), so say that rather than alarming the user. The report has a one-page summary (mood, key, tempo, LUFS, genre/mood/instruments, vocal range, structure line) plus one table of everything to fix, most urgent first, with where it was found (mix or vocal). Read that first, then open the detailed reports it links. `--deep` on a 4.5-minute song takes ~10 min on 2 cores (~6 min with `--fast`). Example output: `docs/example-report/` in the GitHub repo. For a vocal-only file add `--mode vocal` (automatic when the name contains vocal/보컬) so the quick run includes the pitch check; with `--deep`, a file with almost no backing is detected and checked as a vocal directly.

## 2. Workflow (step by step)

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

## 2b. Hearing the song as a listener

```bash
python3 "$L" describe song.mp3 --out song_described          # transcribes too if a model is found
python3 "$L" words song.mp3 --lang ko --out song_lyrics      # lyrics/speech only; also writes 자막.srt
python3 "$L" describe song.mp3 --words song_lyrics/words.json # reuse an earlier transcription
```

With the separation model, `describe` first splits the song into `보컬.wav` and `반주.wav` in the output folder (reused on later runs), transcribes the vocal stem, fills gaps from the full mix (lines marked "원곡에서 받아씀"), and uses the vocal stem to find where singing is. You can also run the pitch check on a full mix this way:

```bash
python3 "$L" separate song.mp3 --out stems            # stems/보컬.wav, stems/반주.wav
python3 "$L" check stems/보컬.wav --mode vocal        # key is taken from stems/반주.wav automatically
```

`describe` writes `곡설명.md`, `describe.json` and `곡흐름.png`: a one-line mood guess, key and tempo, a timeline table (section, repeat letter, loudness bar, drums, bass, singing, lyrics), a plain-language "as heard" walkthrough, repeated lyric hooks, and the full lyrics. Open the picture before you speak.

How it decides (so you can explain or overrule it):
- **Sections**: novelty curve on beat-synchronous chroma + timbre + loudness; repeats are long diagonal stripes in a chroma recurrence matrix. Same letter = same melody/harmony.
- **Chorus**: a repeated section that *returns after other parts* (back-to-back repeats count once) and whose repeats share lyrics — either overall or at least one matching line (a hook). Same melody with different lyrics is a verse. Without lyrics it falls back to "repeated and clearly louder"; if neither holds it does not name a chorus.
- **Sections** also react to drums entering or leaving (percussive level is part of the novelty features), and short songs get at least 5 parts.
- **Drums**: percussive part (HPSS) hitting regularly on the beat (autocorrelation ≥ 0.45). Vocal-only tracks score ~0.25.
- **Mood**: the one-line guess comes from key (major/minor), tempo, loudness and brightness. With Essentia models, the music AI's genre, mood and instrument guesses (with confidence %) are added: above ~30 % is fairly sure, ~15 % is "maybe". Per-section instruments show what enters or leaves.
- **Vocal-only files**: the instrument AI was trained on full mixes and hears drums in a cappella tracks; when the beat-regularity check finds no drums anywhere, drum-like labels are dropped, and when there is also no bass anywhere, instrument names are skipped.
- **Lyrics**: whisper.cpp; lines that look made up during instrumentals (subtitle credits, "please subscribe", bracketed song titles, stretched syllables, repeated lines) are flagged or dropped. With separation, the vocal stem is transcribed and only gaps where the voice is audible but no words came out are re-transcribed from the mix ("원곡에서 받아씀"). Sung lyrics have mistakes, especially over loud backing; vocal-only files are much better. About 25–40 s per minute of audio on 2 CPU cores with the small model.

## 3. What each finding means and how to fix it

Menu names differ between vocal synths and versions; treat these as general directions.

| Finding (Korean label) | Meaning | Typical fix |
|---|---|---|
| 클리핑 (clipping) | Samples hit full scale; audible crackle | Lower track/master gain; limiter ceiling −1 dB |
| 소리가 0dB까지 꽉 참 | Lossy file (mp3/opus) touching 0 dB; can't prove real clipping | Check the original wav; master is very hot |
| 트루 피크 넘침 (true peak over) | Fine now, may distort after mp3/AAC encoding | Limiter ceiling −1 dBTP |
| 딸깍·틱 소리 (click) | Very short tick | Inspect note joins; re-render that part; add short fades at edit points |
| 노트 끝 틱 (note-end tick) | A loud note stops dead and ticks | Add a short fade / extend the note tail |
| 틱이 몰린 구간 (clustered ticks) | Many ticks within 10 s in a mix | Often a deliberate stutter/glitch edit; otherwise check edit seams |
| 음높이를 믿기 어려운 곳 | Harmony or noise; pitch checks skipped there | Check with a single-voice export |
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
- Click: >3 kHz energy in a 1 ms bin over 8× the 150 ms local median, with 3–8 ms neighbours under a third of it, skipping note/drum onsets where the level jumps more than 1.5× (in mix mode downgraded to "listen and decide")
- Abrupt cut: from above −25 dB to below −60 dB within 10 ms
- Pitch glitch: fine pitch (yin, 5.8 ms hop) more than 3 semitones from the local median and back within 80 ms, with an abrupt step, at ≥60 % of phrase level, confirmed by the spectrum (new fundamental present, old one gone, no octave confusion); skipped where pitch is unreliable (harmony/noise clusters)
- Off-pitch: notes ≥0.25 s whose median is >30 cents off after removing the global tuning offset
- Sibilance: 5–10 kHz energy within 6 dB of vowel level (75th percentile), >50 % of the frame, ≥30 ms

Tuned on synthetic tests, two vocal-only synth tracks and three full mixes. If you see false positives or misses on real material, note them in `references/log.md` with the date and adjust.
