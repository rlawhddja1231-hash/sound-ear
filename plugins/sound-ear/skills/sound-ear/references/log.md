# Field log

Newest first. Record date, file type, what was found, false positives, misses, and threshold changes.

## 2026-10-26 — Suno "watermark" probe (no detectable signature)
- Suno announced (2026-08-06) an inaudible watermark + Musixmatch fingerprinting, technique undisclosed. Checked whether a long-term-average-spectrum (LTAS) signature exists: 4 public Suno songs captured in Chrome via WebAudio AnalyserNode (fft 8192, 48 kHz, 35 s each) vs 3 human recordings (archive.org mp3, Wikimedia ogg).
- Suno: hard cutoff at 20.1 kHz (AAC), no narrow non-harmonic peaks shared across songs (one song had 11.2 k / 14.0 k peaks, others none). Humans: cutoff 16–20 k, ogg files even show more high-band residue. Band levels overlap completely.
- Conclusion: nothing in the average spectrum separates Suno from human mixes; a watermark, if present, is below this method's floor or spread-spectrum/neural. An "AI-made?" check would need a trained classifier on hundreds of labelled songs, and would break with each Suno model update. Not added. Revisit only if Suno/Musixmatch publish a detector API.
- Capture notes: Suno plays `cloudfront /1/clip/ID.m4a` through MediaSource (fragmented mp4 — decodeAudioData fails), so only live capture through createMediaElementSource on the blob element works; the analysis code is in suno.com localStorage._sec (window._run).

## 2026-10-26 — lyrics comparison, HTML report, progress, long files — v1.8.0
- `lyrics` / `full --lyrics`: written lyrics vs transcript aligned with difflib on normalized character streams (Korean → jamo, Japanese → hiragana via optional pykakasi, Latin → lowercase, punctuation and spaces dropped). First the score was jamo-level, which let "울지 말라고" with two wrong syllables pass at 89 %; switched to syllable-level: any syllable with a differing jamo is marked, a line with one marked syllable is 🟠, under 60 % is 🔴. Tested only with simulated transcripts here (Hugging Face is blocked in this sandbox, so no whisper model) — the alignment is right; real-song behaviour still to be checked on the PC.
- Japanese: kakasi reads 君 as くん while the singer says きみ → one false mark per such kanji. Documented as model habit.
- `보고서.html`: images and 2.4 s mp3 snippets (80 kbps mono) are base64-embedded; a 21 s test file gave a 2.5 MB page. Snippets capped at 24, images at 20.
- Progress: `progress_step/tick` print and overwrite `<out>/진행.txt`; whisper progress through `new_segment_callback` (falls back silently if the pywhispercpp version lacks it). Estimates: check 0.08×dur (mix) / 0.2×dur (vocal), separate 1.05×dur, words 0.5×dur (small) / 1.7×dur (medium), describe 0.17×dur.
- Long files: >20 min → 10-minute wav pieces in `_조각/` (ffmpeg, pcm_f32le). `check` and `full` run per piece; problems shifted to absolute time; edge artefacts dropped (앞 무음/시작 잘림 except piece 1, 끝 잘림 except last, 뚝 끊김 within 0.3 s of an edge); whole-file items deduped across pieces; LUFS merged as length-weighted power mean; peak = max. Memory for a 10-min piece check ≈ 1.5 GB RSS.
- librosa 1.0.0 is what pip installs now; selftest and mixtest still PASS.

## 2026-10-25 — Japanese stem lyrics, faster describe, vocal-only detection — v1.7.0
- Japanese synth vocal (구마, separated stem, medium): whole-file transcription gave one made-up line ("ご視聴ありがとうございました"); splitting at pauses into ≤ 25 s pieces gave 37 real lines and was faster (317 s vs 345 s). On full mixes the same split made things worse (Korean 마법: fewer correct lines, 215 s vs 124 s), so it is used only for stems and vocal-only files.
- Language is picked from the three loudest 30 s windows instead of the first 30 s.
- `describe`: HPSS median filtering done once instead of three times — 69 s → 39 s on a 4-min song, identical output.
- Vocal-only detection without separation: energy below 70 Hz < 1 % of the total (6 vocal-only files ≤ 0.0001, 7 mixes 0.09–0.48).
- Quick `full` measured: 4-min song 3.5 min, 4:52 song 4 min (lyrics are most of it). Docs now say 3–4 min.
- UNI SV2 demo 「목성은 사랑을 한다」 (YouTube): piano ballad where verse and chorus share one repeat letter (7×) → all were called chorus. When a letter repeats ≥ 5 times, only the repeats that share a lyric line with another repeat and are not quieter get the chorus name (7 → 4). Alone, Pray To Me and 마법 unchanged.
- SeeU cover of the same song (YouTube): 0 must-fix, 3 to check. Same key (G major) as the UNI version.
- Report: more than 3 🟠 of one kind are folded into one row; lossy "0 dB" wording fixed for a handful of hits.

## 2026-10-25 — setup command, quick mode by default, one-line verdict — v1.6.0
- `setup`: installs packages, gathers attached model files (joins .part1/.part2, unpacks an essentia tar), downloads what it can reach, prints ✅/❌ and URLs for the rest. Tested on an empty home: Kim_Vocal_2.onnx downloaded from GitHub; the rest came from attached files.
- `full` is now quick by default (no separation, small lyrics model); `--deep` is the old behaviour. Users found the 10-minute run too slow for a first look.
- `종합보고서.md` starts with a one-line verdict and says that 🟠 items are common even in professional songs.
- Abrupt-cut threshold now follows the song's level (median − 8 dB, at most −25 dB): a 0.25 s dropout in a −19 LUFS synthetic song was missed with the fixed −25 dB rule.
- Also in this version (found on a cover of 구마): pitch-glitch skip at note boundaries; clicks found only in the separated vocal become 🟠 (separation leftovers); `full` reuses words.json (`--redo-words`); `describe --vocal`; whisper `suppress_nst` + `no_context` (medium stopped printing "(♪~♪)"); more junk-line filters; sibilance flags that coincide with backing hi-hats are dropped and >8 groups become one summary line.
- Example report in `docs/example-report/` (synthetic song, planted click and dropout, both found).

## 2026-10-25 — one-command full report, medium lyrics model, better chorus — v1.5.0
- `full` command and `종합보고서.md`. On "Alone" (4:26): ~9.5 min with the medium model (separation cached), 0 must-fix, 1 true-peak note.
- Medium model (ggml-medium-q5_0, 539 MB, SHA256 19fea4b3…) on the separated vocal: the first verse that small turned into "[끝]" / subtitle junk came out as real lyrics, and many lines improved ("보랏빛 하늘로 피어나는데", "그런 소망 따라 흩날리는 무수한 나비"). ~1.6× real time vs ~0.4× for small. Largest model is the default; `--fast` picks the smallest.
- Gap filling now re-transcribes only stretches where the vocal stem is audible but no words came out (was: the whole mix again).
- Junk filter: bracketed song titles up to 40 chars, "한국어 노래가 정말 잘 어울려요"-type prompt echoes, stretched syllables.
- Chorus: back-to-back repeats count once (a chorus must return after other parts); a shared hook line (≥ 0.5 bigram similarity) between repeats counts. Result: Alone → G (4×), Pray To Me → "Now pray to me" (unchanged), 마법이 사라진 세상에서 → the "마법과 함께라면" sections (3×) — before, the last two had no chorus.
- Sections: percussive level added to novelty features; at least 5 parts for short songs. New `tests/mixtest.py` (synthetic song) checks drumless intro, drums, chorus repeat and loudness.
- Off-pitch: notes that glide (pitch std > 25 cents) are expression, not errors; threshold 32 cents.

## 2026-10-25 — vocal separation — v1.4.0
- Added `separate` (UVR MDX-Net Kim_Vocal_2.onnx via onnxruntime, single pass; `--clean` averages a sign-flipped second pass, 2× slower, −36 dB difference). ~1× real time on 2 cores.
- On the new SeeU demo "Alone": separated vocals were clean. Transcribing the vocal stem split lines properly and fixed many words ("아픔", "지나쳐 가는", "쇼윈도"), and singing start times became exact (0:06 instead of 0:00 on a 30 s clip). But whisper returned only "[끝]" for the first verse on the stem → describe now transcribes both and fills gaps from the mix ("원곡에서 받아씀").
- Singing presence from stem energy: threshold p90 − 20 dB marked the outro as sung (backing leakage) → p90 − 14 dB.
- Pitch check on the separated vocal: 0 glitches, 1 borderline off-pitch (a glide). Key estimated from the vocal alone was B♭ major (melody), mix/backing said E♭ major → `check --mode vocal` now takes the key from a sibling 반주.wav (or `--key-from`); out-of-key flags dropped 13 → 4.
- Whole describe on a 4:26 song with separation + double transcription + music AI: ~4.7 min. Self-test 7/7.

## 2026-10-25 — full workflow on a Korean vocal-synth pop song — v1.3.1
- New SeeU voice demo "Alone" (4:26, full mix, lossy recording). check + describe (lyrics + music AI) took ~3.5 min.
- Click false positive at 1:00.4 was a kick drum (same shape as the kicks around it). Clicks are now skipped when low-band (<150 Hz) energy in the next 40 ms rises >1.25× in mixes (>2× otherwise). Self-test still 7/7.
- The one-line loudness word said "quiet" for a dense pop song: it used mean RMS dB. Now uses integrated LUFS (> −10 "dense and loud", > −16 "strong", else "soft").
- Structure: 4 repeats of one vocal section were found but no chorus was named (lyrics transcribed differently each time over the dense backing) — honest but a known limit; vocal separation would help.

## 2026-10-24 — instruments, genre and mood with Essentia — v1.3.0
- Added optional Essentia models (discogs-effnet embeddings + MTG-Jamendo instrument/moodtheme heads + Discogs-400 genre head). 4-minute song in ~6 s on 2 CPU cores.
- Results: Korean folk-pop song → guitar/acoustic guitar/piano, K-pop/ballad; Gumi synth-pop → synthesizer/drums/bass, K-pop/dance-pop/synth-pop; Japanese vocaloid track → synthesizer/bass/sampler, happy hardcore/J-pop. All plausible.
- Vocal-only Bad Apple: the AI said "drums 40 %" (trained on full mixes). Our beat-regularity check says no drums → drum-like labels dropped, and with no bass anywhere, instrument names are skipped.
- Models are CC BY-NC-ND: not bundled; users download them.

## 2026-10-24 — lyrics and "describe" (listening like a person) — v1.2.0
- Added `words` (whisper.cpp via pywhispercpp, ggml-small-q5_1) and `describe`.
- Lyrics: Korean song 4:02 in 103 s, English song 3:24 and Japanese song 2:27 in ~2 min on 2 cores. English and Japanese came out almost word-perfect; Korean had a few misheard words over loud backing. Filter added for bracket tags like "[끝]".
- Sections: agglomerative clustering gave one 1:45-long chunk → switched to a Foote novelty curve with peak picking. Repeat matching by mean feature similarity found almost no repeats → switched to diagonal stripes in a chroma recurrence matrix (stack_memory + path_enhance).
- Chorus: "most repeated" was wrong on a strophic Korean song (same melody sung many times). Rule now: repeats with the same lyrics = chorus (found "Now pray to me" in the English song at both places); otherwise only if clearly louder; otherwise no chorus is named. Repeated lyric lines are listed separately as hook candidates.
- Drums: fixed percussive-energy thresholds said "no drums" everywhere → beat-regularity of the percussive onset envelope: vocal-only 0.25–0.28, drummed sections 0.5–0.9. Threshold 0.45.
- Section loudness uses the 75th percentile so breaths/gaps in vocal-only tracks don't make every section "quiet".
- Self-test 7/7 PASS.

## 2026-10-24 — vocal-only synth recordings + one more synth mix — v1.1.0
- Tested on three vocal-synth tracks: a Synthesizer V vocal-only demo (1:27), a Kasane Teto vocal-only track (3:40, with harmonies), and a Synthesizer V full mix (3:24). All lossy (opus) recordings.
- Pitch glitches: the first vocal-only run flagged 39 "glitches", almost all false. Fixes, in order: (1) only count glitches where the level is ≥60 % of the phrase median (was 30 %) — note tails gliding down were flagged; (2) octave jumps (12/19/24 semitones) during quieter frames are pitch-tracker confusion; (3) verify with the spectrum: the new pitch's harmonics must beat the old pitch's, there must be no strong partial an octave below, the old fundamental must actually disappear, and the new fundamental must actually exist; (4) a glitch needs an abrupt step (>2 semitones between 5.8 ms frames) — smooth V-shaped dips at note joins are natural; (5) skip very quiet voiced chunks (breaths); (6) if ≥4 raw candidates fall within ±0.5 s the pitch is unreliable (harmony/noise) — counted separately. Result: 39 → 0 on the clean demo, 6 → 0 on the harmony track; self-test still finds the planted 45 ms glitch.
- Pitch tracker no longer invents a pitch in near-silence (a song-ending silence had been read as E6 for 1.4 s).
- Clicks: when a loud note stops dead and leaves a tick, it is now labelled "노트 끝 틱" (note-end tick) instead of a generic click. In full mixes, ≥4 clicks within 10 s are merged into one "clustered ticks" line (often a deliberate stutter/glitch edit).
- Self-test still 7/7 PASS, 0 extra.

## 2026-10-24 — second real song (full mix, lossy opus recording, 4:02) — v1.0.1
- −8.8 LUFS, G minor, ~123 BPM.
- 3 of 11 "clicks" were note/drum onsets → clicks are now skipped when the level 3–23 ms after the spike is >1.5× (≈+3.5 dB) the level before it (a real click has similar level on both sides). Remaining 8 were very short (2–5 ms) high-frequency bursts that look like real ticks.
- Self-test still 7/7 PASS.

## 2026-10-24 — first real song (full mix, lossy opus recording, 2:26)
- −6.5 LUFS, F minor, ~172 BPM.
- False positives fixed: (1) lossy decode overshoot reported as hundreds of "clipping" lines → lossy formats now give one summary line at "listen and decide"; (2) hi-hats / consonants flagged as clicks in a full mix → clicks downgraded in mix mode; (3) a deliberate gate/stutter effect flagged as abrupt cut → message now says it is fine if intentional.

## 2026-10-24 — synthetic self-test
- 21.5 s synthetic vocal, 7 planted problems (off-pitch +40 cents, click, clipping, 45 ms pitch glitch, abrupt cut, sibilance burst, cut-off ending): 7/7 found, 0 extra.
- Fixes during tuning: added fine yin pitch track (pyin alone smoothed out short glitches); ignore glitches during fades; mono files no longer lose 3 dB on decode (which had hidden clipping); sibilance threshold relaxed from −2 dB to −6 dB.
- Not yet tested on a real vocal-only synth export.
