# Field log

Newest first. Record date, file type, what was found, false positives, misses, and threshold changes.

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
