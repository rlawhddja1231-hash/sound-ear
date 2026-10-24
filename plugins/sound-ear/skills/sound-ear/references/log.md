# Field log

Newest first. Record date, file type, what was found, false positives, misses, and threshold changes.

## 2026-10-24 — first real song (full mix, lossy opus recording, 2:26)
- −6.5 LUFS, F minor, ~172 BPM.
- False positives fixed: (1) lossy decode overshoot reported as hundreds of "clipping" lines → lossy formats now give one summary line at "listen and decide"; (2) hi-hats / consonants flagged as clicks in a full mix → clicks downgraded in mix mode; (3) a deliberate gate/stutter effect flagged as abrupt cut → message now says it is fine if intentional.

## 2026-10-24 — synthetic self-test
- 21.5 s synthetic vocal, 7 planted problems (off-pitch +40 cents, click, clipping, 45 ms pitch glitch, abrupt cut, sibilance burst, cut-off ending): 7/7 found, 0 extra.
- Fixes during tuning: added fine yin pitch track (pyin alone smoothed out short glitches); ignore glitches during fades; mono files no longer lose 3 dB on decode (which had hidden clipping); sibilance threshold relaxed from −2 dB to −6 dB.
- Not yet tested on a real vocal-only synth export.
