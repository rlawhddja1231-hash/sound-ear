# sound-ear

Lets Claude "hear" audio files. The bundled script turns songs, vocals, sound effects or video soundtracks into numbers and pictures (waveform, spectrogram, pitch curve) so Claude can find problems with exact timestamps: clicks, clipping, abrupt cuts, pitch glitches, off-pitch notes, wobbly notes, harsh sibilance, loud breaths, cut-off endings and loudness issues. It also estimates BPM, key and vocal range, transcribes lyrics (whisper.cpp), describes song structure, drums, mood and lyric hooks like a listener would, cuts short snippets so you can hear each spot, and compares two versions after you fix them. Built for checking vocal-synth songs (Synthesizer V, VOCALOID, UTAU, CeVIO) before release.

Setup: needs Python 3 and ffmpeg; the skill installs the Python packages it needs.

Usage: attach an audio file and ask Claude to listen and find problems. For pitch checks, give a vocal-only export.

Full guide (Korean + English): https://github.com/rlawhddja1231-hash/sound-ear
