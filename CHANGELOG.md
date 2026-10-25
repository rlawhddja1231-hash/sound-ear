# 바뀐 것 (Changelog)

[English below](#english)

판마다 무엇이 바뀌었는지 적은 기록입니다. 받는 곳: [Releases](https://github.com/rlawhddja1231-hash/sound-ear/releases) — Claude Code는 `/plugin` 업데이트, claude.ai는 새 `sound-ear.zip`을 다시 올리면 됩니다.

## v1.7.0 — 2026-10-25 · 일본어 가사, 더 빠른 곡 설명, 보컬만 있는 파일 알아채기
- **갈라낸 목소리 가사**: 조용한 틈마다 25초 안쪽으로 잘라 받아씀 → 일본어 합성 보컬 가사가 1줄에서 37줄로. (완성곡은 자르면 오히려 나빠져 통째로 받아씀)
- **언어 자동 고르기**: 처음 30초가 아니라 노래가 가장 빽빽한 곳으로 정함 — 전주가 길어도 맞음.
- **곡 설명 약 1.8배 빠름** (4분 곡 69초 → 39초, 결과 같음).
- **보컬만 있는 파일을 저절로 알아챔**: 이름에 '보컬'이 없어도 빠른 검사에서 음정까지 봄.
- **피아노 발라드 후렴**: 절과 후렴이 비슷해 전부 '후렴'이라 하던 것 고침.
- 보고서: 같은 종류 🟠가 많으면 한 줄로 묶음. 손실 압축 '0dB' 문구 다듬음.

## v1.6.0 — 2026-10-25 · 쓰기 편하게
- **`setup` 준비 명령**: 프로그램·모델을 한 번에 확인·설치, 못 받은 건 주소 안내, 올린 파일은 제자리에(.part1/.part2 합치기).
- **빠른 검사가 기본** (4분 곡 3~4분), 자세히는 `--deep`(보컬 분리·큰 모델·목소리 음정).
- **보고서 맨 위 한마디 평가** ("꼭 고칠 곳은 없음 👍" 등).
- **예시 보고서** [docs/example-report](docs/example-report/종합보고서.md).
- 고침: 조용한 곡의 뚝 끊김 놓침, 노트 끝 음 튐 헛짚음, 분리 목소리의 틱은 🟠로, 하이햇을 ㅅ·ㅊ로 헛짚음, 큰 모델이 "(♪~♪)"만 내던 것.

## v1.5.0 — 2026-10-25 · 명령 하나로 다 듣기
- `full` 명령 하나로 분리 → 흠 찾기 → 목소리 음정 → 곡 설명 → **종합보고서.md**.
- 큰 받아쓰기 모델(medium) 지원 — 한국어 가사가 훨씬 정확.
- 후렴 찾기 개선(돌아오는 반복·훅 가사), 짧은 곡도 구간 나눔, 합성 완성곡 시험(mixtest) 추가.

## v1.4.0 — 2026-10-25 · 보컬 분리
- 완성곡에서 목소리와 반주 가르기(UVR 모델) → 가사·음정이 더 정확. 조는 반주에서 잼.

## v1.3.1 — 2026-10-25
- 킥 드럼을 딸깍 소리로 헛짚던 것 고침. 곡의 세기를 사람이 듣는 크기(LUFS)로 판단.

## v1.3.0 — 2026-10-24 · 악기·장르·분위기
- 음악 AI(Essentia, 비상업용 모델은 따로 받음)로 악기 40가지·장르 400가지·분위기 56가지를 구간별로.

## v1.2.0 — 2026-10-24 · 가사와 곡 설명
- 가사·말 받아쓰기(whisper.cpp)와 자막 파일(.srt).
- 곡 설명: 전주·절·후렴·간주, 반복, 드럼·저음, 세기 변화, 분위기, 반복되는 가사.

## v1.1.0 — 2026-10-24
- 실제 합성 보컬에서 음 튐 헛짚음을 크게 줄임(39곳 → 0곳), 화음 구간 따로 셈.

## v1.0.1 — 2026-10-24
- 북·건반 첫소리를 딸깍 소리로 헛짚던 것 고침.

## v1.0.0 — 2026-10-24 · 처음 공개
- 소리 크기·클리핑·딸깍·뚝 끊김·음 튐·음정·ㅅ·ㅊ·숨소리·끝 잘림 찾기, BPM·조·음역, 확대 그림과 소리 조각, 두 판 비교.

---

<a id="english"></a>
# English

- **1.7.0** — Stems and vocal-only files transcribed in pause-split ≤25 s pieces (Japanese synth vocal 1 → 37 lines); language detected from the densest singing; `describe` ~1.8× faster; vocal-only files detected automatically; chorus naming fixed for ballads; grouped report rows.
- **1.6.0** — `setup` command; quick `full` by default, `--deep` for separation and vocal pitch; one-line verdict; example report; fixes for missed dropouts, note-end pitch glitches, hi-hat sibilance, separated-vocal clicks, "(♪~♪)" output.
- **1.5.0** — One-command `full` report; medium lyrics model; better chorus detection; synthetic mix test.
- **1.4.0** — Vocal separation (UVR MDX-Net).
- **1.3.1** — Kick drums no longer flagged as clicks; loudness wording from LUFS.
- **1.3.0** — Instruments, genre and mood (Essentia, optional non-commercial models).
- **1.2.0** — Lyrics transcription (whisper.cpp), subtitles, song description.
- **1.1.0** — Far fewer false pitch glitches on real synth vocals.
- **1.0.1** — Note/drum onsets no longer flagged as clicks.
- **1.0.0** — First release.
