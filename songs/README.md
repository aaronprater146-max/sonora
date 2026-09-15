# songs

Every track here was rendered by `tools/make_song.py`. CC0 -- do anything
with them, including commercially, with no attribution.

| track | style | length | command |
|---|---|---|---|
| `aurora.flac` | cinematic-pop | 2:59 | `python3 tools/make_song.py --style cinematic-pop --key Am --seed 146 --out aurora.flac` |
| `nightshift.flac` | trap-dark | 2:20 | `python3 tools/make_song.py --style trap-dark --key Fm --shape short --seed 21 --out nightshift.flac` |
| `neon.flac` | edm-festival | 1:16 | `python3 tools/make_song.py --style edm-festival --key Em --shape short --seed 5 --out neon.flac` |

Same seed + same arguments = the same track, every time.

## Techno / rave

| file | key | bpm | length | notes |
|------|-----|-----|--------|-------|
| `afterhours.flac` | Am | 142 | 2:03 | rolling kick, 16th acid line, snare-roll builds, 9 dB breakdown |
| `strobe.flac`     | Fm | 141 | 2:05 | darker, hoover stabs, biggest drop contrast |

Reproduce exactly:

    ./tools/fetch.sh make_song.py -- --style techno-rave --key Am --seed 7  --out afterhours.flac
    ./tools/fetch.sh make_song.py -- --style techno-rave --key Fm --seed 146 --out strobe.flac
