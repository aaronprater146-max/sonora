# songs

Every track here was rendered by `tools/make_song.py`. CC0 -- do anything
with them, including commercially, with no attribution.

| track | style | length | command |
|---|---|---|---|
| `aurora.flac` | cinematic-pop | 2:59 | `python3 tools/make_song.py --style cinematic-pop --key Am --seed 146 --out aurora.flac` |
| `nightshift.flac` | trap-dark | 2:20 | `python3 tools/make_song.py --style trap-dark --key Fm --shape short --seed 21 --out nightshift.flac` |
| `neon.flac` | edm-festival | 1:16 | `python3 tools/make_song.py --style edm-festival --key Em --shape short --seed 5 --out neon.flac` |
| `submerged.flac` | destructed-drums | 4:10 | `python3 tools/make_song.py --style destructed-drums --key 'Fm' --bpm 120 --seed 4 --out submerged.flac` |
| `ghostyear.flac` | industrial-rock | 4:05 | `python3 tools/make_song.py --style industrial-rock --key 'D#m' --bpm 83 --seed 9 --out ghostyear.flac` |

Same seed + same arguments = the same track, every time.

## Techno / rave

| file | key | bpm | length | notes |
|------|-----|-----|--------|-------|
| `afterhours.flac` | Am | 142 | 2:03 | rolling kick, 16th acid line, snare-roll builds, 9 dB breakdown |
| `strobe.flac`     | Fm | 141 | 2:05 | darker, hoover stabs, biggest drop contrast |

Reproduce exactly:

    ./tools/fetch.sh make_song.py -- --style techno-rave --key Am --seed 7  --out afterhours.flac
    ./tools/fetch.sh make_song.py -- --style techno-rave --key Fm --seed 146 --out strobe.flac

## Drum-only tracks

`submerged.flac` is drums and percussion and nothing else -- no pad, no bass,
no lead.  Every hit is damaged individually by `C.mangle`: reversed, pitched up
or down, bit-crushed, ring-modulated or cut short, with the probabilities
weighted so nothing repeats the same way twice.  Measured hit-to-hit spectral
variation is 0.275 against 0.144 for the same kit left alone.

    python3 tools/make_song.py --style destructed-drums --key 'Fm' --bpm 120 --seed 4 --out submerged.flac
