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
no lead.

The kit is a plain one-two backbeat (kick on 1 and 3, snare on 2 and 4, hats on
eighths, with breakbeat-ish displacement and a fill every four bars in the
driving sections).  It is **dry and gated**: reverb send is 0.05, the kit's own
room is 0.02, and `C.gate` cuts anything that is not loud enough to earn it.
Measured hit-to-gap level went from 20.7 dB to 49.1 dB and true silence from
19% of the runtime to 42% -- that is the difference between hearing a drum and
hearing a wash.

The damage is an accent, not a constant.  `C.mangle` now takes a *share of hits*
rather than a severity, and that share ramps with section energy: the intro and
the breakdowns are clean (0%), the verses get fills, and the choruses are where
things come apart (55%).  Like-for-like hit variation is 0.22 clean against
0.32 damaged, while the intro measures 0.17.

    python3 tools/make_song.py --style destructed-drums --key 'Fm' --bpm 120 --seed 4 --out submerged.flac
