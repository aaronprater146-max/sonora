# songs

Every track here was rendered by `tools/make_song.py`. CC0 -- do anything
with them, including commercially, with no attribution.

| track | style | length | command |
|---|---|---|---|
| `aurora.flac` | cinematic-pop | 2:59 | `python3 tools/make_song.py --style cinematic-pop --key Am --seed 146 --out aurora.flac` |
| `nightshift.flac` | trap-dark | 2:20 | `python3 tools/make_song.py --style trap-dark --key Fm --shape short --seed 21 --out nightshift.flac` |
| `neon.flac` | edm-festival | 1:16 | `python3 tools/make_song.py --style edm-festival --key Em --shape short --seed 5 --out neon.flac` |

Same seed + same arguments = the same track, every time.
