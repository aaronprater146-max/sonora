# Sonora

**A free, offline, unlimited music generator.** No model download, no API key, no
credits, no rate limit. Every note is synthesised from DSP code in this repo.

It does not sound like a chiptune. The instruments are physical models and
virtual-analogue synths (PolyBLEP band-limited oscillators, Zavalishin/TPT
ladder filters, Karplus-Strong waveguides, modal piano with inharmonicity,
formant-based voices), the space is real convolution against generated impulse
responses, and the master bus is a genuine glue → multiband → limiter chain
normalised to a target LUFS.

```bash
pip install -r requirements.txt
python3 tools/make_song.py --style cinematic-pop --key Am --out my_song.flac
```

That is the whole thing. Run it as many times as you want; each run is free.

---

## Sounds

`python3 tools/make_song.py --list`

| style | bpm | what it is |
|---|---|---|
| `cinematic-pop` | 84–96 | hybrid orchestral / modern pop, trailer-friendly |
| `trap-dark` | 68–78 | 808s, fast hats, choir tops |
| `edm-festival` | 124–130 | supersaws, offbeat bass, heavy pumping |
| `lofi-keys` | 72–84 | dusty rhodes, soft drums, tape-ish |
| `orchestral` | 76–92 | strings, brass, taiko, choir |
| `synthwave` | 100–116 | analogue arps, neon pads |
| `ambient-piano` | 60–72 | felt piano, strings, wordless voice |
| `world-fusion` | 88–104 | harp, kalimba, handpan, flute, sitar |
| `gospel-soul` | 74–88 | rhodes + hammond + choir |
| `rock-anthem` | 104–124 | power chords, big drums, gang chorus |

```bash
# the same seed always gives the same song
python3 tools/make_song.py --style trap-dark --seed 42 --out trap.flac

# longer arrangement, force a tempo
python3 tools/make_song.py --style orchestral --shape epic --bpm 84 --out epic.flac

# use the pre-rendered sample library instead of live synthesis (~5x faster)
SONORA_SAMPLES=1 python3 tools/make_song.py --style ambient-piano --out amb.flac
```

Output formats: `.wav` (24-bit), `.flac`, `.ogg`.

---

## What is actually inside

```
src/sonora/
  core.py        DSP: oscillators, filters, reverbs, compressors, loudness
  theory.py      scales, chords, voice-leading, motifs, rhythm templates
  arrange.py     10 styles, section planner, stem rendering, the mix
  master.py      bus processing and loudness
  sampler.py     plays the sample library
  instruments/
    drums.py     14 synthesised drums (808/909-style, acoustic, taiko…)
    strings.py   bowed section, spiccato, tremolo, Karplus-Strong plucks
    keys.py      piano, felt piano, rhodes, wurlitzer, hammond, clav, bells
    synth.py     virtual analogue synth + 13 production presets
    guitar.py    steel, nylon, electric with cabinet sim, strums, chugs
    voice.py     formant (source-filter) singer + choir
    world.py     flutes, kalimba, koto, sitar, handpan, harmonica…
tools/
  make_song.py        render a complete song
  build_samplelib.py  render the sound library into samplelib/
  play_instrument.py  audition any one of 60 instruments
  fetch.sh            clone → run → delete (for tiny workspaces)
samplelib/       302 generated samples, CC0
songs/           rendered songs
```

### Why it does not sound like a toy

* **Band-limited oscillators.** Saws are PolyBLEP-corrected, so bright leads
  have no aliasing to fold back into the low mids.
* **Stable resonant filters.** The 4-pole ladder uses `G = g/(1+g)` one-pole
  sections, so high resonance screams instead of exploding into a square wave.
* **Real waveguides.** Plucked strings are Karplus-Strong with fractional
  delay, running as a *block* recursion (each period is one vectorised numpy
  operation) — genuine string physics at ~1000× the speed of a sample loop.
* **Detuned ensembles.** A "violin section" is 8 independent players with
  their own vibrato phase, timing and bow noise, seated across the stereo field.
* **Voice leading.** Chords are voiced as an inversion chosen to minimise
  movement from the bar before, so progressions flow instead of lurching.
* **Macrodynamics.** Sections are automated before the master bus, so verses
  really are quieter than choruses (about 9 dB of movement) instead of being
  flattened by the limiter.
* **A real master chain.** Glue compression, 3-band control, stereo width,
  bus saturation, lookahead limiting, and BS.1770-gated loudness normalisation.

---

## The sound library

`samplelib/` holds 302 samples in 15 packs (piano, violin, cello, choir,
harp, guitar, flute, glockenspiel, kalimba, handpan, rhodes, felt piano…).

Every one of them was **rendered by this repo's own synthesisers** — nothing
was recorded, downloaded, or lifted from anywhere. That is what makes them
safe to redistribute: they are original works, published under **CC0**
(see `LICENSE-SAMPLES`). Rebuild or extend them at any time:

```bash
python3 tools/build_samplelib.py --all                 # every pack
python3 tools/build_samplelib.py --pack piano --every 3 --rr 4
```

`every 3` samples every minor third; `rr` is the number of round-robin
variants (they stop repeated notes from machine-gunning). Set
`SONORA_SAMPLES=1` and the arranger plays these instead of synthesising live.

---

## Using it from an agent with a tiny workspace

`tools/fetch.sh` pulls the tool, runs it, and deletes it again:

```bash
./tools/fetch.sh make_song.py -- --style edm-festival --seed 7 --out /tmp/s.flac
```

The clone lands in a temp dir and is removed on exit, so the workspace only
ever holds the finished audio.

---

## Honest limitations

* **No sung lyrics.** `voice.py` is a formant synthesiser: it produces
  convincing wordless "ahh/ooo" choirs and vocal pads, but it cannot pronounce
  words. Real lyric synthesis needs a large generative model (ACE-Step, YuE…)
  running on a GPU. Sonora is the CPU-only, zero-dependency, unlimited
  alternative — and a vocals engine can be added behind the same arranger.
* **CPU cost.** Roughly 0.5–1× realtime on two cores: a 2:40 song takes about
  90 seconds. With `SONORA_SAMPLES=1` it is several times faster.
* Quality is "good demo / strong sync brief" rather than "Grammy master".
  It is a synthesiser, not a mixing engineer.

## Licence

Code: MIT. Generated samples and songs: CC0 (public domain).
No third-party audio, models or data are used anywhere.
