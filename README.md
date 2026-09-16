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
| `slow-groove` | 58–62 | half-time industrial groove: swung closed hats, clicks, gated |
| `destructed-drums` | 118–122 | dry gated one-two beat, damaged only in the loud sections |
| `industrial-rock` | 80–88 | static synths, glitched edits, tight dry drums |
| `techno-rave` | 138–150 | four-on-the-floor, offbeat open hats, 303 acid line, hoover drops |
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
  arrange.py     14 styles, section planner, stem rendering, the mix
  ears.py        the analysis library: loudness, rhythm, melody, harmony,
                 texture, structure -- see the three tools below
  feel.py        rhythm analysis -- onset rate, kick grid, so "is this
                 actually a song or one long note?" is a measurement,
                 not an opinion
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
* **A backwards reverb, as a technique.** `core.reverse_reverb()` reverses a
  phrase, runs it through a long room and reverses the result back -- so the
  room's decay runs *into* the note instead of away from it, and a tail
  becomes a swell that grows out of nothing.  Cut the swell off at its peak
  and the ear hears something reversed: it curves up and snaps back to
  silence, leaving a ghost of the phrase behind it.  It is used per-riff
  (built from each bar's own audio), at section boundaries (cut on the
  downbeat of a drop), and as a whisper of a six-second room on the master.
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

## Songs

`songs/` holds rendered tracks with the exact command that made them. Every
one is free to use for anything, commercial included (CC0).

## Honest limitations

* **No sung lyrics.** `voice.py` is a formant synthesiser: it produces
  convincing wordless "ahh/ooo" choirs and vocal pads, but it cannot pronounce
  words. Real lyric synthesis needs a large generative model (ACE-Step, YuE…)
  running on a GPU. Sonora is the CPU-only, zero-dependency, unlimited
  alternative — and a vocals engine can be added behind the same arranger.
* **CPU cost.** About 0.25× realtime on two cores: a 2:59 song renders in ~40 s.
  With `SONORA_SAMPLES=1` it is several times faster still.
* **Memory.** Roughly 4 bytes per sample per channel plus headroom — a three
  minute stereo mix peaks around 900 MB of RSS. Everything streams in blocks,
  but a 5 minute `--shape epic` render wants about 2 GB.
* Quality is "good demo / strong sync brief" rather than "Grammy master".
  It is a synthesiser, not a mixing engineer.

## Licence

Code: MIT. Generated samples and songs: CC0 (public domain).
No third-party audio, models or data are used anywhere.

## The three analysis tools

Generated music hides its faults in the mix.  These are for finding them, and
for taking apart a reference track you want to sound like.

### 1. analyze -- understand a song

    python3 tools/analyze.py song.mp3

```
  song.mp3
  120s window (measured at 19-139s)   83 bpm   D# minor   -13.8 LUFS   crest 10.7 dB

  [  ok  ] melody moves 1.6 notes/s
          range 19 semitones, held 0.21s a note
  [  ok  ] rhythmic: 8.4 onsets/s
  [  ok  ] arrangement breathes (8.1 dB)
  [  ok  ] dynamics intact (crest 10.7 dB)
```

Four checks: **melody** (does the pitch actually move?), **rhythm** (how many
transients per second), **arrangement** (is there any contrast between
sections?), **dynamics** (crest factor -- is the limiter doing all the work?).
`--fail` exits non-zero if any check fails, so you can batch-filter renders.
A sustained pad scores **under 0.6 notes/s** and is reported as
`FAIL: no melody -- this is a sustained note`.  That is the most useful number
here, and it is the one that caught our own output.

### 2. split -- hear each instrument

    python3 tools/split.py song.mp3 --out /tmp/stems

Writes `kick`, `snare`, `hats`, `percussive`, `harmonic`, `texture` and
`tonal` as separate FLACs, using harmonic/percussive separation plus a band
split and a tonal/noise split.  No model downloads, no weights, no GPU.

### 3. mimic -- describe an instrument well enough to rebuild it

    python3 tools/mimic.py /tmp/stems/kick.flac
    python3 tools/mimic.py /tmp/stems/tonal.flac

Measures a drum (fundamental, pitch sweep, ring, click, noise ratio) or a tone
(pitch, harmonic series, wave shape, filter cutoff, envelope, drive) and prints
a Sonora preset you can paste straight into `synth.py` or the drum kit.

There is also `tools/stems.py`, which renders Sonora's *own* layers
separately.  That is what found the bug this release fixes: the pad was
sitting level with the drums and 2.6 dB above the lead, so all you could
hear was the wash.

## Known limits

* Tempo is estimated from the onset envelope and can land on the wrong octave
  for sparse arrangements.  It is reliable on a dense, loud section.
* `split` is harmonic/percussive separation, not source separation -- it will
  not peel a vocal out of a full mix.  It needs roughly 2 GB of RAM for a
  four-minute song; pass `--window 60` on a small machine.
* `mimic` measures what it hears.  A drum's ring lives in the harmonic stem,
  so measure the original mix if you want the decay.
