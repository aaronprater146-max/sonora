# songs

Every track here was rendered by `tools/make_song.py`. CC0 -- do anything
with them, including commercially, with no attribution.

| track | style | length | command |
|---|---|---|---|
| `aurora.flac` | cinematic-pop | 2:59 | `python3 tools/make_song.py --style cinematic-pop --key Am --seed 146 --out aurora.flac` |
| `nightshift.flac` | trap-dark | 2:20 | `python3 tools/make_song.py --style trap-dark --key Fm --shape short --seed 21 --out nightshift.flac` |
| `neon.flac` | edm-festival | 1:16 | `python3 tools/make_song.py --style edm-festival --key Em --shape short --seed 5 --out neon.flac` |
| `submerged.flac` | destructed-drums | 4:10 | `python3 tools/make_song.py --style destructed-drums --key 'Fm' --bpm 120 --seed 4 --out submerged.flac` |
| `undertow.flac` | slow-groove | 4:10 | `python3 tools/make_song.py --style slow-groove --key 'Fm' --bpm 60 --seed 4 --out undertow.flac` |
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

## The half-time groove

`undertow.flac` is `submerged` at half speed (60 bpm) with a jazz-leaning
groove: kick on 1 and 3, snare on 2 and 4, and a swung closed-hat ride on
1, 2, 2-and, 3, 4, 4-and.

There is **no open hat anywhere** -- not in the kit, not in the grids.  An open
hat landing on the same beat every bar is the most tiring sound in programmed
drums.  Instead the closed hats are humanised three ways at once:

* **swing** -- the eighth-note offbeats are pushed late, so the "and" of 2
  lands at step 6.41 instead of 6.0
* **velocity** -- downbeats sit near 1.0, offbeats near 0.55, each with ±20%
  of random drift on top
* **hat_vary** -- each bar independently drops some offbeats and leans on
  others, so no two bars of hats are identical

Measured on three consecutive bars:

    bar 1   0.0  4.0        8.0  12.0  14.41     <- the "and" of 2 sat out
    bar 2   0.0  4.0  6.41  8.0  12.0  14.41
    bar 3   0.0  4.0  6.41  8.0  12.0            <- the "and" of 4 sat out

The gate threshold had to come down from 0.17 to 0.11 to stop it eating the
softer offbeat hats -- the gate is normalised to the loudest hit, and a closed
hat is a lot quieter than a kick.

## The room, and the backwards swells

**The noise sweep is gone.** The build used to be `reverse_sweep()` --
bandpassed noise rising over two seconds.  It sounded like a backwards open
hat because that is more or less what it was.  `render_fx()` no longer plays
it at all.

**The end drone is gone.** The outro dropped a sliding saw tone under the last
bars: two and a half seconds of one pitched note, the only pitched sound on a
drum record.  Off for this style (`outro_tail=False`).

**There is a room now.** A six-second, dark, predelayed room
(`IRS["infinite"]`) on the master at `space_db=-46`, band-limited to
320 Hz-5.2 kHz so it cannot smear the kick or add fizz, and predelayed 55 ms
so it never touches a transient.  It is levelled by **measured RMS, not by
peak**: this room is six seconds of dense tail, and peak-normalising it put
the wash in *front* of the drums -- the one thing that must never happen on a
record whose whole point is the drum.

    measured on the finished file
      hit-to-gap        53.3 dB   (83.4 with dead silence; 12.6 on the first try)
      wash in the gaps   -50 dBFS, 39 dB under the hits
      wet rms           -45.7 dBFS

**And the swells.** `core.reverse_reverb()` is the technique: reverse the
phrase, run it through a long room, reverse the result back.  A room's decay
normally runs away from a hit; reversed it runs into it, so a tail becomes a
swell that grows out of nothing.  Cut it off at its peak -- 50 ms, hard -- and
the ear hears something reversed: it curves up and snaps back to silence.

It is used at two scales, and neither one is a master effect:

* **per riff** (`riff_swells`) -- each swell is built out of the bar it belongs
  to, so it carries that riff's own colour.  Roughly one bar in four gets one,
  at random: 18 of them on this track.
* **per section** (`section_swells`) -- taken from the mix of the two beats
  before a drop and cut on its downbeat.  4 of them.  These replaced the
  noise build.

One swell, measured in isolation:

    0.00s  -75 dB
    1.00s  -49 dB   ####################
    1.25s  -41 dB   ############################
    1.50s  -34 dB   ###################################
    1.75s  -24 dB   #############################################
    1.95s  -23 dB   ##############################################   <- cut

The cut is anchored on the swell's own loudest moment, because a riff that
stops playing half a beat early would otherwise leave a hole of silence
exactly where the cut is supposed to land, and the trick disappears.

Knobs, all on the style:

    space_db=-45.0        master room: wet RMS in dBFS, at full open
    swell=0.34            per-riff swell level
    swell_density=0.55    roughly one every four bars
    outro_tail=False      no end drone

## The build is the return fader now

The section-boundary swell is gone too -- `section_swells()` was deleted.
Between 46 and 56 seconds the old record stacked a noise riser, a ghost swell
built out of that riser, a sub boom and a crash on the downbeat of the first
chorus, and did it again at 132 s and 208 s: three synthesised noises on every
drop.

All of it is gone, everywhere, for this style:

    riser=False        no noise sweep across the pre-chorus
    impact=False       no sub boom on the downbeat
    chorus_crash=False no crash cymbal on the downbeat
    outro_tail=False   no end drone

(They default to `True`, so the other styles keep their builds.)

What replaces it is `reverb_ride()` -- the master reverb's dry/wet, drawn as
an arrangement decision rather than as an effect.  Almost dry at the top, it
opens slowly across the first minute, is wide open on the downbeat of the
first drop, and snaps shut in 120 ms.  Then it goes back to nearly dry and
opens again, faster each time, because by then the listener knows what is
coming.

The room, measured out of the master:

      0-16s   -74.7 dBFS   ####                     intro: essentially dry
     30-40s   -48.9 dBFS   #####################    verse: opening
     44-48s   -41.9 dBFS   ##########################
     48-52s   -41.7 dBFS   ##########################   the swell (peak -21.8)
     56-64s   -64.7 dBFS   ##########               snapped shut: -23 dB
    100-112s  -72.1 dBFS   #####                    bridge: almost dry again
    126-132s  -42.8 dBFS   #########################    quicker build
    136-144s  -63.7 dBFS   ##########               shut
    200-208s  -44.8 dBFS   ########################     quicker still
    212-220s  -60.5 dBFS   #############            shut
    232-248s  -57.7 dBFS   ##################       outro: opens and stays

## The swarm

A fly swarm runs under the whole record.  It is the one thing that does not
go through the master room: it is added after the room, and it has its own,
which is frozen.

**The flies.** `instruments/swarm.py`.  Each one is a buzzing pulse train --
six harmonics of a wingbeat between 165 and 430 Hz, plus filtered air -- and
each wanders in pitch, drifts in distance and moves in the stereo field on its
own smoothed random walk.  No two beats are the same, which is why a real
swarm never sounds like a loop.  28 of them in one loop, 22 in the other.
Nothing is sampled and nothing is downloaded: there is no recording in it to
license, so it is CC0 the same as the rest of the library.

**Seamless.** Each loop is generated `xfade` seconds longer than asked for and
the overhang is folded back over the head, equal power, so the join measures
quieter than a normal step between samples.  The two loops are 9.31 s and
13.77 s -- lengths with no common measure, so the pair does not come round
again inside four minutes.

**The freeze.** Freeze is just regeneration: whatever comes out goes back in,
a little quieter and a little darker, forever (feedback 0.985, 0.31 s round
trip, damped each pass).  Because the input loops, the answer loops, so it is
computed once and tiled.  Two details matter:

* the room is fed one period and folded, because it is still ringing when the
  next period starts -- that is `steady_room()`
* the regeneration runs on **two** periods and only the second is kept.  The
  damping filter and the chorus both need a run-up, and a run-up on a looping
  buffer is a click at the join; generation after generation of them added up
  to a seam you could hear (0.26 against a 99.9th-percentile step of 0.05).
  After the fix: 0.017.

**Consumed.** The dry flies fade out on exactly the curve the frozen ones fade
in (equal power, so there is no dip in the middle), and the bed lifts a
little on the way because it is meant to be taking over.  By 78% of the
record the dry swarm is 5% of what it was and the freeze is everything.

    fly bed, measured out of the master
      0-20s   -51.0 dBFS   start: dry flies
     60-76s   -47.6 dBFS
    140-156s  -46.1 dBFS
    200-220s  -43.7 dBFS   the freeze has taken over (+7 dB)
    232-248s  -47.3 dBFS   outro

Hit-to-gap is 43.5 dB.  It was 53.3 dB before the swarm existed and 12.6 dB
the first time the room was levelled by peak instead of by RMS, so the drum
is still 43 dB clear of everything underneath it.

Knobs:

    space_db=-45.0        master room at full open
    ride / ride_build     the return fader; builds of 100%, 30%, 18% of each gap
    fly_db=-57.0          the dry swarm
    fly_freeze_db=-54.0   the swarm, frozen
    fly_consume=0.78      when the handover finishes, as a fraction of the song

The swarm is deliberately quiet -- about 34 dB under the music.  Raise
`fly_db` toward -50 if it should be more of a feature.
