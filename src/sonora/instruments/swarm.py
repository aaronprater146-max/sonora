"""Insect swarms, synthesised from nothing.

Every fly in here is generated: a buzzing wingbeat that wanders in pitch,
in distance and in position, summed by the dozen into a swarm.  Nothing is
sampled, nothing is downloaded and there is no recording to licence -- the
sound is CC0 by construction, the same as the rest of the library.

The swarm is meant to run forever.  `swarm_loop()` hands back a buffer whose
end folds seamlessly into its start, and `frozen()` hands back that same
swarm held in a reverb that never lets go.
"""
from __future__ import annotations

import numpy as np

from .. import core as C

FLOAT = C.FLOAT


def drift(n: int, seed: int, rate: float = 0.7,
          lo: float = 0.0, hi: float = 1.0) -> np.ndarray:
    """smoothed random walk: a new control point every 1/rate seconds"""
    g = np.random.default_rng(seed)
    k = max(2, int(n * rate / C.SR) + 2)
    pts = g.uniform(lo, hi, k + 2)
    idx = np.linspace(0.0, float(k), n, endpoint=False)
    i0 = idx.astype(np.int64)
    f = idx - i0
    return pts[i0] * (1.0 - f) + pts[i0 + 1] * f


def fly(dur: float, seed: int, f0: float | None = None) -> np.ndarray:
    """One fly.  A wingbeat is a pulse train -- buzz plus air -- and no two
    beats are quite the same, which is the whole reason a real swarm never
    sounds like a loop."""
    n = C.sec(dur)
    g = np.random.default_rng(seed)
    f0 = float(f0) if f0 else float(g.uniform(165.0, 430.0))
    t = np.arange(n) / C.SR
    # the pitch wanders: flies do not hold a note
    f = f0 * (1.0 + 0.20 * drift(n, seed + 1, rate=1.7, lo=-1.0, hi=1.0))
    ph = 2 * np.pi * np.cumsum(f) / C.SR
    y = np.zeros(n, dtype=np.float64)
    for h in range(1, 7):                       # a wing closing is a pulse
        y += np.sin(h * ph) / (h ** 1.3)
    air = C.bandpass(C.noise(n, seed + 2), f0 * 5.5, 0.7)
    beat = 0.5 + 0.5 * np.sin(2 * np.pi * f0 * 0.5 * t + g.uniform(0.0, 6.283))
    y = y * (0.65 + 0.35 * beat) + air * 0.40 * beat
    # it flies about: slow changes in distance, faster tremolo on top
    y *= 0.15 + 0.85 * drift(n, seed + 3, rate=0.5) ** 1.6
    y *= 0.55 + 0.45 * drift(n, seed + 4, rate=2.4)
    m = np.max(np.abs(y))
    return (y / (m + 1e-9)).astype(FLOAT)


def swarm_loop(seconds: float = 9.31, seed: int = 1, flies: int = 16,
               xfade: float = 1.6, spread: float = 0.85) -> np.ndarray:
    """A seamless stereo loop of a swarm.

    It is generated `xfade` seconds longer than asked for and the overhang is
    folded back over the head, equal power, so the join cannot be heard.
    Two loops of coprime length layered together stop the ear from ever
    catching the repeat.
    """
    n = C.sec(seconds + xfade)
    out = np.zeros((n, 2), dtype=np.float64)
    g = np.random.default_rng(seed)
    for i in range(flies):
        y = fly(seconds + xfade, seed + 37 * i).astype(np.float64)
        base = float(g.uniform(-spread, spread))
        pan = np.clip(base + 0.30 * drift(n, seed + 91 * i, rate=0.3,
                                          lo=-1.0, hi=1.0), -1.0, 1.0)
        l = np.sqrt(np.clip(0.5 * (1.0 - pan), 0.0, 1.0))
        r = np.sqrt(np.clip(0.5 * (1.0 + pan), 0.0, 1.0))
        out[:, 0] += y * l
        out[:, 1] += y * r
    L, k = C.sec(seconds), C.sec(xfade)
    loop = out[:L].copy()
    a = np.linspace(0.0, np.pi / 2.0, k)
    loop[:k, 0] = out[:k, 0] * np.sin(a) + out[L:L + k, 0] * np.cos(a)
    loop[:k, 1] = out[:k, 1] * np.sin(a) + out[L:L + k, 1] * np.cos(a)
    m = np.max(np.abs(loop))
    return (loop / (m + 1e-9)).astype(FLOAT)


def steady_room(loop: np.ndarray, ir: np.ndarray) -> np.ndarray:
    """The reverb of a signal that loops forever.

    Feed a short room with one period and it is still ringing when the next
    one starts, so the honest answer is the sum of every period's tail,
    wrapped.  Folded like this it is exact and it costs one short
    convolution instead of a four minute one.
    """
    L = loop.shape[0]
    reps = int(np.ceil(ir.shape[0] / L)) + 1
    z = C._conv(np.tile(np.asarray(loop, dtype=FLOAT), (reps, 1)), ir)
    out = np.zeros((L, 2), dtype=np.float64)
    for j in range(reps + 1):
        seg = z[j * L:j * L + L]
        out[:seg.shape[0]] += seg
    return out


def frozen(loop: np.ndarray, seed: int = 1, feedback: float = 0.985,
           delay: float = 0.31, damp: float = 5200.0,
           gens: int = 260) -> np.ndarray:
    """The swarm, held in a room that never lets go.

    Freeze is just regeneration: whatever comes out goes back in, a little
    quieter and a little darker, forever.  Run it on a loop and the answer is
    a loop, so it can be computed once and tiled for the whole record.
    """
    ir = C.make_ir(seconds=2.6, decay=1.0, predelay=0.02, size=0.9,
                   damping=0.55, brightness=0.40, early=12, seed=seed)
    steady = steady_room(loop, ir)
    del ir
    L = steady.shape[0]
    # Two periods, and only the second one is kept.  The damping filter and
    # the chorus both need a run-up, and a run-up on a looping buffer is a
    # click at the join: generation after generation of them add up to a
    # seam you can hear.  Running on two periods and throwing the first away
    # gives the true periodic answer, so the freeze loops as cleanly as the
    # swarm it was made from.
    x = np.tile(steady, (2, 1))
    out = x.copy()
    cur = x
    D = int(delay * C.SR) % L
    ref = float(np.max(np.abs(x))) + 1e-12
    for _ in range(gens):
        cur = np.roll(cur, D, axis=0)
        cur = C.lowpass(cur, damp)
        cur = cur * feedback
        out += cur
        if float(np.max(np.abs(cur))) < 1e-4 * ref:
            break
    # a held tail should still move, or it is a sample and not a room
    out = C.stereo_chorus(out.astype(FLOAT), rate=0.07, depth=0.006,
                          mix=0.5, seed=seed + 5)
    out = out[L:2 * L]
    m = float(np.max(np.abs(out)))
    return (out / (m + 1e-9)).astype(FLOAT)
