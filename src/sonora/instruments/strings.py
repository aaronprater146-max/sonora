"""
String instruments: bowed section, pizzicato, spiccato, tremolo, harp, guitar.

Bowed strings are modal/additive: ~48 partials with per-partial decay,
bow-scrape noise, vibrato with a delayed onset, and a convolution body
resonance.  Plucked strings use Karplus-Strong implemented with a
*block* recursion (the loop delay only depends on samples >= one period
back, so each period can be computed with one vectorised numpy op) --
that gives real waveguide sound at ~1000x the speed of a sample loop.
"""
from __future__ import annotations

import math

import numpy as np

from ..core import (SR, FLOAT, sec, m2f, noise, rng, db2amp, osc_sine, tanh_sat,
                    exp_decay, filt_env, lowpass, highpass, bandpass, peak_eq,
                    fftconvolve1, fade, resample, ensure2, pan, calibrate)

# --------------------------------------------------------------------------
# body resonances (convolution IRs generated, never sampled -> CC0)
# --------------------------------------------------------------------------

BODY = {
    # (freq, gain, Q/decay) resonances + overall body length
    "violin": ([(285, 1.0, 9), (455, .72, 11), (710, .55, 12), (1180, .45, 13),
                (2350, .36, 14), (3300, .3, 15), (4600, .22, 16), (6200, .15, 17),
                (8300, .09, 18)], 0.30),
    "viola": ([(220, 1.0, 8), (350, .7, 10), (560, .5, 11), (900, .35, 12),
               (1750, .28, 13), (2600, .2, 14)], 0.34),
    "cello": ([(125, 1.0, 7), (215, .75, 9), (350, .6, 10), (560, .48, 11),
               (1100, .38, 12), (1750, .3, 13), (2600, .22, 14), (3800, .14, 15),
               (5400, .08, 16)], 0.42),
    "bass": ([(62, 1.0, 6), (110, .7, 8), (185, .5, 9), (300, .35, 10),
              (620, .25, 11), (1000, .18, 12)], 0.5),
    "guitar": ([(98, 1.0, 7), (186, .8, 8), (395, .55, 9), (760, .4, 10),
                (1450, .3, 11), (2400, .2, 12)], 0.36),
    "nylon": ([(105, 1.0, 6), (200, .7, 7), (420, .45, 8), (700, .3, 9),
               (1200, .2, 10)], 0.30),
    "harp": ([(160, 1.0, 8), (320, .7, 9), (640, .5, 10), (1180, .38, 11),
              (2100, .28, 12), (3400, .2, 13)], 0.30),
    "piano": ([(120, 1.0, 6), (240, .65, 7), (480, .55, 8), (900, .45, 9),
               (1600, .36, 10), (2800, .28, 11), (4300, .18, 12), (6400, .1, 13)], 0.28),
}


def body_ir(kind: str = "violin", seconds: float | None = None, seed: int = 3,
            spread: float = 0.0) -> np.ndarray:
    """modal impulse response for an instrument body (mono)"""
    res, length = BODY.get(kind, BODY["violin"])
    seconds = seconds or length
    n = sec(seconds)
    t = np.arange(n) / SR
    ir = np.zeros(n)
    g = rng(seed)
    for k, (f, a, q) in enumerate(res):
        fk = f * (1.0 + spread * 0.02)
        tau = q / (2.0 * math.pi * fk) * 6.0
        ph = g.uniform(0, 6.283)
        ir += a * np.exp(-t / tau) * np.sin(2 * math.pi * fk * t + ph)
    # a little "wood": filtered noise burst at the very start
    nb = min(n, sec(0.012))
    ir[:nb] += bandpass(noise(nb, seed + 1), 1800.0, 1.0) * 0.35
    ir *= np.hanning(n) ** 0.35
    return (ir / (np.max(np.abs(ir)) + 1e-9)).astype(FLOAT)


# --------------------------------------------------------------------------
# bowed strings
# --------------------------------------------------------------------------

def bowed(midi: float, dur: float = 2.0, bright: float = 0.7, attack: float = 0.09,
          vibrato: float = 5.0, vib_depth: float = 9.0, vib_delay: float = 0.22,
          bow_noise: float = 0.5, n_partials: int = 60, players: int = 1,
          detune: float = 6.0, seed: int = 1, body: str = "violin",
          release: float = 0.35, dyn: float = 1.0) -> np.ndarray:
    """one bowed note (mono). `players` > 1 stacks detuned copies."""
    n = sec(dur + release + 0.05)
    t = np.arange(n) / SR
    out = np.zeros(n, dtype=np.float64)
    g = rng(seed)
    for p in range(players):
        if players > 1:
            cents = g.uniform(-detune, detune) if p else g.uniform(-detune * .3, detune * .3)
            off = g.uniform(0, 0.02)
        else:
            cents, off = 0.0, 0.0
        f = m2f(midi) * (2.0 ** (cents / 1200.0))
        # vibrato with delayed onset
        vib_env = np.clip((t - vib_delay) / max(0.05, vib_delay * 2.0), 0.0, 1.0)
        vib_rate = vibrato * (1.0 + 0.06 * (p - players / 2))
        cents_t = vib_depth * vib_env * np.sin(2 * math.pi * vib_rate * t + g.uniform(0, 6.283))
        freq_t = f * (2.0 ** (cents_t / 1200.0))
        ph = np.cumsum(freq_t) / SR
        # amplitude envelope
        env = (1.0 - np.exp(-t / max(0.008, attack * 0.35))) * np.exp(-np.maximum(0.0, t - (dur - 0.02)) / max(0.02, release * 0.5))
        env = np.minimum(env, 1.0)
        # spectrum: sawtooth-ish rolloff, brightened by `bright`
        kc = 9.0 + 46.0 * bright
        taumax = 2.2 + 6.0 * (1.0 - bright)
        sig = np.zeros(n)
        for k in range(1, n_partials + 1):
            ff = f * k
            if ff > SR * 0.46:
                break
            a = (1.0 / k ** (1.05 - 0.42 * bright)) * math.exp(-(k - 1) / kc)
            tau = taumax / (1.0 + (k - 1) * 0.42)
            sig += a * np.exp(-t / tau) * np.sin(2 * math.pi * k * ph + g.uniform(0, 6.283) * (k > 3))
        # bow: scrape transient + continuous breath of the bow on the string
        if bow_noise > 0:
            ns = noise(n, seed + 7 + p)
            scrape = bandpass(ns, 2200 + 2600 * bright, 0.8) * np.exp(-t / (attack * 0.9)) * 0.55
            scratch = bandpass(ns, 4200.0, 1.2) * env * 0.055
            sig += (scrape + scratch) * bow_noise
        sig *= env
        out += sig
    out /= max(1, players) ** 0.62
    ir = body_ir(body, seed=seed + 5)
    out = fftconvolve1(out, ir)[:n]
    out = highpass(out, 90.0 if body in ("cello", "bass") else 150.0, 0.707)
    out = tanh_sat(out * (0.9 * dyn), 1.25)
    return calibrate(fade(out.astype(FLOAT), 0.004, release * 0.7), -6.0, 99.5)


def spiccato(midi: float, dur: float = 0.35, bright: float = 0.7, seed: int = 1,
             body: str = "violin", dyn: float = 1.0) -> np.ndarray:
    """short, bouncy bowed note: fast attack, fast decay, more bow bite"""
    y = bowed(midi, min(0.28, dur * 0.7), bright=bright, attack=0.012,
              vibrato=0.0, vib_depth=0.0, bow_noise=0.9, seed=seed, body=body,
              release=dur * 0.55, dyn=dyn)
    return fade(y, 0.001, 0.04)


def tremolo(midi: float, dur: float = 2.0, rate: float = 9.0, seed: int = 1,
            body: str = "violin", dyn: float = 1.0) -> np.ndarray:
    y = bowed(midi, dur, attack=0.05, seed=seed, body=body, dyn=dyn)
    n = y.shape[0]
    t = np.arange(n) / SR
    am = 0.55 + 0.45 * np.sin(2 * math.pi * rate * t)
    return (y * am).astype(FLOAT)


# --------------------------------------------------------------------------
# plucked strings: block Karplus-Strong
# --------------------------------------------------------------------------

def pluck(freq: float, dur: float = 2.0, t60: float | None = None,
          bright: float = 0.55, pos: float = 0.28, seed: int = 1,
          body: str = "guitar", level: float = 1.0, stretch: float = 0.0,
          exc_bright: float = 0.6) -> np.ndarray:
    """block-vectorised Karplus-Strong plucked string.

    freq      fundamental in Hz
    t60       time to -60 dB (default: dur*1.6)
    bright    0..1 loop-filter damping (1 = mellow, 0 = zingy)
    pos       pluck position along the string (0 = bridge, .5 = middle)
    """
    t60 = t60 or max(0.25, dur * 1.7)
    N = SR / max(20.0, float(freq))
    Ni = int(math.floor(N))
    frac = N - Ni
    if Ni < 2:
        Ni, frac = 2, 0.0
    exc_len = max(Ni + 4, int(Ni * 2.2))
    pad = exc_len + 4
    total = pad + sec(dur + 0.12)
    y = np.zeros(total, dtype=np.float64)

    # excitation: shaped noise burst, comb-filtered by the pluck position
    e = noise(exc_len, seed).astype(np.float64)
    e = lowpass(e, 2200.0 + 7000.0 * exc_bright, 0.707)
    d = max(1, int(round(np.clip(pos, 0.02, 0.98) * Ni)))
    if d < exc_len:
        e[d:] -= e[:-d] * 0.85          # notch the harmonic at the pluck point
    e *= np.exp(-np.arange(exc_len) / (exc_len * 0.55)) * 0.9
    y[pad - exc_len:pad] = e

    damp = math.exp(-6.907755 / max(0.05, t60 * (SR / N)))
    blend = 0.25 + 0.75 * np.clip(bright, 0.0, 1.0)
    i = pad
    while i < total:
        j = min(total, i + Ni)
        L = j - i
        a = y[i - Ni:i - Ni + L]
        b = y[i - Ni - 1:i - Ni - 1 + L]
        c = y[i - Ni - 2:i - Ni - 2 + L]
        r1 = (1 - frac) * a + frac * b
        r0 = (1 - frac) * b + frac * c
        y[i:j] = damp * ((1 - blend) * r1 + blend * 0.5 * (r1 + r0))
        i = j
    y = y[pad:]
    if stretch > 0:  # slight inharmonic stiffness: gentle all-pass smear
        from scipy.signal import lfilter
        ap = 0.55 * stretch
        yb, ya = [ap, 0.0, 1.0], [1.0, 0.0, ap]
        y = lfilter(yb, ya, y)
    ir = body_ir(body, seed=seed + 11)
    y = fftconvolve1(y, ir)[:y.shape[0]]
    y = highpass(y, 70.0 if body in ("bass", "cello") else 120.0, 0.707)
    y = fade(y, 0.001, 0.03) * level
    return calibrate(y, -6.0, 99.5)


def pluck_note(midi: float, dur: float = 2.0, **kw) -> np.ndarray:
    return pluck(m2f(midi), dur, **kw)


def harp_note(midi: float, dur: float = 3.0, seed: int = 1, dyn: float = 1.0) -> np.ndarray:
    return pluck(m2f(midi), dur, t60=max(1.2, dur * 1.4), bright=0.25, pos=0.16,
                 seed=seed, body="harp", level=0.9 * dyn, exc_bright=0.45)


# --------------------------------------------------------------------------
# section player: seats a group of players across the stereo field
# --------------------------------------------------------------------------

SEATS = {
    "violin1": (-0.55, 0.0),
    "violin2": (0.55, 0.0),
    "viola": (-0.28, 0.0),
    "cello": (0.28, 0.0),
    "bass": (0.0, 0.0),
}


def section(notes, dur: float = 2.0, size: int = 8, body: str = "violin",
            spread: float = 0.7, pan_center: float = 0.0, seed: int = 1,
            style: str = "legato", **kw) -> np.ndarray:
    """render a chord (list of midi notes) as a string section -> stereo"""
    notes = list(notes)
    out = np.zeros((sec(dur + 2.0), 2), dtype=np.float64)
    g = rng(seed)
    per_voice = max(1, size // max(1, len(notes)))
    for vi, nt in enumerate(notes):
        p = pan_center + spread * (2.0 * (vi + 0.5) / max(1, len(notes)) - 1.0)
        p = float(np.clip(p, -1, 1))
        sub = np.zeros(sec(dur + 2.0), dtype=np.float64)
        for k in range(per_voice):
            if style == "spiccato":
                y = spiccato(nt, dur, seed=seed + 13 * vi + k, body=body, **kw)
            elif style == "tremolo":
                y = tremolo(nt, dur, seed=seed + 13 * vi + k, body=body, **kw)
            else:
                y = bowed(nt, dur, players=2, seed=seed + 13 * vi + k, body=body, **kw)
            n = min(y.shape[0], sub.shape[0])
            off = int(sec(abs(float(g.uniform(0, 0.018)))))
            sub[off:off + n] += y[:n] * (0.9 + 0.2 * float(g.random()))
        from ..core import mix_at
        mix_at(out, sub, 0.0, gain=1.0 / max(1.0, size ** 0.35), p=p)
    return calibrate(out, -3.0, 99.0)


def pizz_section(notes, dur: float = 1.6, pan_center: float = 0.0,
                 spread: float = 0.6, body: str = "cello", seed: int = 1) -> np.ndarray:
    out = np.zeros((sec(dur + 2.0), 2), dtype=np.float64)
    for i, nt in enumerate(notes):
        p = float(np.clip(pan_center + spread * (2 * (i + 0.5) / len(notes) - 1), -1, 1))
        y = pluck(m2f(nt), dur, t60=dur * 1.2, bright=0.45, pos=0.3,
                  seed=seed + i, body=body, level=0.8)
        from ..core import mix_at
        mix_at(out, y, i * 0.012, gain=1.0, p=p)
    return calibrate(out, -4.0, 99.0)
