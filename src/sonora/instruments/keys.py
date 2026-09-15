"""
Keyboard instruments: piano / felt piano, Rhodes, Wurlitzer, organ, clavinet,
bells & glockenspiel.  All modal or FM synthesis -- nothing sampled.
"""
from __future__ import annotations

import math

import numpy as np

from ..core import (SR, FLOAT, sec, m2f, noise, rng, db2amp, osc_sine, osc_saw,
                    tanh_sat, filt_env, lowpass, highpass, bandpass, peak_eq,
                    low_shelf, high_shelf, fftconvolve1, fade, chorus, pan,
                    calibrate)
from .strings import body_ir, pluck

# --------------------------------------------------------------------------
# piano (modal, with inharmonicity + double decay + hammer + pedal noise)
# --------------------------------------------------------------------------

def piano(midi: float, dur: float = 2.0, dyn: float = 1.0, bright: float = 0.72,
          pedal: float = 0.6, felt: float = 0.0, seed: int = 1,
          release: float = 0.35, detune: float = 0.35) -> np.ndarray:
    n = sec(dur + release + 0.4)
    t = np.arange(n) / SR
    f = m2f(midi)
    # inharmonicity coefficient: large for short/thick strings
    B = 0.0008 * (2.0 ** ((60.0 - midi) / 24.0))
    n_part = 20
    out = np.zeros(n, dtype=np.float64)
    g = rng(seed)
    # hammer strike noise (felt = softer, darker)
    hn = noise(sec(0.05), seed + 1)
    hn = lowpass(hn, 3200.0 if felt else 6500.0, 0.8)
    hn = hn * np.exp(-np.arange(hn.shape[0]) / (0.004 * SR))
    strike = np.zeros(n)
    strike[:hn.shape[0]] = hn
    # double decay: fast at first, slower tail
    taus = []
    for k in range(1, n_part + 1):
        fk = f * k * math.sqrt(1.0 + B * k * k)
        if fk > SR * 0.45:
            break
        a = (1.0 / k ** (0.92 + 0.55 * (1 - bright))) * math.exp(-(k - 1) / (6.0 + 34 * bright))
        tau = (7.5 - 4.5 * bright) / (1.0 + (k - 1) * 0.42) * (0.35 + 0.65 * pedal)
        tau *= 2.0 ** ((60.0 - midi) / 36.0)      # bass rings much longer
        taus.append(tau)
        # detuned unison strings
        for dd in (-detune, detune):
            out += a * 0.5 * np.exp(-t / tau) * np.sin(
                2 * math.pi * fk * (2 ** (dd / 1200.0)) * t + g.uniform(0, 6.283))
    out *= (1.0 - np.exp(-t / 0.002))
    out += strike * (0.10 + 0.25 * dyn) * (0.4 if felt else 1.0)
    # soundboard + pedal/string resonance
    ir = body_ir("piano", seed=seed + 3)
    out = fftconvolve1(out, ir)[:n]
    # damper release
    if release > 0.01:
        rel = np.zeros(n)
        i0 = sec(max(0.0, dur))
        dn = noise(min(n - i0, sec(0.09)), seed + 5)
        dn = bandpass(dn, 2600.0, 0.9) * np.exp(-np.arange(dn.shape[0]) / (0.012 * SR))
        rel[i0:i0 + dn.shape[0]] = dn
        out += rel * 0.16 * (1.0 - felt)
    out = tanh_sat(out * (1.1 * dyn), 1.15)
    out = highpass(out, 40.0, 0.707)
    if felt:
        out = lowpass(out, 3600.0, 0.707)
    return calibrate(fade(out.astype(FLOAT), 0.002, 0.25), -5.0, 99.5)


def felt_piano(midi, dur=2.0, **kw):
    return piano(midi, dur, felt=0.8, bright=0.28, **kw)


# --------------------------------------------------------------------------
# electric pianos (FM)
# --------------------------------------------------------------------------

def rhodes(midi: float, dur: float = 2.0, dyn: float = 1.0, bite: float = 0.5,
           seed: int = 1, tremolo_rate: float = 0.0, release: float = 0.4) -> np.ndarray:
    n = sec(dur + release + 0.2)
    t = np.arange(n) / SR
    f = m2f(midi)
    env = np.exp(-t / (1.6 + 2.5 * (1 - dyn))) * (1 - np.exp(-t / 0.002))
    index = (2.6 + 5.0 * bite * dyn) * np.exp(-t / (0.35 + 0.5 * (1 - bite)))
    mod = np.sin(2 * math.pi * f * 1.0 * t)          # 1:1 tine ratio
    car = np.sin(2 * math.pi * f * t + index * mod)
    bark = np.sin(2 * math.pi * f * t + (index * 0.22) * np.sin(2 * math.pi * f * 14.0 * t)) * 0.25
    hammer = np.zeros(n)
    m = min(n, sec(0.02))
    hammer[:m] = bandpass(noise(m, seed), 4200.0, 1.2) * np.exp(-np.arange(m) / (0.003 * SR))
    y = (car * 0.8 + bark) * env + hammer * 0.12 * dyn
    y = tanh_sat(y * (0.9 * dyn + 0.4), 1.3)
    y = highpass(y, 90.0, 0.707)
    if tremolo_rate:
        y = y * (0.8 + 0.2 * np.sin(2 * math.pi * tremolo_rate * t))
    return calibrate(fade(y.astype(FLOAT), 0.002, 0.2), -6.0, 99.5)


def wurlitzer(midi: float, dur: float = 2.0, dyn: float = 1.0, seed: int = 1,
              tremolo_rate: float = 5.0, release: float = 0.3) -> np.ndarray:
    n = sec(dur + release + 0.2)
    t = np.arange(n) / SR
    f = m2f(midi)
    env = np.exp(-t / 1.9) * (1 - np.exp(-t / 0.0015))
    idx = 1.4 * dyn * np.exp(-t / 0.5)
    y = np.sin(2 * math.pi * f * t + idx * np.sin(2 * math.pi * f * 2.0 * t))
    y += 0.18 * np.sin(2 * math.pi * f * 3.0 * t) * np.exp(-t / 0.25)
    y = (y * env) * (0.85 + 0.15 * np.sin(2 * math.pi * tremolo_rate * t))
    y = tanh_sat(y * dyn * 1.2, 1.5)
    return calibrate(fade(highpass(y, 120.0, 0.707).astype(FLOAT), 0.002, 0.2), -6.0, 99.5)


# --------------------------------------------------------------------------
# organ
# --------------------------------------------------------------------------

DRAWBARS = [0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0]


def organ(midi: float, dur: float = 2.0, drawbars: tuple = (8, 4, 2, 1),
          level: float = 1.0, click: float = 0.5, leslie: float = 0.0,
          drive: float = 1.6, release: float = 0.18, seed: int = 1) -> np.ndarray:
    n = sec(dur + release + 0.1)
    t = np.arange(n) / SR
    f = m2f(midi)
    g = rng(seed)
    y = np.zeros(n)
    gains = {0.5: 0.55, 1.0: 0.45, 1.5: 0.5, 2.0: 0.85, 3.0: 0.5, 4.0: 0.7, 6.0: 0.4, 8.0: 1.0}
    for d in drawbars:
        r = d / 8.0
        y += gains.get(d, 0.5) * np.sin(2 * math.pi * f * r * t + g.uniform(0, 6.283))
        y += 0.06 * np.sin(2 * math.pi * f * r * 2.001 * t)
    if click:
        m = min(n, sec(0.012))
        ck = noise(m, seed + 2) * np.exp(-np.arange(m) / (0.0025 * SR))
        ck = bandpass(ck, 2400.0, 1.1)
        y[:m] += ck * 0.35 * click
    env = np.ones(n)
    i0 = sec(dur)
    env[i0:] *= np.exp(-(t[i0:] - dur) / (release * 0.35))
    y = y * env * (0.16 * level)
    y = tanh_sat(y, drive)
    y = lowpass(highpass(y, 34.0, 0.707), 9000.0, 0.707).astype(FLOAT)
    if leslie > 0:
        y = chorus(y, rate=0.75 * leslie, depth=0.0045, voices=3, mix=0.55 * leslie, seed=seed)
    return calibrate(fade(y, 0.003, 0.06), -6.0, 99.5)


# --------------------------------------------------------------------------
# clavinet
# --------------------------------------------------------------------------

def clav(midi: float, dur: float = 0.6, dyn: float = 1.0, bright: float = 0.7,
         seed: int = 1) -> np.ndarray:
    n = sec(dur + 0.4)
    t = np.arange(n) / SR
    f = m2f(midi)
    y = np.zeros(n)
    for k, a in zip(range(1, 16), [1 / k ** 1.1 for k in range(1, 16)]):
        if f * k > SR * 0.45:
            break
        y += a * np.sin(2 * math.pi * f * k * t + k * 0.9)
    y *= np.exp(-t / (0.28 + 0.5 * (1 - bright))) * (1 - np.exp(-t / 0.001))
    m = min(n, sec(0.03))
    y[:m] += noise(m, seed) * np.exp(-np.arange(m) / (0.004 * SR)) * 0.25
    y = tanh_sat(y * 0.28 * dyn, 2.2 if bright > 0.5 else 1.4)
    y = highpass(y, 140.0, 0.707)
    return calibrate(fade(y.astype(FLOAT), 0.001, 0.08), -6.0, 99.5)


# --------------------------------------------------------------------------
# bells / mallets (inharmonic additive)
# --------------------------------------------------------------------------

BELL_PARTIALS = {
    "glock": [(1.0, 1.0, 2.6), (2.76, .6, 1.9), (5.4, .35, 1.3), (8.9, .2, .9), (13.3, .1, .6)],
    "bell": [(0.5, .8, 6.0), (1.0, 1.0, 5.0), (1.19, .5, 3.6), (1.56, .45, 2.8),
             (2.0, .6, 2.2), (2.66, .35, 1.6), (3.01, .25, 1.2), (4.1, .18, .9)],
    "kalimba": [(1.0, 1.0, 1.4), (3.0, .35, .8), (5.2, .2, .5), (8.1, .12, .35)],
    "steelpan": [(1.0, 1.0, 2.0), (2.0, .55, 1.4), (3.01, .5, 1.1), (4.2, .3, .8)],
    "marimba": [(1.0, 1.0, .55), (4.0, .28, .3), (10.0, .1, .18)],
    "vibes": [(1.0, 1.0, 4.0), (4.0, .35, 2.0), (10.0, .12, 1.0)],
}


def bell(midi: float, dur: float = 2.0, kind: str = "glock", dyn: float = 1.0,
         seed: int = 1) -> np.ndarray:
    n = sec(dur + 0.6)
    t = np.arange(n) / SR
    f = m2f(midi)
    y = np.zeros(n)
    g = rng(seed)
    for r, a, tau in BELL_PARTIALS.get(kind, BELL_PARTIALS["glock"]):
        fr = f * r
        if fr > SR * 0.46:
            continue
        y += a * np.exp(-t / tau) * np.sin(2 * math.pi * fr * t + g.uniform(0, 6.283))
    if kind in ("steelpan", "vibes"):
        y *= (1 + 0.12 * np.sin(2 * math.pi * 5.2 * t))
    m = min(n, sec(0.02))
    if kind in ("glock", "marimba", "vibes", "kalimba"):
        y[:m] += noise(m, seed + 1) * np.exp(-np.arange(m) / (0.002 * SR)) * 0.2
    y *= (1 - np.exp(-t / 0.0015))
    return calibrate(fade(tanh_sat(y * 0.32 * dyn, 1.2).astype(FLOAT), 0.001, 0.15), -6.0, 99.5)


def music_box(midi, dur=2.0, **kw):
    return bell(midi, dur, kind="glock", **kw)
