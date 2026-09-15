"""
Flutes, reeds and world instruments -- all synthesised.

flutes: sine-ish tone + filtered breath noise + chiff on the attack
plucked world instruments reuse the Karplus-Strong core with their own
body resonance and excitation colour.
"""
from __future__ import annotations

import math

import numpy as np

from ..core import (SR, FLOAT, sec, m2f, noise, rng, osc_sine, osc_saw, osc_square,
                    tanh_sat, lowpass, highpass, bandpass, peak_eq, high_shelf,
                    fftconvolve1, fade, mix_at, pan, additive, adsr, calibrate,
                    filt_env)
from .strings import pluck


def flute(midi: float, dur: float = 1.2, breath: float = 0.45, dyn: float = 1.0,
          vibrato: float = 5.0, vib_depth: float = 8.0, bright: float = 0.5,
          attack: float = 0.05, release: float = 0.2, seed: int = 1) -> np.ndarray:
    n = sec(dur + release + 0.1)
    t = np.arange(n) / SR
    f = m2f(midi)
    g = rng(seed)
    vib_env = np.clip((t - 0.15) / 0.5, 0, 1)
    freq = f * (2.0 ** ((vib_depth * vib_env * np.sin(2 * math.pi * vibrato * t + g.uniform(0, 6.283))) / 1200.0))
    ph = np.cumsum(freq) / SR
    y = np.sin(2 * math.pi * ph) * 0.9
    y += 0.14 * np.sin(4 * math.pi * ph) + (0.05 * bright) * np.sin(6 * math.pi * ph)
    env = (1 - np.exp(-t / max(0.01, attack * 0.5))) * adsr(n, 0.0, 0.0, 1.0, release,
                                                            hold=max(0.0, dur - attack))
    y *= env
    if breath:
        ns = noise(n, seed + 2)
        air = bandpass(ns, f * 2.0, 0.7) * 0.5 + highpass(ns, 3500.0, 0.7) * 0.35
        chiff = np.zeros(n)
        m = min(n, sec(0.05))
        chiff[:m] = noise(m, seed + 3) * np.exp(-np.arange(m) / (0.012 * SR))
        chiff = bandpass(chiff, 2600.0, 1.0)
        y += (air * env * 0.16 + chiff * 0.35) * breath
    y = tanh_sat(y * (0.55 * dyn), 1.2)
    return calibrate(fade(highpass(y, 150.0, 0.707).astype(FLOAT), 0.008, release * 0.7), -8.0, 99.5)


def shakuhachi(midi: float, dur: float = 1.5, dyn: float = 1.0, seed: int = 1, **kw) -> np.ndarray:
    return flute(midi, dur, breath=1.0, bright=0.35, attack=0.09, dyn=dyn, seed=seed, **kw)


def pan_flute(midi: float, dur: float = 0.8, dyn: float = 1.0, seed: int = 1) -> np.ndarray:
    return flute(midi, dur, breath=0.8, bright=0.4, attack=0.03, dyn=dyn, seed=seed)


def whistle(midi: float, dur: float = 0.6, dyn: float = 1.0, seed: int = 1) -> np.ndarray:
    n = sec(dur + 0.3)
    t = np.arange(n) / SR
    f = m2f(midi)
    ph = np.cumsum(f * (1 + 0.02 * np.exp(-t / 0.05) + 0.004 * np.sin(2 * math.pi * 5 * t))) / SR
    y = np.sin(2 * math.pi * ph) + 0.03 * np.sin(4 * math.pi * ph)
    y += noise(n, seed) * 0.012
    env = (1 - np.exp(-t / 0.03)) * adsr(n, 0, 0, 1.0, 0.12, hold=dur)
    return calibrate(fade((y * env * 0.5 * dyn).astype(FLOAT), 0.01, 0.1), -9.0, 99.5)


def kalimba(midi: float, dur: float = 1.6, dyn: float = 1.0, seed: int = 1) -> np.ndarray:
    from .keys import bell
    return bell(midi, dur, kind="kalimba", dyn=dyn * 0.9, seed=seed)


def koto(midi: float, dur: float = 2.0, dyn: float = 1.0, seed: int = 1) -> np.ndarray:
    y = pluck(m2f(midi), dur, t60=dur * 1.3, bright=0.55, pos=0.12, seed=seed,
              body="harp", level=0.9 * dyn, exc_bright=0.7, stretch=0.15)
    return peak_eq(y, 1200.0, 3.0, 1.2).astype(FLOAT)


def sitar(midi: float, dur: float = 2.4, dyn: float = 1.0, seed: int = 1) -> np.ndarray:
    """very bright, long, sympathetic: high stiffness + shimmering partials"""
    y = pluck(m2f(midi), dur, t60=dur * 2.2, bright=0.15, pos=0.1, seed=seed,
              body="guitar", level=0.8 * dyn, exc_bright=0.9, stretch=0.6)
    n = y.shape[0]
    t = np.arange(n) / SR
    f = m2f(midi)
    symp = np.zeros(n)
    for r, a in [(1.0, .18), (1.5, .12), (2.0, .1), (3.0, .07)]:
        symp += a * np.exp(-t / (dur * 0.9)) * np.sin(2 * math.pi * f * r * t)
    return calibrate(y + symp * 0.25, -6.0, 99.5)


def handpan(midi: float, dur: float = 3.0, dyn: float = 1.0, seed: int = 1) -> np.ndarray:
    n = sec(dur + 0.5)
    t = np.arange(n) / SR
    f = m2f(midi)
    y = np.zeros(n)
    g = rng(seed)
    for r, a, tau in [(1.0, 1.0, dur * .5), (2.0, .35, dur * .35), (2.76, .3, dur * .3),
                      (5.4, .12, dur * .18), (8.9, .06, dur * .12)]:
        y += a * np.exp(-t / tau) * np.sin(2 * math.pi * f * r * t + g.uniform(0, 6.283))
    m = min(n, sec(0.01))
    y[:m] += noise(m, seed) * np.exp(-np.arange(m) / (0.003 * SR)) * 0.15
    y *= (1 - np.exp(-t / 0.002))
    return calibrate(fade(tanh_sat(y * 0.3 * dyn, 1.2).astype(FLOAT), 0.001, 0.2), -6.0, 99.5)


def steelpan(midi: float, dur: float = 2.0, dyn: float = 1.0, seed: int = 1) -> np.ndarray:
    from .keys import bell
    return bell(midi, dur, kind="steelpan", dyn=dyn, seed=seed)


def harmonica(midi: float, dur: float = 1.0, dyn: float = 1.0, seed: int = 1) -> np.ndarray:
    """free-reed: strong odd+even harmonics with a buzzy, slightly detuned pair"""
    n = sec(dur + 0.3)
    t = np.arange(n) / SR
    f = m2f(midi)
    y = np.zeros(n)
    for k, a in zip(range(1, 14), [1 / (k ** 0.85) for k in range(1, 14)]):
        if f * k > SR * 0.45:
            break
        y += a * np.sin(2 * math.pi * f * k * t) + 0.7 * a * np.sin(
            2 * math.pi * f * k * 1.004 * t + 0.4)
    env = (1 - np.exp(-t / 0.02)) * adsr(n, 0, 0.02, 1.0, 0.12, hold=dur)
    y = y * env * 0.11 * dyn
    y += noise(n, seed) * 0.02 * env
    y = tanh_sat(y, 1.6)
    return calibrate(fade(bandpass(y, 1400.0, 0.5).astype(FLOAT), 0.01, 0.1), -7.0, 99.5)


def didgeridoo(midi: float = 33.0, dur: float = 2.0, dyn: float = 1.0, seed: int = 1) -> np.ndarray:
    n = sec(dur + 0.4)
    t = np.arange(n) / SR
    f = m2f(midi)
    am = 0.5 + 0.5 * np.sin(2 * math.pi * 4.2 * t)
    y = osc_saw(f * (1 + 0.01 * np.sin(2 * math.pi * 0.7 * t)), n)
    y = y * (0.6 + 0.4 * am)
    y = filt_env(y, 700 + 900 * am, 1.4, "lp")
    y += bandpass(noise(n, seed), 900.0, 0.8) * 0.1
    env = adsr(n, 0.06, 0.1, 1.0, 0.2, hold=dur)
    return calibrate(fade((y * env * 0.5 * dyn).astype(FLOAT), 0.02, 0.2), -5.0, 99.5)


def accordion(midi: float, dur: float = 1.5, dyn: float = 1.0, seed: int = 1) -> np.ndarray:
    n = sec(dur + 0.35)
    t = np.arange(n) / SR
    f = m2f(midi)
    y = np.zeros(n)
    for det in (-6, 6):
        for k in range(1, 12):
            ff = f * k * (2 ** (det / 1200.0))
            if ff > SR * 0.45:
                break
            y += (1 / k ** 1.1) * np.sin(2 * math.pi * ff * t + k * 0.3)
    env = (1 - np.exp(-t / 0.03)) * adsr(n, 0, 0.05, 1.0, 0.18, hold=dur)
    y = y * env * 0.07 * dyn
    return calibrate(fade(bandpass(lowpass(y, 6000.0, 0.707), 700.0, 0.4).astype(FLOAT), 0.01, 0.15), -7.0, 99.5)
