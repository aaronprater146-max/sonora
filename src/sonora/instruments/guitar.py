"""
Guitars: steel-string, nylon, electric (with cabinet sim), 12-string,
plus strum / fingerpick / palm-mute performance models.
"""
from __future__ import annotations

import math

import numpy as np

from ..core import (SR, FLOAT, sec, m2f, noise, rng, tanh_sat, lowpass, highpass,
                    bandpass, peak_eq, low_shelf, high_shelf, fftconvolve1, fade,
                    mix_at, pan, osc_saw, filt_env, resample, calibrate)
from .strings import pluck, body_ir


def steel(midi: float, dur: float = 2.0, dyn: float = 1.0, bright: float = 0.45,
          pos: float = 0.22, seed: int = 1, body: str = "guitar", mic: float = 0.5) -> np.ndarray:
    return pluck(m2f(midi), dur, t60=max(0.8, dur * 1.5), bright=bright, pos=pos,
                 seed=seed, body=body, level=0.85 * dyn, exc_bright=0.55 + 0.3 * mic)


def nylon(midi: float, dur: float = 2.0, dyn: float = 1.0, seed: int = 1) -> np.ndarray:
    return pluck(m2f(midi), dur, t60=dur * 1.2, bright=0.75, pos=0.3, seed=seed,
                 body="nylon", level=0.9 * dyn, exc_bright=0.35)


def electric(midi: float, dur: float = 2.0, dyn: float = 1.0, gain: float = 0.55,
             pickup: str = "bridge", cab: bool = True, seed: int = 1) -> np.ndarray:
    """plucked string -> pickup EQ -> preamp drive -> cabinet IR"""
    y = pluck(m2f(midi), dur, t60=dur * 2.0, bright=0.5, pos=0.18, seed=seed,
              body="guitar", level=1.0)
    y = peak_eq(y, 3000.0 if pickup == "bridge" else 1200.0, 4.0, 0.9)
    y = low_shelf(y, 160.0, -3.0)
    drive = 1.0 + 8.0 * max(0.0, gain)
    y = tanh_sat(y * (1.0 + 3.0 * gain), drive)
    if cab:
        ir = cab_ir()
        y = fftconvolve1(y, ir)[:y.shape[0]] * 0.9
    y = highpass(lowpass(y, 7000.0, 0.707), 90.0, 0.707)
    return calibrate(y * 0.55 * dyn, -6.0, 99.5)


def cab_ir(seconds: float = 0.09, seed: int = 5) -> np.ndarray:
    """4x12-ish cabinet: cone breakup + early box reflections"""
    n = sec(seconds)
    t = np.arange(n) / SR
    g = rng(seed)
    ir = np.zeros(n)
    for f, a in [(120, 1.0), (240, .8), (480, .5), (900, .35), (1600, .3), (2600, .2),
                 (3800, .15), (5200, .1)]:
        ir += a * np.exp(-t / (0.012 + 0.02 / (1 + f / 400))) * np.sin(
            2 * math.pi * f * t + g.uniform(0, 6.283))
    nb = sec(0.004)
    ir[:nb] += noise(nb, seed) * 0.5
    ir = highpass(lowpass(ir, 7200.0, 0.707), 70.0, 0.707)
    ir = ir / (np.sqrt(np.sum(ir ** 2)) + 1e-9) * 0.35   # unity-energy cabinet
    return ir.astype(FLOAT)


def strum(notes, dur: float = 2.0, style: str = "strum", up: bool = False,
          spread: float = 0.018, body: str = "steel", dyn: float = 1.0,
          pan_spread: float = 0.35, seed: int = 1, gain: float = 0.0) -> np.ndarray:
    """strum a chord -> stereo. body: steel | nylon | electric"""
    notes = list(notes)
    if up:
        notes = notes[::-1]
    out = np.zeros((sec(dur + 3.0), 2), dtype=np.float64)
    g = rng(seed)
    for i, nt in enumerate(notes):
        off = i * spread * (1.0 + 0.25 * float(g.uniform(-1, 1)))
        vel = dyn * (0.72 + 0.28 * float(g.random()))
        if body == "nylon":
            y = nylon(nt, dur, vel, seed=seed + i)
        elif body == "electric":
            y = electric(nt, dur, vel, gain=gain, seed=seed + i)
        else:
            y = steel(nt, dur, vel, seed=seed + i)
        p = pan_spread * (2 * (i + 0.5) / len(notes) - 1) if len(notes) > 1 else 0.0
        mix_at(out, y, off, gain=0.9, p=float(p))
    return calibrate(out, -3.0, 99.0)


def fingerpick(notes, dur: float = 2.4, seed: int = 1, dyn: float = 1.0) -> np.ndarray:
    return strum(notes, dur, spread=0.075, body="steel", seed=seed, dyn=dyn, pan_spread=0.4)


def power_chords(roots, dur: float = 1.2, gain: float = 0.7, seed: int = 1,
                 bpm: float = 100.0, pattern: str = "x.x.x.x.") -> np.ndarray:
    """palm-muted power chord chugs (root + fifth + octave)"""
    spb = 60.0 / bpm
    out = np.zeros((sec(len(roots) * spb * 4 + 3), 2), dtype=np.float64)
    for bar, root in enumerate(roots):
        for s, ch in enumerate(pattern):
            if ch != "x":
                continue
            t = bar * 4 * spb + s * spb / 2
            for iv in (0, 7, 12):
                y = electric(root + iv, min(0.55, dur), 0.9, gain=gain, seed=seed + s + iv)
                y = lowpass(y, 4200.0, 0.707)
                mix_at(out, y, t, gain=0.5, p=0.0)
    return calibrate(out, -3.0, 99.0)


def arpeggio(notes, dur: float = 0.9, rate: float = 0.14, seed: int = 1,
             body: str = "steel", dyn: float = 1.0) -> np.ndarray:
    out = np.zeros((sec(len(notes) * rate + dur + 3), 2), dtype=np.float64)
    for i, nt in enumerate(notes):
        y = (nylon(nt, dur, dyn, seed=seed + i) if body == "nylon"
             else steel(nt, dur, dyn, seed=seed + i))
        p = 0.35 * (2 * ((i % len(notes)) + 0.5) / len(notes) - 1)
        mix_at(out, y, i * rate, gain=0.85, p=float(p))
    return calibrate(out, -4.0, 99.0)
