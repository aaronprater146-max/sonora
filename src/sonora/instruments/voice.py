"""
Word-less voice: a formant (source-filter) singer.

Real lyric synthesis needs a big model; this gives you the next best thing
that runs anywhere -- a glottal source with vibrato and breath, pushed
through 5 parallel formant resonators tuned to a vowel.  Stack a dozen of
them with independent vibrato and you get a convincing choir pad.
"""
from __future__ import annotations

import math

import numpy as np

from ..core import (SR, FLOAT, sec, m2f, noise, rng, osc_saw, tanh_sat, lowpass,
                    highpass, bandpass, peak_eq, low_shelf, high_shelf, fade,
                    mix_at, pan, adsr, calibrate)

# vowel formants (F1..F5, amplitude, bandwidth) for a soprano/alto speaker
VOWELS = {
    "ah": [(800, 1.00, 90), (1150, 0.55, 110), (2900, 0.30, 170), (3900, 0.12, 250), (5200, 0.06, 300)],
    "oo": [(325, 1.00, 70), (700, 0.30, 90), (2530, 0.10, 160), (3500, 0.05, 250), (4600, 0.02, 300)],
    "ee": [(350, 1.00, 60), (2000, 0.50, 110), (2800, 0.35, 170), (3800, 0.15, 250), (5000, 0.06, 300)],
    "oh": [(500, 1.00, 80), (900, 0.45, 110), (2600, 0.16, 170), (3600, 0.08, 250), (4800, 0.03, 300)],
    "eh": [(550, 1.00, 70), (1750, 0.45, 120), (2600, 0.30, 170), (3700, 0.12, 250), (5000, 0.05, 300)],
    "mm": [(280, 1.00, 60), (1100, 0.18, 120), (2400, 0.06, 200), (3300, 0.03, 300)],
}


def _formant_bank(x, formants, sr=SR):
    out = np.zeros_like(x, dtype=np.float64)
    for f, a, bw in formants:
        w0 = 2 * math.pi * f / sr
        alpha = math.sin(w0) * math.sinh(math.log(2) / 2 * bw * w0 / math.sin(w0)) if 0 < w0 < math.pi else 0.5
        b = [alpha, 0, -alpha]
        a_ = [1 + alpha, -2 * math.cos(w0), 1 - alpha]
        from scipy.signal import lfilter
        out += a * lfilter([c / a_[0] for c in b], [c / a_[0] for c in a_], x)
    return out


def vowel(midi: float, dur: float = 1.5, vowel_: str = "ah", dyn: float = 1.0,
          vibrato: float = 5.2, vib_depth: float = 14.0, vib_delay: float = 0.18,
          breath: float = 0.35, gender: float = 1.0, rasp: float = 0.15,
          attack: float = 0.06, release: float = 0.3, seed: int = 1,
          bright: float = 0.55, formant_shift: float = 1.0) -> np.ndarray:
    """one sung note. gender 1 = soprano/alto, 0.72 = tenor, 0.6 = bass"""
    n = sec(dur + release + 0.1)
    t = np.arange(n) / SR
    f = m2f(midi)
    g = rng(seed)
    vib_env = np.clip((t - vib_delay) / max(0.05, vib_delay * 2.5), 0, 1)
    cents = vib_depth * vib_env * (np.sin(2 * math.pi * vibrato * t + g.uniform(0, 6.283)) +
                                   0.25 * np.sin(2 * math.pi * vibrato * 1.7 * t))
    freq = f * (2.0 ** (cents / 1200.0))
    ph = np.cumsum(freq) / SR
    # glottal source: pulse train with a soft rolloff + a little jitter
    src = np.zeros(n)
    roll = 1.35 - 0.5 * bright
    for k in range(1, 26):
        if f * k > SR * 0.45:
            break
        src += (1.0 / k ** roll) * np.sin(2 * math.pi * k * ph + g.uniform(0, 6.283) * (k > 1))
    src = src / 2.6
    if rasp:
        jit = 1.0 + rasp * 0.02 * np.sin(2 * math.pi * 47 * t)
        src *= jit
    # breath
    if breath:
        ns = noise(n, seed + 4)
        ns = bandpass(ns, 2400.0, 0.6) * 0.5 + highpass(ns, 5000.0, 0.7) * 0.25
        src = src * (1 - breath * 0.25) + ns * breath * 0.22
    env = (1 - np.exp(-t / max(0.01, attack * 0.4))) * adsr(n, 0.0, 0.0, 1.0, release,
                                                            hold=max(0.0, dur - attack))
    src *= env
    # formants
    fm = [(f * formant_shift * gender, a, bw * (0.9 + 0.2 * gender))
          for f, a, bw in VOWELS.get(vowel_, VOWELS["ah"])]
    y = _formant_bank(src, fm)
    y = high_shelf(low_shelf(y, 250.0, 2.0), 3200.0, 3.0 * bright)
    y = highpass(y, 110.0, 0.707)
    y = tanh_sat(y * (0.5 * dyn), 1.25)
    return calibrate(fade(y.astype(FLOAT), 0.01, release * 0.8), -7.0, 99.5)


def choir(notes, dur: float = 2.0, vowel_: str = "ah", size: int = 10,
          spread: float = 0.8, pan_center: float = 0.0, seed: int = 1,
          gender: float = 1.0, breath: float = 0.4, dyn: float = 1.0) -> np.ndarray:
    """stacked singers -> stereo choir"""
    notes = list(notes)
    out = np.zeros((sec(dur + 2.5), 2), dtype=np.float64)
    g = rng(seed)
    per = max(1, size // max(1, len(notes)))
    for vi, nt in enumerate(notes):
        base_p = pan_center + spread * (2 * (vi + 0.5) / max(1, len(notes)) - 1)
        for k in range(per):
            p = float(np.clip(base_p + g.uniform(-0.12, 0.12), -1, 1))
            y = vowel(nt, dur, vowel_, dyn=dyn * g.uniform(0.75, 1.0),
                      vibrato=g.uniform(4.4, 6.0), vib_depth=g.uniform(9, 18),
                      vib_delay=g.uniform(0.12, 0.4), breath=breath,
                      gender=gender * g.uniform(0.94, 1.06), seed=seed + 17 * vi + k,
                      formant_shift=g.uniform(0.96, 1.04))
            mix_at(out, y, abs(float(g.uniform(0, 0.03))), gain=1.0 / (size ** 0.42), p=p)
    return calibrate(out, -4.0, 99.0)


def oohs(notes, dur=2.0, **kw):
    return choir(notes, dur, vowel_="oo", breath=0.5, **kw)


def aahs(notes, dur=2.0, **kw):
    return choir(notes, dur, vowel_="ah", **kw)


def vocal_chops(notes, dur: float = 0.35, seed: int = 1, vowel_: str = "ah", **kw) -> np.ndarray:
    """short staccato 'vocal' stabs -- the modern pop/edm ear-candy layer"""
    out = np.zeros((sec(len(notes) * dur + 3), 2), dtype=np.float64)
    for i, nt in enumerate(notes):
        y = vowel(nt, dur * 0.9, vowel_, attack=0.012, release=dur * 0.5,
                  vib_depth=6, breath=0.3, seed=seed + i, **kw)
        mix_at(out, y, i * dur, gain=0.8, p=float(np.sin(i * 2.1) * 0.4))
    return calibrate(out, -5.0, 99.0)
