"""
sonora.core -- DSP core for the Sonora music factory.

Design rules that keep output "modern AAA" instead of "8-bit":
  * rendered at 48 kHz / float32 with band-limited (PolyBLEP) oscillators,
    so bright saws do not alias
  * time-varying filters are chunked RBJ biquads run through scipy's C
    lfilter (continuous state) -> analogue-sounding sweeps at ~50x the
    speed of a python sample loop
  * space = real convolution against generated impulse responses
  * the bus is a proper chain: saturate -> EQ -> glue comp -> multiband ->
    stereo widener -> lookahead limiter -> LUFS normalise
"""
from __future__ import annotations

import math
import os
from typing import Sequence

import numpy as np

SR = int(os.environ.get("SONORA_SR", 48000))
FLOAT = np.float32


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------

def db2amp(db):
    return 10.0 ** (float(db) / 20.0)


def amp2db(a):
    return 20.0 * math.log10(max(1e-12, float(a)))


def m2f(m):
    return 440.0 * (2.0 ** ((float(m) - 69.0) / 12.0))


def f2m(f):
    return 69.0 + 12.0 * math.log2(max(1e-9, float(f)) / 440.0)


def sec(n):
    return int(round(float(n) * SR))


def buf(seconds, channels=1):
    n = sec(seconds)
    return np.zeros(n, dtype=FLOAT) if channels == 1 else np.zeros((n, channels), dtype=FLOAT)


def ensure2(x):
    x = np.asarray(x, dtype=FLOAT)
    return np.repeat(x[:, None], 2, axis=1) if x.ndim == 1 else x


def mono(x):
    x = np.asarray(x, dtype=FLOAT)
    return x if x.ndim == 1 else x.mean(axis=1).astype(FLOAT)


def fit(x, n):
    x = np.asarray(x, dtype=FLOAT)
    if x.shape[0] == n:
        return x
    return x[:n] if x.shape[0] > n else np.pad(x, (0, n - x.shape[0]))


def norm(x, peak_db=-1.0):
    x = np.asarray(x, dtype=FLOAT)
    p = float(np.max(np.abs(x))) if x.size else 0.0
    return x if p < 1e-9 else (x * (db2amp(peak_db) / p)).astype(FLOAT)


def rms(x):
    x = np.asarray(x, dtype=FLOAT)
    return float(np.sqrt(np.mean(x * x)) + 1e-12) if x.size else 1e-12


def tanh_sat(x, drive=1.0, ceiling=0.85):
    """soft saturation with an automatic pre-gain.

    `ceiling` normalises the input to a predictable level *before* shaping,
    so `drive` means the same thing no matter how hot the upstream synth
    happens to be.  Without this every instrument would saturate into a
    square wave (which is exactly what "8-bit" sounds like).
    """
    x = np.asarray(x)
    if drive <= 1.0001 and not ceiling:
        return x.astype(FLOAT)
    if ceiling:
        # subsample for the level estimate: a percentile over every sample of
        # a three minute mix is a 136 MB temporary for no audible benefit
        step = max(1, x.shape[0] // 200000)
        pk = float(np.percentile(np.abs(x[::step]).astype(np.float64), 99.0)) if x.size else 0.0
        g = ceiling / pk if pk > 1e-9 else 1.0
    else:
        g = 1.0
    inv = 1.0 / math.tanh(drive)
    out = np.empty(x.shape, dtype=FLOAT)
    step = 1 << 18
    for i in range(0, x.shape[0], step):
        j = min(x.shape[0], i + step)
        out[i:j] = np.tanh(np.asarray(x[i:j], dtype=np.float64) * g * drive) * inv
    return out


def soft_clip(x, ceil=0.9):
    x = np.asarray(x, dtype=np.float64)
    return (np.clip(np.tanh(x / ceil) * ceil, -1.0, 1.0)).astype(FLOAT)


# --------------------------------------------------------------------------
# noise (deterministic, fast)
# --------------------------------------------------------------------------

def rng(seed=1):
    return np.random.default_rng(seed)


def noise(n, seed=1, pink=False):
    g = rng(seed)
    w = g.uniform(-1.0, 1.0, int(n)).astype(np.float64)
    if not pink:
        return w.astype(FLOAT)
    # cheap Paul Kellet pink filter
    b = np.zeros(7)
    out = np.empty(int(n))
    for i in range(int(n)):
        b[0] = 0.99886 * b[0] + w[i] * 0.0555179
        b[1] = 0.99332 * b[1] + w[i] * 0.0750759
        b[2] = 0.96900 * b[2] + w[i] * 0.1538520
        b[3] = 0.86650 * b[3] + w[i] * 0.3104856
        b[4] = 0.55000 * b[4] + w[i] * 0.5329522
        b[5] = -0.7616 * b[5] - w[i] * 0.0168980
        out[i] = (b[0] + b[1] + b[2] + b[3] + b[4] + b[5] + b[6] + w[i] * 0.5362) * 0.11
        b[6] = w[i] * 0.115926
    return out.astype(FLOAT)


# --------------------------------------------------------------------------
# envelopes
# --------------------------------------------------------------------------

def adsr(n, a=0.005, d=0.08, s=0.7, r=0.15, hold=0.0, curve=2.0):
    n = int(n)
    if n <= 0:
        return np.zeros(0, dtype=FLOAT)
    ai, di, hi, ri = sec(a), sec(d), sec(hold), sec(r)
    sustain_n = max(0, n - ai - di - hi - ri)
    parts = []
    if ai:
        parts.append(np.linspace(0.0, 1.0, ai) ** (1.0 / curve))
    if di:
        parts.append(1.0 + (max(1e-6, s) - 1.0) * np.linspace(0.0, 1.0, di) ** curve)
    if hi:
        parts.append(np.full(hi, s))
    if sustain_n:
        parts.append(np.full(sustain_n, s))
    if ri:
        start = s if (sustain_n or hi) else 1.0
        parts.append(start * (1.0 - np.linspace(0.0, 1.0, ri) ** curve))
    return fit(np.concatenate(parts).astype(FLOAT), n)


def exp_decay(n, tau, expo=1.0):
    t = np.arange(int(n), dtype=np.float64) / SR
    return np.exp(-(t / max(1e-6, tau)) ** expo).astype(FLOAT)


# --------------------------------------------------------------------------
# oscillators
# --------------------------------------------------------------------------

def _poly_blep(t, dt):
    out = np.zeros_like(t)
    m = t < dt
    x = t[m] / dt[m]
    out[m] = x + x - x * x - 1.0
    m = t > (1.0 - dt)
    x = (t[m] - 1.0) / dt[m]
    out[m] = x * x + x + x + 1.0
    return out


def _phase(freq, n, phase=0.0):
    f = np.broadcast_to(np.asarray(freq, dtype=np.float64), (n,))
    return (phase + np.cumsum(f) / SR) % 1.0


def osc_saw(freq, n, phase=0.0, detune_cents=0.0):
    f = np.asarray(freq, dtype=np.float64) * (2.0 ** (detune_cents / 1200.0))
    f = np.broadcast_to(f, (int(n),))
    dt = np.broadcast_to(f / SR, (int(n),))
    t = _phase(f, n, phase)
    return (2.0 * t - 1.0 - _poly_blep(t, dt)).astype(FLOAT)


def osc_square(freq, n, phase=0.0, duty=0.5):
    return (osc_saw(freq, n, phase) - osc_saw(freq, n, (phase + duty) % 1.0)).astype(FLOAT)


def osc_pulse(freq, n, phase=0.0, pw=0.5):
    return osc_square(freq, n, phase, min(0.98, max(0.02, pw)))


def osc_tri(freq, n, phase=0.0):
    return (4.0 * np.abs(_phase(freq, n, phase) - 0.5) - 1.0).astype(FLOAT)


def osc_sine(freq, n, phase=0.0, fm=None, fm_index=0.0):
    f = np.broadcast_to(np.asarray(freq, dtype=np.float64), (int(n),)).copy()
    if fm is not None:
        f = f * (1.0 + fm_index * np.asarray(fm, dtype=np.float64))
    return np.sin(2.0 * math.pi * _phase(f, n, phase)).astype(FLOAT)


def supersaw(freq, n, voices=7, spread=0.55, seed=3, curve=1.0):
    """detuned saw stack; centre + alternating +/- detune pairs"""
    out = np.zeros(int(n), dtype=np.float64)
    ph = rng(seed).uniform(0, 1, voices)
    for i in range(voices):
        if i == 0:
            cents = 0.0
        else:
            s = 1.0 if i % 2 else -1.0
            cents = s * spread * (i / max(1, voices - 1)) ** curve * 100.0
        out += osc_saw(freq, n, float(ph[i]), cents)
    return (out * (1.55 / max(1, voices))).astype(FLOAT)


def additive(freq, n, partials, amps, phases=None, decay=None):
    n = int(n)
    t = np.arange(n, dtype=np.float64) / SR
    out = np.zeros(n, dtype=np.float64)
    ph = np.zeros(len(partials)) if phases is None else np.asarray(phases, dtype=np.float64)
    for i, (p, a) in enumerate(zip(partials, amps)):
        env = np.ones(n)
        if decay is not None and decay[i] and decay[i] > 0:
            env = np.exp(-t / decay[i])
        out += a * env * np.sin(2.0 * math.pi * float(freq) * p * t + ph[i])
    return out.astype(FLOAT)


# --------------------------------------------------------------------------
# filters: RBJ biquads through scipy lfilter (fast, C loop)
# --------------------------------------------------------------------------

from scipy.signal import lfilter, butter  # noqa: E402


def _bq(mode, f0, q, gain_db=0.0, sr=SR):
    w0 = 2.0 * math.pi * float(np.clip(f0, 8.0, sr * 0.47)) / sr
    cw, sw = math.cos(w0), math.sin(w0)
    alpha = sw / (2.0 * max(0.05, q))
    A = 10.0 ** (gain_db / 40.0)
    if mode == "lp":
        b = [(1 - cw) / 2, 1 - cw, (1 - cw) / 2]
        a = [1 + alpha, -2 * cw, 1 - alpha]
    elif mode == "hp":
        b = [(1 + cw) / 2, -(1 + cw), (1 + cw) / 2]
        a = [1 + alpha, -2 * cw, 1 - alpha]
    elif mode == "bp":
        b = [alpha, 0.0, -alpha]
        a = [1 + alpha, -2 * cw, 1 - alpha]
    elif mode == "notch":
        b = [1, -2 * cw, 1]
        a = [1 + alpha, -2 * cw, 1 - alpha]
    elif mode == "peak":
        b = [1 + alpha * A, -2 * cw, 1 - alpha * A]
        a = [1 + alpha / A, -2 * cw, 1 - alpha / A]
    elif mode == "ls":
        b = [A * ((A + 1) - (A - 1) * cw + 2 * math.sqrt(A) * alpha),
             2 * A * ((A - 1) - (A + 1) * cw),
             A * ((A + 1) - (A - 1) * cw - 2 * math.sqrt(A) * alpha)]
        a = [(A + 1) + (A - 1) * cw + 2 * math.sqrt(A) * alpha,
             -2 * ((A - 1) + (A + 1) * cw),
             (A + 1) + (A - 1) * cw - 2 * math.sqrt(A) * alpha]
    elif mode == "hs":
        b = [A * ((A + 1) + (A - 1) * cw + 2 * math.sqrt(A) * alpha),
             -2 * A * ((A - 1) + (A + 1) * cw),
             A * ((A + 1) + (A - 1) * cw - 2 * math.sqrt(A) * alpha)]
        a = [(A + 1) - (A - 1) * cw + 2 * math.sqrt(A) * alpha,
             2 * ((A - 1) - (A + 1) * cw),
             (A + 1) - (A - 1) * cw - 2 * math.sqrt(A) * alpha]
    else:
        raise ValueError(mode)
    a0 = a[0]
    return [c / a0 for c in b], [c / a0 for c in a]


def _apply(x, b, a, chunk: int = 1 << 16):
    """run a biquad over a long signal in blocks, carrying filter state.

    Keeps float64 coefficients (low frequency EQ needs the precision) but
    never materialises a float64 copy of the whole song -- on a three minute
    mix that was 136 MB per filter stage, and there are a dozen of them.
    """
    x = np.asarray(x)
    n = x.shape[0]
    if n == 0:
        return x.astype(FLOAT)
    nzi = max(len(a), len(b)) - 1
    zi = np.zeros((x.shape[1], nzi)) if x.ndim == 2 else np.zeros(nzi)
    out = np.empty(x.shape, dtype=FLOAT)
    for i in range(0, n, chunk):
        j = min(n, i + chunk)
        y, zi = lfilter(b, a, np.asarray(x[i:j], dtype=np.float64), axis=0, zi=zi)
        out[i:j] = y.astype(FLOAT)
    return out


def lowpass(x, f0, q=0.707):
    b, a = _bq("lp", f0, q)
    return _apply(x, b, a)


def highpass(x, f0, q=0.707):
    b, a = _bq("hp", f0, q)
    return _apply(x, b, a)


def bandpass(x, f0, q=1.0):
    b, a = _bq("bp", f0, q)
    return _apply(x, b, a)


def notch(x, f0, q=4.0):
    b, a = _bq("notch", f0, q)
    return _apply(x, b, a)


def peak_eq(x, f0, gain_db, q=1.0):
    b, a = _bq("peak", f0, q, gain_db)
    return _apply(x, b, a)


def low_shelf(x, f0, gain_db, q=0.707):
    b, a = _bq("ls", f0, q, gain_db)
    return _apply(x, b, a)


def high_shelf(x, f0, gain_db, q=0.707):
    b, a = _bq("hs", f0, q, gain_db)
    return _apply(x, b, a)


def filt_env(x, cutoff, q=0.707, mode="lp", chunk=64):
    """time-varying biquad: cutoff is scalar or per-sample array.

    Coefficients are recalculated every `chunk` samples but the filter
    state is carried, so sweeps are smooth and it runs at C speed.
    """
    x = np.asarray(x, dtype=np.float64)
    n = x.shape[0]
    if n == 0:
        return x.astype(FLOAT)
    if np.isscalar(cutoff) or (np.ndim(cutoff) == 0):
        b, a = _bq(mode, float(cutoff), q)
        return lfilter(b, a, x).astype(FLOAT)
    fc = np.broadcast_to(np.asarray(cutoff, dtype=np.float64), (n,))
    qa = np.broadcast_to(np.asarray(q, dtype=np.float64), (n,))
    out = np.empty(n, dtype=np.float64)
    zi = np.zeros(2)
    for i in range(0, n, chunk):
        j = min(n, i + chunk)
        b, a = _bq(mode, fc[i], qa[i])
        y, zi = lfilter(b, a, x[i:j], zi=zi)
        out[i:j] = y
    return out.astype(FLOAT)


def ladder(x, cutoff, res=0.25, drive=1.0, chunk=32):
    """4-pole Moog-style ladder, topology preserving transform.

    Uses G = g/(1+g) one-pole sections, which are unconditionally stable,
    plus tanh in the feedback path, so high resonance screams instead of
    exploding the way a naive f = 2*tanh(...) cascade does.
    """
    x = np.asarray(x, dtype=np.float64)
    n = x.shape[0]
    if n == 0:
        return x.astype(FLOAT)
    fc = np.clip(np.broadcast_to(np.asarray(cutoff, dtype=np.float64), (n,)),
                 20.0, SR * 0.34)
    out = np.empty(n, dtype=np.float64)
    s = [0.0, 0.0, 0.0, 0.0]
    k = 4.0 * float(np.clip(res, 0.0, 0.98))
    for i in range(0, n, chunk):
        j = min(n, i + chunk)
        fi = float(fc[i])
        g = math.tan(math.pi * fi / SR)
        G = g / (1.0 + g)
        for t in range(i, j):
            xi = x[t]
            u = math.tanh((xi - k * (s[3] - 0.5 * xi)) * drive)
            v = (u - s[0]) * G
            y = v + s[0]
            s[0] = y + v
            v = (y - s[1]) * G
            y = v + s[1]
            s[1] = y + v
            v = (y - s[2]) * G
            y = v + s[2]
            s[2] = y + v
            v = (y - s[3]) * G
            y = v + s[3]
            s[3] = y + v
            out[t] = y
    return out.astype(FLOAT)


def onepole_lp(x, cutoff):
    x = np.asarray(x, dtype=np.float64)
    n = x.shape[0]
    a = np.exp(-2.0 * math.pi * np.clip(np.broadcast_to(
        np.asarray(cutoff, dtype=np.float64), (n,)), 5.0, SR * 0.49) / SR)
    out = np.empty(n)
    y = 0.0
    for i in range(n):
        y = (1.0 - a[i]) * x[i] + a[i] * y
        out[i] = y
    return out.astype(FLOAT)


def crossover(x, fc: float):
    """Linkwitz-Riley 4th order split -> (low, high).

    Cascading two RBJ Q=0.707 sections makes the two outputs sum back to
    the input exactly.  A single LP + HP pair does not: their numerators
    add up to a null at the crossover frequency, which is how you get a
    mysterious -15 dB hole at 125 Hz in a finished mix.
    """
    x = ensure2(x)
    lo = lowpass(lowpass(x, fc, 0.7071), fc, 0.7071)
    hi = (x.astype(np.float32) - lo).astype(FLOAT)
    return lo, hi


def dc_block(x):
    return highpass(np.asarray(x, dtype=FLOAT), 16.0, 0.707)


def tilt_eq(x, tilt_db, pivot=900.0):
    return high_shelf(low_shelf(np.asarray(x, dtype=FLOAT), pivot, -tilt_db * 0.5),
                      pivot * 7.0, tilt_db)


# --------------------------------------------------------------------------
# time based fx
# --------------------------------------------------------------------------

def delay(x, time_s, fb=0.35, mix=0.3, damp=6000.0, pingpong=False):
    x = ensure2(x)
    d = max(1, sec(time_s))
    n = x.shape[0]
    out = np.zeros((n, 2), dtype=np.float64)
    for ch in range(2):
        src = x[:, ch].astype(np.float64)
        line = np.zeros(d)
        y = np.zeros(n)
        idx = 0
        prev = 0.0
        a = math.exp(-2.0 * math.pi * damp / SR)
        fb_c = fb * (0.92 if ch == 1 and pingpong else 1.0)
        src = src if ch == 0 or not pingpong else np.roll(src, max(1, d // 3))
        for i in range(n):
            v = line[idx]
            line[idx] = src[i] + v * fb_c
            v2 = (1 - a) * v + a * prev
            prev = v2
            y[i] = v2
            idx = idx + 1 if idx + 1 < d else 0
        out[:, ch] = (1 - mix) * src + mix * y
    return out.astype(FLOAT)


def make_ir(seconds=2.2, decay=2.2, predelay=0.012, size=1.0, damping=0.55,
            brightness=0.6, seed=7, early=16, tail_hp=0.0):
    """generated stereo impulse response: early reflections + dense decaying tail"""
    n = sec(seconds)
    g = rng(seed)
    ir = np.zeros((n, 2), dtype=np.float64)
    pd = sec(predelay)
    for ch in range(2):
        gg = rng(seed + 31 * ch)
        taps = gg.uniform(0.0, 1.0, early)
        for k in range(early):
            t = pd + int(sec(0.0035 + 0.05 * size) * (k + 1) * (0.65 + 0.7 * taps[k]))
            if t < n - 4:
                ir[t, ch] += (0.86 ** k) * (0.45 + 0.55 * taps[k])
    t = np.arange(n, dtype=np.float64) / SR
    env = np.exp(-decay * t / max(0.2, seconds) * 2.0)
    for ch in range(2):
        gg = rng(seed + 977 + ch)
        tail = gg.uniform(-1, 1, n)
        # frequency dependent decay: cascade of one-poles whose coefficient
        # opens over time -> highs die first, like a real room
        y = tail * env
        lp = 0.0
        co = 0.35 + 0.55 * damping
        for i in range(n):
            lp += co * (y[i] - lp)
            y[i] = lp
        y *= np.exp(-t * 1.2)
        ir[:, ch] += y * 0.9 + np.roll(y, 137) * 0.25
    ir = np.stack([low_shelf(ir[:, 0].astype(FLOAT), 120.0, -3.0),
                   low_shelf(ir[:, 1].astype(FLOAT), 120.0, -3.0)], 1)
    ir = np.stack([high_shelf(ir[:, 0].astype(FLOAT), 3800.0, -7.0 * damping + 4.0 * brightness),
                   high_shelf(ir[:, 1].astype(FLOAT), 3800.0, -7.0 * damping + 4.0 * brightness)], 1)
    if tail_hp > 0:
        ir = np.stack([highpass(ir[:, 0], tail_hp), highpass(ir[:, 1], tail_hp)], 1)
    return (ir / (np.max(np.abs(ir)) + 1e-9)).astype(FLOAT)


def _conv(x2, ir, block: int = 1 << 18):
    """stereo convolution by overlap-add.

    fftconvolve() on a three minute stem needs a 2^24 point transform -- about
    270 MB per temporary -- which kills small machines.  Blocking it keeps the
    peak allocation at a few MB and is usually faster too.
    """
    from scipy.signal import fftconvolve
    x2 = ensure2(x2)
    n, m = x2.shape[0], ir.shape[0]
    out = np.zeros((n + m, 2), dtype=np.float64)
    irs = [np.ascontiguousarray(ir[:, ch % ir.shape[1]], dtype=np.float64) for ch in range(2)]
    step = max(block, m)
    for start in range(0, n, step):
        end = min(n, start + step)
        for ch in range(2):
            seg = fftconvolve(x2[start:end, ch].astype(np.float64), irs[ch])
            out[start:start + seg.shape[0], ch] += seg
    return out


def reverb(x, ir, mix=0.25, predelay_s=0.0):
    x = ensure2(x)
    n = x.shape[0]
    if predelay_s > 0:
        p = sec(predelay_s)
        xp = np.pad(x, ((p, 0), (0, 0)))[:n + p]
    else:
        xp = x
    w = _conv(xp, ir)[:n]
    return ((1.0 - mix) * x.astype(np.float64) + mix * w).astype(FLOAT)


def send_reverb(x, ir, send=0.3, predelay_s=0.0):
    x = ensure2(x)
    n = x.shape[0]
    if predelay_s > 0:
        p = sec(predelay_s)
        xp = np.pad(x, ((p, 0), (0, 0)))[:n + p]
    else:
        xp = x
    return (_conv(xp, ir)[:n] * send).astype(FLOAT)


def convolve(x, ir, mix=1.0):
    """short-IR convolution (body resonances, cabinets, speaker sims)"""
    x = np.asarray(x, dtype=np.float64)
    n = x.shape[0]
    w = fftconvolve1(x, ir)[:n]
    return ((1 - mix) * x + mix * w).astype(FLOAT) if mix < 1.0 else w.astype(FLOAT)


def fftconvolve1(x, ir):
    from scipy.signal import fftconvolve
    return fftconvolve(np.asarray(x, dtype=np.float64), np.asarray(ir, dtype=np.float64))


def chorus(x, rate=0.6, depth=0.0035, voices=3, mix=0.5, seed=11):
    x = np.asarray(x, dtype=np.float64)
    n = x.shape[0]
    t = np.arange(n) / SR
    out = np.zeros(n)
    ph = rng(seed).uniform(0, 6.283, voices)
    for v in range(voices):
        d = depth * (0.6 + 0.8 * v / max(1, voices - 1))
        lfo = d * (0.5 + 0.5 * np.sin(2 * math.pi * rate * t * (1 + 0.13 * v) + ph[v]))
        idx = np.clip(np.arange(n) - lfo * SR, 0, n - 1.0001)
        i0 = idx.astype(int)
        f = idx - i0
        out += x[i0] * (1 - f) + x[i0 + 1] * f
    out /= voices
    return ((1 - mix) * x + mix * out).astype(FLOAT)


def stereo_chorus(x, rate=0.5, depth=0.004, mix=0.6, seed=5):
    l = chorus(x[:, 0], rate, depth, 3, mix, seed)
    r = chorus(x[:, 1], rate * 1.07, depth * 1.15, 3, mix, seed + 1)
    return np.stack([l, r], 1).astype(FLOAT)


# --------------------------------------------------------------------------
# dynamics (block-rate detectors -> fast, still musical)
# --------------------------------------------------------------------------

def _gain_curve(lvl, thresh, ratio, knee):
    g = np.zeros_like(lvl)
    lo, hi = thresh - knee / 2, thresh + knee / 2
    below = lvl < lo
    above = lvl > hi
    mid = ~(below | above)
    g[above] = (thresh - lvl[above]) * (1 - 1 / ratio)
    d = lvl[mid] - lo
    g[mid] = -(1 - 1 / ratio) * d * d / (2 * max(1e-6, knee))
    return g


def _expand(g, n, block, out=None):
    """linear-interpolate a block-rate control signal up to sample rate.

    Done in chunks into a pre-allocated float32 buffer: np.interp over a
    three minute song builds an int64 index array plus a float64 result,
    which is ~200 MB of pure garbage on a small machine.
    """
    if out is None:
        out = np.empty(n, dtype=np.float32)
    nb = g.shape[0]
    src = np.arange(nb, dtype=np.float32) * np.float32(block)
    step = 1 << 16
    for i in range(0, n, step):
        j = min(n, i + step)
        out[i:j] = np.interp(np.arange(i, j, dtype=np.float32), src, g).astype(np.float32)
    return out


def compressor(x, thresh_db=-18.0, ratio=3.0, attack=0.008, release=0.12,
               knee=6.0, makeup=0.0, block=32):
    x = ensure2(x)
    n = x.shape[0]
    nb = max(1, n // block)
    blocks = np.empty(nb, dtype=np.float32)
    step = nb * block
    for i in range(0, step, 1 << 16):
        j = min(step, i + (1 << 16))
        seg = x[i:j].astype(np.float32)
        k = (j - i) // block
        sq = (seg[:, 0] ** 2 + seg[:, 1] ** 2) * np.float32(0.5)
        blocks[i // block:i // block + k] = sq.reshape(k, block).mean(axis=1)
    lvl = 10 * np.log10(blocks.astype(np.float64) + 1e-12)
    aa = math.exp(-block / max(1e-6, attack * SR))
    ar = math.exp(-block / max(1e-6, release * SR))
    sm = np.empty(nb, dtype=np.float64)
    e = -120.0
    for i in range(nb):
        c = aa if lvl[i] > e else ar
        e = c * e + (1 - c) * lvl[i]
        sm[i] = e
    g = np.exp(_gain_curve(sm, thresh_db, ratio, knee) * (math.log(10) / 20.0))
    g = (g * db2amp(makeup)).astype(np.float32)
    gfull = _expand(g, n, block)
    out = np.empty((n, 2), dtype=np.float32)
    for i in range(0, n, 1 << 16):
        j = min(n, i + (1 << 16))
        out[i:j, 0] = x[i:j, 0] * gfull[i:j]
        out[i:j, 1] = x[i:j, 1] * gfull[i:j]
    return out


def limiter(x, ceil_db=-1.0, release=0.05, lookahead=0.002, block=32):
    x = ensure2(x)
    n = x.shape[0]
    peak = np.abs(x).max(axis=1).astype(np.float32)
    la = max(1, sec(lookahead))
    nb = max(1, (n + la) // block)
    pad = np.pad(peak, (0, max(0, nb * block + la - n)))
    win = np.lib.stride_tricks.sliding_window_view(pad, la)[::block]
    env = win.max(axis=1)[:nb].astype(np.float64)
    ceil = db2amp(ceil_db)
    g = np.minimum(1.0, ceil / np.maximum(env, 1e-9))
    rr = math.exp(-block / max(1e-6, release * SR))
    gs = np.empty(nb, dtype=np.float32)
    prev = 1.0
    for i in range(nb):
        prev = g[i] if g[i] < prev else rr * prev + (1 - rr) * g[i]
        gs[i] = prev
    del win, pad, peak
    gfull = _expand(gs, n, block)
    out = np.empty((n, 2), dtype=np.float32)
    for i in range(0, n, 1 << 16):
        j = min(n, i + (1 << 16))
        out[i:j, 0] = x[i:j, 0] * gfull[i:j]
        out[i:j, 1] = x[i:j, 1] * gfull[i:j]
    return out


def multiband(x, low=(-20.0, 2.2), mid=(-20.0, 1.8), high=(-22.0, 2.6),
              xover=(150.0, 3600.0)):
    x = ensure2(x).astype(FLOAT)
    lo = lowpass(x, xover[0])
    rest = x - lo
    md = lowpass(rest, xover[1])
    hi = rest - md
    del rest
    out = compressor(lo, *low)
    del lo
    hi = compressor(hi, *high)
    out += hi
    del hi
    md = compressor(md, *mid)
    out += md
    return out.astype(FLOAT)


def stereo_width(x, width=1.25):
    x = ensure2(x).astype(np.float64)
    m = (x[:, 0] + x[:, 1]) * 0.5
    s = (x[:, 0] - x[:, 1]) * 0.5 * width
    return np.stack([m + s, m - s], 1).astype(FLOAT)


def sidechain_gain(times, n, amount_db=6.0, attack=0.004, release=0.18):
    """the ducking envelope for a list of hit times (no signal needed)"""
    g = np.ones(n, dtype=np.float32)
    amt = db2amp(-abs(amount_db))
    la, lr = max(1, sec(attack)), max(1, sec(release))
    for t in times:
        i0 = sec(t)
        if i0 >= n:
            continue
        e = min(n, i0 + la)
        if e > i0:
            g[i0:e] = np.minimum(g[i0:e], 1 + (amt - 1) * np.linspace(0, 1, e - i0))
        s, e2 = i0 + la, min(n, i0 + la + lr)
        if e2 > s:
            g[s:e2] = np.minimum(g[s:e2], amt + (1 - amt) * np.linspace(0, 1, e2 - s))
    return g


def sidechain(x, times, amount_db=6.0, attack=0.004, release=0.18):
    x = ensure2(x)
    g = sidechain_gain(times, x.shape[0], amount_db, attack, release)
    out = np.empty_like(x)
    for i in range(0, x.shape[0], 1 << 16):
        j = min(x.shape[0], i + (1 << 16))
        out[i:j] = x[i:j] * g[i:j, None]
    return out


# --------------------------------------------------------------------------
# loudness
# --------------------------------------------------------------------------

def _k_weight(x):
    """BS.1770 K-weighting: RLB high-pass + 4 dB high shelf (chunked)"""
    from scipy.signal import butter
    x = np.asarray(x)
    b, a = butter(2, 38.0 / (SR / 2), "highpass")
    n = x.shape[0]
    out = np.empty(n, dtype=np.float64)
    zi = np.zeros(max(len(a), len(b)) - 1)
    step = 1 << 18
    for i in range(0, n, step):
        j = min(n, i + step)
        y, zi = lfilter(b, a, np.asarray(x[i:j], dtype=np.float64), zi=zi)
        out[i:j] = y
    return high_shelf(out.astype(FLOAT), 1681.0, 4.0, 0.7071).astype(np.float64)


def lufs(x):
    """integrated loudness with the BS.1770 absolute + relative gates.

    Works one channel at a time so a three minute song never needs a
    second full-length float64 copy of the mix.
    """
    x = ensure2(x)
    n = x.shape[0]
    win, hop = sec(0.4), sec(0.1)
    if n < win:
        return -70.0
    ms = np.zeros((n - win + hop) // hop, dtype=np.float64)
    for ch in range(2):
        y = _k_weight(x[:, ch])
        for i, k in enumerate(range(win, n, hop)):
            ms[i] += np.mean(y[k - win:k] ** 2)
    ms *= 0.5
    if ms.size == 0:
        return -70.0
    loud = -0.691 + 10.0 * np.log10(np.maximum(ms, 1e-12))
    keep = loud > -70.0
    if not np.any(keep):
        return -70.0
    keep &= loud > (loud[keep].mean() - 10.0)
    if not np.any(keep):
        return -70.0
    return float(-0.691 + 10.0 * np.log10(ms[keep].mean() + 1e-12))


def normalize_lufs(x, target=-11.0, ceil_db=-1.0):
    cur = lufs(x)
    if cur < -69:
        return x
    y = x * db2amp(min(30.0, target - cur))
    return limiter(y, ceil_db)


# --------------------------------------------------------------------------
# mix utilities
# --------------------------------------------------------------------------

def pan(x, p):
    """equal-power pan. mono -> stereo, stereo -> constant-power balance"""
    a = (float(np.clip(p, -1.0, 1.0)) + 1.0) * math.pi / 4.0
    x = np.asarray(x, dtype=np.float64)
    if x.ndim == 2 and x.shape[1] == 2:
        gl, gr = math.sqrt(2.0) * math.cos(a), math.sqrt(2.0) * math.sin(a)
        return np.stack([x[:, 0] * gl, x[:, 1] * gr], 1).astype(FLOAT)
    return np.stack([x * math.cos(a), x * math.sin(a)], 1).astype(FLOAT)


def calibrate(x, target_db: float = -3.0, perc: float = 99.0) -> np.ndarray:
    """normalise a rendered instrument to a predictable level.

    Uses a high percentile instead of the absolute peak so one stray
    transient cannot drag the whole sound down.
    """
    x = np.asarray(x, dtype=np.float64)
    if x.size == 0:
        return x.astype(FLOAT)
    p = float(np.percentile(np.abs(x), perc))
    if p < 1e-9:
        return x.astype(FLOAT)
    y = x * (db2amp(target_db) / p)
    # soft-limit anything that overshoots instead of hard clipping
    m = np.abs(y) > 0.7
    if np.any(m):
        a = np.abs(y[m])
        y[m] = np.sign(y[m]) * (0.7 + 0.3 * np.tanh((a - 0.7) / 0.3))
    return y.astype(FLOAT)


def mix_at(dst, src, at, gain=1.0, p=None):
    """add `src` into `dst` at time `at` (seconds).

    Kept in float32 on purpose: a full song is ~100 MB per stem and the
    float64 temporaries were blowing up memory on small machines.
    """
    src = ensure2(src).astype(FLOAT)
    i0 = sec(at)
    n = min(src.shape[0], dst.shape[0] - i0)
    if n <= 0 or i0 < 0:
        return dst
    if p is None or abs(p) < 1e-6:
        dst[i0:i0 + n] += src[:n] * np.float32(gain)
    else:
        dst[i0:i0 + n] += pan(src[:n], p) * np.float32(gain)
    return dst


def resample(x, ratio):
    x = np.asarray(x, dtype=np.float64)
    if abs(ratio - 1.0) < 1e-7:
        return x.astype(FLOAT)
    n = max(1, int(round(x.shape[0] / ratio)))
    idx = np.clip(np.arange(n) * ratio, 0, x.shape[0] - 1.0001)
    i0 = idx.astype(int)
    f = idx - i0
    return (x[i0] * (1 - f) + x[i0 + 1] * f).astype(FLOAT)


def fade(x, fin=0.005, fout=0.05):
    x = np.array(np.asarray(x, dtype=np.float64), copy=True)
    a, b = sec(fin), sec(fout)
    n = x.shape[0]
    ramp = (slice(None), None) if x.ndim == 2 else (slice(None),)
    if a:
        k = min(a, n)
        x[:k] *= np.linspace(0, 1, k)[ramp]
    if b:
        k = min(b, n)
        x[-k:] *= np.linspace(1, 0, k)[ramp]
    return x.astype(FLOAT)


def bitcrush(y: np.ndarray, bits: int = 5, hold: int = 1) -> np.ndarray:
    """reduce bit depth and, with hold > 1, sample rate as well"""
    bits = int(np.clip(bits, 1, 16))
    levels = 2.0 ** (bits - 1)
    q = np.round(np.clip(np.asarray(y, dtype=FLOAT), -1.0, 1.0) * levels) / levels
    if hold > 1:
        n = q.shape[0]
        k = np.clip((np.arange(n) // hold) * hold, 0, n - 1)
        q = q[k]
    return q.astype(FLOAT)


def ringmod(y: np.ndarray, freq: float, mix: float = 0.5) -> np.ndarray:
    """multiply by a sine: turns a drum into a bell, a gong or a machine"""
    t = np.arange(y.shape[0], dtype=np.float32) / SR
    m = np.sin(2 * np.pi * freq * t)[:, None]
    y = np.asarray(y, dtype=FLOAT)
    return (y * (1.0 - mix) + y * m * mix).astype(FLOAT)


def varispeed(y: np.ndarray, semis: float) -> np.ndarray:
    """resample in place, keeping the slot length: pitch shift + time change"""
    n = y.shape[0]
    if n < 8 or abs(semis) < 0.01:
        return y
    idx = np.clip(np.arange(n, dtype=np.float64) * (2.0 ** (semis / 12.0)), 0, n - 1)
    i0 = idx.astype(np.int64)
    i1 = np.minimum(i0 + 1, n - 1)
    f = (idx - i0)[:, None].astype(FLOAT)
    return (y[i0] * (1.0 - f) + y[i1] * f).astype(FLOAT)


def mangle(y: np.ndarray, seed: int = 1, amount: float = 1.0) -> np.ndarray:
    """Damage every hit differently.

    This is what makes programmed drums sound like they were run through a
    broken machine on purpose: each hit independently gets reversed, pitched,
    bit-crushed, ring-modulated or cut short.  Nothing repeats the same way
    twice, which is the difference between "processed" and "destroyed".
    """
    if amount <= 0.01:
        return y
    from .feel import onsets                      # lazy: feel imports core
    t, _, _ = onsets(np.asarray(mono(y), dtype=FLOAT), threshold=1.0)
    if t.size < 4:
        return y
    n = y.shape[0]
    edges = np.unique(np.concatenate(([0], (t * SR).astype(np.int64), [n])))
    edges = edges[(edges >= 0) & (edges <= n)]
    if edges.size < 3:
        return y
    out = np.empty_like(y)
    r = rng(seed)
    for a, b in zip(edges[:-1], edges[1:]):
        if b - a < 32:
            out[a:b] = y[a:b]
            continue
        seg = y[a:b]
        u = r.random(5)
        if u[0] < 0.15 * amount:                                   # backwards
            seg = np.ascontiguousarray(seg[::-1])
        if u[1] < 0.58 * amount:                                   # detuned hit
            seg = varispeed(seg, float(r.uniform(-12.0, 12.0)))
        if u[2] < 0.72 * amount:                                   # crushed
            seg = bitcrush(seg, int(r.integers(2, 8)), int(r.integers(1, 6)))
        if u[3] < 0.34 * amount:                                   # metallic
            seg = ringmod(seg, float(r.uniform(30.0, 2600.0)),
                          float(r.uniform(0.25, 0.75)))
        if u[4] < 0.33 * amount:                                   # cut short
            k = max(32, int(seg.shape[0] * float(r.uniform(0.12, 0.6))))
            seg = seg[:k]
        if seg.shape[0] >= b - a:
            out[a:b] = seg[:b - a]
        else:
            out[a:a + seg.shape[0]] = seg
            out[a + seg.shape[0]:b] = 0.0
        del seg
    return out


def stutter(y: np.ndarray, step_s: float, seed: int = 1,
            drop: float = 0.22, chop: float = 0.20) -> np.ndarray:
    """Gate a part into 16ths the way a sampler would: steps go missing or get
    cut in half.  This is the 'glitched' in glitched guitar."""
    n = y.shape[0]
    step = max(16, int(step_s * SR))
    g = np.ones(n, dtype=np.float32)
    r = rng(seed)
    i = 0
    while i < n:
        j = min(n, i + step)
        u = float(r.random())
        if u < drop:
            g[i:j] = 0.0                       # step goes missing
        elif u < drop + chop:
            g[i:min(n, i + step // 2)] = 0.0   # only the back half sounds
        i = j
    # 0.5 ms ramps so it clicks like an edit, not like a bug
    c = np.concatenate(([0.0], np.cumsum(g, dtype=np.float64)))
    a = np.clip(np.arange(n) - 12, 0, n - 1)
    b = np.clip(np.arange(n) + 12, 0, n - 1)
    g = ((c[b] - c[a]) / np.maximum(1, b - a)).astype(FLOAT)
    return (y * g[:, None]).astype(FLOAT)


def bend(y: np.ndarray, semis: float = 0.8, rate: float = 2.5) -> np.ndarray:
    """Varispeed pitch bend: starts `semis` sharp and settles back to pitch.
    The cheap, convincing version of a bent guitar string."""
    n = y.shape[0]
    if n < 64:
        return y
    t = np.arange(n, dtype=np.float64) / SR
    curve = semis * np.exp(-rate * t)
    idx = np.cumsum(2.0 ** (curve / 12.0))
    idx *= (n - 1) / idx[-1]
    i0 = np.clip(idx.astype(np.int64), 0, n - 1)
    i1 = np.minimum(i0 + 1, n - 1)
    f = (idx - i0)[:, None].astype(FLOAT)
    return (y[i0] * (1.0 - f) + y[i1] * f).astype(FLOAT)


def write(path, x, sr=SR, subtype=None):
    import soundfile as sf
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    x = np.clip(np.asarray(x, dtype=FLOAT), -1.0, 1.0)
    if subtype is None:
        subtype = "PCM_24" if str(path).endswith(".wav") else None
    sf.write(path, x, sr, subtype=subtype)
    return path


def read(path):
    import soundfile as sf
    x, sr = sf.read(path, dtype="float32", always_2d=False)
    return x, sr
