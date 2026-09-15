"""
Synthesised drum kit -- no samples, no copyright, nothing 8-bit.

Every drum is built from oscillators + shaped noise + filtered transients and
then driven through saturation, so kicks have sub weight, snares have wires
and hats have real metallic inharmonicity (the classic 6-oscillator ratio
stack) instead of plain white noise.
"""
from __future__ import annotations

import math

import numpy as np

from ..core import (SR, FLOAT, sec, m2f, noise, rng, db2amp, osc_saw, osc_sine,
                    osc_square, tanh_sat, exp_decay, adsr, filt_env, lowpass,
                    highpass, bandpass, peak_eq, low_shelf, high_shelf,
                    make_ir, reverb, norm, fit, ensure2, pan, mix_at, lfilter,
                    calibrate)

# --------------------------------------------------------------------------
# individual drums (all return mono float32 with their natural tail)
# --------------------------------------------------------------------------

def kick(tune: float = 48.0, decay: float = 0.45, punch: float = 0.7,
         click: float = 0.5, sat: float = 1.7, sub: float = 0.6,
         eight08: bool = False, dur: float | None = None, seed: int = 1) -> np.ndarray:
    f = m2f(tune)
    dur = dur or (decay * (2.2 if eight08 else 1.4) + 0.25)
    n = sec(dur)
    t = np.arange(n) / SR
    top = f * (4.2 if not eight08 else 2.4)
    tau = 0.018 if not eight08 else 0.055
    freq = f + (top - f) * np.exp(-t / tau)
    body = osc_sine(freq, n)
    amp = np.exp(-t / (decay * 0.42)) * (1 - np.exp(-t / 0.0015))
    # weight: an octave-down sine that only lives in the first 120 ms
    subw = np.sin(2 * math.pi * np.cumsum(freq * 0.5) / SR) * np.exp(-t / (decay * 0.3)) * sub
    # beater click
    cn = noise(min(n, sec(0.02)), seed)
    cl = np.zeros(n)
    cl[:len(cn)] = cn
    cl = bandpass(cl, 2200.0, 0.9) * np.exp(-t / 0.0045) * click * 0.9
    blip = np.sin(2 * math.pi * 2600 * t) * np.exp(-t / 0.006) * 0.18 * click
    y = body * amp + subw * amp * 0.8 + cl + blip
    y = tanh_sat(y * (1.0 + punch * 1.4), drive=sat)
    y = highpass(y, 24.0, 0.707)
    y = lowpass(y, 9000.0, 0.707) if not eight08 else lowpass(y, 4200.0, 0.707)
    y = peak_eq(y, 62.0, 3.0 * sub, 1.1)
    return (y * np.exp(-t / (dur * 0.9))).astype(FLOAT)


def snare(tune: float = 186.0, decay: float = 0.19, wires: float = 0.65,
          crack: float = 0.8, body: float = 0.8, room: float = 0.22,
          dur: float | None = None, seed: int = 2) -> np.ndarray:
    dur = dur or (decay * 2.0 + 0.3)
    n = sec(dur)
    t = np.arange(n) / SR
    f = m2f(tune) if tune < 100 else tune
    # two tonal membranes with a fast downward pitch bend
    fenv = f * (1.0 + 0.55 * np.exp(-t / 0.03))
    tone = (np.sin(2 * math.pi * np.cumsum(fenv) / SR) +
            0.7 * np.sin(2 * math.pi * np.cumsum(fenv * 1.58) / SR) +
            0.35 * np.sin(2 * math.pi * np.cumsum(fenv * 2.13) / SR))
    tone *= np.exp(-t / (decay * 0.55)) * 0.42 * body
    # shell / body: band-limited noise around 1.7 kHz
    ns = noise(n, seed)
    shell = bandpass(ns, 1750.0, 0.8) * np.exp(-t / (decay * 0.75)) * 0.5 * body
    # wires: bright noise with a fast rattle AM
    ratt = (0.55 + 0.45 * np.sin(2 * math.pi * 190 * t) * np.exp(-t / 0.03))
    wire = highpass(ns, 3600.0, 0.7) * np.exp(-t / (decay * 0.42)) * ratt * wires * 0.42
    # stick crack
    cr = np.zeros(n)
    m = min(n, sec(0.03))
    cr[:m] = noise(m, seed + 9)
    cr = highpass(cr, 4200.0, 0.8) * np.exp(-t / 0.0035) * crack * 0.75
    y = tone + shell + wire + cr
    y = tanh_sat(y, 1.5)
    y = highpass(y, 130.0, 0.707)
    y = peak_eq(y, 320.0, 2.5, 1.0)
    if room > 0:
        ir = make_ir(0.28, decay=3.4, size=0.25, damping=0.75, brightness=0.5, seed=seed + 3)
        y = reverb(ensure2(y), ir, room)[:, 0] * 0.85 + y * 0.5
    return y.astype(FLOAT)


def hat(open_: bool = False, decay: float | None = None, tone: float = 1.0,
        level: float = 1.0, dur: float | None = None, seed: int = 5) -> np.ndarray:
    decay = decay if decay is not None else (0.42 if open_ else 0.055)
    dur = dur or (decay * 1.6 + 0.15)
    n = sec(dur)
    t = np.arange(n) / SR
    # 808-style inharmonic square stack
    ratios = [205.3, 304.4, 369.6, 522.7, 540.0, 800.0]
    met = np.zeros(n)
    for i, r in enumerate(ratios):
        met += osc_square(r * tone, n, phase=i * 0.13) * (1.0 / (1 + i * 0.35))
    met = bandpass(met, 9200.0, 0.7)
    met = highpass(met, 6200.0, 0.8)
    ns = noise(n, seed)
    air = bandpass(ns, 10500.0, 1.1) * 0.5 + highpass(ns, 7500.0, 0.7) * 0.25
    env = np.exp(-t / decay) * (1 - np.exp(-t / 0.0008))
    y = (met * 0.55 + air * 0.7) * env
    y = tanh_sat(y * 1.6, 1.8)
    y = highpass(y, 5200.0, 0.707)
    y = y * level * (0.85 if open_ else 1.0)
    return y.astype(FLOAT)


def clap(spread: float = 0.011, decay: float = 0.16, level: float = 1.0,
         dur: float | None = None, seed: int = 7) -> np.ndarray:
    dur = dur or (decay * 2.2 + 0.25)
    n = sec(dur)
    t = np.arange(n) / SR
    y = np.zeros(n)
    g = rng(seed)
    for k, off in enumerate([0.0, spread, spread * 2.0, spread * 3.1]):
        i0 = sec(off)
        m = min(n - i0, sec(0.02))
        y[i0:i0 + m] += noise(m, seed + k) * (1.0 - k * 0.16)
    y = bandpass(y, 1150.0, 1.5)
    y *= np.exp(-t / 0.012)
    # tail
    tail = noise(n, seed + 21)
    tail = bandpass(tail, 1500.0, 1.1) * np.exp(-t / (decay * 0.6)) * 0.55
    y = y + tail
    y = tanh_sat(y, 1.4)
    y = highpass(y, 420.0, 0.707)
    return (y * level * 0.9).astype(FLOAT)


def tom(tune: float = 120.0, decay: float = 0.32, shell: float = 0.6,
        dur: float | None = None, seed: int = 11) -> np.ndarray:
    f = m2f(tune) if tune < 100 else tune
    dur = dur or (decay * 1.8 + 0.2)
    n = sec(dur)
    t = np.arange(n) / SR
    fenv = f * (1.0 + 0.9 * np.exp(-t / 0.035))
    body = (np.sin(2 * math.pi * np.cumsum(fenv) / SR) +
            0.5 * np.sin(2 * math.pi * np.cumsum(fenv * 1.51) / SR) +
            0.22 * np.sin(2 * math.pi * np.cumsum(fenv * 2.42) / SR))
    body *= np.exp(-t / (decay * 0.6)) * 0.5
    atk = np.zeros(n)
    m = min(n, sec(0.012))
    atk[:m] = noise(m, seed)
    atk = bandpass(atk, 2400.0, 0.9) * np.exp(-t / 0.006) * shell * 0.5
    y = tanh_sat(body + atk, 1.4)
    return highpass(y, 70.0, 0.707).astype(FLOAT)


def shaker(decay: float = 0.085, level: float = 1.0, bright: float = 1.0,
           dur: float | None = None, seed: int = 13) -> np.ndarray:
    dur = dur or (decay * 3 + 0.1)
    n = sec(dur)
    t = np.arange(n) / SR
    ns = noise(n, seed)
    y = bandpass(ns, 6200.0 * bright, 2.2) * 0.8 + highpass(ns, 4200.0, 0.8) * 0.35
    env = (1 - np.exp(-t / 0.006)) * np.exp(-t / decay)
    y = y * env * level
    return highpass(y, 2600.0, 0.707).astype(FLOAT)


def tambourine(decay: float = 0.16, level: float = 1.0, dur: float | None = None,
               seed: int = 17) -> np.ndarray:
    dur = dur or (decay * 2.6 + 0.1)
    n = sec(dur)
    t = np.arange(n) / SR
    ns = noise(n, seed)
    y = np.zeros(n)
    for f, q, g in [(7600, 5.0, 1.0), (9400, 6.0, .8), (11800, 7.0, .6), (5300, 4.0, .7)]:
        y += bandpass(ns, f, q) * g
    y *= (1 - np.exp(-t / 0.004)) * np.exp(-t / decay) * 0.28 * level
    return highpass(y, 3600.0, 0.707).astype(FLOAT)


def cowbell(tune: float = 800.0, decay: float = 0.32, level: float = 1.0,
            dur: float | None = None) -> np.ndarray:
    dur = dur or (decay * 1.6 + 0.1)
    n = sec(dur)
    t = np.arange(n) / SR
    y = osc_square(540.0, n) * 0.7 + osc_square(tune, n) * 0.7
    y = bandpass(y, 2640.0, 1.3)
    y *= np.exp(-t / decay) * (1 - np.exp(-t / 0.002)) * 0.5 * level
    return y.astype(FLOAT)


def rim(level: float = 1.0, dur: float = 0.09, seed: int = 19) -> np.ndarray:
    n = sec(dur)
    t = np.arange(n) / SR
    ns = noise(n, seed)
    y = bandpass(ns, 1700.0, 3.0) * 0.8 + bandpass(ns, 420.0, 6.0) * 0.9
    y *= np.exp(-t / 0.012) * level
    return (y * 0.6).astype(FLOAT)


def crash(decay: float = 2.6, level: float = 1.0, dur: float | None = None,
          seed: int = 23) -> np.ndarray:
    dur = dur or (decay + 0.5)
    n = sec(dur)
    t = np.arange(n) / SR
    y = np.zeros(n)
    base = 40.0
    for i, r in enumerate([1.0, 1.41, 1.78, 2.19, 2.67, 3.11, 3.74, 4.31, 5.07, 6.3, 7.9]):
        y += osc_square(base * r * 8.3, n, phase=i * 0.21) / (1 + i * 0.55)
    y = highpass(y, 3200.0, 0.7)
    ns = noise(n, seed)
    shimmer = (bandpass(ns, 9000.0, 0.9) * 0.7 + highpass(ns, 6000.0, 0.7) * 0.4)
    env = (1 - np.exp(-t / 0.006)) * np.exp(-t / decay)
    y = (y * 0.32 + shimmer * 0.85) * env
    # the metallic wash keeps ringing a little longer than the noise
    y += bandpass(ns, 12500.0, 1.4) * np.exp(-t / (decay * 0.55)) * 0.12
    y = tanh_sat(y * 1.4, 1.5)
    return (highpass(y, 2600.0, 0.707) * level * 0.75).astype(FLOAT)


def ride(decay: float = 1.3, level: float = 1.0, dur: float | None = None,
         seed: int = 29) -> np.ndarray:
    dur = dur or (decay + 0.4)
    n = sec(dur)
    t = np.arange(n) / SR
    y = np.zeros(n)
    for i, r in enumerate([1.0, 1.52, 2.05, 2.71, 3.44, 4.62, 6.1]):
        y += osc_square(330.0 * r, n, phase=i * 0.17) / (1 + i * 0.8)
    y = bandpass(y, 6800.0, 0.8)
    env = np.exp(-t / decay) * (1 - np.exp(-t / 0.0015))
    ping = (np.sin(2 * math.pi * np.cumsum(np.full(n, 1180.0)) / SR) *
            np.exp(-t / 0.06) * 0.25)
    ns = noise(n, seed)
    air = highpass(ns, 7000.0, 0.7) * env * 0.3
    y = (y * 0.35 + air + ping) * level
    return (highpass(y, 2400.0, 0.707) * 0.8).astype(FLOAT)


def taiko(tune: float = 36.0, decay: float = 0.5, level: float = 1.0,
          dur: float | None = None, seed: int = 31) -> np.ndarray:
    f = m2f(tune) if tune < 100 else tune
    dur = dur or (decay * 2.2 + 0.3)
    n = sec(dur)
    t = np.arange(n) / SR
    fenv = f * (1.0 + 0.5 * np.exp(-t / 0.05))
    body = np.sin(2 * math.pi * np.cumsum(fenv) / SR) * np.exp(-t / (decay * 0.5))
    ns = noise(n, seed)
    atk = bandpass(ns, 380.0, 1.1) * np.exp(-t / 0.03) * 0.6
    y = tanh_sat(body * 0.8 + atk, 1.8)
    return (lowpass(y, 2600.0, 0.707) * level).astype(FLOAT)


# --------------------------------------------------------------------------
# kit registry
# --------------------------------------------------------------------------

def hit(name: str, **kw) -> np.ndarray:
    """render one drum by name, e.g. hit('kick', tune=44, decay=.6)"""
    f = {
        "kick": kick, "snare": snare, "hat": hat, "openhat": lambda **k: hat(True, **k),
        "clap": clap, "tom": tom, "shaker": shaker, "tamb": tambourine,
        "cowbell": cowbell, "rim": rim, "crash": crash, "ride": ride, "taiko": taiko,
        "808": lambda **k: kick(eight08=True, **k),
    }[name]
    # relative kit balance (dBFS) so a rendered pattern sits right without
    # any per-song fiddling
    bal = {"kick": 0.0, "808": 1.0, "snare": -0.5, "hat": -4.5, "openhat": -5.5,
           "clap": -2.5, "tom": -3.0, "shaker": -7.0, "tamb": -6.0,
           "cowbell": -6.0, "rim": -7.0, "crash": -5.0, "ride": -7.5,
           "taiko": -1.0}
    return calibrate(f(**kw), bal.get(name, -4.0), perc=100.0)


KITS = {
    "modern": {"kick": dict(tune=36, decay=0.42, punch=0.75, click=0.55, sub=0.75),
               "snare": dict(tune=190, decay=0.17, wires=0.7, crack=0.85, room=0.2),
               "hat": dict(tone=1.0, level=0.85),
               "openhat": dict(tone=1.0, level=0.7),
               "clap": dict(decay=0.15),
               "crash": dict(decay=2.4)},
    "trap": {"kick": dict(tune=31, decay=0.75, punch=0.8, click=0.4, sub=1.0, eight08=True),
             "snare": dict(tune=200, decay=0.13, wires=0.8, crack=0.95, room=0.12),
             "hat": dict(tone=1.08, level=0.8),
             "clap": dict(decay=0.13),
             "808": dict(tune=28, decay=0.9, sub=1.0)},
    "acoustic": {"kick": dict(tune=38, decay=0.32, punch=0.5, click=0.75, sub=0.35),
                 "snare": dict(tune=185, decay=0.22, wires=0.75, crack=0.8, room=0.32),
                 "hat": dict(tone=0.95, level=0.75),
                 "ride": dict(decay=1.5),
                 "crash": dict(decay=2.8),
                 "tom": dict(tune=140)},
    "epic": {"kick": dict(tune=33, decay=0.7, punch=0.85, click=0.4, sub=0.65),
             "snare": dict(tune=170, decay=0.28, wires=0.45, crack=0.7, room=0.4),
             "taiko": dict(tune=36, decay=0.7),
             "hat": dict(tone=1.0, level=0.85),
             "openhat": dict(tone=1.0, level=0.7),
             "crash": dict(decay=3.0),
             "tamb": dict(decay=0.2)},
    "lofi": {"kick": dict(tune=37, decay=0.3, punch=0.4, click=0.3, sub=0.5),
             "snare": dict(tune=195, decay=0.16, wires=0.5, crack=0.5, room=0.35),
             "rim": dict(),
             "hat": dict(tone=0.85, level=0.55)},
    "rave": {"kick": dict(tune=34, decay=0.42, punch=0.95, click=0.65, sub=0.55),
             "snare": dict(tune=215, decay=0.12, wires=0.9, crack=1.0, room=0.10),
             "clap": dict(decay=0.16),
             "hat": dict(tone=1.1, level=0.85),
             "openhat": dict(tone=1.1, level=0.8),
             "shaker": dict(decay=0.06, level=0.9),
             "crash": dict(decay=2.2)},
    "edm": {"kick": dict(tune=35, decay=0.5, punch=0.9, click=0.6, sub=0.8),
            "snare": dict(tune=210, decay=0.14, wires=0.85, crack=1.0, room=0.15),
            "clap": dict(decay=0.18),
            "hat": dict(tone=1.05, level=0.8),
            "openhat": dict(tone=1.05, level=0.75),
            "crash": dict(decay=2.2)},
}


# --------------------------------------------------------------------------
# patterns
# --------------------------------------------------------------------------

def _grid(steps_per_bar: int = 16):
    return np.zeros(steps_per_bar)


STYLE_GRIDS = {
    # name: dict(instrument -> 16-step string, 'x' = accent, 'o' = normal, 'g' = ghost)
    "pop": {
        "kick": "x.......x.......",
        "snare": "....x.......x...",
        "hat": "..x...x...x...x.",
        "openhat": "..............o.",
        "clap": "....o.......o...",
    },
    "trap": {
        "kick": "x......x..x.....",
        "snare": "....x.......x...",
        "hat": "xxx.xxx.xxx.xxox",
        "clap": "....x.......x...",
        "808": "x.......x...x...",
    },
    "edm": {
        "kick": "x...x...x...x...",
        "snare": "................",
        "clap": "....x.......x...",
        "hat": "..x...x...x...x.",
        "openhat": "......o.......o.",
    },
    "four_on_floor": {
        "kick": "x...x...x...x...",
        "snare": "....x.......x...",
        "hat": "..x...x...x...x.",
        "openhat": "......o.......o.",
        "crash": "x...............",
    },
    "lofi": {
        "kick": "x.....x...x.....",
        "snare": "....x.......x...",
        "hat": "..x...x...x...x.",
        "rim": "..........o.....",
    },
    "epic": {
        "kick": "x.......x...x...",
        "taiko": "x...x...x...x...",
        "snare": "....x.......x...",
        "hat": "..x...x...x...x.",
        "openhat": "..............o.",
        "crash": "x...............",
        "tamb": "..o...o...o...o.",
    },
    "ballad": {
        "kick": "x.......x.......",
        "snare": "....x.......x...",
        "hat": "....x.......x...",
        "ride": "x.x.x.x.x.x.x.x.",
    },
    "break": {
        "kick": "x..x....x.......",
        "snare": "....x..x....x.x.",
        "hat": "x.x.x.x.x.x.x.x.",
        "crash": "x...............",
    },
    "rave": {
        "kick": "x...x...x...x...",
        "clap": "....x.......x...",
        "hat": "x.x.x.x.x.x.x.x.",
        "openhat": "..o...o...o...o.",
        "shaker": "xxxxxxxxxxxxxxxx",
        "crash": "x...............",
    },
    "rave_intro": {
        "kick": "x...x...x...x...",
        "hat": "x.x.x.x.x.x.x.x.",
        "shaker": "x.x.x.x.x.x.x.x.",
        "crash": "x...............",
    },
    "none": {},
}


def pattern(style: str, bars: int = 4, bpm: float = 90.0, seed: int = 1,
            fill_every: int = 0, variation: float = 0.25) -> list[tuple[float, str, float]]:
    """-> list of (time_seconds, instrument, velocity)"""
    g = np.random.default_rng(seed)
    grid = STYLE_GRIDS.get(style, STYLE_GRIDS["pop"])
    spb = 60.0 / bpm
    step = spb / 4.0
    events = []
    for bar in range(bars):
        is_fill = fill_every and (bar % fill_every == fill_every - 1) and bar < bars - 1
        for inst, pat in grid.items():
            if inst in ("crash", "ride") and bar % 4 != 0:
                continue
            for s, ch in enumerate(pat):
                if ch == ".":
                    continue
                vel = {"x": 1.0, "o": 0.72, "g": 0.42}[ch]
                vel *= float(1.0 + g.uniform(-variation, variation) * 0.5)
                # hats get a swinging, human feel
                if inst in ("hat", "openhat", "ride", "tamb") and s % 2 == 1:
                    vel *= 0.78
                events.append((bar * 4 * spb + s * step, inst, float(np.clip(vel, 0.15, 1.2))))
        if is_fill:
            # 16th note snare/tom roll into the next bar
            for k in range(4):
                t = bar * 4 * spb + (12 + k) * step
                inst = "snare" if k % 2 == 0 else "tom"
                events.append((t, inst, 0.55 + k * 0.14))
    return sorted(events)


def render_pattern(events, kit: str = "modern", length: float | None = None,
                   tail: float = 3.0, room: float = 0.0, room_size: float = 1.2,
                   seed: int = 1, pan_map: dict | None = None) -> np.ndarray:
    """render a drum pattern to stereo"""
    cfg = KITS.get(kit, KITS["modern"])
    end = max([e[0] for e in events], default=0.0)
    n = sec((length if length else end) + tail)
    out = np.zeros((n, 2), dtype=np.float64)
    pan_map = pan_map or {}
    cache: dict[tuple, np.ndarray] = {}
    for i, (t, inst, vel) in enumerate(events):
        base = "808" if inst == "808" else inst
        params = dict(cfg.get(base, {}))
        if inst == "hat" and vel > 0.95:
            params["level"] = params.get("level", 1.0) * 1.15
        key = (inst, tuple(sorted((k, float(v)) for k, v in params.items())))
        y = cache.get(key)
        if y is None:
            y = hit(inst, seed=seed + i, **params)
            # tiny per-hit tuning variation so repeated hits are not identical
            cache[key] = y
        if vel < 1.0:
            y = y * (vel ** 1.4)
        p = pan_map.get(inst, 0.0)
        mix_at(out, y, t, gain=1.0, p=p)
    if room > 0:
        ir = make_ir(1.4 * room_size, decay=3.0, size=room_size, damping=0.7,
                     brightness=0.55, seed=seed + 77)
        wet = _send(out, ir)
        out = out * (1 - room * 0.35) + wet * room
    return out.astype(FLOAT)


def _send(x, ir):
    from ..core import send_reverb
    return send_reverb(x.astype(FLOAT), ir, 1.0)


def demo_kit(path: str = "kit_demo.wav"):
    """one bar of every drum, useful for auditioning the kit"""
    evs = []
    names = ["kick", "snare", "hat", "openhat", "clap", "tom", "rim", "shaker",
             "tamb", "cowbell", "crash", "ride", "taiko"]
    for i, nm in enumerate(names):
        evs.append((i * 0.5, nm, 1.0))
    y = render_pattern(evs, kit="modern", tail=3.5)
    from ..core import write
    write(path, y)
    return path
