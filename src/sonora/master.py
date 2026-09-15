"""
Master bus: glue, multiband, width, loudness.  This is the difference
between "a bunch of synths" and something that sits next to a commercial
release on a playlist.
"""
from __future__ import annotations

import numpy as np

from .core import (SR, FLOAT, ensure2, highpass, lowpass, tilt_eq, peak_eq,
                   low_shelf, high_shelf, compressor, multiband, limiter,
                   stereo_width, tanh_sat, normalize_lufs, lufs, fade,
                   db2amp, amp2db, rms)


def master(x: np.ndarray, target_lufs: float = -11.0, glue: float = 1.0,
           width: float = 1.12, air: float = 1.0, punch: float = 1.0,
           ceil_db: float = -1.0, low_cut: float = 18.0) -> np.ndarray:
    x = ensure2(x).astype(np.float64)
    # 1. clean up rumble that only eats headroom
    x = highpass(x.astype(FLOAT), low_cut, 0.707).astype(np.float64)
    # 2. static tone shaping: fix the two faults every synth mix has --
    #    a boxy 300-800 build-up and too much unsourced high end
    x = peak_eq(x.astype(FLOAT), 42.0, -2.5, 0.7).astype(np.float64)
    x = peak_eq(x.astype(FLOAT), 130.0, 3.2 * punch, 0.8).astype(np.float64)
    x = peak_eq(x.astype(FLOAT), 450.0, -2.0, 1.0).astype(np.float64)
    x = peak_eq(x.astype(FLOAT), 3500.0, 2.6, 0.7).astype(np.float64)
    x = high_shelf(x.astype(FLOAT), 7000.0, -0.5 * air, 0.7).astype(np.float64)
    x = lowpass(x.astype(FLOAT), 19000.0, 0.707).astype(np.float64)
    x = tilt_eq(x.astype(FLOAT), 0.3 * air).astype(np.float64)
    # 3. glue
    x = compressor(x.astype(FLOAT), -20.0, 1.0 + 1.6 * glue, 0.018, 0.16,
                   knee=10.0, makeup=1.5 * glue).astype(np.float64)
    # 4. multiband control (keeps the low end from pumping the whole mix)
    x = multiband(x.astype(FLOAT),
                  low=(-19.0, 1.0 + 1.5 * glue),
                  mid=(-20.0, 1.0 + 1.1 * glue),
                  high=(-22.0, 1.0 + 1.8 * glue)).astype(np.float64)
    # 5. stereo image
    x = stereo_width(x.astype(FLOAT), width).astype(np.float64)
    # 6. gentle bus saturation for perceived loudness
    x = tanh_sat(x, 1.18, ceiling=0.92).astype(np.float64)
    # 7. limiter + loudness
    x = normalize_lufs(x.astype(FLOAT), target_lufs, ceil_db).astype(np.float64)
    return x.astype(FLOAT)


def stem_fx(x: np.ndarray, hp: float | None = None, lp: float | None = None,
            comp: tuple | None = None, sat: float = 0.0, width: float = 1.0,
            peak: tuple | None = None, gain_db: float = 0.0) -> np.ndarray:
    """per-stem insert chain"""
    from .core import db2amp
    x = ensure2(x)
    if hp:
        x = highpass(x, hp, 0.707)
    if lp:
        x = lowpass(x, lp, 0.707)
    if comp:
        x = compressor(x, *comp)
    if sat:
        x = tanh_sat(x, 1.0 + sat * 3.0, ceiling=0.9)
    if peak:
        x = peak_eq(x, *peak)
    if width != 1.0:
        x = stereo_width(x, width)
    if gain_db:
        x = (x.astype(np.float64) * db2amp(gain_db)).astype(FLOAT)
    return x


def report(x: np.ndarray) -> str:
    x = ensure2(x)
    m, s = x[:, 0].astype(np.float64), x[:, 1].astype(np.float64)
    cor = float(np.corrcoef(m, s)[0, 1]) if x.shape[0] > 2 else 0.0
    return (f"{x.shape[0]/SR:6.2f}s  LUFS {lufs(x):6.2f}  peak {amp2db(np.abs(x).max()):6.2f} dBFS  "
            f"rms {amp2db(rms(x)):6.2f} dBFS  correlation {cor:+.2f}")
