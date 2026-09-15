"""Rhythmic analysis -- answers 'is this a song or one long note?'."""
from __future__ import annotations

import numpy as np

from .core import SR, mono


def spectral_flux(x, hop: int = 512, n_fft: int = 1024):
    """onset strength envelope"""
    x = np.asarray(mono(x), dtype=np.float64)
    win = np.hanning(n_fft)
    n_frames = max(1, (x.shape[0] - n_fft) // hop)
    S = np.empty((n_frames, n_fft // 2 + 1), dtype=np.float64)
    for i in range(n_frames):
        seg = x[i * hop:i * hop + n_fft] * win
        S[i] = np.abs(np.fft.rfft(seg))
    S = np.log1p(S * 20.0)
    flux = np.diff(S, axis=0).clip(min=0).sum(axis=1)
    return flux, hop / SR


def onsets(x, hop: int = 512, threshold: float = 1.1):
    """onset times (seconds) by adaptive peak picking on the flux envelope"""
    flux, dt = spectral_flux(x, hop)
    if flux.size < 8:
        return np.zeros(0), flux, dt
    win = int(round(0.12 / dt))
    med = np.array([np.median(flux[max(0, i - win):i + win]) for i in range(flux.size)])
    thr = med * threshold + 0.06 * flux.mean()
    peaks = []
    i = 1
    while i < flux.size - 1:
        if flux[i] > flux[i - 1] and flux[i] >= flux[i + 1] and flux[i] > thr[i]:
            peaks.append(i)
            i += max(1, int(round(0.035 / dt)))
        else:
            i += 1
    return np.array(peaks) * dt, flux, dt


def report(x, bpm: float = 120.0) -> str:
    x = mono(x)
    t, flux, dt = onsets(x)
    dur = x.shape[0] / SR
    n = t.size
    rate = n / max(1e-6, dur)
    # inter-onset intervals -> how much of it lands on a steady grid
    if n > 4:
        ioi = np.diff(t)
        ioi = ioi[(ioi > 0.02) & (ioi < 2.0)]
        beat = 60.0 / bpm
        grid_err = np.minimum(np.abs(ioi - beat / 2), np.abs(ioi - beat)).mean()
        steadiness = float(np.mean(np.abs(ioi - np.median(ioi)) < 0.03))
    else:
        grid_err, steadiness = 9.9, 0.0
    # low-band (kick) rate is what makes a track feel like it is moving
    from .core import lowpass, highpass
    low = lowpass(np.asarray(x, dtype='float32'), 150.0, 0.707)
    lt, _, _ = onsets(low, threshold=1.3)
    kick_rate = lt.size / max(1e-6, dur)
    return (f"{n} onsets ({rate:.1f}/s), kick-band {kick_rate:.1f}/s, "
            f"grid fit ±{grid_err*1000:.0f} ms, steady IOIs {steadiness*100:.0f}%")
