"""
Virtual analogue synthesiser: 2 oscillators + sub + noise, unison spread,
time-varying ladder/SVF filter, ADSR, LFO, drive, chorus.

The preset dictionary covers the modern production palette: wide supersaw
stacks, reese bass, moog-style mono bass, plucks, brass, dream pads.
"""
from __future__ import annotations

import math

import numpy as np

from ..core import (SR, FLOAT, sec, m2f, noise, rng, db2amp, osc_saw, osc_square,
                    osc_tri, osc_sine, supersaw, tanh_sat, adsr, filt_env, ladder,
                    lowpass, highpass, bandpass, chorus, fade, ensure2, calibrate)


# --------------------------------------------------------------------------
# one voice
# --------------------------------------------------------------------------

def voice(midi: float, dur: float = 1.0, vel: float = 1.0,
          wave_a: str = "saw", wave_b: str = "saw", oct_a: int = 0, oct_b: int = 0,
          detune: float = 12.0, mix: float = 0.6, sub: float = 0.0,
          noise_lvl: float = 0.0, unison: int = 3, spread: float = 0.4,
          cutoff: float = 3000.0, res: float = 0.7, env_amt: float = 0.6,
          ftype: str = "lp24", keytrack: float = 0.0, amp_a: float = 0.01,
          amp_d: float = 0.15, amp_s: float = 0.7, amp_r: float = 0.25,
          f_a: float = 0.01, f_d: float = 0.3, f_s: float = 0.35, f_r: float = 0.3,
          lfo_rate: float = 0.0, lfo_depth: float = 0.0, lfo_target: str = "filter",
          drive: float = 1.2, glide: float = 0.0, seed: int = 1,
          vel_to_cutoff: float = 0.5, vel_to_amp: float = 0.6) -> np.ndarray:
    """render one synth note (mono)"""
    n = sec(dur + amp_r + 0.08)
    t = np.arange(n) / SR
    f = m2f(midi)
    g = rng(seed)

    # pitch: glide from a fifth below on very short glide times, plus LFO
    freq = np.full(n, f)
    if glide > 0.001:
        k = np.clip(t / glide, 0, 1) ** 2
        freq = f * (0.5 + 0.5 * k)
    if lfo_target == "pitch" and lfo_depth:
        freq = freq * (2.0 ** ((lfo_depth * 100.0 * np.sin(2 * math.pi * lfo_rate * t)) / 1200.0))

    def osc(wave, freq_arr, oct_, ph=0.0):
        if wave == "none":
            return np.zeros(n)
        ff = freq_arr * (2.0 ** oct_)
        if wave == "saw":
            return osc_saw(ff, n, ph)
        if wave == "supersaw":
            return supersaw(float(ff[0]), n, voices=max(3, unison), spread=spread, seed=seed)
        if wave == "square":
            return osc_square(ff, n, ph)
        if wave == "pulse":
            return osc_square(ff, n, ph, 0.42)
        if wave == "tri":
            return osc_tri(ff, n, ph)
        if wave == "sine":
            return osc_sine(ff, n, ph)
        if wave == "noise":
            return noise(n, seed)
        return osc_saw(ff, n, ph)

    a = osc(wave_a, freq, oct_a, float(g.uniform(0, 1)))
    b = osc(wave_b, freq * (2 ** (detune / 1200.0)), oct_b, float(g.uniform(0, 1)))
    y = a * (1 - mix) + b * mix
    if unison > 1 and wave_a not in ("supersaw", "noise"):
        stack = np.zeros(n)
        for i in range(unison):
            c = (i / max(1, unison - 1) - 0.5) * 2.0 * spread * 100.0 * (1 if i % 2 else -1)
            stack += osc_saw(freq * (2 ** (c / 1200.0)), n, float(g.uniform(0, 1)))
        y = y * 0.6 + stack * (0.4 / unison) * 1.9
    if sub:
        y = y + osc_sine(freq * 0.5, n) * sub
    if noise_lvl:
        y = y + noise(n, seed + 3) * noise_lvl

    # filter: envelope in octaves above the base cutoff
    vcut = cutoff * (2.0 ** (vel_to_cutoff * (vel - 1.0) * 2.0))
    fenv = adsr(n, f_a, f_d, f_s, f_r, hold=max(0.0, dur - f_a - f_d))
    fenv = np.maximum(fenv, np.exp(-np.maximum(0.0, t - dur) / max(0.02, f_r)))
    fc = vcut * (2.0 ** (env_amt * 4.0 * fenv)) * (f / 220.0) ** keytrack
    fc = np.clip(fc, 40.0, SR * 0.45)
    if lfo_target == "filter" and lfo_depth:
        fc = fc * (2.0 ** (lfo_depth * 2.0 * np.sin(2 * math.pi * lfo_rate * t)))
    if ftype == "lp24":
        y = ladder(y, fc, res, drive=drive)
    elif ftype == "lp12":
        y = filt_env(y, fc, 1.0 / max(0.4, res * 1.6), "lp")
    elif ftype == "hp":
        y = filt_env(y, fc, 1.0 / max(0.4, res * 1.6), "hp")
    elif ftype == "bp":
        y = filt_env(y, fc, 1.0 / max(0.2, res * 0.8), "bp")
    else:
        y = filt_env(y, fc, 0.9, "lp")

    aenv = adsr(n, amp_a, amp_d, amp_s, amp_r, hold=max(0.0, dur - amp_a - amp_d))
    y = y * aenv * (vel ** vel_to_amp)
    y = tanh_sat(y * 1.1, drive)
    if lfo_target == "amp" and lfo_depth:
        y = y * (1 - lfo_depth * 0.5 * (0.5 + 0.5 * np.sin(2 * math.pi * lfo_rate * t)))
    return calibrate(fade(y.astype(FLOAT), 0.003, 0.02), -5.0, 99.5)


# --------------------------------------------------------------------------
# presets
# --------------------------------------------------------------------------

PRESETS: dict[str, dict] = {
    "saw_lead": dict(wave_a="saw", wave_b="saw", unison=5, spread=0.5, cutoff=2600,
                     env_amt=0.45, res=0.9, amp_a=0.008, amp_d=0.2, amp_s=0.75, amp_r=0.2),
    "supersaw": dict(wave_a="supersaw", wave_b="saw", unison=7, spread=0.75, mix=0.35,
                     cutoff=3200, env_amt=0.35, res=0.8, amp_s=0.85, amp_r=0.35, drive=1.4),
    "wide_pad": dict(wave_a="supersaw", wave_b="supersaw", unison=9, spread=0.9, mix=0.5,
                     oct_b=-1, cutoff=1500, env_amt=0.5, res=0.5, amp_a=0.5, amp_d=0.6,
                     amp_s=0.85, amp_r=1.2, f_a=0.9, f_d=1.2, lfo_rate=0.18,
                     lfo_depth=0.25, lfo_target="filter", drive=1.15),
    "dream_pad": dict(wave_a="tri", wave_b="saw", unison=7, spread=0.8, mix=0.4, oct_b=1,
                      cutoff=1900, env_amt=0.4, res=0.6, amp_a=0.35, amp_d=0.5,
                      amp_s=0.8, amp_r=1.5, drive=1.2),
    "pluck": dict(wave_a="saw", wave_b="square", mix=0.25, unison=2, spread=0.2,
                  cutoff=3800, env_amt=0.85, res=1.2, amp_a=0.002, amp_d=0.16,
                  amp_s=0.0, amp_r=0.12, f_a=0.002, f_d=0.13, f_s=0.0, f_r=0.1, drive=1.3),
    "glass_pluck": dict(wave_a="tri", wave_b="sine", mix=0.5, oct_b=2, unison=1,
                        cutoff=5200, env_amt=0.7, res=1.0, amp_a=0.001, amp_d=0.4,
                        amp_s=0.0, amp_r=0.25, f_d=0.25, drive=1.1),
    "square_lead": dict(wave_a="square", wave_b="saw", mix=0.3, unison=3, spread=0.3,
                        cutoff=2200, env_amt=0.6, res=1.1, amp_s=0.8, amp_r=0.15),
    "sub_bass": dict(wave_a="sine", wave_b="saw", mix=0.25, oct_b=-1, sub=0.6,
                     cutoff=420, env_amt=0.35, res=0.7, ftype="lp24", amp_a=0.006,
                     amp_d=0.25, amp_s=0.8, amp_r=0.1, drive=1.8, unison=1),
    "808bass": dict(wave_a="sine", wave_b="none", sub=0.35, cutoff=900, env_amt=0.6,
                    ftype="lp24", res=0.5, amp_a=0.004, amp_d=0.55, amp_s=0.55,
                    amp_r=0.35, f_d=0.5, drive=2.0, unison=1),
    "reese": dict(wave_a="saw", wave_b="saw", mix=0.5, detune=28, unison=5, spread=0.7,
                  cutoff=700, env_amt=0.7, ftype="lp24", res=0.8, amp_s=0.9,
                  amp_r=0.12, drive=2.2, lfo_rate=0.3, lfo_depth=0.15),
    "growl_bass": dict(wave_a="saw", wave_b="square", mix=0.4, unison=4, spread=0.5,
                       cutoff=520, env_amt=0.9, ftype="lp24", res=1.1, f_d=0.2,
                       amp_s=0.85, amp_r=0.1, drive=2.6, lfo_rate=6.0,
                       lfo_depth=0.35, lfo_target="filter"),
    "moog_bass": dict(wave_a="saw", wave_b="saw", mix=0.35, detune=8, unison=3,
                      spread=0.25, cutoff=380, env_amt=0.75, ftype="lp24", res=0.55,
                      amp_a=0.005, amp_d=0.3, amp_s=0.7, amp_r=0.12, f_d=0.35, drive=1.7),
    "brass": dict(wave_a="saw", wave_b="saw", mix=0.5, unison=4, spread=0.25,
                  cutoff=900, env_amt=0.95, res=0.9, f_a=0.05, f_d=0.25, f_s=0.5,
                  amp_a=0.04, amp_d=0.2, amp_s=0.8, amp_r=0.18, drive=1.5),
    "epic_brass": dict(wave_a="saw", wave_b="square", mix=0.3, unison=7, spread=0.35,
                       oct_b=-1, cutoff=1400, env_amt=0.9, res=1.0, f_a=0.06,
                       f_d=0.4, amp_a=0.05, amp_d=0.3, amp_s=0.85, amp_r=0.3, drive=1.6),
    "choir_lead": dict(wave_a="tri", wave_b="saw", mix=0.25, unison=5, spread=0.3,
                       cutoff=1600, env_amt=0.4, res=0.7, amp_a=0.12, amp_d=0.3,
                       amp_s=0.8, amp_r=0.4, drive=1.2),
    # ---- industrial / "deliberately damaged" -----------------------------
    # noise_lvl and wave_b="noise" are how you get static: the oscillator
    # keeps the pitch, the noise keeps it from ever sounding clean
    "static_bed": dict(wave_a="saw", wave_b="noise", mix=0.42, unison=7,
                       spread=0.95, detune=26, cutoff=760, env_amt=0.75,
                       ftype="lp24", res=1.25, noise_lvl=0.5,
                       amp_a=1.1, amp_d=0.9, amp_s=0.72, amp_r=1.7,
                       f_a=1.4, f_d=1.7, f_s=0.3,
                       lfo_rate=0.17, lfo_depth=0.45, lfo_target="filter",
                       drive=1.9),
    "distorted_bass": dict(wave_a="saw", wave_b="square", mix=0.4, unison=3,
                           spread=0.28, detune=13, oct_b=-1, sub=0.5,
                           cutoff=330, env_amt=0.9, ftype="lp24", res=0.95,
                           f_a=0.004, f_d=0.18, f_s=0.22,
                           amp_a=0.004, amp_d=0.24, amp_s=0.82, amp_r=0.09,
                           drive=3.0, vel_to_cutoff=0.7),
    "glitch_lead": dict(wave_a="square", wave_b="noise", mix=0.2, unison=3,
                        spread=0.22, detune=9, cutoff=2500, env_amt=0.85,
                        ftype="lp24", res=1.35, noise_lvl=0.14,
                        amp_a=0.003, amp_d=0.14, amp_s=0.5, amp_r=0.1,
                        f_a=0.003, f_d=0.11, f_s=0.18, drive=2.8),
    "bend_guitar": dict(wave_a="saw", wave_b="saw", mix=0.5, unison=5,
                        spread=0.28, detune=7, cutoff=1150, env_amt=0.9,
                        res=1.05, amp_a=0.012, amp_d=0.55, amp_s=0.68,
                        amp_r=0.4, f_a=0.02, f_d=0.7, drive=2.5),
    "acid": dict(wave_a="saw", wave_b="square", mix=0.35, unison=1, detune=0,
                 cutoff=170, env_amt=1.0, ftype="lp24", res=0.88,
                 f_a=0.002, f_d=0.11, f_s=0.04, f_r=0.06,
                 amp_a=0.002, amp_d=0.16, amp_s=0.0, amp_r=0.04,
                 drive=2.2, vel_to_cutoff=0.9),
    # ---- machine bass ---------------------------------------------------
    # Nothing about this one drifts: no vibrato, no glide, one short envelope
    # repeated until the bar ends.  The darkness is a resonant ladder that
    # snaps shut over every step, and a sub an octave down that the filter
    # never quite reaches.  The click is the sequencer itself.
    # DOOM 2016 Mick Gordon / UAC Report style brutal industrial bass:
    # Heavy multi-wave distortion array with aggressive ladder filter bite,
    # saturated sub-weight, tight transient clamp, and industrial drive.
    "grid_bass": dict(wave_a="saw", wave_b="square", mix=0.45, unison=3,
                      spread=0.20, detune=14, oct_b=0, sub=0.40,
                      noise_lvl=0.075,
                      cutoff=240.0, env_amt=0.60, ftype="lp24", res=0.94,
                      f_a=0.003, f_d=0.085, f_s=0.20, f_r=0.05,
                      amp_a=0.002, amp_d=0.045, amp_s=0.62, amp_r=0.030,
                      drive=3.6, vel_to_cutoff=0.6, vel_to_amp=0.8),
    "hoover": dict(wave_a="supersaw", wave_b="saw", mix=0.4, unison=9, spread=0.85,
                   cutoff=520, env_amt=0.85, ftype="lp24", res=0.55,
                   f_a=0.18, f_d=0.6, f_s=0.5,
                   amp_a=0.06, amp_d=0.3, amp_s=0.9, amp_r=0.35, drive=1.9),
    "stab": dict(wave_a="saw", wave_b="saw", mix=0.5, unison=5, spread=0.55,
                 cutoff=2100, env_amt=0.75, res=0.8,
                 amp_a=0.002, amp_d=0.11, amp_s=0.0, amp_r=0.07,
                 f_a=0.002, f_d=0.13, f_s=0.0, drive=1.7),
    "organ_stab": dict(wave_a="square", wave_b="saw", mix=0.45, unison=3, spread=0.2,
                       oct_b=1, cutoff=2600, env_amt=0.5, res=0.7,
                       amp_a=0.003, amp_d=0.13, amp_s=0.0, amp_r=0.08, drive=1.5),
    "organ_pad": dict(wave_a="saw", wave_b="saw", mix=0.5, oct_b=1, unison=3,
                      spread=0.15, cutoff=2200, env_amt=0.2, amp_a=0.02, amp_s=0.9,
                      amp_r=0.25, drive=1.3),
}


def note(midi: float, dur: float = 1.0, preset: str = "saw_lead", vel: float = 1.0,
         seed: int = 1, **over) -> np.ndarray:
    p = dict(PRESETS.get(preset, PRESETS["saw_lead"]))
    p.update(over)
    return voice(midi, dur, vel, seed=seed, **p)


def render(events, preset: str = "saw_lead", length: float | None = None,
           tail: float = 1.5, seed: int = 1, **over) -> np.ndarray:
    """events: (time_seconds, midi, duration_seconds, velocity) -> mono mix"""
    end = max([e[0] + e[2] for e in events], default=0.0)
    n = sec((length or end) + tail)
    out = np.zeros(n, dtype=np.float64)
    for i, (t, m, d, v) in enumerate(events):
        y = note(m, d, preset, v, seed=seed + i, **over)
        j = sec(t)
        k = min(y.shape[0], n - j)
        if k > 0:
            out[j:j + k] += y[:k]
    return out.astype(FLOAT)
