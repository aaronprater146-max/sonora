#!/usr/bin/env python3
"""Describe an instrument well enough to rebuild it.

Point it at a stem (or at a single hit) and it measures the thing and prints
a Sonora preset you can paste into the synth or the drum kit.

    python3 tools/mimic.py /tmp/split/kick.flac      -> a drum recipe
    python3 tools/mimic.py /tmp/split/tonal.flac     -> a synth recipe

It measures, it does not guess:
  drums   fundamental, pitch sweep, decay, click, noise ratio, brightness
  tones   f0, harmonic series, wave shape, filter cutoff, envelope, drive
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

import sonora.core as C
from sonora import ears

HOP, N_FFT = 512, 2048


# ----------------------------------------------------------------- helpers

def _env(y: np.ndarray, sr: int, lp: float = 60.0):
    m = np.asarray(C.mono(y), dtype=np.float32)
    return np.abs(C.filt_env(m, lp, 0.7, "lp")).astype(np.float64)


def _spectra(y: np.ndarray, sr: int):
    m = np.asarray(C.mono(y), dtype=np.float32)
    S = np.abs(np.fft.rfft(m[:N_FFT] * np.hanning(N_FFT))) if m.size >= N_FFT else None
    return S


def periodic(x: np.ndarray, sr: int, lo: float = 60.0, hi: float = 1000.0) -> float:
    """How pitched is this?  0 = pure noise, 1 = a steady tone.

    Drums vs tones is the first decision the tool has to make, and "it has a
    lot of onsets" is not an answer -- a bassline has onsets too.  Periodic
    energy is.
    """
    m = np.asarray(C.mono(x), dtype=np.float64)
    n = 8192
    if m.size < n * 2:
        return 0.0
    e = np.abs(C.filt_env(m.astype(np.float32), 30.0, 0.7, "lp")).astype(np.float64)
    best, bi = -1.0, 0
    for i in range(0, m.size - n, n // 2):          # a handful of loud windows
        v = float(e[i:i + n].mean())
        if v > best:
            best, bi = v, i
    w = m[bi:bi + n] * np.hanning(n)
    size = 1 << int(np.ceil(np.log2(2 * w.size)))
    F = np.fft.rfft(w, size)
    ac = np.fft.irfft(F * np.conj(F))[:w.size]
    if ac[0] <= 0:
        return 0.0
    ac /= ac[0]
    a, b = max(2, int(sr / hi)), min(ac.size - 1, int(sr / lo))
    return float(ac[a:b].max()) if b > a else 0.0


def _peak_hz(S, sr, lo=20.0, hi=8000.0):
    f = np.fft.rfftfreq(N_FFT, 1 / sr)
    sel = (f >= lo) & (f <= hi)
    return float(f[sel][np.argmax(S[sel])])


def hz_to_midi(f: float) -> float:
    return 69.0 + 12.0 * math.log2(max(f, 1e-6) / 440.0)


# ------------------------------------------------------------------- drums

def hits(y: np.ndarray, sr: int, n: int = 12, win: float = 0.6):
    """isolate the strongest individual hits of a percussion stem"""
    t, _, _ = ears.onset_times(y)
    e = _env(y, sr, 80.0)
    m = np.asarray(C.mono(y), dtype=np.float32)
    out = []
    for t0 in t[:400]:
        i = int(t0 * sr)
        j = min(m.size, i + int(win * sr))
        if j - i < 512:
            continue
        seg = m[i:j]
        out.append((float(e[i:j].max()), seg))
    out.sort(key=lambda s: -s[0])
    return [s[1] for s in out[:n]]


def drum_recipe(y: np.ndarray, sr: int) -> dict:
    """measure a drum: pitch, how far it sweeps, how long it rings, how much
    of it is noise, and how much click is in the attack"""
    hs = hits(y, sr)
    if not hs:
        return dict(kind="percussion", note="no clear hits found")

    win = 2048
    decays, peaks, sweeps, flats, clicks, subs = [], [], [], [], [], []
    for seg in hs:
        n = seg.size
        e = np.abs(C.filt_env(seg, 80.0, 0.7, "lp")).astype(np.float64)
        pk = float(e.max())
        if pk < 1e-6:
            continue
        # ring: peak down to -25 dB, over the whole hit, not just one FFT frame
        k = int(np.argmax(e))
        tail = e[k:]
        below = np.flatnonzero(tail < pk * 0.056)
        decays.append(float((below[0] if below.size else tail.size) / sr))
        # pitch at the attack, and again 60 ms in -- the sweep is the sound
        w = min(win, n)
        S0 = np.abs(np.fft.rfft(seg[:w] * np.hanning(w)))
        f0 = _peak_hz(S0, sr, 25.0, 12000.0)
        peaks.append(f0)
        i1 = int(0.06 * sr)
        if i1 + win <= n:
            S1 = np.abs(np.fft.rfft(seg[i1:i1 + win] * np.hanning(win)))
            f1 = _peak_hz(S1, sr, 25.0, 12000.0)
            sweeps.append(1200.0 * math.log2(max(f1, 1.0) / max(f0, 1.0)))
        # click: how much of the first 5 ms lives above 3 kHz
        atk = seg[:max(128, int(0.005 * sr))]
        if np.abs(atk).max() > 1e-9:
            hi = np.abs(np.fft.rfft(atk * np.hanning(atk.size)))
            fh = np.fft.rfftfreq(atk.size, 1 / sr)
            tot = float((hi ** 2).sum()) + 1e-12
            clicks.append(float((hi[fh > 3000] ** 2).sum() / tot))
        # noise vs tone, and how much of it is down low
        lg = np.log(np.abs(np.fft.rfft(seg[:w] * np.hanning(w))) ** 2 + 1e-12)
        flats.append(float(np.exp(lg.mean()) / (np.exp(lg).mean() + 1e-12)))
        fl = np.fft.rfftfreq(w, 1 / sr)
        tot = float((S0 ** 2).sum()) + 1e-12
        subs.append(float((S0[fl < 120] ** 2).sum() / tot))

    f0 = float(np.median(peaks)) if peaks else 100.0
    decay = float(np.median(decays)) if decays else 0.15
    flat = float(np.median(flats)) if flats else 0.0
    click = float(np.median(clicks)) if clicks else 0.0
    sweep = float(np.median(sweeps)) if sweeps else 0.0
    sub = float(np.median(subs)) if subs else 0.5
    kind = ("kick" if f0 < 140 and sub > 0.2 else
            "hat" if f0 > 2500 or flat > 0.25 else
            "snare" if flat > 0.02 else "tom")
    return dict(
        kind=kind,
        measured=dict(hits=len(hs), fundamental_hz=round(f0, 1),
                      pitch_sweep_cents=round(sweep, 0),
                      decay_s=round(decay, 3),
                      click_ratio=round(click, 3),
                      noise_flatness=round(flat, 4),
                      sub_energy=round(sub, 3)),
        kit_entry={kind: dict(tune=int(round(hz_to_midi(f0))),
                              decay=round(min(1.2, max(0.05, decay * 1.6)), 3),
                              punch=round(min(1.0, 0.35 + click * 3.0), 2),
                              click=round(min(1.0, click * 2.2), 2),
                              sub=round(min(1.0, sub * 1.2), 2))})


# ------------------------------------------------------------------- tones

WAVE_TEMPLATES = {          # relative level of harmonics 1..8
    "saw": [1, .5, .333, .25, .2, .167, .143, .125],
    "square": [1, 0, .333, 0, .2, 0, .143, 0],
    "tri": [1, 0, .111, 0, .04, 0, .02, 0],
    "sine": [1, .02, .01, .005, .003, .002, .001, .001],
}


def tone_recipe(y: np.ndarray, sr: int) -> dict:
    import librosa
    m = np.asarray(C.mono(y), dtype=np.float32)
    f0, voiced, _ = librosa.pyin(m, fmin=50.0, fmax=1400.0, sr=sr,
                                 hop_length=HOP, frame_length=N_FFT)
    f0 = np.asarray(f0, dtype=np.float64)
    v = np.asarray(voiced, dtype=bool) & np.isfinite(f0) & (f0 > 0)
    if v.sum() < 8:
        return dict(kind="tone", note="no stable pitch -- try the texture stem")
    pitch = float(np.median(f0[v]))

    # harmonic amplitudes: read the spectrum at each multiple of f0
    S = np.abs(librosa.stft(m, n_fft=N_FFT, hop_length=HOP)).mean(axis=1)
    freqs = librosa.fft_frequencies(sr=sr, n_fft=N_FFT)
    amps = []
    for h in range(1, 9):
        target = pitch * h
        if target >= sr / 2:
            amps.append(0.0)
            continue
        i = int(np.argmin(np.abs(freqs - target)))
        amps.append(float(S[max(0, i - 1):i + 2].max()))
    a = np.array(amps)
    if a.max() <= 0:
        return dict(kind="tone", note="silent")
    a = a / a.max()

    # which wave is it?  correlate the log series against each template
    best, best_r = "saw", -9
    for name, tpl in WAVE_TEMPLATES.items():
        t = np.array(tpl)
        r = float(np.corrcoef(np.log10(a + 1e-3), np.log10(t + 1e-3))[0, 1])
        if r > best_r:
            best, best_r = name, r

    # cutoff: where the harmonics fall behind the ideal (unfiltered) series
    ideal = np.array(WAVE_TEMPLATES[best])
    ratio = (a + 1e-6) / (ideal + 1e-6)
    roll = [h for h in range(8) if ratio[h] < 0.35]
    cutoff = float(pitch * (roll[0] + 1)) if roll else 8000.0
    cutoff = float(np.clip(cutoff, 120.0, 12000.0))

    # envelope: attack, decay, sustain, release from the amplitude envelope
    e = _env(y, sr, 40.0)
    e = e / (e.max() + 1e-9)
    strike = int(np.argmax(e[:sr]))
    attack = float(max(0.002, strike / sr))
    after = e[strike:]
    sus = float(np.percentile(after, 60)) if after.size else 0.0
    below = np.flatnonzero(after < 0.1)
    decay = float((below[0] if below.size else after.size) / sr)

    # drive: even-order harmonics and inharmonic energy mean distortion
    even = float(np.mean(a[1::2]))
    drive = float(np.clip(1.0 + even * 6.0, 1.0, 4.0))
    tx = ears.texture(y, sr)

    return dict(
        kind="tone",
        measured=dict(pitch_hz=round(pitch, 1),
                      harmonics=[round(float(v), 3) for v in a],
                      wave_match=f"{best} (r={best_r:.2f})",
                      cutoff_hz=round(cutoff), attack_s=round(attack, 3),
                      decay_s=round(decay, 3), sustain=round(sus, 2),
                      flatness=round(tx["flatness"], 3)),
        preset=dict(wave_a=best, wave_b="saw" if best != "saw" else "square",
                    mix=0.35, unison=3, spread=0.3, detune=8,
                    cutoff=int(round(cutoff)), env_amt=0.7, res=0.9,
                    amp_a=round(attack, 3), amp_d=round(max(0.05, decay * 0.6), 3),
                    amp_s=round(float(np.clip(sus, 0.0, 0.95)), 2),
                    amp_r=round(max(0.08, decay * 0.4), 3),
                    drive=round(drive, 2)))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("audio")
    ap.add_argument("--drum", action="store_true", help="treat it as percussion")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    y, sr = ears.load(args.audio)
    tx = ears.texture(y, sr)
    name = os.path.basename(args.audio).lower()
    pitch = periodic(y, sr)
    perc = (args.drum
            or any(k in name for k in ("kick", "snare", "hat", "perc", "drum",
                                       "tom", "clap", "cymbal"))
            or (pitch < 0.35 and tx["flatness"] > 0.02))
    r = drum_recipe(y, sr) if perc else tone_recipe(y, sr)
    r["file"] = args.audio
    r["analysed_as"] = "percussion" if perc else "tone"
    r["periodic"] = round(pitch, 3)

    if args.json:
        print(json.dumps(r, indent=2))
        return 0
    print(f"\n  {os.path.basename(args.audio)}  ->  {r['analysed_as']}")
    if perc and r["measured"].get("decay_s", 1) < 0.02:
        print("  note: under 20 ms of ring -- harmonic/percussive separation\n"
              "        puts a drum's body in the harmonic stem and only its\n"
              "        transient here.  Measure the original mix for the decay.")
    print("  " + "-" * 66)
    for k, v in r["measured"].items():
        print(f"    {k:20s} {v}")
    print("\n  paste this into sonora:")
    body = r.get("kit_entry") or r.get("preset")
    print("    " + json.dumps(body, indent=4).replace("\n", "\n    "))
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
