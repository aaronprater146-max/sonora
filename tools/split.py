#!/usr/bin/env python3
"""Pull a song apart so you can hear each instrument on its own.

    python3 tools/split.py song.mp3 --out /tmp/stems

No model downloads and no GPU: harmonic/percussive separation (median
filtering on the spectrogram) followed by a band split of the percussive
part and a tonal/noise split of the harmonic part.  You get

    kick.flac snare.flac hats.flac       <- the drum kit, by band
    tonal.flac                           <- pitched material: bass, chords, lead
    texture.flac                         <- unpitched material: static, noise, air
    percussive.flac  harmonic.flac       <- the two-way split, if you want it

Point this at a reference track and you can finally answer "what is that
sound made of?" instead of guessing.
"""
from __future__ import annotations

import argparse
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

import sonora.core as C
from sonora import ears


def hpss(x: np.ndarray, margin: float = 3.0, hop: int = 1024,
         block: float = 40.0):
    """harmonic / percussive separation by median filtering the spectrogram.

    The full-length transform is hundreds of megabytes for a four-minute song,
    so this runs in overlapping blocks and crossfades them.  A bigger hop
    halves the memory and costs almost nothing in separation quality.
    """
    import librosa
    n = x.shape[0]
    step = int(block * 48000)
    adv = max(1, step - int(5.0 * 48000))            # 5 s overlap
    H = np.zeros_like(x)
    P = np.zeros_like(x)
    if n <= step:
        for ch in range(2):
            h, pp = librosa.effects.hpss(np.asarray(x[:, ch], dtype=np.float32),
                                         margin=(margin, margin * 1.6),
                                         hop_length=hop)
            H[:h.shape[0], ch] = h
            P[:pp.shape[0], ch] = pp
        return H, P
    fade = int(2.0 * 48000)
    for i in range(0, n, adv):
        j = min(n, i + step)
        h = np.zeros((j - i, 2), dtype=np.float32)
        pp = np.zeros((j - i, 2), dtype=np.float32)
        for ch in range(2):
            a, b = librosa.effects.hpss(np.asarray(x[i:j, ch], dtype=np.float32),
                                        margin=(margin, margin * 1.6),
                                        hop_length=hop)
            h[:a.shape[0], ch] = a
            pp[:b.shape[0], ch] = b
        w = np.ones(j - i, dtype=np.float32)[:, None]
        if i > 0:                                     # crossfade into the tail
            k = min(fade, j - i)
            ramp = np.linspace(0.0, 1.0, k, dtype=np.float32)[:, None]
            w[:k] *= ramp
        if j < n:
            k = min(fade, j - i)
            ramp = np.linspace(1.0, 0.0, k, dtype=np.float32)[:, None]
            w[-k:] *= ramp
        H[i:j] += h * w
        P[i:j] += pp * w
        del h, pp, w
    return H, P


def mask_split(x: np.ndarray, lo: float, hi: float):
    """keep only the band [lo, hi]"""
    y = C.highpass(np.asarray(x, dtype=np.float32), lo, 0.707)
    if hi < 20000:
        y = C.lowpass(y, hi, 0.707)
    return y


def _audible(y: np.ndarray, thresh: float = 1e-5) -> bool:
    """peak, without materialising abs() of a four-minute stem"""
    for i in range(0, y.shape[0], 1 << 20):
        if float(np.abs(y[i:i + (1 << 20)]).max()) > thresh:
            return True
    return False


def _level_db(y: np.ndarray) -> float:
    """rms in dB, chunked so float64 never holds the whole stem"""
    acc, n = 0.0, 0
    for i in range(0, y.shape[0], 1 << 20):
        b = np.asarray(y[i:i + (1 << 20)], dtype=np.float32)
        acc += float((b * b).sum())
        n += b.size
    return 20 * np.log10(math.sqrt(acc / max(1, n)) + 1e-12) if n else -120.0


def flatness_mask(x: np.ndarray, n_fft: int = 2048):
    """soft mask: 1 where the spectrum is noise-like, 0 where it is tonal"""
    import librosa
    m = np.asarray(C.mono(x), dtype=np.float32)
    S = np.abs(librosa.stft(m, n_fft=n_fft, hop_length=512)) + 1e-9
    flat = librosa.feature.spectral_flatness(S=S)          # 1 x frames
    thr = float(np.percentile(flat, 60))
    # soft knee around the threshold
    w = np.clip((flat - thr) / max(1e-6, 0.25 * thr) * 0.5 + 0.5, 0.0, 1.0)
    return np.repeat(w.astype(np.float32), n_fft // 2 + 1, axis=0)[:, :S.shape[1]]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("audio")
    ap.add_argument("--out", default="/tmp/split")
    ap.add_argument("--margin", type=float, default=1.0,
                    help="higher = more aggressive separation")
    ap.add_argument("--window", type=float, default=0.0,
                    help="only split the first N seconds (0 = everything)")
    ap.add_argument("--report", action="store_true",
                    help="run the full Ears analysis on every stem (slow)")
    args = ap.parse_args()

    x, sr = ears.load(args.audio)
    if args.window > 1:
        x = x[:int(args.window * sr)]
    os.makedirs(args.out, exist_ok=True)
    print(f"splitting {args.audio}  ({x.shape[0]/sr:.1f}s)")

    harm, perc = hpss(x, args.margin)
    del x

    def parts():
        yield "kick", mask_split(perc, 20.0, 150.0)
        yield "snare", mask_split(perc, 150.0, 900.0)
        yield "hats", mask_split(perc, 3000.0, 20000.0)
        yield "percussive", perc
        yield "harmonic", harm
        try:
            import librosa
            m = np.asarray(C.mono(harm), dtype=np.float32)
            S = librosa.stft(m, n_fft=2048, hop_length=512)
            w = flatness_mask(harm)
            noise = librosa.istft(S * w, hop_length=512, length=m.size)
            noise = np.stack([noise, noise], axis=1).astype(np.float32)
            del S, w
            yield "texture", noise
            yield "tonal", (harm - noise).astype(np.float32)
            del noise
        except Exception as exc:
            print(f"  (tonal/texture split skipped: {exc})")

    print(f"\n{'stem':12s} {'level':>7s}   what it is")
    print("-" * 74)
    wrote = 0
    for name, y in parts():
        if not _audible(y):
            del y
            continue
        path = os.path.join(args.out, f"{name}.flac")
        C.write(path, y)
        rms = _level_db(y)
        line = f"{name:12s} {rms:6.1f} dB"
        if args.report:                 # pyin on every stem costs a lot of RAM
            a = ears.analyze(path, window=40.0)
            line += (f"   onsets {a['rhythm']['onsets_per_s']:.1f}/s  "
                     f"flat {a['texture']['flatness']:.3f}  "
                     f"centroid {a['texture']['centroid_hz']:.0f} Hz")
        else:
            # a 30 s sample reads the same as the whole stem and costs a
            # fraction of the memory
            seg = y[len(y) // 3: len(y) // 3 + 30 * sr]
            t, _, _ = ears.onset_times(seg)
            line += f"   onsets {t.size / max(1e-6, seg.shape[0]/sr):.1f}/s"
            del seg
        print(line, flush=True)
        wrote += 1
        del y

    print(f"\nwrote {wrote} stems to {args.out}/")
    print("next: python3 tools/mimic.py " + os.path.join(args.out, "kick.flac"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
