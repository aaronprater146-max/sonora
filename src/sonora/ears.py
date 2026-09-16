"""Ears -- analysis, stem separation and instrument mimicry.

Three jobs, in the order a producer actually does them:

  analyze   understand a song: is it a song or a drone?  where is the melody?
            what is the tempo, the key, the structure, the loudness?
  split     pull it apart so you can *hear* each instrument on its own
  mimic     describe each instrument well enough to rebuild it in Sonora

Everything here is plain numpy/scipy/librosa -- no model downloads, no GPU,
no weights.  It runs on a 2-core box in seconds.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field

import numpy as np

try:
    import librosa
    HAS_LIBROSA = True
except ImportError:            # pragma: no cover - analysis is optional
    HAS_LIBROSA = False

import soundfile as sf

from .core import SR, FLOAT, mono, ensure2, highpass, lowpass, write

HOP = 512
N_FFT = 2048


# --------------------------------------------------------------------------
# io
# --------------------------------------------------------------------------

def load(path: str, sr: int = SR):
    """read any audio file as (stereo float32, samplerate)"""
    y, native = sf.read(path, dtype="float32", always_2d=True)
    y = ensure2(y)
    if native != sr:
        y = librosa.resample(y.T, orig_sr=native, target_sr=sr).T if HAS_LIBROSA \
            else _resample_naive(y, native, sr)
        y = np.ascontiguousarray(y, dtype=FLOAT)
    return y.astype(FLOAT), sr


def _resample_naive(y, src, dst):
    n = int(round(y.shape[0] * dst / src))
    idx = np.clip((np.arange(n) * src / dst).astype(np.int64), 0, y.shape[0] - 1)
    return y[idx]


# --------------------------------------------------------------------------
# loudness
# --------------------------------------------------------------------------

def _k_weight(x: np.ndarray, sr: int) -> np.ndarray:
    """ITU-R BS.1770 K-weighting (chunked, so 5-minute songs do not balloon)"""
    from scipy.signal import sosfilt
    # stage 1: high shelf ~ +4 dB above 1 kHz, stage 2: RLB high-pass
    from .core import high_shelf as _hs
    out = np.empty_like(x)
    for ch in range(x.shape[1]):
        c = np.asarray(x[:, ch], dtype=np.float32)
        c = _hs(c, 1681.0, 4.0, 0.707)
        c = highpass(c, 38.0, 0.5)
        out[:, ch] = c
    return out


def loudness(x: np.ndarray, sr: int = SR) -> dict:
    """integrated LUFS, short-term peak, true peak, crest factor"""
    y = np.asarray(ensure2(x), dtype=np.float64)
    if y.shape[0] < sr // 10:
        return dict(lufs=-70.0, lufs_short_max=-70.0, peak_db=-70.0,
                    rms_db=-70.0, crest_db=0.0)
    y = _k_weight(y, sr)
    w = int(0.4 * sr)
    n = y.shape[0] // w
    if n < 1:
        return dict(lufs=-70.0, lufs_short_max=-70.0, peak_db=-70.0,
                    rms_db=-70.0, crest_db=0.0)
    ms = np.array([(y[i * w:(i + 1) * w] ** 2).mean() for i in range(n)])
    l = -0.691 + 10 * np.log10(np.maximum(ms, 1e-12))
    # two-stage gate, as specified
    def _mean_gated(v, rel):
        t = v.max() + rel
        keep = v[v > t]
        return keep.mean() if keep.size else v.mean()
    loud = _mean_gated(_mean_gated(l, -10.0), -20.0)
    # short-term 3 s window
    k = max(1, int(3.0 / 0.4))
    st = np.array([l[i:i + k].mean() for i in range(0, max(1, n - k + 1))])
    peak = float(np.abs(np.asarray(x, dtype=np.float64)).max())
    rms = float(np.sqrt((np.asarray(x, dtype=np.float64) ** 2).mean()))
    return dict(lufs=float(loud),
                lufs_short_max=float(st.max()) if st.size else loud,
                peak_db=float(20 * math.log10(max(peak, 1e-9))),
                rms_db=float(20 * math.log10(max(rms, 1e-9))),
                crest_db=float(20 * math.log10(max(peak, 1e-9) / max(rms, 1e-9))))


# --------------------------------------------------------------------------
# rhythm -- "is this a song or one long note?"
# --------------------------------------------------------------------------

def onset_times(x: np.ndarray, sr: int = SR, **kw):
    from .feel import onsets
    return onsets(np.asarray(mono(x), dtype=FLOAT), **kw)


def tempo(x: np.ndarray, sr: int = SR, lo: float = 80.0, hi: float = 160.0) -> float:
    """tempo from the onset envelope, with the octave ambiguity resolved.

    Half and double tempo are always strong candidates (a kick on 1 and 3 is
    periodic at two beats as well as one), so a plain argmax lands on the wrong
    octave about half the time.  We score every local peak of the onset
    autocorrelation with a harmonic sum, fold it into a musical range, and
    break near-ties in favour of the faster grid -- the finer grid carries more
    information, and a beat that lands on every kick is a better description of
    the music than one that lands on every other kick.
    """
    m = np.asarray(mono(x), dtype=np.float64)
    hop, n_fft = HOP, N_FFT
    n = max(64, (m.size - n_fft) // hop)
    win = np.hanning(n_fft)
    S = np.empty((n, n_fft // 2 + 1), dtype=np.float32)
    for i in range(n):
        S[i] = np.abs(np.fft.rfft(m[i * hop:i * hop + n_fft] * win))
    flux = np.diff(np.log1p(S * 20.0), axis=0).clip(min=0).sum(axis=1).astype(np.float64)
    flux -= flux.mean()
    if flux.std() < 1e-9:
        return 0.0
    size = 1 << int(np.ceil(np.log2(2 * flux.size)))
    F = np.fft.rfft(flux, size)
    ac = np.fft.irfft(F * np.conj(F))[:flux.size]
    if ac[0] <= 0:
        return 0.0
    ac /= ac[0]
    fps = sr / hop
    lmin, lmax = int(fps * 60.0 / (hi * 4)), min(ac.size - 1, int(fps * 60.0 / (lo * 0.5)))
    lmin = max(2, lmin)
    if lmax - lmin < 8:
        return 0.0

    def at(lag):                       # autocorrelation at a fractional lag
        i = int(round(lag))
        return float(ac[i]) if 0 <= i < ac.size else 0.0

    peaks = [i for i in range(lmin, lmax - 1)
             if ac[i] > ac[i - 1] and ac[i] >= ac[i + 1] and ac[i] > 0.02]
    if not peaks:
        return 0.0
    cands = []
    for lag in peaks:
        bpm = 60.0 * fps / lag
        while bpm < lo:
            bpm *= 2.0
        while bpm > hi:
            bpm /= 2.0
        blag = 60.0 * fps / bpm          # lag of the folded beat
        score = sum(w * at(blag * k) for k, w in ((1, 1.0), (2, 0.6), (3, 0.35), (4, 0.2)))
        prior = math.exp(-0.5 * (math.log2(bpm / 118.0) / 0.95) ** 2)
        cands.append((score * prior, bpm, ac[lag]))
    cands.sort(key=lambda c: -c[0])
    best_score = cands[0][0]
    # near-ties go to the faster grid
    for sc, bpm, raw in cands:
        if sc >= best_score * 0.92:
            return float(bpm)
    return float(cands[0][1])


def rhythm(x: np.ndarray, sr: int = SR) -> dict:
    m = np.asarray(mono(x), dtype=np.float32)
    dur = m.size / sr
    t, _, _ = onset_times(m)
    bpm = tempo(m) if dur > 4 else 0.0
    # how much of the movement is low end (kick/bass) vs top (hats/perc)
    from .core import lowpass as _lp, highpass as _hp
    lo_t, _, _ = onset_times(_lp(m, 140.0, 0.707), threshold=1.2)
    hi_t, _, _ = onset_times(_hp(m, 4000.0, 0.707), threshold=1.2)
    return dict(duration=float(dur), tempo=bpm, onsets=float(t.size),
                onsets_per_s=float(t.size / max(1e-6, dur)),
                low_onsets_per_s=float(lo_t.size / max(1e-6, dur)),
                high_onsets_per_s=float(hi_t.size / max(1e-6, dur)))


# --------------------------------------------------------------------------
# texture -- noise vs tone, brightness, balance
# --------------------------------------------------------------------------

def texture(x: np.ndarray, sr: int = SR) -> dict:
    m = np.asarray(mono(x), dtype=np.float32)
    S = np.abs(librosa.stft(m, n_fft=N_FFT, hop_length=HOP)) + 1e-9
    flat = librosa.feature.spectral_flatness(S=S).mean() if HAS_LIBROSA else 0.0
    cent = float(librosa.feature.spectral_centroid(S=S, sr=sr).mean())
    roll = float(librosa.feature.spectral_rolloff(S=S, sr=sr, roll_percent=0.85).mean())
    freqs = librosa.fft_frequencies(sr=sr, n_fft=N_FFT)
    p = (S ** 2).mean(axis=1)
    tot = p.sum() + 1e-20
    # power per Hz, not per band: otherwise a 14 kHz band always beats a
    # 190 Hz one and every mix looks brighter on top than it really is
    bands = {}
    for lo, hi, name in ((20, 60, "sub"), (60, 250, "low"), (250, 500, "lowmid"),
                         (500, 2000, "mid"), (2000, 6000, "himid"),
                         (6000, 20000, "high")):
        sel = (freqs >= lo) & (freqs < hi)
        bw = float(sel.sum()) * (sr / N_FFT)
        bands[name] = float(10 * math.log10(p[sel].sum() / max(bw, 1.0) + 1e-20))
    bands = {k: (v - max(bands.values())) for k, v in bands.items()}
    return dict(flatness=float(flat), centroid_hz=cent, rolloff_hz=roll,
                bands_db=bands)


# --------------------------------------------------------------------------
# melody -- the thing our own output was missing
# --------------------------------------------------------------------------

def melody(x: np.ndarray, sr: int = SR) -> dict:
    """A melody is a pitch line that MOVES and is PROMINENT.

    A drone scores high on sustain and near zero on motion; a real tune
    changes pitch a few times a second and sits above the accompaniment.
    """
    m = np.asarray(mono(x), dtype=np.float32)
    dur = m.size / sr
    if not HAS_LIBROSA:
        return dict(notes_per_s=0.0, motion=0.0, sustain=1.0, range_semis=0.0,
                    voiced=0.0, prominence=0.0, median_hold_s=0.0)
    # the melody lives where a voice or lead would: 200 Hz - 2 kHz
    from .core import highpass as _hp, lowpass as _lp
    band = _lp(_hp(m, 180.0, 0.707), 2400.0, 0.707)
    f0, voiced, _ = librosa.pyin(band, fmin=80.0, fmax=1200.0, sr=sr,
                                 hop_length=HOP, frame_length=N_FFT)
    f0 = np.asarray(f0, dtype=np.float64)
    voiced = np.asarray(voiced, dtype=bool) & np.isfinite(f0) & (f0 > 0)
    vfrac = float(voiced.mean()) if voiced.size else 0.0
    if voiced.sum() < 8:
        return dict(notes_per_s=0.0, motion=0.0, sustain=1.0, range_semis=0.0,
                    voiced=vfrac, prominence=0.0, median_hold_s=0.0)
    semi = 12 * np.log2(f0[voiced] / 440.0) + 69.0
    # quantise to the semitone grid, then count how often it actually changes
    q = np.round(semi)
    changes = int((np.diff(q) != 0).sum())
    frame_dur = HOP / sr
    voiced_time = voiced.sum() * frame_dur
    notes_per_s = changes / max(1e-6, voiced_time)
    # mean run length of a held pitch, in seconds
    runs = np.diff(np.flatnonzero(np.diff(q) != 0))
    hold = float(np.median(runs) * frame_dur) if runs.size else voiced_time
    # prominence: energy in the melodic band vs everything else
    S = np.abs(librosa.stft(m, n_fft=N_FFT, hop_length=HOP)) ** 2
    freqs = librosa.fft_frequencies(sr=sr, n_fft=N_FFT)
    sel = (freqs >= 180) & (freqs < 2400)
    prom = float(S[sel].mean() / (S.mean() + 1e-12))
    return dict(notes_per_s=float(notes_per_s),
                motion=float(min(1.0, notes_per_s / 4.0)),
                sustain=float(min(1.0, hold / 2.0)),
                range_semis=float(np.percentile(semi, 95) - np.percentile(semi, 5)),
                voiced=vfrac, prominence=float(prom),
                median_hold_s=hold)


# --------------------------------------------------------------------------
# harmony
# --------------------------------------------------------------------------

_KEYS = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
_MAJ = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
_MIN = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])


def harmony(x: np.ndarray, sr: int = SR) -> dict:
    m = np.asarray(mono(x), dtype=np.float32)
    if not HAS_LIBROSA:
        return dict(key="?", mode="?", chords_per_min=0.0)
    C = librosa.feature.chroma_cqt(y=m, sr=sr, hop_length=HOP)
    prof = C.mean(axis=1)
    best, bm, bmode = -9, "?", "?"
    for i in range(12):
        for prof_t, name in ((_MAJ, "major"), (_MIN, "minor")):
            r = float(np.corrcoef(prof, np.roll(prof_t, i))[0, 1])
            if r > best:
                best, bm, bmode = r, _KEYS[i], name
    # chord changes: compare half-second blocks.  Adjacent STFT frames are
    # 11 ms apart and always look alike, which reads as zero harmony movement.
    step = max(1, int(0.5 * sr / HOP))
    B = C[:, ::step]
    B = B / (np.linalg.norm(B, axis=0, keepdims=True) + 1e-9)
    change = np.clip(1.0 - (B[:, :-1] * B[:, 1:]).sum(axis=0), 0, 2)
    dur = m.size / sr
    cpm = float((change > 0.12).sum() / max(1e-6, dur) * 60.0)
    return dict(key=bm, mode=bmode, chords_per_min=cpm)


# --------------------------------------------------------------------------
# structure
# --------------------------------------------------------------------------

def structure(x: np.ndarray, sr: int = SR) -> dict:
    """self-similarity -> section boundaries, then label them by energy"""
    m = np.asarray(mono(x), dtype=np.float32)
    dur = m.size / sr
    rms = np.sqrt(np.asarray(
        [ (m[i:i + sr // 2].astype(np.float64) ** 2).mean()
          for i in range(0, m.size - sr // 2, sr // 2)]))
    if rms.size < 8:
        return dict(sections=[], contrast_db=0.0, repeats=0.0)
    db = 20 * np.log10(rms + 1e-9)
    contrast = float(db.max() - db.min())
    # sections from the loudness contour, coarse
    sm = np.convolve(db, np.ones(9) / 9, mode="same") if db.size > 9 else db
    lab = np.digitize(sm, [db.min() + 0.33 * (db.max() - db.min()),
                           db.min() + 0.66 * (db.max() - db.min())])
    names = ["quiet", "mid", "loud"]
    secs, cur, i0 = [], int(lab[0]), 0
    for i in range(1, lab.size):
        if lab[i] != cur:
            secs.append(dict(name=names[cur], start=float(i0 * 0.5),
                             dur=float((i - i0) * 0.5), db=float(db[i0:i].mean())))
            cur, i0 = int(lab[i]), i
    secs.append(dict(name=names[cur], start=float(i0 * 0.5),
                     dur=float(max(0.0, dur - i0 * 0.5)),
                     db=float(db[i0:].mean())))
    # repetition: how much of the tune comes back (a chorus that returns)
    if HAS_LIBROSA and dur > 20:
        try:
            C = librosa.feature.chroma_cqt(y=m, sr=sr, hop_length=HOP)
            # a raw 11k x 11k recurrence matrix is a gigabyte; a beat of
            # chroma per half second is plenty to hear "the chorus came back"
            step = max(1, int(C.shape[1] / 400))
            C = C[:, ::step]
            R = librosa.segment.recurrence_matrix(
                C, mode="affinity", metric="cosine", sym=True, width=12)
            R = librosa.segment.path_enhance(R, 12)
            # for each moment, is there a strong match somewhere else?
            n = R.shape[0]
            gap = max(1, int(8.0 / (dur / n)))          # ignore near neighbours
            best = np.zeros(n)
            for i in range(n):
                lo = np.arange(max(0, i - gap))
                hi = np.arange(min(n, i + gap), n)
                cand = np.concatenate((R[i, lo], R[i, hi]))
                best[i] = cand.max() if cand.size else 0.0
            rep = float((best > 0.55).mean())
        except Exception:
            rep = 0.0
    else:
        rep = 0.0
    return dict(sections=secs, contrast_db=contrast, repeats=rep)


# --------------------------------------------------------------------------
# the verdict
# --------------------------------------------------------------------------

def verdict(a: dict) -> list[tuple[str, str, str]]:
    """(status, headline, detail) -- status in OK / WARN / FAIL"""
    out = []
    r, mel, lo, tx = a["rhythm"], a["melody"], a["loudness"], a["texture"]
    st = a["structure"]

    if mel["voiced"] < 0.25 and r["onsets_per_s"] > 2.0:
        # a drum record is supposed to have no melody; saying FAIL is a lie
        out.append(("--", "no pitched material (a drum / percussion track)",
                    f"{r['onsets_per_s']:.1f} onsets/s and nothing holding a pitch"))
    elif mel["notes_per_s"] >= 1.5:
        out.append(("OK", f"melody moves {mel['notes_per_s']:.1f} notes/s",
                    f"range {mel['range_semis']:.0f} semitones, "
                    f"held {mel['median_hold_s']:.2f}s a note"))
    elif mel["notes_per_s"] >= 0.6:
        out.append(("WARN", f"melody barely moves ({mel['notes_per_s']:.1f} notes/s)",
                    "a tune should change pitch at least twice a second"))
    else:
        out.append(("FAIL", "no melody -- this is a sustained note",
                    f"pitch changes {mel['notes_per_s']:.2f}/s and holds "
                    f"{mel['median_hold_s']:.2f}s a note"))

    tgt = 4.0
    if r["onsets_per_s"] >= 3.0:
        out.append(("OK", f"rhythmic: {r['onsets_per_s']:.1f} onsets/s",
                    f"tempo {r['tempo']:.0f} bpm, "
                    f"{r['low_onsets_per_s']:.1f}/s low, {r['high_onsets_per_s']:.1f}/s high"))
    else:
        out.append(("FAIL", "not rhythmic", f"only {r['onsets_per_s']:.1f} onsets/s"))

    if st["contrast_db"] >= 5.0:
        rep = ("repeats n/a, no pitched material to compare"
               if mel["voiced"] < 0.25 else
               f"{st['repeats']*100:.0f}% of the tune comes back")
        out.append(("OK", f"arrangement breathes ({st['contrast_db']:.1f} dB)",
                    f"{len(st['sections'])} sections, {rep}"))
    else:
        out.append(("WARN", "flat arrangement",
                    f"only {st['contrast_db']:.1f} dB between the quietest "
                    "and loudest part"))

    if lo["crest_db"] >= 9.0:
        out.append(("OK", f"dynamics intact (crest {lo['crest_db']:.1f} dB)",
                    f"{lo['lufs']:.1f} LUFS, peak {lo['peak_db']:.2f} dBFS"))
    elif lo["crest_db"] >= 6.0:
        out.append(("WARN", f"squashed (crest {lo['crest_db']:.1f} dB)",
                    f"{lo['lufs']:.1f} LUFS is loud for the peak it has"))
    else:
        out.append(("FAIL", f"crushed (crest {lo['crest_db']:.1f} dB)",
                    f"{lo['lufs']:.1f} LUFS with peak {lo['peak_db']:.2f} dBFS "
                    "-- the limiter is doing all the work"))
    return out


def loudest_window(x: np.ndarray, sr: int, window: float = 120.0):
    """The loudest stretch of a track is where the song actually is.

    Analysing four minutes of audio at 48 kHz costs several hundred MB and
    answers the same question as the best two minutes of it, so by default we
    measure the part a listener would call the tune.
    """
    n = x.shape[0]
    want = int(window * sr)
    if n <= want:
        return x, 0.0
    step = sr // 4
    m = np.asarray(mono(x), dtype=np.float32)
    nb = (n - want) // step
    if nb < 1:
        return x, 0.0
    rms = np.array([float((m[i * step:i * step + want].astype(np.float64) ** 2).mean())
                    for i in range(nb)])
    i = int(np.argmax(rms))
    return x[i * step:i * step + want], float(i * step / sr)


def analyze(path: str, window: float = 120.0) -> dict:
    x, sr = load(path)
    full = loudness(x, sr)                      # loudness is cheap, keep it whole
    x, offset = loudest_window(x, sr, window)
    r = rhythm(x, sr)
    # melody is expensive; only run it where there is something to hear
    mel = melody(x, sr)
    a = dict(file=path, loudness=full, rhythm=r, melody=mel,
             texture=texture(x, sr), harmony=harmony(x, sr),
             structure=structure(x, sr), window_start=offset)
    a["verdict"] = verdict(a)
    return a


def card(a: dict) -> str:
    L, R, M, T, H, S = (a["loudness"], a["rhythm"], a["melody"], a["texture"],
                        a["harmony"], a["structure"])
    icon = {"OK": "  ok  ", "WARN": " warn ", "FAIL": " FAIL ", "--": " n/a  "}
    w = a.get("window_start", 0.0)
    measured = (f"measured at {w:.0f}-{w + R['duration']:.0f}s"
                if w > 0.5 else "whole file")
    lines = [f"  {a['file']}",
             f"  {R['duration']:.0f}s window ({measured})   {R['tempo']:.0f} bpm"
             f"   {H['key']} {H['mode']}"
             f"   {L['lufs']:.1f} LUFS   peak {L['peak_db']:.2f} dBFS"
             f"   crest {L['crest_db']:.1f} dB",
             ""]
    for status, head, detail in a["verdict"]:
        lines.append(f"  [{icon[status]}] {head}")
        lines.append(f"          {detail}")
    lines += ["",
              f"  rhythm    {R['onsets_per_s']:.1f} onsets/s  "
              f"(low {R['low_onsets_per_s']:.1f}  high {R['high_onsets_per_s']:.1f})",
              (f"  melody    {M['notes_per_s']:.2f} notes/s   "
               f"hold {M['median_hold_s']:.2f}s   range {M['range_semis']:.0f} st   "
               f"prominence {M['prominence']:.2f}"
               if M["voiced"] >= 0.25 else
               f"  melody    n/a -- unpitched ({M['voiced']*100:.0f}% of frames "
               f"hold a steady pitch)"),
              f"  texture   flatness {T['flatness']:.3f}   "
              f"centroid {T['centroid_hz']:.0f} Hz   rolloff {T['rolloff_hz']:.0f} Hz",
              f"  harmony   {H['chords_per_min']:.0f} chord changes/min",
              f"  structure {len(S['sections'])} sections, "
              f"{S['contrast_db']:.1f} dB contrast, "
              + ("repeats n/a (unpitched)" if M["voiced"] < 0.25
                 else f"{S['repeats']*100:.0f}% repeats"),
              "  bands     " + "  ".join(f"{k} {v:+.0f}" for k, v in T["bands_db"].items())]
    return "\n".join(lines)
