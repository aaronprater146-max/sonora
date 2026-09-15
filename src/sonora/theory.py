"""music theory + generative helpers (scales, chords, progressions, motifs)."""
from __future__ import annotations

import math
from typing import Sequence

import numpy as np

NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]


def note(name: str) -> int:
    """'C4' / 'F#3' / 'Bb2' -> midi number"""
    name = name.strip()
    step = name[0].upper()
    i = 1
    acc = 0
    while i < len(name) and name[i] in "#b":
        acc += 1 if name[i] == "#" else -1
        i += 1
    octv = int(name[i:])
    base = NOTE_NAMES.index(step)
    return (octv + 1) * 12 + base + acc


def name(m: int) -> str:
    return f"{NOTE_NAMES[int(m) % 12]}{int(m) // 12 - 1}"


SCALES = {
    "major": [0, 2, 4, 5, 7, 9, 11],
    "minor": [0, 2, 3, 5, 7, 8, 10],
    "harmonic_minor": [0, 2, 3, 5, 7, 8, 11],
    "melodic_minor": [0, 2, 3, 5, 7, 9, 11],
    "dorian": [0, 2, 3, 5, 7, 9, 10],
    "phrygian": [0, 2, 3, 5, 7, 8, 10],
    "lydian": [0, 2, 4, 6, 7, 9, 11],
    "mixolydian": [0, 2, 4, 5, 7, 9, 10],
    "locrian": [0, 1, 3, 5, 6, 8, 10],
    "pent_major": [0, 2, 4, 7, 9],
    "pent_minor": [0, 3, 5, 7, 10],
    "blues": [0, 3, 5, 6, 7, 10],
    "hirajoshi": [0, 2, 3, 7, 8],
    "lydian_dominant": [0, 2, 4, 6, 7, 9, 10],
}

CHORDS = {
    "": [0, 4, 7],
    "m": [0, 3, 7],
    "m7": [0, 3, 7, 10],
    "m9": [0, 3, 7, 10, 14],
    "m11": [0, 3, 7, 10, 14, 17],
    "maj7": [0, 4, 7, 11],
    "maj9": [0, 4, 7, 11, 14],
    "6": [0, 4, 7, 9],
    "m6": [0, 3, 7, 9],
    "7": [0, 4, 7, 10],
    "9": [0, 4, 7, 10, 14],
    "13": [0, 4, 7, 10, 14, 21],
    "7b9": [0, 4, 7, 10, 13],
    "7#9": [0, 4, 7, 10, 15],
    "sus2": [0, 2, 7],
    "sus4": [0, 5, 7],
    "7sus4": [0, 5, 7, 10],
    "add9": [0, 4, 7, 14],
    "madd9": [0, 3, 7, 14],
    "dim": [0, 3, 6],
    "dim7": [0, 3, 6, 9],
    "m7b5": [0, 3, 6, 10],
    "aug": [0, 4, 8],
    "5": [0, 7],
    "maj7#11": [0, 4, 7, 11, 18],
}

# degree -> semitone offset from tonic, per quality of scale
DEGREE = {"I": 0, "II": 2, "III": 4, "IV": 5, "V": 7, "VI": 9, "VII": 11}


def chord(root: int, quality: str = "") -> list[int]:
    return [root + i for i in CHORDS.get(quality, CHORDS[""])]


def chord_notes(tonic: int, scale: str, degree: int, ext: str = "") -> list[int]:
    """diatonic chord built on scale degree (0-indexed) of `scale`"""
    s = SCALES[scale]
    n = len(s)
    out = []
    for k in (0, 2, 4, 6):
        idx = degree + k
        out.append(tonic + s[idx % n] + 12 * (idx // n))
    notes = out[:3]
    if ext == "7":
        notes.append(out[3])
    elif ext in ("9", "add9"):
        notes.extend([out[3], out[2] + 12])
    return notes


def scale_notes(tonic: int, scale: str, lo: int = 48, hi: int = 84) -> list[int]:
    s = SCALES.get(scale, SCALES["major"])
    out = []
    o = -4
    while True:
        for d in s:
            m = tonic + d + 12 * (o + (tonic // 12))
            if m > hi:
                return sorted(set(out))
            if m >= lo:
                out.append(m)
        o += 1
        if o > 12:
            return sorted(set(out))


# stylistic chord loops: (scale degree, extension) per bar
PROGRESSIONS = {
    "pop": [("I", ""), ("V", ""), ("vi", ""), ("IV", "")],
    "epic": [("vi", ""), ("IV", ""), ("I", ""), ("V", "")],
    "sad": [("i", ""), ("VI", ""), ("III", ""), ("VII", "")],
    "trap": [("i", ""), ("VI", ""), ("iv", ""), ("v", "")],
    "edm": [("vi", ""), ("IV", ""), ("V", ""), ("I", "")],
    "lofi": [("ii", "7"), ("V", "7"), ("I", "maj7"), ("vi", "7")],
    "jazz": [("ii", "7"), ("V", "7"), ("I", "maj7"), ("vi", "7")],
    "cinematic": [("i", ""), ("VI", ""), ("iv", ""), ("V", "")],
    "heroic": [("I", ""), ("V", ""), ("vi", ""), ("iii", ""), ("IV", ""), ("I", ""), ("IV", ""), ("V", "")],
    "soul": [("I", "maj7"), ("vi", "7"), ("ii", "7"), ("V", "7")],
}

ROMAN = {"i": 0, "ii": 1, "iii": 2, "iv": 3, "v": 4, "vi": 5, "vii": 6,
         "I": 0, "II": 1, "III": 2, "IV": 3, "V": 4, "VI": 5, "VII": 6}


def progression(tonic: int, scale: str, template: Sequence[tuple[str, str]],
                bars: int = 4) -> list[list[int]]:
    out = []
    for i in range(bars):
        deg, ext = template[i % len(template)]
        minorish = deg.islower()
        s = scale
        if scale == "major" and minorish:
            s = "minor"
        if scale in ("minor", "harmonic_minor") and not minorish:
            s = "harmonic_minor" if scale == "harmonic_minor" else "minor"
        out.append(chord_notes(tonic, s, ROMAN[deg], ext))
    return out


def rhythm(style: str, bar: int, seed: int = 1) -> list[tuple[float, float, float]]:
    """returns (beat_offset, duration_beats, velocity) hits for one bar"""
    g = np.random.default_rng(seed * 7919 + bar)
    p = {
        "quarter": [(0, 1, 1.0), (1, 1, .9), (2, 1, 1.0), (3, 1, .9)],
        "half": [(0, 2, 1.0), (2, 2, .9)],
        "whole": [(0, 4, 1.0)],
        "syncop": [(0, 1.5, 1.0), (1.5, .5, .8), (2, 1, 1.0), (3, .5, .85), (3.5, .5, .7)],
        "trap": [(0, 1.5, 1.0), (1.5, .5, .8), (2.5, 1, .95), (3.5, .5, .7)],
        "arp8": [(i * .5, .5, 1.0 if i % 2 == 0 else .75) for i in range(8)],
        "arp16": [(i * .25, .25, 1.0 if i % 4 == 0 else .7) for i in range(16)],
        "offbeat": [(.5, .5, .8), (1.5, .5, .8), (2.5, .5, .8), (3.5, .5, .8)],
        "ballad": [(0, 1, 1.0), (1.5, .5, .7), (2.5, 1.5, .9)],
        "push": [(.75, .75, .9), (1.75, .75, .85), (2.5, 1.0, 1.0), (3.5, .5, .7)],
        "acid": [(0, .25, 1.0), (.5, .25, .62), (.75, .25, .9), (1.25, .25, .58),
                 (1.5, .25, .85), (2.0, .5, 1.0), (2.75, .25, .88), (3.0, .25, .6),
                 (3.5, .5, .9)],
        "offbeat16": [(i * .25, .25, 1.0 if i % 4 == 0 else .7) for i in range(16)],
        "roll": [(i * .25, .25, min(1.0, .45 + i * .06)) for i in range(16)],
    }[style]
    return [(float(a), float(b), float(c)) for a, b, c in p]


def motif(scale_notes_: Sequence[int], length: int, seed: int = 1,
          center: int = 60, span: int = 12, rest_prob: float = 0.12) -> list[int | None]:
    """generate a singable motif: stepwise motion with one or two leaps"""
    g = np.random.default_rng(seed)
    pool = [n for n in scale_notes_ if center - span <= n <= center + span]
    if not pool:
        pool = list(scale_notes_)
    i = len(pool) // 2
    out: list[int | None] = []
    for k in range(length):
        step = g.choice([-2, -1, -1, 0, 1, 1, 2, 3, -3], p=[.05, .18, .18, .07, .18, .18, .10, .03, .03])
        i = int(np.clip(i + step, 0, len(pool) - 1))
        out.append(None if (k > 0 and g.random() < rest_prob) else pool[i])
    return out


def transpose(seq, semis: int):
    return [None if n is None else n + semis for n in seq]


def humanize(times: Sequence[float], amount: float = 0.006, seed: int = 1):
    g = np.random.default_rng(seed)
    return [t + float(g.uniform(-amount, amount)) for t in times]


def voicing(chords, low: int = 60, high: int = 79, spread: float = 0.0):
    """smooth 4-part voicings for a chord loop.

    Picks, for each chord, the inversion whose notes move the least from the
    previous chord and that sits inside [low, high].  This is what stops a
    generated backing track from sounding like a pile of random block chords.
    """
    out, prev = [], None
    mid = (low + high) / 2.0
    for ch in chords:
        pcs = sorted(set(int(n) % 12 for n in ch))
        best = None
        for base in range(low, high - 11):
            notes = sorted(base + ((pc - base) % 12) for pc in pcs)
            if notes[-1] > high:
                continue
            score = abs((notes[0] + notes[-1]) / 2.0 - mid) * 0.6
            if prev is not None:
                score += sum(abs(a - b) for a, b in zip(notes, prev)) * 1.0
                score += abs(notes[0] - prev[0]) * 0.5
            if best is None or score < best[0]:
                best = (score, notes)
        if best is None:
            notes = sorted(min(high - 12, max(low, int(n))) for n in ch)
        else:
            notes = best[1]
        out.append(notes)
        prev = notes
    return out


def bass_line(chords, low: int = 33, high: int = 45) -> list[int]:
    """root notes folded into the bass register with minimal movement"""
    out, prev = [], None
    for ch in chords:
        pc = int(ch[0]) % 12
        cands = [n for n in range(low, high + 1) if n % 12 == pc]
        if not cands:
            cands = [low + pc % 12]
        n = cands[len(cands) // 2] if prev is None else min(cands, key=lambda c: abs(c - prev))
        out.append(n)
        prev = n
    return out
