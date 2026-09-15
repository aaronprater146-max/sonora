#!/usr/bin/env python3
"""Audition a single instrument:  python3 tools/play_instrument.py violin 60 2"""
from __future__ import annotations
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from sonora import core as C
from sonora.instruments import strings as S, keys as K, synth as Y, guitar as G, voice as V, world as W, drums as D

ONE = {
    # name: (callable(midi, dur),)
    "piano": lambda m, d: K.piano(m, d), "felt_piano": lambda m, d: K.felt_piano(m, d),
    "rhodes": lambda m, d: K.rhodes(m, d), "organ": lambda m, d: K.organ(m, d),
    "clav": lambda m, d: K.clav(m, d), "glock": lambda m, d: K.bell(m, d, kind="glock"),
    "bell": lambda m, d: K.bell(m, d, kind="bell"), "marimba": lambda m, d: K.bell(m, d, kind="marimba"),
    "vibes": lambda m, d: K.bell(m, d, kind="vibes"),
    "violin": lambda m, d: S.bowed(m, d, body="violin", players=3),
    "viola": lambda m, d: S.bowed(m, d, body="viola", players=3),
    "cello": lambda m, d: S.bowed(m, d, body="cello", players=3),
    "bass_arco": lambda m, d: S.bowed(m, d, body="bass", players=2),
    "spiccato": lambda m, d: S.spiccato(m, d), "tremolo": lambda m, d: S.tremolo(m, d),
    "pizz": lambda m, d: S.pluck(C.m2f(m), d, body="cello"),
    "harp": lambda m, d: S.harp_note(m, d),
    "guitar": lambda m, d: G.steel(m, d), "nylon": lambda m, d: G.nylon(m, d),
    "electric": lambda m, d: G.electric(m, d, gain=0.6),
    "voice_ah": lambda m, d: V.vowel(m, d, "ah"), "voice_oo": lambda m, d: V.vowel(m, d, "oo"),
    "choir": lambda m, d: V.choir([m, m + 4, m + 7], d, size=9),
    "flute": lambda m, d: W.flute(m, d), "shakuhachi": lambda m, d: W.shakuhachi(m, d),
    "whistle": lambda m, d: W.whistle(m, d), "kalimba": lambda m, d: W.kalimba(m, d),
    "koto": lambda m, d: W.koto(m, d), "sitar": lambda m, d: W.sitar(m, d),
    "handpan": lambda m, d: W.handpan(m, d), "steelpan": lambda m, d: W.steelpan(m, d),
    "harmonica": lambda m, d: W.harmonica(m, d), "accordion": lambda m, d: W.accordion(m, d),
    **{f"synth_{k}": (lambda k: (lambda m, d: Y.note(m, d, k)))(k) for k in Y.PRESETS},
    **{f"drum_{k}": (lambda k: (lambda m, d: D.hit(k)))(k)
       for k in ["kick", "snare", "hat", "openhat", "clap", "tom", "rim", "shaker",
                 "tamb", "cowbell", "crash", "ride", "taiko", "808"]},
}


def main() -> int:
    if len(sys.argv) < 2 or sys.argv[1] in ("-l", "--list"):
        print("available instruments:")
        for k in sorted(ONE):
            print("  " + k)
        return 0
    name = sys.argv[1]
    midi = float(sys.argv[2]) if len(sys.argv) > 2 else 60.0
    dur = float(sys.argv[3]) if len(sys.argv) > 3 else 2.0
    out = sys.argv[4] if len(sys.argv) > 4 else f"{name}.wav"
    if name not in ONE:
        print(f"unknown instrument {name!r} (try --list)")
        return 1
    y = ONE[name](midi, dur)
    y = C.ensure2(y)
    C.write(out, y / max(1e-6, float(abs(y).max())) * 0.9)
    print(f"wrote {out}  ({len(y)/C.SR:.2f}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
