#!/usr/bin/env python3
"""
make_song.py -- render a complete, mastered song.

    python3 tools/make_song.py --style cinematic-pop --key Am --out song.wav
    python3 tools/make_song.py --list
    python3 tools/make_song.py --style trap-dark --seed 42 --minutes 3 --out trap.flac

No network, no model, no licence, no limit.
"""
from __future__ import annotations

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

# honour --sr before sonora is imported (core reads it at import time)
for _i, _a in enumerate(sys.argv):
    if _a == "--sr" and _i + 1 < len(sys.argv):
        os.environ["SONORA_SR"] = str(int(sys.argv[_i + 1]))

import numpy as np  # noqa: E402

from sonora import core as C, arrange as A, master as M  # noqa: E402


def parse_key(k: str) -> int:
    """'Am' -> 57, 'C' -> 48, 'F#' -> 54, '57' -> 57"""
    from sonora import theory as T
    return T.parse_key(k)


def main() -> int:
    ap = argparse.ArgumentParser(description="Sonora song generator")
    ap.add_argument("--style", default="cinematic-pop", help="see --list")
    ap.add_argument("--list", action="store_true", help="list styles and exit")
    ap.add_argument("--key", default=None, help="tonic, e.g. Am / C / F# / 57")
    ap.add_argument("--bpm", type=float, default=None)
    ap.add_argument("--seed", type=int, default=None, help="any integer; same seed = same song")
    ap.add_argument("--shape", default="default", choices=["default", "short", "epic", "rave", "industrial",
                            "drowning", "groove"])
    ap.add_argument("--out", default="song.wav", help=".wav, .flac or .ogg")
    ap.add_argument("--sr", type=int, default=48000)
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    if args.list:
        print(f"{'style':16s}  {'bpm':10s}  description")
        for k, v in A.STYLES.items():
            print(f"{k:16s}  {str(v['bpm']):10s}  {v['desc']}")
        return 0

    import random
    seed = args.seed if args.seed is not None else random.randrange(1, 10 ** 9)
    if args.key is None:
        args.key = ["Am", "Cm", "Dm", "Em", "Fm", "Gm", "C", "A", "E", "F#"][seed % 10]
    if args.shape == "default":
        args.shape = {"techno-rave": "rave", "edm-festival": "rave",
                      "industrial-rock": "industrial",
                      "destructed-drums": "drowning",
                      "slow-groove": "groove"}.get(args.style, "default")
    tonic = parse_key(args.key)
    t0 = time.time()
    p = A.plan(args.style, seed=seed, tonic=int(round(tonic)) if not isinstance(tonic, int) else tonic,
               shape=args.shape, bpm=args.bpm)
    st = p["st"]
    if not args.quiet:
        print(f"sonora {args.style}  key={args.key}  bpm={p['bpm']:.1f}  "
              f"seed={seed}  sections={len(p['sections'])}  length={p['total']:.1f}s")

    def prog(msg):
        if not args.quiet:
            print("   " + msg)

    y = A.render(p, progress=prog)

    C.write(args.out, y, sr=args.sr)
    if not args.quiet:
        try:
            from . import feel
        except Exception:
            import importlib, sys, os
            sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
            from sonora import feel
        print("   " + feel.report(y, p["bpm"]))
        print(f"\nwrote {args.out}   {M.report(y)}")
        print(f"rendered in {time.time()-t0:.1f}s")
        print(f"seed {seed} -- reuse it with --seed {seed} to get this exact song again")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
