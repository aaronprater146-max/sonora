#!/usr/bin/env python3
"""Understand a song: is it a song, or one long note?

    python3 tools/analyze.py song.mp3
    python3 tools/analyze.py song.mp3 --json

Prints the verdict a producer would give you:

    [  ok  ] melody moves 1.6 notes/s
    [ FAIL ] crushed (crest 4.1 dB)

Four checks -- melody, rhythm, arrangement, dynamics -- plus tempo, key,
chord rate, spectral balance and how much of the tune comes back.  It is
the fastest way to find out why a piece of generated music does not feel
like music.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from sonora import ears


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("audio", nargs="+")
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--window", type=float, default=120.0,
                    help="measure the loudest N seconds (0 = the whole file)")
    ap.add_argument("--fail", action="store_true",
                    help="exit 1 if any check fails (for CI / batch filtering)")
    args = ap.parse_args()

    bad = 0
    for path in args.audio:
        a = ears.analyze(path, window=args.window)
        if args.json:
            a["verdict"] = [list(v) for v in a["verdict"]]
            print(json.dumps(a, indent=2, default=str))
        else:
            print(ears.card(a))
            print()
        bad += sum(1 for v in a["verdict"] if v[0] == "FAIL")
    return 1 if (args.fail and bad) else 0


if __name__ == "__main__":
    raise SystemExit(main())
