#!/usr/bin/env python3
"""Render the individual layers of a Sonora song as separate audio files.

Generated music hides its faults in the mix: a lead that never moves, a pad
that is really just one long note, a bass that never rests.  Render the layers
on their own and you can hear (and measure) exactly which one is lying.

    python3 tools/stems.py --style techno-rave --seed 7 --out /tmp/stems

Writes <out>/drums.flac, bass.flac, pad.flac, arp.flac, lead.flac ... plus
mix.flac, and prints the Ears report for every layer that has content.
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

import sonora.core as C
from sonora import theory as T
from sonora.arrange import (LAYERS, STYLES, plan, Cache)
from sonora import ears


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--style", default="techno-rave")
    ap.add_argument("--key", default="Am")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--shape", default="default")
    ap.add_argument("--bpm", type=float, default=None)
    ap.add_argument("--out", default="/tmp/stems")
    args = ap.parse_args()

    tonic = T.parse_key(args.key)
    p = plan(args.style, seed=args.seed, tonic=tonic, shape=args.shape, bpm=args.bpm)
    st = p["st"]
    total = p["total"]
    os.makedirs(args.out, exist_ok=True)

    # one layer at a time: holding every bus at once is a gigabyte for a
    # four-minute song, and we only need each long enough to write it out
    rows = []
    for name, fn, _send in LAYERS:
        bus = np.zeros((C.sec(total + 5.0), 2), dtype=np.float32)
        cache = Cache()
        empty = True
        for sec in p["sections"]:
            y = fn(p, sec, cache)
            if y is None:
                continue
            gdb = st.get("mix", {}).get(name, 0.0)
            if gdb:
                y = (y * np.float32(10.0 ** (gdb / 20.0))).astype(np.float32)
            t0 = C.sec(sec["start"])
            n = min(y.shape[0], bus.shape[0] - t0)
            if n > 0:
                bus[t0:t0 + n] += y[:n]
                empty = False
            del y
        if empty or float(np.abs(bus).max()) < 1e-5:
            del bus
            continue
        path = os.path.join(args.out, f"{name}.flac")
        C.write(path, (bus / float(np.abs(bus).max()) * 0.7).astype(np.float32))
        rows.append((name, float(20 * np.log10(
            np.sqrt((bus.astype(np.float64) ** 2).mean()) + 1e-12))))
        del bus

    print(f"sonora {args.style}  {args.key}  {p['bpm']:.1f} bpm  seed {args.seed}\n")
    top = max(r[1] for r in rows)
    print(f"{'layer':10s} {'level':>7s}   ears report")
    print("-" * 78)
    for name, db in sorted(rows, key=lambda r: -r[1]):
        a = ears.analyze(os.path.join(args.out, f"{name}.flac"), window=45.0)
        mel, rhy = a["melody"], a["rhythm"]
        print(f"{name:10s} {db - top:6.1f} dB   "
              f"melody {mel['notes_per_s']:.2f} n/s  "
              f"hold {mel['median_hold_s']:.2f}s  "
              f"range {mel['range_semis']:.0f}st  "
              f"onsets {rhy['onsets_per_s']:.1f}/s  "
              f"flat {a['texture']['flatness']:.3f}")

    print(f"\nwrote {len(rows)} layers to {args.out}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
