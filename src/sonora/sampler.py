"""
Sampler: plays the generated sample library stored in samplelib/.

Each pack is a folder of 16-bit FLAC files + manifest.json.  Notes between
sampled pitches are resampled (max +-1.5 semitones when sampled every 3),
sustained packs are looped with a crossfade, and every pack has 2 round
robin variants so repeated notes do not machine-gun.
"""
from __future__ import annotations

import json
import os
from functools import lru_cache

import numpy as np

from .core import (SR, FLOAT, sec, m2f, resample, fade, highpass, lowpass,
                   tanh_sat, ensure2, mix_at)

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_ROOT = os.path.join(HERE, "..", "..", "samplelib")


class Pack:
    def __init__(self, name: str, root: str = DEFAULT_ROOT):
        self.name = name
        self.dir = os.path.join(root, name)
        with open(os.path.join(self.dir, "manifest.json")) as f:
            self.manifest = json.load(f)
        self.sr = int(self.manifest["sr"])
        self.mode = self.manifest.get("mode", "oneshot")
        self.by_midi: dict[int, list[dict]] = {}
        for e in self.manifest["notes"]:
            self.by_midi.setdefault(int(e["midi"]), []).append(e)
        self.midis = sorted(self.by_midi)

    @property
    def available(self) -> bool:
        return bool(self.midis)

    def _pick(self, midi: int, variant: int):
        best = min(self.midis, key=lambda m: abs(m - midi))
        opts = self.by_midi[best]
        return opts[variant % len(opts)], best

    @lru_cache(maxsize=512)
    def _load(self, fname: str) -> np.ndarray:
        import soundfile as sf
        x, _ = sf.read(os.path.join(self.dir, fname), dtype="float32", always_2d=False)
        return np.asarray(x, dtype=np.float64)

    def note(self, midi: float, dur: float = 1.0, vel: float = 1.0,
             variant: int = 0, release: float = 0.25) -> np.ndarray:
        entry, src_midi = self._pick(int(round(midi)), variant)
        x = self._load(entry["file"]).copy()
        if self.sr != SR:
            x = resample(x.astype(FLOAT), self.sr / SR).astype(np.float64)
        ratio = 2.0 ** ((float(midi) - src_midi) / 12.0)   # <1 = stretch = lower
        if abs(ratio - 1.0) > 1e-4:
            x = resample(x.astype(FLOAT), float(ratio)).astype(np.float64)
        want = sec(dur + release + 0.05)
        if self.mode == "loop" and "loop" in entry and x.shape[0] < want:
            a, b = entry["loop"]
            a, b = sec(a * SR / self.sr), sec(b * SR / self.sr)
            a, b = min(a, x.shape[0] - 2), min(b, x.shape[0])
            if b - a > 256:
                xf = sec(0.08)
                head, tail = x[:b], x[a:]
                body = tail
                reps = int(np.ceil((want - x.shape[0]) / max(1, (b - a)))) + 1
                out = [head]
                cur = b
                prev_end = None
                for _ in range(reps):
                    seg = body.copy()
                    if prev_end is not None and xf > 0:
                        seg[:xf] *= np.linspace(0, 1, min(xf, seg.shape[0]))
                        seg[:xf] += prev_end * np.linspace(1, 0, min(xf, seg.shape[0]))
                    out.append(seg)
                    prev_end = seg[-xf:] if xf > 0 else None
                    cur += seg.shape[0]
                    if cur >= want:
                        break
                x = np.concatenate(out)
        x = x[:want] if x.shape[0] > want else np.pad(x, (0, want - x.shape[0]))
        if self.mode == "oneshot":
            x = fade(x.astype(FLOAT), 0.002, release).astype(np.float64)
        else:
            x = fade(x.astype(FLOAT), 0.006, release).astype(np.float64)
        return (x * float(vel)).astype(FLOAT)

    def chord(self, notes, dur: float = 1.0, vel: float = 1.0, seed: int = 1,
              spread: float = 0.6) -> np.ndarray:
        out = np.zeros((sec(dur + 1.0), 2), dtype=np.float32)
        for i, nt in enumerate(notes):
            y = self.note(nt, dur, vel, variant=seed + i)
            p = spread * (2 * (i + 0.5) / max(1, len(notes)) - 1)
            mix_at(out, y, 0.0, gain=1.0 / max(1.0, len(notes) ** 0.45), p=float(p))
        return out


@lru_cache(maxsize=32)
def load(name: str, root: str = DEFAULT_ROOT) -> Pack:
    return Pack(name, root)


def available(name: str, root: str = DEFAULT_ROOT) -> bool:
    p = os.path.join(root, name, "manifest.json")
    return os.path.exists(p)


def list_packs(root: str = DEFAULT_ROOT) -> list[str]:
    if not os.path.isdir(root):
        return []
    return sorted(d for d in os.listdir(root)
                  if os.path.exists(os.path.join(root, d, "manifest.json")))
