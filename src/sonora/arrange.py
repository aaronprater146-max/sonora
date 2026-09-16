"""
Arrangement engine: turns a style + key + seed into a finished stereo mix.

Everything here is deterministic -- the same seed always produces the same
song, and there is no model download, no API key and no per-render cost.
"""
from __future__ import annotations

import math
from typing import Callable

import numpy as np

from . import core as C
from . import theory as T
from . import master as M
from .instruments import drums as D, strings as S, keys as K, synth as Y, guitar as G, voice as V, world as W

# --------------------------------------------------------------------------
# styles
# --------------------------------------------------------------------------

STYLES: dict[str, dict] = {
    "cinematic-pop": dict(
        bpm=(84, 96), scale="minor", prog="epic", kit="epic", drum="epic",
        bass="sub_bass", bass_rhythm="whole", pad="wide_pad", lead="saw_lead",
        arp="pluck", keys="piano", strings=True, choir=True, bells=True,
        perc="tamb", ir="hall", rev=0.32, sidechain=2.5, lufs=-10.5, width=1.10,
        desc="hybrid orchestral / modern pop -- the trailer-friendly one"),
    "trap-dark": dict(
        bpm=(68, 78), scale="minor", prog="trap", kit="trap", drum="trap",
        bass="808bass", bass_rhythm="trap", pad="dream_pad", lead="choir_lead",
        arp="glass_pluck", keys="felt_piano", strings=True, choir=True, bells=True,
        perc="shaker", ir="room", rev=0.22, sidechain=6.0, lufs=-8.5, width=1.08,
        desc="dark 808 / hi-hat heavy beat with choir tops"),
    "edm-festival": dict(
        bpm=(124, 130), scale="minor", prog="edm", kit="edm", drum="four_on_floor",
        bass="reese", bass_rhythm="offbeat", pad="supersaw", lead="supersaw",
        arp="square_lead", keys="", strings=False, choir=False, bells=True,
        perc="shaker", ir="hall", rev=0.26, sidechain=7.0, lufs=-7.5, width=1.10,
        desc="big-room drop machine: supersaws, offbeat bass, heavy pumping"),
    "lofi-keys": dict(
        bpm=(72, 84), scale="minor", prog="lofi", kit="lofi", drum="lofi",
        bass="moog_bass", bass_rhythm="ballad", pad="dream_pad", lead="",
        arp="glass_pluck", keys="rhodes", strings=False, choir=False, bells=True,
        perc="shaker", ir="room", rev=0.3, sidechain=1.5, lufs=-13.0, width=1.10,
        desc="dusty rhodes + soft drums, tape-ish and warm"),
    "orchestral": dict(
        bpm=(76, 92), scale="minor", prog="cinematic", kit="epic", drum="epic",
        bass="sub_bass", bass_rhythm="whole", pad="dream_pad", lead="brass",
        arp="", keys="piano", strings=True, choir=True, bells=True,
        perc="tamb", ir="hall", rev=0.42, sidechain=0.0, lufs=-13.0, width=1.10,
        desc="strings, brass, taiko and choir -- no drums until the peaks"),
    "synthwave": dict(
        bpm=(100, 116), scale="minor", prog="pop", kit="edm", drum="edm",
        bass="moog_bass", bass_rhythm="arp8", pad="wide_pad", lead="square_lead",
        arp="pluck", keys="", strings=False, choir=False, bells=False,
        perc="shaker", ir="hall", rev=0.3, sidechain=3.0, lufs=-10.0, width=1.10,
        desc="analogue arpeggios, gated-ish drums, neon pads"),
    "ambient-piano": dict(
        bpm=(60, 72), scale="major", prog="sad", kit="lofi", drum="none",
        bass="sub_bass", bass_rhythm="whole", pad="dream_pad", lead="",
        arp="", keys="felt_piano", strings=True, choir=True, bells=True,
        perc="", ir="hall", rev=0.5, sidechain=0.0, lufs=-15.0, width=1.10,
        desc="slow felt piano with strings and wordless voice"),
    "world-fusion": dict(
        bpm=(88, 104), scale="dorian", prog="soul", kit="acoustic", drum="break",
        bass="moog_bass", bass_rhythm="syncop", pad="dream_pad", lead="flute",
        arp="kalimba", keys="", strings=True, choir=True, bells=True,
        perc="shaker", ir="hall", rev=0.34, sidechain=1.0, lufs=-11.5, width=1.10,
        desc="harp, kalimba, handpan, flute and strings over live-feeling drums"),
    "destructed-drums": dict(
        bpm=(118, 122), scale="minor", prog="cinematic",
        kit="drowning", drum="drowning",
        bass="", pad="", lead="", arp="", keys="",
        strings=False, choir=False, bells=False, perc="shaker",
        ir="room", rev=0.16, sidechain=0.0, lufs=-11.0, width=1.08,
        glue=0.50, air=0.90, punch=1.35, tilt=0.0,
        mix=dict(drums=0.0, perc=-2.0, fx=2.0),
        destroy=1.0, drop_break=True, post_auto=True, auto_depth=-8.5,
        desc="drums only, every hit damaged differently: crushed, ring-modulated, reversed"),
    "industrial-rock": dict(
        bpm=(80, 88), scale="minor", prog="cinematic", kit="industrial",
        drum="industrial", bass="distorted_bass", bass_rhythm="push",
        pad="static_bed", lead="glitch_lead", arp="", keys="bend_guitar",
        strings=False, choir=False, bells=False, perc="",
        ir="room", rev=0.11, sidechain=2.5, lufs=-11.0, width=1.06,
        glue=0.55, air=0.75, punch=1.30, tilt=-2.5,
        # the fix for "all I can hear is one sustained note": the pad is a
        # bed, not a wall, and the lead sits in front of everything
        mix=dict(pad=-12.0, lead=7.0, bass=9.0, keys=-4.0, fx=1.0),
        stutter=True, bend=True, drop_break=True, post_auto=True,
        auto_depth=-9.0,
        desc="damaged static synths, glitched edits, tight dry drums"),
    "techno-rave": dict(
        bpm=(138, 150), scale="minor", prog="edm", kit="rave", drum="rave",
        bass="acid", bass_rhythm="acid", pad="wide_pad", lead="supersaw",
        arp="stab", keys="", strings=False, choir=False, bells=False,
        perc="shaker", ir="hall", rev=0.20, sidechain=7.5, lufs=-7.5, width=1.10,
        glue=0.72, air=1.35, punch=1.20, drop_break=True, post_auto=True,
        desc="four-on-the-floor, offbeat open hats, 303 acid line, hoover drops"),
    "gospel-soul": dict(
        bpm=(74, 88), scale="major", prog="soul", kit="acoustic", drum="ballad",
        bass="moog_bass", bass_rhythm="push", pad="organ_pad", lead="choir_lead",
        arp="glass_pluck", keys="rhodes", strings=True, choir=True, bells=False,
        perc="tamb", ir="hall", rev=0.32, sidechain=1.0, lufs=-11.0, width=1.10,
        desc="rhodes + hammond + choir, Sunday-morning energy"),
    "rock-anthem": dict(
        bpm=(104, 124), scale="minor", prog="heroic", kit="modern", drum="four_on_floor",
        bass="moog_bass", bass_rhythm="arp8", pad="", lead="epic_brass",
        arp="", keys="", strings=True, choir=True, bells=False,
        perc="tamb", ir="room", rev=0.24, sidechain=2.0, lufs=-8.5, width=1.10,
        desc="power-chord guitars, big drums, gang-vocal chorus"),
}

ARRANGEMENTS: dict[str, list[tuple[str, int, float]]] = {
    "default": [
        ("intro", 4, 0.20), ("verse", 8, 0.45), ("pre", 4, 0.60),
        ("chorus", 8, 0.85), ("verse", 8, 0.55), ("pre", 4, 0.70),
        ("chorus", 8, 0.95), ("bridge", 8, 0.40), ("chorus", 8, 1.00),
        ("outro", 4, 0.25),
    ],
    "short": [("intro", 4, 0.25), ("verse", 8, 0.5), ("chorus", 8, 0.9),
              ("verse", 8, 0.6), ("chorus", 8, 1.0), ("outro", 4, 0.25)],
    "drowning": [("intro", 8, 0.35), ("verse", 12, 0.60), ("pre", 6, 0.74),
                 ("chorus", 12, 1.00), ("verse", 12, 0.64), ("bridge", 10, 0.30),
                 ("pre", 6, 0.80), ("chorus", 12, 1.00), ("bridge", 8, 0.34),
                 ("verse", 12, 0.72), ("pre", 6, 0.84), ("chorus", 12, 0.95),
                 ("outro", 8, 0.34)],
    "industrial": [("intro", 8, 0.40), ("verse", 8, 0.62), ("pre", 4, 0.74),
                   ("chorus", 8, 1.00), ("verse", 8, 0.66), ("pre", 4, 0.78),
                   ("chorus", 8, 1.00), ("bridge", 8, 0.32), ("pre", 4, 0.80),
                   ("chorus", 8, 1.00), ("verse", 8, 0.70), ("outro", 8, 0.36)],
    "rave": [("intro", 8, 0.45), ("verse", 8, 0.62), ("pre", 8, 0.74),
             ("chorus", 8, 1.00), ("bridge", 8, 0.30), ("pre", 8, 0.80),
             ("chorus", 8, 1.00), ("verse", 8, 0.86), ("outro", 8, 0.42)],
    "epic": [("intro", 8, 0.2), ("verse", 8, 0.45), ("pre", 8, 0.65),
             ("chorus", 8, 0.9), ("verse", 8, 0.5), ("pre", 8, 0.7),
             ("chorus", 8, 1.0), ("bridge", 8, 0.35), ("chorus", 8, 1.0),
             ("outro", 8, 0.25)],
}


# --------------------------------------------------------------------------
# planning
# --------------------------------------------------------------------------

def plan(style: str = "cinematic-pop", seed: int = 1, tonic: int | str = 57,
         shape: str = "default", bpm: float | None = None) -> dict:
    st = STYLES.get(style, STYLES["cinematic-pop"])
    g = np.random.default_rng(seed * 7919 + 13)
    if isinstance(tonic, str):
        tonic = T.note(tonic)
    bpm = bpm or float(g.uniform(*st["bpm"]))
    spb = 60.0 / bpm
    sections = []
    t = 0.0
    for i, (name, bars, energy) in enumerate(ARRANGEMENTS.get(shape, ARRANGEMENTS["default"])):
        dur = bars * 4 * spb
        sections.append(dict(name=name, index=i, bars=bars, start=t, dur=dur,
                             energy=energy, bpm=bpm))
        t += dur
    total = t
    # harmony: one 4-bar loop per section, transposed for lift in later choruses
    prog_tpl = T.PROGRESSIONS.get(st["prog"], T.PROGRESSIONS["pop"])
    for s in sections:
        shift = 0
        if s["name"] == "chorus" and s["index"] >= 6:
            shift = 0  # keep the hook in the same key, vary the voicing instead
        chords = []
        for bar in range(max(4, s["bars"])):
            ch = T.progression(tonic + shift, st["scale"], prog_tpl, bars=1)[0]
            chords.append(ch)
        s["chords"] = chords[:max(1, s["bars"])]
    # melody motif for the whole song (4-bar cells, varied per section)
    scale_notes = T.scale_notes(tonic, st["scale"], tonic - 5, tonic + 19)
    s0 = sections[0]
    motif = T.motif(scale_notes, 16, seed=seed, center=tonic + 12, span=14)
    for i, s in enumerate(sections):
        m = list(motif)
        if s["name"] == "bridge":
            m = T.transpose(m, -2)
            m = m[::-1]
        elif s["name"] == "pre":
            m = [n for k, n in enumerate(m) if k % 2 == 0]
        elif s["energy"] > 0.8:
            m = T.transpose(m, 12) if i % 2 else m
        s["motif"] = m
    return dict(style=style, st=st, bpm=bpm, spb=spb, tonic=tonic,
                scale=st["scale"], sections=sections, total=total, seed=seed)


# --------------------------------------------------------------------------
# fx generators
# --------------------------------------------------------------------------

def riser(dur: float = 2.0, seed: int = 1) -> np.ndarray:
    n = C.sec(dur)
    t = np.arange(n) / C.SR
    u = t / max(0.01, dur)
    ns = C.noise(n, seed)
    y = C.filt_env(ns, 300.0 + 8000.0 * u ** 2, 1.2, "bp") * (0.2 + 0.8 * u ** 2)
    saw = C.osc_saw(80.0 * (1 + 7 * u ** 2), n)
    y = y * 0.6 + C.filt_env(saw, 400 + 4000 * u, 0.7, "hp") * 0.25 * u
    y *= np.exp(-np.maximum(0, t - dur + 0.12) / 0.03)
    return C.fade(C.tanh_sat(y, 1.4).astype(C.FLOAT), 0.2, 0.03) * 0.6


def downlifter(dur: float = 1.2, seed: int = 1) -> np.ndarray:
    n = C.sec(dur)
    t = np.arange(n) / C.SR
    u = t / max(0.01, dur)
    y = C.osc_saw(600.0 * (1 - 0.85 * u), n) * (1 - u) ** 1.5
    y = C.filt_env(y, 3000 * (1 - 0.9 * u) + 200, 1.0, "lp")
    return C.fade(y.astype(C.FLOAT), 0.01, 0.2) * 0.5


def impact(seed: int = 1, boom: float = 1.0) -> np.ndarray:
    n = C.sec(3.0)
    t = np.arange(n) / C.SR
    sub = np.sin(2 * math.pi * np.cumsum(70.0 * (1 - 0.55 * np.clip(t * 3, 0, 1))) / C.SR)
    sub *= np.exp(-t / 0.55) * boom
    body = C.bandpass(C.noise(n, seed), 180.0, 0.7) * np.exp(-t / 0.25) * 0.7
    crack = C.highpass(C.noise(n, seed + 1), 3000.0, 0.7) * np.exp(-t / 0.05) * 0.5
    y = C.tanh_sat(sub + body + crack, 1.8)
    return C.fade(y.astype(C.FLOAT), 0.001, 0.3) * 0.8


def reverse_sweep(dur: float = 1.6, seed: int = 1) -> np.ndarray:
    n = C.sec(dur)
    t = np.arange(n) / C.SR
    y = C.filt_env(C.noise(n, seed), 2500 + 5000 * (t / dur), 0.8, "bp")
    y = y * (t / dur) ** 3
    return C.fade(y.astype(C.FLOAT), 0.01, 0.01) * 0.7


def vinyl_tail(dur: float = 1.0, seed: int = 1) -> np.ndarray:
    n = C.sec(dur)
    t = np.arange(n) / C.SR
    y = C.noise(n, seed) * np.exp(-t / 0.25) * 0.05
    return C.bandpass(y, 4000.0, 0.5).astype(C.FLOAT)


# --------------------------------------------------------------------------
# stem renderers
# --------------------------------------------------------------------------

class Cache:
    def __init__(self):
        self.d = {}

    def get(self, k):
        return self.d.get(k)

    def put(self, k, v):
        if len(self.d) < 4000:
            self.d[k] = v
        return v


def _drum_variant(style: str, energy: float, name: str) -> str:
    if style == "drowning":
        # a drum record does not stop drumming for the break, it thins out
        if name == "bridge":
            return "drowning_break"
        if name in ("intro", "outro"):
            return "drowning_intro"
        return "drowning_drive" if energy >= 0.72 else "drowning"
    if name == "bridge":
        return "none" if energy < 0.45 else "break"
    if style == "industrial":
        if name in ("intro", "outro"):
            return "industrial_intro"
        return "industrial_drive" if energy >= 0.72 else "industrial"
    if style == "rave":
        return "rave_intro" if name in ("intro", "outro") else "rave"
    if name in ("intro", "outro"):
        return "none"
    if energy < 0.5:
        return "pop"
    if energy < 0.7:
        return style if style in ("trap", "epic", "edm") else "pop"
    return style


def render_drums(p: dict, sec: dict, cache: Cache) -> np.ndarray:
    st = p["st"]
    variant = _drum_variant(st["drum"], sec["energy"], sec["name"])
    if variant == "none":
        return None
    key = ("drums", variant, sec["bars"], sec["index"])
    y = cache.get(key)
    if y is None:
        evs = D.pattern(variant, bars=sec["bars"], bpm=p["bpm"],
                        seed=p["seed"] + sec["index"], fill_every=8 if sec["bars"] >= 8 else 4)
        # thin the kit out at low energy
        if sec["energy"] < 0.5:
            evs = [e for e in evs if e[1] not in ("snare", "clap", "crash") or
                   (int((e[0] / p["spb"]) % 4) == 2)]
        if sec["name"] == "pre" and sec["bars"] >= 4:
            # the build: 8th notes for a bar, then 16ths, climbing in velocity
            spb, step = p["spb"], p["spb"] / 4.0
            for bar in (sec["bars"] - 2, sec["bars"] - 1):
                div = 2 if bar == sec["bars"] - 2 else 1
                n_hits = int(round(4 * spb / (step * div)))
                for k in range(n_hits):
                    evs.append((bar * 4 * spb + k * step * div, "snare",
                                0.3 + 0.7 * (k / max(1, n_hits - 1)) ** 1.5))
            evs.sort()
        y = D.render_pattern(evs, kit=st["kit"], length=sec["dur"], tail=2.5,
                             room=0.10, seed=p["seed"])
        y = M.stem_fx(y, hp=26.0, comp=(-14.0, 2.2, 0.006, 0.09), sat=0.12,
                      width=1.02, gain_db=-5.0)
        if st.get("destroy"):
            # more damage where the record is loudest: the drops are the
            # parts that should sound like the machine is coming apart
            amt = st["destroy"] * (0.55 + 0.45 * float(sec["energy"]))
            y = C.mangle(y, seed=p["seed"] + sec["index"] * 17, amount=amt)
        cache.put(key, y)
    return y


def _maybe_sample(pack: str, notes, dur: float, seed: int, fallback):
    """use the generated sample library when it is present (much faster too)"""
    import os
    if not os.environ.get("SONORA_SAMPLES"):
        return fallback()
    try:
        from . import sampler
        if sampler.available(pack):
            return sampler.load(pack).chord(list(notes), dur, seed=seed)
    except Exception:
        pass
    return fallback()


def render_bass(p: dict, sec: dict, cache: Cache) -> np.ndarray:
    st = p["st"]
    if not st["bass"]:
        return None
    if st.get("drop_break") and sec["name"] == "bridge" and sec["energy"] < 0.45:
        return None          # breakdown: no bass, so the drop has somewhere to go
    out = np.zeros((C.sec(sec["dur"] + 2.0), 2), dtype=np.float32)
    preset = st["bass"]
    drop = -36 if preset in ("808bass", "reese") else -24
    roots = T.bass_line(sec["chords"])
    for bar, (chord, root) in enumerate(zip(sec["chords"], roots)):
        root = max(24, root + (12 if drop == -36 else 0))
        pat = T.rhythm(st["bass_rhythm"], bar, seed=p["seed"] + bar)
        for (off, durn, vel) in pat:
            nt = root
            if st["bass_rhythm"] == "acid":
                seq = [0, 0, 12, 0, 7, 0, 12, 3, 0, 12, 0, 7, 12, 0, 7, 10]
                nt = root + seq[int(round(off * 4)) % len(seq)]
            elif st["bass_rhythm"] in ("arp8", "offbeat", "push", "syncop"):
                idx = int(off * 2) % len(chord)
                nt = root + [0, 7, 12, 7][idx % 4]
            dur = durn * p["spb"]
            key = ("bass", preset, nt, round(dur, 3), round(vel, 2))
            y = _stack(cache, key, lambda: Y.note(nt, dur * 0.95, preset, vel,
                                                  seed=p["seed"] + nt))
            C.mix_at(out, y, bar * 4 * p["spb"] + off * p["spb"], gain=0.9, p=0.0)
            if preset in ("sub_bass", "808bass", "reese"):
                k2 = ("bass_h", nt + 12, round(dur, 3), round(vel, 2))
                y2 = _stack(cache, k2, lambda: Y.note(nt + 12, dur * 0.9, "moog_bass",
                                                      vel * 0.8, seed=p["seed"] + nt))
                C.mix_at(out, y2, bar * 4 * p["spb"] + off * p["spb"], gain=0.16, p=0.0)
    return M.stem_fx(out, hp=22.0, comp=(-9.0, 3.2, 0.008, 0.10), sat=0.22,
                     width=1.00, gain_db=-12.0)


def _stack(cache, key, fn):
    y = cache.get(key)
    if y is None:
        y = fn()
        cache.put(key, y)
    return y


def _chord_synth(notes, dur, preset, seed) -> np.ndarray:
    out = np.zeros((C.sec(dur + 2.5), 2), dtype=np.float32)
    for i, nt in enumerate(notes):
        y = Y.note(nt, dur, preset, 0.85, seed=seed + i)
        pan = (i - (len(notes) - 1) / 2) * (0.5 / max(1, len(notes) - 1)) * 1.6
        C.mix_at(out, y, 0.0, gain=1.0 / max(1.0, len(notes) ** 0.5), p=float(pan))
    return out


def render_pad(p: dict, sec: dict, cache: Cache) -> np.ndarray:
    st = p["st"]
    if not st["pad"]:
        return None
    out = np.zeros((C.sec(sec["dur"] + 2.5), 2), dtype=np.float32)
    bar_dur = 4 * p["spb"]
    voic = T.voicing(sec["chords"], 60, 79)
    for bar, notes in enumerate(voic):
        dur = bar_dur * 1.06
        key = ("pad", st["pad"], tuple(notes), round(dur, 2))
        y = _stack(cache, key, lambda: _chord_synth(notes, dur, st["pad"], p["seed"]))
        C.mix_at(out, y, bar * bar_dur, gain=0.55 + 0.3 * sec["energy"], p=0.0)
    return M.stem_fx(out, hp=170.0, lp=8500.0, comp=(-20.0, 1.6, 0.05, 0.25),
                     sat=0.05, width=1.10, gain_db=-5.5)


def render_keys(p: dict, sec: dict, cache: Cache) -> np.ndarray:
    st = p["st"]
    if not st.get("keys"):
        return None
    # sampled instruments when the name matches, otherwise any synth preset
    fn = {"piano": K.piano, "felt_piano": K.felt_piano, "rhodes": K.rhodes,
          "organ": K.organ, "clav": K.clav}.get(st["keys"])
    if fn is None:
        preset = st["keys"]
        fn = lambda nt, dur, **kw: Y.note(nt, dur, preset, 0.85,
                                          seed=p["seed"] + nt)
    out = np.zeros((C.sec(sec["dur"] + 2.5), 2), dtype=np.float32)
    bar_dur = 4 * p["spb"]
    voic = T.voicing(sec["chords"], 52, 77)
    for bar, notes in enumerate(voic):
        pat = T.rhythm("ballad" if sec["energy"] < 0.6 else "syncop", bar,
                       seed=p["seed"] + bar)
        for (off, durn, vel) in pat:
            for i, nt in enumerate(notes):
                dur = durn * p["spb"] * 1.8
                key = ("keys", st["keys"], nt, round(dur, 2))
                y = _stack(cache, key, lambda: fn(nt, dur, dyn=0.9, seed=p["seed"] + nt))
                C.mix_at(out, y, bar * bar_dur + off * p["spb"] + i * 0.007,
                         gain=0.5 * vel * (0.65 if i else 1.0),
                         p=float((i - len(notes) / 2) * 0.11))
    if st.get("bend"):
        out = C.bend(out, semis=0.75, rate=2.2)
    return M.stem_fx(out, hp=45.0, lp=11000.0, comp=(-18.0, 1.8, 0.02, 0.2), sat=0.06,
                     width=1.08, gain_db=-7.0)


def render_arp(p: dict, sec: dict, cache: Cache) -> np.ndarray:
    st = p["st"]
    if not st.get("arp"):
        return None
    if st.get("drop_break") and sec["name"] == "bridge" and sec["energy"] < 0.45:
        return None          # breakdown: pad + riser only
    if st["arp"] == "kalimba":
        fn = lambda nt, dur, **k: W.kalimba(nt, dur, **k)
    else:
        fn = lambda nt, dur, **k: Y.note(nt, dur, st["arp"], **k)
    out = np.zeros((C.sec(sec["dur"] + 2.5), 2), dtype=np.float32)
    bar_dur = 4 * p["spb"]
    step = p["spb"] / 4.0
    voic = T.voicing(sec["chords"], 67, 86)
    for bar, notes in enumerate(voic):
        pool = notes + [notes[0] + 12, notes[-1] + 12]
        for s in range(16):
            if sec["energy"] < 0.5 and s % 2:
                continue
            if st["arp"] == "stab" and s % 2 == 0:
                continue          # rave stabs live on the offbeat
            nt = pool[s % len(pool)]
            dur = step * 1.6
            key = ("arp", st["arp"], nt, round(dur, 3))
            y = _stack(cache, key, lambda: fn(nt, dur, seed=p["seed"] + nt))
            vel = 0.55 + 0.35 * (1.0 if s % 4 == 0 else 0.6)
            C.mix_at(out, y, bar * bar_dur + s * step, gain=0.42 * vel,
                     p=float(0.45 * math.sin(s * 0.9)))
    return M.stem_fx(out, hp=380.0, lp=6200.0, comp=(-20.0, 2.0, 0.02, 0.2),
                     sat=0.05, width=1.10, gain_db=-11.0)


def render_lead(p: dict, sec: dict, cache: Cache) -> np.ndarray:
    st = p["st"]
    if not st.get("lead") or sec["energy"] < 0.6:
        return None
    out = np.zeros((C.sec(sec["dur"] + 2.5), 2), dtype=np.float32)
    bar_dur = 4 * p["spb"]
    motif = sec["motif"]
    n_bars = sec["bars"]
    notes_per_bar = max(1, len(motif) // 4)
    for bar in range(n_bars):
        for k in range(notes_per_bar):
            i = (bar * notes_per_bar + k) % len(motif)
            nt = motif[i]
            if nt is None:
                continue
            off = (i % notes_per_bar) * (bar_dur / notes_per_bar)
            dur = (bar_dur / notes_per_bar) * 0.9
            if st["lead"] == "flute":
                key = ("lead_flute", nt, round(dur, 2))
                y = _stack(cache, key, lambda: W.flute(nt, dur, seed=p["seed"] + nt))
                gain = 0.6
            elif st["lead"] == "choir_lead":
                key = ("lead_choir", nt, round(dur, 2))
                y = _stack(cache, key, lambda: V.vowel(nt, dur, "ah", seed=p["seed"] + nt))
                gain = 0.7
            else:
                key = ("lead", st["lead"], nt, round(dur, 2))
                y = _stack(cache, key, lambda: Y.note(nt, dur, st["lead"], 0.95,
                                                      seed=p["seed"] + nt))
                gain = 0.5
            C.mix_at(out, y, bar * bar_dur + off, gain=gain, p=0.12)
    if st.get("stutter"):
        out = C.stutter(out, p["spb"] / 4.0, seed=p["seed"] + sec["index"] + 91,
                       drop=0.14, chop=0.12)
    return M.stem_fx(out, hp=200.0, lp=11000.0, comp=(-16.0, 2.4, 0.015, 0.15), sat=0.1,
                     width=1.05, gain_db=-5.5)


def render_strings(p: dict, sec: dict, cache: Cache) -> np.ndarray:
    st = p["st"]
    if not st["strings"] or sec["energy"] < 0.35:
        return None
    out = np.zeros((C.sec(sec["dur"] + 3.0), 2), dtype=np.float32)
    bar_dur = 4 * p["spb"]
    voic = T.voicing(sec["chords"], 57, 84)
    roots = T.bass_line(sec["chords"], 45, 57)
    for bar, (notes, root) in enumerate(zip(voic, roots)):
        full = sorted(set(list(notes) + [root]))
        style = "legato"
        if sec["energy"] > 0.8 and bar % 2 == 1:
            style = "spiccato"
        key = ("strings", tuple(full), round(bar_dur, 2), style, bar % 2)
        y = _stack(cache, key, lambda: _maybe_sample(
            "violin_section", full, bar_dur * 1.02, p["seed"] + bar,
            lambda: S.section(full, bar_dur * 1.02,
                              size=8 if sec["energy"] > 0.7 else 5,
                              style=style, spread=0.75, seed=p["seed"] + bar)))
        C.mix_at(out, y, bar * bar_dur, gain=0.5 + 0.35 * sec["energy"], p=0.0)
    return M.stem_fx(out, hp=105.0, lp=8000.0, comp=(-19.0, 1.6, 0.04, 0.25),
                     sat=0.04, width=1.10, gain_db=-3.0)


def render_choir(p: dict, sec: dict, cache: Cache) -> np.ndarray:
    st = p["st"]
    if not st["choir"] or sec["energy"] < 0.6:
        return None
    out = np.zeros((C.sec(sec["dur"] + 3.0), 2), dtype=np.float32)
    bar_dur = 4 * p["spb"]
    voic = T.voicing(sec["chords"], 60, 79)
    for bar, notes in enumerate(voic):
        key = ("choir", tuple(notes), round(bar_dur, 2), bar % 2)
        y = _stack(cache, key, lambda: _maybe_sample(
            "choir_ah", notes, bar_dur, p["seed"] + bar,
            lambda: V.choir(notes, bar_dur, vowel_="ah", size=8, spread=0.85,
                            seed=p["seed"] + bar)))
        C.mix_at(out, y, bar * bar_dur, gain=0.35 + 0.3 * sec["energy"], p=0.0)
    return M.stem_fx(out, hp=170.0, lp=9000.0, comp=(-20.0, 1.8, 0.03, 0.2), sat=0.05,
                     width=1.10, gain_db=-4.5)


def render_bells(p: dict, sec: dict, cache: Cache) -> np.ndarray:
    st = p["st"]
    if not st.get("bells"):
        return None
    out = np.zeros((C.sec(sec["dur"] + 3.0), 2), dtype=np.float32)
    bar_dur = 4 * p["spb"]
    g = np.random.default_rng(p["seed"] * 31 + sec["index"])
    voic = T.voicing(sec["chords"], 72, 96)
    for bar, notes in enumerate(voic):
        for s in range(2):
            if g.random() > 0.5 + 0.3 * sec["energy"]:
                continue
            nt = notes[int(g.integers(0, len(notes)))]
            key = ("bell", nt)
            y = _stack(cache, key, lambda: K.bell(nt, 2.0, kind="glock",
                                                  seed=p["seed"] + nt))
            C.mix_at(out, y, bar * bar_dur + float(g.uniform(0, 3.0)),
                     gain=0.22 + 0.2 * sec["energy"], p=float(g.uniform(-0.7, 0.7)))
    return M.stem_fx(out, hp=700.0, sat=0.03, width=1.10, gain_db=-12.0)


def render_guitar(p: dict, sec: dict, cache: Cache) -> np.ndarray:
    if p["style"] != "rock-anthem" or sec["energy"] < 0.45:
        return None
    out = np.zeros((C.sec(sec["dur"] + 3.0), 2), dtype=np.float32)
    bar_dur = 4 * p["spb"]
    roots = T.bass_line(sec["chords"], 40, 52)
    for bar, root in enumerate(roots):
        y = G.power_chords([root] * 4, bpm=p["bpm"], gain=0.75, seed=p["seed"] + bar)
        C.mix_at(out, y, bar * bar_dur, gain=0.5 + 0.2 * sec["energy"], p=-0.25)
        y2 = G.power_chords([root] * 4, bpm=p["bpm"], gain=0.6, seed=p["seed"] + bar + 99)
        C.mix_at(out, y2, bar * bar_dur, gain=0.45 + 0.2 * sec["energy"], p=0.25)
    return M.stem_fx(out, hp=105.0, lp=6500.0, comp=(-14.0, 2.5, 0.01, 0.1),
                     sat=0.18, width=1.10, gain_db=-6.0)


def render_perc(p: dict, sec: dict, cache: Cache) -> np.ndarray:
    st = p["st"]
    if not st.get("perc") or sec["energy"] < 0.45:
        return None
    out = np.zeros((C.sec(sec["dur"] + 1.5), 2), dtype=np.float32)
    bar_dur = 4 * p["spb"]
    for bar in range(sec["bars"]):
        for s in range(8):
            key = ("perc", st["perc"], s % 2)
            y = _stack(cache, key, lambda: D.hit(st["perc"], level=1.0))
            vel = 0.9 if s % 2 == 0 else 0.55
            C.mix_at(out, y, bar * bar_dur + s * p["spb"] / 2.0, gain=vel,
                     p=float(0.3 * (1 if s % 2 == 0 else -1)))
    return M.stem_fx(out, hp=2500.0, width=1.10, gain_db=8.0)


def render_fx(p: dict, sec: dict, cache: Cache) -> np.ndarray:
    out = np.zeros((C.sec(sec["dur"] + 3.0), 2), dtype=np.float32)
    if sec["name"] == "pre":
        C.mix_at(out, riser(sec["dur"] * 0.5, seed=p["seed"]), sec["dur"] * 0.5,
                 gain=0.5, p=0.0)
    if sec["index"] == 0:
        C.mix_at(out, reverse_sweep(2.0, seed=p["seed"]), 0.0, gain=0.35, p=0.0)
    if sec["name"] == "chorus" and sec["energy"] > 0.8:
        C.mix_at(out, impact(seed=p["seed"] + sec["index"]), 0.0, gain=0.5, p=0.0)
        C.mix_at(out, D.hit("crash", decay=2.8, level=1.0), 0.0, gain=0.5, p=0.1)
    if sec["name"] == "outro":
        C.mix_at(out, downlifter(1.6, seed=p["seed"]), 0.0, gain=0.4, p=0.0)
    return M.stem_fx(out, hp=40.0, gain_db=4.0)


# --------------------------------------------------------------------------
# the mix
# --------------------------------------------------------------------------

IRS = {"hall": dict(seconds=3.4, decay=1.5, size=1.0, damping=0.5, brightness=0.6),
       "room": dict(seconds=1.3, decay=2.6, size=0.5, damping=0.7, brightness=0.5),
       "plate": dict(seconds=2.2, decay=1.8, size=0.6, damping=0.35, brightness=0.85)}

LAYERS = [("drums", render_drums, 0.0), ("bass", render_bass, 0.0),
          ("pad", render_pad, 0.22), ("keys", render_keys, 0.18),
          ("arp", render_arp, 0.24), ("lead", render_lead, 0.22),
          ("strings", render_strings, 0.28), ("choir", render_choir, 0.28),
          ("guitar", render_guitar, 0.12), ("bells", render_bells, 0.32),
          ("perc", render_perc, 0.14), ("fx", render_fx, 0.22)]


def automation(p: dict, n: int, depth: float = -11.0,
               exp: float = 1.25) -> np.ndarray:
    """macrodynamics: quiet sections are actually quieter.

    Without this the compressor flattens verse and chorus into the same
    loudness, which is the single most obvious tell of generated music.
    """
    g = np.ones(n, dtype=np.float32)
    fade = max(1, C.sec(0.6))
    for sec in p["sections"]:
        db = depth * (1.0 - float(sec["energy"])) ** exp
        i0, i1 = C.sec(sec["start"]), C.sec(sec["start"] + sec["dur"])
        i0, i1 = min(i0, n), min(i1, n)
        if i1 <= i0:
            continue
        prev = g[i0 - 1] if i0 > 0 else np.float32(10 ** (db / 20.0))
        target = np.float32(10 ** (db / 20.0))
        k = min(fade, i1 - i0)
        if k > 1:
            g[i0:i0 + k] = np.linspace(float(prev), float(target), k).astype(np.float32)
        if i1 - i0 - k > 0:
            g[i0 + k:i1] = target
    # smooth so the level moves like a fader, not a switch (running mean,
    # kept in float32 -- np.convolve would promote this to 67 MB of float64)
    c = np.cumsum(g, dtype=np.float64)
    sm = np.empty(n, dtype=np.float32)
    half = fade // 2
    for i in range(0, n, 1 << 16):
        j = min(n, i + (1 << 16))
        a = np.clip(np.arange(i, j) - half, 0, n - 1).astype(np.int64)
        b = np.clip(np.arange(i, j) + half, 0, n - 1).astype(np.int64)
        sm[i:j] = ((c[b] - c[a]) / np.maximum(1, b - a)).astype(np.float32)
    return sm


def _mem(tag):
    import os, resource
    if os.environ.get("SONORA_MEMTRACE"):
        print(f"      [{tag}] rss={resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024:.0f}MB",
              flush=True)


def render(p: dict, progress=None) -> np.ndarray:
    st = p["st"]
    total = p["total"]
    mix = np.zeros((C.sec(total + 5.0), 2), dtype=np.float32)
    ir = C.make_ir(seed=p["seed"] + 3, **IRS.get(st["ir"], IRS["hall"]))
    send_bus = np.zeros_like(mix)
    cache = Cache()

    for i, sec in enumerate(p["sections"]):
        if progress:
            progress(f"[{i+1}/{len(p['sections'])}] {sec['name']} (energy {sec['energy']:.2f})")
        t0 = C.sec(sec["start"])
        for lname, fn, send in LAYERS:
            y = fn(p, sec, cache)
            if y is None:
                continue
            n = min(y.shape[0], mix.shape[0] - t0)
            if n <= 0:
                continue
            gdb = st.get("mix", {}).get(lname, 0.0)
            if gdb:
                y = (y * np.float32(10.0 ** (gdb / 20.0))).astype(np.float32)
            mix[t0:t0 + n] += y[:n]
            if send > 0.001:
                send_bus[t0:t0 + n] += C.send_reverb(y[:n], ir, send)
            del y

    mix += send_bus
    del send_bus
    cache.d.clear()          # rendered stems can be hundreds of MB
    _mem("stems mixed")

    # macrodynamics.  Most styles set them before the bus so the glue works
    # with them; dance styles (post_auto) ride the fader after the master so
    # the limiter cannot squash the breakdown back up to the level of the drop.
    post = bool(st.get("post_auto"))
    if not post:
        a = automation(p, mix.shape[0])
        _mem("automation")
        mix = (mix.astype(np.float32) * a[:, None])

    # sidechain: duck everything above 120 Hz against the kick
    if st["sidechain"] > 0.1:
        times = []
        for sec in p["sections"]:
            if sec["energy"] < 0.5:
                continue
            for b in range(sec["bars"] * 4):
                times.append(sec["start"] + b * p["spb"])
        sub, high = C.crossover(mix, 120.0)
        _mem("crossover")
        # the ducking gain only depends on the kick times, so build it once
        # and apply it in place instead of allocating another full mix
        g = C.sidechain_gain(times, mix.shape[0], st["sidechain"])
        for i in range(0, mix.shape[0], 1 << 16):
            j = min(mix.shape[0], i + (1 << 16))
            mix[i:j] = sub[i:j] + high[i:j] * g[i:j, None]
        del sub, high, g
        _mem("sidechain")

    import os
    if os.environ.get("SONORA_DEBUG"):
        C.write(os.environ["SONORA_DEBUG"], mix)
    mix = M.master(mix, target_lufs=st["lufs"], width=st["width"],
                  glue=st.get("glue", 1.0), air=st.get("air", 1.0),
                  punch=st.get("punch", 1.0), tilt=st.get("tilt", 0.0))
    if post:
        a = automation(p, mix.shape[0], depth=st.get("auto_depth", -11.5), exp=1.15)
        for i in range(0, mix.shape[0], 1 << 18):
            j = min(mix.shape[0], i + (1 << 18))
            mix[i:j] *= a[i:j, None]
        del a
    _mem("master")
    end = C.sec(total + 2.2)
    return C.fade(mix[:end], 0.02, 1.6)
