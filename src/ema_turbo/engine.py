"""The speech engine: pieces of spoken text in, 48 kHz audio out, one decoded window at a time. NumPy only.

Every piece goes through EMA Lightning's three stages: plan (the text stage and its frame timeline), think
(the aligner and the four sampling steps) and decode (one window of latents to audio). Each stage takes any
batch and runs as one ONNX Runtime call; Playhead (scheduler.py) decides which work runs in each batch. The
arithmetic around the models (rounding durations to frames, the frame timeline, the noise) is EMA
Lightning's, step for step, so a seed gives the same audio.
"""
import threading
from dataclasses import dataclass
from itertools import pairwise

import numpy as np

from .rng import randn

RATE = 48000
FIRST_WINDOW = 25  # EMA Lightning's first stream window, one second
# A stream's windows grow: 0.2 s, 0.8 s, 2 s, then full windows. The first audio needs only a fifth of a second
# decoded, and each next window is decoded long before the audio before it has played.
STREAM_WINDOWS = (5, 20, 50)
WINDOW = 250  # ten seconds for every later window: a sentence is usually decoded in one go
LOW_MEMORY_WINDOW = 125  # five seconds: the decoder's activations are its memory peak
CONTEXT = 8  # frames decoded on each side of a window; the decoder reaches 4
MAX_WORD_FRAMES = 250
MAX_FRAMES = 3000


@dataclass(eq=False)
class Piece:
    text: str
    pause: float
    seed: int
    ids: np.ndarray
    cw: np.ndarray  # word of each letter
    wstart: np.ndarray  # first letter of that word
    h: np.ndarray = None
    dur: np.ndarray = None
    fw: np.ndarray = None  # word of each frame
    fp: np.ndarray = None  # position of each frame inside its word
    latents: np.ndarray = None
    spans: list = None  # its decoder windows, fixed when it is planned

    @property
    def letters(self):
        return self.ids.size

    @property
    def frames(self):
        return self.fw.size


def windows(frames, first=WINDOW, window=WINDOW):
    """(start, end) of every decoded window: `first` frames first (or each size of a tuple in turn), then `window`
    frames each."""
    sizes = list(first) if isinstance(first, (tuple, list)) else [first]
    spans, s = [], 0
    while s < frames:
        e = min(frames, s + (sizes.pop(0) if sizes else window))
        spans.append((s, e))
        s = e
    return spans


def batches(items, size):
    for i in range(0, len(items), size):
        yield items[i:i + size]


class Engine:
    def __init__(self, sessions, meta):
        self.sessions = sessions
        self.vocab = list(meta["vocab"])
        self.stoi = {ch: i for i, ch in enumerate(self.vocab)}
        self.steps = len(meta["times"])
        self.latent_dim = int(meta["latent_dim"])
        self.hop = int(meta["hop"])
        self.window = WINDOW
        self.decode_sizes = ()  # no padded decoder sizes on a CPU
        self.group_windows = True  # but a decode batch still holds windows of one length (scheduler.py)
        self.lock = threading.RLock()

    def piece(self, text, pause, seed):
        ids = [self.stoi.get(ch, 1) for ch in text]
        starts = [i for i, ch in enumerate(text) if ch != " " and (i == 0 or text[i - 1] == " ")] or [0]
        bounds = [0] + starts[1:] + [len(text)]
        cw, wstart = [], []
        for w, (a, b) in enumerate(pairwise(bounds)):
            cw += [w] * (b - a)
            wstart += [a] * (b - a)
        return Piece(text, pause, seed, np.array(ids, np.int64), np.array(cw, np.int64), np.array(wstart, np.int64))

    def plan(self, pieces, speed=1.0):
        """Durations for a batch, then each piece's exact frame timeline."""
        L = max(p.letters for p in pieces)
        ids = self.stack([p.ids for p in pieces], L, 0)
        h, dur = self.sessions["text"].run(None, {"ids": ids, "mask": ids != 0})
        dur = dur / np.float32(speed)
        for i, p in enumerate(pieces):
            d = dur[i, :p.letters]
            n_words = int(p.cw[-1]) + 1
            sums = np.zeros(n_words, np.float32)
            np.add.at(sums, p.cw, d)
            n = np.clip(np.round(sums), 1, MAX_WORD_FRAMES).astype(np.int64)
            frames = min(int(n.sum()), MAX_FRAMES)
            fw = np.repeat(np.arange(n_words), n)[:frames]
            fp = ((np.arange(frames) - (np.cumsum(n) - n)[fw]) / n[fw].astype(np.float64)).astype(np.float32)
            p.h, p.dur, p.fw, p.fp = h[i, :p.letters].copy(), d.copy(), fw, fp

    def think(self, pieces):
        """Latents for a batch, each piece from its own seeded noise."""
        L, T = max(p.letters for p in pieces), max(p.frames for p in pieces)
        latents = self.sessions["sound"].run(None, {
            "h": self.stack([p.h for p in pieces], L, 0.0),
            "dur": self.stack([p.dur for p in pieces], L, 0.0),
            "mask": self.stack([np.ones(p.letters, bool) for p in pieces], L, False),
            "cw": self.stack([p.cw for p in pieces], L, -1),
            "wstart": self.stack([p.wstart for p in pieces], L, 0),
            "fw": self.stack([p.fw for p in pieces], T, -1),
            "fp": self.stack([p.fp for p in pieces], T, 0.0),
            "fmask": self.stack([np.ones(p.frames, bool) for p in pieces], T, False),
            "noise": self.stack([self.noise(p) for p in pieces], T, 0.0, axis=1),
        })[0]
        for i, p in enumerate(pieces):
            p.latents = latents[i, :p.frames].copy()

    def decode_size(self, piece, span):
        s, e = span
        return min(piece.frames, e + CONTEXT) - max(0, s - CONTEXT)

    def decode(self, items):
        """Audio for a batch of (piece, window), each cut to its own window. Windows of one length share a call;
        nothing is padded, so the decoder needs no masks."""
        out = [None] * len(items)
        spans = [(max(0, s - CONTEXT), min(p.frames, e + CONTEXT)) for p, (s, e) in items]
        by_length = {}
        for i, (a, b) in enumerate(spans):
            by_length.setdefault(b - a, []).append(i)
        for group in by_length.values():
            z = np.stack([items[i][0].latents[spans[i][0]:spans[i][1]] for i in group])
            audio = self.sessions["decoder"].run(None, {"z": z})[0]
            for row, i in enumerate(group):
                (s, e), (a, _) = items[i][1], spans[i]
                out[i] = audio[row, (s - a) * self.hop:(e - a) * self.hop].copy()
        return out

    def noise(self, piece):
        return randn(piece.seed, (self.steps, piece.frames, self.latent_dim))

    @staticmethod
    def stack(rows, size, fill, axis=0):
        shape = list(rows[0].shape)
        shape[axis] = size
        out = np.full([len(rows)] + shape, fill, dtype=rows[0].dtype)
        for i, row in enumerate(rows):
            index = [i] + [slice(None)] * row.ndim
            index[axis + 1] = slice(0, row.shape[axis])
            out[tuple(index)] = row
        return out
