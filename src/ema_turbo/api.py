"""EMA Turbo: EMA Lightning's voice, without PyTorch, faster and lighter.

    tts = EMA()
    speech = tts.say("Merhaba, nasılsınız?", path="merhaba.wav")
    speeches = tts.say(["Birinci cümle.", "İkinci cümle."])
    for chunk in tts.stream("Uzun bir metin..."):
        play(chunk)

The same calls, options and results as EMA Lightning's EMA. Based on EMA Lightning's api.py (Apache-2.0).
"""
import json
import os
import random
import statistics
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from . import runtime
from .audio import RATE, RATES, Resampler, write_wav
from .chunker import chunk
from .engine import LOW_MEMORY_WINDOW, STREAM_WINDOWS, Engine, windows
from .frontend import Frontend
from .live import Live
from .scheduler import DONE, Playhead

PROBE = "Bugün hava çok güzel, yarın da yağmur yağacakmış; toplantı öğleden sonra başlayacak."
FPS = 25  # frames per second of the acoustic model
DEFAULT_BATCH = 2  # sentences or windows per call until best_batch_size() has measured this machine
CACHE = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")) / "ema_turbo" / "batch_size_v1.json"


@dataclass(frozen=True)
class Word:
    """One spoken word and when it is heard, in seconds from the start of the audio.

    `text` is the word as it was read aloud, after normalization: "5" comes back as "beş".
    """

    text: str
    start: float
    end: float


@dataclass(frozen=True)
class Speech:
    """One spoken text: float32 audio in [-1, 1] at sample_rate, the seed that made it, and its words' times."""

    audio: np.ndarray = field(repr=False)
    sample_rate: int
    duration: float
    seed: int
    words: tuple[Word, ...] = field(default=(), repr=False)


class EMA:
    def __init__(self, models=None, threads=None, low_memory=False, providers=None):
        """Load the three ONNX models.

        models: a folder with text.onnx, sound.onnx, decoder.onnx and meta.json; by default $EMA_TURBO_MODELS,
        then the cache (exported there once from EMA Lightning's weights if PyTorch is installed).
        threads: ONNX Runtime threads per call (default: one per physical core).
        low_memory: decode at most five seconds at a time and give memory back after every call: about half the
        peak memory, about 10% slower. The audio is the same.
        providers: ONNX Runtime execution providers (default: the CPU).
        """
        path = runtime.find(models)
        meta = runtime.meta(path)
        engine = Engine(runtime.sessions(path, threads, providers, low_memory), meta)
        if low_memory:
            engine.window = LOW_MEMORY_WINDOW
        self._setup(engine, Frontend(meta["vocab"]))
        self.models = path

    @classmethod
    def _from_parts(cls, engine, frontend):
        self = cls.__new__(cls)
        self._setup(engine, frontend)
        self.models = None
        return self

    def _setup(self, engine, frontend):
        self._engine = engine
        self._frontend = frontend
        self._batch_size = None
        self._default_batch = _read_cache().get(f"cpu-{os.cpu_count()}", DEFAULT_BATCH)
        self._playhead = Playhead(self._engine, self._list_batch)

    def warmup(self):
        """Run every model once and measure the batch size, so the first real call is as fast as the rest.
        Prints the first-audio time and the speed of a sentence."""
        start = time.perf_counter()
        self.best_batch_size()
        self.say(PROBE, seed=0)
        first = statistics.median(self._first_audio() for _ in range(3))
        timings = [self._timed_probe() for _ in range(3)]
        ms, seconds = statistics.median(t for t, _ in timings), timings[0][1]
        print(f"EMA Turbo ready in {time.perf_counter() - start:.1f} s: first audio in {first:.0f} ms, "
              f"a {seconds:.1f} s sentence in {ms:.0f} ms ({seconds * 1000 / ms:.1f}x real time), "
              f"batch size {self._batch_size}")
        return self

    lightning = warmup  # EMA Lightning's name for its fast path; EMA Turbo is always on it

    def best_batch_size(self):
        """The smallest batch that reaches 90% of this machine's best throughput, measured once and cached.
        Until it is measured (here or by warmup()), calls use the cached value or 2, so no call waits for it."""
        if self._batch_size:
            return self._batch_size
        key = f"cpu-{os.cpu_count()}"
        cached = _read_cache()
        if key not in cached:
            rates = {size: self._throughput(size) for size in (1, 2, 4, 8)}
            cached[key] = min(s for s, r in rates.items() if r >= 0.9 * max(rates.values()))
            try:
                CACHE.parent.mkdir(parents=True, exist_ok=True)
                CACHE.write_text(json.dumps(cached))
            except OSError:  # a read-only home: measure again next time
                pass
        self._batch_size = cached[key]
        return self._batch_size

    def say(self, text, speed=1.0, seed=None, sample_rate=RATE, path=None):
        """Speech for one text, or a list of Speech for a list of texts (batched)."""
        _check(speed, seed, sample_rate)
        if isinstance(text, str):
            seed = _seed(seed)
            request = self._submit(text, speed, seed)
            speech = _speech(list(self._receive(request, sample_rate)), sample_rate, seed, _words(request.pieces))
            if path is not None:
                write_wav(path, speech.audio, sample_rate)
            return speech
        texts = _texts(text)
        seeds = [_seed(seed) for _ in texts]
        requests = [self._submit(t, speed, s) for t, s in zip(texts, seeds, strict=True)]
        out = [_speech(list(self._receive(r, sample_rate)), sample_rate, s, _words(r.pieces))
               for r, s in zip(requests, seeds, strict=True)]
        if path is not None:
            Path(path).mkdir(parents=True, exist_ok=True)
            width = len(str(max(len(out) - 1, 0)))
            for i, speech in enumerate(out):
                write_wav(Path(path) / f"{i:0{width}d}.wav", speech.audio, sample_rate)
        return out

    def stream(self, text, speed=1.0, seed=None, sample_rate=RATE):
        """float32 audio chunks as they are made: a fifth of a second first, then growing chunks.

        Call it from as many threads as you like; every stream shares the models through the same queues.
        Stopping early (leaving the loop) drops the rest of that text's work.
        """
        _check(speed, seed, sample_rate)
        if not isinstance(text, str):
            raise TypeError("stream() takes one text; for many, call stream() once per text")
        return self._stream(text, speed, _seed(seed), sample_rate)

    def _stream(self, text, speed, seed, sample_rate):
        """A stream's request, submitted when its first chunk is asked for."""
        yield from self._receive(self._submit(text, speed, seed, first=STREAM_WINDOWS), sample_rate)

    def _submit(self, text, speed, seed, first=None):
        pieces = self._pieces(text, speed, seed)
        return self._playhead.submit(pieces, speed) if first is None else self._playhead.submit(pieces, speed, first)

    def live(self, speed=1.0, seed=None, sample_rate=RATE):
        """A live reply: feed() text as it is written (an LLM's tokens), iterate the audio as it is made.
        See live.py. Each piece starts the moment its text is complete."""
        _check(speed, seed, sample_rate)
        return Live(self, speed, _seed(seed), sample_rate)

    def stream_text(self, texts, speed=1.0, seed=None, sample_rate=RATE):
        """Audio chunks for text that arrives in parts (any iterable of strings, such as an LLM's token stream),
        spoken while the rest is still arriving. Leaving the loop early stops everything (barge-in)."""
        live = self.live(speed, seed, sample_rate)
        errors = []

        def feed():
            try:
                for text in texts:
                    if live._closed:
                        return
                    live.feed(text)
                live.end()
            except Exception as error:  # the text source failed: stop speaking, report it to the listener
                errors.append(error)
                live.cancel()

        threading.Thread(target=feed, name="ema-turbo-feed", daemon=True).start()
        yield from live
        if errors:
            raise errors[0]

    def _raw(self, request):
        """48 kHz float32 arrays from a request's outbox. If the caller stops waiting, its work is dropped."""
        try:
            while (item := request.outbox.get()) is not DONE:
                if isinstance(item, BaseException):
                    raise item
                yield item
        finally:
            request.cancel()

    def _receive(self, request, sample_rate):
        """float32 chunks from a request's outbox, resampled. If the caller stops waiting, its work is dropped."""
        resampler = Resampler(sample_rate)
        for item in self._raw(request):
            out = resampler.push(item)
            if out.size:
                yield out.astype(np.float32, copy=False)
        tail = resampler.flush()
        if tail is not None and tail.size:
            yield tail.astype(np.float32, copy=False)

    def _pieces(self, text, speed, seed):
        return [self._engine.piece(piece, pause, (seed * 1_000_003 + i) % 2**63)
                for i, (piece, pause) in enumerate(chunk(self._frontend(text), speed))]

    def _list_batch(self):
        return self._batch_size or self._default_batch

    def _first_audio(self):
        """Milliseconds until a stream of the probe sentence hands over its first chunk."""
        start = time.perf_counter()
        chunks = self._stream(PROBE, 1.0, 0, RATE)
        next(chunks)
        took = 1000 * (time.perf_counter() - start)
        for _ in chunks:  # let it finish, so the next measurement starts on idle models
            pass
        return took

    def _timed_probe(self):
        """Milliseconds for say() to return the probe sentence, and the seconds of audio it made."""
        start = time.perf_counter()
        speech = self.say(PROBE, seed=0)
        return 1000 * (time.perf_counter() - start), speech.duration

    def _throughput(self, size):
        """Seconds of audio per second for a batch of `size` copies of the probe sentence."""
        (piece,) = self._pieces(PROBE, 1.0, 0)
        pieces = [self._engine.piece(piece.text, 0.0, i) for i in range(size)]
        best = float("inf")
        with self._engine.lock:
            self._engine.plan(pieces, 1.0)
            for _ in range(2):
                start = time.perf_counter()
                self._engine.think(pieces)
                for span in windows(pieces[0].frames, self._engine.window, self._engine.window):
                    self._engine.decode([(p, span) for p in pieces])
                best = min(best, time.perf_counter() - start)
        return size * pieces[0].frames / FPS / best


def _read_cache():
    try:
        return json.loads(CACHE.read_text()) if CACHE.exists() else {}
    except (OSError, ValueError):
        return {}


def _check(speed, seed, sample_rate):
    if isinstance(speed, bool) or not isinstance(speed, (int, float)) or not 0.25 <= speed <= 4:
        raise ValueError("speed must be a number from 0.25 to 4")
    if seed is not None and (isinstance(seed, bool) or not isinstance(seed, int) or seed < 0):
        raise ValueError("seed must be a non-negative integer")
    if sample_rate not in RATES:
        raise ValueError(f"sample_rate must be one of {RATES}")


def _texts(text):
    if not isinstance(text, (list, tuple)) or not all(isinstance(t, str) for t in text):
        raise TypeError("text must be a string or a list of strings")
    return list(text)


def _seed(seed):
    return random.SystemRandom().randrange(2**31) if seed is None else seed


def _speech(chunks, sample_rate, seed, words=()):
    audio = np.concatenate(chunks).astype(np.float32) if chunks else np.zeros(0, np.float32)
    return Speech(audio, sample_rate, len(audio) / sample_rate, seed, words)


def _words(pieces):
    """Every word's start and end, read off the frame plan (one frame = 1/25 s) the audio was made from."""
    words, offset = [], 0.0
    for p in pieces:
        if p.fw is None:  # never planned: nothing was spoken
            continue
        fw = p.fw.tolist()
        first, last = {}, {}
        for frame, w in enumerate(fw):
            first.setdefault(w, frame)
            last[w] = frame
        for w, text in enumerate(p.text.split()):
            if w in first:
                words.append(Word(text, round(offset + first[w] / FPS, 3), round(offset + (last[w] + 1) / FPS, 3)))
        # A piece's audio is its frames plus the pause the scheduler inserts after it
        # (scheduler.py emits round(pause * RATE) silent samples); keep the two in step.
        offset += len(fw) / FPS + round(p.pause * RATE) / RATE
    return tuple(words)
