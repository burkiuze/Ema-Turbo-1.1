"""Live speech: speak text while it is still being written, for voice agents and live conversations.

An LLM writes a reply a few words at a time. Waiting for the whole reply before speaking adds seconds; EMA
Turbo's live mode cuts the text as it arrives, at the first clause or sentence end it can, and starts each piece
the moment it is complete. Every piece goes into the same first-come-first-served queues as any other call, so
the next piece is being made while the one before it plays.

    with tts.live() as live:                      # one per reply
        for token in llm_reply:                   # from any thread
            live.feed(token)
        live.end()
    for chunk in live: ...                        # in another thread: audio in order

    for chunk in tts.stream_text(llm_reply):      # the same, with the feeding done for you
        speaker.write(chunk)

live.cancel() (or leaving the loop) is barge-in: everything not yet spoken is dropped at once.

Where the text is cut. The first piece is cut early, so the first audio comes quickly: at the first clause mark
(, ; :), at a sentence end, or after eight words with no punctuation. Later pieces are
cut at sentence ends, at clause marks after at least four words, or after twenty words. A period after a number,
a single letter or a common abbreviation ("3.", "a.", "Dr.") is not a sentence end. Cuts only happen at a space,
so a word or a number that arrives in parts ("14:" "30'da") is never split. A clause cut is followed by EMA
Lightning's clause pause (0.12 s), a sentence end by its sentence pause (0.25 s).
"""
import queue
import re
import threading

import numpy as np

from .audio import RATE, Resampler
from .chunker import CLAUSE_PAUSE, SENTENCE_PAUSE
from .engine import STREAM_WINDOWS
from .scheduler import DONE

SENTENCE_END = re.compile(r"[.!?…]+[\"'’”)\]]*(?=\s)")
CLAUSE_END = re.compile(r"[,;:](?=\s)")
SPACE = re.compile(r"\s+")
FIRST_CLAUSE_WORDS, CLAUSE_WORDS = 1, 4
FIRST_MAX_WORDS, MAX_WORDS = 8, 20
ABBREVIATIONS = frozenset(
    "dr prof doç doc av sn st vb vs örn no bkz yy mr mrs ms jr sr tel cad sok mah apt ltd şti alb yzb gen müh "
    "uzm op hz".split())


class Segmenter:
    """Text deltas in, speakable pieces out: [(text, pause after it)]. Pure; no threads."""

    def __init__(self):
        self.buffer = ""
        self.count = 0  # pieces cut so far

    def push(self, text):
        self.buffer += text
        out = []
        while (cut := self._next_cut()) is not None:
            end, pause = cut
            piece, self.buffer = self.buffer[:end].strip(), self.buffer[end:]
            if piece:
                out.append((piece, pause))
                self.count += 1
        return out

    def flush(self):
        piece, self.buffer = self.buffer.strip(), ""
        if not piece:
            return []
        self.count += 1
        return [(piece, 0.0)]

    def _next_cut(self):
        """The earliest place the buffer can be cut, or None. It depends only on the text, never on how the text
        was split into deltas, so the same reply is always cut the same way."""
        first = self.count == 0
        min_clause = FIRST_CLAUSE_WORDS if first else CLAUSE_WORDS
        max_words = FIRST_MAX_WORDS if first else MAX_WORDS
        text = self.buffer
        cuts = [(m.end(), SENTENCE_PAUSE) for m in SENTENCE_END.finditer(text)
                if not _abbreviation(text, m.start())]
        cuts += [(m.end(), CLAUSE_PAUSE) for m in CLAUSE_END.finditer(text) if _words(text[:m.end()]) >= min_clause]
        spaces = (m.start() for m in SPACE.finditer(text) if m.start() > 0)
        long = next((s for s in spaces if _words(text[:s]) >= max_words), None)
        if long is not None:
            cuts.append((long, CLAUSE_PAUSE))
        return min(cuts) if cuts else None


def _words(text):
    return len(text.split())


def _abbreviation(text, dot):
    """True when the period at `dot` ends a number, a single letter or a known abbreviation."""
    word = text[:dot].rsplit(None, 1)[-1] if text[:dot].strip() else ""
    word = word.strip("\"'(“‘").lower()
    return word.isdigit() or len(word) == 1 or word in ABBREVIATIONS


class Live:
    """One live reply: feed() text from one thread, iterate audio in another. See the module docstring."""

    def __init__(self, tts, speed, seed, sample_rate):
        self._tts, self._speed, self._seed, self._rate = tts, speed, seed, sample_rate
        self._segmenter = Segmenter()
        self._requests = queue.SimpleQueue()  # (request, pause after it), then DONE
        self._submitted = []
        self._lock = threading.Lock()
        self._ended = False
        self._closed = False

    def feed(self, text):
        """Add text as it arrives (any size: a token, a word, a sentence). Complete pieces start at once."""
        if not isinstance(text, str):
            raise TypeError("feed() takes text")
        with self._lock:
            if self._ended:
                raise RuntimeError("this reply has ended; start a new one with tts.live()")
            if not self._closed:
                self._submit(self._segmenter.push(text))

    def end(self):
        """No more text: speak what is left. Safe to call twice."""
        with self._lock:
            if self._ended:
                return
            self._ended = True
            if not self._closed:
                self._submit(self._segmenter.flush())
            self._requests.put(DONE)

    def cancel(self):
        """Barge-in: stop speaking now and drop every piece not yet spoken. Safe from any thread, also while
        another thread is waiting for the next chunk: that wait ends at once."""
        with self._lock:
            self._closed = True
            for request in self._submitted:
                if not request.closed:
                    request.cancel()
                    request.outbox.put(DONE)  # wake a listener waiting on this piece
            if not self._ended:
                self._ended = True
                self._requests.put(DONE)

    def _submit(self, pieces):
        first = self._segmenter.count - len(pieces)  # the number of this batch's first piece in the reply
        for i, (text, pause) in enumerate(pieces):
            seed = (self._seed * 1_000_003 + 7919 * (first + i)) % 2**63
            request = self._tts._submit(text, self._speed, seed, first=STREAM_WINDOWS)
            self._submitted.append(request)
            self._requests.put((request, pause))

    def __iter__(self):
        resampler, pause = Resampler(self._rate), 0.0
        try:
            while (item := self._requests.get()) is not DONE:
                request, after = item
                if self._closed:
                    break
                if pause:  # the pause after the piece before, now that another piece follows it
                    out = resampler.push(np.zeros(round(pause * RATE), np.float32))
                    if out.size:
                        yield out
                for raw in self._tts._raw(request):
                    out = resampler.push(raw)
                    if out.size:
                        yield out.astype(np.float32, copy=False)
                    if self._closed:
                        return
                pause = after
            if not self._closed:
                tail = resampler.flush()
                if tail is not None and tail.size:
                    yield tail.astype(np.float32, copy=False)
        finally:
            self.cancel()

    def __enter__(self):
        return self

    def __exit__(self, kind, value, trace):
        if kind is not None:
            self.cancel()
        else:
            self.end()
        return False
