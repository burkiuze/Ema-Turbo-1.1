"""Audio helpers: sample-rate conversion from 48 kHz, window by window, and WAV writing. NumPy only.

The resampler carries its state from window to window, so the output never depends on where the decoder's
windows were cut. It is EMA Lightning's resampler (the same linear-phase windowed-sinc low-pass, the same
bookkeeping), with the convolution done as one matrix product over a strided view.
"""
import wave

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

RATE = 48000
RATES = (48000, 24000, 16000, 8000)


class Resampler:
    """48 kHz in, 48 kHz / factor out, through a linear-phase windowed-sinc low-pass."""

    def __init__(self, rate):
        self.factor = RATE // rate
        self.half = 32 * self.factor
        self.taps = None
        self.buffer = None
        self.start = -self.half  # absolute input index of buffer[0]; the signal is preceded by zeros
        self.next = 0  # absolute input index at the centre of the next output sample
        self.end = 0  # absolute input index one past the last real sample

    def push(self, chunk):
        if self.factor == 1:
            return chunk
        if self.buffer is None:
            self.buffer = np.zeros(self.half, np.float32)
            n = np.arange(-self.half, self.half + 1, dtype=np.float64)
            cutoff = 0.45 / self.factor
            taps = 2 * cutoff * np.sinc(2 * cutoff * n) * np.blackman(2 * self.half + 1)
            self.taps = (taps / taps.sum()).astype(np.float32)
        self.buffer = np.concatenate([self.buffer, chunk.astype(np.float32, copy=False)])
        self.end += chunk.size
        return self._emit(self.end - 1 - self.half)

    def flush(self):
        if self.factor == 1 or self.buffer is None:
            return None
        self.buffer = np.concatenate([self.buffer, np.zeros(self.half + self.factor, np.float32)])
        return self._emit(self.end - 1)

    def _emit(self, last):
        """Every output whose centre is at or before input index `last`."""
        if last < self.next:
            return np.zeros(0, np.float32)
        count = (last - self.next) // self.factor + 1
        a = self.next - self.half - self.start
        b = a + (count - 1) * self.factor + 2 * self.half + 1
        frames = sliding_window_view(self.buffer[a:b], self.taps.size)[:: self.factor]
        out = frames @ self.taps  # a cross-correlation, like conv1d; the taps are symmetric anyway
        self.next += count * self.factor
        drop = self.next - self.half - self.start
        self.buffer, self.start = self.buffer[drop:], self.start + drop
        return out.astype(np.float32, copy=False)


def write_wav(path, audio, rate):
    pcm = (np.clip(audio, -1.0, 1.0) * 32767.0).round().astype("<i2")
    with wave.open(path if hasattr(path, "write") else str(path), "wb") as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(rate)
        f.writeframes(pcm.tobytes())


def wav_bytes(audio, rate):
    """A WAV file in memory, for web apps."""
    import io

    buf = io.BytesIO()
    write_wav(buf, audio, rate)
    return buf.getvalue()
