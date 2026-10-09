import math
import wave

import numpy as np
import pytest

from ema_turbo.audio import Resampler, wav_bytes, write_wav


def run(resampler, signal, sizes):
    parts, i = [], 0
    for n in sizes:
        parts.append(resampler.push(signal[i:i + n]))
        i += n
    parts.append(resampler.push(signal[i:]))
    tail = resampler.flush()
    return np.concatenate(parts + ([tail] if tail is not None else []))


def test_48k_passes_through_untouched():
    x = np.random.default_rng(0).standard_normal(1000).astype(np.float32)
    assert np.array_equal(run(Resampler(48000), x, [300, 300]), x)


@pytest.mark.parametrize("rate", [24000, 16000, 8000])
def test_output_never_depends_on_where_the_input_was_cut(rate):
    x = np.random.default_rng(0).standard_normal(48000).astype(np.float32)
    whole = run(Resampler(rate), x, [])
    cut = run(Resampler(rate), x, [1, 7, 1000, 4096, 13, 20000])
    assert whole.size == math.ceil(48000 / (48000 // rate))
    assert np.allclose(whole, cut, atol=1e-6)


def test_keeps_speech_band_and_removes_aliasing():
    t = np.arange(48000) / 48000
    low = run(Resampler(8000), np.sin(2 * np.pi * 1000 * t).astype(np.float32), [])
    high = run(Resampler(8000), np.sin(2 * np.pi * 6000 * t).astype(np.float32), [])
    assert 0.95 < np.abs(low[400:-400]).max() < 1.05
    assert np.abs(high[400:-400]).max() < 0.01


@pytest.mark.parametrize("rate", [24000, 16000, 8000])
def test_matches_ema_lightnings_resampler(rate):
    torch = pytest.importorskip("torch")
    lightning = pytest.importorskip("ema_lightning.audio")
    x = np.random.default_rng(1).standard_normal(96000).astype(np.float32)
    a, b = lightning.Resampler(rate), Resampler(rate)
    want, got = [], []
    for s, e in ((0, 1000), (1000, 50000), (50000, 96000)):
        want.append(a.push(torch.from_numpy(x[s:e])).numpy())
        got.append(b.push(x[s:e]))
    want.append(a.flush().numpy())
    got.append(b.flush())
    assert np.abs(np.concatenate(want) - np.concatenate(got)).max() < 1e-5


def test_write_wav_round_trip(tmp_path):
    audio = np.sin(np.linspace(0, 100, 1600)).astype(np.float32)
    write_wav(tmp_path / "a.wav", audio, 16000)
    with wave.open(str(tmp_path / "a.wav")) as f:
        assert (f.getframerate(), f.getnchannels(), f.getsampwidth(), f.getnframes()) == (16000, 1, 2, 1600)
    assert wav_bytes(audio, 16000) == (tmp_path / "a.wav").read_bytes()
