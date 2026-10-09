import time
import wave

import numpy as np
import pytest
from conftest import FIXTURES, same_speech

from ema_turbo import EMA, Speech
from ema_turbo.runtime import MissingModels
from ema_turbo.scheduler import DONE

TEXT = "Bugün hava çok güzel. Yarın da yağmur yağacakmış, toplantı öğleden sonra başlayacak."
LONG = " ".join(["Bu cümle yaklaşık olarak dört saniye sürecek kadar uzun bir cümledir."] * 5)


def test_say_returns_speech(tts):
    s = tts.say(TEXT, seed=7)
    assert isinstance(s, Speech) and s.seed == 7 and s.sample_rate == 48000
    assert s.audio.dtype == np.float32 and s.audio.ndim == 1 and s.audio.size > 0
    assert s.duration == pytest.approx(s.audio.size / 48000)
    assert np.abs(s.audio).max() <= 1.0


@pytest.mark.parametrize("rate", [48000, 24000, 16000, 8000])
def test_every_sample_rate_gives_the_same_speech(tts, rate):
    full = tts.say(LONG, seed=3)
    s = tts.say(LONG, seed=3, sample_rate=rate)
    assert s.sample_rate == rate and s.audio.dtype == np.float32
    assert s.duration == pytest.approx(full.duration, abs=0.01)


@pytest.mark.parametrize("rate", [48000, 24000, 16000, 8000])
def test_stream_joined_matches_say(tts, rate):
    chunks = list(tts.stream(LONG, seed=3, sample_rate=rate))
    assert len(chunks) > 2 and all(c.dtype == np.float32 for c in chunks)
    same_speech(np.concatenate(chunks), tts.say(LONG, seed=3, sample_rate=rate).audio)


def test_first_chunk_of_a_stream_is_a_fifth_of_a_second(tts):
    first = next(iter(tts.stream(LONG, seed=0)))
    assert first.size == 5 * 1920


def _raw(request):
    chunks = []
    while (item := request.outbox.get(timeout=30)) is not DONE:
        chunks.append(item)
    return chunks


def test_say_decodes_a_full_window_from_the_start(tts):
    request = tts._submit(LONG, 1.0, 0)
    chunks = _raw(request)
    assert chunks[0].size == min(request.pieces[0].frames, tts._engine.window) * 1920


def test_a_streams_first_window_is_decoded_alone(tts):
    engine, calls = tts._engine, []
    real = engine.decode

    def spy(items):
        calls.append([span for _, span in items])
        return real(items)

    engine.decode = spy
    try:
        list(tts.stream(LONG, seed=0))
    finally:
        engine.decode = real
    assert calls[0] == [(0, 5)]


def test_leaving_a_stream_early_drops_its_work(tts):
    chunks = tts.stream(" ".join([LONG] * 4), seed=0)
    assert next(chunks).size > 0
    chunks.close()
    deadline = time.time() + 10
    while (tts._playhead.sentences or tts._playhead.windows) and time.time() < deadline:
        time.sleep(0.05)
    assert not tts._playhead.sentences and not tts._playhead.windows
    assert tts.say(TEXT, seed=0).audio.size > 0


def test_seed_decides_the_audio(tts):
    a, b, c = tts.say(TEXT, seed=1), tts.say(TEXT, seed=1), tts.say(TEXT, seed=2)
    assert np.array_equal(a.audio, b.audio)
    assert not np.array_equal(a.audio, c.audio)
    assert isinstance(tts.say(TEXT).seed, int)


def test_list_matches_saying_each_text(tts):
    texts = [TEXT, "Evet.", LONG, "", "Kısa bir cümle daha."]
    many = tts.say(texts, seed=5, sample_rate=16000)
    assert [s.seed for s in many] == [5] * len(texts)
    for text, s in zip(texts, many, strict=True):
        same_speech(s.audio, tts.say(text, seed=5, sample_rate=16000).audio)
    assert tts.say([]) == []


def test_unspeakable_text_is_silent_not_an_error(tts):
    for text in ["", "   ", "...", "?!", "\x00\x01"]:
        assert tts.say(text).audio.size == 0
        assert list(tts.stream(text)) == []


@pytest.mark.parametrize("kwargs", [dict(speed=0), dict(speed=5), dict(speed="fast"), dict(speed=True),
                                    dict(seed=-1), dict(seed=1.5), dict(sample_rate=44100)])
def test_bad_settings_raise_before_any_work(tts, kwargs):
    with pytest.raises(ValueError):
        tts.say(TEXT, **kwargs)
    with pytest.raises(ValueError):
        tts.say([TEXT], **kwargs)
    with pytest.raises(ValueError):
        tts.stream(TEXT, **kwargs)


def test_wrong_text_types_raise(tts):
    with pytest.raises(TypeError):
        tts.stream([TEXT])
    with pytest.raises(TypeError):
        tts.say(123)


def test_path_writes_wav_files(tts, tmp_path):
    s = tts.say(TEXT, seed=0, sample_rate=24000, path=tmp_path / "one.wav")
    with wave.open(str(tmp_path / "one.wav")) as f:
        assert f.getframerate() == 24000 and f.getnframes() == s.audio.size
    tts.say([TEXT] * 11, seed=0, path=tmp_path / "many")
    assert sorted(p.name for p in (tmp_path / "many").iterdir()) == [f"{i:02d}.wav" for i in range(11)]


def test_sentence_cuts_leave_a_pause(tts):
    chunks = _raw(tts._submit(LONG, 1.0, 0))
    assert any(c.size == 12000 and not c.any() for c in chunks)


def test_words_cover_every_spoken_word_in_order(tts):
    s = tts.say(TEXT, seed=0)
    assert [w.text for w in s.words] == tts._frontend(TEXT).split()
    assert s.words[0].start == 0.0
    for a, b in zip(s.words, s.words[1:], strict=False):
        assert a.start < a.end <= b.start
    assert s.words[-1].end == pytest.approx(s.duration, abs=1e-3)


def test_words_skip_the_pause_between_pieces(tts):
    s = tts.say(LONG, seed=0)
    gaps = [b.start - a.end for a, b in zip(s.words, s.words[1:], strict=False)]
    assert max(gaps) == pytest.approx(0.25, abs=1e-3)
    assert s.words[-1].end == pytest.approx(s.duration, abs=1e-3)


def test_low_memory_sounds_the_same(models):
    fast, lean = EMA(models=models), EMA(models=models, low_memory=True)
    assert lean._engine.window < fast._engine.window
    same_speech(lean.say(LONG, seed=2).audio, fast.say(LONG, seed=2).audio, db=60)


def test_warmup_measures_once_and_chains(tts, capsys):
    assert tts.warmup() is tts and tts.lightning() is tts
    assert "EMA Turbo ready" in capsys.readouterr().out
    assert tts.best_batch_size() in (1, 2, 4, 8)


def test_models_are_found_by_argument_or_environment(monkeypatch, tmp_path):
    import ema_turbo.runtime as runtime

    with pytest.raises(MissingModels):
        EMA(models=tmp_path)
    monkeypatch.setenv("EMA_TURBO_MODELS", str(FIXTURES / "tiny_word_pos"))
    assert EMA().models == FIXTURES / "tiny_word_pos"
    monkeypatch.delenv("EMA_TURBO_MODELS")
    monkeypatch.setattr(runtime, "CACHE", tmp_path / "empty")
    monkeypatch.setitem(__import__("sys").modules, "torch", None)  # no PyTorch to export with
    with pytest.raises(MissingModels, match="ema_turbo.export"):
        EMA()
