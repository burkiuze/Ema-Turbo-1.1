import numpy as np

from ema_turbo.engine import CONTEXT, FIRST_WINDOW, STREAM_WINDOWS, windows
from ema_turbo.rng import randn

TEXT = "bugün hava çok güzel, yarın da yağmur yağacakmış. toplantı öğleden sonra başlayacak."


def test_windows_layout():
    assert windows(0) == []
    assert windows(10) == [(0, 10)]
    assert windows(300) == [(0, 250), (250, 300)]  # say(): ten seconds each
    assert windows(126, FIRST_WINDOW) == [(0, 25), (25, 126)]  # a stream's first sentence
    assert windows(126, FIRST_WINDOW, 100) == [(0, 25), (25, 125), (125, 126)]
    assert windows(400, STREAM_WINDOWS) == [(0, 5), (5, 25), (25, 75), (75, 325), (325, 400)]  # a stream: growing
    assert windows(12, STREAM_WINDOWS) == [(0, 5), (5, 12)]


def test_piece_marks_words_like_the_release_model(tts):
    p = tts._engine.piece("merhaba dünya.", 0.0, 0)
    assert p.cw.tolist() == [0] * 8 + [1] * 6
    assert p.wstart.tolist() == [0] * 8 + [8] * 6


def test_plan_builds_an_exact_timeline(tts):
    engine = tts._engine
    p = engine.piece(TEXT, 0.0, 0)
    engine.plan([p], 1.0)
    assert p.frames == p.fp.size > 0
    assert (p.fw[1:] >= p.fw[:-1]).all()
    assert ((p.fp >= 0) & (p.fp < 1)).all()
    assert p.fp.dtype == np.float32 and p.fw.dtype == np.int64


def test_slower_speech_has_more_frames(tts):
    engine = tts._engine
    a, b = engine.piece(TEXT, 0.0, 0), engine.piece(TEXT, 0.0, 0)
    engine.plan([a], 1.0)
    engine.plan([b], 0.5)
    assert b.frames > 1.5 * a.frames


def run(engine, pieces, speed=1.0):
    engine.plan(pieces, speed)
    engine.think(pieces)
    return [np.concatenate(engine.decode([(p, span) for span in windows(p.frames, 100, 100)])) for p in pieces]


def test_a_batch_sounds_like_each_piece_alone(tts):
    engine = tts._engine
    texts = [TEXT, "merhaba.", "kısa bir cümle daha, biraz uzun olsun diye."]
    together = run(engine, [engine.piece(t, 0.0, i) for i, t in enumerate(texts)])
    for i, t in enumerate(texts):
        (alone,) = run(engine, [engine.piece(t, 0.0, i)])
        assert together[i].shape == alone.shape and np.allclose(together[i], alone, atol=1e-4)


def test_windowed_audio_matches_whole_decode(tts):
    engine = tts._engine
    p = engine.piece(TEXT + " " + TEXT, 0.0, 3)
    (windowed,) = run(engine, [p])
    whole = engine.decode([(p, (0, p.frames))])[0]
    assert windowed.shape == whole.shape and np.allclose(windowed, whole, atol=1e-5)
    assert CONTEXT >= 4


def test_windows_of_one_length_share_a_decoder_call(tts):
    engine, calls = tts._engine, []
    real = engine.sessions["decoder"]

    class Spy:
        def run(self, outputs, feeds):
            calls.append(feeds["z"].shape[:2])
            return real.run(outputs, feeds)

    pieces = [engine.piece("merhaba dünya.", 0.0, i) for i in range(3)] + [engine.piece(TEXT, 0.0, 9)]
    engine.plan(pieces)
    engine.think(pieces)
    engine.sessions["decoder"] = Spy()
    try:
        out = engine.decode([(p, (0, p.frames)) for p in pieces])
    finally:
        engine.sessions["decoder"] = real
    assert sorted(calls) == sorted([(3, pieces[0].frames), (1, pieces[3].frames)])
    assert [o.size for o in out] == [p.frames * engine.hop for p in pieces]


def test_noise_is_the_seeded_noise(tts):
    engine = tts._engine
    p = engine.piece("merhaba.", 0.0, 42)
    engine.plan([p])
    assert np.array_equal(engine.noise(p), randn(42, (4, p.frames, 64)))
