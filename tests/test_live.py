"""Live speech: the text cutter, then live replies on the tiny models."""
import random
import threading
import time

import numpy as np
import pytest
from conftest import same_speech

from ema_turbo.chunker import CLAUSE_PAUSE, SENTENCE_PAUSE
from ema_turbo.live import Segmenter

REPLY = ("Merhaba, size nasıl yardımcı olabilirim? Dr. Ayşe saat 14:30'da gelecek. Toplantı 3. katta, herkes hazır "
         "olsun. Bu cümlede hiç noktalama yok ve uzun mu uzun bir metin olarak devam ediyor ve ediyor ve ediyor")


def cut(parts):
    s = Segmenter()
    out = []
    for part in parts:
        out += s.push(part)
    return out + s.flush()


def test_the_first_piece_is_cut_early_and_later_ones_at_sentences():
    assert cut([REPLY]) == [
        ("Merhaba,", CLAUSE_PAUSE),
        ("size nasıl yardımcı olabilirim?", SENTENCE_PAUSE),
        ("Dr. Ayşe saat 14:30'da gelecek.", SENTENCE_PAUSE),
        ("Toplantı 3. katta, herkes hazır olsun.", SENTENCE_PAUSE),
        ("Bu cümlede hiç noktalama yok ve uzun mu uzun bir metin olarak devam ediyor ve ediyor ve ediyor", 0.0),
    ]
    later = cut(["Tamam. " + " ".join(f"kelime{i}" for i in range(25))])
    assert later[1] == (" ".join(f"kelime{i}" for i in range(20)), CLAUSE_PAUSE)  # later pieces: twenty words


def test_cuts_do_not_depend_on_how_the_text_arrives():
    whole = cut([REPLY])
    rng = random.Random(0)
    for _ in range(50):
        points = sorted(rng.sample(range(1, len(REPLY)), rng.randrange(1, 60)))
        parts = [REPLY[a:b] for a, b in zip([0] + points, points + [len(REPLY)], strict=True)]
        assert cut(parts) == whole


def test_nothing_is_cut_before_a_space_arrives():
    s = Segmenter()
    assert s.push("Merhaba,") == []
    assert s.push(" size") == [("Merhaba,", CLAUSE_PAUSE)]
    assert s.push(" 14:") == [] and s.push("30'da.") == []
    assert s.flush() == [("size 14:30'da.", 0.0)]


def test_first_piece_without_punctuation_is_cut_after_eight_words():
    words = " ".join(f"kelime{i}" for i in range(12))
    assert cut([words])[0] == (" ".join(f"kelime{i}" for i in range(8)), CLAUSE_PAUSE)


def test_nothing_in_nothing_out():
    assert cut([]) == [] and cut(["   "]) == []


def collect(live):
    return np.concatenate(list(live)) if live else np.zeros(0, np.float32)


def test_a_live_reply_speaks_every_piece_in_order(tts):
    live = tts.live(seed=4)
    for part in ("Merhaba, ", "nasılsınız? Bugün ", "size yardımcı olacağım."):
        live.feed(part)
    live.end()
    audio = np.concatenate(list(live))
    pieces = cut(["Merhaba, nasılsınız? Bugün size yardımcı olacağım."])
    want = []
    for i, (text, pause) in enumerate(pieces):
        want.append(tts.say(text, seed=(4 * 1_000_003 + 7919 * i) % 2**63).audio)
        if i + 1 < len(pieces):
            want.append(np.zeros(round(pause * 48000), np.float32))
    same_speech(audio, np.concatenate(want))


def test_stream_text_does_not_depend_on_how_the_text_arrives(tts):
    text = "Merhaba, size nasıl yardımcı olabilirim? Bugün hava çok güzel."
    a = np.concatenate(list(tts.stream_text([text], seed=1, sample_rate=16000)))
    b = np.concatenate(list(tts.stream_text(iter([text[:7], text[7:20], text[20:]]), seed=1, sample_rate=16000)))
    assert np.array_equal(a, b) or np.allclose(a, b, atol=1e-5)


def test_the_first_chunk_comes_before_the_reply_is_written(tts):
    gate = threading.Event()

    def llm():
        yield "Merhaba, "
        yield "size "
        gate.wait(10)  # the rest of the reply is not written until the first audio is out
        yield "nasıl yardımcı olabilirim?"

    chunks = tts.stream_text(llm(), seed=0)
    first = next(chunks)
    assert first.size > 0
    gate.set()
    assert sum(c.size for c in chunks) > 0


def test_cancel_drops_the_rest(tts):
    live = tts.live(seed=0)
    live.feed(" ".join(["Bu cümle yaklaşık olarak dört saniye sürecek kadar uzun bir cümledir."] * 6))
    chunks = iter(live)
    assert next(chunks).size > 0
    live.cancel()
    assert list(chunks) == []
    deadline = time.time() + 10
    while (tts._playhead.sentences or tts._playhead.windows) and time.time() < deadline:
        time.sleep(0.05)
    assert not tts._playhead.sentences and not tts._playhead.windows


def test_leaving_stream_text_early_stops_the_feeding(tts):
    fed = []

    def llm():
        for i in range(200):
            fed.append(i)
            time.sleep(0.005)
            yield f"kelime{i}. "

    chunks = tts.stream_text(llm())
    next(chunks)
    chunks.close()
    time.sleep(0.3)
    assert len(fed) < 200


def test_a_failing_text_source_reaches_the_listener(tts):
    def llm():
        yield "Merhaba, "
        raise ConnectionError("model went away")

    with pytest.raises(ConnectionError):
        list(tts.stream_text(llm()))


def test_feeding_after_the_end_is_an_error(tts):
    live = tts.live()
    live.end()
    live.end()  # twice is fine
    with pytest.raises(RuntimeError):
        live.feed("geç kaldım")
    assert list(live) == []


def test_live_checks_its_settings(tts):
    with pytest.raises(ValueError):
        tts.live(speed=9)
    with pytest.raises(TypeError):
        tts.live().feed(5)


def test_cancel_from_another_thread_wakes_the_listener(tts):
    live = tts.live(seed=0)
    live.feed(" ".join(["Bu cümle yaklaşık olarak dört saniye sürecek kadar uzun bir cümledir."] * 8))
    got, done = [], threading.Event()

    def listen():
        for chunk in live:
            got.append(chunk)
        done.set()

    threading.Thread(target=listen, daemon=True).start()
    deadline = time.time() + 10
    while not got and time.time() < deadline:
        time.sleep(0.01)
    live.cancel()  # the listener may be waiting for the next chunk right now
    assert done.wait(5), "the listener was left waiting after cancel()"
