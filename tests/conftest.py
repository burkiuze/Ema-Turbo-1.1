"""Shared fixtures: tiny models with EMA Lightning's real structure, exported to ONNX (make_fixtures.py), and a
stand-in normalizer when normalizer-tr is missing."""
import os
import sys
import tempfile
import types
from pathlib import Path

import numpy as np
import pytest

os.environ["XDG_CACHE_HOME"] = tempfile.mkdtemp()

try:
    import normalizer_tr  # noqa: F401
except ImportError:
    class _Result:
        def __init__(self, text):
            self.normalized_text = text

    class _Normalizer:
        calls = []

        def normalize(self, text, *, ambiguity_policy="preserve", **_):
            self.calls.append(ambiguity_policy)
            return _Result(text.replace("5", "beş").replace("2", "iki"))

    sys.modules["normalizer_tr"] = types.SimpleNamespace(Normalizer=_Normalizer)

from ema_turbo import EMA  # noqa: E402

FIXTURES = Path(__file__).parent / "fixtures"
MODELS = ["tiny_letter_pos", "tiny_word_pos"]
VOCAB = ["<pad>", "<unk>"] + list(" !\"%&'(),-./:;?abcdefghijklmnopqrstuvwxyzçöüğış")


@pytest.fixture(params=MODELS)
def models(request):
    return FIXTURES / request.param


@pytest.fixture
def tts(models):
    return EMA(models=models)


def same_speech(a, b, db=40.0):
    """The same audio made two ways: the same length, and the difference at least `db` dB quieter than the speech."""
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    assert a.shape == b.shape, f"lengths differ: {a.shape} vs {b.shape}"
    diff = float(np.sum((a - b) ** 2))
    if diff == 0.0:
        return
    speech = float(np.sum(a**2))
    quieter = 10 * np.log10(speech / diff) if speech > 0 else float("-inf")
    assert quieter >= db, f"the difference is only {quieter:.1f} dB quieter than the speech"
