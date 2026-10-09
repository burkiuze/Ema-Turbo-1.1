"""EMA Turbo against EMA Lightning itself, on the same tiny weights: the same audio for the same seed.

Runs only where PyTorch and EMA Lightning are installed (python -m pip install ema-lightning)."""
import numpy as np
import pytest
from conftest import FIXTURES, MODELS, VOCAB, same_speech

from ema_turbo import EMA

torch = pytest.importorskip("torch")
lightning = pytest.importorskip("ema_lightning")

TEXTS = ["Merhaba, size nasıl yardımcı olabilirim?", "Bugün hava çok güzel. Yarın da yağmur yağacakmış.",
         " ".join(["Bu cümle yaklaşık olarak dört saniye sürecek kadar uzun bir cümledir."] * 3), ""]


def reference(name):
    from ema_lightning.decoder import Decoder
    from ema_lightning.frontend import Frontend
    from ema_lightning.model import Acoustic

    state = torch.load(FIXTURES / name / "torch.pt", weights_only=True)
    cfg = dict(d=32, n_heads=2, n_layers=2, text_conv=1, text_attn=1, dur_hidden=16, align_heads=2, ff_mult=2,
               letter_pos=name == "tiny_letter_pos", latent_dim=64, distilled=dict(times=[0.0, 0.25, 0.5, 0.75]))
    model = Acoustic(cfg, VOCAB, shared=True)
    model.load_state_dict(state["model"])
    decoder = Decoder(ch=64, rb_kernels=(3,), rb_dilations=((1, 3),))
    decoder.load_state_dict(state["decoder"])
    model, decoder = model.eval().requires_grad_(False), decoder.eval().requires_grad_(False)
    return lightning.EMA._from_parts(model, decoder, Frontend(VOCAB), "cpu")


@pytest.mark.parametrize("name", MODELS)
@pytest.mark.parametrize("rate", [48000, 16000])
def test_the_same_seed_gives_ema_lightnings_audio(name, rate):
    ema, turbo = reference(name), EMA(models=FIXTURES / name)
    for seed in (0, 7, 123456):
        for text in TEXTS:
            want, got = ema.say(text, seed=seed, sample_rate=rate), turbo.say(text, seed=seed, sample_rate=rate)
            same_speech(got.audio, want.audio, db=50)
            assert [(w.text, w.start, w.end) for w in got.words] == [(w.text, w.start, w.end) for w in want.words]


def test_streams_match_too():
    ema, turbo = reference(MODELS[0]), EMA(models=FIXTURES / MODELS[0])
    text = TEXTS[2]
    same_speech(np.concatenate(list(turbo.stream(text, seed=3))), np.concatenate(list(ema.stream(text, seed=3))), db=50)
