"""Write the tiny test models: EMA Lightning's architecture at a tiny size with random weights, exported the
way real models are. Needs PyTorch; the tests themselves only read the ONNX files.

    python tests/make_fixtures.py
"""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ema_turbo.export import export_modules  # noqa: E402
from ema_turbo.export_torch.decoder import Decoder  # noqa: E402
from ema_turbo.export_torch.model import Acoustic  # noqa: E402

VOCAB = ["<pad>", "<unk>"] + list(" !\"%&'(),-./:;?abcdefghijklmnopqrstuvwxyzçöüğış")
HERE = Path(__file__).parent / "fixtures"


def tiny_model(letter_pos):
    torch.manual_seed(0)
    cfg = dict(d=32, n_heads=2, n_layers=2, text_conv=1, text_attn=1, dur_hidden=16, align_heads=2, ff_mult=2,
               letter_pos=letter_pos, latent_dim=64, distilled=dict(times=[0.0, 0.25, 0.5, 0.75]))
    model = Acoustic(cfg, VOCAB, shared=True)
    model.chardur.out.bias.data.fill_(1.0)  # about two frames per letter, like the real model
    for p in model.aligner.parameters():
        p.data.normal_(0, 0.1)
    return model.eval().requires_grad_(False)


def tiny_decoder():
    torch.manual_seed(1)
    return Decoder(ch=64, rb_kernels=(3,), rb_dilations=((1, 3),)).eval().requires_grad_(False)


if __name__ == "__main__":
    for name, letter_pos in (("tiny_letter_pos", True), ("tiny_word_pos", False)):
        export_modules(tiny_model(letter_pos), tiny_decoder(), HERE / name, {"tiny": name})
        torch.save({"model": tiny_model(letter_pos).state_dict(), "decoder": tiny_decoder().state_dict()},
                   HERE / name / "torch.pt")
        print("wrote", HERE / name)
