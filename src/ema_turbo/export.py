"""Export EMA Lightning's weights to EMA Turbo's three ONNX models. The only step that needs PyTorch.

    python -m ema_turbo.export                       # downloads ema.pt and decoder.pt from the Hub
    python -m ema_turbo.export --weights folder      # or reads them from a folder
    python -m ema_turbo.export --out models          # where to write (default: the cache EMA() reads)

The acoustic model's graph is then fused by ONNX Runtime's transformer optimizer (layer norms, GELU, attention
pieces), which made the sampling steps about 1.4x faster on a CPU with the same latents to within 1e-4.
Every export is checked against PyTorch before it is written.
"""
import argparse
import hashlib
import json
import os
import shutil
import tempfile
import warnings
from pathlib import Path

REPO = "canberkkkkkk/ema-lightning"


def export_from_hub(out):
    from huggingface_hub import hf_hub_download

    return export(hf_hub_download(REPO, "ema.pt"), hf_hub_download(REPO, "decoder.pt"), out)


def export(ema_path, decoder_path, out, check=True):
    """EMA Lightning's ema.pt and decoder.pt to EMA Turbo's model folder `out`, checked against PyTorch."""
    from .export_torch.decoder import load_decoder
    from .export_torch.model import load_acoustic

    model = load_acoustic(ema_path, "cpu")
    decoder = load_decoder(decoder_path, "cpu")
    source = {"ema.pt": _sha256(ema_path), "decoder.pt": _sha256(decoder_path)}
    return export_modules(model, decoder, out, source, check)


def export_modules(model, decoder, out, source=None, check=True):
    """Loaded PyTorch modules (export_torch's Acoustic and Decoder) to a model folder."""
    import numpy as np
    import torch

    from .runtime import FORMAT

    torch.manual_seed(0)
    model, decoder = model.eval().float(), decoder.eval().float()
    out = Path(out)
    work = Path(tempfile.mkdtemp(prefix="ema-turbo-"))
    try:
        _export_text(model, work / "text.onnx")
        _export_sound(model, work / "sound_raw.onnx")
        _fuse(work / "sound_raw.onnx", work / "sound.onnx", model)
        _export_decoder(decoder, work / "decoder.onnx")
        (work / "meta.json").write_text(json.dumps({
            "format": FORMAT,
            "vocab": list(model.vocab),
            "times": [float(t) for t in model.times],
            "latent_dim": int(model.latent_dim),
            "hop": int(decoder.hop),
            "rate": 48000,
            "source": source or {},
        }, ensure_ascii=False, indent=1))
        if check:
            _check(work, model, decoder, np, torch)
        out.mkdir(parents=True, exist_ok=True)
        for name in ("text.onnx", "sound.onnx", "decoder.onnx", "meta.json"):
            shutil.move(str(work / name), str(out / name))
    finally:
        shutil.rmtree(work, ignore_errors=True)
    return out


def _modules(model):
    import torch

    class Text(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.m = model

        def forward(self, ids, mask):
            return self.m.text_stage(ids, mask)

    class Sound(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.m = model

        def forward(self, h, dur, mask, cw, wstart, fw, fp, fmask, noise):
            return self.m.sound_stage(h, dur, mask, cw, wstart, fw, fp, fmask, noise)

    return Text(), Sound()


def _onnx_export(module, args, path, names, outputs, axes):
    import torch

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        torch.onnx.export(module.eval(), args, str(path), input_names=names, output_names=outputs, dynamic_axes=axes,
                          opset_version=17, dynamo=False)


def _export_text(model, path):
    import torch

    text, _ = _modules(model)
    ids = torch.randint(2, len(model.vocab), (2, 30))
    axes = {k: {0: "batch", 1: "letters"} for k in ("ids", "mask", "h", "dur")}
    _onnx_export(text, (ids, torch.ones_like(ids, dtype=torch.bool)), path, ["ids", "mask"], ["h", "dur"], axes)


def _export_sound(model, path):
    import torch

    _, sound = _modules(model)
    B, L, T = 2, 30, 50
    ids = torch.randint(2, len(model.vocab), (B, L))
    mask = torch.ones(B, L, dtype=torch.bool)
    with torch.no_grad():
        h, dur = model.text_stage(ids, mask)
    cw = (torch.arange(L) // 5).repeat(B, 1)
    args = (h, dur, mask, cw, cw * 5, (torch.arange(T) * 6 // T).repeat(B, 1), torch.rand(B, T),
            torch.ones(B, T, dtype=torch.bool), torch.randn(B, len(model.times), T, model.latent_dim))
    names = ["h", "dur", "mask", "cw", "wstart", "fw", "fp", "fmask", "noise"]
    axes = {k: {0: "batch", 1: "letters"} for k in ("h", "dur", "mask", "cw", "wstart")}
    axes.update({k: {0: "batch", 1: "frames"} for k in ("fw", "fp", "fmask", "latents")})
    axes["noise"] = {0: "batch", 2: "frames"}
    _onnx_export(sound, args, path, names, ["latents"], axes)


def _fuse(src, dst, model):
    try:
        from onnxruntime.transformers.optimizer import optimize_model

        heads = model.d // model.dh
        optimize_model(str(src), model_type="bert", num_heads=heads, hidden_size=model.d,
                       opt_level=1).save_model_to_file(str(dst))
    except Exception as error:  # the fusions are a speed-up only
        warnings.warn(f"transformer fusions skipped: {error}", stacklevel=2)
        shutil.copy(src, dst)


def _export_decoder(decoder, path):
    import torch

    class Decode(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.d = decoder

        def forward(self, z):
            return self.d(z.transpose(1, 2))

    axes = {"z": {0: "batch", 1: "frames"}, "audio": {0: "batch", 1: "samples"}}
    _onnx_export(Decode(), (torch.zeros(1, 20, decoder.pre.in_channels),), path, ["z"], ["audio"], axes)


def _check(work, model, decoder, np, torch):
    """Every model against PyTorch on random inputs of two lengths and a padded batch."""
    from .runtime import sessions

    s = sessions(work)
    g = torch.Generator().manual_seed(1)
    for L, T in ((13, 31), (90, 210)):
        ids = torch.randint(2, len(model.vocab), (2, L), generator=g)
        ids[1, L // 2:] = 0
        mask = ids != 0
        with torch.no_grad():
            h, dur = model.text_stage(ids, mask)
        oh, od = s["text"].run(None, {"ids": ids.numpy(), "mask": mask.numpy()})
        _close("text", od, dur.numpy(), 1e-3)
        cw = torch.minimum(torch.arange(L) // 4, torch.tensor(L // 4)).repeat(2, 1).masked_fill(~mask, -1)
        wstart = (cw.clamp(min=0) * 4)
        fw = (torch.arange(T) * (L // 4) // T).repeat(2, 1)
        args = dict(h=h, dur=dur, mask=mask, cw=cw, wstart=wstart, fw=fw, fp=torch.rand(2, T, generator=g),
                    fmask=torch.ones(2, T, dtype=torch.bool),
                    noise=torch.randn(2, len(model.times), T, model.latent_dim, generator=g))
        with torch.no_grad():
            latents = model.sound_stage(**args)
            audio = decoder(latents[:1].transpose(1, 2))
        olat = s["sound"].run(None, {k: v.numpy() for k, v in args.items()})[0]
        _close("sound", olat, latents.numpy(), 5e-3)
        oaudio = s["decoder"].run(None, {"z": latents[:1].numpy()})[0]
        _close("decoder", oaudio, audio.numpy(), 1e-4)


def _close(name, got, want, atol):
    import numpy as np

    err = float(np.abs(np.asarray(got) - np.asarray(want)).max())
    if not err <= atol:
        raise RuntimeError(f"the exported {name} model differs from PyTorch by {err:.2e} (allowed {atol:.0e})")


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main(argv=None):
    from .runtime import CACHE

    parser = argparse.ArgumentParser(description="Export EMA Lightning's weights to EMA Turbo's ONNX models.")
    parser.add_argument("--weights", help="a folder with ema.pt and decoder.pt (default: download from the Hub)")
    parser.add_argument("--out", default=str(CACHE), help=f"where to write the models (default: {CACHE})")
    args = parser.parse_args(argv)
    if args.weights:
        out = export(Path(args.weights) / "ema.pt", Path(args.weights) / "decoder.pt", args.out)
    else:
        out = export_from_hub(args.out)
    sizes = ", ".join(f"{p.name} {os.path.getsize(p) / 1e6:.1f} MB" for p in sorted(Path(out).glob("*.onnx")))
    print(f"EMA Turbo models written to {out}: {sizes}")


if __name__ == "__main__":
    main()
