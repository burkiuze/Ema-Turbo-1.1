"""The three ONNX models and ONNX Runtime sessions for them.

A model folder holds text.onnx (letters to text features and durations), sound.onnx (the aligner and the four
sampling steps), decoder.onnx (latents to 48 kHz audio) and meta.json (the alphabet, the sampling times, the
frame size, and the SHA-256 of the EMA Lightning weights they were exported from). `python -m ema_turbo.export`
writes one from EMA Lightning's ema.pt and decoder.pt; that is the only step that needs PyTorch.
"""
import json
import os
from pathlib import Path

FILES = ("text.onnx", "sound.onnx", "decoder.onnx", "meta.json")
FORMAT = 1
CACHE = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")) / "ema_turbo" / f"models-v{FORMAT}"


class MissingModels(FileNotFoundError):
    pass


def find(models=None):
    """The model folder: the argument, $EMA_TURBO_MODELS, or the cache (exported there once if PyTorch is around)."""
    for candidate in (models, os.environ.get("EMA_TURBO_MODELS")):
        if candidate:
            path = Path(candidate)
            if not complete(path):
                raise MissingModels(f"{path} must hold {', '.join(FILES)}")
            return path
    if complete(CACHE):
        return CACHE
    try:
        import torch  # noqa: F401
    except ImportError:
        raise MissingModels(
            "EMA Turbo's ONNX models were not found. Pass EMA(models='folder') or set EMA_TURBO_MODELS to a folder "
            f"with {', '.join(FILES)}, or run `python -m ema_turbo.export` once on a machine with PyTorch."
        ) from None
    from .export import export_from_hub

    print(f"EMA Turbo: exporting the models once to {CACHE} (needs PyTorch, about 10 seconds)")
    export_from_hub(CACHE)
    return CACHE


def complete(path):
    return all((Path(path) / f).is_file() for f in FILES)


def meta(path):
    data = json.loads((Path(path) / "meta.json").read_text())
    if data.get("format") != FORMAT:
        raise MissingModels(f"{path} holds models in format {data.get('format')}, this EMA Turbo reads {FORMAT}")
    return data


_SHARED = None  # whether this process runs ONNX Runtime on one shared thread pool


def shared_pool(threads=None):
    """One thread pool for every session in the process, created once.

    With a pool per session, the three models' threads spin after each call and fight the next model's threads
    for the same cores. One shared pool measured 40% sooner first audio (104 -> 61 ms) and 5% more throughput on
    a 2-core CPU. If the process's pools were made before (by EMA Turbo or anyone else), they are used as they are;
    if they cannot be shared at all, each session gets its own pool without spinning."""
    global _SHARED
    if _SHARED is None:
        import onnxruntime as ort

        try:
            ort.set_global_thread_pool_sizes(int(threads or 0), 1)
            _SHARED = True
        except Exception as error:
            _SHARED = "already" in str(error)
    return _SHARED


def sessions(path, threads=None, providers=None, low_memory=False):
    """One ONNX Runtime session per model, on the shared thread pool, with denormals flushed to zero (measured 8%
    faster, same audio). low_memory: no memory arena, so the memory of a call is given back after it."""
    import onnxruntime as ort

    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    options.add_session_config_entry("session.set_denormal_as_zero", "1")
    options.enable_cpu_mem_arena = not low_memory
    if shared_pool(threads):
        options.use_per_session_threads = False
    else:
        options.add_session_config_entry("session.intra_op.allow_spinning", "0")
        if threads:
            options.intra_op_num_threads = int(threads)
    providers = providers or ["CPUExecutionProvider"]
    return {name: ort.InferenceSession(str(Path(path) / f"{name}.onnx"), options, providers=providers)
            for name in ("text", "sound", "decoder")}
