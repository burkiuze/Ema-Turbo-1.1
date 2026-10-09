# Changelog

## EMA Turbo 1.1.0

For live conversations.

- Live mode: `live()` (feed text as an LLM writes it, iterate audio in another
  thread, `cancel()` for barge-in) and `stream_text()` (any iterable of text
  deltas). The first piece is cut at the first clause or sentence end; cuts
  depend only on the text, never on how it arrives.
- Streams decode growing windows (0.2 s, 0.8 s, 2 s, then full ones), and a
  stream's first sentence is thought and decoded alone.
- One shared ONNX Runtime thread pool for the three models.
- First audio of `stream()`: 66-127 ms (1.0: 164-364 ms); live mode: 69-76 ms
  after an LLM's first token on a 2-core CPU.
- The Streamlit demo shows the first-audio time; examples for live LLM replies
  and a live WebSocket endpoint.

## EMA Turbo 1.0.0

First release: EMA Lightning 1.0.4 on ONNX Runtime.

- No PyTorch at run time: `numpy`, `onnxruntime` and `normalizer-tr` only.
- The same audio for the same seed: PyTorch's CPU `randn` reproduced in NumPy;
  the exported graphs checked against PyTorch at export.
- About 2x faster on a 2-core CPU, first audio of a stream about 4.5x sooner,
  about a fifth of the memory once loaded.
- `low_memory=True`: five-second decoder windows and no memory arena, half the
  peak memory, the same audio.
- `warmup()` (also `lightning()`), `python -m ema_turbo.export`.
- A Streamlit demo (`app.py`) without PyTorch.
