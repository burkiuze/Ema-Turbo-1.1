"""PyTorch's seeded normal noise on a CPU, without PyTorch.

EMA Lightning draws each sentence's noise with torch.randn on a torch.Generator seeded with manual_seed.
On a CPU that is a 32-bit Mersenne Twister seeded with the low 32 bits of the seed; each value is a uniform
float from the low 24 bits of one draw; and every block of 16 is turned into normals by Box-Muller, the first
8 as radii times cosines and the last 8 as radii times sines (a size that is not a multiple of 16 redraws its
last 16 values). NumPy's legacy RandomState runs the same Mersenne Twister, so the same seed gives the same
draws, and the normals agree to about 1e-6 (PyTorch uses single-precision vector math for the logarithm and
the sines).
"""
import numpy as np

_SCALE = np.float32(1.0 / (1 << 24))


def randn(seed, shape):
    """float32 standard normals of `shape`, as torch.randn(shape, generator=Generator().manual_seed(seed))."""
    n = int(np.prod(shape))
    if n < 16:  # PyTorch draws fewer than 16 values another way; EMA's noise always has a multiple of 64
        raise ValueError("randn reproduces PyTorch for 16 values or more")
    tail = 16 if n % 16 else 0
    draws = np.random.RandomState(seed & 0xFFFFFFFF).randint(0, 2**32, size=n + tail, dtype=np.uint64)
    u = ((draws & 0xFFFFFF).astype(np.float32) * _SCALE).astype(np.float64)
    out = np.empty(n, np.float64)
    whole = n - n % 16
    out[:whole] = _box_muller(u[:whole])
    if tail:
        out[n - 16:] = _box_muller(u[n:])
    return out.astype(np.float32).reshape(shape)


def _box_muller(u):
    blocks = u.reshape(-1, 16)
    radius = np.sqrt(-2.0 * np.log(1.0 - blocks[:, :8]))
    theta = 2.0 * np.pi * blocks[:, 8:]
    return np.concatenate([radius * np.cos(theta), radius * np.sin(theta)], 1).reshape(-1)
