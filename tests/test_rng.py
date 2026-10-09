import numpy as np
import pytest

from ema_turbo.rng import randn


def test_the_same_seed_gives_the_same_noise():
    assert np.array_equal(randn(5, (4, 30, 64)), randn(5, (4, 30, 64)))
    assert not np.array_equal(randn(5, (4, 30, 64)), randn(6, (4, 30, 64)))
    assert randn(5, (4, 30, 64)).dtype == np.float32


def test_it_is_standard_normal():
    x = randn(1, (200_000,))
    assert abs(x.mean()) < 0.01 and abs(x.std() - 1) < 0.01


def test_only_the_low_32_bits_of_the_seed_count_like_pytorch():
    assert np.array_equal(randn(7, (64,)), randn(7 + 2**32, (64,)))


@pytest.mark.parametrize("shape", [(4, 1, 64), (4, 250, 64), (100,), (17,), (3, 37)])
@pytest.mark.parametrize("seed", [0, 3, 5_000_015, 2**33 + 5])
def test_matches_pytorch(seed, shape):
    torch = pytest.importorskip("torch")
    want = torch.randn(shape, generator=torch.Generator().manual_seed(seed)).numpy()
    assert np.abs(randn(seed, shape) - want).max() < 1e-5


def test_fewer_than_16_values_are_refused():
    with pytest.raises(ValueError):
        randn(0, (15,))
