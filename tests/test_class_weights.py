import pytest
import torch

from meldrt.train.class_weights import class_weights

COUNTS = [400, 100, 25]


def test_none_gives_equal_weights():
    assert torch.allclose(class_weights(COUNTS, "none"), torch.ones(3))


def test_sqrt_inv_ratios():
    w = class_weights(COUNTS, "sqrt_inv")
    # 16x fewer examples gives a 4x larger weight
    assert w[2] / w[0] == pytest.approx(4.0)
    assert w[1] / w[0] == pytest.approx(2.0)


def test_inv_ratios():
    w = class_weights(COUNTS, "inv")
    assert w[2] / w[0] == pytest.approx(16.0)


@pytest.mark.parametrize("scheme", ["none", "inv", "sqrt_inv"])
def test_weights_have_mean_one(scheme):
    assert class_weights(COUNTS, scheme).mean() == pytest.approx(1.0)


def test_unknown_scheme_is_an_error():
    with pytest.raises(ValueError):
        class_weights(COUNTS, "banana")
