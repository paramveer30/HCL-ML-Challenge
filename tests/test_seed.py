import random

import numpy as np
import torch

from meldrt.utils.seed import set_seed


def draw():
    return random.random(), float(np.random.rand()), float(torch.rand(1))


def test_same_seed_gives_same_numbers():
    set_seed(7)
    first = draw()
    set_seed(7)
    assert draw() == first


def test_different_seed_gives_different_numbers():
    set_seed(7)
    first = draw()
    set_seed(8)
    assert draw() != first
