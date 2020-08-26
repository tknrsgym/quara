import numpy as np
import pytest

from quara.math import norm


def test_l2_norm():
    x = np.array([1.0, 2.0], dtype=np.float64)
    y = np.array([4.0, 6.0], dtype=np.float64)

    actual = norm.l2_norm(x, y)
    assert actual == 5.0
