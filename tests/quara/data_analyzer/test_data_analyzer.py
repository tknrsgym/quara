import numpy as np
import numpy.testing as npt
import pytest

from quara.data_analyzer import data_analyzer
from quara.math import norm


def test_calc_mse():
    xs = [
        np.array([2.0, 3.0], dtype=np.float64),
        np.array([4.0, 5.0], dtype=np.float64),
    ]
    y = np.array([1.0, 2.0], dtype=np.float64)

    actual = data_analyzer.calc_mse(xs, y, norm.l2_norm)
    npt.assert_almost_equal(actual, 10.0, decimal=14)
