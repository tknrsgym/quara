import numpy as np
import numpy.testing as npt
import pytest

from quara.math import entropy


def test_round_varz():
    # success
    actual = entropy.round_varz(0.1, 0.0)
    expected = 0.1
    npt.assert_almost_equal(actual, expected, decimal=15)

    actual = entropy.round_varz(np.float64(0.1), np.float64(0.0))
    expected = 0.1
    npt.assert_almost_equal(actual, expected, decimal=15)

    actual = entropy.round_varz(0.5, 0.8)
    expected = 0.8
    npt.assert_almost_equal(actual, expected, decimal=15)

    actual = entropy.round_varz(np.float64(0.5), np.float64(0.8))
    expected = 0.8
    npt.assert_almost_equal(actual, expected, decimal=15)

    # raise ValueError
    with pytest.raises(ValueError):
        entropy.round_varz(-0.1, 0.0)

    with pytest.raises(ValueError):
        entropy.round_varz(0.5, -0.8)

    with pytest.raises(ValueError):
        entropy.round_varz(0.5, 0.8j)


"""
def test_relative_entropy():
    assert False


def test_gradient_relative_entropy_2nd():
    assert False


def test_hessian_relative_entropy_2nd():
    assert False
"""
