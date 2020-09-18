import numpy as np
import numpy.testing as npt
import pytest

from quara.math import func_proj


def test_proj_to_self():
    proj = func_proj.proj_to_self()

    var = np.array([1.0, 2.0], dtype=np.float64)
    actual = proj(var)
    expected = np.array([1.0, 2.0], dtype=np.float64)
    npt.assert_almost_equal(actual, expected, decimal=15)


def test_proj_to_hyperplane():
    ### var_a = [2.0, 0.0]
    var_a = np.array([2.0, 0.0], dtype=np.float64)
    proj = func_proj.proj_to_hyperplane(var_a)

    # case1
    var = np.array([1.0, 1.0], dtype=np.float64)
    actual = proj(var)
    expected = np.array([2.0, 1.0], dtype=np.float64)
    npt.assert_almost_equal(actual, expected, decimal=15)

    # case2
    var = np.array([1.0, 0.0], dtype=np.float64)
    actual = proj(var)
    expected = np.array([2.0, 0.0], dtype=np.float64)
    npt.assert_almost_equal(actual, expected, decimal=15)

    # case3
    var = np.array([1.0, -1.0], dtype=np.float64)
    actual = proj(var)
    expected = np.array([2.0, -1.0], dtype=np.float64)
    npt.assert_almost_equal(actual, expected, decimal=15)

    # case4
    var = np.array([2.0, 1.0], dtype=np.float64)
    actual = proj(var)
    expected = np.array([2.0, 1.0], dtype=np.float64)
    npt.assert_almost_equal(actual, expected, decimal=15)

    ### var_a = [1.0, 1.0]
    var_a = np.array([1.0, 1.0], dtype=np.float64)
    proj = func_proj.proj_to_hyperplane(var_a)

    # case5
    var = np.array([-1.0, 1.0], dtype=np.float64)
    actual = proj(var)
    expected = np.array([0.0, 2.0], dtype=np.float64)
    npt.assert_almost_equal(actual, expected, decimal=15)

    # case6
    var = np.array([0.0, 0.0], dtype=np.float64)
    actual = proj(var)
    expected = np.array([1.0, 1.0], dtype=np.float64)
    npt.assert_almost_equal(actual, expected, decimal=15)

    # case7
    var = np.array([1.0, -1.0], dtype=np.float64)
    actual = proj(var)
    expected = np.array([2.0, 0.0], dtype=np.float64)
    npt.assert_almost_equal(actual, expected, decimal=15)

    # case8
    var = np.array([0.0, 2.0], dtype=np.float64)
    actual = proj(var)
    expected = np.array([0.0, 2.0], dtype=np.float64)
    npt.assert_almost_equal(actual, expected, decimal=15)


def test_proj_to_nonnegative():
    proj = func_proj.proj_to_nonnegative()

    # case1
    var = np.array([1.0, 2.0], dtype=np.float64)
    actual = proj(var)
    expected = np.array([1.0, 2.0], dtype=np.float64)
    npt.assert_almost_equal(actual, expected, decimal=15)

    # case2
    var = np.array([-1.0, 2.0], dtype=np.float64)
    actual = proj(var)
    expected = np.array([0.0, 2.0], dtype=np.float64)
    npt.assert_almost_equal(actual, expected, decimal=15)

    # case3
    var = np.array([1.0, -2.0], dtype=np.float64)
    actual = proj(var)
    expected = np.array([1.0, 0.0], dtype=np.float64)
    npt.assert_almost_equal(actual, expected, decimal=15)

    # case4
    var = np.array([-1.0, -2.0], dtype=np.float64)
    actual = proj(var)
    expected = np.array([0.0, 0.0], dtype=np.float64)
    npt.assert_almost_equal(actual, expected, decimal=15)

