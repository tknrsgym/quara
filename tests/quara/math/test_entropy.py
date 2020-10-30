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


def test_relative_entropy():
    q = np.array([2 ** 3, 3 ** 2], dtype=np.float64)
    p = np.array([2, 3], dtype=np.float64)
    actual = entropy.relative_entropy(q, p)
    expected = 2 ** 4 * np.log(2) + 3 ** 2 * np.log(3)
    npt.assert_almost_equal(actual, expected, decimal=15)

    # q < eps_q
    q = np.array([2 ** 3, 3 ** 2], dtype=np.float64)
    p = np.array([2, 3], dtype=np.float64)
    actual = entropy.relative_entropy(q, p, eps_q=8.5)
    expected = 3 ** 2 * np.log(3)
    npt.assert_almost_equal(actual, expected, decimal=15)

    # p < eps_p
    q = np.array([2 ** 3, 3 ** 2], dtype=np.float64)
    p = np.array([2, 3], dtype=np.float64)
    actual = entropy.relative_entropy(q, p, eps_p=4)
    expected = 2 ** 3 * np.log(4) + 3 ** 2 * np.log(4)
    npt.assert_almost_equal(actual, expected, decimal=15)

    # q/p < eps_p
    q = np.array([2 ** 3, 3], dtype=np.float64)
    p = np.array([2, 3], dtype=np.float64)
    actual = entropy.relative_entropy(q, p, eps_p=2)
    expected = 2 ** 3 * np.log(4) + 3 * np.log(2)
    npt.assert_almost_equal(actual, expected, decimal=15)

    # p has negative entry
    q = np.array([2 ** 3, 3 ** 2], dtype=np.float64)
    p = np.array([-2, 3], dtype=np.float64)
    with pytest.raises(ValueError):
        entropy.relative_entropy(q, p)

    # eps_p is negative
    q = np.array([2 ** 3, 3 ** 2], dtype=np.float64)
    p = np.array([2, 3], dtype=np.float64)
    with pytest.raises(ValueError):
        entropy.relative_entropy(q, p, eps_p=-1)

    # eps_q is negative
    q = np.array([2 ** 3, 3 ** 2], dtype=np.float64)
    p = np.array([2, 3], dtype=np.float64)
    with pytest.raises(ValueError):
        entropy.relative_entropy(q, p, eps_q=-1)


def test_gradient_relative_entropy_2nd():
    q = np.array([1, 2], dtype=np.float64)
    p = np.array([3, 4], dtype=np.float64)
    grad_p = np.array([[12, 24], [36, 48]], dtype=np.float64)
    actual = entropy.gradient_relative_entropy_2nd(q, p, grad_p)
    expected = np.array([-22, -32], dtype=np.float64)
    npt.assert_almost_equal(actual, expected, decimal=15)

    # q < eps_q
    q = np.array([1, 2], dtype=np.float64)
    p = np.array([3, 4], dtype=np.float64)
    grad_p = np.array([[12, 24], [36, 48]], dtype=np.float64)
    actual = entropy.gradient_relative_entropy_2nd(q, p, grad_p, eps_q=1.5)
    expected = np.array([-18, -24], dtype=np.float64)
    npt.assert_almost_equal(actual, expected, decimal=15)

    # p < eps_p
    q = np.array([1, 2], dtype=np.float64)
    p = np.array([3, 4], dtype=np.float64)
    grad_p = np.array([[12, 24], [36, 48]], dtype=np.float64)
    actual = entropy.gradient_relative_entropy_2nd(q, p, grad_p, eps_p=4)
    expected = np.array([-21, -30], dtype=np.float64)
    npt.assert_almost_equal(actual, expected, decimal=15)

    # p has negative entry
    q = np.array([1, 2], dtype=np.float64)
    p = np.array([-3, 4], dtype=np.float64)
    grad_p = np.array([[12, 24], [36, 48]], dtype=np.float64)
    with pytest.raises(ValueError):
        entropy.gradient_relative_entropy_2nd(q, p, grad_p)

    # eps_p is negative
    q = np.array([1, 2], dtype=np.float64)
    p = np.array([3, 4], dtype=np.float64)
    grad_p = np.array([[12, 24], [36, 48]], dtype=np.float64)
    with pytest.raises(ValueError):
        entropy.gradient_relative_entropy_2nd(q, p, grad_p, eps_p=-1)


def test_hessian_relative_entropy_2nd():
    q = np.array([1, 2], dtype=np.float64)
    p = np.array([3, 4], dtype=np.float64)
    grad_p = np.array([[12, 24], [36, 48]], dtype=np.float64)
    hess_p = [
        np.array([[12, 24], [36, 48]], dtype=np.float64),
        np.array([[60, 72], [84, 96]], dtype=np.float64),
    ]
    actual = entropy.hessian_relative_entropy_2nd(q, p, grad_p, hess_p)
    expected = np.array([[144, 204], [194, 288]], dtype=np.float64)
    npt.assert_almost_equal(actual, expected, decimal=15)

    # q < eps_q
    q = np.array([1, 2], dtype=np.float64)
    p = np.array([3, 4], dtype=np.float64)
    grad_p = np.array([[12, 24], [36, 48]], dtype=np.float64)
    hess_p = [
        np.array([[12, 24], [36, 48]], dtype=np.float64),
        np.array([[60, 72], [84, 96]], dtype=np.float64),
    ]
    actual = entropy.hessian_relative_entropy_2nd(q, p, grad_p, hess_p, eps_q=1.5)
    expected = np.array([[132, 180], [174, 240]], dtype=np.float64)
    npt.assert_almost_equal(actual, expected, decimal=15)

    # p < eps_p
    q = np.array([1, 2], dtype=np.float64)
    p = np.array([3, 4], dtype=np.float64)
    grad_p = np.array([[12, 24], [36, 48]], dtype=np.float64)
    hess_p = [
        np.array([[12, 24], [36, 48]], dtype=np.float64),
        np.array([[60, 72], [84, 96]], dtype=np.float64),
    ]
    actual = entropy.hessian_relative_entropy_2nd(q, p, grad_p, hess_p, eps_p=4)
    expected = np.array([[138, 192], [183, 264]], dtype=np.float64)
    npt.assert_almost_equal(actual, expected, decimal=15)

    # p has negative entry
    q = np.array([1, 2], dtype=np.float64)
    p = np.array([3, -4], dtype=np.float64)
    grad_p = np.array([[12, 24], [36, 48]], dtype=np.float64)
    hess_p = [
        np.array([[12, 24], [36, 48]], dtype=np.float64),
        np.array([[60, 72], [84, 96]], dtype=np.float64),
    ]
    with pytest.raises(ValueError):
        entropy.hessian_relative_entropy_2nd(q, p, grad_p, hess_p)

    # eps_p is negative
    q = np.array([1, 2], dtype=np.float64)
    p = np.array([3, 4], dtype=np.float64)
    grad_p = np.array([[12, 24], [36, 48]], dtype=np.float64)
    hess_p = [
        np.array([[12, 24], [36, 48]], dtype=np.float64),
        np.array([[60, 72], [84, 96]], dtype=np.float64),
    ]
    with pytest.raises(ValueError):
        entropy.hessian_relative_entropy_2nd(q, p, grad_p, hess_p, eps_p=-1)
