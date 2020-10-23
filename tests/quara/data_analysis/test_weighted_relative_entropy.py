import numpy as np
import numpy.testing as npt
import pytest

from quara.data_analysis.weighted_relative_entropy import (
    WeightedRelativeEntropy,
    WeightedRelativeEntropyOption,
)


# parameters for test
mat_p = np.array(
    [
        [1, 1, 0, 0],
        [1, -1, 0, 0],
        [1, 0, 1, 0],
        [1, 0, -1, 0],
        [1, 0, 0, 1],
        [1, 0, 0, -1],
    ],
    dtype=np.float64,
) / np.sqrt(2)


def _func_prob_dist(index: int):
    def _process(var: np.array) -> np.array:
        return mat_p[2 * index : 2 * (index + 1)] @ var

    return _process


def func_prob_dists(x: int = None):
    funcs = []
    for index in range(int(mat_p.shape[0] / 2)):
        funcs.append(_func_prob_dist(index))
    return funcs


def _func_gradient_prob_dist(index: int):
    def _process(alpha: int, var: np.array) -> np.array:
        return np.array(
            [mat_p[2 * index, alpha], mat_p[2 * index + 1, alpha]], dtype=np.float64
        )

    return _process


def func_gradient_prob_dists():
    funcs = []
    for index in range(3):
        funcs.append(_func_gradient_prob_dist(index))
    return funcs


def _func_hessian_prob_dist(index: int):
    def _process(alpha: int, beta: int, var: np.array):
        return np.array([0.0, 0.0], dtype=np.float64)

    return _process


def func_hessian_prob_dists():
    funcs = []
    for index in range(3):
        funcs.append(_func_hessian_prob_dist(index))
    return funcs


prob_dists_q = [
    np.array([0.5, 0.5], dtype=np.float64),
    np.array([0.5, 0.5], dtype=np.float64),
    np.array([1.0, 0.0], dtype=np.float64),
]

weights = [1.0, 1.0, 2.0]


class TestWeightedRelativeEntropy:
    def test_value(self):
        func = WeightedRelativeEntropy(
            4,
            func_prob_dists(),
            func_gradient_prob_dists(),
            func_hessian_prob_dists(),
            prob_dists_q,
        )

        # case1: var = [1, 0, 0, 1]/sqrt(2)
        var = np.array([1, 0, 0, 1], dtype=np.float64) / np.sqrt(2)
        actual = func.value(var)
        expected = 0
        npt.assert_almost_equal(actual, 0, decimal=15)

        # case2: var = [1, 0, 0, 0.9]/sqrt(2)
        var = np.array([1, 0, 0, 0.9], dtype=np.float64) / np.sqrt(2)
        actual = func.value(var)
        expected = np.log(2 / 1.9)
        npt.assert_almost_equal(actual, expected, decimal=15)

        # case3: var = [1, 0, 0, 0.9]/sqrt(2), weights = [1.0, 1.0, 2.0]
        func = WeightedRelativeEntropy(
            4,
            func_prob_dists(),
            func_gradient_prob_dists(),
            func_hessian_prob_dists(),
            prob_dists_q,
            weights=weights,
        )
        var = np.array([1, 0, 0, 0.9], dtype=np.float64) / np.sqrt(2)
        actual = func.value(var)
        expected = 2 * np.log(2 / 1.9)
        npt.assert_almost_equal(actual, expected, decimal=15)

    def test_gradient(self):
        func = WeightedRelativeEntropy(
            4,
            func_prob_dists(),
            func_gradient_prob_dists(),
            func_hessian_prob_dists(),
            prob_dists_q,
        )

        # case1: var = [1, 0, 0, 1]/sqrt(2)
        var = np.array([1, 0, 0, 1], dtype=np.float64) / np.sqrt(2)
        actual = func.gradient(var)
        expected = -np.array([5, 0, 0, 1], dtype=np.float64) / np.sqrt(2)
        npt.assert_almost_equal(actual, expected, decimal=15)

        # case2: var = [1, 0, 0, 0.9]/sqrt(2)
        var = np.array([1, 0, 0, 0.9], dtype=np.float64) / np.sqrt(2)
        actual = func.gradient(var)
        expected = -np.array([4 + 2 / 1.9, 0, 0, 2 / 1.9], dtype=np.float64) / np.sqrt(
            2
        )
        npt.assert_almost_equal(actual, expected, decimal=15)

        # case3: var = [1, 0, 0, 0.9]/sqrt(2), weights = [1.0, 1.0, 2.0]
        func = WeightedRelativeEntropy(
            4,
            func_prob_dists(),
            func_gradient_prob_dists(),
            func_hessian_prob_dists(),
            prob_dists_q,
            weights=weights,
        )
        var = np.array([1, 0, 0, 0.9], dtype=np.float64) / np.sqrt(2)
        actual = func.gradient(var)
        expected = -np.array([4 + 4 / 1.9, 0, 0, 4 / 1.9], dtype=np.float64) / np.sqrt(
            2
        )
        npt.assert_almost_equal(actual, expected, decimal=15)
