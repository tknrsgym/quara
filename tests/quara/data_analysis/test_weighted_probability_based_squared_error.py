import numpy as np
import numpy.testing as npt

from quara.data_analysis.weighted_probability_based_squared_error import (
    WeightedProbabilityBasedSquaredErrorFunction,
    WeightedProbabilityBasedSquaredErrorOption,
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
    # def _process(var: np.array) -> np.float64:
    #    return np.dot(mat_p[index], var)

    def _process(var: np.array) -> np.array:
        return mat_p[2 * index : 2 * (index + 1)] @ var

    return _process


def func_prob_dists(x: int = None):
    # TODO 確率分布を返す関数はリスト長=3、確率を返す関数はリスト長=3なので、両立しない。要検討。
    funcs = []
    for index in range(int(mat_p.shape[0] / 2)):
        funcs.append(_func_prob_dist(index))
    return funcs


def _func_gradient_prob_dist(index: int):
    def _process(alpha: int, var: np.array) -> np.float64:
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

weight_matrices = [
    np.array([[1, 0], [0, 1]], dtype=np.float64),
    np.array([[1, 0], [0, 1]], dtype=np.float64),
    np.array([[2, 0], [0, 2]], dtype=np.float64),
]


class TestWeightedProbabilityBasedSquaredErrorFunction:
    def test_value(self):
        func = WeightedProbabilityBasedSquaredErrorFunction(
            4,
            func_prob_dists(),
            func_gradient_prob_dists(),
            func_hessian_prob_dists(),
            prob_dists_q,
        )

        # case1: var = [1, 0, 0, 1]/sqrt(2)
        var = np.array([1, 0, 0, 1], dtype=np.float64) / np.sqrt(2)
        actual = func.value(var)
        npt.assert_almost_equal(actual, 0.0, decimal=15)

        # case2: var = [1, 0, 0, 0.9]/sqrt(2)
        var = np.array([1, 0, 0, 0.9], dtype=np.float64) / np.sqrt(2)
        actual = func.value(var)
        print(actual)
        npt.assert_almost_equal(actual, 0.005, decimal=15)

        # case3: var = [1, 0, 0, 0.9]/sqrt(2), weight_matrices = {I, I, 2I}
        func = WeightedProbabilityBasedSquaredErrorFunction(
            4,
            func_prob_dists(),
            func_gradient_prob_dists(),
            func_hessian_prob_dists(),
            prob_dists_q,
            weight_matrices=weight_matrices,
        )
        var = np.array([1, 0, 0, 0.9], dtype=np.float64) / np.sqrt(2)
        actual = func.value(var)
        npt.assert_almost_equal(actual, 0.01, decimal=15)

    def test_gradient(self):
        func = WeightedProbabilityBasedSquaredErrorFunction(
            4,
            func_prob_dists(),
            func_gradient_prob_dists(),
            func_hessian_prob_dists(),
            prob_dists_q,
        )

        # case1: var = [1, 0, 0, 1]/sqrt(2)
        var = np.array([1, 0, 0, 1], dtype=np.float64) / np.sqrt(2)
        actual = func.gradient(var)
        expected = np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float64)
        npt.assert_almost_equal(actual, expected, decimal=15)

        # case2: var = [1, 0, 0, 0.9]/sqrt(2)
        var = np.array([1, 0, 0, 0.9], dtype=np.float64) / np.sqrt(2)
        actual = func.gradient(var)
        expected = np.array([0.0, 0.0, 0.0, -np.sqrt(2) / 10], dtype=np.float64)
        npt.assert_almost_equal(actual, expected, decimal=15)

        # case3: var = [1, 0, 0, 0.9]/sqrt(2), weight_matrices = {I, I, 2I}
        func = WeightedProbabilityBasedSquaredErrorFunction(
            4,
            func_prob_dists(),
            func_gradient_prob_dists(),
            func_hessian_prob_dists(),
            prob_dists_q,
            weight_matrices=weight_matrices,
        )
        var = np.array([1, 0, 0, 0.9], dtype=np.float64) / np.sqrt(2)
        actual = func.gradient(var)
        print(actual)
        expected = np.array([0.0, 0.0, 0.0, -2 * np.sqrt(2) / 10], dtype=np.float64)
        npt.assert_almost_equal(actual, expected, decimal=15)

    def test_hessian(self):
        # TODO
        func = WeightedProbabilityBasedSquaredErrorFunction(
            4,
            func_prob_dists(),
            func_gradient_prob_dists(),
            func_hessian_prob_dists(),
            prob_dists_q,
        )

        # case1: var = [1, 0, 0, 1]/sqrt(2)
        var = np.array([1, 0, 0, 1], dtype=np.float64) / np.sqrt(2)
        actual = func.hessian(var)
        expected = np.array(
            [
                [0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0],
            ],
            dtype=np.float64,
        )
        # npt.assert_almost_equal(actual, expected, decimal=15)
