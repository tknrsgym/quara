import numpy as np
import numpy.testing as npt
import pytest

from quara.data_analysis.weighted_probability_based_squared_error import (
    WeightedProbabilityBasedSquaredError,
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

weight_matrices = [
    np.array([[1, 0], [0, 1]], dtype=np.float64),
    np.array([[1, 0], [0, 1]], dtype=np.float64),
    np.array([[2, 0], [0, 2]], dtype=np.float64),
]


class TestWeightedProbabilityBasedSquaredErrorFunction:
    def test_access_weight_matrices(self):
        weight_matrices = [
            np.eye(4, dtype=np.float64),
            np.eye(4, dtype=np.float64),
            np.array([[0, 1], [1, 0]], dtype=np.float64),
        ]
        loss_func = WeightedProbabilityBasedSquaredError(
            4, weight_matrices=weight_matrices
        )
        for a, e in zip(loss_func.weight_matrices, weight_matrices):
            npt.assert_almost_equal(a, e, decimal=15)

        # Test that "weight_matrices" cannot be updated
        with pytest.raises(AttributeError):
            loss_func.weight_matrices = weight_matrices

        # Test that "weight_matrices" are not real
        weight_matrices = [
            np.eye(4, dtype=np.float64),
            np.eye(4, dtype=np.float64),
            np.eye(4, dtype=np.complex128),
        ]
        with pytest.raises(ValueError):
            WeightedProbabilityBasedSquaredError(4, weight_matrices=weight_matrices)

        # Test that "weight_matrices" are not symmetric
        weight_matrices = [
            np.eye(4, dtype=np.float64),
            np.eye(4, dtype=np.float64),
            np.array([[1, 1], [0, 1]], dtype=np.float64),
        ]
        with pytest.raises(ValueError):
            WeightedProbabilityBasedSquaredError(4, weight_matrices=weight_matrices)

    def test_set_weight_matrices(self):
        loss_func = WeightedProbabilityBasedSquaredError(4)
        weight_matrices = [
            np.eye(4, dtype=np.float64),
            np.eye(4, dtype=np.float64),
            np.array([[0, 1], [1, 0]], dtype=np.float64),
        ]
        loss_func.set_weight_matrices(weight_matrices)
        for a, e in zip(loss_func.weight_matrices, weight_matrices):
            npt.assert_almost_equal(a, e, decimal=15)

        # Test that "weight_matrices" are not real
        weight_matrices = [
            np.eye(4, dtype=np.float64),
            np.eye(4, dtype=np.float64),
            np.eye(4, dtype=np.complex128),
        ]
        with pytest.raises(ValueError):
            loss_func.set_weight_matrices(weight_matrices)

        # Test that "weight_matrices" are not symmetric
        weight_matrices = [
            np.eye(4, dtype=np.float64),
            np.eye(4, dtype=np.float64),
            np.array([[1, 1], [0, 1]], dtype=np.float64),
        ]
        with pytest.raises(ValueError):
            loss_func.set_weight_matrices(weight_matrices)

    def test_update_on_value_true(self):
        # tests _update_on_value_true, _update_on_gradient_true, _update_on_hessian_true and setter/getter of ProbabilityBasedLossFunction in this test case.

        # case1: set_func_prob_dists -> set_func_gradient_dists -> set_func_hessian_dists
        loss_func = WeightedProbabilityBasedSquaredError(4, prob_dists_q=prob_dists_q)
        assert loss_func.on_value == False
        assert loss_func.on_gradient == False
        assert loss_func.on_hessian == False
        assert loss_func.on_func_prob_dists == False
        assert loss_func.on_func_gradient_dists == False
        assert loss_func.on_func_hessian_dists == False

        loss_func.set_func_prob_dists(func_prob_dists())
        assert loss_func.on_value == True
        assert loss_func.on_gradient == False
        assert loss_func.on_hessian == False
        assert loss_func.on_func_prob_dists == True
        assert loss_func.on_func_gradient_dists == False
        assert loss_func.on_func_hessian_dists == False
        assert loss_func.size_prob_dists() == 3
        assert len(loss_func.func_prob_dists) == 3

        loss_func.set_func_gradient_dists(func_gradient_prob_dists())
        assert loss_func.on_value == True
        assert loss_func.on_gradient == True
        assert loss_func.on_hessian == False
        assert loss_func.on_func_prob_dists == True
        assert loss_func.on_func_gradient_dists == True
        assert loss_func.on_func_hessian_dists == False
        assert loss_func.size_prob_dists() == 3
        assert len(loss_func.func_prob_dists) == 3
        assert len(loss_func.func_gradient_dists) == 3

        loss_func.set_func_hessian_dists(func_hessian_prob_dists())
        assert loss_func.on_value == True
        assert loss_func.on_gradient == True
        assert loss_func.on_hessian == True
        assert loss_func.on_func_prob_dists == True
        assert loss_func.on_func_gradient_dists == True
        assert loss_func.on_func_hessian_dists == True
        assert loss_func.size_prob_dists() == 3
        assert len(loss_func.func_prob_dists) == 3
        assert len(loss_func.func_gradient_dists) == 3
        assert len(loss_func.func_hessian_dists) == 3

        # case2: set_func_hessian_dists -> set_func_gradient_dists -> set_func_prob_dists
        loss_func = WeightedProbabilityBasedSquaredError(4, prob_dists_q=prob_dists_q)
        assert loss_func.on_value == False
        assert loss_func.on_gradient == False
        assert loss_func.on_hessian == False
        assert loss_func.on_func_prob_dists == False
        assert loss_func.on_func_gradient_dists == False
        assert loss_func.on_func_hessian_dists == False

        loss_func.set_func_hessian_dists(func_hessian_prob_dists())
        assert loss_func.on_value == False
        assert loss_func.on_gradient == False
        assert loss_func.on_hessian == False
        assert loss_func.on_func_prob_dists == False
        assert loss_func.on_func_gradient_dists == False
        assert loss_func.on_func_hessian_dists == True
        assert len(loss_func.func_hessian_dists) == 3

        loss_func.set_func_gradient_dists(func_gradient_prob_dists())
        assert loss_func.on_value == False
        assert loss_func.on_gradient == False
        assert loss_func.on_hessian == False
        assert loss_func.on_func_prob_dists == False
        assert loss_func.on_func_gradient_dists == True
        assert loss_func.on_func_hessian_dists == True
        assert len(loss_func.func_gradient_dists) == 3
        assert len(loss_func.func_hessian_dists) == 3

        loss_func.set_func_prob_dists(func_prob_dists())
        assert loss_func.on_value == True
        assert loss_func.on_gradient == True
        assert loss_func.on_hessian == True
        assert loss_func.on_func_prob_dists == True
        assert loss_func.on_func_gradient_dists == True
        assert loss_func.on_func_hessian_dists == True
        assert loss_func.size_prob_dists() == 3
        assert len(loss_func.func_prob_dists) == 3
        assert len(loss_func.func_gradient_dists) == 3
        assert len(loss_func.func_hessian_dists) == 3

    def test_value(self):
        func = WeightedProbabilityBasedSquaredError(
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
        npt.assert_almost_equal(actual, 0.005, decimal=15)

        # case3: var = [1, 0, 0, 0.9]/sqrt(2), weight_matrices = {I, I, 2I}
        func = WeightedProbabilityBasedSquaredError(
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
        func = WeightedProbabilityBasedSquaredError(
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
        func = WeightedProbabilityBasedSquaredError(
            4,
            func_prob_dists(),
            func_gradient_prob_dists(),
            func_hessian_prob_dists(),
            prob_dists_q,
            weight_matrices=weight_matrices,
        )
        var = np.array([1, 0, 0, 0.9], dtype=np.float64) / np.sqrt(2)
        actual = func.gradient(var)
        expected = np.array([0.0, 0.0, 0.0, -2 * np.sqrt(2) / 10], dtype=np.float64)
        npt.assert_almost_equal(actual, expected, decimal=15)

    def test_hessian(self):
        func = WeightedProbabilityBasedSquaredError(
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
                [6.0, 0.0, 0.0, 0.0],
                [0.0, 2.0, 0.0, 0.0],
                [0.0, 0.0, 2.0, 0.0],
                [0.0, 0.0, 0.0, 2.0],
            ],
            dtype=np.float64,
        )
        npt.assert_almost_equal(actual, expected, decimal=14)

        # case2: var = [1, 0, 0, 0.9]/sqrt(2)
        var = np.array([1, 0, 0, 0.9], dtype=np.float64) / np.sqrt(2)
        actual = func.hessian(var)
        expected = np.array(
            [
                [6.0, 0.0, 0.0, 0.0],
                [0.0, 2.0, 0.0, 0.0],
                [0.0, 0.0, 2.0, 0.0],
                [0.0, 0.0, 0.0, 2.0],
            ],
            dtype=np.float64,
        )
        npt.assert_almost_equal(actual, expected, decimal=14)

        # case3: var = [1, 0, 0, 0.9]/sqrt(2), weight_matrices = {I, I, 2I}
        func = WeightedProbabilityBasedSquaredError(
            4,
            func_prob_dists(),
            func_gradient_prob_dists(),
            func_hessian_prob_dists(),
            prob_dists_q,
            weight_matrices=weight_matrices,
        )
        var = np.array([1, 0, 0, 0.9], dtype=np.float64) / np.sqrt(2)
        actual = func.hessian(var)
        expected = np.array(
            [
                [8.0, 0.0, 0.0, 0.0],
                [0.0, 2.0, 0.0, 0.0],
                [0.0, 0.0, 2.0, 0.0],
                [0.0, 0.0, 0.0, 4.0],
            ],
            dtype=np.float64,
        )
        npt.assert_almost_equal(actual, expected, decimal=14)

    def test_on_prob_dists_q_False(self):
        loss_func = WeightedProbabilityBasedSquaredError(
            4, func_prob_dists(), func_gradient_prob_dists(), func_hessian_prob_dists(),
        )
        assert loss_func.on_value == False
        assert loss_func.on_gradient == False
        assert loss_func.on_hessian == False
        assert loss_func.on_func_prob_dists == True
        assert loss_func.on_func_gradient_dists == True
        assert loss_func.on_func_hessian_dists == True
