import numpy as np
import numpy.testing as npt
import pytest

from quara.loss_function.weighted_probability_based_squared_error import (
    WeightedProbabilityBasedSquaredError,
    WeightedProbabilityBasedSquaredErrorOption,
)
from quara.minimization_algorithm.projected_gradient_descent_backtracking import (
    ProjectedGradientDescentBacktracking,
    ProjectedGradientDescentBacktrackingOption,
)
from quara.objects.composite_system import CompositeSystem
from quara.objects.elemental_system import ElementalSystem
from quara.objects.matrix_basis import get_normalized_pauli_basis
from quara.objects.povm import (
    Povm,
    get_x_povm,
    get_y_povm,
    get_z_povm,
)
from quara.objects.state import get_z0_1q
from quara.protocol.qtomography.standard.loss_minimization_estimator import (
    LossMinimizationEstimator,
)
from quara.protocol.qtomography.standard.standard_qst import StandardQst


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

data = [
    (10000, np.array([0.5, 0.5], dtype=np.float64)),
    (10000, np.array([0.5, 0.5], dtype=np.float64)),
    (10000, np.array([1.0, 0.0], dtype=np.float64)),
]

weight_matrices = [
    np.array([[1, 0], [0, 1]], dtype=np.float64),
    np.array([[1, 0], [0, 1]], dtype=np.float64),
    np.array([[2, 0], [0, 2]], dtype=np.float64),
]


def get_test_qst(on_para_eq_constraint=True):
    e_sys = ElementalSystem(0, get_normalized_pauli_basis())
    c_sys = CompositeSystem([e_sys])

    povm_x = get_x_povm(c_sys)
    povm_y = get_y_povm(c_sys)
    povm_z = get_z_povm(c_sys)
    povms = [povm_x, povm_y, povm_z]

    qst = StandardQst(povms, on_para_eq_constraint=on_para_eq_constraint, seed=7)
    return qst


class TestWeightedProbabilityBasedSquaredErrorOption:
    def test_access_mode_weight(self):
        loss_option = WeightedProbabilityBasedSquaredErrorOption(mode_weight="identity")
        assert loss_option.mode_weight == "identity"

        loss_option = WeightedProbabilityBasedSquaredErrorOption(mode_weight="custom")
        assert loss_option.mode_weight == "custom"

        loss_option = WeightedProbabilityBasedSquaredErrorOption(weights=[1, 2, 3])
        assert loss_option.mode_weight == "custom"

        loss_option = WeightedProbabilityBasedSquaredErrorOption(
            mode_weight="inverse_sample_covariance"
        )
        assert loss_option.mode_weight == "inverse_sample_covariance"

        loss_option = WeightedProbabilityBasedSquaredErrorOption(
            mode_weight="inverse_unbiased_covariance"
        )
        assert loss_option.mode_weight == "inverse_unbiased_covariance"

        loss_option = WeightedProbabilityBasedSquaredErrorOption(
            mode_weight="unbiased_inverse_covariance"
        )
        assert loss_option.mode_weight == "unbiased_inverse_covariance"

        # Test that "mode_weight" is not specified
        with pytest.raises(ValueError):
            loss_option = WeightedProbabilityBasedSquaredErrorOption()


class TestWeightedProbabilityBasedSquaredErrorFunction:
    def test_access_weight_matrices(self):
        # success(dtype=float)
        weight_matrices = [
            np.eye(4, dtype=float),
            np.eye(4, dtype=float),
            np.array([[0, 1], [1, 0]], dtype=float),
        ]
        loss_func = WeightedProbabilityBasedSquaredError(
            4, weight_matrices=weight_matrices
        )
        for a, e in zip(loss_func.weight_matrices, weight_matrices):
            npt.assert_almost_equal(a, e, decimal=15)

        # success(dtype=np.float64)
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
        # success(dtype=float)
        loss_func = WeightedProbabilityBasedSquaredError(4)
        weight_matrices = [
            np.eye(4, dtype=float),
            np.eye(4, dtype=float),
            np.array([[0, 1], [1, 0]], dtype=float),
        ]
        loss_func.set_weight_matrices(weight_matrices)
        for a, e in zip(loss_func.weight_matrices, weight_matrices):
            npt.assert_almost_equal(a, e, decimal=15)

        # success(dtype=np.float64)
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
        assert loss_func.on_func_gradient_prob_dists == False
        assert loss_func.on_func_hessian_prob_dists == False

        loss_func.set_func_prob_dists(func_prob_dists())
        assert loss_func.on_value == True
        assert loss_func.on_gradient == False
        assert loss_func.on_hessian == False
        assert loss_func.on_func_prob_dists == True
        assert loss_func.on_func_gradient_prob_dists == False
        assert loss_func.on_func_hessian_prob_dists == False
        assert loss_func.size_prob_dists() == 3
        assert len(loss_func.func_prob_dists) == 3

        loss_func.set_func_gradient_prob_dists(func_gradient_prob_dists())
        assert loss_func.on_value == True
        assert loss_func.on_gradient == True
        assert loss_func.on_hessian == False
        assert loss_func.on_func_prob_dists == True
        assert loss_func.on_func_gradient_prob_dists == True
        assert loss_func.on_func_hessian_prob_dists == False
        assert loss_func.size_prob_dists() == 3
        assert len(loss_func.func_prob_dists) == 3
        assert len(loss_func.func_gradient_prob_dists) == 3

        loss_func.set_func_hessian_prob_dists(func_hessian_prob_dists())
        assert loss_func.on_value == True
        assert loss_func.on_gradient == True
        assert loss_func.on_hessian == True
        assert loss_func.on_func_prob_dists == True
        assert loss_func.on_func_gradient_prob_dists == True
        assert loss_func.on_func_hessian_prob_dists == True
        assert loss_func.size_prob_dists() == 3
        assert len(loss_func.func_prob_dists) == 3
        assert len(loss_func.func_gradient_prob_dists) == 3
        assert len(loss_func.func_hessian_prob_dists) == 3

        # case2: set_func_hessian_dists -> set_func_gradient_dists -> set_func_prob_dists
        loss_func = WeightedProbabilityBasedSquaredError(4, prob_dists_q=prob_dists_q)
        assert loss_func.on_value == False
        assert loss_func.on_gradient == False
        assert loss_func.on_hessian == False
        assert loss_func.on_func_prob_dists == False
        assert loss_func.on_func_gradient_prob_dists == False
        assert loss_func.on_func_hessian_prob_dists == False

        loss_func.set_func_hessian_prob_dists(func_hessian_prob_dists())
        assert loss_func.on_value == False
        assert loss_func.on_gradient == False
        assert loss_func.on_hessian == False
        assert loss_func.on_func_prob_dists == False
        assert loss_func.on_func_gradient_prob_dists == False
        assert loss_func.on_func_hessian_prob_dists == True
        assert len(loss_func.func_hessian_prob_dists) == 3

        loss_func.set_func_gradient_prob_dists(func_gradient_prob_dists())
        assert loss_func.on_value == False
        assert loss_func.on_gradient == False
        assert loss_func.on_hessian == False
        assert loss_func.on_func_prob_dists == False
        assert loss_func.on_func_gradient_prob_dists == True
        assert loss_func.on_func_hessian_prob_dists == True
        assert len(loss_func.func_gradient_prob_dists) == 3
        assert len(loss_func.func_hessian_prob_dists) == 3

        loss_func.set_func_prob_dists(func_prob_dists())
        assert loss_func.on_value == True
        assert loss_func.on_gradient == True
        assert loss_func.on_hessian == True
        assert loss_func.on_func_prob_dists == True
        assert loss_func.on_func_gradient_prob_dists == True
        assert loss_func.on_func_hessian_prob_dists == True
        assert loss_func.size_prob_dists() == 3
        assert len(loss_func.func_prob_dists) == 3
        assert len(loss_func.func_gradient_prob_dists) == 3
        assert len(loss_func.func_hessian_prob_dists) == 3

    def test_is_option_sufficient(self):
        func = WeightedProbabilityBasedSquaredError(
            4,
            func_prob_dists(),
            func_gradient_prob_dists(),
            func_hessian_prob_dists(),
            prob_dists_q,
        )
        assert func.is_option_sufficient() == True

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

        # case4: var = [0, 0, 1]/sqrt(2), on_para_eq_constraint=True
        qt = get_test_qst(on_para_eq_constraint=True)
        loss_func = WeightedProbabilityBasedSquaredError(
            qt.num_variables, prob_dists_q=prob_dists_q
        )
        loss_func.set_func_prob_dists_from_standard_qt(qt)

        var = np.array([0, 0, 1], dtype=np.float64) / np.sqrt(2)
        actual = loss_func.value(var)
        expected = 0
        npt.assert_almost_equal(actual, expected, decimal=15)

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

        # case4: var = [0, 0, 1]/sqrt(2), on_para_eq_constraint=True
        qt = get_test_qst(on_para_eq_constraint=True)
        loss_func = WeightedProbabilityBasedSquaredError(
            qt.num_variables, prob_dists_q=prob_dists_q
        )
        loss_func.set_func_prob_dists_from_standard_qt(qt)
        loss_func.set_func_gradient_prob_dists_from_standard_qt(qt)

        var = np.array([0, 0, 1], dtype=np.float64) / np.sqrt(2)
        actual = loss_func.gradient(var)
        expected = np.array([0.0, 0.0, 0.0], dtype=np.float64)
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

        # case4: var = [0, 0, 1]/sqrt(2), on_para_eq_constraint=True
        qt = get_test_qst(on_para_eq_constraint=True)
        loss_func = WeightedProbabilityBasedSquaredError(
            qt.num_variables, prob_dists_q=prob_dists_q
        )
        loss_func.set_func_prob_dists_from_standard_qt(qt)
        loss_func.set_func_gradient_prob_dists_from_standard_qt(qt)
        loss_func.set_func_hessian_prob_dists_from_standard_qt(qt)

        var = np.array([0, 0, 1], dtype=np.float64) / np.sqrt(2)
        actual = loss_func.hessian(var)
        expected = np.array(
            [[2.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 2.0],], dtype=np.float64,
        )
        npt.assert_almost_equal(actual, expected, decimal=15)

    def test_on_prob_dists_q_False(self):
        loss_func = WeightedProbabilityBasedSquaredError(
            4, func_prob_dists(), func_gradient_prob_dists(), func_hessian_prob_dists(),
        )
        assert loss_func.on_value == False
        assert loss_func.on_gradient == False
        assert loss_func.on_hessian == False
        assert loss_func.on_func_prob_dists == True
        assert loss_func.on_func_gradient_prob_dists == True
        assert loss_func.on_func_hessian_prob_dists == True

    def test_set_weight_by_mode(self):
        qt = get_test_qst(on_para_eq_constraint=True)
        func = WeightedProbabilityBasedSquaredError(4)

        # case: mode_weight="identity"
        loss_option = WeightedProbabilityBasedSquaredErrorOption(mode_weight="identity")
        func.set_from_standard_qtomography_option_data(
            qt, loss_option, data, True, True
        )

        assert func.weight_matrices == None

        # case: mode_weight="custom"
        weights = [
            np.array([[1, 2], [2, 3]], dtype=np.float64),
            np.array([[4, 5], [5, 6]], dtype=np.float64),
            np.array([[7, 8], [8, 9]], dtype=np.float64),
        ]
        loss_option = WeightedProbabilityBasedSquaredErrorOption(
            mode_weight="custom", weights=weights
        )
        func.set_from_standard_qtomography_option_data(
            qt, loss_option, data, True, True
        )

        npt.assert_almost_equal(func.weight_matrices, weights, decimal=15)

        # case: mode_weight="inverse_sample_covariance"
        loss_option = WeightedProbabilityBasedSquaredErrorOption(
            mode_weight="inverse_sample_covariance"
        )
        func.set_from_standard_qtomography_option_data(
            qt, loss_option, data, True, True
        )

        expected = [
            np.array([[38461.53846153846, 0], [0, 0]], dtype=np.float64),
            np.array([[38461.53846153846, 0], [0, 0]], dtype=np.float64),
            np.array([[999999.0000010062, 0], [0, 0]], dtype=np.float64),
        ]
        npt.assert_almost_equal(func.weight_matrices, expected, decimal=15)

        # case: mode_weight="inverse_unbiased_covariance"
        loss_option = WeightedProbabilityBasedSquaredErrorOption(
            mode_weight="inverse_unbiased_covariance"
        )
        func.set_from_standard_qtomography_option_data(
            qt, loss_option, data, True, True
        )

        expected = [
            np.array([[38457.84022246239, 0], [0, 0]], dtype=np.float64),
            np.array([[38457.84022246239, 0], [0, 0]], dtype=np.float64),
            np.array([[999998.9999009963, 0], [0, 0]], dtype=np.float64),
        ]
        npt.assert_almost_equal(func.weight_matrices, expected, decimal=15)

    def test_calc_estimate(self):
        empi_dists = [
            (10000, np.array([0.5, 0.5], dtype=np.float64)),
            (10000, np.array([0.5, 0.5], dtype=np.float64)),
            (10000, np.array([1, 0], dtype=np.float64)),
        ]
        loss = WeightedProbabilityBasedSquaredError(4)
        loss_option = WeightedProbabilityBasedSquaredErrorOption(
            mode_weight="inverse_sample_covariance"
        )

        qst = get_test_qst(on_para_eq_constraint=False)
        algo = ProjectedGradientDescentBacktracking()
        algo_option = ProjectedGradientDescentBacktrackingOption(
            on_algo_eq_constraint=True, on_algo_ineq_constraint=True
        )

        estimator = LossMinimizationEstimator()

        actual = estimator.calc_estimate(
            qst,
            empi_dists,
            loss,
            loss_option,
            algo,
            algo_option,
            is_computation_time_required=True,
        )
        expected = [1 / np.sqrt(2), 0, 0, 1 / np.sqrt(2)]
        assert actual.estimated_qoperation.is_physical()
        npt.assert_almost_equal(actual.estimated_var, expected, decimal=7)
