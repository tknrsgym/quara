import numpy as np
import numpy.testing as npt
import pytest

from quara.loss_function.standard_qtomography_based_weighted_relative_entropy import (
    StandardQTomographyBasedWeightedRelativeEntropy,
    StandardQTomographyBasedWeightedRelativeEntropyOption,
)
from quara.loss_function.weighted_relative_entropy import (
    WeightedRelativeEntropy,
    WeightedRelativeEntropyOption,
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
from quara.protocol.qtomography.standard.standard_qst import StandardQst


# parameters for test
mat_p = (
    np.array(
        [
            [1, 1, 0, 0],
            [1, -1, 0, 0],
            [1, 0, 1, 0],
            [1, 0, -1, 0],
            [1, 0, 0, 1],
            [1, 0, 0, -1],
        ],
        dtype=np.float64,
    )
    / np.sqrt(2)
)


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


def get_test_qst(on_para_eq_constraint=True):
    e_sys = ElementalSystem(0, get_normalized_pauli_basis())
    c_sys = CompositeSystem([e_sys])

    povm_x = get_x_povm(c_sys)
    povm_y = get_y_povm(c_sys)
    povm_z = get_z_povm(c_sys)
    povms = [povm_x, povm_y, povm_z]

    qst = StandardQst(povms, on_para_eq_constraint=on_para_eq_constraint, seed=7)
    return qst


class TestStandardQTomographyBasedWeightedRelativeEntropy:
    def test_access_weights(self):
        weights = [1.0, 2.0, 3.0]
        loss_func = StandardQTomographyBasedWeightedRelativeEntropy(4, weights=weights)
        for a, e in zip(loss_func.weights, weights):
            npt.assert_almost_equal(a, e, decimal=15)

        # Test that "weights" cannot be updated
        with pytest.raises(AttributeError):
            loss_func.weights = weights

        # Test that "weights" are not real
        weights = [1.0, 2.0, 3.0j]
        with pytest.raises(ValueError):
            StandardQTomographyBasedWeightedRelativeEntropy(4, weights=weights)

    def test_set_weights(self):
        loss_func = StandardQTomographyBasedWeightedRelativeEntropy(4)
        weights = [1.0, 2.0, 3.0]
        loss_func.set_weights(weights)
        for a, e in zip(loss_func.weights, weights):
            npt.assert_almost_equal(a, e, decimal=15)

        # Test that "weights" are not real
        weights = [1.0, 2.0, 3.0j]
        with pytest.raises(ValueError):
            loss_func.set_weights(weights)

    def test_is_option_sufficient(self):
        func = StandardQTomographyBasedWeightedRelativeEntropy(
            4,
            prob_dists_q,
        )
        assert func.is_option_sufficient() == True

    def test_value(self):
        qt = get_test_qst(on_para_eq_constraint=False)

        loss_func = StandardQTomographyBasedWeightedRelativeEntropy(
            4,
            prob_dists_q,
        )
        loss_func.set_func_prob_dists_from_standard_qt(qt)

        # case1: var = [1, 0, 0, 1]/sqrt(2)
        var = np.array([1, 0, 0, 1], dtype=np.float64) / np.sqrt(2)
        actual = loss_func.value(var)
        expected = 0
        npt.assert_almost_equal(actual, 0, decimal=15)

        # case2: var = [1, 0, 0, 0.9]/sqrt(2)
        var = np.array([1, 0, 0, 0.9], dtype=np.float64) / np.sqrt(2)
        actual = loss_func.value(var)
        expected = np.log(2 / 1.9)
        npt.assert_almost_equal(actual, expected, decimal=15)

        # case3: var = [1, 0, 0, 0.9]/sqrt(2), weights = [1.0, 1.0, 2.0]
        loss_func = StandardQTomographyBasedWeightedRelativeEntropy(
            4,
            prob_dists_q,
            weights=weights,
        )
        loss_func.set_func_prob_dists_from_standard_qt(qt)

        var = np.array([1, 0, 0, 0.9], dtype=np.float64) / np.sqrt(2)
        actual = loss_func.value(var)
        expected = 2 * np.log(2 / 1.9)
        npt.assert_almost_equal(actual, expected, decimal=14)

        # case4: var = [0, 0, 1]/sqrt(2), on_para_eq_constraint=True
        qt = get_test_qst(on_para_eq_constraint=True)
        loss_func = StandardQTomographyBasedWeightedRelativeEntropy(
            qt.num_variables, prob_dists_q=prob_dists_q
        )
        loss_func.set_func_prob_dists_from_standard_qt(qt)

        var = np.array([0, 0, 1], dtype=np.float64) / np.sqrt(2)
        actual = loss_func.value(var)
        expected = 0
        npt.assert_almost_equal(actual, expected, decimal=15)

    def test_gradient(self):
        qt = get_test_qst(on_para_eq_constraint=False)

        loss_func = StandardQTomographyBasedWeightedRelativeEntropy(
            4,
            prob_dists_q,
        )
        loss_func.set_func_prob_dists_from_standard_qt(qt)
        loss_func.set_func_gradient_prob_dists_from_standard_qt(qt)

        # case1: var = [1, 0, 0, 1]/sqrt(2)
        var = np.array([1, 0, 0, 1], dtype=np.float64) / np.sqrt(2)
        actual = loss_func.gradient(var)
        expected = -np.array([5, 0, 0, 1], dtype=np.float64) / np.sqrt(2)
        npt.assert_almost_equal(actual, expected, decimal=15)

        # case2: var = [1, 0, 0, 0.9]/sqrt(2)
        var = np.array([1, 0, 0, 0.9], dtype=np.float64) / np.sqrt(2)
        actual = loss_func.gradient(var)
        expected = -np.array([4 + 2 / 1.9, 0, 0, 2 / 1.9], dtype=np.float64) / np.sqrt(
            2
        )
        npt.assert_almost_equal(actual, expected, decimal=15)

        # case3: var = [1, 0, 0, 0.9]/sqrt(2), weights = [1.0, 1.0, 2.0], weights = [1.0, 1.0, 2.0]
        loss_func = StandardQTomographyBasedWeightedRelativeEntropy(
            4,
            prob_dists_q,
            weights=weights,
        )
        loss_func.set_func_prob_dists_from_standard_qt(qt)
        loss_func.set_func_gradient_prob_dists_from_standard_qt(qt)

        var = np.array([1, 0, 0, 0.9], dtype=np.float64) / np.sqrt(2)
        actual = loss_func.gradient(var)
        expected = -np.array([4 + 4 / 1.9, 0, 0, 4 / 1.9], dtype=np.float64) / np.sqrt(
            2
        )
        npt.assert_almost_equal(actual, expected, decimal=15)

        # case4: var = [0, 0, 1]/sqrt(2), on_para_eq_constraint=True
        qt = get_test_qst(on_para_eq_constraint=True)
        loss_func = StandardQTomographyBasedWeightedRelativeEntropy(
            qt.num_variables, prob_dists_q=prob_dists_q
        )
        loss_func.set_func_prob_dists_from_standard_qt(qt)
        loss_func.set_func_gradient_prob_dists_from_standard_qt(qt)

        var = np.array([0, 0, 1], dtype=np.float64) / np.sqrt(2)
        actual = loss_func.gradient(var)
        expected = -np.array([0, 0, 1], dtype=np.float64) / np.sqrt(2)
        npt.assert_almost_equal(actual, expected, decimal=15)
