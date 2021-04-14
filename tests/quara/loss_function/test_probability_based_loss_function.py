from abc import abstractmethod
from typing import List

import numpy as np
import numpy.testing as npt
import pytest

from quara.loss_function.probability_based_loss_function import (
    ProbabilityBasedLossFunction,
)
from quara.loss_function.weighted_probability_based_squared_error import (
    WeightedProbabilityBasedSquaredError,
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
    def _process(var: np.array) -> np.float64:
        return np.dot(mat_p[index], var)

    return _process


def func_prob_dists(x: int = None):
    funcs = []
    for index in range(len(mat_p)):
        funcs.append(_func_prob_dist(index))
    return funcs


prob_dists_q = [
    np.array([0.5, 0.5], dtype=np.float64),
    np.array([0.5, 0.5], dtype=np.float64),
    np.array([1.0, 0.0], dtype=np.float64),
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


class TestProbabilityBasedLossFunction:
    def test_is_option_sufficient(self):
        loss_func = ProbabilityBasedLossFunction(4)
        assert loss_func.is_option_sufficient() == True

    def test_access_prob_dists_q(self):
        loss_func = ProbabilityBasedLossFunction(4)
        assert loss_func.prob_dists_q is None

        loss_func = ProbabilityBasedLossFunction(4, prob_dists_q=prob_dists_q)
        assert len(loss_func.prob_dists_q) == 3

        # Test that "prob_dists_q" cannot be updated
        with pytest.raises(AttributeError):
            loss_func.prob_dists_q = prob_dists_q

    def test_access_on_prob_dists_q(self):
        loss_func = ProbabilityBasedLossFunction(4)
        assert loss_func.on_prob_dists_q == False

        loss_func = ProbabilityBasedLossFunction(4, prob_dists_q=prob_dists_q)
        assert loss_func.on_prob_dists_q == True

        # Test that "on_prob_dists_q" cannot be updated
        with pytest.raises(AttributeError):
            loss_func.on_prob_dists_q = False

    def test_set_prob_dists_q(self):
        qt = get_test_qst(on_para_eq_constraint=False)
        loss_func = WeightedProbabilityBasedSquaredError(qt.num_variables)
        assert loss_func.prob_dists_q is None

        loss_func.set_prob_dists_q(prob_dists_q)
        assert loss_func.on_prob_dists_q == True
        assert len(loss_func.prob_dists_q) == 3

    def test_set_func_prob_dists_from_standard_qt___num_var(self):
        # Arrange
        qt = get_test_qst(on_para_eq_constraint=True)
        loss_func = WeightedProbabilityBasedSquaredError()
        assert loss_func.on_func_prob_dists == False
        assert loss_func.on_func_gradient_prob_dists == False
        assert loss_func.on_func_hessian_prob_dists == False
        assert loss_func.on_prob_dists_q == False

        # Act
        loss_func.set_func_prob_dists_from_standard_qt(qt)

        # Assert
        assert loss_func.num_var == 3
        assert loss_func.on_func_prob_dists == True
        func_prob_dists = loss_func.func_prob_dists
        expected = [[0.5, 0.5], [0.5, 0.5], [1, 0]]
        var = np.array([0, 0, 1], dtype=np.float64) / np.sqrt(2)  # len(var) = 3
        for index in range(len(func_prob_dists)):
            npt.assert_almost_equal(
                func_prob_dists[index](var), expected[index], decimal=15
            )

    def test_set_func_prob_dists_from_standard_qt___on_para_eq_constraint_True(self):
        # Arrange
        qt = get_test_qst(on_para_eq_constraint=True)
        loss_func = WeightedProbabilityBasedSquaredError(qt.num_variables)
        assert loss_func.on_func_prob_dists == False
        assert loss_func.on_func_gradient_prob_dists == False
        assert loss_func.on_func_hessian_prob_dists == False
        assert loss_func.on_prob_dists_q == False

        # Act
        loss_func.set_func_prob_dists_from_standard_qt(qt)

        # Assert
        assert loss_func.on_func_prob_dists == True
        func_prob_dists = loss_func.func_prob_dists
        expected = [[0.5, 0.5], [0.5, 0.5], [1, 0]]
        var = np.array([0, 0, 1], dtype=np.float64) / np.sqrt(2)  # len(var) = 3
        for index in range(len(func_prob_dists)):
            npt.assert_almost_equal(
                func_prob_dists[index](var), expected[index], decimal=15
            )

    def test_set_func_prob_dists_from_standard_qt___on_para_eq_constraint_False(self):
        # Arrange
        qt = get_test_qst(on_para_eq_constraint=False)
        loss_func = WeightedProbabilityBasedSquaredError(qt.num_variables)
        assert loss_func.on_func_prob_dists == False
        assert loss_func.on_func_gradient_prob_dists == False
        assert loss_func.on_func_hessian_prob_dists == False
        assert loss_func.on_prob_dists_q == False

        # Act
        loss_func.set_func_prob_dists_from_standard_qt(qt)

        # Assert
        assert loss_func.on_func_prob_dists == True
        func_prob_dists = loss_func.func_prob_dists
        expected = [[0.5, 0.5], [0.5, 0.5], [1, 0]]
        var = np.array([1, 0, 0, 1], dtype=np.float64) / np.sqrt(2)  # len(var) = 4
        for index in range(len(func_prob_dists)):
            npt.assert_almost_equal(
                func_prob_dists[index](var), expected[index], decimal=15
            )

    def test_set_func_gradient_prob_dists_from_standard_qt___on_para_eq_constraint_True(
        self,
    ):
        # Arrange
        qt = get_test_qst(on_para_eq_constraint=True)
        loss_func = WeightedProbabilityBasedSquaredError(qt.num_variables)
        assert loss_func.on_func_prob_dists == False
        assert loss_func.on_func_gradient_prob_dists == False
        assert loss_func.on_func_hessian_prob_dists == False
        assert loss_func.on_prob_dists_q == False

        # Act
        loss_func.set_func_gradient_prob_dists_from_standard_qt(qt)

        # Assert
        assert loss_func.on_func_gradient_prob_dists == True
        func_gradient_dists = loss_func.func_gradient_prob_dists
        expected = [
            [
                [1 / np.sqrt(2), -1 / np.sqrt(2)],
                [0, 0],
                [0, 0],
            ],
            [
                [0, 0],
                [1 / np.sqrt(2), -1 / np.sqrt(2)],
                [0, 0],
            ],
            [
                [0, 0],
                [0, 0],
                [1 / np.sqrt(2), -1 / np.sqrt(2)],
            ],
        ]
        var = np.array([0, 0, 1], dtype=np.float64) / np.sqrt(2)  # len(var) = 3
        for index in range(len(func_gradient_dists)):
            for var_index in range(qt.num_variables):
                npt.assert_almost_equal(
                    func_gradient_dists[index](var_index, var),
                    expected[index][var_index],
                    decimal=15,
                )

    def test_set_func_gradient_prob_dists_from_standard_qt___on_para_eq_constraint_False(
        self,
    ):
        # Arrange
        qt = get_test_qst(on_para_eq_constraint=False)
        loss_func = WeightedProbabilityBasedSquaredError(qt.num_variables)
        assert loss_func.on_func_prob_dists == False
        assert loss_func.on_func_gradient_prob_dists == False
        assert loss_func.on_func_hessian_prob_dists == False
        assert loss_func.on_prob_dists_q == False

        # Act
        loss_func.set_func_gradient_prob_dists_from_standard_qt(qt)

        # Assert
        assert loss_func.on_func_gradient_prob_dists == True
        func_gradient_dists = loss_func.func_gradient_prob_dists
        expected = [
            [
                [1 / np.sqrt(2), 1 / np.sqrt(2)],
                [1 / np.sqrt(2), -1 / np.sqrt(2)],
                [0, 0],
                [0, 0],
            ],
            [
                [1 / np.sqrt(2), 1 / np.sqrt(2)],
                [0, 0],
                [1 / np.sqrt(2), -1 / np.sqrt(2)],
                [0, 0],
            ],
            [
                [1 / np.sqrt(2), 1 / np.sqrt(2)],
                [0, 0],
                [0, 0],
                [1 / np.sqrt(2), -1 / np.sqrt(2)],
            ],
        ]
        var = np.array([1, 0, 0, 1], dtype=np.float64) / np.sqrt(2)  # len(var) = 4
        for index in range(len(func_gradient_dists)):
            for var_index in range(qt.num_variables):
                npt.assert_almost_equal(
                    func_gradient_dists[index](var_index, var),
                    expected[index][var_index],
                    decimal=15,
                )

    def test_set_func_hessian_prob_dists_from_standard_qt___on_para_eq_constraint_True(
        self,
    ):
        # Arrange
        qt = get_test_qst(on_para_eq_constraint=True)
        loss_func = WeightedProbabilityBasedSquaredError(qt.num_variables)
        assert loss_func.on_func_prob_dists == False
        assert loss_func.on_func_gradient_prob_dists == False
        assert loss_func.on_func_hessian_prob_dists == False
        assert loss_func.on_prob_dists_q == False

        # Act
        loss_func.set_func_hessian_prob_dists_from_standard_qt(qt)

        # Assert
        assert loss_func.on_func_hessian_prob_dists == True
        func_hessian_prob_dists = loss_func.func_hessian_prob_dists
        expected = [0, 0]
        var = np.array([0, 0, 1], dtype=np.float64) / np.sqrt(2)  # len(var) = 3
        for index in range(len(func_hessian_prob_dists)):
            for var_index1 in range(qt.num_variables):
                for var_index2 in range(qt.num_variables):
                    npt.assert_almost_equal(
                        func_hessian_prob_dists[index](var_index1, var_index2, var),
                        expected,
                        decimal=15,
                    )

    def test_set_func_hessian_prob_dists_from_standard_qt___on_para_eq_constraint_False(
        self,
    ):
        # Arrange
        qt = get_test_qst(on_para_eq_constraint=False)
        loss_func = WeightedProbabilityBasedSquaredError(qt.num_variables)
        assert loss_func.on_func_prob_dists == False
        assert loss_func.on_func_gradient_prob_dists == False
        assert loss_func.on_func_hessian_prob_dists == False
        assert loss_func.on_prob_dists_q == False

        # Act
        loss_func.set_func_hessian_prob_dists_from_standard_qt(qt)

        # Assert
        assert loss_func.on_func_hessian_prob_dists == True
        func_hessian_prob_dists = loss_func.func_hessian_prob_dists
        expected = [0, 0]
        var = np.array([1, 0, 0, 1], dtype=np.float64) / np.sqrt(2)  # len(var) = 4
        for index in range(len(func_hessian_prob_dists)):
            for var_index1 in range(qt.num_variables):
                for var_index2 in range(qt.num_variables):
                    npt.assert_almost_equal(
                        func_hessian_prob_dists[index](var_index1, var_index2, var),
                        expected,
                        decimal=15,
                    )

    def test_set_func_xxx_prob_dists_from_standard_qt(self):
        # tests set_func_prob_dists_from_standard_qt, set_func_gradient_prob_dists_from_standard_qt, set_func_hessian_prob_dists_from_standard_qt

        # case1: set_prob_dists_q -> set_func_prob_dists_from_standard_qt -> set_func_gradient_prob_dists_from_standard_qt -> set_func_hessian_prob_dists_from_standard_qt
        qt = get_test_qst(on_para_eq_constraint=False)
        loss_func = WeightedProbabilityBasedSquaredError(qt.num_variables)
        assert loss_func.on_value == False
        assert loss_func.on_gradient == False
        assert loss_func.on_hessian == False
        assert loss_func.on_func_prob_dists == False
        assert loss_func.on_func_gradient_prob_dists == False
        assert loss_func.on_func_hessian_prob_dists == False
        assert loss_func.on_prob_dists_q == False

        loss_func.set_prob_dists_q(prob_dists_q)
        assert loss_func.on_value == False
        assert loss_func.on_gradient == False
        assert loss_func.on_hessian == False
        assert loss_func.on_func_prob_dists == False
        assert loss_func.on_func_gradient_prob_dists == False
        assert loss_func.on_func_hessian_prob_dists == False
        assert loss_func.on_prob_dists_q == True

        loss_func.set_func_prob_dists_from_standard_qt(qt)
        assert loss_func.on_value == True
        assert loss_func.on_gradient == False
        assert loss_func.on_hessian == False
        assert loss_func.on_func_prob_dists == True
        assert loss_func.on_func_gradient_prob_dists == False
        assert loss_func.on_func_hessian_prob_dists == False
        assert loss_func.on_prob_dists_q == True
        assert loss_func.size_prob_dists() == 3
        assert len(loss_func.func_prob_dists) == 3

        loss_func.set_func_gradient_prob_dists_from_standard_qt(qt)
        assert loss_func.on_value == True
        assert loss_func.on_gradient == True
        assert loss_func.on_hessian == False
        assert loss_func.on_func_prob_dists == True
        assert loss_func.on_func_gradient_prob_dists == True
        assert loss_func.on_func_hessian_prob_dists == False
        assert loss_func.on_prob_dists_q == True
        assert loss_func.size_prob_dists() == 3
        assert len(loss_func.func_prob_dists) == 3
        assert len(loss_func.func_gradient_prob_dists) == 3

        loss_func.set_func_hessian_prob_dists_from_standard_qt(qt)
        assert loss_func.on_value == True
        assert loss_func.on_gradient == True
        assert loss_func.on_hessian == True
        assert loss_func.on_func_prob_dists == True
        assert loss_func.on_func_gradient_prob_dists == True
        assert loss_func.on_func_hessian_prob_dists == True
        assert loss_func.on_prob_dists_q == True
        assert loss_func.size_prob_dists() == 3
        assert len(loss_func.func_prob_dists) == 3
        assert len(loss_func.func_gradient_prob_dists) == 3
        assert len(loss_func.func_hessian_prob_dists) == 3

        # case2: set_func_hessian_prob_dists_from_standard_qt -> set_func_gradient_prob_dists_from_standard_qt -> set_func_prob_dists_from_standard_qt -> set_prob_dists_q
        qt = get_test_qst(on_para_eq_constraint=False)
        loss_func = WeightedProbabilityBasedSquaredError(qt.num_variables)
        assert loss_func.on_value == False
        assert loss_func.on_gradient == False
        assert loss_func.on_hessian == False
        assert loss_func.on_func_prob_dists == False
        assert loss_func.on_func_gradient_prob_dists == False
        assert loss_func.on_func_hessian_prob_dists == False
        assert loss_func.on_prob_dists_q == False

        loss_func.set_func_hessian_prob_dists_from_standard_qt(qt)
        assert loss_func.on_value == False
        assert loss_func.on_gradient == False
        assert loss_func.on_hessian == False
        assert loss_func.on_func_prob_dists == False
        assert loss_func.on_func_gradient_prob_dists == False
        assert loss_func.on_func_hessian_prob_dists == True
        assert loss_func.on_prob_dists_q == False
        assert len(loss_func.func_hessian_prob_dists) == 3

        loss_func.set_func_gradient_prob_dists_from_standard_qt(qt)
        assert loss_func.on_value == False
        assert loss_func.on_gradient == False
        assert loss_func.on_hessian == False
        assert loss_func.on_func_prob_dists == False
        assert loss_func.on_func_gradient_prob_dists == True
        assert loss_func.on_func_hessian_prob_dists == True
        assert loss_func.on_prob_dists_q == False
        assert len(loss_func.func_gradient_prob_dists) == 3
        assert len(loss_func.func_hessian_prob_dists) == 3

        loss_func.set_func_prob_dists_from_standard_qt(qt)
        assert loss_func.on_value == False
        assert loss_func.on_gradient == False
        assert loss_func.on_hessian == False
        assert loss_func.on_func_prob_dists == True
        assert loss_func.on_func_gradient_prob_dists == True
        assert loss_func.on_func_hessian_prob_dists == True
        assert loss_func.on_prob_dists_q == False
        assert loss_func.size_prob_dists() == 3
        assert len(loss_func.func_prob_dists) == 3
        assert len(loss_func.func_gradient_prob_dists) == 3
        assert len(loss_func.func_hessian_prob_dists) == 3

        loss_func.set_prob_dists_q(prob_dists_q)
        assert loss_func.on_value == True
        assert loss_func.on_gradient == True
        assert loss_func.on_hessian == True
        assert loss_func.on_func_prob_dists == True
        assert loss_func.on_func_gradient_prob_dists == True
        assert loss_func.on_func_hessian_prob_dists == True
        assert loss_func.on_prob_dists_q == True
        assert loss_func.size_prob_dists() == 3
        assert len(loss_func.func_prob_dists) == 3
        assert len(loss_func.func_gradient_prob_dists) == 3
        assert len(loss_func.func_hessian_prob_dists) == 3
