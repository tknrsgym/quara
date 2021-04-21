import numpy as np
import numpy.testing as npt
import pytest

from quara.objects.composite_system import CompositeSystem
from quara.objects.elemental_system import ElementalSystem
from quara.objects.matrix_basis import get_normalized_pauli_basis
from quara.objects.povm import (
    Povm,
    get_x_povm,
    get_y_povm,
    get_z_povm,
)
from quara.objects.qoperations import SetQOperations
from quara.objects.state import State, get_x0_1q, get_y0_1q, get_z0_1q, get_z1_1q
from quara.protocol.qtomography.standard.standard_qtomography import StandardQTomography
from quara.protocol.qtomography.standard.standard_qst import StandardQst
from quara.qcircuit.experiment import Experiment


def get_test_data():
    e_sys = ElementalSystem(0, get_normalized_pauli_basis())
    c_sys = CompositeSystem([e_sys])

    povm_x = get_x_povm(c_sys)
    povm_y = get_y_povm(c_sys)
    povm_z = get_z_povm(c_sys)
    povms = [povm_x, povm_y, povm_z]

    schedules = []
    for index in range(len(povms)):
        schedule = [("state", 0), ("povm", index)]
        schedules.append(schedule)

    experiment = Experiment(states=[None], gates=[], povms=povms, schedules=schedules)
    set_qoperations = SetQOperations(states=[get_z0_1q(c_sys)], gates=[], povms=[])

    return experiment, set_qoperations


def get_test_data_qst(on_para_eq_constraint=True):
    e_sys = ElementalSystem(0, get_normalized_pauli_basis())
    c_sys = CompositeSystem([e_sys])

    povm_x = get_x_povm(c_sys)
    povm_y = get_y_povm(c_sys)
    povm_z = get_z_povm(c_sys)
    povms = [povm_x, povm_y, povm_z]

    qst = StandardQst(povms, on_para_eq_constraint=on_para_eq_constraint, seed=7)

    return qst, c_sys


def get_test_data_povmt(on_para_eq_constraint=True):
    e_sys = ElementalSystem(0, get_normalized_pauli_basis())
    c_sys = CompositeSystem([e_sys])

    # |+><+|
    state_x0 = get_x0_1q(c_sys)
    # |+i><+i|
    state_y0 = get_y0_1q(c_sys)
    # |0><0|
    state_z0 = get_z0_1q(c_sys)
    # |1><1|
    state_z1 = get_z1_1q(c_sys)
    tester_objects = [state_x0, state_y0, state_z0, state_z1]

    povmt = StandardQst(
        tester_objects, on_para_eq_constraint=on_para_eq_constraint, seed=777
    )

    return povmt, c_sys


class TestStandardQTomography:
    def test_init(self):
        experiment, set_qoperations = get_test_data()
        qt = StandardQTomography(experiment, set_qoperations)

        assert qt._coeffs_0th == None
        assert qt._coeffs_1st == None

    def test_get_coeffs_0th(self):
        experiment, set_qoperations = get_test_data()
        qt = StandardQTomography(experiment, set_qoperations)
        coeffs_0th = {(0, 0): 0, (0, 1): 1, (1, 0): 2, (1, 1): 3}
        qt._coeffs_0th = coeffs_0th

        assert qt.get_coeffs_0th(0, 0) == 0
        assert qt.get_coeffs_0th(0, 1) == 1
        assert qt.get_coeffs_0th(1, 0) == 2
        assert qt.get_coeffs_0th(1, 1) == 3

    def test_get_coeffs_1st(self):
        experiment, set_qoperations = get_test_data()
        qt = StandardQTomography(experiment, set_qoperations)
        coeffs_1st = {
            (0, 0): np.array([0, 0]),
            (0, 1): np.array([1, 1]),
            (1, 0): np.array([2, 2]),
            (1, 1): np.array([3, 3]),
        }
        qt._coeffs_1st = coeffs_1st

        assert np.all(qt.get_coeffs_1st(0, 0) == np.array([0, 0]))
        assert np.all(qt.get_coeffs_1st(0, 1) == np.array([1, 1]))
        assert np.all(qt.get_coeffs_1st(1, 0) == np.array([2, 2]))
        assert np.all(qt.get_coeffs_1st(1, 1) == np.array([3, 3]))

    def test_calc_matA(self):
        experiment, set_qoperations = get_test_data()
        qt = StandardQTomography(experiment, set_qoperations)
        coeffs_1st = {
            (0, 0): np.array([0, 0]),
            (0, 1): np.array([1, 1]),
            (1, 0): np.array([2, 2]),
            (1, 1): np.array([3, 3]),
        }
        qt._coeffs_1st = coeffs_1st

        actual = qt.calc_matA()
        expected = np.array([[0, 0], [1, 1], [2, 2], [3, 3]])
        assert np.all(actual == expected)

    def test_calc_vecB(self):
        experiment, set_qoperations = get_test_data()
        qt = StandardQTomography(experiment, set_qoperations)
        coeffs_0th = {(0, 0): 0, (0, 1): 1, (1, 0): 2, (1, 1): 3}
        qt._coeffs_0th = coeffs_0th

        actual = qt.calc_vecB()
        expected = np.array([0, 1, 2, 3])
        assert np.all(actual == expected)

    def test_is_fullrank_matA(self):
        experiment, set_qoperations = get_test_data()
        qt = StandardQTomography(experiment, set_qoperations)

        # case: True
        coeffs_1st = {
            (0, 0): np.array([0, 1]),
            (0, 1): np.array([2, 3]),
            (1, 0): np.array([4, 5]),
            (1, 1): np.array([6, 7]),
        }
        qt._coeffs_1st = coeffs_1st

        assert qt.is_fullrank_matA() == True

        # case: False
        coeffs_1st = {
            (0, 0): np.array([0, 0]),
            (0, 1): np.array([1, 1]),
            (1, 0): np.array([2, 2]),
            (1, 1): np.array([3, 3]),
        }
        qt._coeffs_1st = coeffs_1st

        assert qt.is_fullrank_matA() == False

    def test_calc_prob_dist(self):
        qst, c_sys = get_test_data_qst()
        state = get_z0_1q(c_sys)

        # schedule_index = 0
        actual = qst.calc_prob_dist(state, 0)
        npt.assert_almost_equal(
            actual, np.array([0.5, 0.5], dtype=np.float64), decimal=15
        )

        # schedule_index = 1
        actual = qst.calc_prob_dist(state, 1)
        npt.assert_almost_equal(
            actual, np.array([0.5, 0.5], dtype=np.float64), decimal=15
        )

        # schedule_index = 2
        actual = qst.calc_prob_dist(state, 2)
        npt.assert_almost_equal(actual, np.array([1, 0], dtype=np.float64), decimal=15)

    def test_calc_prob_dists(self):
        # tomography on_para_eq_constraint=True, qoperation on_para_eq_constraint=True
        qst, c_sys = get_test_data_qst(on_para_eq_constraint=True)
        state = get_z0_1q(c_sys)

        actual = qst.calc_prob_dists(state)
        expected = [
            np.array([0.5, 0.5], dtype=np.float64),
            np.array([0.5, 0.5], dtype=np.float64),
            np.array([1, 0], dtype=np.float64),
        ]
        npt.assert_almost_equal(actual, expected, decimal=15)

        # tomography on_para_eq_constraint=True, qoperation on_para_eq_constraint=False
        qst, c_sys = get_test_data_qst(on_para_eq_constraint=False)
        state = get_z0_1q(c_sys)

        actual = qst.calc_prob_dists(state)
        expected = [
            np.array([0.5, 0.5], dtype=np.float64),
            np.array([0.5, 0.5], dtype=np.float64),
            np.array([1, 0], dtype=np.float64),
        ]
        npt.assert_almost_equal(actual, expected, decimal=15)

        # tomography on_para_eq_constraint=False, qoperation on_para_eq_constraint=True
        qst, c_sys = get_test_data_qst(on_para_eq_constraint=False)
        state = get_z0_1q(c_sys)

        actual = qst.calc_prob_dists(state)
        expected = [
            np.array([0.5, 0.5], dtype=np.float64),
            np.array([0.5, 0.5], dtype=np.float64),
            np.array([1, 0], dtype=np.float64),
        ]
        npt.assert_almost_equal(actual, expected, decimal=15)

        # tomography on_para_eq_constraint=False, qoperation on_para_eq_constraint=False
        qst, c_sys = get_test_data_qst(on_para_eq_constraint=False)
        state = get_z0_1q(c_sys)
        state._on_para_eq_constraint = False

        actual = qst.calc_prob_dists(state)
        expected = [
            np.array([0.5, 0.5], dtype=np.float64),
            np.array([0.5, 0.5], dtype=np.float64),
            np.array([1, 0], dtype=np.float64),
        ]
        npt.assert_almost_equal(actual, expected, decimal=15)

    def test_calc_covariance_mat_single(self):
        qst, c_sys = get_test_data_qst()
        state = get_z0_1q(c_sys)

        # schedule_index = 0
        actual = qst.calc_covariance_mat_single(state, 0, 10)
        expected = np.array([[0.025, -0.025], [-0.025, 0.025]], dtype=np.float64)
        npt.assert_almost_equal(actual, expected, decimal=15)

        # schedule_index = 1
        actual = qst.calc_covariance_mat_single(state, 1, 5)
        expected = np.array([[0.05, -0.05], [-0.05, 0.05]], dtype=np.float64)
        npt.assert_almost_equal(actual, expected, decimal=15)

        # schedule_index = 2
        actual = qst.calc_covariance_mat_single(state, 2, 10)
        expected = np.array([[0.0, 0.0], [0.0, 0.0]], dtype=np.float64)
        npt.assert_almost_equal(actual, expected, decimal=15)

    def test_calc_covariance_mat_total(self):
        qst, c_sys = get_test_data_qst()
        state = get_z0_1q(c_sys)

        actual = qst.calc_covariance_mat_total(state, [10, 5, 10])
        expected = np.array(
            [
                [0.025, -0.025, 0, 0, 0, 0],
                [-0.025, 0.025, 0, 0, 0, 0],
                [0, 0, 0.05, -0.05, 0, 0],
                [0, 0, -0.05, 0.05, 0, 0],
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
            ],
            dtype=np.float64,
        )
        npt.assert_almost_equal(actual, expected, decimal=15)

    def test_calc_covariance_linear_mat_total(self):
        # on_para_eq_constraint=True
        qst, c_sys = get_test_data_qst(on_para_eq_constraint=True)
        state = get_z0_1q(c_sys)

        actual = qst.calc_covariance_linear_mat_total(state, [10, 5, 10])
        expected = np.array(
            [
                [0.05, 0, 0],
                [0, 0.1, 0],
                [0, 0, 0],
            ],
            dtype=np.float64,
        )
        npt.assert_almost_equal(actual, expected, decimal=15)

        # on_para_eq_constraint=False
        qst, c_sys = get_test_data_qst(on_para_eq_constraint=False)
        state = get_z0_1q(c_sys)
        state._on_para_eq_constraint = False

        actual = qst.calc_covariance_linear_mat_total(state, [10, 5, 10])
        expected = np.array(
            [
                [0, 0, 0, 0],
                [0, 0.05, 0, 0],
                [0, 0, 0.1, 0],
                [0, 0, 0, 0],
            ],
            dtype=np.float64,
        )
        npt.assert_almost_equal(actual, expected, decimal=15)

    def test_calc_mse_linear_analytical(self):
        # Case 1: qst, on_par_eq_constraint = True
        # Arrange
        qst, c_sys = get_test_data_qst()
        state = get_z0_1q(c_sys)
        # Act
        actual = qst.calc_mse_linear_analytical(state, [10, 5, 10])
        # Assert
        npt.assert_almost_equal(actual, 0.15, decimal=15)

        # Case 2: qst, on_par_eq_constraint = False
        # Arange
        qst, c_sys = get_test_data_qst(on_para_eq_constraint=False)
        state = get_z0_1q(c_sys)
        state = State(
            vec=get_z0_1q(c_sys).vec, c_sys=c_sys, on_para_eq_constraint=False
        )
        # Act
        actual = qst.calc_mse_linear_analytical(state, [10, 5, 10])
        # Assert
        npt.assert_almost_equal(actual, 0.15, decimal=15)

    def test_calc_mse_empi_dists_analytical(self):
        # Case 1: qst, on_par_eq_constraint = True
        # Arrange
        qst, c_sys = get_test_data_qst()
        state = get_z0_1q(c_sys)
        # Act
        actual = qst.calc_mse_empi_dists_analytical(state, [10, 5, 10])
        # Assert
        npt.assert_almost_equal(actual, 0.15, decimal=15)

        # Case 2: qst, on_par_eq_constraint = False
        # Arange
        qst, c_sys = get_test_data_qst(on_para_eq_constraint=False)
        state = get_z0_1q(c_sys)
        state = State(
            vec=get_z0_1q(c_sys).vec, c_sys=c_sys, on_para_eq_constraint=False
        )
        # Act
        actual = qst.calc_mse_empi_dists_analytical(state, [10, 5, 10])
        # Assert
        npt.assert_almost_equal(actual, 0.15, decimal=15)

    def test_calc_fisher_matrix_qoperation(self):
        # Case 1: qst, on_par_eq_constraint = True
        # Arrange
        qst, c_sys = get_test_data_qst()
        state = get_z0_1q(c_sys)
        # Act
        actual = qst.calc_fisher_matrix(2, state)
        # Assert
        expected = np.array(
            [
                [0, 0, 0],
                [0, 0, 0],
                [0, 0, 50000000.499999985],
            ],
            dtype=np.float64,
        )
        npt.assert_almost_equal(actual, expected, decimal=15)

        # Case 2: qst, on_par_eq_constraint = False
        # Arrange
        qst, c_sys = get_test_data_qst(on_para_eq_constraint=False)
        vec = np.array([1, 0, 0, 1], dtype=np.float64) / np.sqrt(2)
        state = State(c_sys, vec, on_para_eq_constraint=False)
        # Act
        actual = qst.calc_fisher_matrix(2, state)
        # Assert
        expected = np.array(
            [
                [50000000.499999985, 0, 0, -49999999.49999997],
                [0, 0, 0, 0],
                [0, 0, 0, 0],
                [-49999999.49999997, 0, 0, 50000000.499999985],
            ],
            dtype=np.float64,
        )
        npt.assert_almost_equal(actual, expected, decimal=15)

    def test_calc_fisher_matrix_var(self):
        # Case 1: qst, on_par_eq_constraint = True
        # Arrange
        qst, c_sys = get_test_data_qst()
        var = np.array([0, 0, 1], dtype=np.float64) / np.sqrt(2)
        # Act
        actual = qst.calc_fisher_matrix(2, var)
        # Assert
        expected = np.array(
            [
                [0, 0, 0],
                [0, 0, 0],
                [0, 0, 50000000.499999985],
            ],
            dtype=np.float64,
        )
        npt.assert_almost_equal(actual, expected, decimal=15)

        # Case 2: qst, on_par_eq_constraint = False
        # Arrange
        qst, c_sys = get_test_data_qst(on_para_eq_constraint=False)
        vec = np.array([1, 0, 0, 1], dtype=np.float64) / np.sqrt(2)
        var = np.array([1, 0, 0, 1], dtype=np.float64) / np.sqrt(2)
        # Act
        actual = qst.calc_fisher_matrix(2, var)
        # Assert
        expected = np.array(
            [
                [50000000.499999985, 0, 0, -49999999.49999997],
                [0, 0, 0, 0],
                [0, 0, 0, 0],
                [-49999999.49999997, 0, 0, 50000000.499999985],
            ],
            dtype=np.float64,
        )
        npt.assert_almost_equal(actual, expected, decimal=15)

    def test_calc_fisher_matrix_total_qoperation(self):
        # Case 1: qst, on_par_eq_constraint = True
        # Arrange
        qst, c_sys = get_test_data_qst()
        state = get_z0_1q(c_sys)
        weights = [3, 2, 1]
        # Act
        actual = qst.calc_fisher_matrix_total(state, weights)
        # Assert
        expected = np.array(
            [
                [6, 0, 0],
                [0, 4, 0],
                [0, 0, 50000000.499999985],
            ],
            dtype=np.float64,
        )
        npt.assert_almost_equal(actual, expected, decimal=15)

        # Case 2: qst, on_par_eq_constraint = False
        # Arrange
        qst, c_sys = get_test_data_qst(on_para_eq_constraint=False)
        vec = np.array([1, 0, 0, 1], dtype=np.float64) / np.sqrt(2)
        state = State(c_sys, vec, on_para_eq_constraint=False)
        weights = [3, 2, 1]
        # Act
        actual = qst.calc_fisher_matrix_total(state, weights)
        # Assert
        expected = np.array(
            [
                [50000010.499999985, 0, 0, -49999999.49999997],
                [0, 6, 0, 0],
                [0, 0, 4, 0],
                [-49999999.49999997, 0, 0, 50000000.499999985],
            ],
            dtype=np.float64,
        )
        npt.assert_almost_equal(actual, expected, decimal=15)

    def test_calc_fisher_matrix_total_var(self):
        # Case 1: qst, on_par_eq_constraint = True
        # Arrange
        qst, c_sys = get_test_data_qst()
        var = np.array([0, 0, 1], dtype=np.float64) / np.sqrt(2)
        weights = [3, 2, 1]
        # Act
        actual = qst.calc_fisher_matrix_total(var, weights)
        # Assert
        expected = np.array(
            [
                [6, 0, 0],
                [0, 4, 0],
                [0, 0, 50000000.499999985],
            ],
            dtype=np.float64,
        )
        npt.assert_almost_equal(actual, expected, decimal=15)

        # Case 2: qst, on_par_eq_constraint = False
        # Arrange
        qst, c_sys = get_test_data_qst(on_para_eq_constraint=False)
        vec = np.array([1, 0, 0, 1], dtype=np.float64) / np.sqrt(2)
        var = np.array([1, 0, 0, 1], dtype=np.float64) / np.sqrt(2)
        weights = [3, 2, 1]
        # Act
        actual = qst.calc_fisher_matrix_total(var, weights)
        # Assert
        expected = np.array(
            [
                [50000010.499999985, 0, 0, -49999999.49999997],
                [0, 6, 0, 0],
                [0, 0, 4, 0],
                [-49999999.49999997, 0, 0, 50000000.499999985],
            ],
            dtype=np.float64,
        )
        npt.assert_almost_equal(actual, expected, decimal=15)

    def test_calc_cramer_rao_bound_qoperation(self):
        # Case 1: qst, on_par_eq_constraint = True
        # Arrange
        qst, c_sys = get_test_data_qst()
        state = get_z0_1q(c_sys)
        list_N = [4, 2, 1]
        N = 10
        # Act
        actual = qst.calc_cramer_rao_bound(state, N, list_N)
        # Assert
        expected = 0.37500001999999977
        npt.assert_almost_equal(actual, expected, decimal=15)

        # Case 2: qst, on_par_eq_constraint = False
        # Arrange
        qst, c_sys = get_test_data_qst(on_para_eq_constraint=False)
        vec = np.array([1, 0, 0, 1], dtype=np.float64) / np.sqrt(2)
        state = State(c_sys, vec, on_para_eq_constraint=False)
        list_N = [4, 2, 1]
        N = 10
        # Act
        actual = qst.calc_cramer_rao_bound(state, N, list_N)
        # Assert
        expected = 0.5178571598881306
        npt.assert_almost_equal(actual, expected, decimal=15)

    def test_calc_cramer_rao_bound_var(self):
        # Case 1: qst, on_par_eq_constraint = True
        # Arrange
        qst, c_sys = get_test_data_qst()
        var = np.array([0, 0, 1], dtype=np.float64) / np.sqrt(2)
        list_N = [4, 2, 1]
        N = 10
        # Act
        actual = qst.calc_cramer_rao_bound(var, N, list_N)
        # Assert
        expected = 0.37500001999999977
        npt.assert_almost_equal(actual, expected, decimal=15)

        # Case 2: qst, on_par_eq_constraint = False
        # Arrange
        qst, c_sys = get_test_data_qst(on_para_eq_constraint=False)
        var = np.array([1, 0, 0, 1], dtype=np.float64) / np.sqrt(2)
        list_N = [4, 2, 1]
        N = 10
        # Act
        actual = qst.calc_cramer_rao_bound(var, N, list_N)
        # Assert
        expected = 0.5178571598881306
        npt.assert_almost_equal(actual, expected, decimal=15)
