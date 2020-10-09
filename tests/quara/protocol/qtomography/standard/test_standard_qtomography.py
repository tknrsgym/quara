import numpy as np
import numpy.testing as npt
import pytest

from quara.objects.composite_system import CompositeSystem
from quara.objects.elemental_system import ElementalSystem
from quara.objects.matrix_basis import get_normalized_pauli_basis
from quara.objects.povm import (
    Povm,
    get_x_measurement,
    get_y_measurement,
    get_z_measurement,
)
from quara.objects.qoperations import SetQOperations
from quara.objects.state import get_z0_1q
from quara.protocol.qtomography.standard.standard_qtomography import StandardQTomography
from quara.protocol.qtomography.standard.standard_qst import StandardQst
from quara.qcircuit.experiment import Experiment


def get_test_data():
    e_sys = ElementalSystem(0, get_normalized_pauli_basis())
    c_sys = CompositeSystem([e_sys])

    povm_x = get_x_measurement(c_sys)
    povm_y = get_y_measurement(c_sys)
    povm_z = get_z_measurement(c_sys)
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

    povm_x = get_x_measurement(c_sys)
    povm_y = get_y_measurement(c_sys)
    povm_z = get_z_measurement(c_sys)
    povms = [povm_x, povm_y, povm_z]

    qst = StandardQst(povms, on_para_eq_constraint=on_para_eq_constraint, seed=7)

    return qst, c_sys


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
        # on_para_eq_constraint=True
        qst, c_sys = get_test_data_qst(on_para_eq_constraint=True)
        state = get_z0_1q(c_sys)

        actual = qst.calc_prob_dists(state)
        expected = [
            np.array([0.5, 0.5], dtype=np.float64),
            np.array([0.5, 0.5], dtype=np.float64),
            np.array([1, 0], dtype=np.float64),
        ]
        npt.assert_almost_equal(actual, expected, decimal=15)

        # on_para_eq_constraint=False
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
        expected = np.array([[0.05, 0, 0], [0, 0.1, 0], [0, 0, 0],], dtype=np.float64,)
        npt.assert_almost_equal(actual, expected, decimal=15)

        # on_para_eq_constraint=False
        qst, c_sys = get_test_data_qst(on_para_eq_constraint=False)
        state = get_z0_1q(c_sys)
        state._on_para_eq_constraint = False

        actual = qst.calc_covariance_linear_mat_total(state, [10, 5, 10])
        expected = np.array(
            [[0, 0, 0, 0], [0, 0.05, 0, 0], [0, 0, 0.1, 0], [0, 0, 0, 0],],
            dtype=np.float64,
        )
        npt.assert_almost_equal(actual, expected, decimal=15)

    def test_calc_mse_linear(self):
        qst, c_sys = get_test_data_qst()
        state = get_z0_1q(c_sys)

        actual = qst.calc_mse_linear(state, [10, 5, 10])
        npt.assert_almost_equal(actual, 0.15, decimal=15)
