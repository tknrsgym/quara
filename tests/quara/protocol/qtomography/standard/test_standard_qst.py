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
from quara.objects.state import get_z0_1q
from quara.protocol.qtomography.standard.standard_qst import StandardQst


def get_test_data():
    e_sys = ElementalSystem(0, get_normalized_pauli_basis())
    c_sys = CompositeSystem([e_sys])

    povm_x = get_x_measurement(c_sys)
    povm_y = get_y_measurement(c_sys)
    povm_z = get_z_measurement(c_sys)
    povms = [povm_x, povm_y, povm_z]

    qst = StandardQst(povms, on_para_eq_constraint=False, seed=7)

    return qst, c_sys


class TestStandardQst:
    def test_init_on_para_eq_constraint_True(self):
        e_sys = ElementalSystem(0, get_normalized_pauli_basis())
        c_sys = CompositeSystem([e_sys])

        povm_x = get_x_measurement(c_sys)
        povm_y = get_y_measurement(c_sys)
        povm_z = get_z_measurement(c_sys)
        povms = [povm_x, povm_y, povm_z]

        qst = StandardQst(povms, on_para_eq_constraint=True)

        # num_variables
        assert qst.num_variables == 3

        # _map_experiment_to_setqoperations
        assert qst._map_experiment_to_setqoperations == {("state", 0): ("state", 0)}
        # _map_setqoperations_to_experiment
        assert qst._map_setqoperations_to_experiment == {("state", 0): ("state", 0)}

        arrayA = [
            [1, 0, 0],
            [-1, 0, 0],
            [0, 1, 0],
            [0, -1, 0],
            [0, 0, 1],
            [0, 0, -1],
        ]
        expectedA = np.array(arrayA, dtype=np.float64) / np.sqrt(2)
        # get_coeffs_1st
        npt.assert_almost_equal(qst.get_coeffs_1st(0, 0), expectedA[0], decimal=15)
        npt.assert_almost_equal(qst.get_coeffs_1st(0, 1), expectedA[1], decimal=15)
        npt.assert_almost_equal(qst.get_coeffs_1st(1, 0), expectedA[2], decimal=15)
        npt.assert_almost_equal(qst.get_coeffs_1st(1, 1), expectedA[3], decimal=15)
        npt.assert_almost_equal(qst.get_coeffs_1st(2, 0), expectedA[4], decimal=15)
        npt.assert_almost_equal(qst.get_coeffs_1st(2, 1), expectedA[5], decimal=15)
        # calc_matA
        npt.assert_almost_equal(qst.calc_matA(), expectedA, decimal=15)

        arrayB = [
            1 / np.sqrt(2),
            1 / np.sqrt(2),
            1 / np.sqrt(2),
            1 / np.sqrt(2),
            1 / np.sqrt(2),
            1 / np.sqrt(2),
        ]
        expectedB = np.array(arrayB, dtype=np.float64)
        # get_coeffs_0th
        npt.assert_almost_equal(qst.get_coeffs_0th(0, 0), expectedB[0], decimal=15)
        npt.assert_almost_equal(qst.get_coeffs_0th(0, 1), expectedB[1], decimal=15)
        npt.assert_almost_equal(qst.get_coeffs_0th(1, 0), expectedB[2], decimal=15)
        npt.assert_almost_equal(qst.get_coeffs_0th(1, 1), expectedB[3], decimal=15)
        npt.assert_almost_equal(qst.get_coeffs_0th(2, 0), expectedB[4], decimal=15)
        npt.assert_almost_equal(qst.get_coeffs_0th(2, 1), expectedB[5], decimal=15)
        # calc_vecB
        npt.assert_almost_equal(qst.calc_vecB(), expectedB, decimal=15)

        # is_fullrank_matA
        assert qst.is_fullrank_matA() == True

    def test_init_on_para_eq_constraint_False(self):
        qst, _ = get_test_data()

        # num_variables
        assert qst.num_variables == 4

        # _map_experiment_to_setqoperations
        assert qst._map_experiment_to_setqoperations == {("state", 0): ("state", 0)}
        # _map_setqoperations_to_experiment
        assert qst._map_setqoperations_to_experiment == {("state", 0): ("state", 0)}

        arrayA = [
            [1, 1, 0, 0],
            [1, -1, 0, 0],
            [1, 0, 1, 0],
            [1, 0, -1, 0],
            [1, 0, 0, 1],
            [1, 0, 0, -1],
        ]
        expectedA = np.array(arrayA, dtype=np.float64) / np.sqrt(2)
        # get_coeffs_1st
        npt.assert_almost_equal(qst.get_coeffs_1st(0, 0), expectedA[0], decimal=15)
        npt.assert_almost_equal(qst.get_coeffs_1st(0, 1), expectedA[1], decimal=15)
        npt.assert_almost_equal(qst.get_coeffs_1st(1, 0), expectedA[2], decimal=15)
        npt.assert_almost_equal(qst.get_coeffs_1st(1, 1), expectedA[3], decimal=15)
        npt.assert_almost_equal(qst.get_coeffs_1st(2, 0), expectedA[4], decimal=15)
        npt.assert_almost_equal(qst.get_coeffs_1st(2, 1), expectedA[5], decimal=15)
        # calc_matA
        npt.assert_almost_equal(qst.calc_matA(), expectedA, decimal=15)

        arrayB = [0, 0, 0, 0, 0, 0]
        expectedB = np.array(arrayB, dtype=np.float64)
        # get_coeffs_0th
        npt.assert_almost_equal(qst.get_coeffs_0th(0, 0), expectedB[0], decimal=15)
        npt.assert_almost_equal(qst.get_coeffs_0th(0, 1), expectedB[1], decimal=15)
        npt.assert_almost_equal(qst.get_coeffs_0th(1, 0), expectedB[2], decimal=15)
        npt.assert_almost_equal(qst.get_coeffs_0th(1, 1), expectedB[3], decimal=15)
        npt.assert_almost_equal(qst.get_coeffs_0th(2, 0), expectedB[4], decimal=15)
        npt.assert_almost_equal(qst.get_coeffs_0th(2, 1), expectedB[5], decimal=15)
        # calc_vecB
        npt.assert_almost_equal(qst.calc_vecB(), expectedB, decimal=15)

        # is_fullrank_matA
        assert qst.is_fullrank_matA() == True

    def test_init_exception(self):
        e_sys0 = ElementalSystem(0, get_normalized_pauli_basis())
        c_sys0 = CompositeSystem([e_sys0])
        e_sys1 = ElementalSystem(1, get_normalized_pauli_basis())
        c_sys1 = CompositeSystem([e_sys1])

        povm_x = get_x_measurement(c_sys0)
        povm_y = get_y_measurement(c_sys0)
        povm_z = get_z_measurement(c_sys1)
        povms = [povm_x, povm_y, povm_z]

        # is_valid_experiment == False
        with pytest.raises(ValueError):
            StandardQst(povms, on_para_eq_constraint=False)

    def test_is_valid_experiment(self):
        qst, _ = get_test_data()
        assert qst.is_valid_experiment() == True

    def test_generate_empi_dist(self):
        qst, c_sys = get_test_data()
        state = get_z0_1q(c_sys)

        # schedule_index = 0
        actual = qst.generate_empi_dist(0, state, 10)
        expected = (10, np.array([0.5, 0.5], dtype=np.float64))
        assert actual[0] == expected[0]
        npt.assert_almost_equal(actual[1], expected[1], decimal=15)

        # schedule_index = 1
        actual = qst.generate_empi_dist(1, state, 10)
        expected = (10, np.array([0.6, 0.4], dtype=np.float64))
        assert actual[0] == expected[0]
        npt.assert_almost_equal(actual[1], expected[1], decimal=15)

        # schedule_index = 2
        actual = qst.generate_empi_dist(2, state, 10)
        expected = (10, np.array([1, 0], dtype=np.float64))
        assert actual[0] == expected[0]
        npt.assert_almost_equal(actual[1], expected[1], decimal=15)

    def test_generate_empi_dists(self):
        qst, c_sys = get_test_data()
        state = get_z0_1q(c_sys)

        actual = qst.generate_empi_dists(state, 10)
        expected = [
            (10, np.array([0.5, 0.5], dtype=np.float64)),
            (10, np.array([0.6, 0.4], dtype=np.float64)),
            (10, np.array([1, 0], dtype=np.float64)),
        ]
        for a, e in zip(actual, expected):
            assert a[0] == e[0]
            npt.assert_almost_equal(a[1], e[1], decimal=15)

    def test_generate_empi_dists_sequence(self):
        qst, c_sys = get_test_data()
        state = get_z0_1q(c_sys)

        actual = qst.generate_empi_dists_sequence(state, [10, 20])
        print(actual)
        expected = [
            [
                (10, np.array([0.5, 0.5], dtype=np.float64)),
                (10, np.array([0.3, 0.7], dtype=np.float64)),
                (10, np.array([1, 0], dtype=np.float64)),
            ],
            [
                (20, np.array([0.55, 0.45], dtype=np.float64)),
                (20, np.array([0.5, 0.5], dtype=np.float64)),
                (20, np.array([1, 0], dtype=np.float64)),
            ],
        ]
        for a_dists, e_dists in zip(actual, expected):
            for a, e in zip(a_dists, e_dists):
                assert a[0] == e[0]
                npt.assert_almost_equal(a[1], e[1], decimal=15)
