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
from quara.objects.state import State, get_z0_1q
from quara.protocol.qtomography.standard.standard_qst import StandardQst


def get_test_data():
    e_sys = ElementalSystem(0, get_normalized_pauli_basis())
    c_sys = CompositeSystem([e_sys])

    povm_x = get_x_povm(c_sys)
    povm_y = get_y_povm(c_sys)
    povm_z = get_z_povm(c_sys)
    povms = [povm_x, povm_y, povm_z]

    qst = StandardQst(povms, on_para_eq_constraint=False, seed=7)

    return qst, c_sys


class TestStandardQst:
    def test_init_on_para_eq_constraint_True(self):
        e_sys = ElementalSystem(0, get_normalized_pauli_basis())
        c_sys = CompositeSystem([e_sys])

        povm_x = get_x_povm(c_sys)
        povm_y = get_y_povm(c_sys)
        povm_z = get_z_povm(c_sys)
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
            0.5,
            0.5,
            0.5,
            0.5,
            0.5,
            0.5,
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

        povm_x = get_x_povm(c_sys0)
        povm_y = get_y_povm(c_sys0)
        povm_z = get_z_povm(c_sys1)
        povms = [povm_x, povm_y, povm_z]

        # is_valid_experiment == False
        with pytest.raises(ValueError):
            StandardQst(povms, on_para_eq_constraint=False)

    def test_estimation_object_type(self):
        qst, _ = get_test_data()
        assert qst.estimation_object_type() == State

    def test_is_valid_experiment(self):
        qst, _ = get_test_data()
        assert qst.is_valid_experiment() == True

    def test_reset_seed(self):
        # Assert
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

        # Act
        qst.reset_seed()

        # Assert
        actual = qst.generate_empi_dists(state, 10)
        expected = [
            (10, np.array([0.5, 0.5], dtype=np.float64)),
            (10, np.array([0.6, 0.4], dtype=np.float64)),
            (10, np.array([1, 0], dtype=np.float64)),
        ]
        for a, e in zip(actual, expected):
            assert a[0] == e[0]
            npt.assert_almost_equal(a[1], e[1], decimal=15)

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

    def test_generate_empi_dist__seed_or_stream(self):
        qst, c_sys = get_test_data()
        state = get_z0_1q(c_sys)

        # seed_or_stream : default
        np.random.seed(7)
        actual1 = qst.generate_empi_dist(0, state, 10)
        # seed_or_stream : int
        actual2 = qst.generate_empi_dist(0, state, 10, seed_or_stream=7)
        # seed_or_stream : np.random.RandomState
        actual3 = qst.generate_empi_dist(
            0, state, 10, seed_or_stream=np.random.RandomState(7)
        )
        npt.assert_almost_equal(actual1[1], actual2[1], decimal=15)
        npt.assert_almost_equal(actual2[1], actual3[1], decimal=15)

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

    def test_generate_empty_estimation_obj_with_setting_info(self):
        qst, c_sys = get_test_data()
        setting_info = qst.generate_empty_estimation_obj_with_setting_info()

        expected = np.array([0, 0, 0, 0])
        npt.assert_almost_equal(setting_info.to_stacked_vector(), expected, decimal=15)
        assert setting_info.on_para_eq_constraint == False

    def test_validate_schedules(self):
        e_sys = ElementalSystem(0, get_normalized_pauli_basis())
        c_sys = CompositeSystem([e_sys])

        povm_x = get_x_povm(c_sys)
        povm_y = get_y_povm(c_sys)
        povm_z = get_z_povm(c_sys)
        povms = [povm_x, povm_y, povm_z]

        # Act
        qst = StandardQst(povms, on_para_eq_constraint=True)
        # Assert
        actual = qst._experiment.schedules
        assert len(actual) == 3
        for i, a in enumerate(actual):
            expected = [("state", 0), ("povm", i)]
            assert a == expected

        # Case 2:
        # Act
        qst = StandardQst(povms, on_para_eq_constraint=True, schedules="all")
        # Assert
        actual = qst._experiment.schedules
        assert len(actual) == 3
        for i, a in enumerate(actual):
            expected = [("state", 0), ("povm", i)]
            assert a == expected

        # Case 3:
        # Act
        schedules = [[("state", 0), ("povm", 2)], [("state", 0), ("povm", 1)]]
        qst = StandardQst(povms, on_para_eq_constraint=True, schedules=schedules)
        # Assert
        actual = qst._experiment.schedules
        assert actual == schedules

        # Case 4:
        # Act
        invalid_schedules = "invalid str"
        with pytest.raises(ValueError):
            _ = StandardQst(
                povms, on_para_eq_constraint=True, schedules=invalid_schedules
            )

        # Case 5:
        # invalid_schedules = [[("state", 0), ("gate", 0), ("povm", 2)], [("state", 0), ("gate", 0), ("povm", 1)]]
        # with pytest.raises(ValueError):
        #     _ = StandardQst(
        #         povms, on_para_eq_constraint=True, schedules=invalid_schedules
        #     )

        # Case 6:
        # invalid_schedules = [[("state", 0), ("gate": 0), ("povm", 2)], [("state", 1), ("povm", 1)]]
        # with pytest.raises(ValueError):
        #     _ = StandardQst(povms, on_para_eq_constraint=True, schedules=invalid_schedules)
