import numpy as np
import numpy.testing as npt
import pytest

from quara.objects.composite_system import CompositeSystem
from quara.objects.elemental_system import ElementalSystem
from quara.objects.matrix_basis import get_normalized_pauli_basis
from quara.objects.state import (
    get_x0_1q,
    get_y0_1q,
    get_z0_1q,
    get_z1_1q,
)
from quara.objects.povm import Povm
from quara.objects.povm_typical import generate_povm_from_name
from quara.protocol.qtomography.standard.standard_povmt import StandardPovmt


def get_test_data():
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

    # Act
    povmt = StandardPovmt(
        tester_objects, on_para_eq_constraint=False, num_outcomes=2, seed=7
    )

    return povmt, c_sys


class TestStandardPovmt:
    def test_init_on_para_eq_constraint_false(self):
        # Array
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
        states = [state_x0, state_y0, state_z0, state_z1]

        # Case 1: m = 2
        # Act
        actual = StandardPovmt(states, num_outcomes=2, on_para_eq_constraint=False)
        assert actual.num_variables == 8

        # Assert
        expected_A = (1 / np.sqrt(2)) * np.array(
            [
                [1, 1, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 1, 1, 0, 0],
                [1, 0, 1, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 1, 0, 1, 0],
                [1, 0, 0, 1, 0, 0, 0, 0],
                [0, 0, 0, 0, 1, 0, 0, 1],
                [1, 0, 0, -1, 0, 0, 0, 0],
                [0, 0, 0, 0, 1, 0, 0, -1],
            ]
        )
        npt.assert_almost_equal(actual.calc_matA(), expected_A, decimal=15)

        expected_b = np.array(
            [
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
            ]
        )
        npt.assert_almost_equal(actual.calc_vecB(), expected_b, decimal=15)

        # Case 1: m = 3
        # Act
        actual = StandardPovmt(states, num_outcomes=3, on_para_eq_constraint=False)
        assert actual.num_variables == 12

        # Assert
        expected_A = (1 / np.sqrt(2)) * np.array(
            [
                [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0],
                [1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0],
                [1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1],
                [1, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 1, 0, 0, -1, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, -1],
            ]
        )
        npt.assert_almost_equal(actual.calc_matA(), expected_A, decimal=15)

        expected_b = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        npt.assert_almost_equal(actual.calc_vecB(), expected_b, decimal=15)

    def test_init_on_para_eq_constraint_true(self):
        # Array
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
        states = [state_x0, state_y0, state_z0, state_z1]

        # Case 1: m = 2
        # Act
        actual = StandardPovmt(states, num_outcomes=2, on_para_eq_constraint=True)

        # Assert
        assert actual.num_variables == 4
        # Assert
        expected_A = (1 / np.sqrt(2)) * np.array(
            [
                [1, 1, 0, 0],
                [-1, -1, 0, 0],
                [1, 0, 1, 0],
                [-1, 0, -1, 0],
                [1, 0, 0, 1],
                [-1, 0, 0, -1],
                [1, 0, 0, -1],
                [-1, 0, 0, 1],
            ]
        )
        npt.assert_almost_equal(actual.calc_matA(), expected_A, decimal=15)

        expected_b = np.array([0, 1, 0, 1, 0, 1, 0, 1])
        npt.assert_almost_equal(actual.calc_vecB(), expected_b, decimal=15)

        # Case 2: m = 3
        # Act
        actual = StandardPovmt(states, num_outcomes=3, on_para_eq_constraint=True)

        # Assert
        assert actual.num_variables == 8
        # Assert
        expected_A = (1 / np.sqrt(2)) * np.array(
            [
                [1, 1, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 1, 1, 0, 0],
                [-1, -1, 0, 0, -1, -1, 0, 0],
                [1, 0, 1, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 1, 0, 1, 0],
                [-1, 0, -1, 0, -1, 0, -1, 0],
                [1, 0, 0, 1, 0, 0, 0, 0],
                [0, 0, 0, 0, 1, 0, 0, 1],
                [-1, 0, 0, -1, -1, 0, 0, -1],
                [1, 0, 0, -1, 0, 0, 0, 0],
                [0, 0, 0, 0, 1, 0, 0, -1],
                [-1, 0, 0, 1, -1, 0, 0, 1],
            ]
        )
        npt.assert_almost_equal(actual.calc_matA(), expected_A, decimal=15)

        expected_b = np.array([0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1])
        npt.assert_almost_equal(actual.calc_vecB(), expected_b, decimal=15)

    def test_init_exception(self):
        # Array
        e_sys0 = ElementalSystem(0, get_normalized_pauli_basis())
        c_sys0 = CompositeSystem([e_sys0])
        e_sys1 = ElementalSystem(1, get_normalized_pauli_basis())
        c_sys1 = CompositeSystem([e_sys1])

        state_x = get_x0_1q(c_sys0)
        state_y = get_y0_1q(c_sys0)
        state_z = get_z0_1q(c_sys1)  # invalid
        states = [state_x, state_y, state_z]

        # Act & Assert
        with pytest.raises(ValueError):
            _ = StandardPovmt(states, num_outcomes=2, on_para_eq_constraint=False)

    def test_estimation_object_type(self):
        # Array
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
        states = [state_x0, state_y0, state_z0, state_z1]

        # Case 1: m = 2
        # Act
        actual = StandardPovmt(states, num_outcomes=2, on_para_eq_constraint=True)

        # Act & Assert
        assert actual.estimation_object_type() == Povm

    def test_generate_empty_estimation_obj_with_setting_info(self):
        # Array
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
        states = [state_x0, state_y0, state_z0, state_z1]

        # Act
        actual = StandardPovmt(states, num_outcomes=2, on_para_eq_constraint=False)
        setting_info = actual.generate_empty_estimation_obj_with_setting_info()

        expected = np.array([0, 0, 0, 0, 0, 0, 0, 0])
        npt.assert_almost_equal(setting_info.to_stacked_vector(), expected, decimal=15)
        assert setting_info.on_para_eq_constraint == False

    def test_calc_mse_linear_analytical_mode_qoperation(self):
        # setup system
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

        # true_object: origin
        a0, a1, a2, a3 = 1, 0, 0, 0
        m1 = (1 / np.sqrt(2)) * np.array([a0, a1, a2, a3])
        m2 = (1 / np.sqrt(2)) * np.array([2 - a0, -a1, -a2, -a3])

        ### Case 1: on_par_eq_constraint = True
        # Arange
        true_object = Povm(vecs=[m1, m2], c_sys=c_sys, on_para_eq_constraint=True)
        num_outcomes = len(true_object.vecs)
        povmt = StandardPovmt(
            tester_objects, num_outcomes=num_outcomes, on_para_eq_constraint=True
        )

        # Act
        actual = povmt.calc_mse_linear_analytical(
            true_object, [100] * len(tester_objects), mode="qoperation"
        )

        # Assert
        npt.assert_almost_equal(actual, 0.04, decimal=15)

        ### Case 2: on_par_eq_constraint = False
        # Arange
        true_object = Povm(vecs=[m1, m2], c_sys=c_sys, on_para_eq_constraint=False)
        num_outcomes = len(true_object.vecs)
        povmt = StandardPovmt(
            tester_objects, num_outcomes=num_outcomes, on_para_eq_constraint=False
        )

        # Act
        actual = povmt.calc_mse_linear_analytical(
            true_object, [100] * len(tester_objects), mode="qoperation"
        )

        # Assert
        npt.assert_almost_equal(actual, 0.04, decimal=15)

    def test_calc_mse_linear_analytical_mode_var(self):
        # setup system
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

        # true_object: origin
        a0, a1, a2, a3 = 1, 0, 0, 0
        m1 = (1 / np.sqrt(2)) * np.array([a0, a1, a2, a3])
        m2 = (1 / np.sqrt(2)) * np.array([2 - a0, -a1, -a2, -a3])

        ### Case 1: on_par_eq_constraint = True
        # Arange
        true_object = Povm(vecs=[m1, m2], c_sys=c_sys, on_para_eq_constraint=True)
        num_outcomes = len(true_object.vecs)
        povmt = StandardPovmt(
            tester_objects, num_outcomes=num_outcomes, on_para_eq_constraint=True
        )

        # Act
        actual = povmt.calc_mse_linear_analytical(
            true_object, [100] * len(tester_objects), mode="var"
        )

        # Assert
        npt.assert_almost_equal(actual, 0.02, decimal=15)

        ### Case 2: on_par_eq_constraint = False
        # Arange
        true_object = Povm(vecs=[m1, m2], c_sys=c_sys, on_para_eq_constraint=False)
        num_outcomes = len(true_object.vecs)
        povmt = StandardPovmt(
            tester_objects, num_outcomes=num_outcomes, on_para_eq_constraint=False
        )

        # Act
        actual = povmt.calc_mse_linear_analytical(
            true_object, [100] * len(tester_objects), mode="var"
        )

        # Assert
        npt.assert_almost_equal(actual, 0.04, decimal=15)

    def test_calc_cramer_rao_bound(self):
        # setup system
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

        # true_object: z0
        a0, a1, a2, a3 = 1, 0, 0, 1
        m1 = (1 / np.sqrt(2)) * np.array([a0, a1, a2, a3])
        m2 = (1 / np.sqrt(2)) * np.array([2 - a0, -a1, -a2, -a3])

        ### Case 1: on_par_eq_constraint = True
        # Arange
        true_object = Povm(vecs=[m1, m2], c_sys=c_sys, on_para_eq_constraint=True)
        num_outcomes = len(true_object.vecs)
        povmt = StandardPovmt(
            tester_objects, num_outcomes=num_outcomes, on_para_eq_constraint=True
        )

        # Act
        actual = povmt.calc_cramer_rao_bound(
            true_object, 100, [100] * len(tester_objects)
        )

        # Assert
        npt.assert_almost_equal(actual, 0.0200000008, decimal=15)

        ### Case 2: on_par_eq_constraint = False
        # Arange
        true_object = Povm(vecs=[m1, m2], c_sys=c_sys, on_para_eq_constraint=False)
        num_outcomes = len(true_object.vecs)
        povmt = StandardPovmt(
            tester_objects, num_outcomes=num_outcomes, on_para_eq_constraint=False
        )

        # Act
        print(povmt.calc_fisher_matrix_total(true_object, [1, 1, 1, 1]))
        actual = povmt.calc_cramer_rao_bound(
            true_object, 100, [100] * len(tester_objects)
        )

        # Assert
        npt.assert_almost_equal(actual, 0.07999999984703485, decimal=11)

    def test_validate_schedules(self):
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

        # Act
        povmt = StandardPovmt(
            tester_objects, on_para_eq_constraint=True, num_outcomes=2
        )
        # Assert
        actual = povmt._experiment._schedules
        assert len(actual) == 4
        for i, a in enumerate(actual):
            expected = [("state", i), ("povm", 0)]
            assert a == expected

        # Case 2:
        # Act
        povmt = StandardPovmt(
            tester_objects, on_para_eq_constraint=True, num_outcomes=2, schedules="all"
        )
        # Assert
        actual = povmt._experiment._schedules
        assert len(actual) == 4
        for i, a in enumerate(actual):
            expected = [("state", i), ("povm", 0)]
            assert a == expected

        # Case 3:
        # Act
        schedules = [[("state", 2), ("povm", 0)], [("state", 1), ("povm", 0)]]
        povmt = StandardPovmt(
            tester_objects,
            on_para_eq_constraint=True,
            num_outcomes=2,
            schedules=schedules,
        )
        # Assert
        actual = povmt._experiment._schedules
        assert actual == schedules

        # Case 4:
        # Act
        invalid_schedules = "invalid str"
        with pytest.raises(ValueError):
            _ = StandardPovmt(
                tester_objects,
                on_para_eq_constraint=True,
                num_outcomes=2,
                schedules=invalid_schedules,
            )

    def test_generate_empi_dist(self):
        povmt, c_sys = get_test_data()
        povm = generate_povm_from_name("z", c_sys)

        # schedule_index = 0
        actual = povmt.generate_empi_dist(0, povm, 10)
        expected = (10, np.array([0.5, 0.5], dtype=np.float64))
        assert actual[0] == expected[0]
        npt.assert_almost_equal(actual[1], expected[1], decimal=15)

        # schedule_index = 1
        actual = povmt.generate_empi_dist(1, povm, 10)
        expected = (10, np.array([0.6, 0.4], dtype=np.float64))
        assert actual[0] == expected[0]
        npt.assert_almost_equal(actual[1], expected[1], decimal=15)

        # schedule_index = 2
        actual = povmt.generate_empi_dist(2, povm, 10)
        expected = (10, np.array([1.0, 0.0], dtype=np.float64))
        assert actual[0] == expected[0]
        npt.assert_almost_equal(actual[1], expected[1], decimal=15)

        # schedule_index = 3
        actual = povmt.generate_empi_dist(3, povm, 10)
        expected = (10, np.array([0.0, 1.0], dtype=np.float64))
        assert actual[0] == expected[0]
        npt.assert_almost_equal(actual[1], expected[1], decimal=15)

    def test_generate_empi_dist__seed_or_stream(self):
        povmt, c_sys = get_test_data()
        povm = generate_povm_from_name("z", c_sys)

        # seed_or_stream : default
        np.random.seed(7)
        actual1 = povmt.generate_empi_dist(0, povm, 10)
        # seed_or_stream : int
        actual2 = povmt.generate_empi_dist(0, povm, 10, seed_or_stream=7)
        # seed_or_stream : np.random.RandomState
        actual3 = povmt.generate_empi_dist(
            0, povm, 10, seed_or_stream=np.random.RandomState(7)
        )
        npt.assert_almost_equal(actual1[1], actual2[1], decimal=15)
        npt.assert_almost_equal(actual2[1], actual3[1], decimal=15)

    def test_generate_empi_dists(self):
        povmt, c_sys = get_test_data()
        povm = generate_povm_from_name("z", c_sys)

        actual = povmt.generate_empi_dists(povm, 10)
        expected = [
            (10, np.array([0.5, 0.5], dtype=np.float64)),
            (10, np.array([0.6, 0.4], dtype=np.float64)),
            (10, np.array([1.0, 0.0], dtype=np.float64)),
            (10, np.array([0.0, 1.0], dtype=np.float64)),
        ]
        for a, e in zip(actual, expected):
            assert a[0] == e[0]
            npt.assert_almost_equal(a[1], e[1], decimal=15)

    def test_generate_empi_dists_sequence(self):
        povmt, c_sys = get_test_data()
        povm = generate_povm_from_name("z", c_sys)

        actual = povmt.generate_empi_dists_sequence(povm, [10, 20])
        print(actual)
        expected = [
            [
                (10, np.array([0.5, 0.5], dtype=np.float64)),
                (10, np.array([0.3, 0.7], dtype=np.float64)),
                (10, np.array([1.0, 0.0], dtype=np.float64)),
                (10, np.array([0.0, 1.0], dtype=np.float64)),
            ],
            [
                (20, np.array([0.55, 0.45], dtype=np.float64)),
                (20, np.array([0.5, 0.5], dtype=np.float64)),
                (20, np.array([1.0, 0.0], dtype=np.float64)),
                (20, np.array([0.0, 1.0], dtype=np.float64)),
            ],
        ]
        for a_dists, e_dists in zip(actual, expected):
            for a, e in zip(a_dists, e_dists):
                assert a[0] == e[0]
                npt.assert_almost_equal(a[1], e[1], decimal=15)
