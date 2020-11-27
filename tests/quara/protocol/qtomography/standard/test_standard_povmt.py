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
from quara.protocol.qtomography.standard.standard_povmt import StandardPovmt


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
        actual = StandardPovmt(states, measurement_n=2, on_para_eq_constraint=False)
        assert actual.num_variables == 4  # TODO

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

        expected_b = np.array([0, 0, 0, 0, 0, 0, 0, 0,])
        npt.assert_almost_equal(actual.calc_vecB(), expected_b, decimal=15)

        # Case 1: m = 3
        # Act
        actual = StandardPovmt(states, measurement_n=3, on_para_eq_constraint=False)
        assert actual.num_variables == 4  # TODO

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
        actual = StandardPovmt(states, measurement_n=2, on_para_eq_constraint=True)

        # Assert
        assert actual.num_variables == 3  # TODO
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
        actual = StandardPovmt(states, measurement_n=3, on_para_eq_constraint=True)

        # Assert
        assert actual.num_variables == 3  # TODO
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
            _ = StandardPovmt(states, measurement_n=2, on_para_eq_constraint=False)

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
        actual = StandardPovmt(states, measurement_n=2, on_para_eq_constraint=False)
        setting_info = actual.generate_empty_estimation_obj_with_setting_info()

        expected = np.array([0, 0, 0, 0, 0, 0, 0, 0])
        npt.assert_almost_equal(setting_info.to_stacked_vector(), expected, decimal=15)
        assert setting_info.on_para_eq_constraint == False
        assert setting_info.on_algo_eq_constraint == False
        assert setting_info.on_algo_ineq_constraint == False

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
        measurement_n = len(true_object.vecs)
        povmt = StandardPovmt(
            tester_objects, measurement_n=measurement_n, on_para_eq_constraint=True
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
        measurement_n = len(true_object.vecs)
        povmt = StandardPovmt(
            tester_objects, measurement_n=measurement_n, on_para_eq_constraint=False
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
        measurement_n = len(true_object.vecs)
        povmt = StandardPovmt(
            tester_objects, measurement_n=measurement_n, on_para_eq_constraint=True
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
        measurement_n = len(true_object.vecs)
        povmt = StandardPovmt(
            tester_objects, measurement_n=measurement_n, on_para_eq_constraint=False
        )

        # Act
        actual = povmt.calc_mse_linear_analytical(
            true_object, [100] * len(tester_objects), mode="var"
        )

        # Assert
        npt.assert_almost_equal(actual, 0.04, decimal=15)
