import math

import numpy as np
import numpy.testing as npt
import pytest

from quara.objects import matrix_basis
from quara.objects.composite_system import CompositeSystem
from quara.objects.elemental_system import ElementalSystem
from quara.objects.matrix_basis import MatrixBasis
from quara.objects.povm import get_z_povm
from quara.objects.state import (
    State,
    to_density_matrix_from_var,
    to_var_from_density_matrix,
    to_vec_from_density_matrix_with_sparsity,
    convert_var_index_to_state_index,
    convert_state_index_to_var_index,
    convert_var_to_state,
    convert_vec_to_var,
    calc_gradient_from_state,
    get_bell_2q,
    get_x0_1q,
    get_x1_1q,
    get_y0_1q,
    get_y1_1q,
    get_z0_1q,
    get_z1_1q,
)
from quara.settings import Settings


class TestState:
    def test_init_error(self):
        e_sys = ElementalSystem(1, matrix_basis.get_comp_basis())
        c_sys = CompositeSystem([e_sys])

        # vec is not one-dimensional array
        with pytest.raises(ValueError):
            State(c_sys, np.array([[1, 0], [0, 0]], dtype=np.float64))

        # size is not square
        with pytest.raises(ValueError):
            State(c_sys, np.array([1, 0, 0], dtype=np.float64))

        # entries of vec are not real numbers
        with pytest.raises(ValueError):
            State(c_sys, np.array([1, 0, 0, 0], dtype=np.complex128))

        # dim of CompositeSystem does not equal dim of vec
        with pytest.raises(ValueError):
            State(c_sys, np.array([1], dtype=np.float64))

    def test_init_is_physicality_required(self):
        e_sys = ElementalSystem(1, matrix_basis.get_comp_basis())
        c_sys = CompositeSystem([e_sys])

        # trace of density matrix does not equal 1
        with pytest.raises(ValueError):
            State(c_sys, np.array([0.5, 0, 0, 0], dtype=np.float64))
        with pytest.raises(ValueError):
            State(
                c_sys,
                np.array([0.5, 0, 0, 0], dtype=np.float64),
                is_physicality_required=True,
            )

        # density matrix is not positive semidefinite
        with pytest.raises(ValueError):
            State(c_sys, np.array([2, 0, 0, -1], dtype=np.float64))
        with pytest.raises(ValueError):
            State(
                c_sys,
                np.array([2, 0, 0, -1], dtype=np.float64),
                is_physicality_required=True,
            )

        # case: when is_physicality_required is False, it is not happened ValueError
        State(
            c_sys,
            np.array([2, 0, 0, -1], dtype=np.float64),
            is_physicality_required=False,
        )
        State(
            c_sys,
            np.array([0.5, 0, 0, 0], dtype=np.float64),
            is_physicality_required=False,
        )

    def test_access_is_physicality_required(self):
        e_sys = ElementalSystem(0, matrix_basis.get_comp_basis())
        c_sys = CompositeSystem([e_sys])
        state = State(c_sys, np.array([1, 0, 0, 0], dtype=np.float64))
        assert state.is_physicality_required == True

        # Test that "is_physicality_required" cannot be updated
        with pytest.raises(AttributeError):
            state.is_physicality_required = False

    def test_access_vec(self):
        e_sys = ElementalSystem(0, matrix_basis.get_normalized_pauli_basis())
        c_sys = CompositeSystem([e_sys])
        state = get_x0_1q(c_sys)
        actual = state.vec
        expected = 1 / np.sqrt(2) * np.array([1, 1, 0, 0], dtype=np.float64)
        npt.assert_almost_equal(actual, expected, decimal=15)

        # Test that "vec" cannot be updated
        with pytest.raises(AttributeError):
            state.vec = (
                1 / np.sqrt(2) * np.array([1, 1, 0, 0], dtype=np.float64)
            )  # New vec

    def test_access_dim(self):
        e_sys = ElementalSystem(0, matrix_basis.get_normalized_pauli_basis())
        c_sys = CompositeSystem([e_sys])
        state = get_x0_1q(c_sys)
        assert state.dim == 2

        # Test that "dim" cannot be updated
        with pytest.raises(AttributeError):
            state.dim = 2  # New dim

    def test_is_physical(self):
        e_sys = ElementalSystem(1, matrix_basis.get_comp_basis())
        c_sys = CompositeSystem([e_sys])

        state = State(
            c_sys,
            np.array([1, 0, 0, 0], dtype=np.float64),
        )
        assert state.is_physical() == True

        state = State(
            c_sys,
            np.array([0.5, 0, 0, 0], dtype=np.float64),
            is_physicality_required=False,
        )
        assert state.is_physical() == False

        state = State(
            c_sys,
            np.array([2, 0, 0, -1], dtype=np.float64),
            is_physicality_required=False,
        )
        assert state.is_physical() == False

    def test_set_zero(self):
        e_sys = ElementalSystem(0, matrix_basis.get_normalized_pauli_basis())
        c_sys = CompositeSystem([e_sys])
        state = get_z0_1q(c_sys)
        state.set_zero()

        expected = np.zeros((4), dtype=np.float64)
        npt.assert_almost_equal(state.vec, expected, decimal=15)
        assert state.dim == 2
        assert state.is_physicality_required == False
        assert state.is_estimation_object == True
        assert state.on_para_eq_constraint == True
        assert state.on_algo_eq_constraint == True
        assert state.on_algo_ineq_constraint == True
        assert state.eps_proj_physical == Settings.get_atol() / 10.0

    def test_generate_zero_obj(self):
        e_sys = ElementalSystem(0, matrix_basis.get_normalized_pauli_basis())
        c_sys = CompositeSystem([e_sys])
        state = get_z0_1q(c_sys)
        zero = state.generate_zero_obj()

        expected = np.zeros((4), dtype=np.float64)
        npt.assert_almost_equal(zero.vec, expected, decimal=15)
        assert zero.dim == state.dim
        assert zero.is_physicality_required == False
        assert zero.is_estimation_object == False
        assert zero.on_para_eq_constraint == state.on_para_eq_constraint
        assert zero.on_algo_eq_constraint == state.on_algo_eq_constraint
        assert zero.on_algo_ineq_constraint == state.on_algo_ineq_constraint
        assert zero.eps_proj_physical == Settings.get_atol() / 10.0

    def test_generate_origin_obj(self):
        e_sys = ElementalSystem(0, matrix_basis.get_normalized_pauli_basis())
        c_sys = CompositeSystem([e_sys])
        state = get_z0_1q(c_sys)
        origin = state.generate_origin_obj()

        expected = np.array([1, 0, 0, 0], dtype=np.float64) / np.sqrt(2)
        npt.assert_almost_equal(origin.vec, expected, decimal=15)
        assert origin.dim == state.dim
        assert origin.is_physicality_required == False
        assert origin.is_estimation_object == False
        assert origin.on_para_eq_constraint == state.on_para_eq_constraint
        assert origin.on_algo_eq_constraint == state.on_algo_eq_constraint
        assert origin.on_algo_ineq_constraint == state.on_algo_ineq_constraint
        assert origin.eps_proj_physical == Settings.get_atol() / 10.0

    def test_copy(self):
        e_sys = ElementalSystem(0, matrix_basis.get_normalized_pauli_basis())
        c_sys = CompositeSystem([e_sys])
        state = get_z0_1q(c_sys)
        actual = state.copy()

        expected = np.array([1, 0, 0, 1], dtype=np.float64) / np.sqrt(2)
        npt.assert_almost_equal(actual.vec, expected, decimal=15)
        assert actual.dim == state.dim
        assert actual.is_physicality_required == state.is_physicality_required
        assert actual.is_estimation_object == state.is_estimation_object
        assert actual.on_para_eq_constraint == state.on_para_eq_constraint
        assert actual.on_algo_eq_constraint == state.on_algo_eq_constraint
        assert actual.on_algo_ineq_constraint == state.on_algo_ineq_constraint
        assert actual.eps_proj_physical == state.eps_proj_physical

    def test_add(self):
        # Arrange
        e_sys = ElementalSystem(0, matrix_basis.get_normalized_pauli_basis())
        c_sys = CompositeSystem([e_sys])

        vec1 = np.array([10, 20, 30, 40], dtype=np.float64)
        state1 = State(c_sys, vec1, is_physicality_required=False)
        vec2 = np.array([1, 2, 3, 4], dtype=np.float64)
        state2 = State(c_sys, vec2, is_physicality_required=False)

        # Act
        actual = state1 + state2

        # Assert
        expected_vec = np.array([11, 22, 33, 44], dtype=np.float64)
        assert type(actual) == State
        assert len(actual.vec) == len(expected_vec)
        npt.assert_almost_equal(actual.vec, expected_vec, decimal=15)

        assert actual.is_physicality_required == False
        assert actual.is_estimation_object == False
        assert actual.on_para_eq_constraint == state1.on_para_eq_constraint
        assert actual.on_algo_eq_constraint == state1.on_algo_eq_constraint
        assert actual.on_algo_ineq_constraint == state1.on_algo_ineq_constraint
        assert actual.eps_proj_physical == state1.eps_proj_physical

    def test_add_is_physicality_required(self):
        # Arrange
        e_sys = ElementalSystem(0, matrix_basis.get_normalized_pauli_basis())
        c_sys = CompositeSystem([e_sys])

        state1 = get_x0_1q(c_sys)
        state2 = get_z0_1q(c_sys)

        # Act
        actual = state1 + state2

        # Assert
        expected_vec = np.array([2, 1, 0, 1], dtype=np.float64) / np.sqrt(2)
        assert type(actual) == State
        assert len(actual.vec) == len(expected_vec)
        npt.assert_almost_equal(actual.vec, expected_vec, decimal=15)

        assert actual.is_physicality_required == False
        assert actual.is_estimation_object == False
        assert actual.on_para_eq_constraint == state1.on_para_eq_constraint
        assert actual.on_algo_eq_constraint == state1.on_algo_eq_constraint
        assert actual.on_algo_ineq_constraint == state1.on_algo_ineq_constraint
        assert actual.eps_proj_physical == state1.eps_proj_physical

    def test_add_exception(self):
        # Arrange
        e_sys = ElementalSystem(0, matrix_basis.get_normalized_pauli_basis())
        c_sys = CompositeSystem([e_sys])

        state1 = get_x0_1q(c_sys)

        # Case 1: different type
        # Arrange
        povm = get_z_povm(c_sys)

        # Act & Assert
        with pytest.raises(TypeError):
            _ = state1 + povm

        # Case 2: different on_para_eq_constraint
        # Arrange
        vec = np.array([10, 20, 30, 40], dtype=np.float64)
        state2 = State(
            c_sys=c_sys,
            vec=vec,
            is_physicality_required=False,
            on_para_eq_constraint=False,
        )

        # Act & Assert
        with pytest.raises(ValueError):
            _ = state1 + state2

        # Case 3: different CompositeSystem
        # Arrange
        e_sys2 = ElementalSystem(2, matrix_basis.get_normalized_pauli_basis())
        c_sys2 = CompositeSystem([e_sys])

        state2 = get_x0_1q(c_sys2)

        # Act & Assert
        with pytest.raises(ValueError):
            _ = state1 + state2

    def test_sub(self):
        # Arrange
        e_sys = ElementalSystem(0, matrix_basis.get_normalized_pauli_basis())
        c_sys = CompositeSystem([e_sys])

        vec1 = np.array([10, 20, 30, 40], dtype=np.float64)
        state1 = State(c_sys, vec1, is_physicality_required=False)
        vec2 = np.array([1, 2, 3, 4], dtype=np.float64)
        state2 = State(c_sys, vec2, is_physicality_required=False)

        # Act
        actual = state1 - state2

        # Assert
        expected_vec = np.array([9, 18, 27, 36], dtype=np.float64)
        assert type(actual) == State
        assert len(actual.vec) == len(expected_vec)
        npt.assert_almost_equal(actual.vec, expected_vec, decimal=15)

        assert actual.is_physicality_required == False
        assert actual.is_estimation_object == False
        assert actual.on_para_eq_constraint == state1.on_para_eq_constraint
        assert actual.on_algo_eq_constraint == state1.on_algo_eq_constraint
        assert actual.on_algo_ineq_constraint == state1.on_algo_ineq_constraint
        assert actual.eps_proj_physical == state1.eps_proj_physical

    def test_sub_is_physicality_required(self):
        # Arrange
        e_sys = ElementalSystem(0, matrix_basis.get_normalized_pauli_basis())
        c_sys = CompositeSystem([e_sys])

        state1 = get_x0_1q(c_sys)
        state2 = get_z0_1q(c_sys)

        # Act
        actual = state1 - state2

        # Assert
        expected_vec = np.array([0, 1, 0, -1], dtype=np.float64) / np.sqrt(2)
        assert type(actual) == State
        assert len(actual.vec) == len(expected_vec)
        npt.assert_almost_equal(actual.vec, expected_vec, decimal=15)

        assert actual.is_physicality_required == False
        assert actual.is_estimation_object == False
        assert actual.on_para_eq_constraint == state1.on_para_eq_constraint
        assert actual.on_algo_eq_constraint == state1.on_algo_eq_constraint
        assert actual.on_algo_ineq_constraint == state1.on_algo_ineq_constraint
        assert actual.eps_proj_physical == state1.eps_proj_physical

    def test_sub_exception(self):
        # Arrange
        e_sys = ElementalSystem(0, matrix_basis.get_normalized_pauli_basis())
        c_sys = CompositeSystem([e_sys])

        state1 = get_x0_1q(c_sys)

        # Case 1: different type
        # Arrange
        povm = get_z_povm(c_sys)

        # Act & Assert
        with pytest.raises(TypeError):
            _ = state1 - povm

        # Case 2: different on_para_eq_constraint
        # Arrange
        vec = np.array([10, 20, 30, 40], dtype=np.float64)
        state2 = State(
            c_sys=c_sys,
            vec=vec,
            is_physicality_required=False,
            on_para_eq_constraint=False,
        )

        # Act & Assert
        with pytest.raises(ValueError):
            _ = state1 - state2

        # Case 3: different CompositeSystem
        # Arrange
        e_sys2 = ElementalSystem(2, matrix_basis.get_normalized_pauli_basis())
        c_sys2 = CompositeSystem([e_sys])

        state2 = get_x0_1q(c_sys2)

        # Act & Assert
        with pytest.raises(ValueError):
            _ = state1 - state2

    def test_mul(self):
        # Arrange
        e_sys = ElementalSystem(0, matrix_basis.get_normalized_pauli_basis())
        c_sys = CompositeSystem([e_sys])

        vec1 = np.array([10, 20, 30, 40], dtype=np.float64)
        state = State(c_sys, vec1, is_physicality_required=False)

        # Case 1: type(int)
        # Act
        actual = state * 10

        # Assert
        expected_vec = np.array([100, 200, 300, 400], dtype=np.float64)
        assert type(actual) == State
        assert len(actual.vec) == len(expected_vec)
        npt.assert_almost_equal(actual.vec, expected_vec, decimal=15)

        assert actual.is_physicality_required == False
        assert actual.is_estimation_object == False
        assert actual.on_para_eq_constraint == state.on_para_eq_constraint
        assert actual.on_algo_eq_constraint == state.on_algo_eq_constraint
        assert actual.on_algo_ineq_constraint == state.on_algo_ineq_constraint
        assert actual.eps_proj_physical == state.eps_proj_physical

        # Case 2: type(float)
        # Act
        actual = state * 0.1

        # Assert
        expected_vec = np.array([1, 2, 3, 4], dtype=np.float64)
        assert type(actual) == State
        assert len(actual.vec) == len(expected_vec)
        npt.assert_almost_equal(actual.vec, expected_vec, decimal=15)

        assert actual.is_physicality_required == False
        assert actual.is_estimation_object == False
        assert actual.on_para_eq_constraint == state.on_para_eq_constraint
        assert actual.on_algo_eq_constraint == state.on_algo_eq_constraint
        assert actual.on_algo_ineq_constraint == state.on_algo_ineq_constraint
        assert actual.eps_proj_physical == state.eps_proj_physical

        # Case 3: type(np.int64)
        # Act
        actual = state * np.int64(10)

        # Assert
        expected_vec = np.array([100, 200, 300, 400], dtype=np.float64)
        assert type(actual) == State
        assert len(actual.vec) == len(expected_vec)
        npt.assert_almost_equal(actual.vec, expected_vec, decimal=15)

        assert actual.is_physicality_required == False
        assert actual.is_estimation_object == False
        assert actual.on_para_eq_constraint == state.on_para_eq_constraint
        assert actual.on_algo_eq_constraint == state.on_algo_eq_constraint
        assert actual.on_algo_ineq_constraint == state.on_algo_ineq_constraint
        assert actual.eps_proj_physical == state.eps_proj_physical

        # Case 4: type(np.float64)
        # Act
        actual = state * np.float64(0.1)

        # Assert
        expected_vec = np.array([1, 2, 3, 4], dtype=np.float64)
        assert type(actual) == State
        assert len(actual.vec) == len(expected_vec)
        npt.assert_almost_equal(actual.vec, expected_vec, decimal=15)

        assert actual.is_physicality_required == False
        assert actual.is_estimation_object == False
        assert actual.on_para_eq_constraint == state.on_para_eq_constraint
        assert actual.on_algo_eq_constraint == state.on_algo_eq_constraint
        assert actual.on_algo_ineq_constraint == state.on_algo_ineq_constraint
        assert actual.eps_proj_physical == state.eps_proj_physical

    def test_mul_convex_combination(self):
        # Arrange
        e_sys = ElementalSystem(0, matrix_basis.get_normalized_pauli_basis())
        c_sys = CompositeSystem([e_sys])

        state1 = get_x0_1q(c_sys)
        state2 = get_z0_1q(c_sys)

        # Act
        actual = 0.3 * state1 + 0.7 * state2

        # Assert
        expected_vec = np.array([1, 0.3, 0, 0.7], dtype=np.float64) / np.sqrt(2)
        assert type(actual) == State
        assert len(actual.vec) == len(expected_vec)
        npt.assert_almost_equal(actual.vec, expected_vec, decimal=15)

        assert actual.is_physicality_required == False
        assert actual.is_estimation_object == False
        assert actual.on_para_eq_constraint == state1.on_para_eq_constraint
        assert actual.on_algo_eq_constraint == state1.on_algo_eq_constraint
        assert actual.on_algo_ineq_constraint == state1.on_algo_ineq_constraint
        assert actual.eps_proj_physical == state1.eps_proj_physical
        # check is_physical
        assert actual.is_physical() == True

    def test_mul_exception(self):
        # Arrange
        e_sys = ElementalSystem(0, matrix_basis.get_normalized_pauli_basis())
        c_sys = CompositeSystem([e_sys])

        state = get_x0_1q(c_sys)

        # Case 1: different type(POVM)
        # Arrange
        povm = get_z_povm(c_sys)

        # Act & Assert
        with pytest.raises(TypeError):
            _ = state * povm

        # Case 2: different type(complex)
        # Act & Assert
        with pytest.raises(TypeError):
            _ = state * 1j

    def test_rmul(self):
        # Arrange
        e_sys = ElementalSystem(0, matrix_basis.get_normalized_pauli_basis())
        c_sys = CompositeSystem([e_sys])

        vec1 = np.array([10, 20, 30, 40], dtype=np.float64)
        state = State(c_sys, vec1, is_physicality_required=False)

        # Case 1: type(int)
        # Act
        actual = 10 * state

        # Assert
        expected_vec = np.array([100, 200, 300, 400], dtype=np.float64)
        assert type(actual) == State
        assert len(actual.vec) == len(expected_vec)
        npt.assert_almost_equal(actual.vec, expected_vec, decimal=15)

        assert actual.is_physicality_required == False
        assert actual.is_estimation_object == False
        assert actual.on_para_eq_constraint == state.on_para_eq_constraint
        assert actual.on_algo_eq_constraint == state.on_algo_eq_constraint
        assert actual.on_algo_ineq_constraint == state.on_algo_ineq_constraint
        assert actual.eps_proj_physical == state.eps_proj_physical

        # Case 2: type(float)
        # Act
        actual = 0.1 * state

        # Assert
        expected_vec = np.array([1, 2, 3, 4], dtype=np.float64)
        assert type(actual) == State
        assert len(actual.vec) == len(expected_vec)
        npt.assert_almost_equal(actual.vec, expected_vec, decimal=15)

        assert actual.is_physicality_required == False
        assert actual.is_estimation_object == False
        assert actual.on_para_eq_constraint == state.on_para_eq_constraint
        assert actual.on_algo_eq_constraint == state.on_algo_eq_constraint
        assert actual.on_algo_ineq_constraint == state.on_algo_ineq_constraint
        assert actual.eps_proj_physical == state.eps_proj_physical

        # Case 3: type(np.int64)
        # Act
        actual = np.int64(10) * state

        # Assert
        expected_vec = np.array([100, 200, 300, 400], dtype=np.float64)
        assert type(actual) == State
        assert len(actual.vec) == len(expected_vec)
        npt.assert_almost_equal(actual.vec, expected_vec, decimal=15)

        assert actual.is_physicality_required == False
        assert actual.is_estimation_object == False
        assert actual.on_para_eq_constraint == state.on_para_eq_constraint
        assert actual.on_algo_eq_constraint == state.on_algo_eq_constraint
        assert actual.on_algo_ineq_constraint == state.on_algo_ineq_constraint
        assert actual.eps_proj_physical == state.eps_proj_physical

        # Case 4: type(np.float64)
        # Act
        actual = np.float64(0.1) * state

        # Assert
        expected_vec = np.array([1, 2, 3, 4], dtype=np.float64)
        assert type(actual) == State
        assert len(actual.vec) == len(expected_vec)
        npt.assert_almost_equal(actual.vec, expected_vec, decimal=15)

        assert actual.is_physicality_required == False
        assert actual.is_estimation_object == False
        assert actual.on_para_eq_constraint == state.on_para_eq_constraint
        assert actual.on_algo_eq_constraint == state.on_algo_eq_constraint
        assert actual.on_algo_ineq_constraint == state.on_algo_ineq_constraint
        assert actual.eps_proj_physical == state.eps_proj_physical

    def test_rmul_exception(self):
        # Arrange
        e_sys = ElementalSystem(0, matrix_basis.get_normalized_pauli_basis())
        c_sys = CompositeSystem([e_sys])

        state = get_x0_1q(c_sys)

        # Case 1: different type(POVM)
        # Arrange
        povm = get_z_povm(c_sys)

        # Act & Assert
        with pytest.raises(TypeError):
            _ = povm * state

        # Case 2: different type(complex)
        # Act & Assert
        with pytest.raises(TypeError):
            _ = 1j * state

    def test_truediv(self):
        # Arrange
        e_sys = ElementalSystem(0, matrix_basis.get_normalized_pauli_basis())
        c_sys = CompositeSystem([e_sys])

        vec1 = np.array([10, 20, 30, 40], dtype=np.float64)
        state = State(c_sys, vec1, is_physicality_required=False)

        # Case 1: type(int)
        # Act
        actual = state / 10

        # Assert
        expected_vec = np.array([1, 2, 3, 4], dtype=np.float64)
        assert type(actual) == State
        assert len(actual.vec) == len(expected_vec)
        npt.assert_almost_equal(actual.vec, expected_vec, decimal=15)

        assert actual.is_physicality_required == False
        assert actual.is_estimation_object == False
        assert actual.on_para_eq_constraint == state.on_para_eq_constraint
        assert actual.on_algo_eq_constraint == state.on_algo_eq_constraint
        assert actual.on_algo_ineq_constraint == state.on_algo_ineq_constraint
        assert actual.eps_proj_physical == state.eps_proj_physical

        # Case 2: type(float)
        # Act
        actual = state / 0.1

        # Assert
        expected_vec = np.array([100, 200, 300, 400], dtype=np.float64)
        assert type(actual) == State
        assert len(actual.vec) == len(expected_vec)
        npt.assert_almost_equal(actual.vec, expected_vec, decimal=15)

        assert actual.is_physicality_required == False
        assert actual.is_estimation_object == False
        assert actual.on_para_eq_constraint == state.on_para_eq_constraint
        assert actual.on_algo_eq_constraint == state.on_algo_eq_constraint
        assert actual.on_algo_ineq_constraint == state.on_algo_ineq_constraint
        assert actual.eps_proj_physical == state.eps_proj_physical

        # Case 3: type(np.int64)
        # Act
        actual = state / np.int64(10)

        # Assert
        expected_vec = np.array([1, 2, 3, 4], dtype=np.float64)
        assert type(actual) == State
        assert len(actual.vec) == len(expected_vec)
        npt.assert_almost_equal(actual.vec, expected_vec, decimal=15)

        assert actual.is_physicality_required == False
        assert actual.is_estimation_object == False
        assert actual.on_para_eq_constraint == state.on_para_eq_constraint
        assert actual.on_algo_eq_constraint == state.on_algo_eq_constraint
        assert actual.on_algo_ineq_constraint == state.on_algo_ineq_constraint
        assert actual.eps_proj_physical == state.eps_proj_physical

        # Case 4: type(np.float64)
        # Act
        actual = state / np.float64(0.1)

        # Assert
        expected_vec = np.array([100, 200, 300, 400], dtype=np.float64)
        assert type(actual) == State
        assert len(actual.vec) == len(expected_vec)
        npt.assert_almost_equal(actual.vec, expected_vec, decimal=15)

        assert actual.is_physicality_required == False
        assert actual.is_estimation_object == False
        assert actual.on_para_eq_constraint == state.on_para_eq_constraint
        assert actual.on_algo_eq_constraint == state.on_algo_eq_constraint
        assert actual.on_algo_ineq_constraint == state.on_algo_ineq_constraint
        assert actual.eps_proj_physical == state.eps_proj_physical

        # Case 5: 0
        # Act
        actual = state / 0

        # Assert
        expected_vec = np.array(
            [float("inf"), float("inf"), float("inf"), float("inf")], dtype=np.float64
        )
        assert type(actual) == State
        assert len(actual.vec) == len(expected_vec)
        npt.assert_almost_equal(actual.vec, expected_vec, decimal=15)

        assert actual.is_physicality_required == False
        assert actual.is_estimation_object == False
        assert actual.on_para_eq_constraint == state.on_para_eq_constraint
        assert actual.on_algo_eq_constraint == state.on_algo_eq_constraint
        assert actual.on_algo_ineq_constraint == state.on_algo_ineq_constraint
        assert actual.eps_proj_physical == state.eps_proj_physical

    def test_truediv_exception(self):
        # Arrange
        e_sys = ElementalSystem(0, matrix_basis.get_normalized_pauli_basis())
        c_sys = CompositeSystem([e_sys])

        state = get_x0_1q(c_sys)

        # Case 1: different type(POVM)
        # Arrange
        povm = get_z_povm(c_sys)

        # Act & Assert
        with pytest.raises(TypeError):
            _ = state / povm

        # Case 2: different type(complex)
        # Act & Assert
        with pytest.raises(TypeError):
            _ = state / 1j

    def test_to_var(self):
        e_sys = ElementalSystem(0, matrix_basis.get_normalized_pauli_basis())
        c_sys = CompositeSystem([e_sys])

        # case: on_para_eq_constraint = True
        state = get_z0_1q(c_sys)
        var = state.to_var()

        expected = np.array([0, 0, 1], dtype=np.float64) / np.sqrt(2)
        npt.assert_almost_equal(var, expected, decimal=15)

        # case: on_para_eq_constraint = False
        vec = np.array([1, 0, 0, 1], dtype=np.float64) / np.sqrt(2)
        state = State(c_sys, vec, on_para_eq_constraint=False)
        var = state.to_var()

        expected = np.array([1, 0, 0, 1], dtype=np.float64) / np.sqrt(2)
        npt.assert_almost_equal(var, expected, decimal=15)

    def test_to_stacked_vector(self):
        e_sys = ElementalSystem(0, matrix_basis.get_normalized_pauli_basis())
        c_sys = CompositeSystem([e_sys])
        state = get_z0_1q(c_sys)
        vector = state.to_stacked_vector()

        expected = np.array([1, 0, 0, 1], dtype=np.float64) / np.sqrt(2)
        npt.assert_almost_equal(vector, expected, decimal=15)

    def test_calc_gradient(self):
        e_sys = ElementalSystem(0, matrix_basis.get_normalized_pauli_basis())
        c_sys = CompositeSystem([e_sys])

        # case: on_para_eq_constraint=True
        state = get_z0_1q(c_sys)
        actual = state.calc_gradient(0)

        expected = np.array([0, 1, 0, 0], dtype=np.float64)
        npt.assert_almost_equal(actual.vec, expected, decimal=15)
        assert actual.is_physicality_required == False
        assert actual.is_estimation_object == True
        assert actual.on_para_eq_constraint == True
        assert actual.on_algo_eq_constraint == True
        assert actual.on_algo_ineq_constraint == True
        assert actual.eps_proj_physical == Settings.get_atol() / 10.0

        # case: on_para_eq_constraint=False
        vec = np.array([1, 0, 0, 1], dtype=np.float64) / np.sqrt(2)
        state = State(c_sys, vec, on_para_eq_constraint=False)
        actual = state.calc_gradient(0)

        expected = np.array([1, 0, 0, 0], dtype=np.float64)
        npt.assert_almost_equal(actual.vec, expected, decimal=15)
        assert actual.is_physicality_required == False
        assert actual.is_estimation_object == True
        assert actual.on_para_eq_constraint == False
        assert actual.on_algo_eq_constraint == True
        assert actual.on_algo_ineq_constraint == True
        assert actual.eps_proj_physical == Settings.get_atol() / 10.0

    def test_calc_proj_eq_constraint(self):
        e_sys = ElementalSystem(0, matrix_basis.get_normalized_pauli_basis())
        c_sys = CompositeSystem([e_sys])
        vec = np.array([1, 0, 0, 0], dtype=np.float64)
        state = State(c_sys, vec, is_physicality_required=False)
        actual = state.calc_proj_eq_constraint()

        expected = np.array([1, 0, 0, 0], dtype=np.float64) / np.sqrt(2)
        npt.assert_almost_equal(actual.vec, expected, decimal=15)
        assert actual.is_hermitian() == True
        assert actual.is_trace_one() == True

        npt.assert_almost_equal(
            state.vec, np.array([1, 0, 0, 0], dtype=np.float64), decimal=15
        )

    def test_func_calc_proj_eq_constraint(self):
        e_sys = ElementalSystem(0, matrix_basis.get_normalized_pauli_basis())
        c_sys = CompositeSystem([e_sys])
        vec = np.array([1, 0, 0, 0], dtype=np.float64)
        state = State(c_sys, vec, is_physicality_required=False)
        func = state.func_calc_proj_eq_constraint(False)

        # case1: var = [1, 0, 0, 0]/sqrt(2)
        var = np.array([1, 0, 0, 0], dtype=np.float64) / np.sqrt(2)
        actual = func(var)
        expected = np.array([1, 0, 0, 0], dtype=np.float64) / np.sqrt(2)
        npt.assert_almost_equal(actual, expected, decimal=15)

        # case2: var = [1, 0, 0, 0]
        var = np.array([1, 0, 0, 0], dtype=np.float64)
        actual = func(var)
        expected = np.array([1, 0, 0, 0], dtype=np.float64) / np.sqrt(2)
        npt.assert_almost_equal(actual, expected, decimal=15)

    def test_calc_proj_eq_constraint_with_var(self):
        e_sys = ElementalSystem(0, matrix_basis.get_normalized_pauli_basis())
        c_sys = CompositeSystem([e_sys])
        vec = np.array([1, 0, 0, 0], dtype=np.float64)
        state = State(c_sys, vec, is_physicality_required=False)

        # case1: on_para_eq_constraint :default(True)
        actual = state.calc_proj_eq_constraint_with_var(c_sys, state.to_var())
        expected = np.array([0, 0, 0], dtype=np.float64) / np.sqrt(2)
        npt.assert_almost_equal(actual, expected, decimal=15)

        # case2: on_para_eq_constraint=True
        actual = state.calc_proj_eq_constraint_with_var(
            c_sys, state.to_var(), on_para_eq_constraint=True
        )
        expected = np.array([0, 0, 0], dtype=np.float64) / np.sqrt(2)
        npt.assert_almost_equal(actual, expected, decimal=15)

        # case3: on_para_eq_constraint=False
        actual = state.calc_proj_eq_constraint_with_var(
            c_sys, state.to_stacked_vector(), on_para_eq_constraint=False
        )
        expected = np.array([1, 0, 0, 0], dtype=np.float64) / np.sqrt(2)
        npt.assert_almost_equal(actual, expected, decimal=15)

    def test_calc_proj_ineq_constraint(self):
        e_sys = ElementalSystem(0, matrix_basis.get_normalized_pauli_basis())
        c_sys = CompositeSystem([e_sys])
        vec = np.array([1, 2, 0, 0], dtype=np.float64) / np.sqrt(2)
        state = State(c_sys, vec, is_physicality_required=False)
        actual = state.calc_proj_ineq_constraint()

        expected = np.array(
            [3 * np.sqrt(2) / 4, 3 * np.sqrt(2) / 4, 0, 0], dtype=np.float64
        )
        npt.assert_almost_equal(actual.vec, expected, decimal=15)
        assert actual.is_hermitian() == True
        assert actual.is_positive_semidefinite() == True

    def test_func_calc_proj_ineq_constraint(self):
        e_sys = ElementalSystem(0, matrix_basis.get_normalized_pauli_basis())
        c_sys = CompositeSystem([e_sys])
        vec = np.array([1, 2, 0, 0], dtype=np.float64) / np.sqrt(2)
        state = State(c_sys, vec, is_physicality_required=False)
        func = state.func_calc_proj_ineq_constraint(False)

        # case1: var = [1, 2, 0, 0]/sqrt(2)
        var = np.array([1, 2, 0, 0], dtype=np.float64) / np.sqrt(2)
        actual = func(var)

        expected = np.array(
            [3 * np.sqrt(2) / 4, 3 * np.sqrt(2) / 4, 0, 0], dtype=np.float64
        )
        npt.assert_almost_equal(actual, expected, decimal=15)

    def test_calc_proj_ineq_constraint_with_var(self):
        e_sys = ElementalSystem(0, matrix_basis.get_normalized_pauli_basis())
        c_sys = CompositeSystem([e_sys])
        vec = np.array([1, 2, 0, 0], dtype=np.float64) / np.sqrt(2)
        state = State(c_sys, vec, is_physicality_required=False)

        # case1: on_para_eq_constraint: default(True)
        actual = state.calc_proj_ineq_constraint_with_var(c_sys, state.to_var())
        expected = np.array([3 * np.sqrt(2) / 4, 0, 0], dtype=np.float64)
        npt.assert_almost_equal(actual, expected, decimal=15)

        # case2: on_para_eq_constraint=True
        actual = state.calc_proj_ineq_constraint_with_var(
            c_sys, state.to_var(), on_para_eq_constraint=True
        )
        expected = np.array([3 * np.sqrt(2) / 4, 0, 0], dtype=np.float64)
        npt.assert_almost_equal(actual, expected, decimal=15)

        # case3: on_para_eq_constraint=False
        actual = state.calc_proj_ineq_constraint_with_var(
            c_sys, state.to_stacked_vector(), on_para_eq_constraint=False
        )
        expected = np.array(
            [3 * np.sqrt(2) / 4, 3 * np.sqrt(2) / 4, 0, 0], dtype=np.float64
        )
        npt.assert_almost_equal(actual, expected, decimal=15)

    def test_to_density_matrix(self):
        e_sys = ElementalSystem(0, matrix_basis.get_normalized_pauli_basis())
        c_sys = CompositeSystem([e_sys])
        state = get_z0_1q(c_sys)
        actual = state.to_density_matrix()
        expected = np.array([[1, 0], [0, 0]], dtype=np.float64)
        npt.assert_almost_equal(actual, expected, decimal=15)

    def test_to_density_matrix_with_sparsity(self):
        e_sys = ElementalSystem(0, matrix_basis.get_normalized_pauli_basis())
        c_sys = CompositeSystem([e_sys])
        state = get_z0_1q(c_sys)
        actual = state.to_density_matrix_with_sparsity()
        expected = np.array([[1, 0], [0, 0]], dtype=np.float64)
        npt.assert_almost_equal(actual, expected, decimal=15)

    def test_is_trace_one(self):
        # case: True
        e_sys = ElementalSystem(0, matrix_basis.get_comp_basis())
        c_sys = CompositeSystem([e_sys])
        state = State(
            c_sys,
            np.array([1, 0, 0, 0], dtype=np.float64),
            is_physicality_required=False,
        )
        assert state.is_trace_one() == True
        assert state.is_eq_constraint_satisfied() == True

        # case: False
        e_sys = ElementalSystem(0, matrix_basis.get_comp_basis())
        c_sys = CompositeSystem([e_sys])
        state = State(
            c_sys,
            np.array([0, 1, 0, 0], dtype=np.float64),
            is_physicality_required=False,
        )
        assert state.is_trace_one() == False
        assert state.is_eq_constraint_satisfied() == False

        # case: specify atol
        e_sys = ElementalSystem(0, matrix_basis.get_comp_basis())
        c_sys = CompositeSystem([e_sys])
        state = State(
            c_sys,
            np.array([1.001, 0, 0, 0], dtype=np.float64),
            is_physicality_required=False,
        )
        assert state.is_trace_one(atol=1e-2) == True
        assert state.is_eq_constraint_satisfied(atol=1e-2) == True

    def test_is_hermitian(self):
        # case: True
        e_sys = ElementalSystem(0, matrix_basis.get_normalized_pauli_basis())
        c_sys = CompositeSystem([e_sys])
        state = get_z0_1q(c_sys)
        assert state.is_hermitian() == True

        # case: False
        e_sys = ElementalSystem(0, matrix_basis.get_comp_basis())
        c_sys = CompositeSystem([e_sys])
        state = State(
            c_sys,
            np.array([0, 1, 0, 0], dtype=np.float64),
            is_physicality_required=False,
        )
        assert state.is_hermitian() == False

        # case: specify atol
        e_sys = ElementalSystem(0, matrix_basis.get_comp_basis())
        c_sys = CompositeSystem([e_sys])
        state = State(
            c_sys,
            np.array([0, 1, 1.001, 0], dtype=np.float64),
            is_physicality_required=False,
        )
        assert state.is_hermitian(atol=1e-2) == True

    def test_is_positive_semidefinite(self):
        # case: True
        e_sys = ElementalSystem(0, matrix_basis.get_normalized_pauli_basis())
        c_sys = CompositeSystem([e_sys])
        state = get_z0_1q(c_sys)
        assert state.is_positive_semidefinite() == True
        assert state.is_ineq_constraint_satisfied() == True

        # case: False
        e_sys = ElementalSystem(0, matrix_basis.get_comp_basis())
        c_sys = CompositeSystem([e_sys])
        state = State(
            c_sys,
            np.array([-1, 0, 0, 0], dtype=np.float64),
            is_physicality_required=False,
        )
        assert state.is_positive_semidefinite() == False
        assert state.is_ineq_constraint_satisfied() == False

        # case: specify atol
        e_sys = ElementalSystem(0, matrix_basis.get_comp_basis())
        c_sys = CompositeSystem([e_sys])
        state = State(
            c_sys,
            np.array([-0.001, 0, 0, 0], dtype=np.float64),
            is_physicality_required=False,
        )
        assert state.is_positive_semidefinite(atol=1e-2) == True
        assert state.is_ineq_constraint_satisfied(atol=1e-2) == True

    def test_calc_eigenvalues(self):
        e_sys = ElementalSystem(0, matrix_basis.get_normalized_pauli_basis())
        c_sys = CompositeSystem([e_sys])
        state = get_z0_1q(c_sys)
        actual = state.calc_eigenvalues()
        expected = np.array([1, 0], dtype=np.complex128)
        npt.assert_almost_equal(actual, expected, decimal=15)

    def test_comp_basis(self):
        e_sys = ElementalSystem(1, matrix_basis.get_comp_basis())
        c_sys = CompositeSystem([e_sys])

        # test for vec[1, 0, 0, 0]
        state = State(
            c_sys,
            np.array([1, 0, 0, 0], dtype=np.float64),
            is_physicality_required=False,
        )
        assert state.dim == 2
        assert np.all(
            state.to_density_matrix() == np.array([[1, 0], [0, 0]], dtype=np.complex128)
        )
        assert state.is_trace_one() == True
        assert state.is_hermitian() == True
        assert state.is_positive_semidefinite() == True
        assert np.all(state.calc_eigenvalues() == np.array([1, 0], dtype=np.complex128))

        # test for vec[0, 1, 0, 0]
        state = State(
            c_sys,
            np.array([0, 1, 0, 0], dtype=np.float64),
            is_physicality_required=False,
        )
        assert state.dim == 2
        assert np.all(
            state.to_density_matrix() == np.array([[0, 1], [0, 0]], dtype=np.complex128)
        )
        assert state.is_trace_one() == False
        assert state.is_hermitian() == False
        assert state.is_positive_semidefinite() == False
        assert np.all(state.calc_eigenvalues() == np.array([0, 0], dtype=np.complex128))

        # test for vec[0, 0, 1, 0]
        state = State(
            c_sys,
            np.array([0, 0, 1, 0], dtype=np.float64),
            is_physicality_required=False,
        )
        assert state.dim == 2
        assert np.all(
            state.to_density_matrix() == np.array([[0, 0], [1, 0]], dtype=np.complex128)
        )
        assert state.is_trace_one() == False
        assert state.is_hermitian() == False
        assert state.is_positive_semidefinite() == False
        assert np.all(state.calc_eigenvalues() == np.array([0, 0], dtype=np.complex128))

        # test for vec[0, 0, 0, 1]
        state = State(
            c_sys,
            np.array([0, 0, 0, 1], dtype=np.float64),
            is_physicality_required=False,
        )
        assert state.dim == 2
        assert np.all(
            state.to_density_matrix() == np.array([[0, 0], [0, 1]], dtype=np.complex128)
        )
        assert state.is_trace_one() == True
        assert state.is_hermitian() == True
        assert state.is_positive_semidefinite() == True
        assert np.all(state.calc_eigenvalues() == np.array([1, 0], dtype=np.complex128))

    def test_pauli_basis(self):
        e_sys = ElementalSystem(1, matrix_basis.get_pauli_basis())
        c_sys = CompositeSystem([e_sys])

        # test for vec[1, 0, 0, 0]
        state = State(
            c_sys,
            np.array([1, 0, 0, 0], dtype=np.float64),
            is_physicality_required=False,
        )
        assert state.dim == 2
        assert np.all(
            state.to_density_matrix() == np.array([[1, 0], [0, 1]], dtype=np.complex128)
        )
        assert state.is_trace_one() == False
        assert state.is_hermitian() == True
        assert state.is_positive_semidefinite() == True
        npt.assert_almost_equal(
            state.calc_eigenvalues(),
            np.array([1, 1], dtype=np.complex128),
            decimal=15,
        )

        # test for vec [0, 1, 0, 0]
        state = State(
            c_sys,
            np.array([0, 1, 0, 0], dtype=np.float64),
            is_physicality_required=False,
        )
        assert state.dim == 2
        assert np.all(
            state.to_density_matrix() == np.array([[0, 1], [1, 0]], dtype=np.complex128)
        )
        assert state.is_trace_one() == False
        assert state.is_hermitian() == True
        assert state.is_positive_semidefinite() == False
        npt.assert_almost_equal(
            state.calc_eigenvalues(),
            np.array([1, -1], dtype=np.complex128),
            decimal=15,
        )

        # test for vec [0, 0, 1, 0]
        state = State(
            c_sys,
            np.array([0, 0, 1, 0], dtype=np.float64),
            is_physicality_required=False,
        )
        assert state.dim == 2
        assert np.all(
            state.to_density_matrix()
            == np.array([[0, -1j], [1j, 0]], dtype=np.complex128)
        )
        assert state.is_trace_one() == False
        assert state.is_hermitian() == True
        assert state.is_positive_semidefinite() == False
        npt.assert_almost_equal(
            state.calc_eigenvalues(),
            np.array([1, -1], dtype=np.complex128),
            decimal=15,
        )

        # test for vec [0, 0, 0, 1]
        state = State(
            c_sys,
            np.array([0, 0, 0, 1], dtype=np.float64),
            is_physicality_required=False,
        )
        assert state.dim == 2
        assert np.all(
            state.to_density_matrix()
            == np.array([[1, 0], [0, -1]], dtype=np.complex128)
        )
        assert state.is_trace_one() == False
        assert state.is_hermitian() == True
        assert state.is_positive_semidefinite() == False
        npt.assert_almost_equal(
            state.calc_eigenvalues(),
            np.array([1, -1], dtype=np.complex128),
            decimal=15,
        )

    def test_normalized_pauli_basis(self):
        e_sys = ElementalSystem(1, matrix_basis.get_normalized_pauli_basis())
        c_sys = CompositeSystem([e_sys])

        # test for vec[1, 0, 0, 0]
        state = State(
            c_sys,
            np.array([1, 0, 0, 0], dtype=np.float64),
            is_physicality_required=False,
        )
        assert state.dim == 2
        assert np.all(
            state.to_density_matrix()
            == 1 / np.sqrt(2) * np.array([[1, 0], [0, 1]], dtype=np.complex128)
        )
        assert state.is_trace_one() == False
        assert state.is_hermitian() == True
        assert state.is_positive_semidefinite() == True
        npt.assert_almost_equal(
            state.calc_eigenvalues(),
            np.array([1 / np.sqrt(2), 1 / np.sqrt(2)], dtype=np.complex128),
            decimal=15,
        )

        # test for vec [0, 1, 0, 0]
        state = State(
            c_sys,
            np.array([0, 1, 0, 0], dtype=np.float64),
            is_physicality_required=False,
        )
        assert state.dim == 2
        assert np.all(
            state.to_density_matrix()
            == 1 / np.sqrt(2) * np.array([[0, 1], [1, 0]], dtype=np.complex128)
        )
        assert state.is_trace_one() == False
        assert state.is_hermitian() == True
        assert state.is_positive_semidefinite() == False
        npt.assert_almost_equal(
            state.calc_eigenvalues(),
            np.array([1 / np.sqrt(2), -1 / np.sqrt(2)], dtype=np.complex128),
            decimal=15,
        )

        # test for vec [0, 0, 1, 0]
        state = State(
            c_sys,
            np.array([0, 0, 1, 0], dtype=np.float64),
            is_physicality_required=False,
        )
        assert state.dim == 2
        assert np.all(
            state.to_density_matrix()
            == 1 / np.sqrt(2) * np.array([[0, -1j], [1j, 0]], dtype=np.complex128)
        )
        assert state.is_trace_one() == False
        assert state.is_hermitian() == True
        assert state.is_positive_semidefinite() == False
        npt.assert_almost_equal(
            state.calc_eigenvalues(),
            np.array([1 / np.sqrt(2), -1 / np.sqrt(2)], dtype=np.complex128),
            decimal=15,
        )

        # test for vec [0, 0, 0, 1]
        state = State(
            c_sys,
            np.array([0, 0, 0, 1], dtype=np.float64),
            is_physicality_required=False,
        )
        assert state.dim == 2
        assert np.all(
            state.to_density_matrix()
            == 1 / np.sqrt(2) * np.array([[1, 0], [0, -1]], dtype=np.complex128)
        )
        assert state.is_trace_one() == False
        assert state.is_hermitian() == True
        assert state.is_positive_semidefinite() == False
        npt.assert_almost_equal(
            state.calc_eigenvalues(),
            np.array([1 / np.sqrt(2), -1 / np.sqrt(2)], dtype=np.complex128),
            decimal=15,
        )

    def test_convert_basis_form_comp_to_pauli(self):
        pauli_basis = matrix_basis.get_normalized_pauli_basis()

        # CompositeSystem of comp basis
        e_sys1 = ElementalSystem(1, matrix_basis.get_comp_basis())
        c_sys1 = CompositeSystem([e_sys1])

        # converts [1, 0, 0, 0] with comp basis to Pauli basis
        state = State(
            c_sys1,
            np.array([1, 0, 0, 0], dtype=np.float64),
            is_physicality_required=False,
        )
        actual = state.convert_basis(pauli_basis)
        expected = 1 / np.sqrt(2) * np.array([1, 0, 0, 1], dtype=np.complex128)
        assert np.all(actual == expected)

        # converts [0, 1, 0, 0] with comp basis to Pauli basis
        state = State(
            c_sys1,
            np.array([0, 1, 0, 0], dtype=np.float64),
            is_physicality_required=False,
        )
        actual = state.convert_basis(pauli_basis)
        expected = 1 / np.sqrt(2) * np.array([0, 1, 1j, 0], dtype=np.complex128)
        assert np.all(actual == expected)

        # converts [0, 0, 1, 0] with comp basis to Pauli basis
        state = State(
            c_sys1,
            np.array([0, 0, 1, 0], dtype=np.float64),
            is_physicality_required=False,
        )
        actual = state.convert_basis(pauli_basis)
        expected = 1 / np.sqrt(2) * np.array([0, 1, -1j, 0], dtype=np.complex128)
        assert np.all(actual == expected)

        # converts [0, 0, 0, 1] with comp basis to Pauli basis
        state = State(
            c_sys1,
            np.array([0, 0, 0, 1], dtype=np.float64),
            is_physicality_required=False,
        )
        actual = state.convert_basis(pauli_basis)
        expected = 1 / np.sqrt(2) * np.array([1, 0, 0, -1], dtype=np.complex128)
        assert np.all(actual == expected)

    def test_convert_basis_form_pauli_to_comp(self):
        comp_basis = matrix_basis.get_comp_basis()

        # CompositeSystem of Pauli basis
        e_sys2 = ElementalSystem(2, matrix_basis.get_normalized_pauli_basis())
        c_sys2 = CompositeSystem([e_sys2])

        # converts [1, 0, 0, 0] with Pauli basis to comp basis
        state = State(
            c_sys2,
            np.array([1, 0, 0, 0], dtype=np.float64),
            is_physicality_required=False,
        )
        actual = state.convert_basis(comp_basis)
        expected = 1 / np.sqrt(2) * np.array([1, 0, 0, 1], dtype=np.complex128)
        assert np.all(actual == expected)

        # converts [0, 1, 0, 0] with Pauli basis to comp basis
        state = State(
            c_sys2,
            np.array([0, 1, 0, 0], dtype=np.float64),
            is_physicality_required=False,
        )
        actual = state.convert_basis(comp_basis)
        expected = 1 / np.sqrt(2) * np.array([0, 1, 1, 0], dtype=np.complex128)
        assert np.all(actual == expected)

        # converts [0, 0, 1, 0] with Pauli basis to comp basis
        state = State(
            c_sys2,
            np.array([0, 0, 1, 0], dtype=np.float64),
            is_physicality_required=False,
        )
        actual = state.convert_basis(comp_basis)
        expected = 1 / np.sqrt(2) * np.array([0, -1j, 1j, 0], dtype=np.complex128)
        assert np.all(actual == expected)

        # converts [0, 0, 0, 1] with Pauli basis to comp basis
        state = State(
            c_sys2,
            np.array([0, 0, 0, 1], dtype=np.float64),
            is_physicality_required=False,
        )
        actual = state.convert_basis(comp_basis)
        expected = 1 / np.sqrt(2) * np.array([1, 0, 0, -1], dtype=np.complex128)
        assert np.all(actual == expected)

    def test_generate_from_var(self):
        # Arrange
        e_sys = ElementalSystem(0, matrix_basis.get_normalized_pauli_basis())
        c_sys = CompositeSystem([e_sys])

        from_vec = 1 / np.sqrt(2) * np.array([1, 1, 0, 0], dtype=np.float64)
        from_basis = matrix_basis.get_normalized_pauli_basis()
        to_vec = matrix_basis.convert_vec(from_vec, from_basis, c_sys.basis())
        vec = to_vec.real.astype(np.float64)

        init_is_physicality_required = False
        init_is_estimation_object = True
        init_on_para_eq_constraint = False
        init_on_algo_eq_constraint = True
        init_on_algo_ineq_constraint = False
        init_eps_proj_physical = 10 ** (-3)

        source_state = State(
            c_sys,
            vec=vec,
            is_physicality_required=init_is_physicality_required,
            is_estimation_object=init_is_estimation_object,
            on_para_eq_constraint=init_on_para_eq_constraint,
            on_algo_eq_constraint=init_on_algo_eq_constraint,
            on_algo_ineq_constraint=init_on_algo_ineq_constraint,
            eps_proj_physical=init_eps_proj_physical,
        )

        # Case 1: default
        var = np.array([1, 2, 3, 4], dtype=np.float64)
        # Act
        actual = source_state.generate_from_var(var)
        # Assert
        expected = np.array([1, 2, 3, 4], dtype=np.float64)
        assert np.all(actual.vec == expected)
        assert actual.composite_system is c_sys
        assert actual.is_physicality_required is init_is_physicality_required
        assert actual.is_estimation_object is init_is_estimation_object
        assert actual.on_para_eq_constraint is init_on_para_eq_constraint
        assert actual.on_algo_eq_constraint is init_on_algo_eq_constraint
        assert actual.on_algo_ineq_constraint is init_on_algo_ineq_constraint
        assert actual.eps_proj_physical is init_eps_proj_physical

        # Case 2:
        with pytest.raises(ValueError):
            # ValueError: the state is not physically correct.
            _ = source_state.generate_from_var(var, is_physicality_required=True)

        # Case 3:
        # Arrange
        var = np.array([1, 2, 3], dtype=np.float64)
        source_is_estimation_object = False
        source_on_para_eq_constraint = True
        source_on_algo_eq_constraint = False
        source_on_algo_ineq_constraint = True
        source_eps_proj_physical = 10 ** (-2)

        # Act
        actual = source_state.generate_from_var(
            var,
            is_estimation_object=source_is_estimation_object,
            on_para_eq_constraint=source_on_para_eq_constraint,
            on_algo_eq_constraint=source_on_algo_eq_constraint,
            on_algo_ineq_constraint=source_on_algo_ineq_constraint,
            eps_proj_physical=source_eps_proj_physical,
        )

        # Assert
        expected = np.array([1 / np.sqrt(2), 1, 2, 3], dtype=np.float64)
        assert np.all(actual.vec == expected)
        assert actual.composite_system is c_sys
        assert actual.is_physicality_required is init_is_physicality_required
        assert actual.is_estimation_object is source_is_estimation_object
        assert actual.on_para_eq_constraint is source_on_para_eq_constraint
        assert actual.on_algo_eq_constraint is source_on_algo_eq_constraint
        assert actual.on_algo_ineq_constraint is source_on_algo_ineq_constraint
        assert actual.eps_proj_physical == source_eps_proj_physical

    def test_convert_var_to_stacked_vector(self):
        e_sys = ElementalSystem(0, matrix_basis.get_normalized_pauli_basis())
        c_sys = CompositeSystem([e_sys])
        state = get_z0_1q(c_sys)

        # case1: on_para_eq_constraint: default(True)
        var = np.array([2, 3, 4], dtype=np.float64)
        actual = state.convert_var_to_stacked_vector(c_sys, var)
        expected = np.array([1 / np.sqrt(2), 2, 3, 4], dtype=np.float64)
        npt.assert_almost_equal(actual, expected, decimal=15)

        # case2: on_para_eq_constraint=True
        var = np.array([2, 3, 4], dtype=np.float64)
        actual = state.convert_var_to_stacked_vector(
            c_sys, var, on_para_eq_constraint=True
        )
        expected = np.array([1 / np.sqrt(2), 2, 3, 4], dtype=np.float64)
        npt.assert_almost_equal(actual, expected, decimal=15)

        # case3: on_para_eq_constraint=False
        var = np.array([1, 2, 3, 4], dtype=np.float64)
        actual = state.convert_var_to_stacked_vector(
            c_sys, var, on_para_eq_constraint=False
        )
        expected = np.array([1, 2, 3, 4], dtype=np.float64)
        npt.assert_almost_equal(actual, expected, decimal=15)

    def test_convert_stacked_vector_to_var(self):
        e_sys = ElementalSystem(0, matrix_basis.get_normalized_pauli_basis())
        c_sys = CompositeSystem([e_sys])
        state = get_z0_1q(c_sys)
        vec = np.array([1, 2, 3, 4], dtype=np.float64)

        # case1: on_para_eq_constraint: default(True)
        actual = state.convert_stacked_vector_to_var(c_sys, vec)
        expected = np.array([2, 3, 4], dtype=np.float64)
        npt.assert_almost_equal(actual, expected, decimal=15)

        # case2: on_para_eq_constraint=True
        actual = state.convert_stacked_vector_to_var(
            c_sys, vec, on_para_eq_constraint=True
        )
        expected = np.array([2, 3, 4], dtype=np.float64)
        npt.assert_almost_equal(actual, expected, decimal=15)

        # case3: on_para_eq_constraint=False
        actual = state.convert_stacked_vector_to_var(
            c_sys, vec, on_para_eq_constraint=False
        )
        expected = np.array([1, 2, 3, 4], dtype=np.float64)
        npt.assert_almost_equal(actual, expected, decimal=15)

    def test_calc_proj_physical(self):
        e_sys = ElementalSystem(0, matrix_basis.get_normalized_pauli_basis())
        c_sys = CompositeSystem([e_sys])

        # z0 -> z0
        z0 = get_z0_1q(c_sys)
        actual = z0.calc_proj_physical()
        expected = np.array([1, 0, 0, 1], dtype=np.float64) / np.sqrt(2)
        npt.assert_almost_equal(actual.vec, expected, decimal=15)
        assert actual.is_physical(actual.eps_proj_physical) == True

        # [1, 0, 0, 1] -> z0
        vec = np.array([1, 0, 0, 1], dtype=np.float64)
        state = State(c_sys, vec, is_physicality_required=False)
        actual = state.calc_proj_physical()
        expected = np.array([1, 0, 0, 1], dtype=np.float64) / np.sqrt(2)
        npt.assert_almost_equal(actual.vec, expected, decimal=4)
        assert actual.is_physical(actual.eps_proj_physical) == True

        # [1, 0, 0, -1] -> z1
        vec = np.array([1, 0, 0, -1], dtype=np.float64)
        state = State(c_sys, vec, is_physicality_required=False)
        actual = state.calc_proj_physical()
        expected = np.array([1, 0, 0, -1], dtype=np.float64) / np.sqrt(2)
        npt.assert_almost_equal(actual.vec, expected, decimal=4)
        assert actual.is_physical(actual.eps_proj_physical) == True

        # [1/sqrt(2), 1/sqrt(6), 1/sqrt(6), 1/sqrt(6)] -> [1/sqrt(2), 1/sqrt(6), 1/sqrt(6), 1/sqrt(6)]
        vec = np.array(
            [1 / np.sqrt(2), 1 / np.sqrt(6), 1 / np.sqrt(6), 1 / np.sqrt(6)],
            dtype=np.float64,
        )
        state = State(c_sys, vec, is_physicality_required=False)
        actual = state.calc_proj_physical()
        expected = np.array(
            [1 / np.sqrt(2), 1 / np.sqrt(6), 1 / np.sqrt(6), 1 / np.sqrt(6)],
            dtype=np.float64,
        )
        npt.assert_almost_equal(actual.vec, expected, decimal=4)
        assert actual.is_physical(actual.eps_proj_physical) == True

        # [2/sqrt(2), 2/sqrt(6), 2/sqrt(6), 2/sqrt(6)] -> [1/sqrt(2), 1/sqrt(6), 1/sqrt(6), 1/sqrt(6)]
        vec = np.array(
            [2 / np.sqrt(2), 2 / np.sqrt(6), 2 / np.sqrt(6), 2 / np.sqrt(6)],
            dtype=np.float64,
        )
        state = State(c_sys, vec, is_physicality_required=False)
        actual = state.calc_proj_physical()
        expected = np.array(
            [1 / np.sqrt(2), 1 / np.sqrt(6), 1 / np.sqrt(6), 1 / np.sqrt(6)],
            dtype=np.float64,
        )
        npt.assert_almost_equal(actual.vec, expected, decimal=4)
        assert actual.is_physical(actual.eps_proj_physical) == True

        # [1, 0, 0, 2] -> z0
        vec = np.array([1, 0, 0, 2], dtype=np.float64)
        state = State(c_sys, vec, is_physicality_required=False)
        actual = state.calc_proj_physical()
        expected = np.array([1, 0, 0, 1], dtype=np.float64) / np.sqrt(2)
        npt.assert_almost_equal(actual.vec, expected, decimal=4)
        assert actual.is_physical(actual.eps_proj_physical) == True

        # [1, 2, 3, 4] -> [1/sqrt(2), 2/sqrt(2*29), 3/sqrt(2*29), 4/sqrt(2*29)]
        # 29 = 2^2 + 3^2 + 4^2
        vec = np.array([1, 2, 3, 4], dtype=np.float64)
        state = State(c_sys, vec, is_physicality_required=False)
        actual, history = state.calc_proj_physical(is_iteration_history=True)
        expected = (
            np.array(
                [1, 2 / np.sqrt(29), 3 / np.sqrt(29), 4 / np.sqrt(29)],
                dtype=np.float64,
            )
            / np.sqrt(2)
        )
        npt.assert_almost_equal(actual.vec, expected, decimal=4)
        assert actual.is_physical(actual.eps_proj_physical) == True
        assert len(history["p"]) == 28
        assert len(history["q"]) == 28
        assert len(history["x"]) == 28
        assert len(history["y"]) == 28
        assert len(history["error_value"]) == 27
        assert history["y"][0] == None
        assert history["error_value"][0] == None

        # check max_iteration
        max_iteration = 10
        vec = np.array([1, 0, 0, 1], dtype=np.float64)
        state = State(c_sys, vec, is_physicality_required=False)
        actual, history = state.calc_proj_physical(
            max_iteration=max_iteration, is_iteration_history=True
        )
        assert len(history["x"]) == max_iteration + 1

    def test_calc_proj_physical_with_var(self):
        e_sys = ElementalSystem(0, matrix_basis.get_normalized_pauli_basis())
        c_sys = CompositeSystem([e_sys])

        # z0 -> z0
        z0 = get_z0_1q(c_sys)
        actual = z0.calc_proj_physical_with_var(z0.to_var())
        expected = np.array([0, 0, 1], dtype=np.float64) / np.sqrt(2)
        npt.assert_almost_equal(actual, expected, decimal=15)
        # assert actual.is_physical(actual.eps_proj_physical) == True

        # [1, 0, 0, 1] -> z0
        vec = np.array([1, 0, 0, 1], dtype=np.float64)
        state = State(c_sys, vec, is_physicality_required=False)
        actual = z0.calc_proj_physical_with_var(state.to_var())
        expected = np.array([0, 0, 1], dtype=np.float64) / np.sqrt(2)
        npt.assert_almost_equal(actual, expected, decimal=4)
        # assert actual.is_physical(actual.eps_proj_physical) == True

        # check max_iteration
        max_iteration = 10
        vec = np.array([1, 0, 0, 1], dtype=np.float64)
        state = State(c_sys, vec, is_physicality_required=False)
        actual, history = state.calc_proj_physical_with_var(
            state.to_var(), max_iteration=max_iteration, is_iteration_history=True
        )
        assert len(history["x"]) == max_iteration + 1

    def test_calc_stopping_criterion_birgin_raydan_vectors(self):
        e_sys = ElementalSystem(0, matrix_basis.get_normalized_pauli_basis())
        c_sys = CompositeSystem([e_sys])
        state = get_z0_1q(c_sys)

        p_prev = np.array([1, 2, 3, 4], dtype=np.float64)
        p_next = np.array([5, 6, 7, 8], dtype=np.float64)
        q_prev = np.array([11, 12, 13, 14], dtype=np.float64)
        q_next = np.array([15, 16, 17, 18], dtype=np.float64)
        x_prev = np.array([21, 22, 23, 24], dtype=np.float64)
        x_next = np.array([25, 26, 27, 28], dtype=np.float64)
        y_prev = np.array([31, 32, 33, 34], dtype=np.float64)
        y_next = np.array([35, 36, 37, 38], dtype=np.float64)

        value = state._calc_stopping_criterion_birgin_raydan_vectors(
            p_prev, p_next, q_prev, q_next, x_prev, x_next, y_prev, y_next
        )

        assert value == np.float64(-352)

    def test_is_satisfied_stopping_criterion_birgin_raydan_vectors(self):
        e_sys = ElementalSystem(0, matrix_basis.get_normalized_pauli_basis())
        c_sys = CompositeSystem([e_sys])
        state = get_z0_1q(c_sys)

        # case: True
        p_prev = np.array([1, 2, 3, 4], dtype=np.float64) * 10 ** (-7)
        p_next = np.array([5, 6, 7, 8], dtype=np.float64) * 10 ** (-7)
        q_prev = np.array([11, 12, 13, 14], dtype=np.float64) * 10 ** (-7)
        q_next = np.array([15, 16, 17, 18], dtype=np.float64) * 10 ** (-7)
        x_prev = np.array([21, 22, 23, 24], dtype=np.float64) * 10 ** (-7)
        x_next = np.array([25, 26, 27, 28], dtype=np.float64) * 10 ** (-7)
        y_prev = np.array([31, 32, 33, 34], dtype=np.float64) * 10 ** (-7)
        y_next = np.array([35, 36, 37, 38], dtype=np.float64) * 10 ** (-7)
        eps_proj_physical = 10 ** (-4)

        (
            is_stopping,
            error_value,
        ) = state._is_satisfied_stopping_criterion_birgin_raydan_vectors(
            p_prev,
            p_next,
            q_prev,
            q_next,
            x_prev,
            x_next,
            y_prev,
            y_next,
            eps_proj_physical,
        )

        assert is_stopping == True

        # case: False
        p_prev = np.array([1, 2, 3, 4], dtype=np.float64)
        p_next = np.array([5, 6, 7, 8], dtype=np.float64)
        q_prev = np.array([11, 12, 13, 14], dtype=np.float64)
        q_next = np.array([15, 16, 17, 18], dtype=np.float64)
        x_prev = np.array([25, 26, 27, 28], dtype=np.float64)
        x_next = np.array([21, 22, 23, 24], dtype=np.float64)
        y_prev = np.array([35, 36, 37, 38], dtype=np.float64)
        y_next = np.array([31, 32, 33, 34], dtype=np.float64)

        (
            is_stopping,
            error_value,
        ) = state._is_satisfied_stopping_criterion_birgin_raydan_vectors(
            p_prev,
            p_next,
            q_prev,
            q_next,
            x_prev,
            x_next,
            y_prev,
            y_next,
            eps_proj_physical,
        )

        assert is_stopping == False

    def test_is_satisfied_stopping_criterion_birgin_raydan_qoperations(self):
        e_sys = ElementalSystem(0, matrix_basis.get_normalized_pauli_basis())
        c_sys = CompositeSystem([e_sys])
        state = get_z0_1q(c_sys)

        # case: True
        p_prev = State(
            c_sys,
            np.array([1, 2, 3, 4], dtype=np.float64) * 10 ** (-7),
            is_physicality_required=False,
        )
        p_next = State(
            c_sys,
            np.array([5, 6, 7, 8], dtype=np.float64) * 10 ** (-7),
            is_physicality_required=False,
        )
        q_prev = State(
            c_sys,
            np.array([11, 12, 13, 14], dtype=np.float64) * 10 ** (-7),
            is_physicality_required=False,
        )
        q_next = State(
            c_sys,
            np.array([15, 16, 17, 18], dtype=np.float64) * 10 ** (-7),
            is_physicality_required=False,
        )
        x_prev = State(
            c_sys,
            np.array([21, 22, 23, 24], dtype=np.float64) * 10 ** (-7),
            is_physicality_required=False,
        )
        x_next = State(
            c_sys,
            np.array([25, 26, 27, 28], dtype=np.float64) * 10 ** (-7),
            is_physicality_required=False,
        )
        y_prev = State(
            c_sys,
            np.array([31, 32, 33, 34], dtype=np.float64) * 10 ** (-7),
            is_physicality_required=False,
        )
        y_next = State(
            c_sys,
            np.array([35, 36, 37, 38], dtype=np.float64) * 10 ** (-7),
            is_physicality_required=False,
        )
        eps_proj_physical = 10 ** (-4)

        (
            is_stopping,
            error_value,
        ) = state._is_satisfied_stopping_criterion_birgin_raydan_qoperations(
            p_prev,
            p_next,
            q_prev,
            q_next,
            x_prev,
            x_next,
            y_prev,
            y_next,
            eps_proj_physical,
        )

        assert is_stopping == True

        # case: False
        p_prev = State(
            c_sys,
            np.array([1, 2, 3, 4], dtype=np.float64),
            is_physicality_required=False,
        )
        p_next = State(
            c_sys,
            np.array([5, 6, 7, 8], dtype=np.float64),
            is_physicality_required=False,
        )
        q_prev = State(
            c_sys,
            np.array([11, 12, 13, 14], dtype=np.float64),
            is_physicality_required=False,
        )
        q_next = State(
            c_sys,
            np.array([15, 16, 17, 18], dtype=np.float64),
            is_physicality_required=False,
        )
        x_prev = State(
            c_sys,
            np.array([25, 26, 27, 28], dtype=np.float64),
            is_physicality_required=False,
        )
        x_next = State(
            c_sys,
            np.array([21, 22, 23, 24], dtype=np.float64),
            is_physicality_required=False,
        )
        y_prev = State(
            c_sys,
            np.array([35, 36, 37, 38], dtype=np.float64),
            is_physicality_required=False,
        )
        y_next = State(
            c_sys,
            np.array([31, 32, 33, 34], dtype=np.float64),
            is_physicality_required=False,
        )
        eps_proj_physical = 10 ** (-15)

        (
            is_stopping,
            error_value,
        ) = state._is_satisfied_stopping_criterion_birgin_raydan_qoperations(
            p_prev,
            p_next,
            q_prev,
            q_next,
            x_prev,
            x_next,
            y_prev,
            y_next,
            eps_proj_physical,
        )

        assert is_stopping == False

    def test_func_calc_proj_physical(self):
        e_sys = ElementalSystem(0, matrix_basis.get_normalized_pauli_basis())
        c_sys = CompositeSystem([e_sys])
        state = get_z0_1q(c_sys)
        func = state.func_calc_proj_physical(False)

        # z0 -> z0
        var = np.array([1, 0, 0, 1], dtype=np.float64) / np.sqrt(2)
        actual = func(var)
        expected = np.array([1, 0, 0, 1], dtype=np.float64) / np.sqrt(2)
        npt.assert_almost_equal(actual, expected, decimal=15)
        actual_qobj = State(c_sys, actual)
        assert actual_qobj.is_physical() == True

        # [1, 0, 0, 1] -> z0
        var = np.array([1, 0, 0, 1], dtype=np.float64)
        actual = func(var)
        expected = np.array([1, 0, 0, 1], dtype=np.float64) / np.sqrt(2)
        npt.assert_almost_equal(actual, expected, decimal=7)
        actual_qobj = State(c_sys, actual)
        assert actual_qobj.is_physical() == True

        # [1/sqrt(2), 1/sqrt(6), 1/sqrt(6), 1/sqrt(6)] -> [1/sqrt(2), 1/sqrt(6), 1/sqrt(6), 1/sqrt(6)]
        var = np.array(
            [1 / np.sqrt(2), 1 / np.sqrt(6), 1 / np.sqrt(6), 1 / np.sqrt(6)],
            dtype=np.float64,
        )
        actual = func(var)
        expected = np.array(
            [1 / np.sqrt(2), 1 / np.sqrt(6), 1 / np.sqrt(6), 1 / np.sqrt(6)],
            dtype=np.float64,
        )
        npt.assert_almost_equal(actual, expected, decimal=15)
        actual_qobj = State(c_sys, actual)
        assert actual_qobj.is_physical() == True


def test_to_density_matrix_from_var():
    e_sys = ElementalSystem(0, matrix_basis.get_normalized_pauli_basis())
    c_sys = CompositeSystem([e_sys])
    state = get_z0_1q(c_sys)

    # Case 1: on_para_eq_constraint: default(True)
    actual = to_density_matrix_from_var(c_sys, state.to_var())
    expected = np.array([[1, 0], [0, 0]], dtype=np.float64)
    npt.assert_almost_equal(actual, expected, decimal=15)

    # Case 2: on_para_eq_constraint=True
    actual = to_density_matrix_from_var(
        c_sys, state.to_var(), on_para_eq_constraint=True
    )
    expected = np.array([[1, 0], [0, 0]], dtype=np.float64)
    npt.assert_almost_equal(actual, expected, decimal=15)

    # Case 3: on_para_eq_constraint=False
    actual = to_density_matrix_from_var(
        c_sys, state.to_stacked_vector(), on_para_eq_constraint=False
    )
    expected = np.array([[1, 0], [0, 0]], dtype=np.float64)
    npt.assert_almost_equal(actual, expected, decimal=15)


def test_to_var_from_density_matrix():
    e_sys = ElementalSystem(0, matrix_basis.get_normalized_pauli_basis())
    c_sys = CompositeSystem([e_sys])
    state = get_z0_1q(c_sys)

    # Case 1: on_para_eq_constraint: default(True)
    actual = to_var_from_density_matrix(c_sys, state.to_density_matrix())
    expected = np.array([0, 0, 1], dtype=np.float64) / np.sqrt(2)
    npt.assert_almost_equal(actual, expected, decimal=15)

    # Case 2: on_para_eq_constraint=True
    actual = to_var_from_density_matrix(
        c_sys, state.to_density_matrix(), on_para_eq_constraint=True
    )
    expected = np.array([0, 0, 1], dtype=np.float64) / np.sqrt(2)
    npt.assert_almost_equal(actual, expected, decimal=15)

    # Case 3: on_para_eq_constraint=False
    actual = to_var_from_density_matrix(
        c_sys, state.to_density_matrix(), on_para_eq_constraint=False
    )
    expected = np.array([1, 0, 0, 1], dtype=np.float64) / np.sqrt(2)
    npt.assert_almost_equal(actual, expected, decimal=15)


def test_to_vec_from_density_matrix_with_sparsity():
    # Case 1:
    # Arrange
    e_sys = ElementalSystem(0, matrix_basis.get_normalized_pauli_basis())
    c_sys = CompositeSystem([e_sys])
    state = get_x0_1q(c_sys)
    density_matrix = state.to_density_matrix()
    # Act
    actual = to_vec_from_density_matrix_with_sparsity(c_sys, density_matrix)
    # Assert
    expected = state.vec
    npt.assert_almost_equal(actual, expected, decimal=15)

    # Case 2:
    # Arrange
    e_sys = ElementalSystem(0, matrix_basis.get_normalized_pauli_basis())
    c_sys = CompositeSystem([e_sys])
    state = get_y0_1q(c_sys)
    density_matrix = state.to_density_matrix()
    # Act
    actual = to_vec_from_density_matrix_with_sparsity(c_sys, density_matrix)
    # Assert
    expected = state.vec
    npt.assert_almost_equal(actual, expected, decimal=15)

    # Case 3:
    # Arrange
    e_sys = ElementalSystem(0, matrix_basis.get_normalized_pauli_basis())
    c_sys = CompositeSystem([e_sys])
    state = get_z0_1q(c_sys)
    density_matrix = state.to_density_matrix()
    # Act
    actual = to_vec_from_density_matrix_with_sparsity(c_sys, density_matrix)
    # Assert
    expected = state.vec
    npt.assert_almost_equal(actual, expected, decimal=15)


def test_convert_var_index_to_state_index():
    # default
    actual = convert_var_index_to_state_index(1)
    assert actual == 2

    # on_para_eq_constraint=True
    actual = convert_var_index_to_state_index(1, on_para_eq_constraint=True)
    assert actual == 2

    # on_para_eq_constraint=False
    actual = convert_var_index_to_state_index(1, on_para_eq_constraint=False)
    assert actual == 1


def test_convert_state_index_to_var_index():
    # default
    actual = convert_state_index_to_var_index(1)
    assert actual == 0

    # on_para_eq_constraint=True
    actual = convert_state_index_to_var_index(1, on_para_eq_constraint=True)
    assert actual == 0

    # on_para_eq_constraint=False
    actual = convert_state_index_to_var_index(1, on_para_eq_constraint=False)
    assert actual == 1


def test_convert_var_to_state():
    # Arrange
    e_sys = ElementalSystem(0, matrix_basis.get_normalized_pauli_basis())
    c_sys = CompositeSystem([e_sys])

    # Case 1: default
    # Act
    actual = convert_var_to_state(
        c_sys, np.array([1, 2, 3], dtype=np.float64), is_physicality_required=False
    )
    # Assert
    expected = np.array([1 / np.sqrt(2), 1, 2, 3], dtype=np.float64)
    assert actual.is_physicality_required == False
    assert np.all(actual.vec == expected)

    # Case 2: on_para_eq_constraint=True
    # Act
    actual = convert_var_to_state(
        c_sys,
        np.array([1, 2, 3], dtype=np.float64),
        on_para_eq_constraint=True,
        is_physicality_required=False,
    )
    # Assert
    expected = np.array([1 / np.sqrt(2), 1, 2, 3], dtype=np.float64)
    assert actual.is_physicality_required == False
    assert np.all(actual.vec == expected)

    # Case 3: on_para_eq_constraint=False
    # Act
    actual = convert_var_to_state(
        c_sys,
        np.array([1, 2, 3, 4], dtype=np.float64),
        on_para_eq_constraint=False,
        is_physicality_required=False,
    )
    # Assert
    expected = np.array([1, 2, 3, 4], dtype=np.float64)
    assert actual.is_physicality_required == False
    assert np.all(actual.vec == expected)


def test_convert_vec_to_var():
    # Arrange
    e_sys = ElementalSystem(0, matrix_basis.get_normalized_pauli_basis())
    c_sys = CompositeSystem([e_sys])

    # Case 1: default
    # Act
    actual = convert_vec_to_var(
        c_sys, np.array([1 / np.sqrt(2), 1, 2, 3], dtype=np.float64)
    )
    # Assert
    expected = np.array([1, 2, 3], dtype=np.float64)
    assert np.all(actual == expected)

    # Case 2: on_para_eq_constraint=True
    # Act
    actual = convert_vec_to_var(
        c_sys,
        np.array([1 / np.sqrt(2), 1, 2, 3], dtype=np.float64),
        on_para_eq_constraint=True,
    )
    # Assert
    expected = np.array([1, 2, 3], dtype=np.float64)
    assert np.all(actual == expected)

    # Case 3: on_para_eq_constraint=False
    # Act
    actual = convert_vec_to_var(
        c_sys, np.array([1, 2, 3, 4], dtype=np.float64), on_para_eq_constraint=False
    )
    # Assert
    expected = np.array([1, 2, 3, 4], dtype=np.float64)
    assert np.all(actual == expected)


def test_calc_gradient_from_state():
    e_sys = ElementalSystem(0, matrix_basis.get_normalized_pauli_basis())
    c_sys = CompositeSystem([e_sys])

    # default
    actual = calc_gradient_from_state(
        c_sys, np.array([1, 2, 3, 4], dtype=np.float64), 1
    )
    expected = np.array([0, 0, 1, 0], dtype=np.float64)
    assert actual.is_physicality_required == False
    assert np.all(actual.vec == expected)

    # on_para_eq_constraint=True
    actual = calc_gradient_from_state(
        c_sys, np.array([1, 2, 3, 4], dtype=np.float64), 1, on_para_eq_constraint=True
    )
    expected = np.array([0, 0, 1, 0], dtype=np.float64)
    assert actual.is_physicality_required == False
    assert np.all(actual.vec == expected)

    # on_para_eq_constraint=False
    actual = calc_gradient_from_state(
        c_sys, np.array([1, 2, 3, 4], dtype=np.float64), 1, on_para_eq_constraint=False
    )
    expected = np.array([0, 1, 0, 0], dtype=np.float64)
    assert actual.is_physicality_required == False
    assert np.all(actual.vec == expected)


def test_get_x0_1q():
    e_sys = ElementalSystem(0, matrix_basis.get_normalized_pauli_basis())
    c_sys = CompositeSystem([e_sys])
    state = get_x0_1q(c_sys)
    actual = state.to_density_matrix()
    expected = np.array([[0.5, 0.5], [0.5, 0.5]], dtype=np.complex128)
    npt.assert_almost_equal(actual, expected, decimal=15)

    # Test that not 1qubit CompositeSystem
    e_sys0 = ElementalSystem(0, matrix_basis.get_normalized_pauli_basis())
    e_sys1 = ElementalSystem(1, matrix_basis.get_normalized_pauli_basis())
    c_sys = CompositeSystem([e_sys0, e_sys1])
    with pytest.raises(ValueError):
        get_x0_1q(c_sys)


def test_get_x1_1q():
    e_sys = ElementalSystem(0, matrix_basis.get_normalized_pauli_basis())
    c_sys = CompositeSystem([e_sys])
    state = get_x1_1q(c_sys)
    actual = state.to_density_matrix()
    expected = np.array([[0.5, -0.5], [-0.5, 0.5]], dtype=np.complex128)
    npt.assert_almost_equal(actual, expected, decimal=15)

    # Test that not 1qubit CompositeSystem
    e_sys0 = ElementalSystem(0, matrix_basis.get_normalized_pauli_basis())
    e_sys1 = ElementalSystem(1, matrix_basis.get_normalized_pauli_basis())
    c_sys = CompositeSystem([e_sys0, e_sys1])
    with pytest.raises(ValueError):
        get_x1_1q(c_sys)


def test_get_y0_1q():
    e_sys = ElementalSystem(0, matrix_basis.get_normalized_pauli_basis())
    c_sys = CompositeSystem([e_sys])
    state = get_y0_1q(c_sys)
    actual = state.to_density_matrix()
    expected = np.array([[0.5, -0.5j], [0.5j, 0.5]], dtype=np.complex128)
    npt.assert_almost_equal(actual, expected, decimal=15)

    # Test that not 1qubit CompositeSystem
    e_sys0 = ElementalSystem(0, matrix_basis.get_normalized_pauli_basis())
    e_sys1 = ElementalSystem(1, matrix_basis.get_normalized_pauli_basis())
    c_sys = CompositeSystem([e_sys0, e_sys1])
    with pytest.raises(ValueError):
        get_y0_1q(c_sys)


def test_get_y1_1q():
    e_sys = ElementalSystem(0, matrix_basis.get_normalized_pauli_basis())
    c_sys = CompositeSystem([e_sys])
    state = get_y1_1q(c_sys)
    actual = state.to_density_matrix()
    expected = np.array([[0.5, 0.5j], [-0.5j, 0.5]], dtype=np.complex128)
    npt.assert_almost_equal(actual, expected, decimal=15)

    # Test that not 1qubit CompositeSystem
    e_sys0 = ElementalSystem(0, matrix_basis.get_normalized_pauli_basis())
    e_sys1 = ElementalSystem(1, matrix_basis.get_normalized_pauli_basis())
    c_sys = CompositeSystem([e_sys0, e_sys1])
    with pytest.raises(ValueError):
        get_y1_1q(c_sys)


def test_get_z0_1q():
    e_sys = ElementalSystem(0, matrix_basis.get_normalized_pauli_basis())
    c_sys = CompositeSystem([e_sys])
    state = get_z0_1q(c_sys)
    actual = state.to_density_matrix()
    expected = np.array([[1, 0], [0, 0]], dtype=np.complex128)
    npt.assert_almost_equal(actual, expected, decimal=15)

    # Test that not 1qubit CompositeSystem
    e_sys0 = ElementalSystem(0, matrix_basis.get_normalized_pauli_basis())
    e_sys1 = ElementalSystem(1, matrix_basis.get_normalized_pauli_basis())
    c_sys = CompositeSystem([e_sys0, e_sys1])
    with pytest.raises(ValueError):
        get_z0_1q(c_sys)


def test_get_z1_1q():
    e_sys = ElementalSystem(0, matrix_basis.get_normalized_pauli_basis())
    c_sys = CompositeSystem([e_sys])
    state = get_z1_1q(c_sys)
    actual = state.to_density_matrix()
    expected = np.array([[0, 0], [0, 1]], dtype=np.complex128)
    npt.assert_almost_equal(actual, expected, decimal=15)

    # Test that not 1qubit CompositeSystem
    e_sys0 = ElementalSystem(0, matrix_basis.get_normalized_pauli_basis())
    e_sys1 = ElementalSystem(1, matrix_basis.get_normalized_pauli_basis())
    c_sys = CompositeSystem([e_sys0, e_sys1])
    with pytest.raises(ValueError):
        get_z1_1q(c_sys)


def test_get_bell_2q():
    expected = (
        np.array(
            [[1, 0, 0, 1], [0, 0, 0, 0], [0, 0, 0, 0], [1, 0, 0, 1]],
            dtype=np.complex128,
        )
        / 2
    )

    # test for Pauli basis
    e_sys0 = ElementalSystem(0, matrix_basis.get_normalized_pauli_basis())
    e_sys1 = ElementalSystem(1, matrix_basis.get_normalized_pauli_basis())
    c_sys = CompositeSystem([e_sys0, e_sys1])
    state = get_bell_2q(c_sys)
    actual = state.to_density_matrix()
    npt.assert_almost_equal(actual, expected, decimal=15)

    # test for comp basis
    e_sys2 = ElementalSystem(2, matrix_basis.get_comp_basis())
    e_sys3 = ElementalSystem(3, matrix_basis.get_comp_basis())
    c_sys = CompositeSystem([e_sys2, e_sys3])
    state = get_bell_2q(c_sys)
    actual = state.to_density_matrix()
    npt.assert_almost_equal(actual, expected, decimal=15)

    # Test that not 2qubit CompositeSystem
    e_sys2 = ElementalSystem(2, matrix_basis.get_normalized_pauli_basis())
    c_sys = CompositeSystem([e_sys2])
    with pytest.raises(ValueError):
        get_bell_2q(c_sys)
