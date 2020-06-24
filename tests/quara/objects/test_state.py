import numpy as np
import numpy.testing as npt
import pytest

from quara.objects import matrix_basis
from quara.objects.composite_system import CompositeSystem
from quara.objects.elemental_system import ElementalSystem
from quara.objects.matrix_basis import MatrixBasis
from quara.objects.state import (
    State,
    convert_var_index_to_state_index,
    convert_state_index_to_var_index,
    convert_var_to_state,
    convert_state_to_var,
    calc_gradient_from_state,
    get_bell_2q,
    get_x0_1q,
    get_x1_1q,
    get_y0_1q,
    get_y1_1q,
    get_z0_1q,
    get_z1_1q,
)


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

        state = State(c_sys, np.array([1, 0, 0, 0], dtype=np.float64),)
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
        assert state.eps_proj_physical == 10 ** (-4)

    def test_zero_obj(self):
        e_sys = ElementalSystem(0, matrix_basis.get_normalized_pauli_basis())
        c_sys = CompositeSystem([e_sys])
        state = get_z0_1q(c_sys)
        zero = state.zero_obj()

        expected = np.zeros((4), dtype=np.float64)
        npt.assert_almost_equal(zero.vec, expected, decimal=15)
        assert zero.dim == state.dim
        assert zero.is_physicality_required == False
        assert zero.is_estimation_object == True
        assert zero.on_para_eq_constraint == state.on_para_eq_constraint
        assert zero.on_algo_eq_constraint == state.on_algo_eq_constraint
        assert zero.on_algo_ineq_constraint == state.on_algo_ineq_constraint
        assert zero.eps_proj_physical == state.eps_proj_physical

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

    def test_to_density_matrix(self):
        e_sys = ElementalSystem(0, matrix_basis.get_normalized_pauli_basis())
        c_sys = CompositeSystem([e_sys])
        state = get_z0_1q(c_sys)
        actual = state.to_density_matrix()
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

        # case: False
        e_sys = ElementalSystem(0, matrix_basis.get_comp_basis())
        c_sys = CompositeSystem([e_sys])
        state = State(
            c_sys,
            np.array([0, 1, 0, 0], dtype=np.float64),
            is_physicality_required=False,
        )
        assert state.is_trace_one() == False

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

    def test_is_positive_semidefinite(self):
        # case: True
        e_sys = ElementalSystem(0, matrix_basis.get_normalized_pauli_basis())
        c_sys = CompositeSystem([e_sys])
        state = get_z0_1q(c_sys)
        assert state.is_positive_semidefinite() == True

        # case: False
        e_sys = ElementalSystem(0, matrix_basis.get_comp_basis())
        c_sys = CompositeSystem([e_sys])
        state = State(
            c_sys,
            np.array([-1, 0, 0, 0], dtype=np.float64),
            is_physicality_required=False,
        )
        assert state.is_positive_semidefinite() == False

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
        assert np.all(state.calc_eigenvalues() == np.array([0, 1], dtype=np.complex128))

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
            state.calc_eigenvalues(), np.array([1, 1], dtype=np.complex128), decimal=15,
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


def test_convert_state_to_var():
    # Arrange
    e_sys = ElementalSystem(0, matrix_basis.get_normalized_pauli_basis())
    c_sys = CompositeSystem([e_sys])

    # Case 1: default
    # Act
    actual = convert_state_to_var(
        c_sys, np.array([1 / np.sqrt(2), 1, 2, 3], dtype=np.float64)
    )
    # Assert
    expected = np.array([1, 2, 3], dtype=np.float64)
    assert np.all(actual == expected)

    # Case 2: on_para_eq_constraint=True
    # Act
    actual = convert_state_to_var(
        c_sys,
        np.array([1 / np.sqrt(2), 1, 2, 3], dtype=np.float64),
        on_para_eq_constraint=True,
    )
    # Assert
    expected = np.array([1, 2, 3], dtype=np.float64)
    assert np.all(actual == expected)

    # Case 3: on_para_eq_constraint=False
    # Act
    actual = convert_state_to_var(
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
