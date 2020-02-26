import numpy as np
import numpy.testing as npt

from quara.objects.composite_system import CompositeSystem
from quara.objects.elemental_system import ElementalSystem
from quara.objects import matrix_basis
from quara.objects.matrix_basis import MatrixBasis
from quara.objects.state import (
    State,
    get_X0_1q_with_normalized_pauli_basis,
    get_X1_1q_with_normalized_pauli_basis,
    get_Y0_1q_with_normalized_pauli_basis,
    get_Y1_1q_with_normalized_pauli_basis,
    get_Z0_1q_with_normalized_pauli_basis,
    get_Z1_1q_with_normalized_pauli_basis
)
import numpy.testing as npt
import pytest


def test_init_error():
    e_sys = ElementalSystem("q1", matrix_basis.get_comp_basis())
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


def test_comp_basis():
    e_sys = ElementalSystem("q1", matrix_basis.get_comp_basis())
    c_sys = CompositeSystem([e_sys])

    # test for vec[1, 0, 0, 0]
    state = State(c_sys, np.array([1, 0, 0, 0], dtype=np.float64))
    assert state.dim == 2
    assert np.all(
        state.get_density_matrix() == np.array([[1, 0], [0, 0]], dtype=np.complex128)
    )
    assert state.is_trace_one() == True
    assert state.is_hermitian() == True
    assert state.is_positive_semidefinite() == True
    assert np.all(state.get_eigen_values() == np.array([1, 0], dtype=np.complex128))

    # test for vec[0, 1, 0, 0]
    state = State(c_sys, np.array([0, 1, 0, 0], dtype=np.float64))
    assert state.dim == 2
    assert np.all(
        state.get_density_matrix() == np.array([[0, 1], [0, 0]], dtype=np.complex128)
    )
    assert state.is_trace_one() == False
    assert state.is_hermitian() == False
    assert state.is_positive_semidefinite() == False
    assert np.all(state.get_eigen_values() == np.array([0, 0], dtype=np.complex128))

    # test for vec[0, 0, 1, 0]
    state = State(c_sys, np.array([0, 0, 1, 0], dtype=np.float64))
    assert state.dim == 2
    assert np.all(
        state.get_density_matrix() == np.array([[0, 0], [1, 0]], dtype=np.complex128)
    )
    assert state.is_trace_one() == False
    assert state.is_hermitian() == False
    assert state.is_positive_semidefinite() == False
    assert np.all(state.get_eigen_values() == np.array([0, 0], dtype=np.complex128))

    # test for vec[0, 0, 0, 1]
    state = State(c_sys, np.array([0, 0, 0, 1], dtype=np.float64))
    assert state.dim == 2
    assert np.all(
        state.get_density_matrix() == np.array([[0, 0], [0, 1]], dtype=np.complex128)
    )
    assert state.is_trace_one() == True
    assert state.is_hermitian() == True
    assert state.is_positive_semidefinite() == True
    assert np.all(state.get_eigen_values() == np.array([0, 1], dtype=np.complex128))


def test_pauli_basis():
    e_sys = ElementalSystem("q1", matrix_basis.get_pauli_basis())
    c_sys = CompositeSystem([e_sys])

    # test for vec[1, 0, 0, 0]
    state = State(c_sys, np.array([1, 0, 0, 0], dtype=np.float64))
    assert state.dim == 2
    assert np.all(
        state.get_density_matrix() == np.array([[1, 0], [0, 1]], dtype=np.complex128)
    )
    assert state.is_trace_one() == False
    assert state.is_hermitian() == True
    assert state.is_positive_semidefinite() == True
    npt.assert_almost_equal(
        state.get_eigen_values(), np.array([1, 1], dtype=np.complex128), decimal=15,
    )

    # test for vec [0, 1, 0, 0]
    state = State(c_sys, np.array([0, 1, 0, 0], dtype=np.float64))
    assert state.dim == 2
    assert np.all(
        state.get_density_matrix() == np.array([[0, 1], [1, 0]], dtype=np.complex128)
    )
    assert state.is_trace_one() == False
    assert state.is_hermitian() == True
    assert state.is_positive_semidefinite() == False
    npt.assert_almost_equal(
        state.get_eigen_values(), np.array([1, -1], dtype=np.complex128), decimal=15,
    )

    # test for vec [0, 0, 1, 0]
    state = State(c_sys, np.array([0, 0, 1, 0], dtype=np.float64))
    assert state.dim == 2
    assert np.all(
        state.get_density_matrix() == np.array([[0, -1j], [1j, 0]], dtype=np.complex128)
    )
    assert state.is_trace_one() == False
    assert state.is_hermitian() == True
    assert state.is_positive_semidefinite() == False
    npt.assert_almost_equal(
        state.get_eigen_values(), np.array([1, -1], dtype=np.complex128), decimal=15,
    )

    # test for vec [0, 0, 0, 1]
    state = State(c_sys, np.array([0, 0, 0, 1], dtype=np.float64))
    assert state.dim == 2
    assert np.all(
        state.get_density_matrix() == np.array([[1, 0], [0, -1]], dtype=np.complex128)
    )
    assert state.is_trace_one() == False
    assert state.is_hermitian() == True
    assert state.is_positive_semidefinite() == False
    npt.assert_almost_equal(
        state.get_eigen_values(), np.array([1, -1], dtype=np.complex128), decimal=15,
    )


def test_normalized_pauli_basis():
    e_sys = ElementalSystem("q1", matrix_basis.get_normalized_pauli_basis())
    c_sys = CompositeSystem([e_sys])

    # test for vec[1, 0, 0, 0]
    state = State(c_sys, np.array([1, 0, 0, 0], dtype=np.float64))
    assert state.dim == 2
    assert np.all(
        state.get_density_matrix()
        == 1 / np.sqrt(2) * np.array([[1, 0], [0, 1]], dtype=np.complex128)
    )
    assert state.is_trace_one() == False
    assert state.is_hermitian() == True
    assert state.is_positive_semidefinite() == True
    npt.assert_almost_equal(
        state.get_eigen_values(),
        np.array([1 / np.sqrt(2), 1 / np.sqrt(2)], dtype=np.complex128),
        decimal=15,
    )

    # test for vec [0, 1, 0, 0]
    state = State(c_sys, np.array([0, 1, 0, 0], dtype=np.float64))
    assert state.dim == 2
    assert np.all(
        state.get_density_matrix()
        == 1 / np.sqrt(2) * np.array([[0, 1], [1, 0]], dtype=np.complex128)
    )
    assert state.is_trace_one() == False
    assert state.is_hermitian() == True
    assert state.is_positive_semidefinite() == False
    npt.assert_almost_equal(
        state.get_eigen_values(),
        np.array([1 / np.sqrt(2), -1 / np.sqrt(2)], dtype=np.complex128),
        decimal=15,
    )

    # test for vec [0, 0, 1, 0]
    state = State(c_sys, np.array([0, 0, 1, 0], dtype=np.float64))
    assert state.dim == 2
    assert np.all(
        state.get_density_matrix()
        == 1 / np.sqrt(2) * np.array([[0, -1j], [1j, 0]], dtype=np.complex128)
    )
    assert state.is_trace_one() == False
    assert state.is_hermitian() == True
    assert state.is_positive_semidefinite() == False
    npt.assert_almost_equal(
        state.get_eigen_values(),
        np.array([1 / np.sqrt(2), -1 / np.sqrt(2)], dtype=np.complex128),
        decimal=15,
    )

    # test for vec [0, 0, 0, 1]
    state = State(c_sys, np.array([0, 0, 0, 1], dtype=np.float64))
    assert state.dim == 2
    assert np.all(
        state.get_density_matrix()
        == 1 / np.sqrt(2) * np.array([[1, 0], [0, -1]], dtype=np.complex128)
    )
    assert state.is_trace_one() == False
    assert state.is_hermitian() == True
    assert state.is_positive_semidefinite() == False
    npt.assert_almost_equal(
        state.get_eigen_values(),
        np.array([1 / np.sqrt(2), -1 / np.sqrt(2)], dtype=np.complex128),
        decimal=15,
    )


def test_convert_basis_form_comp_to_pauli():
    comp_basis = matrix_basis.get_comp_basis()
    pauli_basis = matrix_basis.get_normalized_pauli_basis()

    # CompositeSystem of comp basis
    e_sys1 = ElementalSystem("q1", matrix_basis.get_comp_basis())
    c_sys1 = CompositeSystem([e_sys1])

    # converts [1, 0, 0, 0] with comp basis to Pauli basis
    state = State(c_sys1, np.array([1, 0, 0, 0], dtype=np.float64))
    actual = state.convert_basis(pauli_basis)
    expected = 1 / np.sqrt(2) * np.array([1, 0, 0, 1], dtype=np.complex128)
    assert np.all(actual == expected)

    # converts [0, 1, 0, 0] with comp basis to Pauli basis
    state = State(c_sys1, np.array([0, 1, 0, 0], dtype=np.float64))
    actual = state.convert_basis(pauli_basis)
    expected = 1 / np.sqrt(2) * np.array([0, 1, 1j, 0], dtype=np.complex128)
    assert np.all(actual == expected)

    # converts [0, 0, 1, 0] with comp basis to Pauli basis
    state = State(c_sys1, np.array([0, 0, 1, 0], dtype=np.float64))
    actual = state.convert_basis(pauli_basis)
    expected = 1 / np.sqrt(2) * np.array([0, 1, -1j, 0], dtype=np.complex128)
    assert np.all(actual == expected)

    # converts [0, 0, 0, 1] with comp basis to Pauli basis
    state = State(c_sys1, np.array([0, 0, 0, 1], dtype=np.float64))
    actual = state.convert_basis(pauli_basis)
    expected = 1 / np.sqrt(2) * np.array([1, 0, 0, -1], dtype=np.complex128)
    assert np.all(actual == expected)


def test_convert_basis_form_pauli_to_comp():
    comp_basis = matrix_basis.get_comp_basis()
    pauli_basis = matrix_basis.get_normalized_pauli_basis()

    # CompositeSystem of Pauli basis
    e_sys2 = ElementalSystem("q2", matrix_basis.get_normalized_pauli_basis())
    c_sys2 = CompositeSystem([e_sys2])

    # converts [1, 0, 0, 0] with Pauli basis to comp basis
    state = State(c_sys2, np.array([1, 0, 0, 0], dtype=np.float64))
    actual = state.convert_basis(comp_basis)
    expected = 1 / np.sqrt(2) * np.array([1, 0, 0, 1], dtype=np.complex128)
    assert np.all(actual == expected)

    # converts [0, 1, 0, 0] with Pauli basis to comp basis
    state = State(c_sys2, np.array([0, 1, 0, 0], dtype=np.float64))
    actual = state.convert_basis(comp_basis)
    expected = 1 / np.sqrt(2) * np.array([0, 1, 1, 0], dtype=np.complex128)
    assert np.all(actual == expected)

    # converts [0, 0, 1, 0] with Pauli basis to comp basis
    state = State(c_sys2, np.array([0, 0, 1, 0], dtype=np.float64))
    actual = state.convert_basis(comp_basis)
    expected = 1 / np.sqrt(2) * np.array([0, -1j, 1j, 0], dtype=np.complex128)
    assert np.all(actual == expected)

    # converts [0, 0, 0, 1] with Pauli basis to comp basis
    state = State(c_sys2, np.array([0, 0, 0, 1], dtype=np.float64))
    actual = state.convert_basis(comp_basis)
    expected = 1 / np.sqrt(2) * np.array([1, 0, 0, -1], dtype=np.complex128)
    assert np.all(actual == expected)


def test_get_X0_1q_with_normalized_pauli_basis():
    e_sys = ElementalSystem("q0", matrix_basis.get_normalized_pauli_basis())
    c_sys = CompositeSystem([e_sys])
    state = get_X0_1q_with_normalized_pauli_basis(c_sys)
    actual = state.get_density_matrix()
    expected = np.array([[0.5, 0.5], [0.5, 0.5]], dtype=np.complex128)
    npt.assert_almost_equal(actual, expected, decimal=15)


def test_get_X1_1q_with_normalized_pauli_basis():
    e_sys = ElementalSystem("q0", matrix_basis.get_normalized_pauli_basis())
    c_sys = CompositeSystem([e_sys])
    state = get_X1_1q_with_normalized_pauli_basis(c_sys)
    actual = state.get_density_matrix()
    expected = np.array([[0.5, -0.5], [-0.5, 0.5]], dtype=np.complex128)
    npt.assert_almost_equal(actual, expected, decimal=15)


def test_get_Y0_1q_with_normalized_pauli_basis():
    e_sys = ElementalSystem("q0", matrix_basis.get_normalized_pauli_basis())
    c_sys = CompositeSystem([e_sys])
    state = get_Y0_1q_with_normalized_pauli_basis(c_sys)
    actual = state.get_density_matrix()
    expected = np.array([[0.5, -0.5j], [0.5j, 0.5]], dtype=np.complex128)
    npt.assert_almost_equal(actual, expected, decimal=15)


def test_get_Y1_1q_with_normalized_pauli_basis():
    e_sys = ElementalSystem("q0", matrix_basis.get_normalized_pauli_basis())
    c_sys = CompositeSystem([e_sys])
    state = get_Y1_1q_with_normalized_pauli_basis(c_sys)
    actual = state.get_density_matrix()
    expected = np.array([[0.5, 0.5j], [-0.5j, 0.5]], dtype=np.complex128)
    npt.assert_almost_equal(actual, expected, decimal=15)


def test_get_Z0_with_normalized_pauli_basis():
    e_sys = ElementalSystem("q0", matrix_basis.get_normalized_pauli_basis())
    c_sys = CompositeSystem([e_sys])
    state = get_Z0_1q_with_normalized_pauli_basis(c_sys)
    actual = state.get_density_matrix()
    expected = np.array([[1, 0], [0, 0]], dtype=np.complex128)
    npt.assert_almost_equal(actual, expected, decimal=15)

def test_get_Z1_with_normalized_pauli_basis():
    e_sys = ElementalSystem("q0", matrix_basis.get_normalized_pauli_basis())
    c_sys = CompositeSystem([e_sys])
    state = get_Z1_1q_with_normalized_pauli_basis(c_sys)
    actual = state.get_density_matrix()
    expected = np.array([[0, 0], [0, 1]], dtype=np.complex128)
    npt.assert_almost_equal(actual, expected, decimal=15)

# TODO implement test of convert_vec
