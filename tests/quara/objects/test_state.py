import numpy as np
import numpy.testing as npt
import pytest

from quara.objects import matrix_basis
from quara.objects.composite_system import CompositeSystem
from quara.objects.elemental_system import ElementalSystem
from quara.objects.matrix_basis import MatrixBasis
from quara.objects.state import (
    State,
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

        # Test that "vec" cannot be updated
        with pytest.raises(AttributeError):
            state.dim = 2  # New dim

    def test_get_density_matrix(self):
        e_sys = ElementalSystem(0, matrix_basis.get_normalized_pauli_basis())
        c_sys = CompositeSystem([e_sys])
        state = get_z0_1q(c_sys)
        actual = state.get_density_matrix()
        expected = np.array([[1, 0], [0, 0]], dtype=np.float64)
        npt.assert_almost_equal(actual, expected, decimal=15)

    def test_is_trace_one(self):
        # case: True
        e_sys = ElementalSystem(0, matrix_basis.get_comp_basis())
        c_sys = CompositeSystem([e_sys])
        state = State(c_sys, np.array([1, 0, 0, 0], dtype=np.float64))
        assert state.is_trace_one() == True

        # case: False
        e_sys = ElementalSystem(0, matrix_basis.get_normalized_pauli_basis())
        c_sys = CompositeSystem([e_sys])
        state = get_z0_1q(c_sys)
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
        state = State(c_sys, np.array([0, 1, 0, 0], dtype=np.float64))
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
        state = State(c_sys, np.array([-1, 0, 0, 0], dtype=np.float64))
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
        state = State(c_sys, np.array([1, 0, 0, 0], dtype=np.float64))
        assert state.dim == 2
        assert np.all(
            state.get_density_matrix()
            == np.array([[1, 0], [0, 0]], dtype=np.complex128)
        )
        assert state.is_trace_one() == True
        assert state.is_hermitian() == True
        assert state.is_positive_semidefinite() == True
        assert np.all(state.calc_eigenvalues() == np.array([1, 0], dtype=np.complex128))

        # test for vec[0, 1, 0, 0]
        state = State(c_sys, np.array([0, 1, 0, 0], dtype=np.float64))
        assert state.dim == 2
        assert np.all(
            state.get_density_matrix()
            == np.array([[0, 1], [0, 0]], dtype=np.complex128)
        )
        assert state.is_trace_one() == False
        assert state.is_hermitian() == False
        assert state.is_positive_semidefinite() == False
        assert np.all(state.calc_eigenvalues() == np.array([0, 0], dtype=np.complex128))

        # test for vec[0, 0, 1, 0]
        state = State(c_sys, np.array([0, 0, 1, 0], dtype=np.float64))
        assert state.dim == 2
        assert np.all(
            state.get_density_matrix()
            == np.array([[0, 0], [1, 0]], dtype=np.complex128)
        )
        assert state.is_trace_one() == False
        assert state.is_hermitian() == False
        assert state.is_positive_semidefinite() == False
        assert np.all(state.calc_eigenvalues() == np.array([0, 0], dtype=np.complex128))

        # test for vec[0, 0, 0, 1]
        state = State(c_sys, np.array([0, 0, 0, 1], dtype=np.float64))
        assert state.dim == 2
        assert np.all(
            state.get_density_matrix()
            == np.array([[0, 0], [0, 1]], dtype=np.complex128)
        )
        assert state.is_trace_one() == True
        assert state.is_hermitian() == True
        assert state.is_positive_semidefinite() == True
        assert np.all(state.calc_eigenvalues() == np.array([0, 1], dtype=np.complex128))

    def test_pauli_basis(self):
        e_sys = ElementalSystem(1, matrix_basis.get_pauli_basis())
        c_sys = CompositeSystem([e_sys])

        # test for vec[1, 0, 0, 0]
        state = State(c_sys, np.array([1, 0, 0, 0], dtype=np.float64))
        assert state.dim == 2
        assert np.all(
            state.get_density_matrix()
            == np.array([[1, 0], [0, 1]], dtype=np.complex128)
        )
        assert state.is_trace_one() == False
        assert state.is_hermitian() == True
        assert state.is_positive_semidefinite() == True
        npt.assert_almost_equal(
            state.calc_eigenvalues(), np.array([1, 1], dtype=np.complex128), decimal=15,
        )

        # test for vec [0, 1, 0, 0]
        state = State(c_sys, np.array([0, 1, 0, 0], dtype=np.float64))
        assert state.dim == 2
        assert np.all(
            state.get_density_matrix()
            == np.array([[0, 1], [1, 0]], dtype=np.complex128)
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
        state = State(c_sys, np.array([0, 0, 1, 0], dtype=np.float64))
        assert state.dim == 2
        assert np.all(
            state.get_density_matrix()
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
        state = State(c_sys, np.array([0, 0, 0, 1], dtype=np.float64))
        assert state.dim == 2
        assert np.all(
            state.get_density_matrix()
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
            state.calc_eigenvalues(),
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
            state.calc_eigenvalues(),
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
            state.calc_eigenvalues(),
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

    def test_convert_basis_form_pauli_to_comp(self):
        comp_basis = matrix_basis.get_comp_basis()

        # CompositeSystem of Pauli basis
        e_sys2 = ElementalSystem(2, matrix_basis.get_normalized_pauli_basis())
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


def test_get_x0_1q():
    e_sys = ElementalSystem(0, matrix_basis.get_normalized_pauli_basis())
    c_sys = CompositeSystem([e_sys])
    state = get_x0_1q(c_sys)
    actual = state.get_density_matrix()
    expected = np.array([[0.5, 0.5], [0.5, 0.5]], dtype=np.complex128)
    npt.assert_almost_equal(actual, expected, decimal=15)


def test_get_x1_1q():
    e_sys = ElementalSystem(0, matrix_basis.get_normalized_pauli_basis())
    c_sys = CompositeSystem([e_sys])
    state = get_x1_1q(c_sys)
    actual = state.get_density_matrix()
    expected = np.array([[0.5, -0.5], [-0.5, 0.5]], dtype=np.complex128)
    npt.assert_almost_equal(actual, expected, decimal=15)


def test_get_y0_1q():
    e_sys = ElementalSystem(0, matrix_basis.get_normalized_pauli_basis())
    c_sys = CompositeSystem([e_sys])
    state = get_y0_1q(c_sys)
    actual = state.get_density_matrix()
    expected = np.array([[0.5, -0.5j], [0.5j, 0.5]], dtype=np.complex128)
    npt.assert_almost_equal(actual, expected, decimal=15)


def test_get_y1_1q():
    e_sys = ElementalSystem(0, matrix_basis.get_normalized_pauli_basis())
    c_sys = CompositeSystem([e_sys])
    state = get_y1_1q(c_sys)
    actual = state.get_density_matrix()
    expected = np.array([[0.5, 0.5j], [-0.5j, 0.5]], dtype=np.complex128)
    npt.assert_almost_equal(actual, expected, decimal=15)


def test_get_z0_1q():
    e_sys = ElementalSystem(0, matrix_basis.get_normalized_pauli_basis())
    c_sys = CompositeSystem([e_sys])
    state = get_z0_1q(c_sys)
    actual = state.get_density_matrix()
    expected = np.array([[1, 0], [0, 0]], dtype=np.complex128)
    npt.assert_almost_equal(actual, expected, decimal=15)


def test_get_z1_1q():
    e_sys = ElementalSystem(0, matrix_basis.get_normalized_pauli_basis())
    c_sys = CompositeSystem([e_sys])
    state = get_z1_1q(c_sys)
    actual = state.get_density_matrix()
    expected = np.array([[0, 0], [0, 1]], dtype=np.complex128)
    npt.assert_almost_equal(actual, expected, decimal=15)


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
    actual = state.get_density_matrix()
    npt.assert_almost_equal(actual, expected, decimal=15)

    # test for comp basis
    e_sys2 = ElementalSystem(2, matrix_basis.get_comp_basis())
    e_sys3 = ElementalSystem(3, matrix_basis.get_comp_basis())
    c_sys = CompositeSystem([e_sys2, e_sys3])
    state = get_bell_2q(c_sys)
    actual = state.get_density_matrix()
    npt.assert_almost_equal(actual, expected, decimal=15)
