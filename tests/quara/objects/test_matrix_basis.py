import numpy as np
import pytest

from quara.objects import matrix_basis
from quara.objects.matrix_basis import MatrixBasis, VectorizedMatrixBasis


def test_get_comp_basis():
    basis = matrix_basis.get_comp_basis()
    assert basis.dim == 2
    assert basis._is_squares() == True
    assert basis._is_same_size() == True
    assert basis._is_basis() == True
    assert basis.is_orthogonal() == True
    assert basis.is_normal() == True
    assert basis.is_hermitian() == False  # computational basisはFalse
    assert basis.is_scalar_mult_of_identity() == False  # computational basisはFalse
    assert basis.is_trace_less() == False  # computational basisはFalse
    assert np.all(basis[0] == np.array([[1, 0], [0, 0]], dtype=np.complex128))
    assert np.all(basis[1] == np.array([[0, 1], [0, 0]], dtype=np.complex128))
    assert np.all(basis[2] == np.array([[0, 0], [1, 0]], dtype=np.complex128))
    assert np.all(basis[3] == np.array([[0, 0], [0, 1]], dtype=np.complex128))
    assert basis.size() == (2, 2)
    assert len(basis) == 4


def test_get_pauli_basis():
    basis = matrix_basis.get_pauli_basis()
    assert basis.dim == 2
    assert basis._is_squares() == True
    assert basis._is_same_size() == True
    assert basis._is_basis() == True
    assert basis.is_orthogonal() == True
    assert basis.is_normal() == False  # Pauli basisはFalse
    assert basis.is_hermitian() == True
    assert basis.is_scalar_mult_of_identity() == True
    assert basis.is_trace_less() == True
    assert np.all(basis[0] == np.array([[1, 0], [0, 1]], dtype=np.complex128))
    assert np.all(basis[1] == np.array([[0, 1], [1, 0]], dtype=np.complex128))
    assert np.all(basis[2] == np.array([[0, -1j], [1j, 0]], dtype=np.complex128))
    assert np.all(basis[3] == np.array([[1, 0], [0, -1]], dtype=np.complex128))
    assert basis.size() == (2, 2)
    assert len(basis) == 4


def test_get_normalized_pauli_basis():
    basis = matrix_basis.get_normalized_pauli_basis()
    assert basis.dim == 2
    assert basis._is_squares() == True
    assert basis._is_same_size() == True
    assert basis._is_basis() == True
    assert basis.is_orthogonal() == True
    assert basis.is_normal() == True
    assert basis.is_hermitian() == True
    assert basis.is_scalar_mult_of_identity() == True
    assert basis.is_trace_less() == True
    assert np.all(
        basis[0] == 1 / np.sqrt(2) * np.array([[1, 0], [0, 1]], dtype=np.complex128)
    )
    assert np.all(
        basis[1] == 1 / np.sqrt(2) * np.array([[0, 1], [1, 0]], dtype=np.complex128)
    )
    assert np.all(
        basis[2] == 1 / np.sqrt(2) * np.array([[0, -1j], [1j, 0]], dtype=np.complex128)
    )
    assert np.all(
        basis[3] == 1 / np.sqrt(2) * np.array([[1, 0], [0, -1]], dtype=np.complex128)
    )
    assert basis.size() == (2, 2)
    assert len(basis) == 4


class TestMatrixBasis:
    def test_raise_not_basis(self):
        # Testing for non-basis inputs
        # Case 1: Not enough matrices.
        source = matrix_basis.get_pauli_basis().basis
        source = source[:-1]
        with pytest.raises(ValueError):
            _ = MatrixBasis(source)

        # Case 2: Not independent (B_3 = B_0 + B_1)
        source = matrix_basis.get_pauli_basis().basis
        source = source[:-1]
        invalid_array = source[0] + source[1]
        source.append(invalid_array)
        with pytest.raises(ValueError):
            _ = MatrixBasis(source)

    def test_is_same_size(self):
        # Case1: All the same size
        source_basis = matrix_basis.get_pauli_basis().basis
        basis = MatrixBasis(source_basis)
        assert basis._is_same_size() == True

        # Case2: Not same size
        source = [
            np.array([[1, 0], [0, 1]]),
            np.array([[0, 1, 0], [0, 0, 0], [0, 0, 0]]),
            np.array([[1, 1], [1, 0]]),
            np.array([[0, 0], [0, 1]]),
        ]
        with pytest.raises(ValueError):
            _ = MatrixBasis(source)

    def test_is_squares(self):
        # Case1: Square matrix
        source_basis = matrix_basis.get_pauli_basis().basis
        basis = MatrixBasis(source_basis)
        assert basis._is_squares() == True

        # Case2: There is a non-square matrix
        source = [
            np.array([[1, 0], [0, 1]]),
            np.array([[0, 1], [0, 0], [0, 0]]),
            np.array([[1, 1], [1, 0]]),
            np.array([[0, 0], [0, 1]]),
        ]
        with pytest.raises(ValueError):
            _ = MatrixBasis(source)

    def test_is_orthogonal(self):
        # Case1: orthorogonal
        source_basis = matrix_basis.get_pauli_basis().basis
        m_basis = MatrixBasis(source_basis)
        assert m_basis.is_orthogonal() == True

        # Case2: basis, but non orthorogonal
        non_orthorogonal_source = [
            np.array([[1, 0], [0, 1]]),
            np.array([[0, 1], [0, 0]]),
            np.array([[1, 1], [1, 0]]),
            np.array([[0, 0], [0, 1]]),
        ]
        non_orthorogonal_basis = MatrixBasis(non_orthorogonal_source)
        assert non_orthorogonal_basis.is_orthogonal() == False

        # Case3: basis, but non orthorogonal
        X = np.array([[1, 0], [0, 0]])
        Y = np.array([[0, 1], [0, 0]])
        Z = X + Y + 2
        non_orthorogonal_source = [np.eye(2), X, Y, Z]
        non_orthorogonal_basis = MatrixBasis(non_orthorogonal_source)
        assert non_orthorogonal_basis.is_orthogonal() == False

    def test_is_normal(self):
        # Case1: Normalized
        source = matrix_basis.get_normalized_pauli_basis().basis
        normalized_basis = MatrixBasis(source)
        assert normalized_basis.is_normal() == True

        # Case2: Not Normalized
        source = matrix_basis.get_pauli_basis().basis
        non_normalized_basis = MatrixBasis(source)
        assert non_normalized_basis.is_normal() == False

    def test_is_hermitian(self):
        # Case1: Hermitian matrix
        source = matrix_basis.get_pauli_basis().basis
        hermitian_basis = MatrixBasis(source)
        assert hermitian_basis.is_hermitian() == True

        # Case2: Non Hermitian matrix
        non_hermitian_source = [
            np.array([[1, 0], [0, 0]]),
            np.array([[0, 1], [0, 0]]),
            np.array([[0, 0], [1, 0]]),
            np.array([[0, 0], [0, 1]]),
        ]
        non_hermitian_basis = MatrixBasis(non_hermitian_source)
        assert non_hermitian_basis.is_hermitian() == False

    def test_is_scalar_mult_of_identity(self):
        # Case1: B_0 = C*I
        source = matrix_basis.get_pauli_basis().basis
        basis = MatrixBasis(source)
        assert basis.is_scalar_mult_of_identity() == True

        # Case2: B_0 != C*I
        source = matrix_basis.get_comp_basis().basis
        basis = MatrixBasis(source)
        assert basis.is_scalar_mult_of_identity() == False

    def test_is_trace_less(self):
        # Case1: Tr[B_alpha] = 0, alpha >= 1
        source = matrix_basis.get_pauli_basis().basis
        basis = MatrixBasis(source)
        assert basis.is_trace_less() == True

        # Case2: Tr[B_alpha] != 0, alpha >= 1
        source = matrix_basis.get_comp_basis().basis
        basis = MatrixBasis(source)
        assert basis.is_trace_less() == False

    def test_to_vect(self):
        source = matrix_basis.get_pauli_basis().basis
        basis = MatrixBasis(source)
        v_basis = basis.to_vect()
        assert np.allclose(v_basis.basis[0], np.array([1, 0, 0, 1]))
        assert np.allclose(v_basis.basis[1], np.array([0, 1, 1, 0]))
        assert np.allclose(v_basis.basis[2], np.array([0, -1j, 1j, 0]))
        assert np.allclose(v_basis.basis[3], np.array([1, 0, 0, -1]))

    def test_get_item(self):
        source_np = matrix_basis.get_pauli_basis().basis
        basis = MatrixBasis(source_np)
        for i in range(len(source_np)):
            assert np.allclose(basis[i], source_np[i])

    def test_str(self):
        source_np = matrix_basis.get_pauli_basis().basis
        basis = MatrixBasis(source_np)
        assert str(basis) == str(source_np)


class TestVectorizedMatrixBasis:
    def test_convert(self):
        source_basis = matrix_basis.get_pauli_basis()
        actual_vec_basis = VectorizedMatrixBasis(source_basis)
        expected_vec_basis = [
            np.array([1, 0, 0, 1]),
            np.array([0, 1, 1, 0]),
            np.array([0, -1j, 1j, 0]),
            np.array([1, 0, 0, -1]),
        ]
        assert len(actual_vec_basis) == len(expected_vec_basis)

        # test member "basis"
        for i, actual in enumerate(actual_vec_basis.basis):
            assert np.allclose(actual, expected_vec_basis[i])

        # test iter
        for i, actual in enumerate(actual_vec_basis):
            assert np.allclose(actual, expected_vec_basis[i])

        # test len
        actual = len(actual_vec_basis)
        assert actual == 4

        # test original matrix basis
        actual_org_basis = actual_vec_basis.org_basis
        for i, actual in enumerate(actual_org_basis):
            assert np.array_equal(actual, source_basis[i])


def test_get_gell_mann_basis():
    basis = matrix_basis.get_gell_mann_basis()

    assert basis.dim == 3
    assert basis._is_squares() == True
    assert basis._is_same_size() == True
    assert basis._is_basis() == True
    assert basis.is_orthogonal() == True
    assert basis.is_normal() == False
    assert basis.is_hermitian() == True
    assert basis.is_scalar_mult_of_identity() == True
    assert basis.is_trace_less() == True
