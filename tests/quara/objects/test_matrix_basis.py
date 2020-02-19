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
        # Case 1: 数が足りていない
        source = matrix_basis.get_pauli_basis().basis
        source = source[:-1]
        with pytest.raises(ValueError):
            _ = MatrixBasis(source)

        # Case 2: 独立ではない（4つ目が1つ目と2つ目の和）
        source = matrix_basis.get_pauli_basis().basis
        source = source[:-1]
        invalid_array = source[0] + source[1]
        source.append(invalid_array)
        with pytest.raises(ValueError):
            _ = MatrixBasis(source)


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
    assert basis.is_squares() == True
    assert basis.is_same_size() == True
    assert basis._is_basis() == True
    assert basis.is_orthogonal() == True
    assert basis.is_normal() == False
    assert basis.is_hermitian() == True
    assert basis.is_scalar_mult_of_identity() == True
    assert basis.is_trace_less() == True
