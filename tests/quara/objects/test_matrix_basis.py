import numpy as np

from quara.objects import matrix_basis
from quara.objects.matrix_basis import VectorizedMatrixBasis


def test_get_comp_basis():
    basis = matrix_basis.get_comp_basis()
    assert basis.dim == 2
    assert basis.is_squares() == True
    assert basis.is_same_size() == True
    assert basis.is_basis() == True
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
    assert basis.is_squares() == True
    assert basis.is_same_size() == True
    assert basis.is_basis() == True
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
    assert basis.is_squares() == True
    assert basis.is_same_size() == True
    assert basis.is_basis() == True
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
    assert np.all(basis[3] == 1 / np.sqrt(2) * [[1, 0], [0, -1]], dtype=np.complex128)

    assert basis.size() == (2, 2)
    assert len(basis) == 4


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
        for i, actual in enumerate(actual_vec_basis):
            assert np.allclose(actual, expected_vec_basis[i])

        # test len
        actual = len(actual_vec_basis)
        assert actual == 4

        actual_org_basis = actual_vec_basis.org_basis
        for i, actual in enumerate(actual_org_basis):
            assert np.array_equal(actual, source_basis[i])
