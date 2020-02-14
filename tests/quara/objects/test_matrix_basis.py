import numpy as np

from quara.objects import matrix_basis


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
