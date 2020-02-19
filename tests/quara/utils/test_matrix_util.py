import pytest

import numpy as np

import quara.utils.matrix_util as util


def test_is_hermitian():
    # cases: Hermitian
    target_matrix = np.array([[1, 0], [0, 1]], dtype=np.complex128)
    assert util.is_hermitian(target_matrix)

    target_matrix = np.array([[0, -1j], [1j, 0]], dtype=np.complex128)
    assert util.is_hermitian(target_matrix)

    target_matrix = np.array([[0, 0], [0, 0]], dtype=np.complex128)
    assert util.is_hermitian(target_matrix)

    # cases: not Hermitian
    target_matrix = np.array([[1, 0], [1j, 1]], dtype=np.complex128)
    assert not util.is_hermitian(target_matrix)

    target_matrix = np.array([[0, -1j], [1j, 0], [1j, 0]], dtype=np.complex128)
    assert not util.is_hermitian(target_matrix)


def test_is_positive_semidefinite():
    # cases: positive semidefinite
    target_matrix = np.array([[1, 0], [0, 1]], dtype=np.complex128)
    assert util.is_positive_semidefinite(target_matrix)

    target_matrix = np.array([[0, 0], [0, 0]], dtype=np.complex128)
    assert util.is_positive_semidefinite(target_matrix)

    # cases: not positive semidefinite
    target_matrix = np.array([[0, -1j], [1j, 0]], dtype=np.complex128)
    assert not util.is_positive_semidefinite(target_matrix)

    target_matrix = np.array([[1, 0], [0, -1]], dtype=np.complex128)
    assert not util.is_positive_semidefinite(target_matrix)

    target_matrix = np.array([[0, -1j], [1j, 0], [1j, 0]], dtype=np.complex128)
    assert not util.is_positive_semidefinite(target_matrix)


def test_partial_trace1():
    whole = np.arange(16).reshape(4, 4)
    # expected = np.array([[5, 9], [21, 25]])
    expected = np.array([[10, 12], [18, 20]])
    actual = util.partial_trace1(whole, 2)
    assert np.array_equal(actual, expected)

    identity = np.eye(2, dtype=np.complex128)
    pauli_x = np.array([[0, 1], [1, 0]], dtype=np.complex128)

    # Tr_1[I \otimes I] = Tr[I]I
    expected = np.array([[2, 0], [0, 2]])
    tensor = np.kron(identity, identity)
    actual = util.partial_trace1(tensor, 2)
    assert np.array_equal(actual, expected)

    # Tr_1[X \otimes I] = Tr[X]I
    expected = np.array([[0, 0], [0, 0]])
    tensor = np.kron(pauli_x, identity)
    actual = util.partial_trace1(tensor, 2)
    assert np.array_equal(actual, expected)

    # Tr_1[I \otimes X] = Tr[I]X
    expected = np.array([[0, 2], [2, 0]])
    tensor = np.kron(identity, pauli_x)
    actual = util.partial_trace1(tensor, 2)
    assert np.array_equal(actual, expected)


def test_is_tp():
    # cases: TP
    target_matrix = np.array(
        [[1, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 1]], dtype=np.complex128
    )
    assert util.is_tp(target_matrix, 2)

    # cases: not TP
    target_matrix = np.array(
        [[1, 0, 0, 0], [1, 0, 0, 0], [1, 0, 0, 0], [1, 0, 0, 0]], dtype=np.complex128
    )
    assert not util.is_tp(target_matrix, 2)
