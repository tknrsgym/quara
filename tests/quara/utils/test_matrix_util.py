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


def test_partial_trace():
    whole = np.arange(16).reshape(4, 4)
    expected = np.array([[5, 9], [21, 25]])
    actual = util.partial_trace(whole, 2)
    assert np.array_equal(actual, expected)


def test_is_tp():
    # cases: TP
    target_matrix = np.array(
        [[1, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 1]], dtype=np.complex128
    )
    assert util.is_tp(target_matrix, 2)

    target_matrix = np.array(
        [
            [
                9.99777211e-01 + 0.00000000e00j,
                -1.10126364e-05 - 1.49172469e-02j,
                4.39788591e-03 - 1.42542774e-02j,
                9.55558552e-01 + 2.94043679e-01j,
            ],
            [
                -1.10126364e-05 + 1.49172469e-02j,
                2.22786124e-04 + 0.00000000e00j,
                2.12627558e-04 + 6.57931277e-05j,
                -4.39776874e-03 + 1.42542154e-02j,
            ],
            [
                4.39788591e-03 + 1.42542774e-02j,
                2.12627558e-04 - 6.57931277e-05j,
                2.22788923e-04 + 0.00000000e00j,
                1.10126364e-05 + 1.49172469e-02j,
            ],
            [
                9.55558552e-01 - 2.94043679e-01j,
                -4.39776874e-03 - 1.42542154e-02j,
                1.10126364e-05 - 1.49172469e-02j,
                9.99777214e-01 + 0.00000000e00j,
            ],
        ],
        dtype=np.complex128,
    )
    assert util.is_tp(target_matrix, 2)

    # cases: not TP
    target_matrix = np.array(
        [[1, 0, 0, 0], [1, 0, 0, 0], [1, 0, 0, 0], [1, 0, 0, 0]], dtype=np.complex128
    )
    assert not util.is_tp(target_matrix, 2)
