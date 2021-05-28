import numpy as np
import numpy.testing as npt
import pytest

import quara.utils.matrix_util as util


def test_is_unitary():
    # cases: unitary
    target_matrix = np.array([[1, 0], [0, 1]], dtype=np.complex128)
    assert util.is_hermitian(target_matrix)

    target_matrix = np.array([[0, -1j], [1j, 0]], dtype=np.complex128)
    assert util.is_hermitian(target_matrix)

    # cases: not Hermitian
    target_matrix = np.array([[0, 0], [0, 0]], dtype=np.complex128)
    assert util.is_hermitian(target_matrix)

    target_matrix = np.array([[1, 0], [1j, 1]], dtype=np.complex128)
    assert not util.is_hermitian(target_matrix)

    target_matrix = np.array([[0, -1j], [1j, 0], [1j, 0]], dtype=np.complex128)
    assert not util.is_hermitian(target_matrix)


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


def test_truncate_imaginary_part():
    # truncate
    target_matrix = np.array(
        [[1, 1e-14 * 1j], [0, 1]],
        dtype=np.complex128,
    )
    actual = util.truncate_imaginary_part(target_matrix)
    expected = np.eye(2)
    npt.assert_almost_equal(actual, expected, decimal=15)

    # not truncate
    target_matrix = np.array(
        [[1, 1e-13 * 1j], [0, 1]],
        dtype=np.complex128,
    )
    actual = util.truncate_imaginary_part(target_matrix)
    npt.assert_almost_equal(actual, target_matrix, decimal=15)


def test_truncate_computational_fluctuation():
    # truncate
    target_matrix = np.array(
        [[1, 1e-14], [0, 1]],
        dtype=np.complex128,
    )
    actual = util.truncate_computational_fluctuation(target_matrix)
    expected = np.eye(2)
    npt.assert_almost_equal(actual, expected, decimal=15)

    # not truncate
    target_matrix = np.array(
        [[1, 1e-13], [0, 1]],
        dtype=np.complex128,
    )
    actual = util.truncate_computational_fluctuation(target_matrix)
    npt.assert_almost_equal(actual, target_matrix, decimal=15)


def test_truncate_hs():
    # truncate
    target_matrix = np.array(
        [[1, 1e-14j], [0, 1]],
        dtype=np.complex128,
    )
    actual = util.truncate_hs(target_matrix)
    expected = np.eye(2)
    npt.assert_almost_equal(actual, expected, decimal=15)

    # not truncate(ValueError)
    target_matrix = np.array(
        [[1, 1e-13j], [0, 1]],
        dtype=np.complex128,
    )
    with pytest.raises(ValueError):
        util.truncate_hs(target_matrix)

    # not truncate(ValueError)
    target_matrix = np.array(
        [[1, 1e-13j], [0, 1]],
        dtype=np.complex128,
    )
    with pytest.raises(ValueError):
        util.truncate_hs(target_matrix, is_zero_imaginary_part_required=True)

    # not truncate(not ValueError)
    target_matrix = np.array(
        [[1, 1e-13j], [0, 1]],
        dtype=np.complex128,
    )
    actual = util.truncate_hs(target_matrix, is_zero_imaginary_part_required=False)
    npt.assert_almost_equal(actual, target_matrix, decimal=15)


def test_truncate_and_normalize():
    # not truncate
    mat = np.array([0.1, 0.2, 0.3, 0.4])
    actual = util.truncate_and_normalize(mat)
    npt.assert_almost_equal(actual, mat, decimal=15)

    # truncate (eps=default)
    mat = np.array([-0.1, 0.2, 0.3, 0.4])
    actual = util.truncate_and_normalize(mat)
    expected = np.array([0.0, 0.2, 0.3, 0.4]) / 0.9
    npt.assert_almost_equal(actual, expected, decimal=15)

    # truncate (eps=0.1)
    mat = np.array([0.09, 0.2, 0.3, 0.4])
    actual = util.truncate_and_normalize(mat, eps=0.1)
    expected = np.array([0.0, 0.2, 0.3, 0.4]) / 0.9
    npt.assert_almost_equal(actual, expected, decimal=15)


def test_calc_se():
    # list of vectors
    xs = [
        np.array([1, 2]),
        np.array([3, 4]),
    ]
    ys = [
        np.array([11, 12]),
        np.array([13, 14]),
    ]
    actual = util.calc_se(xs, ys)
    expected = float(400)
    assert actual == expected

    # list of matrices
    xs = [
        np.array([[1, 2], [3, 4]]),
        np.array([[5, 6], [7, 8]]),
    ]
    ys = [
        np.array([[11, 12], [13, 14]]),
        np.array([[15, 16], [17, 18]]),
    ]
    actual = util.calc_se(xs, ys)
    expected = float(800)
    assert actual == expected


def test_calc_mse_prob_dists():
    xs_list = [
        [
            np.array([1, 2]),
            np.array([3, 4]),
        ],
        [
            np.array([5, 6]),
            np.array([7, 8]),
        ],
    ]
    ys_list = [
        [
            np.array([11, 12]),
            np.array([13, 14]),
        ],
        [
            np.array([15, 16]),
            np.array([17, 18]),
        ],
    ]
    actual = util.calc_mse_prob_dists(xs_list, ys_list)
    expected = float(400), 0.0
    assert actual == expected


def test_calc_covariance_mat():
    # Case: 1
    # Arrange
    q = np.array([0.5, 0.5], dtype=np.float64)
    n = 10

    # Act
    actual = util.calc_covariance_mat(q, n)

    # Assert
    expected = np.array([[0.25, -0.25], [-0.25, 0.25]]) / n
    npt.assert_almost_equal(actual, expected, decimal=15)

    # Case: 2
    # Arrange
    q = np.array([1.0, 0.0], dtype=np.float64)
    n = 10

    # Act
    actual = util.calc_covariance_mat(q, n)

    # Assert
    expected = np.array([[0.0, 0.0], [0.0, 0.0]]) / n
    npt.assert_almost_equal(actual, expected, decimal=15)


def test_calc_covariance_mat_total():
    empi_dists = [
        (10, np.array([0.5, 0.5])),
        (5, np.array([0.5, 0.5])),
        (10, np.array([1.0, 0.0])),
    ]
    actual = util.calc_covariance_mat_total(empi_dists)
    expected = np.array(
        [
            [0.025, -0.025, 0, 0, 0, 0],
            [-0.025, 0.025, 0, 0, 0, 0],
            [0, 0, 0.05, -0.05, 0, 0],
            [0, 0, -0.05, 0.05, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
        ],
        dtype=np.float64,
    )
    npt.assert_almost_equal(actual, expected, decimal=15)


def test_calc_direct_sum():
    # case1: success
    matrices = [
        np.array([[1, 2], [3, 4]]),
        np.array([[11, 12, 13], [14, 15, 16], [17, 18, 19]]),
    ]
    actual = util.calc_direct_sum(matrices)
    expected = np.array(
        [
            [1, 2, 0, 0, 0],
            [3, 4, 0, 0, 0],
            [0, 0, 11, 12, 13],
            [0, 0, 14, 15, 16],
            [0, 0, 17, 18, 19],
        ]
    )
    npt.assert_almost_equal(actual, expected, decimal=15)

    # case2: not matrix
    matrices = [
        np.array([1, 2]),
        np.array([11, 12, 13]),
    ]
    with pytest.raises(ValueError):
        util.calc_direct_sum(matrices)

    # case3: not square matrix
    matrices = [
        np.array([[1, 2], [3, 4], [5, 6]]),
        np.array([[11, 12, 13], [14, 15, 16], [17, 18, 19]]),
    ]
    with pytest.raises(ValueError):
        util.calc_direct_sum(matrices)


def test_calc_conjugate():
    # Arrange
    x = np.array([[1, 2], [3, 4]])
    v = np.array([[5, 6], [7, 8]])

    # Act
    actual = util.calc_conjugate(x, v)

    # Assert
    expected = np.array([[63, 145], [143, 329]])
    npt.assert_almost_equal(actual, expected, decimal=15)


def test_calc_left_inv():
    # case1: success
    # Arrange
    x = np.array([[2, 5], [1, 3]])

    # Act
    actual = util.calc_left_inv(x)

    # Assert
    expected = np.array([[3, -5], [-1, 2]])
    npt.assert_almost_equal(actual, expected, decimal=12)

    # case2: not full rank
    # Arrange
    x = np.array([[2, 5], [4, 10]])

    # Act
    with pytest.raises(ValueError):
        util.calc_left_inv(x)


def test_calc_fisher_matrix():
    # case1: not replace
    # Arrange
    prob_dist = np.array([0.9, 0.1], dtype=np.float64)
    grad_prob_dist = [
        np.array([3, 4], dtype=np.float64),
        np.array([5, 6], dtype=np.float64),
    ]

    # Act
    actual = util.calc_fisher_matrix(prob_dist, grad_prob_dist)

    # Assert
    expected = np.array([[260, 940 / 3], [940 / 3, 3400 / 9]], dtype=np.float64)
    npt.assert_almost_equal(actual, expected, decimal=15)

    # case2: replace
    # Arrange
    prob_dist = np.array([0.9, 0.1], dtype=np.float64)
    grad_prob_dist = [
        np.array([3, 4], dtype=np.float64),
        np.array([5, 6], dtype=np.float64),
    ]
    eps = 0.2

    # Act
    actual = util.calc_fisher_matrix(prob_dist, grad_prob_dist, eps)

    # Assert
    expected = np.array(
        [[90 / 7 + 125, 120 / 7 + 150], [120 / 7 + 150, 160 / 7 + 180]],
        dtype=np.float64,
    )
    npt.assert_almost_equal(actual, expected, decimal=15)

    # case3: error case. some entries < 0
    with pytest.raises(ValueError):
        prob_dist = np.array([0.9, -0.1], dtype=np.float64)
        grad_prob_dist = [
            np.array([3, 4], dtype=np.float64),
            np.array([5, 6], dtype=np.float64),
        ]
        util.calc_fisher_matrix(prob_dist, grad_prob_dist)

    # case4: error case. some entries > 1
    with pytest.raises(ValueError):
        prob_dist = np.array([1.1, 0.1], dtype=np.float64)
        grad_prob_dist = [
            np.array([3, 4], dtype=np.float64),
            np.array([5, 6], dtype=np.float64),
        ]
        util.calc_fisher_matrix(prob_dist, grad_prob_dist)

    # case5: error case. the sum of prob_dist is not 1
    with pytest.raises(ValueError):
        prob_dist = np.array([0.8, 0.1], dtype=np.float64)
        grad_prob_dist = [
            np.array([3, 4], dtype=np.float64),
            np.array([5, 6], dtype=np.float64),
        ]
        util.calc_fisher_matrix(prob_dist, grad_prob_dist)

    # case6: error case. the size of prob_dist and grad_prob_dist are not equal
    with pytest.raises(ValueError):
        prob_dist = np.array([0.8, 0.1, 0.1], dtype=np.float64)
        grad_prob_dist = [
            np.array([3, 4], dtype=np.float64),
            np.array([5, 6], dtype=np.float64),
        ]
        util.calc_fisher_matrix(prob_dist, grad_prob_dist)

    # case7: error case. eps is not a positive number
    with pytest.raises(ValueError):
        prob_dist = np.array([0.9, 0.1], dtype=np.float64)
        grad_prob_dist = [
            np.array([3, 4], dtype=np.float64),
            np.array([5, 6], dtype=np.float64),
        ]
        eps = 0.0
        util.calc_fisher_matrix(prob_dist, grad_prob_dist, eps=eps)


def test_replace_prob_dist():
    # case1: p >= eps
    # Arrange
    prob_dist = np.array([0.9, 0.1], dtype=np.float64)
    eps = 0.1

    # Act
    actual = util.replace_prob_dist(prob_dist, eps)

    # Assert
    expected = np.array([0.9, 0.1], dtype=np.float64)
    npt.assert_almost_equal(actual, expected, decimal=15)

    # case2: p < eps
    # Arrange
    prob_dist = np.array([0.99, 0.01], dtype=np.float64)
    eps = 0.1

    # Act
    actual = util.replace_prob_dist(prob_dist, eps)

    # Assert
    expected = np.array([0.89, 0.1], dtype=np.float64)
    npt.assert_almost_equal(actual, expected, decimal=15)


def test_calc_fisher_matrix_total():
    prob_dists = [
        np.array([0.9, 0.1], dtype=np.float64),
        np.array([0.8, 0.2], dtype=np.float64),
    ]
    grad_prob_dists = [
        [
            np.array([3, 4], dtype=np.float64),
            np.array([5, 6], dtype=np.float64),
        ],
        [
            np.array([7, 8], dtype=np.float64),
            np.array([9, 10], dtype=np.float64),
        ],
    ]
    weights = [1.0, 0.5]

    # Act
    actual = util.calc_fisher_matrix_total(prob_dists, grad_prob_dists, weights)

    # Assert
    expected = np.array(
        [
            [260 + 490 / 16 + 810 / 4, 940 / 3 + 260],
            [940 / 3 + 260, 3400 / 9 + 290],
        ],
        dtype=np.float64,
    )
    npt.assert_almost_equal(actual, expected, decimal=15)
