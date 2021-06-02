import numpy as np
import numpy.testing as npt
import pytest

from quara.data_analysis import data_analysis
from quara.math import norm


def test_calc_mse_general_norm():
    # Arrange
    xs = [
        np.array([2.0, 3.0], dtype=np.float64),
        np.array([4.0, 5.0], dtype=np.float64),
    ]
    y = np.array([1.0, 2.0], dtype=np.float64)

    # Act
    actual = data_analysis.calc_mse_general_norm(xs, y, norm.l2_norm)

    # Assert
    npt.assert_almost_equal(actual, 10.0, decimal=14)


def test_calc_covariance_matrix_of_prob_dist():
    # Case: 1
    # Arrange
    prob_dist = np.array([0.5, 0.5], dtype=np.float64)
    data_num = 10

    # Act
    actual = data_analysis.calc_covariance_matrix_of_prob_dist(prob_dist, data_num)

    # Assert
    expected = np.array([[0.25, -0.25], [-0.25, 0.25]]) / data_num
    npt.assert_almost_equal(actual, expected, decimal=15)

    # Case: 2
    # Arrange
    prob_dist = np.array([1.0, 0.0], dtype=np.float64)
    data_num = 10

    # Act
    actual = data_analysis.calc_covariance_matrix_of_prob_dist(prob_dist, data_num)

    # Assert
    expected = np.array([[0.0, 0.0], [0.0, 0.0]]) / data_num
    npt.assert_almost_equal(actual, expected, decimal=15)


def test_calc_covariance_matrix_of_prob_dists():
    # Arrange
    prob_dists = [
        np.array([0.5, 0.5], dtype=np.float64),
        np.array([0.5, 0.5], dtype=np.float64),
        np.array([1.0, 0.0], dtype=np.float64),
    ]
    data_num = 10

    # Act
    actual = data_analysis.calc_covariance_matrix_of_prob_dists(prob_dists, data_num)

    # Assert
    mat = [
        [0.25, -0.25, 0.0, 0.0, 0.0, 0.0],
        [-0.25, 0.25, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.25, -0.25, 0.0, 0.0],
        [0.0, 0.0, -0.25, 0.25, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    ]
    expected = np.array(mat) / data_num
    npt.assert_almost_equal(actual, expected, decimal=15)


def test_extract_empi_dists_sequences():
    # Assert
    source = [
        [
            [
                (1000, np.array([0.574, 0.426])),
                (1000, np.array([0.577, 0.423])),
                (1000, np.array([0.914, 0.086])),
            ],
            [
                (10000, np.array([0.5676, 0.4324])),
                (10000, np.array([0.5813, 0.4187])),
                (10000, np.array([0.9241, 0.0759])),
            ],
        ],
        [
            [
                (1000, np.array([0.575, 0.425])),
                (1000, np.array([0.586, 0.414])),
                (1000, np.array([0.925, 0.075])),
            ],
            [
                (10000, np.array([0.5643, 0.4357])),
                (10000, np.array([0.5793, 0.4207])),
                (10000, np.array([0.9216, 0.0784])),
            ],
        ],
    ]
    # Act
    actual = data_analysis.extract_empi_dists_sequences(source)

    # Assert
    expected = [
        [
            [
                np.array([0.574, 0.426]),
                np.array([0.577, 0.423]),
                np.array([0.914, 0.086]),
            ],
            [
                np.array([0.575, 0.425]),
                np.array([0.586, 0.414]),
                np.array([0.925, 0.075]),
            ],
        ],
        [
            [
                np.array([0.5676, 0.4324]),
                np.array([0.5813, 0.4187]),
                np.array([0.9241, 0.0759]),
            ],
            [
                np.array([0.5643, 0.4357]),
                np.array([0.5793, 0.4207]),
                np.array([0.9216, 0.0784]),
            ],
        ],
    ]
