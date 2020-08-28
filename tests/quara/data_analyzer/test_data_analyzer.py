import numpy as np
import numpy.testing as npt
import pytest

from quara.data_analyzer import data_analyzer
from quara.math import norm


def test_calc_mse():
    # Arrange
    xs = [
        np.array([2.0, 3.0], dtype=np.float64),
        np.array([4.0, 5.0], dtype=np.float64),
    ]
    y = np.array([1.0, 2.0], dtype=np.float64)

    # Act
    actual = data_analyzer.calc_mse(xs, y, norm.l2_norm)

    # Assert
    npt.assert_almost_equal(actual, 10.0, decimal=14)


def test_calc_covariance_matrix_of_prob_dist():
    # Case: 1
    # Arrange
    prob_dist = np.array([0.5, 0.5], dtype=np.float64)
    data_num = 10

    # Act
    actual = data_analyzer.calc_covariance_matrix_of_prob_dist(prob_dist, data_num)

    # Assert
    expected = np.array([[0.25, -0.25], [-0.25, 0.25]]) / data_num
    npt.assert_almost_equal(actual, expected, decimal=15)

    # Case: 2
    # Arrange
    prob_dist = np.array([1.0, 0.0], dtype=np.float64)
    data_num = 10

    # Act
    actual = data_analyzer.calc_covariance_matrix_of_prob_dist(prob_dist, data_num)

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
    actual = data_analyzer.calc_covariance_matrix_of_prob_dists(prob_dists, data_num)

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


def test_calc_mse_of_linear_estimator():
    # Arrange
    matA = np.eye(6)
    prob_dists = [
        np.array([0.5, 0.5], dtype=np.float64),
        np.array([0.5, 0.5], dtype=np.float64),
        np.array([1.0, 0.0], dtype=np.float64),
    ]
    data_num = 10

    # Act
    actual = data_analyzer.calc_mse_of_linear_estimator(matA, prob_dists, data_num)

    # Assert
    expected = 1.0 / data_num
    npt.assert_almost_equal(actual, expected, decimal=15)
