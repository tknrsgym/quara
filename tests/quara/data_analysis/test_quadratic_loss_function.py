from abc import abstractmethod

import numpy as np
import numpy.testing as npt
import pytest

from quara.data_analysis.quadratic_loss_function import QuadraticLossFunction


class TestQuadraticLossFunction:
    def test_value(self):
        var_ref = np.array([1, 2, 3, 4], dtype=np.float64)
        loss_func = QuadraticLossFunction(var_ref)

        # Case1: var = var_ref
        actual = loss_func.value(var_ref)
        assert actual == 0.0

        # Case2: var != var_ref
        var = np.array([0, 1, 2, 3], dtype=np.float64)
        actual = loss_func.value(var)
        assert actual == 4.0

        # Case3: shape of var is invalid
        var = np.array([0, 1, 2, 3, 4], dtype=np.float64)
        with pytest.raises(ValueError):
            actual = loss_func.value(var)

    def test_gradient(self):
        var_ref = np.array([1, 2, 3, 4], dtype=np.float64)
        loss_func = QuadraticLossFunction(var_ref)

        # Case1: var = var_ref
        actual = loss_func.gradient(var_ref)
        expected = np.array([0, 0, 0, 0], dtype=np.float64)
        npt.assert_almost_equal(actual, expected, decimal=15)

        # Case2: var != var_ref
        var = np.array([0, 1, 2, 3], dtype=np.float64)
        actual = loss_func.gradient(var)
        expected = np.array([-2, -2, -2, -2], dtype=np.float64)
        npt.assert_almost_equal(actual, expected, decimal=15)

        # Case3: shape of var is invalid
        var = np.array([0, 1, 2, 3, 4], dtype=np.float64)
        with pytest.raises(ValueError):
            actual = loss_func.gradient(var)

    def test_hessian(self):
        var_ref = np.array([1, 2, 3, 4], dtype=np.float64)
        loss_func = QuadraticLossFunction(var_ref)

        # Case1: var = var_ref
        actual = loss_func.hessian(var_ref)
        expected = 2 * np.eye(4, dtype=np.float64)
        npt.assert_almost_equal(actual, expected, decimal=15)

        # Case2: var != var_ref
        var = np.array([0, 1, 2, 3], dtype=np.float64)
        actual = loss_func.hessian(var)
        expected = 2 * np.eye(4, dtype=np.float64)
        npt.assert_almost_equal(actual, expected, decimal=15)

        # Case3: shape of var is invalid
        var = np.array([0, 1, 2, 3, 4], dtype=np.float64)
        with pytest.raises(ValueError):
            actual = loss_func.hessian(var)
