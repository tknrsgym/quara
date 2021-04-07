from abc import abstractmethod

import numpy as np
import numpy.testing as npt
import pytest

from quara.loss_function.simple_quadratic_loss_function import (
    SimpleQuadraticLossFunction,
)


class TestSimpleQuadraticLossFunction:
    def test_update_on_value_true(self):
        var_ref = np.array([1, 2, 3, 4], dtype=np.float64)
        loss_func = SimpleQuadraticLossFunction(var_ref)
        loss_func._on_value = False
        loss_func._update_on_value_true()
        assert loss_func.on_value == True

    def test_update_on_gradient_true(self):
        var_ref = np.array([1, 2, 3, 4], dtype=np.float64)
        loss_func = SimpleQuadraticLossFunction(var_ref)
        loss_func._on_gradient = False
        loss_func._update_on_gradient_true()
        assert loss_func.on_gradient == True

    def test_update_on_hessian_true(self):
        var_ref = np.array([1, 2, 3, 4], dtype=np.float64)
        loss_func = SimpleQuadraticLossFunction(var_ref)
        loss_func._on_hessian = False
        loss_func._update_on_hessian_true()
        assert loss_func.on_hessian == True

    def test_is_option_sufficient(self):
        var_ref = np.array([1, 2, 3, 4], dtype=np.float64)
        loss_func = SimpleQuadraticLossFunction(var_ref)
        assert loss_func.is_option_sufficient() == True

    def test_value(self):
        var_ref = np.array([1, 2, 3, 4], dtype=np.float64)
        loss_func = SimpleQuadraticLossFunction(var_ref)
        assert loss_func.on_value == True

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
        loss_func = SimpleQuadraticLossFunction(var_ref)
        assert loss_func.on_gradient == True

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
        loss_func = SimpleQuadraticLossFunction(var_ref)
        assert loss_func.on_hessian == True

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
