from abc import abstractmethod

import numpy as np
import numpy.testing as npt
import pytest

from quara.data_analysis.loss_function import LossFunction


class TestLossFunction:
    def test_access_num_var(self):
        loss_func = LossFunction(4)
        assert loss_func.num_var == 4

        # Test that "num_var" cannot be updated
        with pytest.raises(AttributeError):
            loss_func.num_var = 5

    def test_access_on_value(self):
        loss_func = LossFunction(4)
        assert loss_func.on_value == False

        # Test that "on_value" cannot be updated
        with pytest.raises(AttributeError):
            loss_func.on_value = True

    def test_reset_on_value(self):
        loss_func = LossFunction(4)
        loss_func._on_value = True
        loss_func._reset_on_value()
        assert loss_func.on_value == False

    def test_set_on_value(self):
        loss_func = LossFunction(4)
        loss_func._set_on_value(True)
        assert loss_func.on_value == True
        loss_func._set_on_value(False)
        assert loss_func.on_value == False

    def test_access_on_gradient(self):
        loss_func = LossFunction(4)
        assert loss_func.on_gradient == False

        # Test that "on_gradient" cannot be updated
        with pytest.raises(AttributeError):
            loss_func.on_gradient = True

    def test_reset_on_gradient(self):
        loss_func = LossFunction(4)
        loss_func._on_gradient = True
        loss_func._reset_on_gradient()
        assert loss_func.on_gradient == False

    def test_set_on_gradient(self):
        loss_func = LossFunction(4)
        loss_func._set_on_gradient(True)
        assert loss_func.on_gradient == True
        loss_func._set_on_gradient(False)
        assert loss_func.on_gradient == False

    def test_access_on_hessian(self):
        loss_func = LossFunction(4)
        assert loss_func.on_hessian == False

        # Test that "on_hessian" cannot be updated
        with pytest.raises(AttributeError):
            loss_func.on_hessian = True

    def test_reset_on_hessian(self):
        loss_func = LossFunction(4)
        loss_func._on_hessian = True
        loss_func._reset_on_hessian()
        assert loss_func.on_hessian == False

    def test_set_on_hessian(self):
        loss_func = LossFunction(4)
        loss_func._set_on_hessian(True)
        assert loss_func.on_hessian == True
        loss_func._set_on_hessian(False)
        assert loss_func.on_hessian == False

    def test_validate_var_shape(self):
        loss_func = LossFunction(4)

        var = np.array([1, 2, 3, 4], dtype=np.float64)
        loss_func._validate_var_shape(var)

        var = np.array([1, 2, 3, 4, 5], dtype=np.float64)
        with pytest.raises(ValueError):
            loss_func._validate_var_shape(var)
