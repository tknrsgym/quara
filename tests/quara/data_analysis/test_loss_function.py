from abc import abstractmethod

import numpy as np
import numpy.testing as npt
import pytest

from quara.data_analysis.loss_function import LossFunction


class TestLossFunction:
    def test_access_num_var(self):
        loss_func = LossFunction(4, True, True)
        assert loss_func.num_var == 4

        # Test that "num_var" cannot be updated
        with pytest.raises(AttributeError):
            loss_func.num_var = 5

    def test_access_on_gradient(self):
        loss_func = LossFunction(4, True, True)
        assert loss_func.on_gradient == True

        loss_func = LossFunction(4, False, True)
        assert loss_func.on_gradient == False

        # Test that "on_gradient" cannot be updated
        with pytest.raises(AttributeError):
            loss_func.on_gradient = True

    def test_access_on_hessian(self):
        loss_func = LossFunction(4, True, True)
        assert loss_func.on_hessian == True

        loss_func = LossFunction(4, True, False)
        assert loss_func.on_hessian == False

        # Test that "on_hessian" cannot be updated
        with pytest.raises(AttributeError):
            loss_func.on_hessian = True

    def test_validate_var_shape(self):
        loss_func = LossFunction(4, True, True)

        var = np.array([1, 2, 3, 4], dtype=np.float64)
        loss_func._validate_var_shape(var)

        var = np.array([1, 2, 3, 4, 5], dtype=np.float64)
        with pytest.raises(ValueError):
            loss_func._validate_var_shape(var)
