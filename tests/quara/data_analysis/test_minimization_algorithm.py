import numpy as np
import numpy.testing as npt
import pytest

from quara.data_analysis.minimization_algorithm import (
    MinimizationAlgorithm,
    MinimizationAlgorithmOption,
)
from quara.data_analysis.quadratic_loss_function import (
    QuadraticLossFunction,
    QuadraticLossFunctionOption,
)


class TestMinimizationAlgorithm:
    def test_access_is_gradient_required(self):
        algo = MinimizationAlgorithm()
        assert algo.is_gradient_required == False

        with pytest.raises(AttributeError):
            algo.is_gradient_required = True

    def test_access_is_hessian_required(self):
        algo = MinimizationAlgorithm()
        assert algo.is_hessian_required == False

        with pytest.raises(AttributeError):
            algo.is_hessian_required = True

    def test_access_loss(self):
        algo = MinimizationAlgorithm()
        assert algo.loss == None

        with pytest.raises(AttributeError):
            algo.loss = True

    def test_set_from_loss(self):
        loss_option = QuadraticLossFunctionOption()
        var_ref = np.array([1, 0, 0, 0], dtype=np.float64) / np.sqrt(2)
        loss = QuadraticLossFunction(var_ref)

        algo = MinimizationAlgorithm()
        algo.set_from_loss(loss)
        assert algo.loss is not None
        assert algo.is_loss_sufficient() == True

    def test_is_loss_sufficient(self):
        algo = MinimizationAlgorithm()
        assert algo.is_loss_sufficient() == True

    def test_access_option(self):
        algo = MinimizationAlgorithm()
        assert algo.option == None

        with pytest.raises(AttributeError):
            algo.option = True

    def test_set_from_option(self):
        option = MinimizationAlgorithmOption()

        algo = MinimizationAlgorithm()
        algo.set_from_option(option)
        assert algo.option is not None
        assert algo.is_option_sufficient() == True

    def test_is_option_sufficient(self):
        algo = MinimizationAlgorithm()
        assert algo.is_option_sufficient() == True

    def test_is_loss_and_option_sufficient(self):
        algo = MinimizationAlgorithm()
        assert algo.is_loss_and_option_sufficient() == True
