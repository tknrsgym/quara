import numpy as np
import numpy.testing as npt
import pytest

from quara.loss_function.simple_quadratic_loss_function import (
    SimpleQuadraticLossFunction,
    SimpleQuadraticLossFunctionOption,
)
from quara.minimization_algorithm.minimization_algorithm import (
    MinimizationAlgorithm,
    MinimizationAlgorithmOption,
)


class TestMinimizationAlgorithmOption:
    def test_access_on_algo_eq_constraint(self):
        # default = True
        actual = MinimizationAlgorithmOption()
        assert actual.on_algo_eq_constraint == True

        # on_algo_eq_constraint = False
        actual = MinimizationAlgorithmOption(on_algo_eq_constraint=False)
        assert actual.on_algo_eq_constraint == False

        # Test that the property cannot be updated
        with pytest.raises(AttributeError):
            actual.on_algo_eq_constraint = True

    def test_access_on_algo_ineq_constraint(self):
        # default = True
        actual = MinimizationAlgorithmOption()
        assert actual.on_algo_ineq_constraint == True

        # on_algo_ineq_constraint = False
        actual = MinimizationAlgorithmOption(on_algo_ineq_constraint=False)
        assert actual.on_algo_ineq_constraint == False

        # Test that the property cannot be updated
        with pytest.raises(AttributeError):
            actual.on_algo_ineq_constraint = True


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
        var_ref = np.array([1, 0, 0, 0], dtype=np.float64) / np.sqrt(2)
        loss = SimpleQuadraticLossFunction(var_ref)

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
