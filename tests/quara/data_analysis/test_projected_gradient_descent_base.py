import numpy as np


from quara.data_analysis.quadratic_loss_function import (
    QuadraticLossFunction,
    QuadraticLossFunctionOption,
)
from quara.data_analysis.projected_gradient_descent_base import (
    ProjectedGradientDescentBase,
    ProjectedGradientDescentBaseOption,
)


class TestProjectedGradientDescentBase:
    def test_optimize(self):
        def _func_proj_id(var: np.array, loss_function_option=None) -> np.array:
            return var

        def _func_proj_a(var: np.array, loss_function_option=None) -> np.array:
            proj_value = var - np.dot(var_a, var) * var_a / np.dot(var_a, var_a) + var_a
            return proj_value

        def _func_proj_zero(var: np.array, loss_function_option=None) -> np.array:
            zero = np.zeros(var.shape)
            proj_value = np.maximum(var, zero)
            return proj_value

        var_ref = np.array([1, 1], dtype=np.float64)
        loss = QuadraticLossFunction(var_ref)

        var_a = np.array([2, 0], dtype=np.float64)
        loss_option = QuadraticLossFunctionOption(var_a)

        algo = ProjectedGradientDescentBase()

        var_starts = [
            np.array([3, 3], dtype=np.float64),
            np.array([-3, 3], dtype=np.float64),
            np.array([-3, -3], dtype=np.float64),
            np.array([3, -3], dtype=np.float64),
        ]

        for var_start in var_starts:
            algo_option = ProjectedGradientDescentBaseOption(_func_proj_a, var_start)
            algo.optimize(loss, loss_option, algo_option)

        assert False
