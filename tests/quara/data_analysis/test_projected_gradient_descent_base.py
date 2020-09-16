import numpy as np
import numpy.testing as npt

from quara.data_analysis.quadratic_loss_function import (
    QuadraticLossFunction,
    QuadraticLossFunctionOption,
)
from quara.data_analysis.projected_gradient_descent_base import (
    ProjectedGradientDescentBase,
    ProjectedGradientDescentBaseOption,
)
from quara.math import func_proj


class TestProjectedGradientDescentBase:
    def test_optimize_with_proj_to_self(self):
        proj = func_proj.proj_to_self()
        loss_option = QuadraticLossFunctionOption()
        algo = ProjectedGradientDescentBase()

        var_ref = np.array([1, 1], dtype=np.float64)
        loss = QuadraticLossFunction(var_ref)

        var_starts = [
            np.array([3, 3], dtype=np.float64),
            np.array([-3, 3], dtype=np.float64),
            np.array([-3, -3], dtype=np.float64),
            np.array([3, -3], dtype=np.float64),
        ]
        expected = np.array([1, 1], dtype=np.float64)

        for var_start in var_starts:
            algo_option = ProjectedGradientDescentBaseOption(proj, var_start)
            actual = algo.optimize(loss, loss_option, algo_option)
            npt.assert_almost_equal(actual, expected, decimal=7)

    def test_optimize_with_proj_to_hyperplane(self):
        loss_option = QuadraticLossFunctionOption()
        algo = ProjectedGradientDescentBase()

        # case1: var_ref is multiple of var_a
        var_a = np.array([2, 0], dtype=np.float64)
        proj = func_proj.proj_to_hyperplane(var_a)
        var_ref = np.array([1, 0], dtype=np.float64)
        loss = QuadraticLossFunction(var_ref)

        var_starts = [
            np.array([2, 1], dtype=np.float64),
            np.array([2, 0], dtype=np.float64),
            np.array([2, -1], dtype=np.float64),
        ]
        expected = np.array([2, 0], dtype=np.float64)

        for var_start in var_starts:
            algo_option = ProjectedGradientDescentBaseOption(proj, var_start)
            actual = algo.optimize(loss, loss_option, algo_option)
            npt.assert_almost_equal(actual, expected, decimal=7)

        # case2: var_ref is NOT multiple of var_a
        var_a = np.array([1, 1], dtype=np.float64)
        proj = func_proj.proj_to_hyperplane(var_a)
        var_ref = np.array([1, 0], dtype=np.float64)
        loss = QuadraticLossFunction(var_ref)

        var_starts = [
            np.array([2, 0], dtype=np.float64),
            np.array([1, 1], dtype=np.float64),
            np.array([0, 2], dtype=np.float64),
        ]
        expected = np.array([1.5, 0.5], dtype=np.float64)

        for var_start in var_starts:
            algo_option = ProjectedGradientDescentBaseOption(proj, var_start)
            actual = algo.optimize(loss, loss_option, algo_option)
            npt.assert_almost_equal(actual, expected, decimal=6)

        # case3: var_ref is NOT multiple of var_a
        var_a = np.array([2, 0], dtype=np.float64)
        proj = func_proj.proj_to_hyperplane(var_a)
        var_ref = np.array([1, 1], dtype=np.float64)
        loss = QuadraticLossFunction(var_ref)

        var_starts = [
            np.array([2, 1], dtype=np.float64),
            np.array([2, 0], dtype=np.float64),
            np.array([2, -1], dtype=np.float64),
        ]
        expected = np.array([2, 1], dtype=np.float64)

        for var_start in var_starts:
            algo_option = ProjectedGradientDescentBaseOption(proj, var_start)
            actual = algo.optimize(loss, loss_option, algo_option)
            npt.assert_almost_equal(actual, expected, decimal=7)

    def test_optimize_with_proj_to_nonnegative(self):
        proj = func_proj.proj_to_nonnegative()
        loss_option = QuadraticLossFunctionOption()
        algo = ProjectedGradientDescentBase()

        # case1: var_ref is inside constraint.
        var_ref = np.array([1, 1], dtype=np.float64)
        loss = QuadraticLossFunction(var_ref)

        var_starts = [
            np.array([3, 3], dtype=np.float64),
            np.array([-3, 3], dtype=np.float64),
            np.array([-3, -3], dtype=np.float64),
            np.array([3, -3], dtype=np.float64),
        ]
        expected = np.array([1, 1], dtype=np.float64)

        for var_start in var_starts:
            algo_option = ProjectedGradientDescentBaseOption(proj, var_start)
            actual = algo.optimize(loss, loss_option, algo_option)
            npt.assert_almost_equal(actual, expected, decimal=6)

        # case2: var_ref is outside constraint.
        var_ref = np.array([-1, 0], dtype=np.float64)
        loss = QuadraticLossFunction(var_ref)

        var_starts = [
            np.array([0, 0], dtype=np.float64),
            np.array([1, 0], dtype=np.float64),
            np.array([1, 1], dtype=np.float64),
            np.array([0, 0], dtype=np.float64),
            np.array([2, 3], dtype=np.float64),
        ]
        expected = np.array([0, 0], dtype=np.float64)

        for var_start in var_starts:
            algo_option = ProjectedGradientDescentBaseOption(proj, var_start)
            actual = algo.optimize(loss, loss_option, algo_option)
            npt.assert_almost_equal(actual, expected, decimal=15)

        # case3: var_ref is outside constraint.
        var_ref = np.array([1, -1], dtype=np.float64)
        loss = QuadraticLossFunction(var_ref)

        var_starts = [
            np.array([0, 0], dtype=np.float64),
            np.array([1, 0], dtype=np.float64),
            np.array([1, 1], dtype=np.float64),
            np.array([0, 0], dtype=np.float64),
            np.array([2, 3], dtype=np.float64),
        ]
        expected = np.array([1, 0], dtype=np.float64)

        for var_start in var_starts:
            algo_option = ProjectedGradientDescentBaseOption(proj, var_start)
            actual = algo.optimize(loss, loss_option, algo_option)
            npt.assert_almost_equal(actual, expected, decimal=6)

        # case3: var_ref is in boundary of constraint.
        var_ref = np.array([1, 0], dtype=np.float64)
        loss = QuadraticLossFunction(var_ref)

        var_starts = [
            np.array([0, 0], dtype=np.float64),
            np.array([1, 0], dtype=np.float64),
            np.array([1, 1], dtype=np.float64),
            np.array([0, 0], dtype=np.float64),
            np.array([2, 3], dtype=np.float64),
        ]
        expected = np.array([1, 0], dtype=np.float64)

        for var_start in var_starts:
            algo_option = ProjectedGradientDescentBaseOption(proj, var_start)
            actual = algo.optimize(loss, loss_option, algo_option)
            npt.assert_almost_equal(actual, expected, decimal=6)
