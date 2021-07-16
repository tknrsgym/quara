import numpy as np
import numpy.testing as npt
import pytest

from quara.minimization_algorithm.projected_gradient_descent_backtracking import (
    ProjectedGradientDescentBacktracking,
    ProjectedGradientDescentBacktrackingOption,
)
from quara.loss_function.simple_quadratic_loss_function import (
    SimpleQuadraticLossFunction,
    SimpleQuadraticLossFunctionOption,
)
from quara.objects.composite_system import CompositeSystem
from quara.objects.elemental_system import ElementalSystem
from quara.objects.matrix_basis import get_normalized_pauli_basis
from quara.objects.povm import (
    get_x_povm,
    get_y_povm,
    get_z_povm,
)
from quara.objects.state import convert_var_to_state, get_z0_1q
from quara.protocol.qtomography.standard.standard_qst import StandardQst
from quara.math import func_proj


def get_test_data(on_para_eq_constraint=False):
    e_sys = ElementalSystem(0, get_normalized_pauli_basis())
    c_sys = CompositeSystem([e_sys])

    povm_x = get_x_povm(c_sys)
    povm_y = get_y_povm(c_sys)
    povm_z = get_z_povm(c_sys)
    povms = [povm_x, povm_y, povm_z]

    qst = StandardQst(
        povms,
        on_para_eq_constraint=on_para_eq_constraint,
        seed=7,
    )

    return qst, c_sys


class TestProjectedGradientDescentBacktrackingOption:
    def test_access_mode_stopping_criterion_gradient_descent(self):
        # default
        option = ProjectedGradientDescentBacktrackingOption()
        assert (
            option.mode_stopping_criterion_gradient_descent == "single_difference_loss"
        )

        # mode_stopping_criterion_gradient_descent = "single_difference_loss"
        option = ProjectedGradientDescentBacktrackingOption(
            mode_stopping_criterion_gradient_descent="single_difference_loss"
        )
        assert (
            option.mode_stopping_criterion_gradient_descent == "single_difference_loss"
        )

        # mode_stopping_criterion_gradient_descent = "sum_absolute_difference_loss"
        option = ProjectedGradientDescentBacktrackingOption(
            mode_stopping_criterion_gradient_descent="sum_absolute_difference_loss"
        )
        assert (
            option.mode_stopping_criterion_gradient_descent
            == "sum_absolute_difference_loss"
        )

        # mode_stopping_criterion_gradient_descent = "sum_absolute_difference_variable"
        option = ProjectedGradientDescentBacktrackingOption(
            mode_stopping_criterion_gradient_descent="sum_absolute_difference_variable"
        )
        assert (
            option.mode_stopping_criterion_gradient_descent
            == "sum_absolute_difference_variable"
        )

        # mode_stopping_criterion_gradient_descent = "sum_absolute_difference_projected_gradient"
        option = ProjectedGradientDescentBacktrackingOption(
            mode_stopping_criterion_gradient_descent="sum_absolute_difference_projected_gradient"
        )
        assert (
            option.mode_stopping_criterion_gradient_descent
            == "sum_absolute_difference_projected_gradient"
        )

        # unsupported string
        with pytest.raises(ValueError):
            ProjectedGradientDescentBacktrackingOption(
                mode_stopping_criterion_gradient_descent="unsupported_string"
            )

        # Test that "mode_stopping_criterion_gradient_descent" cannot be updated
        with pytest.raises(AttributeError):
            option = ProjectedGradientDescentBacktrackingOption()
            option.mode_stopping_criterion_gradient_descent = "single_difference_loss"

    def test_access_num_history_stopping_criterion_gradient_descent(self):
        # default
        option = ProjectedGradientDescentBacktrackingOption()
        assert option.num_history_stopping_criterion_gradient_descent == 1

        # num_history_stopping_criterion_gradient_descent = 2
        option = ProjectedGradientDescentBacktrackingOption(
            num_history_stopping_criterion_gradient_descent=2
        )
        assert option.num_history_stopping_criterion_gradient_descent == 2

        # not int
        with pytest.raises(ValueError):
            ProjectedGradientDescentBacktrackingOption(
                num_history_stopping_criterion_gradient_descent=1.0
            )

        # <1
        with pytest.raises(ValueError):
            ProjectedGradientDescentBacktrackingOption(
                num_history_stopping_criterion_gradient_descent=0
            )

        # Test that "num_history_stopping_criterion_gradient_descent" cannot be updated
        with pytest.raises(AttributeError):
            option = ProjectedGradientDescentBacktrackingOption()
            option.num_history_stopping_criterion_gradient_descent = 1

    def test_access_mode_proj_order(self):
        # default
        option = ProjectedGradientDescentBacktrackingOption()
        assert option.mode_proj_order == "eq_ineq"

        # eq_ineq
        option = ProjectedGradientDescentBacktrackingOption(mode_proj_order="eq_ineq")
        assert option.mode_proj_order == "eq_ineq"

        # ineq_eq
        option = ProjectedGradientDescentBacktrackingOption(mode_proj_order="ineq_eq")
        assert option.mode_proj_order == "ineq_eq"

        # unsupported value
        with pytest.raises(ValueError):
            ProjectedGradientDescentBacktrackingOption(mode_proj_order="unsupported")

        # Test that "mode_proj_order" cannot be updated
        with pytest.raises(AttributeError):
            option = ProjectedGradientDescentBacktrackingOption()
            option.mode_proj_order = "eq_ineq"


class TestProjectedGradientDescentBase:
    def test_access_func_proj(self):
        algo = ProjectedGradientDescentBacktracking()
        assert algo.func_proj is None

        with pytest.raises(AttributeError):
            algo.func_proj = func_proj.proj_to_self

        algo = ProjectedGradientDescentBacktracking(func_proj.proj_to_self)
        assert algo.func_proj is not None

    def test_set_constraint_from_standard_qt_and_option(self):
        # case1: use func_calc_proj_physical() if on_algo_eq_constraint=True, on_algo_ineq_constraint=True
        qst, _ = get_test_data()
        algo = ProjectedGradientDescentBacktracking()
        option = ProjectedGradientDescentBacktrackingOption(
            on_algo_eq_constraint=True, on_algo_ineq_constraint=True
        )
        algo.set_constraint_from_standard_qt_and_option(qst, option)
        var = np.array([2, 1, 1, 1], dtype=np.float64) / np.sqrt(2)
        actual = algo.func_proj(var)
        expected = np.array(
            [1, 1 / np.sqrt(3), 1 / np.sqrt(3), 1 / np.sqrt(3)], dtype=np.float64
        ) / np.sqrt(2)
        npt.assert_almost_equal(actual, expected, decimal=7)

        # case2: use func_calc_proj_physical() if on_algo_eq_constraint=True, on_algo_ineq_constraint=False
        qst, _ = get_test_data()
        algo = ProjectedGradientDescentBacktracking()
        option = ProjectedGradientDescentBacktrackingOption(
            on_algo_eq_constraint=True, on_algo_ineq_constraint=False
        )
        algo.set_constraint_from_standard_qt_and_option(qst, option)
        var = np.array([2, 1, 1, 1], dtype=np.float64) / np.sqrt(2)
        actual = algo.func_proj(var)
        expected = np.array([1, 1, 1, 1], dtype=np.float64) / np.sqrt(2)
        npt.assert_almost_equal(actual, expected, decimal=14)

        # case3: use func_calc_proj_ineq_constraint() if on_algo_eq_constraint=False, on_algo_ineq_constraint=True
        qst, _ = get_test_data()
        algo = ProjectedGradientDescentBacktracking()
        option = ProjectedGradientDescentBacktrackingOption(
            on_algo_eq_constraint=False, on_algo_ineq_constraint=True
        )
        algo.set_constraint_from_standard_qt_and_option(qst, option)
        var = np.array([1, 1.1, 0, 0], dtype=np.float64) / np.sqrt(2)
        actual = algo.func_proj(var)
        expected = np.array(
            [7.42462120245875e-01, 7.42462120245875e-01, 0, 0], dtype=np.float64
        )
        npt.assert_almost_equal(actual, expected, decimal=14)

        # case4: use proj_to_self() if on_algo_eq_constraint=False, on_algo_ineq_constraint=False
        qst, _ = get_test_data()
        algo = ProjectedGradientDescentBacktracking()
        option = ProjectedGradientDescentBacktrackingOption(
            on_algo_eq_constraint=False, on_algo_ineq_constraint=False
        )
        algo.set_constraint_from_standard_qt_and_option(qst, option)
        var = np.array([2, 1, 1, 1], dtype=np.float64) / np.sqrt(2)
        actual = algo.func_proj(var)
        npt.assert_almost_equal(actual, var, decimal=14)

    def test_is_loss_sufficient(self):
        # loss is not None
        var_ref = np.array([1, 0, 0, 0], dtype=np.float64) / np.sqrt(2)
        loss = SimpleQuadraticLossFunction(var_ref)
        algo = ProjectedGradientDescentBacktracking()
        algo.set_from_loss(loss)
        assert algo.is_loss_sufficient() == True

        # loss is None
        algo = ProjectedGradientDescentBacktracking()
        assert algo.is_loss_sufficient() == False

        # loss is None
        algo = ProjectedGradientDescentBacktracking()
        algo.set_from_loss(None)
        assert algo.is_loss_sufficient() == False

        # loss.on_value is False
        var_ref = np.array([1, 0, 0, 0], dtype=np.float64) / np.sqrt(2)
        loss = SimpleQuadraticLossFunction(var_ref)
        loss._on_value = False
        algo = ProjectedGradientDescentBacktracking()
        algo.set_from_loss(loss)
        assert algo.is_loss_sufficient() == False

        # loss.on_gradient is False
        var_ref = np.array([1, 0, 0, 0], dtype=np.float64) / np.sqrt(2)
        loss = SimpleQuadraticLossFunction(var_ref)
        loss._on_gradient = False
        algo = ProjectedGradientDescentBacktracking()
        algo.set_from_loss(loss)
        assert algo.is_loss_sufficient() == False

    def test_is_option_sufficient(self):
        # option is not None
        algo = ProjectedGradientDescentBacktracking()
        algo_option = ProjectedGradientDescentBacktrackingOption()
        algo.set_from_option(algo_option)
        assert algo.is_option_sufficient() == True

        # option is None
        algo = ProjectedGradientDescentBacktracking()
        assert algo.is_option_sufficient() == False

        # option is None
        algo = ProjectedGradientDescentBacktracking()
        algo.set_from_option(None)
        assert algo.is_option_sufficient() == False

        # option.mu is None
        algo = ProjectedGradientDescentBacktracking()
        algo_option = ProjectedGradientDescentBacktrackingOption(mu=None)
        algo.set_from_option(algo_option)
        assert algo.is_option_sufficient() == True

        # option.mu is non-positive
        algo = ProjectedGradientDescentBacktracking()
        algo_option = ProjectedGradientDescentBacktrackingOption(mu=0)
        algo.set_from_option(algo_option)
        assert algo.is_option_sufficient() == False

        # option.gamma is None
        algo = ProjectedGradientDescentBacktracking()
        algo_option = ProjectedGradientDescentBacktrackingOption(gamma=None)
        algo.set_from_option(algo_option)
        assert algo.is_option_sufficient() == False

        # option.gamma is non-positive
        algo = ProjectedGradientDescentBacktracking()
        algo_option = ProjectedGradientDescentBacktrackingOption(gamma=0)
        algo.set_from_option(algo_option)
        assert algo.is_option_sufficient() == False

        # option.eps is None
        algo = ProjectedGradientDescentBacktracking()
        algo_option = ProjectedGradientDescentBacktrackingOption(eps=None)
        algo.set_from_option(algo_option)
        assert algo.is_option_sufficient() == True

        # option.eps is non-positive
        algo = ProjectedGradientDescentBacktracking()
        algo_option = ProjectedGradientDescentBacktrackingOption(eps=0)
        algo.set_from_option(algo_option)
        assert algo.is_option_sufficient() == False

    def test_is_loss_and_option_sufficient(self):
        # option or loss is None
        algo = ProjectedGradientDescentBacktracking()
        assert algo.is_loss_and_option_sufficient() == True

        # option.var_start is None
        algo = ProjectedGradientDescentBacktracking()
        algo_option = ProjectedGradientDescentBacktrackingOption()
        algo.set_from_option(algo_option)
        assert algo.is_loss_and_option_sufficient() == True

        # self.option.var_start.shape[0] != self.loss.num_var
        var_ref1 = np.array([0, 0, 0], dtype=np.float64) / np.sqrt(2)
        algo_option = ProjectedGradientDescentBacktrackingOption(var_start=var_ref1)
        var_ref2 = np.array([1, 0, 0, 0], dtype=np.float64) / np.sqrt(2)
        loss = SimpleQuadraticLossFunction(var_ref2)

        algo = ProjectedGradientDescentBacktracking()
        algo.set_from_option(algo_option)
        algo.set_from_loss(loss)
        assert algo.is_loss_and_option_sufficient() == False

    def test_optimize_with_proj_to_self(self):
        loss_option = SimpleQuadraticLossFunctionOption()
        var_ref = np.array([1, 1], dtype=np.float64)
        loss = SimpleQuadraticLossFunction(var_ref)

        proj = func_proj.proj_to_self()
        algo = ProjectedGradientDescentBacktracking(proj)

        var_starts = [
            np.array([3, 3], dtype=np.float64),
            np.array([-3, 3], dtype=np.float64),
            np.array([-3, -3], dtype=np.float64),
            np.array([3, -3], dtype=np.float64),
        ]
        expected = np.array([1, 1], dtype=np.float64)

        for var_start in var_starts:
            algo_option = ProjectedGradientDescentBacktrackingOption(
                var_start=var_start
            )
            actual = algo.optimize(loss, loss_option, algo_option)
            npt.assert_almost_equal(actual.value, expected, decimal=7)

    def test_optimize_with_proj_to_hyperplane(self):
        loss_option = SimpleQuadraticLossFunctionOption()

        # case1: var_ref is multiple of var_a
        var_ref = np.array([1, 0], dtype=np.float64)
        loss = SimpleQuadraticLossFunction(var_ref)

        var_a = np.array([2, 0], dtype=np.float64)
        proj = func_proj.proj_to_hyperplane(var_a)
        algo = ProjectedGradientDescentBacktracking(proj)

        var_starts = [
            np.array([2, 1], dtype=np.float64),
            np.array([2, 0], dtype=np.float64),
            np.array([2, -1], dtype=np.float64),
        ]
        expected = np.array([2, 0], dtype=np.float64)

        for var_start in var_starts:
            algo_option = ProjectedGradientDescentBacktrackingOption(
                var_start=var_start
            )
            actual = algo.optimize(loss, loss_option, algo_option)
            npt.assert_almost_equal(actual.value, expected, decimal=7)

        # case2: var_ref is NOT multiple of var_a
        var_ref = np.array([1, 0], dtype=np.float64)
        loss = SimpleQuadraticLossFunction(var_ref)

        var_a = np.array([1, 1], dtype=np.float64)
        proj = func_proj.proj_to_hyperplane(var_a)
        algo = ProjectedGradientDescentBacktracking(proj)

        var_starts = [
            np.array([2, 0], dtype=np.float64),
            np.array([1, 1], dtype=np.float64),
            np.array([0, 2], dtype=np.float64),
        ]
        expected = np.array([1.5, 0.5], dtype=np.float64)

        for var_start in var_starts:
            algo_option = ProjectedGradientDescentBacktrackingOption(
                var_start=var_start
            )
            actual = algo.optimize(loss, loss_option, algo_option)
            npt.assert_almost_equal(actual.value, expected, decimal=6)

        # case3: var_ref is NOT multiple of var_a
        var_ref = np.array([1, 1], dtype=np.float64)
        loss = SimpleQuadraticLossFunction(var_ref)

        var_a = np.array([2, 0], dtype=np.float64)
        proj = func_proj.proj_to_hyperplane(var_a)
        algo = ProjectedGradientDescentBacktracking(proj)

        var_starts = [
            np.array([2, 1], dtype=np.float64),
            np.array([2, 0], dtype=np.float64),
            np.array([2, -1], dtype=np.float64),
        ]
        expected = np.array([2, 1], dtype=np.float64)

        for var_start in var_starts:
            algo_option = ProjectedGradientDescentBacktrackingOption(
                var_start=var_start
            )
            actual = algo.optimize(loss, loss_option, algo_option)
            npt.assert_almost_equal(actual.value, expected, decimal=7)

    def test_optimize_with_proj_to_nonnegative(self):
        loss_option = SimpleQuadraticLossFunctionOption()

        proj = func_proj.proj_to_nonnegative()
        algo = ProjectedGradientDescentBacktracking(proj)

        # case1: var_ref is inside constraint.
        var_ref = np.array([1, 1], dtype=np.float64)
        loss = SimpleQuadraticLossFunction(var_ref)

        var_starts = [
            np.array([3, 3], dtype=np.float64),
            np.array([-3, 3], dtype=np.float64),
            np.array([-3, -3], dtype=np.float64),
            np.array([3, -3], dtype=np.float64),
        ]
        expected = np.array([1, 1], dtype=np.float64)

        for var_start in var_starts:
            algo_option = ProjectedGradientDescentBacktrackingOption(
                var_start=var_start
            )
            actual = algo.optimize(loss, loss_option, algo_option)
            npt.assert_almost_equal(actual.value, expected, decimal=6)

        # case2: var_ref is outside constraint.
        var_ref = np.array([-1, 0], dtype=np.float64)
        loss = SimpleQuadraticLossFunction(var_ref)

        var_starts = [
            np.array([0, 0], dtype=np.float64),
            np.array([1, 0], dtype=np.float64),
            np.array([1, 1], dtype=np.float64),
            np.array([0, 0], dtype=np.float64),
            np.array([2, 3], dtype=np.float64),
        ]
        expected = np.array([0, 0], dtype=np.float64)

        for var_start in var_starts:
            algo_option = ProjectedGradientDescentBacktrackingOption(
                var_start=var_start
            )
            actual = algo.optimize(loss, loss_option, algo_option)
            npt.assert_almost_equal(actual.value, expected, decimal=15)

        # case3: var_ref is outside constraint.
        var_ref = np.array([1, -1], dtype=np.float64)
        loss = SimpleQuadraticLossFunction(var_ref)

        var_starts = [
            np.array([0, 0], dtype=np.float64),
            np.array([1, 0], dtype=np.float64),
            np.array([1, 1], dtype=np.float64),
            np.array([0, 0], dtype=np.float64),
            np.array([2, 3], dtype=np.float64),
        ]
        expected = np.array([1, 0], dtype=np.float64)

        for var_start in var_starts:
            algo_option = ProjectedGradientDescentBacktrackingOption(
                var_start=var_start
            )
            actual = algo.optimize(loss, loss_option, algo_option)
            npt.assert_almost_equal(actual.value, expected, decimal=6)

        # case3: var_ref is in boundary of constraint.
        var_ref = np.array([1, 0], dtype=np.float64)
        loss = SimpleQuadraticLossFunction(var_ref)

        var_starts = [
            np.array([0, 0], dtype=np.float64),
            np.array([1, 0], dtype=np.float64),
            np.array([1, 1], dtype=np.float64),
            np.array([0, 0], dtype=np.float64),
            np.array([2, 3], dtype=np.float64),
        ]
        expected = np.array([1, 0], dtype=np.float64)

        for var_start in var_starts:
            algo_option = ProjectedGradientDescentBacktrackingOption(
                var_start=var_start
            )
            actual = algo.optimize(loss, loss_option, algo_option)
            npt.assert_almost_equal(actual.value, expected, decimal=6)

    def test_optimize_max_iteration(self):
        loss_option = SimpleQuadraticLossFunctionOption()

        var_ref = np.array([1, 1], dtype=np.float64)
        loss = SimpleQuadraticLossFunction(var_ref)

        proj = func_proj.proj_to_self()
        algo = ProjectedGradientDescentBacktracking(proj)

        var_start = np.array([3, 3], dtype=np.float64)
        expected = np.array([1, 1], dtype=np.float64)

        algo_option = ProjectedGradientDescentBacktrackingOption(var_start=var_start)
        actual = algo.optimize(
            loss,
            loss_option,
            algo_option,
            max_iteration=5,
            on_iteration_history=True,
        )

        assert actual.k == 5

    def test_optimize_on_iteration_history(self):
        loss_option = SimpleQuadraticLossFunctionOption()

        var_ref = np.array([1, 1], dtype=np.float64)
        loss = SimpleQuadraticLossFunction(var_ref)

        proj = func_proj.proj_to_self()
        algo = ProjectedGradientDescentBacktracking(proj)

        var_start = np.array([3, 3], dtype=np.float64)
        expected = np.array([1, 1], dtype=np.float64)

        algo_option = ProjectedGradientDescentBacktrackingOption(var_start=var_start)
        actual = algo.optimize(
            loss, loss_option, algo_option, on_iteration_history=True
        )

        npt.assert_almost_equal(actual.value, expected, decimal=7)
        assert actual.k == 7
        assert len(actual.fx) == actual.k + 1
        assert len(actual.x) == actual.k + 1
        assert len(actual.y) == actual.k
        assert len(actual.alpha) == actual.k
        assert type(actual.computation_time) == float

    def test_optimize_value_error(self):
        # loss.on_value is False
        var_ref = np.array([1, 0, 0, 0], dtype=np.float64) / np.sqrt(2)
        loss = SimpleQuadraticLossFunction(var_ref)
        loss._on_value = False
        algo = ProjectedGradientDescentBacktracking()
        algo.set_from_loss(loss)
        with pytest.raises(ValueError):
            algo.optimize(loss, None, None)

        # loss.on_gradient is False
        var_ref = np.array([1, 0, 0, 0], dtype=np.float64) / np.sqrt(2)
        loss = SimpleQuadraticLossFunction(var_ref)
        loss._on_gradient = False
        algo = ProjectedGradientDescentBacktracking()
        algo.set_from_loss(loss)
        with pytest.raises(ValueError):
            algo.optimize(loss, None, None)
