import numpy as np
import numpy.testing as npt
import pytest

from quara.minimization_algorithm.projected_gradient_descent_with_momentum import (
    ProjectedGradientDescentWithMomentum,
    ProjectedGradientDescentWithMomentumOption,
)
from quara.loss_function.simple_quadratic_loss_function import (
    SimpleQuadraticLossFunction,
)
from quara.objects.composite_system import CompositeSystem
from quara.objects.elemental_system import ElementalSystem
from quara.objects.matrix_basis import get_normalized_pauli_basis
from quara.objects.povm import (
    get_x_povm,
    get_y_povm,
    get_z_povm,
)
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
        seed_data=7,
    )

    return qst, c_sys


class TestProjectedGradientDescentWithMomentumOption:
    def test_access_mode_stopping_criterion_gradient_descent(self):
        # default
        option = ProjectedGradientDescentWithMomentumOption()
        assert (
            option.mode_stopping_criterion_gradient_descent == "single_difference_loss"
        )

        # mode_stopping_criterion_gradient_descent = "single_difference_loss"
        option = ProjectedGradientDescentWithMomentumOption(
            mode_stopping_criterion_gradient_descent="single_difference_loss"
        )
        assert (
            option.mode_stopping_criterion_gradient_descent == "single_difference_loss"
        )

        # mode_stopping_criterion_gradient_descent = "sum_absolute_difference_loss"
        option = ProjectedGradientDescentWithMomentumOption(
            mode_stopping_criterion_gradient_descent="sum_absolute_difference_loss"
        )
        assert (
            option.mode_stopping_criterion_gradient_descent
            == "sum_absolute_difference_loss"
        )

        # mode_stopping_criterion_gradient_descent = "sum_absolute_difference_variable"
        option = ProjectedGradientDescentWithMomentumOption(
            mode_stopping_criterion_gradient_descent="sum_absolute_difference_variable"
        )
        assert (
            option.mode_stopping_criterion_gradient_descent
            == "sum_absolute_difference_variable"
        )

        # mode_stopping_criterion_gradient_descent = "sum_absolute_difference_projected_gradient"
        option = ProjectedGradientDescentWithMomentumOption(
            mode_stopping_criterion_gradient_descent="sum_absolute_difference_projected_gradient"
        )
        assert (
            option.mode_stopping_criterion_gradient_descent
            == "sum_absolute_difference_projected_gradient"
        )

        # unsupported string
        with pytest.raises(ValueError):
            ProjectedGradientDescentWithMomentumOption(
                mode_stopping_criterion_gradient_descent="unsupported_string"
            )

        # Test that "mode_stopping_criterion_gradient_descent" cannot be updated
        with pytest.raises(AttributeError):
            option = ProjectedGradientDescentWithMomentumOption()
            option.mode_stopping_criterion_gradient_descent = "single_difference_loss"

    def test_access_num_history_stopping_criterion_gradient_descent(self):
        # default
        option = ProjectedGradientDescentWithMomentumOption()
        assert option.num_history_stopping_criterion_gradient_descent == 1

        # num_history_stopping_criterion_gradient_descent = 2
        option = ProjectedGradientDescentWithMomentumOption(
            num_history_stopping_criterion_gradient_descent=2
        )
        assert option.num_history_stopping_criterion_gradient_descent == 2

        # not int
        with pytest.raises(ValueError):
            ProjectedGradientDescentWithMomentumOption(
                num_history_stopping_criterion_gradient_descent=1.0
            )

        # <1
        with pytest.raises(ValueError):
            ProjectedGradientDescentWithMomentumOption(
                num_history_stopping_criterion_gradient_descent=0
            )

        # Test that "num_history_stopping_criterion_gradient_descent" cannot be updated
        with pytest.raises(AttributeError):
            option = ProjectedGradientDescentWithMomentumOption()
            option.num_history_stopping_criterion_gradient_descent = 1

    def test_access_mode_proj_order(self):
        # default
        option = ProjectedGradientDescentWithMomentumOption()
        assert option.mode_proj_order == "eq_ineq"

        # eq_ineq
        option = ProjectedGradientDescentWithMomentumOption(mode_proj_order="eq_ineq")
        assert option.mode_proj_order == "eq_ineq"

        # ineq_eq
        option = ProjectedGradientDescentWithMomentumOption(mode_proj_order="ineq_eq")
        assert option.mode_proj_order == "ineq_eq"

        # unsupported value
        with pytest.raises(ValueError):
            ProjectedGradientDescentWithMomentumOption(mode_proj_order="unsupported")

        # Test that "mode_proj_order" cannot be updated
        with pytest.raises(AttributeError):
            option = ProjectedGradientDescentWithMomentumOption()
            option.mode_proj_order = "eq_ineq"


class TestProjectedGradientDescentWithMomentum:
    def test_access_func_proj(self):
        algo = ProjectedGradientDescentWithMomentum()
        assert algo.func_proj is None

        with pytest.raises(AttributeError):
            algo.func_proj = func_proj.proj_to_self

        algo = ProjectedGradientDescentWithMomentum(func_proj.proj_to_self)
        assert algo.func_proj is not None

    def test_set_constraint_from_standard_qt_and_option(self):
        # case1: use func_calc_proj_physical() if on_algo_eq_constraint=True, on_algo_ineq_constraint=True
        qst, _ = get_test_data()
        algo = ProjectedGradientDescentWithMomentum()
        option = ProjectedGradientDescentWithMomentumOption(
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
        algo = ProjectedGradientDescentWithMomentum()
        option = ProjectedGradientDescentWithMomentumOption(
            on_algo_eq_constraint=True, on_algo_ineq_constraint=False
        )
        algo.set_constraint_from_standard_qt_and_option(qst, option)
        var = np.array([2, 1, 1, 1], dtype=np.float64) / np.sqrt(2)
        actual = algo.func_proj(var)
        expected = np.array([1, 1, 1, 1], dtype=np.float64) / np.sqrt(2)
        npt.assert_almost_equal(actual, expected, decimal=14)

        # case3: use func_calc_proj_ineq_constraint() if on_algo_eq_constraint=False, on_algo_ineq_constraint=True
        qst, _ = get_test_data()
        algo = ProjectedGradientDescentWithMomentum()
        option = ProjectedGradientDescentWithMomentumOption(
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
        algo = ProjectedGradientDescentWithMomentum()
        option = ProjectedGradientDescentWithMomentumOption(
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
        algo = ProjectedGradientDescentWithMomentum()
        algo.set_from_loss(loss)
        assert algo.is_loss_sufficient() == True

        # loss is None
        algo = ProjectedGradientDescentWithMomentum()
        assert algo.is_loss_sufficient() == False

        # loss is None
        algo = ProjectedGradientDescentWithMomentum()
        algo.set_from_loss(None)
        assert algo.is_loss_sufficient() == False

        # loss.on_value is False
        var_ref = np.array([1, 0, 0, 0], dtype=np.float64) / np.sqrt(2)
        loss = SimpleQuadraticLossFunction(var_ref)
        loss._on_value = False
        algo = ProjectedGradientDescentWithMomentum()
        algo.set_from_loss(loss)
        assert algo.is_loss_sufficient() == False

        # loss.on_gradient is False
        var_ref = np.array([1, 0, 0, 0], dtype=np.float64) / np.sqrt(2)
        loss = SimpleQuadraticLossFunction(var_ref)
        loss._on_gradient = False
        algo = ProjectedGradientDescentWithMomentum()
        algo.set_from_loss(loss)
        assert algo.is_loss_sufficient() == False

    def test_is_option_sufficient(self):
        # option is not None
        algo = ProjectedGradientDescentWithMomentum()
        algo_option = ProjectedGradientDescentWithMomentumOption()
        algo.set_from_option(algo_option)
        assert algo.is_option_sufficient() == True

        # option is None
        algo = ProjectedGradientDescentWithMomentum()
        assert algo.is_option_sufficient() == False

        # option is None
        algo = ProjectedGradientDescentWithMomentum()
        algo.set_from_option(None)
        assert algo.is_option_sufficient() == False

        # option.r is None
        algo = ProjectedGradientDescentWithMomentum()
        algo_option = ProjectedGradientDescentWithMomentumOption(r=None)
        algo.set_from_option(algo_option)
        assert algo.is_option_sufficient() == True

        # option.r is non-positive
        algo = ProjectedGradientDescentWithMomentum()
        algo_option = ProjectedGradientDescentWithMomentumOption(r=0)
        algo.set_from_option(algo_option)
        assert algo.is_option_sufficient() == False

        # option.moment_0 is None
        algo = ProjectedGradientDescentWithMomentum()
        algo_option = ProjectedGradientDescentWithMomentumOption(moment_0=None)
        algo.set_from_option(algo_option)
        assert algo.is_option_sufficient() == True

        # option.moment_0 is non-positive
        algo = ProjectedGradientDescentWithMomentum()
        algo_option = ProjectedGradientDescentWithMomentumOption(moment_0=0)
        algo.set_from_option(algo_option)
        assert algo.is_option_sufficient() == True

        # option.eps is None
        algo = ProjectedGradientDescentWithMomentum()
        algo_option = ProjectedGradientDescentWithMomentumOption(eps=None)
        algo.set_from_option(algo_option)
        assert algo.is_option_sufficient() == True

        # option.eps is non-positive
        algo = ProjectedGradientDescentWithMomentum()
        algo_option = ProjectedGradientDescentWithMomentumOption(eps=0)
        algo.set_from_option(algo_option)
        assert algo.is_option_sufficient() == False

    def test_is_loss_and_option_sufficient(self):
        # option or loss is None
        algo = ProjectedGradientDescentWithMomentum()
        assert algo.is_loss_and_option_sufficient() == True

        # option.var_start is None
        algo = ProjectedGradientDescentWithMomentum()
        algo_option = ProjectedGradientDescentWithMomentumOption()
        algo.set_from_option(algo_option)
        assert algo.is_loss_and_option_sufficient() == True

        # self.option.var_start.shape[0] != self.loss.num_var
        var_ref1 = np.array([0, 0, 0], dtype=np.float64) / np.sqrt(2)
        algo_option = ProjectedGradientDescentWithMomentumOption(var_start=var_ref1)
        var_ref2 = np.array([1, 0, 0, 0], dtype=np.float64) / np.sqrt(2)
        loss = SimpleQuadraticLossFunction(var_ref2)

        algo = ProjectedGradientDescentWithMomentum()
        algo.set_from_option(algo_option)
        algo.set_from_loss(loss)
        assert algo.is_loss_and_option_sufficient() == False

    def test_optimize_value_error(self):
        algo_option = ProjectedGradientDescentWithMomentumOption()

        # loss.on_value is False
        var_ref = np.array([1, 0, 0, 0], dtype=np.float64) / np.sqrt(2)
        loss = SimpleQuadraticLossFunction(var_ref)
        loss._on_value = False
        algo = ProjectedGradientDescentWithMomentum()
        algo.set_from_loss(loss)
        with pytest.raises(ValueError):
            algo.optimize(loss, None, algo_option)

        # loss.on_gradient is False
        var_ref = np.array([1, 0, 0, 0], dtype=np.float64) / np.sqrt(2)
        loss = SimpleQuadraticLossFunction(var_ref)
        loss._on_gradient = False
        algo = ProjectedGradientDescentWithMomentum()
        algo.set_from_loss(loss)
        with pytest.raises(ValueError):
            algo.optimize(loss, None, algo_option)
