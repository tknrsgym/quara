import numpy as np
import numpy.testing as npt
import pytest

from quara.minimization_algorithm.projected_fast_iterative_shrinkage_thresholding_algorithm import (
    ProjectedFastIterativeShrinkageThresholdingAlgorithm,
    ProjectedFastIterativeShrinkageThresholdingAlgorithmOption,
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


class TestProjectedFastIterativeShrinkageThresholdingAlgorithmOption:
    def test_access_mode_stopping_criterion_gradient_descent(self):
        # default
        option = ProjectedFastIterativeShrinkageThresholdingAlgorithmOption()
        assert (
            option.mode_stopping_criterion_gradient_descent == "single_difference_loss"
        )

        # mode_stopping_criterion_gradient_descent = "single_difference_loss"
        option = ProjectedFastIterativeShrinkageThresholdingAlgorithmOption(
            mode_stopping_criterion_gradient_descent="single_difference_loss"
        )
        assert (
            option.mode_stopping_criterion_gradient_descent == "single_difference_loss"
        )

        # mode_stopping_criterion_gradient_descent = "sum_absolute_difference_loss"
        option = ProjectedFastIterativeShrinkageThresholdingAlgorithmOption(
            mode_stopping_criterion_gradient_descent="sum_absolute_difference_loss"
        )
        assert (
            option.mode_stopping_criterion_gradient_descent
            == "sum_absolute_difference_loss"
        )

        # mode_stopping_criterion_gradient_descent = "sum_absolute_difference_variable"
        option = ProjectedFastIterativeShrinkageThresholdingAlgorithmOption(
            mode_stopping_criterion_gradient_descent="sum_absolute_difference_variable"
        )
        assert (
            option.mode_stopping_criterion_gradient_descent
            == "sum_absolute_difference_variable"
        )

        # mode_stopping_criterion_gradient_descent = "sum_absolute_difference_projected_gradient"
        option = ProjectedFastIterativeShrinkageThresholdingAlgorithmOption(
            mode_stopping_criterion_gradient_descent="sum_absolute_difference_projected_gradient"
        )
        assert (
            option.mode_stopping_criterion_gradient_descent
            == "sum_absolute_difference_projected_gradient"
        )

        # unsupported string
        with pytest.raises(ValueError):
            ProjectedFastIterativeShrinkageThresholdingAlgorithmOption(
                mode_stopping_criterion_gradient_descent="unsupported_string"
            )

        # Test that "mode_stopping_criterion_gradient_descent" cannot be updated
        with pytest.raises(AttributeError):
            option = ProjectedFastIterativeShrinkageThresholdingAlgorithmOption()
            option.mode_stopping_criterion_gradient_descent = "single_difference_loss"

    def test_access_num_history_stopping_criterion_gradient_descent(self):
        # default
        option = ProjectedFastIterativeShrinkageThresholdingAlgorithmOption()
        assert option.num_history_stopping_criterion_gradient_descent == 1

        # num_history_stopping_criterion_gradient_descent = 2
        option = ProjectedFastIterativeShrinkageThresholdingAlgorithmOption(
            num_history_stopping_criterion_gradient_descent=2
        )
        assert option.num_history_stopping_criterion_gradient_descent == 2

        # not int
        with pytest.raises(ValueError):
            ProjectedFastIterativeShrinkageThresholdingAlgorithmOption(
                num_history_stopping_criterion_gradient_descent=1.0
            )

        # <1
        with pytest.raises(ValueError):
            ProjectedFastIterativeShrinkageThresholdingAlgorithmOption(
                num_history_stopping_criterion_gradient_descent=0
            )

        # Test that "num_history_stopping_criterion_gradient_descent" cannot be updated
        with pytest.raises(AttributeError):
            option = ProjectedFastIterativeShrinkageThresholdingAlgorithmOption()
            option.num_history_stopping_criterion_gradient_descent = 1

    def test_access_mode_proj_order(self):
        # default
        option = ProjectedFastIterativeShrinkageThresholdingAlgorithmOption()
        assert option.mode_proj_order == "eq_ineq"

        # eq_ineq
        option = ProjectedFastIterativeShrinkageThresholdingAlgorithmOption(
            mode_proj_order="eq_ineq"
        )
        assert option.mode_proj_order == "eq_ineq"

        # ineq_eq
        option = ProjectedFastIterativeShrinkageThresholdingAlgorithmOption(
            mode_proj_order="ineq_eq"
        )
        assert option.mode_proj_order == "ineq_eq"

        # unsupported value
        with pytest.raises(ValueError):
            ProjectedFastIterativeShrinkageThresholdingAlgorithmOption(
                mode_proj_order="unsupported"
            )

        # Test that "mode_proj_order" cannot be updated
        with pytest.raises(AttributeError):
            option = ProjectedFastIterativeShrinkageThresholdingAlgorithmOption()
            option.mode_proj_order = "eq_ineq"


class TestProjectedFastIterativeShrinkageThresholdingAlgorithm:
    def test_access_func_proj(self):
        algo = ProjectedFastIterativeShrinkageThresholdingAlgorithm()
        assert algo.func_proj is None

        with pytest.raises(AttributeError):
            algo.func_proj = func_proj.proj_to_self

        algo = ProjectedFastIterativeShrinkageThresholdingAlgorithm(
            func_proj.proj_to_self
        )
        assert algo.func_proj is not None

    def test_set_constraint_from_standard_qt_and_option(self):
        # case1: use func_calc_proj_physical() if on_algo_eq_constraint=True, on_algo_ineq_constraint=True
        qst, _ = get_test_data()
        algo = ProjectedFastIterativeShrinkageThresholdingAlgorithm()
        option = ProjectedFastIterativeShrinkageThresholdingAlgorithmOption(
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
        algo = ProjectedFastIterativeShrinkageThresholdingAlgorithm()
        option = ProjectedFastIterativeShrinkageThresholdingAlgorithmOption(
            on_algo_eq_constraint=True, on_algo_ineq_constraint=False
        )
        algo.set_constraint_from_standard_qt_and_option(qst, option)
        var = np.array([2, 1, 1, 1], dtype=np.float64) / np.sqrt(2)
        actual = algo.func_proj(var)
        expected = np.array([1, 1, 1, 1], dtype=np.float64) / np.sqrt(2)
        npt.assert_almost_equal(actual, expected, decimal=14)

        # case3: use func_calc_proj_ineq_constraint() if on_algo_eq_constraint=False, on_algo_ineq_constraint=True
        qst, _ = get_test_data()
        algo = ProjectedFastIterativeShrinkageThresholdingAlgorithm()
        option = ProjectedFastIterativeShrinkageThresholdingAlgorithmOption(
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
        algo = ProjectedFastIterativeShrinkageThresholdingAlgorithm()
        option = ProjectedFastIterativeShrinkageThresholdingAlgorithmOption(
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
        algo = ProjectedFastIterativeShrinkageThresholdingAlgorithm()
        algo.set_from_loss(loss)
        assert algo.is_loss_sufficient() == True

        # loss is None
        algo = ProjectedFastIterativeShrinkageThresholdingAlgorithm()
        assert algo.is_loss_sufficient() == False

        # loss is None
        algo = ProjectedFastIterativeShrinkageThresholdingAlgorithm()
        algo.set_from_loss(None)
        assert algo.is_loss_sufficient() == False

        # loss.on_value is False
        var_ref = np.array([1, 0, 0, 0], dtype=np.float64) / np.sqrt(2)
        loss = SimpleQuadraticLossFunction(var_ref)
        loss._on_value = False
        algo = ProjectedFastIterativeShrinkageThresholdingAlgorithm()
        algo.set_from_loss(loss)
        assert algo.is_loss_sufficient() == False

        # loss.on_gradient is False
        var_ref = np.array([1, 0, 0, 0], dtype=np.float64) / np.sqrt(2)
        loss = SimpleQuadraticLossFunction(var_ref)
        loss._on_gradient = False
        algo = ProjectedFastIterativeShrinkageThresholdingAlgorithm()
        algo.set_from_loss(loss)
        assert algo.is_loss_sufficient() == False

    def test_is_option_sufficient(self):
        # option is not None
        algo = ProjectedFastIterativeShrinkageThresholdingAlgorithm()
        algo_option = ProjectedFastIterativeShrinkageThresholdingAlgorithmOption()
        algo.set_from_option(algo_option)
        assert algo.is_option_sufficient() == True

        # option is None
        algo = ProjectedFastIterativeShrinkageThresholdingAlgorithm()
        assert algo.is_option_sufficient() == False

        # option is None
        algo = ProjectedFastIterativeShrinkageThresholdingAlgorithm()
        algo.set_from_option(None)
        assert algo.is_option_sufficient() == False

        # option.delta is None
        algo = ProjectedFastIterativeShrinkageThresholdingAlgorithm()
        algo_option = ProjectedFastIterativeShrinkageThresholdingAlgorithmOption(
            delta=None
        )
        algo.set_from_option(algo_option)
        assert algo.is_option_sufficient() == True

        # option.delta is non-positive
        algo = ProjectedFastIterativeShrinkageThresholdingAlgorithm()
        algo_option = ProjectedFastIterativeShrinkageThresholdingAlgorithmOption(
            delta=0
        )
        algo.set_from_option(algo_option)
        assert algo.is_option_sufficient() == False

        # option.eps is None
        algo = ProjectedFastIterativeShrinkageThresholdingAlgorithm()
        algo_option = ProjectedFastIterativeShrinkageThresholdingAlgorithmOption(
            eps=None
        )
        algo.set_from_option(algo_option)
        assert algo.is_option_sufficient() == True

        # option.eps is non-positive
        algo = ProjectedFastIterativeShrinkageThresholdingAlgorithm()
        algo_option = ProjectedFastIterativeShrinkageThresholdingAlgorithmOption(eps=0)
        algo.set_from_option(algo_option)
        assert algo.is_option_sufficient() == False

    def test_is_loss_and_option_sufficient(self):
        # option or loss is None
        algo = ProjectedFastIterativeShrinkageThresholdingAlgorithm()
        assert algo.is_loss_and_option_sufficient() == True

        # option.var_start is None
        algo = ProjectedFastIterativeShrinkageThresholdingAlgorithm()
        algo_option = ProjectedFastIterativeShrinkageThresholdingAlgorithmOption()
        algo.set_from_option(algo_option)
        assert algo.is_loss_and_option_sufficient() == True

        # self.option.var_start.shape[0] != self.loss.num_var
        var_ref1 = np.array([0, 0, 0], dtype=np.float64) / np.sqrt(2)
        algo_option = ProjectedFastIterativeShrinkageThresholdingAlgorithmOption(
            var_start=var_ref1
        )
        var_ref2 = np.array([1, 0, 0, 0], dtype=np.float64) / np.sqrt(2)
        loss = SimpleQuadraticLossFunction(var_ref2)

        algo = ProjectedFastIterativeShrinkageThresholdingAlgorithm()
        algo.set_from_option(algo_option)
        algo.set_from_loss(loss)
        assert algo.is_loss_and_option_sufficient() == False

    def test_optimize_value_error(self):
        algo_option = ProjectedFastIterativeShrinkageThresholdingAlgorithmOption()

        # loss.on_value is False
        var_ref = np.array([1, 0, 0, 0], dtype=np.float64) / np.sqrt(2)
        loss = SimpleQuadraticLossFunction(var_ref)
        loss._on_value = False
        algo = ProjectedFastIterativeShrinkageThresholdingAlgorithm()
        algo.set_from_loss(loss)
        with pytest.raises(ValueError):
            algo.optimize(loss, None, algo_option)

        # loss.on_gradient is False
        var_ref = np.array([1, 0, 0, 0], dtype=np.float64) / np.sqrt(2)
        loss = SimpleQuadraticLossFunction(var_ref)
        loss._on_gradient = False
        algo = ProjectedFastIterativeShrinkageThresholdingAlgorithm()
        algo.set_from_loss(loss)
        with pytest.raises(ValueError):
            algo.optimize(loss, None, algo_option)
