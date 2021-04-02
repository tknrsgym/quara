import time
from typing import List

import numpy as np
import numpy.testing as npt
import pytest

from quara.loss_function.weighted_probability_based_squared_error import (
    WeightedProbabilityBasedSquaredError,
    WeightedProbabilityBasedSquaredErrorOption,
)
from quara.minimization_algorithm.projected_gradient_descent_backtracking import (
    ProjectedGradientDescentBacktracking,
    ProjectedGradientDescentBacktrackingOption,
)
from quara.math import func_proj
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
from quara.protocol.qtomography.standard.loss_minimization_estimator import (
    LossMinimizationEstimator,
)


def get_test_data(on_para_eq_constraint=False):
    e_sys = ElementalSystem(0, get_normalized_pauli_basis())
    c_sys = CompositeSystem([e_sys])

    povm_x = get_x_povm(c_sys)
    povm_y = get_y_povm(c_sys)
    povm_z = get_z_povm(c_sys)
    povms = [povm_x, povm_y, povm_z]

    qst = StandardQst(povms, on_para_eq_constraint=on_para_eq_constraint, seed=7)

    return qst, c_sys


class TestLossMinimizationEstimator:
    def test_calc_estimate__is_computation_time_required(self):
        empi_dists = [
            (10000, np.array([0.5, 0.5], dtype=np.float64)),
            (10000, np.array([0.5, 0.5], dtype=np.float64)),
            (10000, np.array([1, 0], dtype=np.float64)),
        ]
        loss = WeightedProbabilityBasedSquaredError(4)
        loss_option = WeightedProbabilityBasedSquaredErrorOption(mode_weight="identity")

        qst, _ = get_test_data()
        algo = ProjectedGradientDescentBacktracking()
        algo_option = ProjectedGradientDescentBacktrackingOption(
            on_algo_eq_constraint=True, on_algo_ineq_constraint=True
        )

        estimator = LossMinimizationEstimator()

        # is_computation_time_required=True
        actual = estimator.calc_estimate(
            qst,
            empi_dists,
            loss,
            loss_option,
            algo,
            algo_option,
            is_computation_time_required=True,
        )
        expected = [1 / np.sqrt(2), 0, 0, 1 / np.sqrt(2)]
        assert actual.estimated_qoperation.is_physical()
        npt.assert_almost_equal(actual.estimated_var, expected, decimal=7)
        assert type(actual.computation_time) == float

        # is_computation_time_required=False
        actual = estimator.calc_estimate(
            qst, empi_dists, loss, loss_option, algo, algo_option
        )
        expected = [1 / np.sqrt(2), 0, 0, 1 / np.sqrt(2)]
        assert actual.estimated_qoperation.is_physical()
        npt.assert_almost_equal(actual.estimated_var, expected, decimal=7)
        assert actual.computation_time == None

    def test_calc_estimate__on_algo_xx_constraint(self):
        empi_dists = [
            (10000, np.array([0.5, 0.5], dtype=np.float64)),
            (10000, np.array([0.5, 0.5], dtype=np.float64)),
            (10000, np.array([1, 0], dtype=np.float64)),
        ]
        loss = WeightedProbabilityBasedSquaredError(4)
        loss_option = WeightedProbabilityBasedSquaredErrorOption(mode_weight="identity")

        # case1: on_algo_eq_constraint=True, on_algo_ineq_constraint=True
        qst, _ = get_test_data()
        algo = ProjectedGradientDescentBacktracking()

        obj_start = (
            qst.generate_empty_estimation_obj_with_setting_info().generate_origin_obj()
        )
        var_start = obj_start.to_var()
        algo_option = ProjectedGradientDescentBacktrackingOption(
            on_algo_eq_constraint=True,
            on_algo_ineq_constraint=True,
            var_start=var_start,
        )

        estimator = LossMinimizationEstimator()

        actual = estimator.calc_estimate(
            qst, empi_dists, loss, loss_option, algo, algo_option
        )
        expected = [1 / np.sqrt(2), 0, 0, 1 / np.sqrt(2)]
        assert actual.estimated_qoperation.is_physical()
        npt.assert_almost_equal(actual.estimated_var, expected, decimal=7)

        # case2: on_algo_eq_constraint=True, on_algo_ineq_constraint=False
        qst, _ = get_test_data()
        algo = ProjectedGradientDescentBacktracking()

        obj_start = (
            qst.generate_empty_estimation_obj_with_setting_info().generate_origin_obj()
        )
        var_start = obj_start.to_var()
        algo_option = ProjectedGradientDescentBacktrackingOption(
            on_algo_eq_constraint=True,
            on_algo_ineq_constraint=False,
            var_start=var_start,
        )

        estimator = LossMinimizationEstimator()

        actual = estimator.calc_estimate(
            qst, empi_dists, loss, loss_option, algo, algo_option
        )
        expected = [1 / np.sqrt(2), 0, 0, 0.707106764760052]
        assert actual.estimated_qoperation.is_physical()
        npt.assert_almost_equal(actual.estimated_var, expected, decimal=15)

        # case3: on_algo_eq_constraint=False, on_algo_ineq_constraint=True
        qst, _ = get_test_data()
        algo = ProjectedGradientDescentBacktracking()

        obj_start = (
            qst.generate_empty_estimation_obj_with_setting_info().generate_origin_obj()
        )
        var_start = obj_start.to_var()
        algo_option = ProjectedGradientDescentBacktrackingOption(
            on_algo_eq_constraint=False,
            on_algo_ineq_constraint=True,
            var_start=var_start,
        )

        estimator = LossMinimizationEstimator()

        actual = estimator.calc_estimate(
            qst, empi_dists, loss, loss_option, algo, algo_option
        )
        expected = [0.707106790461949, 0, 0, 0.707106713475754]
        assert actual.estimated_qoperation.is_physical()
        npt.assert_almost_equal(actual.estimated_var, expected, decimal=15)

        # case4: on_algo_eq_constraint=False, on_algo_ineq_constraint=False
        qst, _ = get_test_data()
        algo = ProjectedGradientDescentBacktracking()

        obj_start = (
            qst.generate_empty_estimation_obj_with_setting_info().generate_origin_obj()
        )
        var_start = obj_start.to_var()
        algo_option = ProjectedGradientDescentBacktrackingOption(
            on_algo_eq_constraint=False,
            on_algo_ineq_constraint=False,
            var_start=var_start,
        )

        estimator = LossMinimizationEstimator()

        actual = estimator.calc_estimate(
            qst, empi_dists, loss, loss_option, algo, algo_option
        )
        expected = [
            7.071067772680395e-01,
            -3.291237808416748e-26,
            -3.291237808416748e-26,
            7.071067976130435e-01,
        ]
        npt.assert_almost_equal(actual.estimated_var, expected, decimal=10)

    def test_calc_estimate__func_proj(self):
        empi_dists = [
            (10000, np.array([0.5, 0.5], dtype=np.float64)),
            (10000, np.array([0.5, 0.5], dtype=np.float64)),
            (10000, np.array([1, 0], dtype=np.float64)),
        ]
        loss = WeightedProbabilityBasedSquaredError(4)
        loss_option = WeightedProbabilityBasedSquaredErrorOption(mode_weight="identity")

        # case1: func_proj=auto setting
        qst, _ = get_test_data()
        algo = ProjectedGradientDescentBacktracking()

        obj_start = (
            qst.generate_empty_estimation_obj_with_setting_info().generate_origin_obj()
        )
        var_start = obj_start.to_var()
        algo_option = ProjectedGradientDescentBacktrackingOption(
            on_algo_eq_constraint=True,
            on_algo_ineq_constraint=True,
            var_start=var_start,
        )

        estimator = LossMinimizationEstimator()

        actual = estimator.calc_estimate(
            qst, empi_dists, loss, loss_option, algo, algo_option
        )
        expected = [1 / np.sqrt(2), 0, 0, 1 / np.sqrt(2)]
        assert actual.estimated_qoperation.is_physical()
        npt.assert_almost_equal(actual.estimated_var, expected, decimal=7)

        # case2: func_proj=func_proj.proj_to_self
        qst, _ = get_test_data()
        algo = ProjectedGradientDescentBacktracking()

        obj_start = (
            qst.generate_empty_estimation_obj_with_setting_info().generate_origin_obj()
        )
        var_start = obj_start.to_var()
        algo_option = ProjectedGradientDescentBacktrackingOption(
            on_algo_eq_constraint=True,
            on_algo_ineq_constraint=True,
            var_start=var_start,
        )
        loss = WeightedProbabilityBasedSquaredError(
            4, func_prob_dists=func_proj.proj_to_self
        )

        estimator = LossMinimizationEstimator()

        actual = estimator.calc_estimate(
            qst, empi_dists, loss, loss_option, algo, algo_option
        )
        expected = [1 / np.sqrt(2), 0, 0, 1 / np.sqrt(2)]
        assert actual.estimated_qoperation.is_physical()
        npt.assert_almost_equal(actual.estimated_var, expected, decimal=7)

        # case3: func_proj=func_proj.proj_to_hyperplane
        qst, _ = get_test_data()
        algo = ProjectedGradientDescentBacktracking()

        obj_start = (
            qst.generate_empty_estimation_obj_with_setting_info().generate_origin_obj()
        )
        var_start = obj_start.to_var()
        algo_option = ProjectedGradientDescentBacktrackingOption(
            on_algo_eq_constraint=True,
            on_algo_ineq_constraint=True,
            var_start=var_start,
        )
        loss = WeightedProbabilityBasedSquaredError(
            4, func_prob_dists=func_proj.proj_to_hyperplane(var_start)
        )

        estimator = LossMinimizationEstimator()

        actual = estimator.calc_estimate(
            qst, empi_dists, loss, loss_option, algo, algo_option
        )
        expected = [1 / np.sqrt(2), 0, 0, 1 / np.sqrt(2)]
        assert actual.estimated_qoperation.is_physical()
        npt.assert_almost_equal(actual.estimated_var, expected, decimal=7)

        # case4: func_proj=func_proj.proj_to_nonnegative
        qst, _ = get_test_data()
        algo = ProjectedGradientDescentBacktracking()

        obj_start = (
            qst.generate_empty_estimation_obj_with_setting_info().generate_origin_obj()
        )
        var_start = obj_start.to_var()
        algo_option = ProjectedGradientDescentBacktrackingOption(
            on_algo_eq_constraint=True,
            on_algo_ineq_constraint=True,
            var_start=var_start,
        )
        loss = WeightedProbabilityBasedSquaredError(
            4, func_prob_dists=func_proj.proj_to_nonnegative
        )

        estimator = LossMinimizationEstimator()

        actual = estimator.calc_estimate(
            qst, empi_dists, loss, loss_option, algo, algo_option
        )
        expected = [1 / np.sqrt(2), 0, 0, 1 / np.sqrt(2)]
        assert actual.estimated_qoperation.is_physical()
        npt.assert_almost_equal(actual.estimated_var, expected, decimal=7)

    def test_calc_estimate_sequence(self):
        qst, _ = get_test_data()
        empi_dists_seq = [
            [
                (100, np.array([0.5, 0.5], dtype=np.float64)),
                (100, np.array([0.5, 0.5], dtype=np.float64)),
                (100, np.array([1, 0], dtype=np.float64)),
            ],
            [
                (10000, np.array([1, 0], dtype=np.float64)),
                (10000, np.array([0.5, 0.5], dtype=np.float64)),
                (10000, np.array([0.5, 0.5], dtype=np.float64)),
            ],
        ]

        loss = WeightedProbabilityBasedSquaredError(4)
        loss_option = WeightedProbabilityBasedSquaredErrorOption(mode_weight="identity")

        qst, _ = get_test_data()
        algo = ProjectedGradientDescentBacktracking()
        algo_option = ProjectedGradientDescentBacktrackingOption()

        estimator = LossMinimizationEstimator()

        actual = estimator.calc_estimate_sequence(
            qst, empi_dists_seq, loss, loss_option, algo, algo_option,
        )

        expected = [
            [1 / np.sqrt(2), 0, 0, 1 / np.sqrt(2)],
            [1 / np.sqrt(2), 1 / np.sqrt(2), 0, 0],
        ]
        for a, e in zip(actual.estimated_qoperation_sequence, expected):
            assert a.is_physical()
            npt.assert_almost_equal(a.to_stacked_vector(), e, decimal=7)

    def test_calc_estimate_sequence_value_error(self):
        qst, _ = get_test_data()
        empi_dists_seq = [
            [
                (100, np.array([0.5, 0.5], dtype=np.float64)),
                (100, np.array([0.5, 0.5], dtype=np.float64)),
                (100, np.array([1, 0], dtype=np.float64)),
            ],
            [
                (10000, np.array([1, 0], dtype=np.float64)),
                (10000, np.array([0.5, 0.5], dtype=np.float64)),
                (10000, np.array([0.5, 0.5], dtype=np.float64)),
            ],
        ]

        # loss.is_option_sufficient() is False
        loss = WeightedProbabilityBasedSquaredError(4)

        def _is_option_sufficient():
            return False

        loss.is_option_sufficient = _is_option_sufficient
        loss_option = WeightedProbabilityBasedSquaredErrorOption(mode_weight="identity")

        qst, _ = get_test_data()
        algo = ProjectedGradientDescentBacktracking()
        algo_option = ProjectedGradientDescentBacktrackingOption()

        estimator = LossMinimizationEstimator()

        with pytest.raises(ValueError):
            estimator.calc_estimate_sequence(
                qst, empi_dists_seq, loss, loss_option, algo, algo_option,
            )

        # algo.is_loss_sufficient() is False
        loss = WeightedProbabilityBasedSquaredError(4)
        loss_option = WeightedProbabilityBasedSquaredErrorOption(mode_weight="identity")

        qst, _ = get_test_data()
        algo = ProjectedGradientDescentBacktracking()

        def _is_loss_sufficient():
            return False

        algo.is_loss_sufficient = _is_loss_sufficient
        algo_option = ProjectedGradientDescentBacktrackingOption()

        estimator = LossMinimizationEstimator()
        with pytest.raises(ValueError):
            estimator.calc_estimate_sequence(
                qst, empi_dists_seq, loss, loss_option, algo, algo_option,
            )

        # algo.is_option_sufficient() is False
        loss = WeightedProbabilityBasedSquaredError(4)
        loss_option = WeightedProbabilityBasedSquaredErrorOption(mode_weight="identity")

        qst, _ = get_test_data()
        algo = ProjectedGradientDescentBacktracking()

        def _is_option_sufficient():
            return False

        algo.is_option_sufficient = _is_option_sufficient
        algo_option = ProjectedGradientDescentBacktrackingOption()

        estimator = LossMinimizationEstimator()
        with pytest.raises(ValueError):
            estimator.calc_estimate_sequence(
                qst, empi_dists_seq, loss, loss_option, algo, algo_option,
            )

        # algo.is_loss_and_option_sufficient() is False
        loss = WeightedProbabilityBasedSquaredError(4)
        loss_option = WeightedProbabilityBasedSquaredErrorOption(mode_weight="identity")

        qst, _ = get_test_data()
        algo = ProjectedGradientDescentBacktracking()

        def _is_loss_and_option_sufficient():
            return False

        algo.is_loss_and_option_sufficient = _is_loss_and_option_sufficient
        algo_option = ProjectedGradientDescentBacktrackingOption()

        estimator = LossMinimizationEstimator()
        with pytest.raises(ValueError):
            estimator.calc_estimate_sequence(
                qst, empi_dists_seq, loss, loss_option, algo, algo_option,
            )

    def test_calc_estimate__pgdb_opton__sum_absolute_difference_loss(self):
        empi_dists = [
            (10000, np.array([0.5, 0.5], dtype=np.float64)),
            (10000, np.array([0.5, 0.5], dtype=np.float64)),
            (10000, np.array([1, 0], dtype=np.float64)),
        ]
        loss = WeightedProbabilityBasedSquaredError(4)
        loss_option = WeightedProbabilityBasedSquaredErrorOption(mode_weight="identity")

        qst, _ = get_test_data()
        algo = ProjectedGradientDescentBacktracking()
        # mode_stopping_criterion_gradient_descent="sum_absolute_difference_loss"
        # num_history_stopping_criterion_gradient_descent=10
        algo_option = ProjectedGradientDescentBacktrackingOption(
            on_algo_eq_constraint=True,
            on_algo_ineq_constraint=True,
            mode_stopping_criterion_gradient_descent="sum_absolute_difference_loss",
            num_history_stopping_criterion_gradient_descent=10,
        )

        estimator = LossMinimizationEstimator()

        # is_computation_time_required=True
        actual = estimator.calc_estimate(
            qst,
            empi_dists,
            loss,
            loss_option,
            algo,
            algo_option,
            is_computation_time_required=True,
        )
        expected = [1 / np.sqrt(2), 0, 0, 1 / np.sqrt(2)]
        assert actual.estimated_qoperation.is_physical()
        npt.assert_almost_equal(actual.estimated_var, expected, decimal=11)

    def test_calc_estimate__pgdb_opton__sum_absolute_difference_variable(self):
        empi_dists = [
            (10000, np.array([0.5, 0.5], dtype=np.float64)),
            (10000, np.array([0.5, 0.5], dtype=np.float64)),
            (10000, np.array([1, 0], dtype=np.float64)),
        ]
        loss = WeightedProbabilityBasedSquaredError(4)
        loss_option = WeightedProbabilityBasedSquaredErrorOption(mode_weight="identity")

        qst, _ = get_test_data()
        algo = ProjectedGradientDescentBacktracking()
        # mode_stopping_criterion_gradient_descent="sum_absolute_difference_variable"
        # num_history_stopping_criterion_gradient_descent=10
        algo_option = ProjectedGradientDescentBacktrackingOption(
            on_algo_eq_constraint=True,
            on_algo_ineq_constraint=True,
            mode_stopping_criterion_gradient_descent="sum_absolute_difference_variable",
            num_history_stopping_criterion_gradient_descent=10,
        )

        estimator = LossMinimizationEstimator()

        # is_computation_time_required=True
        actual = estimator.calc_estimate(
            qst,
            empi_dists,
            loss,
            loss_option,
            algo,
            algo_option,
            is_computation_time_required=True,
        )
        expected = [1 / np.sqrt(2), 0, 0, 1 / np.sqrt(2)]
        assert actual.estimated_qoperation.is_physical()
        npt.assert_almost_equal(actual.estimated_var, expected, decimal=15)

    def test_calc_estimate__pgdb_opton__sum_absolute_difference_projected_gradient(
        self,
    ):
        empi_dists = [
            (10000, np.array([0.5, 0.5], dtype=np.float64)),
            (10000, np.array([0.5, 0.5], dtype=np.float64)),
            (10000, np.array([1, 0], dtype=np.float64)),
        ]
        loss = WeightedProbabilityBasedSquaredError(4)
        loss_option = WeightedProbabilityBasedSquaredErrorOption(mode_weight="identity")

        qst, _ = get_test_data()
        algo = ProjectedGradientDescentBacktracking()
        # mode_stopping_criterion_gradient_descent="sum_absolute_difference_projected_gradient"
        # num_history_stopping_criterion_gradient_descent=10
        algo_option = ProjectedGradientDescentBacktrackingOption(
            on_algo_eq_constraint=True,
            on_algo_ineq_constraint=True,
            mode_stopping_criterion_gradient_descent="sum_absolute_difference_projected_gradient",
            num_history_stopping_criterion_gradient_descent=10,
        )

        estimator = LossMinimizationEstimator()

        # is_computation_time_required=True
        actual = estimator.calc_estimate(
            qst,
            empi_dists,
            loss,
            loss_option,
            algo,
            algo_option,
            is_computation_time_required=True,
        )
        expected = [1 / np.sqrt(2), 0, 0, 1 / np.sqrt(2)]
        assert actual.estimated_qoperation.is_physical()
        npt.assert_almost_equal(actual.estimated_var, expected, decimal=15)
