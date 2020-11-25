import time
from typing import List

import numpy as np
import numpy.testing as npt

from quara.data_analysis.projected_gradient_descent_base import (
    ProjectedGradientDescentBase,
    ProjectedGradientDescentBaseOption,
)
from quara.data_analysis.weighted_probability_based_squared_error import (
    WeightedProbabilityBasedSquaredError,
    WeightedProbabilityBasedSquaredErrorOption,
)
from quara.math import func_proj
from quara.objects.composite_system import CompositeSystem
from quara.objects.elemental_system import ElementalSystem
from quara.objects.matrix_basis import get_normalized_pauli_basis
from quara.objects.povm import (
    get_x_measurement,
    get_y_measurement,
    get_z_measurement,
)
from quara.objects.state import convert_var_to_state, get_z0_1q
from quara.protocol.qtomography.standard.standard_qst import StandardQst
from quara.protocol.qtomography.standard.loss_minimization_estimator import (
    LossMinimizationEstimator,
)


def get_test_data(
    on_para_eq_constraint=False,
    on_algo_eq_constraint=False,
    on_algo_ineq_constraint=False,
):
    e_sys = ElementalSystem(0, get_normalized_pauli_basis())
    c_sys = CompositeSystem([e_sys])

    povm_x = get_x_measurement(c_sys)
    povm_y = get_y_measurement(c_sys)
    povm_z = get_z_measurement(c_sys)
    povms = [povm_x, povm_y, povm_z]

    qst = StandardQst(
        povms,
        on_para_eq_constraint=on_para_eq_constraint,
        on_algo_eq_constraint=on_algo_eq_constraint,
        on_algo_ineq_constraint=on_algo_ineq_constraint,
        seed=7,
    )

    return qst, c_sys


class TestLossMinimizationEstimator:
    def test_calc_estimate__is_computation_time_required(self):
        empi_dists = [
            (10000, np.array([0.5, 0.5], dtype=np.float64)),
            (10000, np.array([0.5, 0.5], dtype=np.float64)),
            (10000, np.array([1, 0], dtype=np.float64)),
        ]
        loss = WeightedProbabilityBasedSquaredError(4)
        loss_option = WeightedProbabilityBasedSquaredErrorOption()

        qst, _ = get_test_data(on_algo_eq_constraint=True, on_algo_ineq_constraint=True)
        algo = ProjectedGradientDescentBase()
        algo_option = ProjectedGradientDescentBaseOption()

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
        assert type(actual.computation_time) == float

        # is_computation_time_required=False
        actual = estimator.calc_estimate(
            qst, empi_dists, loss, loss_option, algo, algo_option
        )
        expected = [1 / np.sqrt(2), 0, 0, 1 / np.sqrt(2)]
        assert actual.estimated_qoperation.is_physical()
        npt.assert_almost_equal(actual.estimated_var, expected, decimal=15)
        assert actual.computation_time == None

    def test_calc_estimate__on_algo_xx_constraint(self):
        empi_dists = [
            (10000, np.array([0.5, 0.5], dtype=np.float64)),
            (10000, np.array([0.5, 0.5], dtype=np.float64)),
            (10000, np.array([1, 0], dtype=np.float64)),
        ]
        loss = WeightedProbabilityBasedSquaredError(4)
        loss_option = WeightedProbabilityBasedSquaredErrorOption()

        # case1: on_algo_eq_constraint=True, on_algo_ineq_constraint=True
        qst, _ = get_test_data(on_algo_eq_constraint=True, on_algo_ineq_constraint=True)
        algo = ProjectedGradientDescentBase()

        obj_start = (
            qst.generate_empty_estimation_obj_with_setting_info().generate_origin_obj()
        )
        var_start = obj_start.to_var()
        algo_option = ProjectedGradientDescentBaseOption(var_start)

        estimator = LossMinimizationEstimator()

        actual = estimator.calc_estimate(
            qst, empi_dists, loss, loss_option, algo, algo_option
        )
        expected = [1 / np.sqrt(2), 0, 0, 1 / np.sqrt(2)]
        assert actual.estimated_qoperation.is_physical()
        npt.assert_almost_equal(actual.estimated_var, expected, decimal=15)

        # case2: on_algo_eq_constraint=True, on_algo_ineq_constraint=False
        qst, _ = get_test_data(
            on_algo_eq_constraint=True, on_algo_ineq_constraint=False
        )
        algo = ProjectedGradientDescentBase()

        obj_start = (
            qst.generate_empty_estimation_obj_with_setting_info().generate_origin_obj()
        )
        var_start = obj_start.to_var()
        algo_option = ProjectedGradientDescentBaseOption(var_start)

        estimator = LossMinimizationEstimator()

        actual = estimator.calc_estimate(
            qst, empi_dists, loss, loss_option, algo, algo_option
        )
        expected = [1 / np.sqrt(2), 0, 0, 0.707106764760052]
        assert actual.estimated_qoperation.is_physical()
        npt.assert_almost_equal(actual.estimated_var, expected, decimal=15)

        # case3: on_algo_eq_constraint=False, on_algo_ineq_constraint=True
        qst, _ = get_test_data(
            on_algo_eq_constraint=False, on_algo_ineq_constraint=True
        )
        algo = ProjectedGradientDescentBase()

        obj_start = (
            qst.generate_empty_estimation_obj_with_setting_info().generate_origin_obj()
        )
        var_start = obj_start.to_var()
        algo_option = ProjectedGradientDescentBaseOption(var_start)

        estimator = LossMinimizationEstimator()

        actual = estimator.calc_estimate(
            qst, empi_dists, loss, loss_option, algo, algo_option
        )
        expected = [0.707106790461949, 0, 0, 0.707106713475754]
        assert actual.estimated_qoperation.is_physical()
        npt.assert_almost_equal(actual.estimated_var, expected, decimal=15)

        # case4: on_algo_eq_constraint=False, on_algo_ineq_constraint=False
        qst, _ = get_test_data(
            on_algo_eq_constraint=False, on_algo_ineq_constraint=False
        )
        algo = ProjectedGradientDescentBase()

        obj_start = (
            qst.generate_empty_estimation_obj_with_setting_info().generate_origin_obj()
        )
        var_start = obj_start.to_var()
        algo_option = ProjectedGradientDescentBaseOption(var_start)

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
        loss_option = WeightedProbabilityBasedSquaredErrorOption()

        # case1: func_proj=auto setting
        qst, _ = get_test_data(on_algo_eq_constraint=True, on_algo_ineq_constraint=True)
        algo = ProjectedGradientDescentBase()

        obj_start = (
            qst.generate_empty_estimation_obj_with_setting_info().generate_origin_obj()
        )
        var_start = obj_start.to_var()
        algo_option = ProjectedGradientDescentBaseOption(var_start)

        estimator = LossMinimizationEstimator()

        actual = estimator.calc_estimate(
            qst, empi_dists, loss, loss_option, algo, algo_option
        )
        expected = [1 / np.sqrt(2), 0, 0, 1 / np.sqrt(2)]
        assert actual.estimated_qoperation.is_physical()
        npt.assert_almost_equal(actual.estimated_var, expected, decimal=15)

        # case2: func_proj=func_proj.proj_to_self
        qst, _ = get_test_data(on_algo_eq_constraint=True, on_algo_ineq_constraint=True)
        algo = ProjectedGradientDescentBase()

        obj_start = (
            qst.generate_empty_estimation_obj_with_setting_info().generate_origin_obj()
        )
        var_start = obj_start.to_var()
        algo_option = ProjectedGradientDescentBaseOption(var_start)
        loss = WeightedProbabilityBasedSquaredError(
            4, func_prob_dists=func_proj.proj_to_self
        )

        estimator = LossMinimizationEstimator()

        actual = estimator.calc_estimate(
            qst, empi_dists, loss, loss_option, algo, algo_option
        )
        expected = [1 / np.sqrt(2), 0, 0, 1 / np.sqrt(2)]
        assert actual.estimated_qoperation.is_physical()
        npt.assert_almost_equal(actual.estimated_var, expected, decimal=15)

        # case3: func_proj=func_proj.proj_to_hyperplane
        qst, _ = get_test_data(on_algo_eq_constraint=True, on_algo_ineq_constraint=True)
        algo = ProjectedGradientDescentBase()

        obj_start = (
            qst.generate_empty_estimation_obj_with_setting_info().generate_origin_obj()
        )
        var_start = obj_start.to_var()
        algo_option = ProjectedGradientDescentBaseOption(var_start)
        loss = WeightedProbabilityBasedSquaredError(
            4, func_prob_dists=func_proj.proj_to_hyperplane(var_start)
        )

        estimator = LossMinimizationEstimator()

        actual = estimator.calc_estimate(
            qst, empi_dists, loss, loss_option, algo, algo_option
        )
        expected = [1 / np.sqrt(2), 0, 0, 1 / np.sqrt(2)]
        assert actual.estimated_qoperation.is_physical()
        npt.assert_almost_equal(actual.estimated_var, expected, decimal=15)

        # case4: func_proj=func_proj.proj_to_nonnegative
        qst, _ = get_test_data(on_algo_eq_constraint=True, on_algo_ineq_constraint=True)
        algo = ProjectedGradientDescentBase()

        obj_start = (
            qst.generate_empty_estimation_obj_with_setting_info().generate_origin_obj()
        )
        var_start = obj_start.to_var()
        algo_option = ProjectedGradientDescentBaseOption(var_start)
        loss = WeightedProbabilityBasedSquaredError(
            4, func_prob_dists=func_proj.proj_to_nonnegative
        )

        estimator = LossMinimizationEstimator()

        actual = estimator.calc_estimate(
            qst, empi_dists, loss, loss_option, algo, algo_option
        )
        expected = [1 / np.sqrt(2), 0, 0, 1 / np.sqrt(2)]
        assert actual.estimated_qoperation.is_physical()
        npt.assert_almost_equal(actual.estimated_var, expected, decimal=15)

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
        loss_option = WeightedProbabilityBasedSquaredErrorOption()

        qst, _ = get_test_data(on_algo_eq_constraint=True, on_algo_ineq_constraint=True)
        algo = ProjectedGradientDescentBase()
        algo_option = ProjectedGradientDescentBaseOption()

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
            npt.assert_almost_equal(a.to_stacked_vector(), e, decimal=15)
