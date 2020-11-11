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
from quara.objects.state import get_z0_1q
from quara.protocol.qtomography.standard.standard_qst import StandardQst
from quara.protocol.qtomography.standard.loss_minimization_estimator import (
    LossMinimizationEstimator,
)
from quara.utils.matrix_util import calc_mse


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
    def test_calc_estimate(self):
        qst, _ = get_test_data()
        empi_dists = [
            (10000, np.array([0.5, 0.5], dtype=np.float64)),
            (10000, np.array([0.5, 0.5], dtype=np.float64)),
            (10000, np.array([1, 0], dtype=np.float64)),
        ]
        loss = WeightedProbabilityBasedSquaredError(4)
        loss_option = WeightedProbabilityBasedSquaredErrorOption()

        # TODO choice func_proj
        proj = func_proj.proj_to_self()
        algo = ProjectedGradientDescentBase(proj)

        obj_start = qst._set_qoperations.states[0].generate_origin_obj()
        var_start = obj_start.to_var()
        algo_option = ProjectedGradientDescentBaseOption(var_start)

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
        """
        expected = [1 / np.sqrt(2), 0, 0, 1 / np.sqrt(2)]
        assert actual.estimated_qoperation.is_physical()
        npt.assert_almost_equal(actual.estimated_var, expected, decimal=15)
        """
        assert type(actual.computation_time) == float

        # is_computation_time_required=False
        actual = estimator.calc_estimate(
            qst, empi_dists, loss, loss_option, algo, algo_option
        )
        # TODO
        # expected = [1 / np.sqrt(2), 0, 0, 1 / np.sqrt(2)]
        # assert actual.estimated_qoperation.is_physical()
        # npt.assert_almost_equal(actual.estimated_var, expected, decimal=15)
        # assert actual.computation_time == None

    """
    def test_calc_estimate_sequence(self):
        qst, _ = get_test_data()
        empi_dists_seq = [
            [
                (100, np.array([0.5, 0.5], dtype=np.float64)),
                (100, np.array([0.5, 0.5], dtype=np.float64)),
                (100, np.array([1, 0], dtype=np.float64)),
            ],
            [
                (10000, np.array([0.5, 0.5], dtype=np.float64)),
                (10000, np.array([0.5, 0.5], dtype=np.float64)),
                (10000, np.array([1, 0], dtype=np.float64)),
            ],
        ]

        estimator = LossMinimizationEstimator()

        # is_computation_time_required=True
        actual = estimator.calc_estimate_sequence(
            qst, empi_dists_seq, is_computation_time_required=True
        )
        expected = [
            [1 / np.sqrt(2), 0, 0, 1 / np.sqrt(2)],
            [1 / np.sqrt(2), 0, 0, 1 / np.sqrt(2)],
        ]
        for a, e in zip(actual.estimated_qoperation_sequence, expected):
            assert a.is_physical()
            npt.assert_almost_equal(a.to_stacked_vector(), e, decimal=15)
        assert len(actual.computation_times) == 2
        for a in actual.computation_times:
            assert type(a) == float

        # is_computation_time_required=False
        actual = estimator.calc_estimate_sequence(qst, empi_dists_seq)
        expected = [
            [1 / np.sqrt(2), 0, 0, 1 / np.sqrt(2)],
            [1 / np.sqrt(2), 0, 0, 1 / np.sqrt(2)],
        ]
        for a, e in zip(actual.estimated_var_sequence, expected):
            npt.assert_almost_equal(a, e, decimal=15)
        assert actual.computation_times == None

    def test_scenario_on_para_eq_constraint_True(self):
        qst, c_sys = get_test_data(on_para_eq_constraint=True)

        # generate empi dists and calc estimate
        true_object = get_z0_1q(c_sys)
        num_data = [100, 1000, 10000, 100000]
        iterations = 2

        result_sequence = []

        for _ in range(iterations):
            empi_dists_seq = qst.generate_empi_dists_sequence(true_object, num_data)

            estimator = LossMinimizationEstimator()
            result = estimator.calc_estimate_sequence(qst, empi_dists_seq)
            result_sequence.append(result.estimated_var_sequence)
            for var in result.estimated_var_sequence:
                assert len(var) == 3
            assert len(result.estimated_qoperation_sequence) == 4

        # calc mse
        result_sequences_tmp = [list(result) for result in zip(*result_sequence)]
        actual = [
            calc_mse(result, [true_object.vec[1:]] * len(result))
            for result in result_sequences_tmp
        ]
        print(f"mse={actual}")
        expected = [
            0.0037000000000000036,
            0.0005530000000000015,
            6.636000000000025e-05,
            6.1338999999999626e-06,
        ]
        npt.assert_almost_equal(actual, expected, decimal=15)

    def test_scenario_on_para_eq_constraint_False(self):
        qst, c_sys = get_test_data()

        # generate empi dists and calc estimate
        true_object = get_z0_1q(c_sys)
        num_data = [100, 1000, 10000, 100000]
        iterations = 2

        result_sequence = []

        for _ in range(iterations):
            empi_dists_seq = qst.generate_empi_dists_sequence(true_object, num_data)

            estimator = LossMinimizationEstimator()
            result = estimator.calc_estimate_sequence(qst, empi_dists_seq)
            result_sequence.append(result.estimated_var_sequence)
            for var in result.estimated_var_sequence:
                assert len(var) == 4
            assert len(result.estimated_qoperation_sequence) == 4

        # calc mse
        result_sequences_tmp = [list(result) for result in zip(*result_sequence)]
        actual = [
            calc_mse(result, [true_object.vec] * len(result))
            for result in result_sequences_tmp
        ]
        print(f"mse={actual}")
        expected = [
            0.0037000000000000045,
            0.0005530000000000005,
            6.635999999999932e-05,
            6.133899999999996e-06,
        ]
        npt.assert_almost_equal(actual, expected, decimal=15)
    """
