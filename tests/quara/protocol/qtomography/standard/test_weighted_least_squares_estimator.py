import time
from typing import List

import numpy as np
import numpy.testing as npt
import pytest

from quara.data_analysis.projected_gradient_descent_base import (
    ProjectedGradientDescentBase,
    ProjectedGradientDescentBaseOption,
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
from quara.protocol.qtomography.standard.weighted_least_squares_estimator import (
    WeightedLeastSquaresEstimator,
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


class TestLossWeightedLeastSquaresEstimator:
    def test_calc_estimate(self):
        empi_dists = [
            (10000, np.array([0.5, 0.5], dtype=np.float64)),
            (10000, np.array([0.5, 0.5], dtype=np.float64)),
            (10000, np.array([1, 0], dtype=np.float64)),
        ]

        # on_para_eq_constraint=True
        qst, _ = get_test_data(
            on_para_eq_constraint=True,
            on_algo_eq_constraint=True,
            on_algo_ineq_constraint=True,
        )
        algo = ProjectedGradientDescentBase()
        algo_option = ProjectedGradientDescentBaseOption()
        estimator = WeightedLeastSquaresEstimator(3)

        actual = estimator.calc_estimate(
            qst,
            empi_dists,
            algo,
            algo_option,
            "scm",
            "extraction",
            is_computation_time_required=True,
        )
        expected = [0, 0, 1 / np.sqrt(2)]
        assert actual.estimated_qoperation.is_physical()
        npt.assert_almost_equal(actual.estimated_var, expected, decimal=15)
        assert type(actual.computation_time) == float

        # on_para_eq_constraint=False
        qst, _ = get_test_data(
            on_para_eq_constraint=False,
            on_algo_eq_constraint=True,
            on_algo_ineq_constraint=True,
        )
        algo = ProjectedGradientDescentBase()
        algo_option = ProjectedGradientDescentBaseOption()
        estimator = WeightedLeastSquaresEstimator(4)

        actual = estimator.calc_estimate(
            qst,
            empi_dists,
            algo,
            algo_option,
            "scm",
            "extraction",
            is_computation_time_required=True,
        )
        expected = [1 / np.sqrt(2), 0, 0, 1 / np.sqrt(2)]
        assert actual.estimated_qoperation.is_physical()
        npt.assert_almost_equal(actual.estimated_var, expected, decimal=15)
        assert type(actual.computation_time) == float

        # on_para_eq_constraint=True, mode_covariance="ucm"
        qst, _ = get_test_data(
            on_para_eq_constraint=True,
            on_algo_eq_constraint=True,
            on_algo_ineq_constraint=True,
        )
        algo = ProjectedGradientDescentBase()
        algo_option = ProjectedGradientDescentBaseOption()
        estimator = WeightedLeastSquaresEstimator(3)

        actual = estimator.calc_estimate(
            qst,
            empi_dists,
            algo,
            algo_option,
            "ucm",
            "extraction",
            is_computation_time_required=True,
        )
        expected = [0, 0, 1 / np.sqrt(2)]
        assert actual.estimated_qoperation.is_physical()
        npt.assert_almost_equal(actual.estimated_var, expected, decimal=15)
        assert type(actual.computation_time) == float

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

        # on_para_eq_constraint=False
        qst, _ = get_test_data(
            on_para_eq_constraint=False,
            on_algo_eq_constraint=True,
            on_algo_ineq_constraint=True,
        )
        algo = ProjectedGradientDescentBase()
        algo_option = ProjectedGradientDescentBaseOption()
        estimator = WeightedLeastSquaresEstimator(4)

        actual = estimator.calc_estimate_sequence(
            qst,
            empi_dists_seq,
            algo,
            algo_option,
            "scm",
            "extraction",
            is_computation_time_required=True,
        )

        expected = [
            [1 / np.sqrt(2), 0, 0, 1 / np.sqrt(2)],
            [1 / np.sqrt(2), 1 / np.sqrt(2), 0, 0],
        ]
        for a, e in zip(actual.estimated_qoperation_sequence, expected):
            assert a.is_physical()
            npt.assert_almost_equal(a.to_stacked_vector(), e, decimal=10)
