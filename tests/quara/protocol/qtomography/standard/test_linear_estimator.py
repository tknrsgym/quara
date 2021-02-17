import time
from typing import List

import numpy as np
import numpy.testing as npt

from quara.objects.composite_system import CompositeSystem
from quara.objects.elemental_system import ElementalSystem
from quara.objects.matrix_basis import get_normalized_pauli_basis
from quara.objects.povm import (
    get_x_povm,
    get_y_povm,
    get_z_povm,
)
from quara.objects.state import get_z0_1q
from quara.protocol.qtomography.standard.standard_qst import StandardQst
from quara.protocol.qtomography.standard.linear_estimator import LinearEstimator
from quara.utils.matrix_util import calc_se


def get_test_data(on_para_eq_constraint=False):
    e_sys = ElementalSystem(0, get_normalized_pauli_basis())
    c_sys = CompositeSystem([e_sys])

    povm_x = get_x_povm(c_sys)
    povm_y = get_y_povm(c_sys)
    povm_z = get_z_povm(c_sys)
    povms = [povm_x, povm_y, povm_z]

    qst = StandardQst(povms, on_para_eq_constraint=on_para_eq_constraint, seed=7)

    return qst, c_sys


class TestLinearEstimator:
    def test_calc_estimate(self):
        qst, _ = get_test_data()
        empi_dists = [
            (10000, np.array([0.5, 0.5], dtype=np.float64)),
            (10000, np.array([0.5, 0.5], dtype=np.float64)),
            (10000, np.array([1, 0], dtype=np.float64)),
        ]

        estimator = LinearEstimator()

        # is_computation_time_required=True
        actual = estimator.calc_estimate(
            qst, empi_dists, is_computation_time_required=True
        )
        expected = [1 / np.sqrt(2), 0, 0, 1 / np.sqrt(2)]
        assert actual.estimated_qoperation.is_physical()
        npt.assert_almost_equal(actual.estimated_var, expected, decimal=15)
        assert type(actual.computation_time) == float

        # is_computation_time_required=False
        actual = estimator.calc_estimate(qst, empi_dists)
        expected = [1 / np.sqrt(2), 0, 0, 1 / np.sqrt(2)]
        assert actual.estimated_qoperation.is_physical()
        npt.assert_almost_equal(actual.estimated_var, expected, decimal=15)
        assert actual.computation_time == None

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

        estimator = LinearEstimator()

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

            estimator = LinearEstimator()
            result = estimator.calc_estimate_sequence(qst, empi_dists_seq)
            result_sequence.append(result.estimated_var_sequence)
            for var in result.estimated_var_sequence:
                assert len(var) == 3
            assert len(result.estimated_qoperation_sequence) == 4

        # calc mse
        result_sequences_tmp = [list(result) for result in zip(*result_sequence)]
        actual = [
            calc_se(result, [true_object.vec[1:]] * len(result)) / len(result)
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

            estimator = LinearEstimator()
            result = estimator.calc_estimate_sequence(qst, empi_dists_seq)
            result_sequence.append(result.estimated_var_sequence)
            for var in result.estimated_var_sequence:
                assert len(var) == 4
            assert len(result.estimated_qoperation_sequence) == 4

        # calc mse
        result_sequences_tmp = [list(result) for result in zip(*result_sequence)]
        actual = [
            calc_se(result, [true_object.vec] * len(result)) / len(result)
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
