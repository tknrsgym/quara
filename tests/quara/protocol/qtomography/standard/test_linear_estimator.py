import time
from typing import List

import numpy as np
import numpy.testing as npt

from quara.objects.composite_system import CompositeSystem
from quara.objects.elemental_system import ElementalSystem
from quara.objects.matrix_basis import get_normalized_pauli_basis
from quara.objects.povm import (
    Povm,
    get_x_measurement,
    get_y_measurement,
    get_z_measurement,
)
from quara.objects.state import get_z0_1q
from quara.protocol.qtomography.standard.standard_qst import StandardQst
from quara.protocol.qtomography.standard.linear_estimator import LinearEstimator
from quara.utils.matrix_util import calc_mse


def get_test_data(on_para_eq_constraint=False):
    e_sys = ElementalSystem(0, get_normalized_pauli_basis())
    c_sys = CompositeSystem([e_sys])

    povm_x = get_x_measurement(c_sys)
    povm_y = get_y_measurement(c_sys)
    povm_z = get_z_measurement(c_sys)
    povms = [povm_x, povm_y, povm_z]

    qst = StandardQst(povms, on_para_eq_constraint=on_para_eq_constraint)

    return qst, c_sys


class TestLinearEstimator:
    def test_calc_estimate_var(self):
        qst, _ = get_test_data()
        empi_dists = [
            (10000, np.array([0.5, 0.5], dtype=np.float64)),
            (10000, np.array([0.5, 0.5], dtype=np.float64)),
            (10000, np.array([1, 0], dtype=np.float64)),
        ]

        estimator = LinearEstimator()

        # is_computation_time_required=True
        actual = estimator.calc_estimate_var(
            qst, empi_dists, is_computation_time_required=True
        )
        expected = [1 / np.sqrt(2), 0, 0, 1 / np.sqrt(2)]
        npt.assert_almost_equal(actual["estimate"], expected, decimal=15)
        assert type(actual["computation_time"]) == float

        # is_computation_time_required=False
        actual = estimator.calc_estimate_var(qst, empi_dists)
        expected = [1 / np.sqrt(2), 0, 0, 1 / np.sqrt(2)]
        npt.assert_almost_equal(actual["estimate"], expected, decimal=15)
        assert not "computation_time" in actual

    def test_calc_estimate_sequence_var(self):
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
        actual = estimator.calc_estimate_sequence_var(
            qst, empi_dists_seq, is_computation_time_required=True
        )
        print(actual)
        expected = [
            [1 / np.sqrt(2), 0, 0, 1 / np.sqrt(2)],
            [1 / np.sqrt(2), 0, 0, 1 / np.sqrt(2)],
        ]
        for a, e in zip(actual["estimate"], expected):
            npt.assert_almost_equal(a, e, decimal=15)
        assert len(actual["computation_time"]) == 2
        for a in actual["computation_time"]:
            assert type(a) == float

        # is_computation_time_required=False
        actual = estimator.calc_estimate_sequence_var(qst, empi_dists_seq)
        print(actual)
        expected = [
            [1 / np.sqrt(2), 0, 0, 1 / np.sqrt(2)],
            [1 / np.sqrt(2), 0, 0, 1 / np.sqrt(2)],
        ]
        for a, e in zip(actual["estimate"], expected):
            npt.assert_almost_equal(a, e, decimal=15)
        assert not "computation_time" in actual

    def test_scenario_on_para_eq_constraint_True(self):
        qst, c_sys = get_test_data(on_para_eq_constraint=True)

        # generate empi dists and calc estimate
        true_object = get_z0_1q(c_sys)
        num_data = [100, 1000, 10000, 100000]
        iterations = 2

        var_sequences = []

        for iteration in range(iterations):
            seeds = [iteration] * len(num_data)
            empi_dists_seq = qst.generate_empi_dists_sequence(
                true_object, num_data, seeds
            )

            estimator = LinearEstimator()
            var_sequence = estimator.calc_estimate_sequence_var(qst, empi_dists_seq)
            print(f"var_sequence={var_sequence}")

            """
            info = {
                "iteration": iteration,
                "empi_dists_seq": empi_dists_seq,
                "var_sequence": var_sequence,
            }
            print(info)
            """
            var_sequences.append(var_sequence["estimate"])
            for var in var_sequence["estimate"]:
                assert len(var) == 3

        # calc mse
        var_sequences_tmp = [list(var_sequence) for var_sequence in zip(*var_sequences)]
        actual = [
            calc_mse(var_sequence, [true_object.vec[1:]] * len(var_sequence))
            for var_sequence in var_sequences_tmp
        ]
        print(f"mse={actual}")
        expected = [4.000e-04, 6.5000e-04, 8.392e-05, 6.442e-07]
        npt.assert_almost_equal(actual, expected, decimal=15)

    def test_scenario_on_para_eq_constraint_False(self):
        qst, c_sys = get_test_data()

        # generate empi dists and calc estimate
        true_object = get_z0_1q(c_sys)
        num_data = [100, 1000, 10000, 100000]
        iterations = 2

        var_sequences = []

        for iteration in range(iterations):
            seeds = [iteration] * len(num_data)
            empi_dists_seq = qst.generate_empi_dists_sequence(
                true_object, num_data, seeds
            )

            estimator = LinearEstimator()
            var_sequence = estimator.calc_estimate_sequence_var(qst, empi_dists_seq)
            print(f"var_sequence={var_sequence}")

            """
            info = {
                "iteration": iteration,
                "empi_dists_seq": empi_dists_seq,
                "var_sequence": var_sequence,
            }
            print(info)
            """
            var_sequences.append(var_sequence["estimate"])
            for var in var_sequence["estimate"]:
                assert len(var) == 4

        # calc mse
        var_sequences_tmp = [list(var_sequence) for var_sequence in zip(*var_sequences)]
        actual = [
            calc_mse(var_sequence, [true_object.vec] * len(var_sequence))
            for var_sequence in var_sequences_tmp
        ]
        print(f"mse={actual}")
        expected = [4.000e-04, 6.5000e-04, 8.392e-05, 6.442e-07]
        npt.assert_almost_equal(actual, expected, decimal=15)
