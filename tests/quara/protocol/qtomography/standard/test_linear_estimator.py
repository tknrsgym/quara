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


def get_test_data():
    e_sys = ElementalSystem(0, get_normalized_pauli_basis())
    c_sys = CompositeSystem([e_sys])

    povm_x = get_x_measurement(c_sys)
    povm_y = get_y_measurement(c_sys)
    povm_z = get_z_measurement(c_sys)
    povms = [povm_x, povm_y, povm_z]

    qst = StandardQst(povms, on_para_eq_constraint=False)

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
        actual = estimator.calc_estimate_var(qst, empi_dists)
        expected = [1 / np.sqrt(2), 0, 0, 1 / np.sqrt(2)]
        npt.assert_almost_equal(actual, expected, decimal=15)

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
        actual = estimator.calc_estimate_sequence_var(qst, empi_dists_seq)
        print(actual)
        expected = [
            [1 / np.sqrt(2), 0, 0, 1 / np.sqrt(2)],
            [1 / np.sqrt(2), 0, 0, 1 / np.sqrt(2)],
        ]
        for a, e in zip(actual, expected):
            npt.assert_almost_equal(a, e, decimal=15)

    def test_scenario(self):
        qst, c_sys = get_test_data()

        # generate empi dists and calc estimate
        true_object = get_z0_1q(c_sys)
        num_data = [100, 1000, 10000, 100000]
        iterations = 2

        var_sequences = []

        start = time.time()
        for iteration in range(iterations):
            seeds = [iteration] * len(num_data)
            empi_dists_seq = qst.generate_empi_dists_sequence(
                true_object, num_data, seeds
            )

            estimator = LinearEstimator()
            var_sequence = estimator.calc_estimate_sequence_var(qst, empi_dists_seq)
            print(f"var_sequence={var_sequence}")

            info = {
                "iteration": iteration,
                "empi_dists_seq": empi_dists_seq,
                "var_sequence": var_sequence,
            }
            # print(info)
            var_sequences.append(var_sequence)

        end = time.time()
        # print(f"time(s)={end - start}")

        # calc mse
        var_sequences_tmp = [list(var_sequence) for var_sequence in zip(*var_sequences)]
        mses = [
            calc_mse(var_sequence, [true_object.vec] * len(var_sequence))
            for var_sequence in var_sequences_tmp
        ]
        print(f"mse={mses}")
        # assert False
