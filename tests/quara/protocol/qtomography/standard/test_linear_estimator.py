import numpy as np

from quara.objects.composite_system import CompositeSystem
from quara.objects.elemental_system import ElementalSystem
from quara.objects.povm import (
    Povm,
    get_x_measurement,
    get_y_measurement,
    get_z_measurement,
)
from quara.objects.state import get_z0_1q
from quara.protocol.qtomography.standard.standard_qst import StandardQst
from quara.protocol.qtomography.standard.linear_estimator import LinearEstimator
from quara.objects.matrix_basis import get_normalized_pauli_basis
from quara.qcircuit.experiment import Experiment


class TestLinearEstimator:
    def test_scenario(self):
        e_sys = ElementalSystem(1, get_normalized_pauli_basis())
        c_sys = CompositeSystem([e_sys])

        povm_x = get_x_measurement(c_sys)
        povm_y = get_y_measurement(c_sys)
        povm_z = get_z_measurement(c_sys)
        povms = [povm_x, povm_y, povm_z]

        qst = StandardQst(povms, on_para_eq_constraint=True)
        print(qst.calc_vecB())
        print(qst.calc_matA())

        # true_object = get_z0_1q(c_sys)
        # calc_prob_dist_for_povm_x = qst.calc_prob_dist(2, get_z0_1q(c_sys))
        # print(calc_prob_dist_for_povm_x)

        """
        empi_dists_sequences_tmp = []
        for index, _ in enumerate(experiment.schedules):
            empi_dists = qst.generate_empi_dist(index, true_object, [100, 1000])
            empi_dists_sequences_tmp.append(empi_dists)
        print(empi_dists_sequences_tmp)
        for empi_dists in empi_dists_sequences_tmp:
            for empi_dist in empi_dists:
                empi_dist
        """

        empi_dists_sequence = [[0.45, 0.55], [0.51, 0.49], [1.0, 0.0]]

        estimator = LinearEstimator()
        estimator.calc_estimates_sequence_var(
            qst, empi_dists_sequence, on_para_eq_constraint=False
        )

        # assert False
