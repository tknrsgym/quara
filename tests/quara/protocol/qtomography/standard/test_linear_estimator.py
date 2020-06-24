import numpy as np

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


class TestLinearEstimator:
    def test_scenario(self):
        # setup system
        e_sys = ElementalSystem(0, get_normalized_pauli_basis())
        c_sys = CompositeSystem([e_sys])

        povm_x = get_x_measurement(c_sys)
        povm_y = get_y_measurement(c_sys)
        povm_z = get_z_measurement(c_sys)
        povms = [povm_x, povm_y, povm_z]

        qst = StandardQst(povms, on_para_eq_constraint=False)

        # generate empi dists
        true_object = get_z0_1q(c_sys)
        empi_dists = qst.generate_empi_dists(true_object, 100)
        print(f"empi_dists={empi_dists}")

        # estimate
        estimator = LinearEstimator()
        var = estimator.calc_estimate_var(qst, empi_dists)
        print(f"var={var}")
