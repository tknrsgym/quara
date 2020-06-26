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
        empi_dists_seq = qst.generate_empi_dists_sequence(
            true_object, [100, 1000, 10000]
        )
        print(f"empi_dists_seq={empi_dists_seq}")

        # estimate
        estimator = LinearEstimator()
        var_sequence = estimator.calc_estimate_sequence_var(qst, empi_dists_seq)
        print(f"estimate var={var_sequence}")
        print(f"true var={true_object.vec}")

        mses = [calc_mse(var, true_object.vec) for var in var_sequence]
        print(f"mse={mses}")


def calc_mse(a: np.array, b: np.array) -> np.float64:
    return ((a - b) ** 2).mean(axis=0)
