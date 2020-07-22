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
from quara.protocol.qtomography.standard.linear_estimator import LinearEstimator
from quara.protocol.qtomography.standard.standard_qst import StandardQst


class TestStandardQTomographyEstimator:
    def test_calc_estimate_qoperation(self):
        e_sys = ElementalSystem(0, get_normalized_pauli_basis())
        c_sys = CompositeSystem([e_sys])

        povm_x = get_x_measurement(c_sys)
        povm_y = get_y_measurement(c_sys)
        povm_z = get_z_measurement(c_sys)
        povms = [povm_x, povm_y, povm_z]

        qst = StandardQst(povms, on_para_eq_constraint=False)

        est = LinearEstimator()
        empi_dists = [
            (10, np.array([0.5, 0.5], dtype=np.float64)),
            (10, np.array([0.5, 0.5], dtype=np.float64)),
            (10, np.array([1, 0], dtype=np.float64)),
        ]

        # is_computation_time_required=True
        actual = est.calc_estimate_qoperation(
            qst, empi_dists, is_computation_time_required=True
        )
        expected = np.array([1, 0, 0, 1], dtype=np.float64) / np.sqrt(2)
        npt.assert_almost_equal(actual["estimate"].vec, expected, decimal=15)
        assert type(actual["computation_time"]) == float

        # is_computation_time_required=False
        actual = est.calc_estimate_qoperation(qst, empi_dists)
        expected = np.array([1, 0, 0, 1], dtype=np.float64) / np.sqrt(2)
        npt.assert_almost_equal(actual["estimate"].vec, expected, decimal=15)
        assert not "computation_time" in actual

    def test_calc_estimate_sequence_qoperation(self):
        e_sys = ElementalSystem(0, get_normalized_pauli_basis())
        c_sys = CompositeSystem([e_sys])

        povm_x = get_x_measurement(c_sys)
        povm_y = get_y_measurement(c_sys)
        povm_z = get_z_measurement(c_sys)
        povms = [povm_x, povm_y, povm_z]

        qst = StandardQst(povms, on_para_eq_constraint=False)

        est = LinearEstimator()
        empi_dists_sequence = [
            [
                (10, np.array([0.5, 0.5], dtype=np.float64)),
                (10, np.array([0.5, 0.5], dtype=np.float64)),
                (10, np.array([1, 0], dtype=np.float64)),
            ],
            [
                (10, np.array([0.5, 0.5], dtype=np.float64)),
                (10, np.array([0.5, 0.5], dtype=np.float64)),
                (10, np.array([1, 0], dtype=np.float64)),
            ],
        ]

        # is_computation_time_required=True
        actual = est.calc_estimate_sequence_qoperation(
            qst, empi_dists_sequence, is_computation_time_required=True
        )
        expected = np.array([1, 0, 0, 1], dtype=np.float64) / np.sqrt(2)
        for a in actual["estimate"]:
            npt.assert_almost_equal(a.vec, expected, decimal=15)
        assert len(actual["computation_time"]) == 2
        for a in actual["computation_time"]:
            assert type(a) == float

        # is_computation_time_required=False
        actual = est.calc_estimate_sequence_qoperation(qst, empi_dists_sequence)
        expected = np.array([1, 0, 0, 1], dtype=np.float64) / np.sqrt(2)
        for a in actual["estimate"]:
            npt.assert_almost_equal(a.vec, expected, decimal=15)
        assert not "computation_time" in actual
