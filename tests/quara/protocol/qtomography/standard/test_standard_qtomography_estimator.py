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
from quara.protocol.qtomography.standard.standard_qst import StandardQst
from quara.protocol.qtomography.standard.standard_qtomography_estimator import (
    StandardQTomographyEstimationResult,
)


class TestStandardQTomographyEstimationResult:
    def get_estimation_result(self):
        e_sys = ElementalSystem(0, get_normalized_pauli_basis())
        c_sys = CompositeSystem([e_sys])
        povm_x = get_x_povm(c_sys)
        povm_y = get_y_povm(c_sys)
        povm_z = get_z_povm(c_sys)
        povms = [povm_x, povm_y, povm_z]
        qtomography = StandardQst(povms, on_para_eq_constraint=False)

        estimated_var_sequence = [
            np.array([1, 0, 0, 1], dtype=np.float64) / np.sqrt(2),
            np.array([1, 0, 0, 1], dtype=np.float64) / np.sqrt(2),
        ]
        computation_times = [0.1, 0.2]

        result = StandardQTomographyEstimationResult(
            estimated_var_sequence, computation_times, qtomography._template_qoperation
        )

        return result

    def test_access_estimated_var(self):
        result = self.get_estimation_result()
        actual = result.estimated_var
        expected = np.array([1, 0, 0, 1], dtype=np.float64) / np.sqrt(2)
        npt.assert_almost_equal(actual, expected, decimal=15)

    def test_access_estimated_var_sequence(self):
        result = self.get_estimation_result()
        actual = result.estimated_var_sequence
        expected = [
            np.array([1, 0, 0, 1], dtype=np.float64) / np.sqrt(2),
            np.array([1, 0, 0, 1], dtype=np.float64) / np.sqrt(2),
        ]
        for a, e in zip(actual, expected):
            npt.assert_almost_equal(a, e, decimal=15)

    def test_access_estimated_qoperation(self):
        result = self.get_estimation_result()
        actual = result.estimated_qoperation
        expected = np.array([1, 0, 0, 1], dtype=np.float64) / np.sqrt(2)
        npt.assert_almost_equal(actual.to_stacked_vector(), expected, decimal=15)

    def test_access_estimated_qoperation_sequence(self):
        result = self.get_estimation_result()
        actual = result.estimated_qoperation_sequence
        expected = [
            np.array([1, 0, 0, 1], dtype=np.float64) / np.sqrt(2),
            np.array([1, 0, 0, 1], dtype=np.float64) / np.sqrt(2),
        ]
        for a, e in zip(actual, expected):
            npt.assert_almost_equal(a.to_stacked_vector(), e, decimal=15)
