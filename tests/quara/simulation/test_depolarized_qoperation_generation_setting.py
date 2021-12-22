import numpy as np
import numpy.testing as npt

from quara.objects import matrix_basis
from quara.objects.composite_system import CompositeSystem
from quara.objects.elemental_system import ElementalSystem
from quara.objects.mprocess import MProcess
from quara.objects.mprocess_typical import generate_mprocess_from_name
from quara.simulation.depolarized_qoperation_generation_setting import (
    DepolarizedQOperationGenerationSetting,
)


class TestDepolarizedQOperationGenerationSetting:
    def test_generate_mprocess(self):
        # Arrange
        e_sys = ElementalSystem(0, matrix_basis.get_normalized_pauli_basis())
        c_sys = CompositeSystem([e_sys])
        qoperation_base = generate_mprocess_from_name(c_sys, "z-type1")
        setting = DepolarizedQOperationGenerationSetting(c_sys, qoperation_base, 0.1)

        # Act
        actual = setting.generate_mprocess()

        # Assert
        assert type(actual) == MProcess
        assert len(actual.hss) == 2
        expected_hss = [
            np.array(
                [[0.5, 0, 0, 0.5], [0, 0, 0, 0], [0, 0, 0, 0], [0.45, 0, 0, 0.45]],
                dtype=np.float64,
            ),
            np.array(
                [[0.5, 0, 0, -0.5], [0, 0, 0, 0], [0, 0, 0, 0], [-0.45, 0, 0, 0.45]],
                dtype=np.float64,
            ),
        ]
        for a, e in zip(actual.hss, expected_hss):
            npt.assert_almost_equal(a, e, decimal=15)
