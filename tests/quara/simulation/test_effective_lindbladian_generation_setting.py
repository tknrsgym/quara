from abc import abstractmethod

import numpy as np
import numpy.testing as npt
import pytest


from quara.simulation.effective_lindbladian_generation_setting import (
    EffectiveLindbladianGenerationSetting,
)
from quara.objects import matrix_basis
from quara.objects.composite_system import CompositeSystem
from quara.objects.elemental_system import ElementalSystem
from quara.objects.effective_lindbladian import EffectiveLindbladian
from quara.objects.state import get_z0_1q


class TestEffectiveLindbladianGenerationSetting:
    def test_init_error(self):
        e_sys = ElementalSystem(0, matrix_basis.get_normalized_pauli_basis())
        c_sys = CompositeSystem([e_sys])
        qoperation_base = get_z0_1q(c_sys)
        lindbladian_base = 1

        ## Test that "lindbladian_base" is not EffectiveLindbladian or str
        with pytest.raises(TypeError):
            EffectiveLindbladianGenerationSetting(
                c_sys, qoperation_base, lindbladian_base
            )

    def test_access_lindbladian_base(self):
        ## init
        # Arrange
        e_sys = ElementalSystem(0, matrix_basis.get_normalized_pauli_basis())
        c_sys = CompositeSystem([e_sys])
        qoperation_base = get_z0_1q(c_sys)
        hs = np.array(
            [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]], dtype=np.float64
        )
        lindbladian_base = EffectiveLindbladian(c_sys, hs)

        # Act
        actual = EffectiveLindbladianGenerationSetting(
            c_sys, qoperation_base, lindbladian_base
        )

        # Assert
        assert actual.lindbladian_base == lindbladian_base

        ## Test that "lindbladian_base" cannot be updated
        with pytest.raises(AttributeError):
            actual.lindbladian_base = lindbladian_base
