import numpy as np
import pytest

from quara.objects.composite_system import CompositeSystem
from quara.objects.elemental_system import ElementalSystem
from quara.objects.matrix_basis import get_normalized_pauli_basis
from quara.objects.state import (
    State,
    get_x0_1q,
    get_x1_1q,
    get_y0_1q,
    get_y1_1q,
    get_z0_1q,
    get_z1_1q,
)
from quara.objects.povm import (
    Povm,
    get_x_measurement,
    get_y_measurement,
    get_z_measurement,
)

from quara.objects.qoperations import SetQOperations
from quara.protocol.qtomography.standard.standard_povmt import StandardPovmt


class TestStandardPovmt:
    def test_init_on_para_eq_constraint_false(self):
        # Array
        e_sys = ElementalSystem(0, get_normalized_pauli_basis())
        c_sys = CompositeSystem([e_sys])

        # |+><+|
        s_0 = 1 / np.sqrt(2) * np.array([1, 1, 0, 0], dtype=np.float64)
        state_0 = State(c_sys=c_sys, vec=s_0)
        # |+i><+i|
        s_1 = 1 / np.sqrt(2) * np.array([1, 0, 1, 0], dtype=np.float64)
        state_1 = State(c_sys=c_sys, vec=s_1)

        # |0><0|
        s_2 = 1 / np.sqrt(2) * np.array([1, 0, 0, 1], dtype=np.float64)
        state_2 = State(c_sys=c_sys, vec=s_2)

        # |1><1|
        s_3 = 1 / np.sqrt(2) * np.array([1, 0, 0, -1], dtype=np.float64)
        state_3 = State(c_sys=c_sys, vec=s_3)

        states = [state_0, state_1, state_2, state_3]

        # Act
        povmt = StandardPovmt(states, measurement_n=2, on_para_eq_constraint=False)
        assert povmt.num_variables == 4  # TODO

    @pytest.mark.skip("Working in Progress")
    def test_init_on_para_eq_constraint_true(self):
        # Array
        e_sys = ElementalSystem(0, get_normalized_pauli_basis())
        c_sys = CompositeSystem([e_sys])

        state_x = get_x0_1q(c_sys)
        state_y = get_y0_1q(c_sys)
        state_z = get_z0_1q(c_sys)
        states = [state_x, state_y, state_z]
        # Act
        povmt = StandardPovmt(states, measurement_n=2, on_para_eq_constraint=True)

        # Assert
        assert povmt.num_variables == 3  # TODO

    def test_init_exception(self):
        # Array
        e_sys0 = ElementalSystem(0, get_normalized_pauli_basis())
        c_sys0 = CompositeSystem([e_sys0])
        e_sys1 = ElementalSystem(1, get_normalized_pauli_basis())
        c_sys1 = CompositeSystem([e_sys1])

        state_x = get_x0_1q(c_sys0)
        state_y = get_y0_1q(c_sys0)
        state_z = get_z0_1q(c_sys1)  # invalid
        states = [state_x, state_y, state_z]

        # Act & Assert
        with pytest.raises(ValueError):
            _ = StandardPovmt(states, measurement_n=2, on_para_eq_constraint=False)
