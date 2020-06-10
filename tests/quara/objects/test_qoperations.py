from typing import List

import numpy as np
import pytest

from quara.objects import matrix_basis
from quara.objects.composite_system import CompositeSystem
from quara.objects.elemental_system import ElementalSystem
from quara.objects.gate import Gate, get_h, get_i, get_x, get_cnot, get_swap, get_cz
from quara.objects.povm import (
    Povm,
    get_x_measurement,
    get_y_measurement,
    get_z_measurement,
    get_xx_measurement,
    get_xy_measurement,
    get_yy_measurement,
    get_zz_measurement,
)
from quara.objects.state import State, get_x0_1q, get_y0_1q, get_z0_1q
from quara.objects import qoperations as qope


class TestSetListQOperation:
    def array_states_povms_gates(self):
        # Array
        e_sys = ElementalSystem(0, matrix_basis.get_comp_basis())
        c_sys = CompositeSystem([e_sys])
        # State
        state_0 = get_x0_1q(c_sys)
        state_1 = get_y0_1q(c_sys)
        states = [state_0, state_1]

        # POVM
        povm_0 = get_x_measurement(c_sys)
        povm_1 = get_x_measurement(c_sys)
        povm_2 = get_x_measurement(c_sys)
        povms = [povm_0, povm_1, povm_2]

        # Gate
        gate_0 = get_i(c_sys)
        gate_1 = get_h(c_sys)
        gate_2 = get_x(c_sys)
        gate_3 = get_x(c_sys)
        gates = [gate_0, gate_1, gate_2, gate_3]
        return states, povms, gates

    def test_init(self):
        # Array
        states, povms, gates = self.array_states_povms_gates()

        # Act
        sl_qope = qope.SetListQOperation(states=states, povms=povms, gates=gates)

        # Assert
        assert sl_qope.states == states
        assert sl_qope.povms == povms
        assert sl_qope.gates == gates

    def test_init_exception(self):
        # Array
        states, povms, gates = self.array_states_povms_gates()

        # Act & Assert
        ng_states = [states[0], 1]
        with pytest.raises(TypeError):
            # TypeError: 'states' must be a list of State.
            _ = qope.SetListQOperation(states=ng_states, povms=povms, gates=gates)

        # Act & Assert
        ng_povms = [states[0], povms[0], povms[1]]
        with pytest.raises(TypeError):
            # TypeError: 'povms' must be a list of Povm.
            _ = qope.SetListQOperation(states=states, povms=ng_povms, gates=gates)

        # Act & Assert
        ng_gates = [gates[0], states[0], gates[2], gates[3]]
        with pytest.raises(TypeError):
            # TypeError: 'gates' must be a list of Gate.
            _ = qope.SetListQOperation(states=states, povms=povms, gates=ng_gates)

    def test_setter(self):
        # Array
        states, povms, gates = self.array_states_povms_gates()

        new_states, new_povms, new_gates = self.array_states_povms_gates()

        sl_qope = qope.SetListQOperation(states=states, povms=povms, gates=gates)

        # Act & Assert
        ng_states = [states[0], 1]
        with pytest.raises(TypeError):
            # TypeError: 'states' must be a list of State.
            sl_qope.states = ng_states
        assert sl_qope.states == states

        # Act & Assert
        sl_qope.states = new_states
        assert sl_qope.states == new_states

        # Act & Assert
        ng_povms = [states[0], povms[0], povms[1]]
        with pytest.raises(TypeError):
            # TypeError: 'povms' must be a list of Povm.
            sl_qope.povms = ng_povms
        assert sl_qope.povms == povms

        # Act & Assert
        sl_qope.povms = new_povms
        assert sl_qope.povms == new_povms

        # Act & Assert
        ng_gates = [gates[0], states[0], gates[2], gates[3]]
        with pytest.raises(TypeError):
            # TypeError: 'gates' must be a list of Gate.
            sl_qope.gates = ng_gates
        assert sl_qope.gates == gates

        # Act & Assert
        sl_qope.gates = new_gates
        assert sl_qope.gates == new_gates

    def test_num(self):
        states, povms, gates = self.array_states_povms_gates()

        sl_qope = qope.SetListQOperation(states=states, povms=povms, gates=gates)

        # Act & Assert
        expected = len(states)
        assert sl_qope.num_states() == expected

        # Act & Assert
        expected = len(povms)
        assert sl_qope.num_povms() == expected

        # Act & Assert
        expected = len(gates)
        assert sl_qope.num_gates() == expected

    def test_var_state(self):
        # Array
        _, povms, gates = self.array_states_povms_gates()

        e_sys = ElementalSystem(0, matrix_basis.get_normalized_pauli_basis())
        c_sys = CompositeSystem([e_sys])

        vec_1 = np.array([1 / np.sqrt(2), 1, 2, 3], dtype=np.float64)
        vec_2 = np.array([1, 2, 3, 4], dtype=np.float64)

        state_1 = State(c_sys=c_sys, vec=vec_1, is_physical=False)
        state_2 = State(c_sys=c_sys, vec=vec_2, is_physical=False)

        states = [state_1, state_2]

        # Case 1:
        sl_qope = qope.SetListQOperation(states=states, povms=povms, gates=gates,)
        # Act
        actual = sl_qope.var_state(0)

        # Assert
        expected = np.array([1, 2, 3], dtype=np.float64)
        assert np.all(actual == expected)

        # Act
        actual = sl_qope.var_state(1)

        # Assert
        expected = np.array([2, 3, 4], dtype=np.float64)
        assert np.all(actual == expected)

        # # Case 2:
        # Array
        state_1 = State(
            c_sys=c_sys, vec=vec_1, is_physical=False, on_para_eq_constraint=True
        )
        state_2 = State(
            c_sys=c_sys, vec=vec_2, is_physical=False, on_para_eq_constraint=False
        )
        states = [state_1, state_2]

        sl_qope = qope.SetListQOperation(states=states, povms=povms, gates=gates)

        # Act
        actual = sl_qope.var_state(0)

        # Assert
        expected = np.array([1, 2, 3], dtype=np.float64)
        assert np.all(actual == expected)

        # Act
        actual = sl_qope.var_state(1)

        # Assert
        expected = np.array([1, 2, 3, 4], dtype=np.float64)
        assert np.all(actual == expected)

    def test_var_states(self):
        # Array
        _, povms, gates = self.array_states_povms_gates()

        e_sys = ElementalSystem(0, matrix_basis.get_normalized_pauli_basis())
        c_sys = CompositeSystem([e_sys])

        vec_1 = np.array([1 / np.sqrt(2), 1, 2, 3], dtype=np.float64)
        vec_2 = np.array([1, 2, 3, 4], dtype=np.float64)

        state_1 = State(c_sys=c_sys, vec=vec_1, is_physical=False)
        state_2 = State(c_sys=c_sys, vec=vec_2, is_physical=False)

        states = [state_1, state_2]

        # Case 1:
        sl_qope = qope.SetListQOperation(states=states, povms=povms, gates=gates)
        # Act
        actual = sl_qope.var_states()

        # Assert
        expected = np.array([1, 2, 3, 2, 3, 4], dtype=np.float64)
        assert np.all(actual == expected)

        # Case 2:
        state_1 = State(
            c_sys=c_sys, vec=vec_1, is_physical=False, on_para_eq_constraint=True
        )
        state_2 = State(
            c_sys=c_sys, vec=vec_2, is_physical=False, on_para_eq_constraint=False
        )

        states = [state_1, state_2]
        sl_qope = qope.SetListQOperation(states=states, povms=povms, gates=gates)

        # Act
        actual = sl_qope.var_states()

        # Assert
        expected = np.array([1, 2, 3, 1, 2, 3, 4], dtype=np.float64)
        assert np.all(actual == expected)

    def test_var_povms(self):
        # Array
        states, povms, gates = self.array_states_povms_gates()
        sl_qope = qope.SetListQOperation(states=states, povms=povms, gates=gates)
        actual = sl_qope.var_povm(1)
        actual = sl_qope.var_povms()

        # TODO

    def test_var_gates(self):
        # Array
        states, povms, gates = self.array_states_povms_gates()
        sl_qope = qope.SetListQOperation(states=states, povms=povms, gates=gates)
        actual = sl_qope.var_gate(1)
        actual = sl_qope.var_gates()

        # TODO