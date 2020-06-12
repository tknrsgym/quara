from typing import List

import numpy as np
import numpy.testing as npt
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
from quara.objects.state import State, get_x0_1q, get_y0_1q, get_z0_1q, get_bell_2q
from quara.objects import qoperations as qope


class TestSetQOperations:
    def array_states_povms_gates(self):
        # Arrange
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
        # Arrange
        states, povms, gates = self.array_states_povms_gates()

        # Act
        sl_qope = qope.SetQOperations(states=states, povms=povms, gates=gates)

        # Assert
        assert sl_qope.states == states
        assert sl_qope.povms == povms
        assert sl_qope.gates == gates

    def test_init_exception(self):
        # Arrange
        states, povms, gates = self.array_states_povms_gates()

        # Act & Assert
        ng_states = [states[0], 1]
        with pytest.raises(TypeError):
            # TypeError: 'states' must be a list of State.
            _ = qope.SetQOperations(states=ng_states, povms=povms, gates=gates)

        # Act & Assert
        ng_povms = [states[0], povms[0], povms[1]]
        with pytest.raises(TypeError):
            # TypeError: 'povms' must be a list of Povm.
            _ = qope.SetQOperations(states=states, povms=ng_povms, gates=gates)

        # Act & Assert
        ng_gates = [gates[0], states[0], gates[2], gates[3]]
        with pytest.raises(TypeError):
            # TypeError: 'gates' must be a list of Gate.
            _ = qope.SetQOperations(states=states, povms=povms, gates=ng_gates)

    def test_setter(self):
        # Arrange
        states, povms, gates = self.array_states_povms_gates()

        new_states, new_povms, new_gates = self.array_states_povms_gates()

        sl_qope = qope.SetQOperations(states=states, povms=povms, gates=gates)

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
        # Arrange
        states, povms, gates = self.array_states_povms_gates()

        sl_qope = qope.SetQOperations(states=states, povms=povms, gates=gates)

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
        # Arrange
        _, povms, gates = self.array_states_povms_gates()

        e_sys = ElementalSystem(0, matrix_basis.get_normalized_pauli_basis())
        c_sys = CompositeSystem([e_sys])

        vec_1 = np.array([1 / np.sqrt(2), 1, 2, 3], dtype=np.float64)
        vec_2 = np.array([1, 2, 3, 4], dtype=np.float64)

        state_1 = State(c_sys=c_sys, vec=vec_1, is_physicality_required=False)
        state_2 = State(c_sys=c_sys, vec=vec_2, is_physicality_required=False)

        states = [state_1, state_2]

        # Case 1:
        sl_qope = qope.SetQOperations(states=states, povms=povms, gates=gates,)
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
        # Arrange
        state_1 = State(
            c_sys=c_sys,
            vec=vec_1,
            is_physicality_required=False,
            on_para_eq_constraint=True,
        )
        state_2 = State(
            c_sys=c_sys,
            vec=vec_2,
            is_physicality_required=False,
            on_para_eq_constraint=False,
        )
        states = [state_1, state_2]

        sl_qope = qope.SetQOperations(states=states, povms=povms, gates=gates)

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
        # Arrange
        _, povms, gates = self.array_states_povms_gates()

        e_sys = ElementalSystem(0, matrix_basis.get_normalized_pauli_basis())
        c_sys = CompositeSystem([e_sys])

        vec_1 = np.array([1 / np.sqrt(2), 1, 2, 3], dtype=np.float64)
        vec_2 = np.array([1, 2, 3, 4], dtype=np.float64)

        state_1 = State(c_sys=c_sys, vec=vec_1, is_physicality_required=False)
        state_2 = State(c_sys=c_sys, vec=vec_2, is_physicality_required=False)

        states = [state_1, state_2]

        # Case 1:
        sl_qope = qope.SetQOperations(states=states, povms=povms, gates=gates)
        # Act
        actual = sl_qope.var_states()

        # Assert
        expected = np.array([1, 2, 3, 2, 3, 4], dtype=np.float64)
        assert np.all(actual == expected)

        # Case 2:
        state_1 = State(
            c_sys=c_sys,
            vec=vec_1,
            is_physicality_required=False,
            on_para_eq_constraint=True,
        )
        state_2 = State(
            c_sys=c_sys,
            vec=vec_2,
            is_physicality_required=False,
            on_para_eq_constraint=False,
        )

        states = [state_1, state_2]
        sl_qope = qope.SetQOperations(states=states, povms=povms, gates=gates)

        # Act
        actual = sl_qope.var_states()

        # Assert
        expected = np.array([1, 2, 3, 1, 2, 3, 4], dtype=np.float64)
        assert np.all(actual == expected)

    def test_var_povms(self):
        # Arrange
        states, _, gates = self.array_states_povms_gates()

        e_sys = ElementalSystem(0, matrix_basis.get_normalized_pauli_basis())
        c_sys = CompositeSystem([e_sys])

        # POVM 1
        vecs = [
            np.array([2, 3, 5, 7], dtype=np.float64),
            np.array([11, 13, 17, 19], dtype=np.float64),
        ]

        povm_1 = Povm(c_sys, vecs, is_physical=False)
        povm_2 = Povm(c_sys, vecs, is_physical=False, on_para_eq_constraint=True)
        povm_3 = Povm(c_sys, vecs, is_physical=False, on_para_eq_constraint=False)
        povms = [povm_1, povm_2, povm_3]

        sl_qope = qope.SetQOperations(states=states, povms=povms, gates=gates)

        # Case 1:
        # Act
        actual = sl_qope.var_povm(0)

        # Assert
        expected = np.array([2, 3, 5, 7], dtype=np.float64)
        npt.assert_almost_equal(actual, expected, decimal=15)

        # Case 2:
        # Act
        actual = sl_qope.var_povm(1)

        # Assert
        expected = np.array([2, 3, 5, 7], dtype=np.float64)
        npt.assert_almost_equal(actual, expected, decimal=15)

        # Case 3:
        # Act
        actual = sl_qope.var_povm(2)

        # Assert
        expected = np.array([2, 3, 5, 7, 11, 13, 17, 19], dtype=np.float64)
        npt.assert_almost_equal(actual, expected, decimal=15)

        # Case 4: All
        # Act
        actual = sl_qope.var_povms()

        # Assert
        expected = np.array(
            [2, 3, 5, 7, 2, 3, 5, 7, 2, 3, 5, 7, 11, 13, 17, 19], dtype=np.float64
        )
        npt.assert_almost_equal(actual, expected, decimal=15)

    def test_var_gates(self):
        # Arrange
        states, povms, _ = self.array_states_povms_gates()

        e_sys = ElementalSystem(0, matrix_basis.get_normalized_pauli_basis())
        c_sys = CompositeSystem([e_sys])
        hs = np.array(
            [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, -1, 0], [0, 0, 0, -1]], dtype=np.float64
        )
        gate_1 = Gate(c_sys=c_sys, hs=hs, on_para_eq_constraint=True)
        gate_2 = Gate(c_sys=c_sys, hs=hs, on_para_eq_constraint=False)
        gates = [gate_1, gate_2]

        sl_qope = qope.SetQOperations(states=states, povms=povms, gates=gates)

        # Act
        actual = sl_qope.var_gate(0)

        # Assert
        expected = np.array([0, 1, 0, 0, 0, 0, -1, 0, 0, 0, 0, -1], dtype=np.float64)
        npt.assert_almost_equal(actual, expected, decimal=15)

        # Act
        actual = sl_qope.var_gate(1)

        # Assert
        expected = np.array(
            [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, -1, 0, 0, 0, 0, -1], dtype=np.float64
        )
        npt.assert_almost_equal(actual, expected, decimal=15)

        # Act
        actual = sl_qope.var_gates()

        # Assert
        expected = np.array(
            [
                0,
                1,
                0,
                0,
                0,
                0,
                -1,
                0,
                0,
                0,
                0,
                -1,
                1,
                0,
                0,
                0,
                0,
                1,
                0,
                0,
                0,
                0,
                -1,
                0,
                0,
                0,
                0,
                -1,
            ],
            dtype=np.float64,
        )
        npt.assert_almost_equal(actual, expected, decimal=15)

    def test_var_total(self):
        e_sys = ElementalSystem(0, matrix_basis.get_normalized_pauli_basis())
        c_sys = CompositeSystem([e_sys])

        # State
        vec_1 = np.array([1 / np.sqrt(2), 1, 2, 3], dtype=np.float64)
        vec_2 = np.array([1, 2, 3, 4], dtype=np.float64)

        state_1 = State(
            c_sys=c_sys, vec=vec_1, is_physical=False, on_para_eq_constraint=True
        )
        state_2 = State(
            c_sys=c_sys, vec=vec_2, is_physical=False, on_para_eq_constraint=False
        )
        states = [state_1, state_2]
        expected_states_var = [1, 2, 3] + [1, 2, 3, 4]

        # Gate
        hs = np.array(
            [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, -1, 0], [0, 0, 0, -1]], dtype=np.float64
        )
        gate_1 = Gate(c_sys=c_sys, hs=hs, on_para_eq_constraint=True)
        gate_2 = Gate(c_sys=c_sys, hs=hs, on_para_eq_constraint=False)
        gates = [gate_1, gate_2]

        expected_gates_var = [0, 1, 0, 0, 0, 0, -1, 0, 0, 0, 0, -1] + [
            1,
            0,
            0,
            0,
            0,
            1,
            0,
            0,
            0,
            0,
            -1,
            0,
            0,
            0,
            0,
            -1,
        ]

        # Povm
        vecs = [
            np.array([2, 3, 5, 7], dtype=np.float64),
            np.array([11, 13, 17, 19], dtype=np.float64),
        ]
        povm_1 = Povm(c_sys, vecs, is_physical=False, on_para_eq_constraint=True)
        povm_2 = Povm(c_sys, vecs, is_physical=False, on_para_eq_constraint=False)
        povms = [povm_1, povm_2]
        expected_povms_var = [2, 3, 5, 7] + [2, 3, 5, 7, 11, 13, 17, 19]

        sl_qope = qope.SetQOperations(states=states, povms=povms, gates=gates)

        # Act
        actual = sl_qope.var_total()

        # Assert
        expected = np.array(
            expected_states_var + expected_gates_var + expected_povms_var,
            dtype=np.float64,
        )
        npt.assert_almost_equal(actual, expected, decimal=15)

    def test_size_var(self):
        e_sys = ElementalSystem(0, matrix_basis.get_normalized_pauli_basis())
        c_sys = CompositeSystem([e_sys])

        # State
        vec_1 = np.array([1 / np.sqrt(2), 1, 2, 3], dtype=np.float64)
        vec_2 = np.array([1, 2, 3, 4], dtype=np.float64)

        state_1 = State(
            c_sys=c_sys, vec=vec_1, is_physical=False, on_para_eq_constraint=True
        )
        state_2 = State(
            c_sys=c_sys, vec=vec_2, is_physical=False, on_para_eq_constraint=False
        )
        states = [state_1, state_2]

        # Gate
        hs = np.array(
            [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, -1, 0], [0, 0, 0, -1]], dtype=np.float64
        )
        gate_1 = Gate(c_sys=c_sys, hs=hs, on_para_eq_constraint=True)
        gate_2 = Gate(c_sys=c_sys, hs=hs, on_para_eq_constraint=False)
        gates = [gate_1, gate_2]

        # Povm
        vecs = [
            np.array([2, 3, 5, 7], dtype=np.float64),
            np.array([11, 13, 17, 19], dtype=np.float64),
        ]
        povm_1 = Povm(c_sys, vecs, is_physical=False, on_para_eq_constraint=True)
        povm_2 = Povm(c_sys, vecs, is_physical=False, on_para_eq_constraint=False)
        povms = [povm_1, povm_2]

        sl_qope = qope.SetQOperations(states=states, povms=povms, gates=gates)

        # Case 1-1: State0
        # Act
        actual = sl_qope.size_var_state(0)

        # Assert
        expected = 3
        assert actual == expected

        # Case 1-2: State1
        # Act
        actual = sl_qope.size_var_state(1)

        # Assert
        expected = 4
        assert actual == expected

        # Case 1-3: State All
        # Act
        actual = sl_qope.size_var_states()

        # Assert
        expected = 3 + 4
        assert actual == expected

        # Case 2-1: Gate0
        # Act
        actual = sl_qope.size_var_gate(0)

        # Assert
        expected = 12
        assert actual == expected

        # Case 2-2: Gate1
        # Act
        actual = sl_qope.size_var_gate(1)

        # Assert
        expected = 16
        assert actual == expected

        # Case 1-3: Gate All
        # Act
        actual = sl_qope.size_var_gates()

        # Assert
        expected = 12 + 16
        assert actual == expected

        # Case 3-1: Povm0
        # Act
        actual = sl_qope.size_var_povm(0)

        # Assert
        expected = 4
        assert actual == expected

        # Case 3-2: Povm1
        # Act
        actual = sl_qope.size_var_povm(1)

        # Assert
        expected = 8
        assert actual == expected

        # Case 3-3: Povm All
        # Act
        actual = sl_qope.size_var_povms()

        # Assert
        expected = 4 + 8
        assert actual == expected

        # Case 4: Total
        actual = sl_qope.size_var_total()

        # Assert
        expected = 3 + 4 + 12 + 16 + 4 + 8
        assert actual == expected

    def test_dim(self):
        # Arrange
        e_sys = ElementalSystem(0, matrix_basis.get_normalized_pauli_basis())
        c_sys_1q = CompositeSystem([e_sys])

        e_sys0 = ElementalSystem(0, matrix_basis.get_normalized_pauli_basis())
        e_sys1 = ElementalSystem(1, matrix_basis.get_normalized_pauli_basis())
        c_sys_2q = CompositeSystem([e_sys0, e_sys1])

        # State
        state_1q = get_x0_1q(c_sys_1q)
        state_2q = get_bell_2q(c_sys_2q)
        states = [state_1q, state_2q]

        # Gate
        gate_1q = get_x(c_sys_1q)
        gate_2q = get_cnot(c_sys_2q, e_sys0)

        gates = [gate_2q, gate_1q]

        # POVM
        povm_1q = get_x_measurement(c_sys_1q)
        povm_2q = get_xx_measurement(c_sys_2q)

        povms = [povm_2q, povm_2q, povm_1q]

        sl_qope = qope.SetQOperations(states=states, gates=gates, povms=povms)

        # Case 1: State
        # Act
        actual = sl_qope.dim_state(0)
        # Assert
        expected = 2
        assert actual == expected

        # Act
        actual = sl_qope.dim_state(1)
        # Assert
        expected = 4
        assert actual == expected

        # Case 2: Gate
        # Act
        actual = sl_qope.dim_gate(0)
        # Assert
        expected = 4
        assert actual == expected

        # Act
        actual = sl_qope.dim_gate(1)
        # Assert
        expected = 2
        assert actual == expected

        # Case 3: POVM
        # Act
        actual = sl_qope.dim_povm(1)
        # Assert
        expected = 4
        assert actual == expected

        # Act
        actual = sl_qope.dim_povm(2)
        # Assert
        expected = 2
        assert actual == expected
