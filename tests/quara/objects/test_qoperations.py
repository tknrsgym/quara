import numpy as np
import numpy.testing as npt
import pytest

from quara.objects import matrix_basis
from quara.objects import mprocess
from quara.objects.composite_system import CompositeSystem
from quara.objects.elemental_system import ElementalSystem
from quara.objects.gate import Gate, get_h, get_i, get_x, get_cnot
from quara.objects.povm import (
    Povm,
    get_x_povm,
    get_xx_povm,
)
from quara.objects.state import State, get_x0_1q, get_y0_1q, get_bell_2q
from quara.objects import qoperations as qope
from quara.objects.mprocess import MProcess
from quara.objects.qoperation_typical import (
    generate_qoperation_object,
)
from quara.objects.composite_system_typical import generate_composite_system


class TestSetQOperations:
    def arrange_qoperations(self):
        # Arrange
        e_sys = ElementalSystem(0, matrix_basis.get_comp_basis())
        c_sys = CompositeSystem([e_sys])
        # State
        state_0 = get_x0_1q(c_sys)
        state_1 = get_y0_1q(c_sys)
        states = [state_0, state_1]

        # POVM
        povm_0 = get_x_povm(c_sys)
        povm_1 = get_x_povm(c_sys)
        povm_2 = get_x_povm(c_sys)
        povms = [povm_0, povm_1, povm_2]

        # Gate
        gate_0 = get_i(c_sys)
        gate_1 = get_h(c_sys)
        gate_2 = get_x(c_sys)
        gate_3 = get_x(c_sys)
        gates = [gate_0, gate_1, gate_2, gate_3]

        # MProcess
        c_sys_1 = generate_composite_system(mode="qubit", num=1, ids_esys=[1])
        names = ["x-type1", "y-type1", "z-type1"]
        mprocesses = [
            generate_qoperation_object(
                mode="mprocess", object_name="mprocess", name=name, c_sys=c_sys_1
            )
            for name in names
        ]

        return states, povms, gates, mprocesses

    def test_init(self):
        # Arrange
        states, povms, gates, mprocesses = self.arrange_qoperations()

        # Act
        sl_qope = qope.SetQOperations(
            states=states, povms=povms, gates=gates, mprocesses=mprocesses
        )

        # Assert
        assert sl_qope.states == states
        assert sl_qope.povms == povms
        assert sl_qope.gates == gates
        assert sl_qope.mprocesses == mprocesses

    def test_init_empty(self):
        states, povms, gates, mprocesses = self.arrange_qoperations()

        # Act
        sl_qope = qope.SetQOperations(povms=povms, gates=gates, mprocesses=mprocesses)

        # Assert
        assert sl_qope.states == []

        # Act
        sl_qope = qope.SetQOperations(states=states, gates=gates, mprocesses=mprocesses)

        # Assert
        assert sl_qope.povms == []

        # Act
        sl_qope = qope.SetQOperations(states=states, povms=povms, mprocesses=mprocesses)

        # Assert
        assert sl_qope.gates == []

        # Act
        sl_qope = qope.SetQOperations(states=states, povms=povms, gates=gates)

        # Assert
        assert sl_qope.mprocesses == []

    def test_init_exception(self):
        # Arrange
        states, povms, gates, mprocesses = self.arrange_qoperations()

        # Act & Assert
        ng_states = [states[0], 1]
        with pytest.raises(TypeError):
            # TypeError: 'states' must be a list of State.
            _ = qope.SetQOperations(
                states=ng_states, povms=povms, gates=gates, mprocesses=mprocesses
            )

        # Act & Assert
        ng_povms = [states[0], povms[0], povms[1]]
        with pytest.raises(TypeError):
            # TypeError: 'povms' must be a list of Povm.
            _ = qope.SetQOperations(
                states=states, povms=ng_povms, gates=gates, mprocesses=mprocesses
            )

        # Act & Assert
        ng_gates = [gates[0], states[0], gates[2], gates[3]]
        with pytest.raises(TypeError):
            # TypeError: 'gates' must be a list of Gate.
            _ = qope.SetQOperations(
                states=states, povms=povms, gates=ng_gates, mprocesses=mprocesses
            )

        # Act & Assert
        ng_mprocesses = [mprocesses[0], states[0], mprocesses[1], mprocesses[2]]
        with pytest.raises(TypeError):
            # TypeError: 'mprocesses' must be a list of MProcess.
            _ = qope.SetQOperations(
                states=states, povms=povms, gates=gates, mprocesses=ng_mprocesses
            )

    def test_setter(self):
        # Arrange
        states, povms, gates, mprocesses = self.arrange_qoperations()

        new_states, new_povms, new_gates, new_mprocesses = self.arrange_qoperations()

        sl_qope = qope.SetQOperations(
            states=states, povms=povms, gates=gates, mprocesses=mprocesses
        )

        # Act & Assert
        ng_states = [states[0], 1]
        with pytest.raises(TypeError):
            # TypeError: 'states' must be a list of State.
            sl_qope.states = ng_states
        assert sl_qope.states == states

        # Act & Assert
        new_states = [states[1], states[0]]
        sl_qope.states = new_states
        assert sl_qope.states == new_states

        # Act & Assert
        ng_povms = [states[0], povms[0], povms[1]]
        with pytest.raises(TypeError):
            # TypeError: 'povms' must be a list of Povm.
            sl_qope.povms = ng_povms
        assert sl_qope.povms == povms

        # Act & Assert
        new_povms = [povms[1], povms[0]]
        sl_qope.povms = new_povms
        assert sl_qope.povms == new_povms

        # Act & Assert
        ng_gates = [gates[0], states[0], gates[2], gates[3]]
        with pytest.raises(TypeError):
            # TypeError: 'gates' must be a list of Gate.
            sl_qope.gates = ng_gates
        assert sl_qope.gates == gates

        # Act & Assert
        new_gates = [gates[0], gates[1]]
        sl_qope.gates = new_gates
        assert sl_qope.gates == new_gates

        # Act & Assert
        ng_mprocesses = [mprocesses[0], states[0]]
        with pytest.raises(TypeError):
            # TypeError: 'mprocesses' must be a list of MProcess.
            sl_qope.mprocesses = ng_mprocesses

        new_mprocesses = [mprocesses[1], mprocesses[0]]
        sl_qope.mprocesses = new_mprocesses
        assert sl_qope.mprocesses == new_mprocesses

    def test_qoperations(self):
        # Arrange
        states, povms, gates, mprocesses = self.arrange_qoperations()

        sl_qope = qope.SetQOperations(
            states=states, povms=povms, gates=gates, mprocesses=mprocesses
        )

        # Act & Assert
        assert len(sl_qope.qoperations("state")) == len(states)

        # Act & Assert
        assert len(sl_qope.qoperations("povm")) == len(povms)

        # Act & Assert
        assert len(sl_qope.qoperations("gate")) == len(gates)

        # Act & Assert
        assert len(sl_qope.qoperations("mprocess")) == len(mprocesses)

        # Act & Assert
        with pytest.raises(ValueError):
            sl_qope.qoperations("unsupported")

    def test_num(self):
        # Arrange
        states, povms, gates, mprocesses = self.arrange_qoperations()

        sl_qope = qope.SetQOperations(
            states=states, povms=povms, gates=gates, mprocesses=mprocesses
        )

        # Act & Assert
        expected = len(states)
        assert sl_qope.num_states() == expected

        # Act & Assert
        expected = len(povms)
        assert sl_qope.num_povms() == expected

        # Act & Assert
        expected = len(gates)
        assert sl_qope.num_gates() == expected

        # Act & Assert
        expected = len(mprocesses)
        assert sl_qope.num_mprocesses() == expected

    def test_num_qoperations(self):
        # Arrange
        states, povms, gates, mprocesses = self.arrange_qoperations()

        sl_qope = qope.SetQOperations(
            states=states, povms=povms, gates=gates, mprocesses=mprocesses
        )

        # Act & Assert
        assert sl_qope.num_qoperations("state") == len(states)

        # Act & Assert
        assert sl_qope.num_qoperations("povm") == len(povms)

        # Act & Assert
        assert sl_qope.num_qoperations("gate") == len(gates)

        # Act & Assert
        assert sl_qope.num_qoperations("mprocess") == len(mprocesses)

        # Act & Assert
        with pytest.raises(ValueError):
            sl_qope.num_qoperations("unsupported")

    def test_var_state(self):
        # Arrange
        _, povms, gates, mprocesses = self.arrange_qoperations()

        e_sys = ElementalSystem(0, matrix_basis.get_normalized_pauli_basis())
        c_sys = CompositeSystem([e_sys])

        vec_1 = np.array([1 / np.sqrt(2), 1, 2, 3], dtype=np.float64)
        vec_2 = np.array([1, 2, 3, 4], dtype=np.float64)

        state_1 = State(c_sys=c_sys, vec=vec_1, is_physicality_required=False)
        state_2 = State(c_sys=c_sys, vec=vec_2, is_physicality_required=False)

        states = [state_1, state_2]

        # Case 1:
        sl_qope = qope.SetQOperations(
            states=states,
            povms=povms,
            gates=gates,
        )
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
        _, povms, gates, mprocesses = self.arrange_qoperations()

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
        actual = sl_qope.var_states()

        # Assert
        expected = np.array([1, 2, 3, 1, 2, 3, 4], dtype=np.float64)
        assert np.all(actual == expected)

    def test_var_povms(self):
        # Arrange
        states, _, gates, mprocesses = self.arrange_qoperations()

        e_sys = ElementalSystem(0, matrix_basis.get_normalized_pauli_basis())
        c_sys = CompositeSystem([e_sys])

        # POVM 1
        vecs = [
            np.array([2, 3, 5, 7], dtype=np.float64),
            np.array([11, 13, 17, 19], dtype=np.float64),
        ]

        povm_1 = Povm(c_sys, vecs, is_physicality_required=False)
        povm_2 = Povm(
            c_sys, vecs, is_physicality_required=False, on_para_eq_constraint=True
        )
        povm_3 = Povm(
            c_sys, vecs, is_physicality_required=False, on_para_eq_constraint=False
        )
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
        states, povms, _, mprocesses = self.arrange_qoperations()

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

    def test_var_mprocesses(self):
        c_sys = generate_composite_system(mode="qubit", num=1, ids_esys=[1])
        # MProcess
        hss = [
            np.array(
                [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]],
                dtype=np.float64,
            ),
            np.array(
                [
                    [17, 18, 19, 20],
                    [21, 22, 23, 24],
                    [25, 26, 27, 28],
                    [29, 30, 31, 32],
                ],
                dtype=np.float64,
            ),
        ]
        mprocess_1 = MProcess(
            hss=hss,
            c_sys=c_sys,
            is_physicality_required=False,
            on_para_eq_constraint=True,
        )
        mprocess_2 = MProcess(
            hss=hss,
            c_sys=c_sys,
            is_physicality_required=False,
            on_para_eq_constraint=False,
        )
        mprocesses = [mprocess_1, mprocess_2]

        sl_qope = qope.SetQOperations(
            states=[], povms=[], gates=[], mprocesses=mprocesses
        )

        # Act
        actual = sl_qope.var_mprocesses()

        # Assert
        expected = np.array(
            list(range(1, 17)) + list(range(21, 33)) + list(range(1, 33))
        )
        # assert np.all(actual == expected)
        npt.assert_almost_equal(actual, expected, decimal=15)

        # Act
        actual = sl_qope.var_mprocess(0)

        # Assert
        expected = np.array(list(range(1, 17)) + list(range(21, 33)))
        npt.assert_almost_equal(actual, expected, decimal=15)

        # Act
        actual = sl_qope.var_mprocess(1)

        # Assert
        expected = np.array(list(range(1, 33)))
        npt.assert_almost_equal(actual, expected, decimal=15)

    def test_var_empty(self):
        # Arrange
        states, povms, gates, mprocesses = self.arrange_qoperations()

        # Empty QOperations
        # Arrange
        sl_qope = qope.SetQOperations(
            states=[], povms=povms, gates=gates, mprocesses=mprocesses
        )
        # Act
        actual = sl_qope.var_states()
        # Assert
        expected = np.array([], dtype=np.float64)
        assert np.all(actual == expected)
        # Act
        actual = sl_qope.size_var_states()
        # Assert
        assert actual == 0

        # Arrange
        sl_qope = qope.SetQOperations(
            states=states, povms=[], gates=gates, mprocesses=mprocesses
        )
        # Act
        actual = sl_qope.var_povms()
        # Assert
        expected = np.array([], dtype=np.float64)
        assert np.all(actual == expected)
        # Act
        actual = sl_qope.size_var_povms()
        # Assert
        assert actual == 0

        # Arrange
        sl_qope = qope.SetQOperations(
            states=states, povms=povms, gates=[], mprocesses=mprocesses
        )
        # Act
        actual = sl_qope.var_gates()
        # Assert
        expected = np.array([], dtype=np.float64)
        assert np.all(actual == expected)
        # Act
        actual = sl_qope.size_var_gates()
        # Assert
        assert actual == 0

        # Arrange
        sl_qope = qope.SetQOperations(
            states=states, povms=povms, gates=gates, mprocesses=[]
        )
        # Act
        actual = sl_qope.var_mprocesses()
        # Assert
        expected = np.array([], dtype=np.float64)
        assert np.all(actual == expected)
        # Act
        actual = sl_qope.size_var_mprocesses()
        # Assert
        assert actual == 0

        # Total
        # Arrange
        sl_qope = qope.SetQOperations(states=[], povms=[], gates=[], mprocesses=[])
        # Act
        actual = sl_qope.var_total()
        # Assert
        expected = np.array([], dtype=np.float64)
        assert np.all(actual == expected)
        # Act
        actual = sl_qope.size_var_total()
        # Assert
        assert actual == 0

    def test_var_total(self):
        e_sys = ElementalSystem(0, matrix_basis.get_normalized_pauli_basis())
        c_sys = CompositeSystem([e_sys])

        # State
        vec_1 = np.array([1 / np.sqrt(2), 1, 2, 3], dtype=np.float64)
        vec_2 = np.array([1, 2, 3, 4], dtype=np.float64)

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
        povm_1 = Povm(
            c_sys, vecs, is_physicality_required=False, on_para_eq_constraint=True
        )
        povm_2 = Povm(
            c_sys, vecs, is_physicality_required=False, on_para_eq_constraint=False
        )
        povms = [povm_1, povm_2]
        expected_povms_var = [2, 3, 5, 7] + [2, 3, 5, 7, 11, 13, 17, 19]

        # MProcess
        hss = [
            np.array(
                [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]],
                dtype=np.float64,
            ),
            np.array(
                [
                    [17, 18, 19, 20],
                    [21, 22, 23, 24],
                    [25, 26, 27, 28],
                    [29, 30, 31, 32],
                ],
                dtype=np.float64,
            ),
        ]

        mprocess_1 = MProcess(
            hss=hss,
            c_sys=c_sys,
            is_physicality_required=False,
            on_para_eq_constraint=True,
        )
        mprocess_2 = MProcess(
            hss=hss,
            c_sys=c_sys,
            is_physicality_required=False,
            on_para_eq_constraint=False,
        )
        mprocesses = [mprocess_1, mprocess_2]
        expected_mproesses_var = (
            list(range(1, 17)) + list(range(21, 33)) + list(range(1, 33))
        )

        sl_qope = qope.SetQOperations(
            states=states, povms=povms, gates=gates, mprocesses=mprocesses
        )

        # Act
        actual = sl_qope.var_total()

        # Assert
        expected = np.array(
            expected_states_var
            + expected_gates_var
            + expected_povms_var
            + expected_mproesses_var,
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
        povm_1 = Povm(
            c_sys, vecs, is_physicality_required=False, on_para_eq_constraint=True
        )
        povm_2 = Povm(
            c_sys, vecs, is_physicality_required=False, on_para_eq_constraint=False
        )
        povms = [povm_1, povm_2]

        # MProcess
        hss = [
            np.array(
                [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]],
                dtype=np.float64,
            ),
            np.array(
                [
                    [17, 18, 19, 20],
                    [21, 22, 23, 24],
                    [25, 26, 27, 28],
                    [29, 30, 31, 32],
                ],
                dtype=np.float64,
            ),
        ]
        mprocess_1 = MProcess(
            hss=hss,
            c_sys=c_sys,
            is_physicality_required=False,
            on_para_eq_constraint=True,
        )
        mprocess_2 = MProcess(
            hss=hss,
            c_sys=c_sys,
            is_physicality_required=False,
            on_para_eq_constraint=False,
        )
        mprocesses = [mprocess_1, mprocess_2]

        sl_qope = qope.SetQOperations(
            states=states, povms=povms, gates=gates, mprocesses=mprocesses
        )

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

        # Case 4-1: MProcess1
        actual = sl_qope.size_var_mprocess(0)

        # Assert
        expected = 28
        assert actual == expected

        # Case 4-2: MProcess2
        actual = sl_qope.size_var_mprocess(1)

        # Assert
        expected = 32
        assert actual == expected

        # Case 4-3: MProcess All
        # Act
        actual = sl_qope.size_var_mprocesses()

        # Assert
        expected = 28 + 32
        assert actual == expected

        # Case 5: Total
        actual = sl_qope.size_var_total()

        # Assert
        expected = 3 + 4 + 12 + 16 + 4 + 8 + 28 + 32
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
        povm_1q = get_x_povm(c_sys_1q)
        povm_2q = get_xx_povm(c_sys_2q)

        povms = [povm_2q, povm_2q, povm_1q]

        # MPRocess
        mprocess_1q = generate_qoperation_object(
            mode="mprocess", object_name="mprocess", name="x-type1", c_sys=c_sys_1q
        )
        mprocess_2q = generate_qoperation_object(
            mode="mprocess", object_name="mprocess", name="bell-type1", c_sys=c_sys_2q
        )
        mprocesses = [mprocess_1q, mprocess_2q]

        sl_qope = qope.SetQOperations(
            states=states, gates=gates, povms=povms, mprocesses=mprocesses
        )

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

        # Case 4: MProcess
        # Act
        actual = sl_qope.dim_mprocess(0)
        # Assert
        expected = 2
        assert actual == expected

        # Act
        actual = sl_qope.dim_mprocess(1)
        # Assert
        expected = 4
        assert actual == expected

    def _arrange_setqoperations(self):
        e_sys = ElementalSystem(0, matrix_basis.get_normalized_pauli_basis())
        c_sys_1q = CompositeSystem([e_sys])

        e_sys0 = ElementalSystem(0, matrix_basis.get_normalized_pauli_basis())
        e_sys1 = ElementalSystem(1, matrix_basis.get_normalized_pauli_basis())
        c_sys_2q = CompositeSystem([e_sys0, e_sys1])

        # State
        vec_1 = np.array([1 / np.sqrt(2), 101, 102, 103], dtype=np.float64)
        vec_2 = np.array([105, 106, 107, 108], dtype=np.float64)
        vec_3 = np.array(range(109, 125), dtype=np.float64)
        state_1 = State(
            c_sys=c_sys_1q,
            vec=vec_1,
            is_physicality_required=False,
            on_para_eq_constraint=True,
        )
        state_2 = State(
            c_sys=c_sys_1q,
            vec=vec_2,
            is_physicality_required=False,
            on_para_eq_constraint=False,
        )
        state_3 = State(
            c_sys=c_sys_2q,
            vec=vec_3,
            is_physicality_required=False,
            on_para_eq_constraint=False,
        )
        states = [state_1, state_2, state_3]

        # Gate
        hs_1 = np.array(
            [
                [201, 202, 203, 204],
                [205, 206, 207, 208],
                [209, 210, 211, 212],
                [213, 214, 215, 216],
            ],
            dtype=np.float64,
        )
        hs_2 = np.array(
            [
                [217, 218, 219, 220],
                [221, 222, 223, 224],
                [225, 226, 227, 228],
                [229, 230, 231, 232],
            ],
            dtype=np.float64,
        )
        hs_3 = np.array(
            [
                [233, 234, 235, 236],
                [237, 238, 239, 240],
                [241, 242, 243, 244],
                [245, 246, 247, 248],
            ],
            dtype=np.float64,
        )
        gate_1 = Gate(
            c_sys=c_sys_1q,
            hs=hs_1,
            on_para_eq_constraint=True,
            is_physicality_required=False,
        )
        gate_2 = Gate(
            c_sys=c_sys_1q,
            hs=hs_2,
            on_para_eq_constraint=False,
            is_physicality_required=False,
        )
        gate_3 = Gate(
            c_sys=c_sys_1q,
            hs=hs_3,
            on_para_eq_constraint=True,
            is_physicality_required=False,
        )
        gates = [gate_1, gate_2, gate_3]

        # Povm
        vecs_1 = [
            np.array([301, 302, 303, 304], dtype=np.float64),
            np.array([305, 306, 307, 305], dtype=np.float64),
        ]
        vecs_2 = [
            np.array([306, 307, 308, 309], dtype=np.float64),
            np.array([310, 311, 312, 313], dtype=np.float64),
        ]
        vecs_3 = [
            np.array([314, 315, 316, 317], dtype=np.float64),
            np.array([318, 319, 320, 321], dtype=np.float64),
        ]
        povm_1 = Povm(
            c_sys_1q, vecs_1, is_physicality_required=False, on_para_eq_constraint=True
        )
        povm_2 = Povm(
            c_sys_1q, vecs_2, is_physicality_required=False, on_para_eq_constraint=False
        )
        povm_3 = Povm(
            c_sys_1q, vecs_3, is_physicality_required=False, on_para_eq_constraint=True
        )
        povms = [povm_1, povm_2, povm_3]

        # MProcess
        hss = [
            np.array(
                [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]],
                dtype=np.float64,
            ),
            np.array(
                [
                    [17, 18, 19, 20],
                    [21, 22, 23, 24],
                    [25, 26, 27, 28],
                    [29, 30, 31, 32],
                ],
                dtype=np.float64,
            ),
        ]

        mprocess_1 = MProcess(
            hss=hss,
            c_sys=c_sys_1q,
            is_physicality_required=False,
            on_para_eq_constraint=True,
        )
        mprocess_2 = MProcess(
            hss=hss,
            c_sys=c_sys_1q,
            is_physicality_required=False,
            on_para_eq_constraint=False,
        )
        mprocesses = [mprocess_1, mprocess_2]

        set_qoperations = qope.SetQOperations(
            states=states, povms=povms, gates=gates, mprocesses=mprocesses
        )
        return set_qoperations

    def test_size_var_total(self):
        # Arrange
        set_qoperations = self._arrange_setqoperations()

        # Act
        actual = set_qoperations.size_var_total()

        # Assert
        expected = 139
        assert actual == expected

    def test_get_operation_mode_to_total_index_map(self):
        # Arrange
        set_qoperations = self._arrange_setqoperations()
        # Act
        actual = set_qoperations._get_operation_mode_to_total_index_map()
        # Assert
        expected = {"state": 0, "gate": 23, "povm": 63, "mprocess": 79}
        assert actual == expected

    def test_index_var_total_from_local_info(self):
        set_qoperations = self._arrange_setqoperations()
        var_total = set_qoperations.var_total()

        # Act & Assert
        # State[0]
        actual = set_qoperations.index_var_total_from_local_info("state", 0, 0)
        assert actual == 0
        assert var_total[actual] == set_qoperations.states[0].to_var()[0]

        actual = set_qoperations.index_var_total_from_local_info("state", 0, 2)
        assert actual == 2
        assert var_total[actual] == set_qoperations.states[0].to_var()[2]

        # State[1]
        actual = set_qoperations.index_var_total_from_local_info("state", 1, 0)
        assert actual == 3
        assert var_total[actual] == set_qoperations.states[1].to_var()[0]

        actual = set_qoperations.index_var_total_from_local_info("state", 1, 3)
        assert actual == 6
        assert var_total[actual] == set_qoperations.states[1].to_var()[3]

        # State[2]
        actual = set_qoperations.index_var_total_from_local_info("state", 2, 15)
        assert actual == 22
        assert var_total[actual] == set_qoperations.states[2].to_var()[15]

        # Gates[0]
        actual = set_qoperations.index_var_total_from_local_info("gate", 0, 0)
        assert actual == 23
        assert var_total[actual] == set_qoperations.gates[0].to_var()[0]

        actual = set_qoperations.index_var_total_from_local_info("gate", 0, 11)
        assert actual == 34
        assert var_total[actual] == set_qoperations.gates[0].to_var()[11]

        # Gates[1]
        actual = set_qoperations.index_var_total_from_local_info("gate", 1, 15)
        assert actual == 50
        assert var_total[actual] == set_qoperations.gates[1].to_var()[15]

        # Gates[2]
        actual = set_qoperations.index_var_total_from_local_info("gate", 2, 5)
        assert actual == 56
        assert var_total[actual] == set_qoperations.gates[2].to_var()[5]

        actual = set_qoperations.index_var_total_from_local_info("gate", 2, 11)
        assert actual == 62
        assert var_total[actual] == set_qoperations.gates[2].to_var()[11]

        # Povm[0]
        actual = set_qoperations.index_var_total_from_local_info("povm", 0, 0)
        assert actual == 63
        assert var_total[actual] == set_qoperations.povms[0].to_var()[0]

        # Povm[1]
        actual = set_qoperations.index_var_total_from_local_info("povm", 1, 1)
        assert actual == 68
        assert var_total[actual] == set_qoperations.povms[1].to_var()[1]

        # Povm[2]

        actual = set_qoperations.index_var_total_from_local_info("povm", 2, 0)
        assert actual == 75
        assert var_total[actual] == set_qoperations.povms[2].to_var()[0]

        actual = set_qoperations.index_var_total_from_local_info("povm", 2, 3)
        assert actual == 78
        assert var_total[actual] == set_qoperations.povms[2].to_var()[3]

        # MProcess
        actual = set_qoperations.index_var_total_from_local_info("mprocess", 0, 0)
        assert actual == 79
        assert var_total[actual] == set_qoperations.mprocesses[0].to_var()[0]

        actual = set_qoperations.index_var_total_from_local_info("mprocess", 0, 1)
        assert actual == 80
        assert var_total[actual] == set_qoperations.mprocesses[0].to_var()[1]

        actual = set_qoperations.index_var_total_from_local_info("mprocess", 0, 27)
        assert actual == 106
        assert var_total[actual] == set_qoperations.mprocesses[0].to_var()[-1]

        actual = set_qoperations.index_var_total_from_local_info("mprocess", 1, 0)
        assert actual == 107
        assert var_total[actual] == set_qoperations.mprocesses[1].to_var()[0]

    def test_local_info_from_index_var_total(self):
        set_qoperations = self._arrange_setqoperations()

        actual = set_qoperations.local_info_from_index_var_total(0)
        expected = dict(
            mode="state",
            index_operations=0,
            index_var_local=0,
        )
        assert actual == expected

        actual = set_qoperations.local_info_from_index_var_total(2)
        expected = dict(
            mode="state",
            index_operations=0,
            index_var_local=2,
        )
        assert actual == expected

        # states[1]
        actual = set_qoperations.local_info_from_index_var_total(3)
        expected = dict(
            mode="state",
            index_operations=1,
            index_var_local=0,
        )
        assert actual == expected

        actual = set_qoperations.local_info_from_index_var_total(6)
        expected = dict(
            mode="state",
            index_operations=1,
            index_var_local=3,
        )
        assert actual == expected

        # states[2]
        actual = set_qoperations.local_info_from_index_var_total(7)
        expected = dict(
            mode="state",
            index_operations=2,
            index_var_local=0,
        )
        assert actual == expected

        actual = set_qoperations.local_info_from_index_var_total(22)
        expected = dict(
            mode="state",
            index_operations=2,
            index_var_local=15,
        )
        assert actual == expected

        # gates[0]
        actual = set_qoperations.local_info_from_index_var_total(23)
        expected = dict(
            mode="gate",
            index_operations=0,
            index_var_local=0,
        )
        assert actual == expected

        actual = set_qoperations.local_info_from_index_var_total(34)
        expected = dict(
            mode="gate",
            index_operations=0,
            index_var_local=11,
        )
        assert actual == expected

        # gates[1]
        actual = set_qoperations.local_info_from_index_var_total(35)
        expected = dict(
            mode="gate",
            index_operations=1,
            index_var_local=0,
        )
        assert actual == expected

        actual = set_qoperations.local_info_from_index_var_total(50)
        expected = dict(
            mode="gate",
            index_operations=1,
            index_var_local=15,
        )
        assert actual == expected

        # gates[2]
        actual = set_qoperations.local_info_from_index_var_total(51)
        expected = dict(
            mode="gate",
            index_operations=2,
            index_var_local=0,
        )
        assert actual == expected

        actual = set_qoperations.local_info_from_index_var_total(62)
        expected = dict(
            mode="gate",
            index_operations=2,
            index_var_local=11,
        )
        assert actual == expected

        # povms[0]
        actual = set_qoperations.local_info_from_index_var_total(63)
        expected = dict(
            mode="povm",
            index_operations=0,
            index_var_local=0,
        )
        assert actual == expected

        actual = set_qoperations.local_info_from_index_var_total(66)
        expected = dict(
            mode="povm",
            index_operations=0,
            index_var_local=3,
        )
        assert actual == expected

        # povms[1]
        actual = set_qoperations.local_info_from_index_var_total(67)
        expected = dict(
            mode="povm",
            index_operations=1,
            index_var_local=0,
        )
        assert actual == expected

        actual = set_qoperations.local_info_from_index_var_total(74)
        expected = dict(
            mode="povm",
            index_operations=1,
            index_var_local=7,
        )
        assert actual == expected

        # povms[2]
        actual = set_qoperations.local_info_from_index_var_total(75)
        expected = dict(
            mode="povm",
            index_operations=2,
            index_var_local=0,
        )
        assert actual == expected

        actual = set_qoperations.local_info_from_index_var_total(78)
        expected = dict(
            mode="povm",
            index_operations=2,
            index_var_local=3,
        )
        assert actual == expected

        # MProcess
        actual = set_qoperations.local_info_from_index_var_total(79)
        expected = dict(
            mode="mprocess",
            index_operations=0,
            index_var_local=0,
        )
        assert actual == expected

        actual = set_qoperations.local_info_from_index_var_total(106)
        expected = dict(
            mode="mprocess",
            index_operations=0,
            index_var_local=27,
        )
        assert actual == expected

        actual = set_qoperations.local_info_from_index_var_total(107)
        expected = dict(
            mode="mprocess",
            index_operations=1,
            index_var_local=0,
        )
        assert actual == expected

        actual = set_qoperations.local_info_from_index_var_total(108)
        expected = dict(
            mode="mprocess",
            index_operations=1,
            index_var_local=1,
        )
        assert actual == expected

    def test_set_qoperations_from_var_total_exception(self):
        # Arrange
        set_qoperations = self._arrange_setqoperations()
        ok_var_total_size = len(set_qoperations.var_total())
        ng_var_total = np.array(range(1000, 1000 + ok_var_total_size + 1))

        # Act & Assert
        with pytest.raises(ValueError):
            set_qoperations.set_qoperations_from_var_total(ng_var_total)

        # Arrange
        ng_var_total = np.array(range(1000, 1000 + ok_var_total_size - 1))

        # Act & Assert
        with pytest.raises(ValueError):
            set_qoperations.set_qoperations_from_var_total(ng_var_total)

    def test_set_qoperations_from_var_total(self):
        # Arrange
        set_qoperations = self._arrange_setqoperations()
        source_states = list(range(1000, 1023))
        source_gates = list(range(2000, 2040))
        source_povms = list(range(3000, 3016))
        source_mprocesses = list(range(4000, 4060))
        source_var_total = np.array(
            source_states + source_gates + source_povms + source_mprocesses, np.float64
        )

        # Act
        actual = set_qoperations.set_qoperations_from_var_total(source_var_total)

        # Assert
        # State
        assert len(actual.states) == len(set_qoperations.states)
        expected_vecs = [
            np.array([1000, 1001, 1002]),
            np.array([1003, 1004, 1005, 1006]),
            np.array(range(1007, 1023)),
        ]
        for i, item in enumerate(zip(actual.states, set_qoperations.states)):
            actual_item, compared_item = item
            assert actual_item._composite_system is compared_item._composite_system
            assert (
                actual_item.on_para_eq_constraint == compared_item.on_para_eq_constraint
            )
            assert (
                actual_item.is_physicality_required
                == compared_item.is_physicality_required
            )
            assert (
                actual_item.is_estimation_object == compared_item.is_estimation_object
            )
            assert (
                actual_item.on_algo_eq_constraint == compared_item.on_algo_eq_constraint
            )
            assert (
                actual_item.on_algo_ineq_constraint
                == compared_item.on_algo_ineq_constraint
            )
            assert actual_item.eps_proj_physical == compared_item.eps_proj_physical
            npt.assert_almost_equal(actual_item.to_var(), expected_vecs[i])

        # Gate
        assert len(actual.gates) == len(set_qoperations.gates)
        expected_vecs = [
            np.array(range(2000, 2012)),
            np.array(range(2012, 2028)),
            np.array(range(2028, 2040)),
        ]
        for i, item in enumerate(zip(actual.gates, set_qoperations.gates)):
            actual_item, compared_item = item
            assert actual_item._composite_system is compared_item._composite_system
            assert (
                actual_item.on_para_eq_constraint == compared_item.on_para_eq_constraint
            )
            assert (
                actual_item.is_physicality_required
                == compared_item.is_physicality_required
            )
            assert (
                actual_item.is_estimation_object == compared_item.is_estimation_object
            )
            assert (
                actual_item.on_algo_eq_constraint == compared_item.on_algo_eq_constraint
            )
            assert (
                actual_item.on_algo_ineq_constraint
                == compared_item.on_algo_ineq_constraint
            )
            assert actual_item.eps_proj_physical == compared_item.eps_proj_physical
            npt.assert_almost_equal(actual_item.to_var(), expected_vecs[i])

        # POVM
        assert len(actual.gates) == len(set_qoperations.gates)
        expected_vecs = [
            np.array(range(3000, 3004)),
            np.array(range(3004, 3012)),
            np.array(range(3012, 3016)),
        ]
        for i, item in enumerate(zip(actual.povms, set_qoperations.povms)):
            actual_item, compared_item = item
            assert actual_item._composite_system is compared_item._composite_system
            assert (
                actual_item.on_para_eq_constraint == compared_item.on_para_eq_constraint
            )
            assert (
                actual_item.is_physicality_required
                == compared_item.is_physicality_required
            )
            assert (
                actual_item.is_estimation_object == compared_item.is_estimation_object
            )
            assert (
                actual_item.on_algo_eq_constraint == compared_item.on_algo_eq_constraint
            )
            assert (
                actual_item.on_algo_ineq_constraint
                == compared_item.on_algo_ineq_constraint
            )
            assert actual_item.eps_proj_physical == compared_item.eps_proj_physical
            npt.assert_almost_equal(actual_item.to_var(), expected_vecs[i])

        # MProcess
        assert len(actual.mprocesses) == len(set_qoperations.mprocesses)

        expected_hss_list = [
            [
                np.array(
                    [
                        [4000.0, 4001.0, 4002.0, 4003.0],
                        [4004.0, 4005.0, 4006.0, 4007.0],
                        [4008.0, 4009.0, 4010.0, 4011.0],
                        [4012.0, 4013.0, 4014.0, 4015.0],
                    ]
                ),
                np.array(
                    [
                        [-3999.0, -4001.0, -4002.0, -4003.0],
                        [4016.0, 4017.0, 4018.0, 4019.0],
                        [4020.0, 4021.0, 4022.0, 4023.0],
                        [4024.0, 4025.0, 4026.0, 4027.0],
                    ]
                ),
            ],
            [
                np.array(
                    [
                        [4028.0, 4029.0, 4030.0, 4031.0],
                        [4032.0, 4033.0, 4034.0, 4035.0],
                        [4036.0, 4037.0, 4038.0, 4039.0],
                        [4040.0, 4041.0, 4042.0, 4043.0],
                    ]
                ),
                np.array(
                    [
                        [4044.0, 4045.0, 4046.0, 4047.0],
                        [4048.0, 4049.0, 4050.0, 4051.0],
                        [4052.0, 4053.0, 4054.0, 4055.0],
                        [4056.0, 4057.0, 4058.0, 4059.0],
                    ]
                ),
            ],
        ]
        for i, item in enumerate(zip(actual.mprocesses, set_qoperations.mprocesses)):
            actual_item, compared_item = item
            assert actual_item._composite_system is compared_item._composite_system
            assert (
                actual_item.on_para_eq_constraint == compared_item.on_para_eq_constraint
            )
            assert (
                actual_item.is_physicality_required
                == compared_item.is_physicality_required
            )
            assert (
                actual_item.is_estimation_object == compared_item.is_estimation_object
            )
            assert (
                actual_item.on_algo_eq_constraint == compared_item.on_algo_eq_constraint
            )
            assert (
                actual_item.on_algo_ineq_constraint
                == compared_item.on_algo_ineq_constraint
            )
            assert actual_item.eps_proj_physical == compared_item.eps_proj_physical
            assert len(actual_item.hss) == len(expected_hss_list[i])

            for a, e in zip(actual_item.hss, expected_hss_list[i]):
                npt.assert_almost_equal(a, e)
