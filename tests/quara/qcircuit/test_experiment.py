from typing import Dict, List

import pytest

from quara.objects import matrix_basis
from quara.objects.composite_system import CompositeSystem
from quara.objects.elemental_system import ElementalSystem
from quara.objects.gate import Gate, get_h, get_i
from quara.objects.povm import (
    Povm,
    get_x_measurement,
    get_y_measurement,
    get_z_measurement,
)
from quara.objects.state import State, get_x0_1q, get_y0_1q
from quara.qcircuit.experiment import (
    Experiment,
    QuaraScheduleItemError,
    QuaraScheduleOrderError,
)


class TestExperiment:
    def array_states_povms_gates(self):
        # Array
        e_sys = ElementalSystem(0, matrix_basis.get_comp_basis())
        c_sys = CompositeSystem([e_sys])
        # State
        state_0 = get_x0_1q(c_sys)
        state_1 = get_y0_1q(c_sys)
        state_list = [state_0, state_1]

        # POVM
        povm_0 = get_x_measurement(c_sys)
        povm_1 = get_x_measurement(c_sys)
        povm_list = [povm_0, povm_1]
        # Gate
        gate_0 = get_i(c_sys)
        gate_1 = get_h(c_sys)
        gate_list = [gate_0, gate_1]
        return state_list, povm_list, gate_list

    def test_property(self):
        # Array
        states, povms, gates = self.array_states_povms_gates()
        schedule_list = [
            [("state", 0), ("gate", 0), ("povm", 0)],
            [("state", 1), ("gate", 1), ("povm", 1)],
        ]

        # Act & Assert
        _ = Experiment(states=states, povms=povms, gates=gates, schedules=schedule_list)

        # Arrange
        source_states = [states[0], None]
        source_povms = [None, povms[0]]
        source_gates = [gates[0], None]

        # Act & Assert
        exp = Experiment(
            states=source_states,
            povms=source_povms,
            gates=source_gates,
            schedules=schedule_list,
        )

        # Assert
        actual, expected = exp.states, source_states
        assert len(actual) == len(expected)
        for a, e in zip(actual, expected):
            assert a == e

        actual, expected = exp.povms, source_povms
        assert len(actual) == len(expected)
        for a, e in zip(actual, expected):
            assert a == e

        actual, expected = exp.gates, source_gates
        assert len(actual) == len(expected)
        for a, e in zip(actual, expected):
            assert a == e

        actual, expected = exp.schedules, schedule_list
        assert len(actual) == len(expected)
        for a, e in zip(actual, expected):
            assert a == e

    def test_init_unexpected_type(self):
        # Array
        ok_states, ok_povms, ok_gates = self.array_states_povms_gates()
        schedule_list = [
            [("state", 0), ("gate", 0), ("povm", 0)],
            [("state", 1), ("gate", 1), ("povm", 1)],
        ]

        # Act & Assert
        # Case1: Invalid states
        ng_states = [ok_states[0], ok_povms[0]]
        with pytest.raises(TypeError):
            # TypeError: 'states' must be a list of State.
            _ = Experiment(
                states=ng_states,
                povms=ok_povms,
                gates=ok_gates,
                schedules=schedule_list,
            )

        # Case2: Invalid povms
        ng_povms = [ok_states[0], ok_povms[0]]
        with pytest.raises(TypeError):
            # TypeError: 'povms' must be a list of Povm.
            _ = Experiment(
                states=ok_states,
                povms=ng_povms,
                gates=ok_gates,
                schedules=schedule_list,
            )

        # Case3: Invalid gates
        ng_gates = [ok_gates[0], ok_povms[0]]
        with pytest.raises(TypeError):
            # TypeError: 'povms' must be a list of Povm.
            _ = Experiment(
                states=ok_states,
                povms=ok_povms,
                gates=ng_gates,
                schedules=schedule_list,
            )

    def test_expeption_order_too_short_schedule(self):
        # Array
        ok_states, ok_povms, ok_gates = self.array_states_povms_gates()
        ng_schedule_list = [
            [("state", 0)],
            [("state", 1), ("gate", 1), ("povm", 1)],
        ]

        # Act & Assert
        with pytest.raises(QuaraScheduleOrderError):
            # There is a schedule with an invalid order.
            _ = Experiment(
                states=ok_states,
                povms=ok_povms,
                gates=ok_gates,
                schedules=ng_schedule_list,
            )

    def test_expeption_order_not_start_with_state(self):
        # Array
        ok_states, ok_povms, ok_gates = self.array_states_povms_gates()
        ng_schedule_list = [
            [("state", 0), ("gate", 0), ("povm", 0)],  # OK
            [("povm", 1), ("gate", 1), ("povm", 1)],  # NG
            [("state", 1), ("gate", 1), ("povm", 1)],  # OK
        ]

        # Act & Assert
        with pytest.raises(QuaraScheduleOrderError):
            # There is a schedule with an invalid order.
            # Detail: The first element of the schedule must be a 'state'.
            _ = Experiment(
                states=ok_states,
                povms=ok_povms,
                gates=ok_gates,
                schedules=ng_schedule_list,
            )

    def test_expeption_order_not_end_with_povm_mprocess(self):
        # Array
        ok_states, ok_povms, ok_gates = self.array_states_povms_gates()
        ng_schedule_list = [
            [("state", 0), ("gate", 0), ("povm", 0)],  # OK
            [("state"), ("gate", 1), ("povm", 1)],  # NG
            [("state", 0), ("gate", 0), ("povm", 0)],  # OK
        ]

        # Act & Assert
        with pytest.raises(QuaraScheduleItemError):
            # There is a schedule with an invalid order.
            # Detail: The last element of the schedule must be either 'povm' or 'mprocess'.
            _ = Experiment(
                states=ok_states,
                povms=ok_povms,
                gates=ok_gates,
                schedules=ng_schedule_list,
            )

        # TODO: mprocessを実装後、mprocessで終わるスケジュールを含めたテストを追加する

    def test_expeption_order_too_many_state(self):
        # Array
        ok_states, ok_povms, ok_gates = self.array_states_povms_gates()
        ng_schedule_list = [
            [("state", 0), ("gate", 0), ("gate", 1), ("povm", 0)],  # OK
            [("state", 0), ("state", 1), ("gate", 1), ("povm", 1)],  # NG
            [("state", 1), ("gate", 1), ("povm", 1)],  # OK
        ]

        # Act & Assert
        with pytest.raises(QuaraScheduleOrderError):
            # There is a schedule with an invalid order.
            # Detail: There are too many States; one schedule can only contain one State.
            _ = Experiment(
                states=ok_states,
                povms=ok_povms,
                gates=ok_gates,
                schedules=ng_schedule_list,
            )

    def test_expeption_order_too_many_povm(self):
        # Array
        ok_states, ok_povms, ok_gates = self.array_states_povms_gates()
        ng_schedule_list = [
            [("state", 0), ("gate", 0), ("povm", 0)],  # OK
            [("state", 0), ("gate", 1), ("povm", 0), ("povm", 1)],  # NG
            [("state", 1), ("gate", 1), ("povm", 1)],  # OK
        ]

        # Act & Assert
        with pytest.raises(QuaraScheduleOrderError):
            # There is a schedule with an invalid order.
            # Detail: There are too many POVMs; one schedule can only contain one POVM.
            _ = Experiment(
                states=ok_states,
                povms=ok_povms,
                gates=ok_gates,
                schedules=ng_schedule_list,
            )

    def test_expeption_item_no_mprocess(self):
        # Array
        ok_states, ok_povms, ok_gates = self.array_states_povms_gates()
        ng_schedule_list = [
            [("state", 0), ("gate", 0), ("povm", 0)],  # OK
            [("state", 0), ("gate", 0), ("mprocess", 0)],  # NG
            [("state", 1), ("gate", 1), ("povm", 1)],  # OK
        ]

        # Act & Assert
        with pytest.raises(QuaraScheduleItemError):
            # There is a schedule with an invalid order.
            # Detail: The first element of the schedule must be a 'state'.
            _ = Experiment(
                states=ok_states,
                povms=ok_povms,
                gates=ok_gates,
                schedules=ng_schedule_list,
            )

    def test_expeption_item_not_tuple(self):
        # Array
        ok_states, ok_povms, ok_gates = self.array_states_povms_gates()
        ng_schedule_list = [
            [("state", 0), ("gate", 0), ("povm", 0)],  # OK
            [1, ("gate", 1), ("povm", 1)],  # NG
            [("state", 0), ("gate", 0), ("povm", 0)],  # OK
        ]

        # Act & Assert
        with pytest.raises(QuaraScheduleItemError):
            # The item in the schedules[1] is invalid.
            # Detail: A schedule item must be a tuple of str and int.
            _ = Experiment(
                states=ok_states,
                povms=ok_povms,
                gates=ok_gates,
                schedules=ng_schedule_list,
            )

    def test_expeption_item_too_short(self):
        # Array
        ok_states, ok_povms, ok_gates = self.array_states_povms_gates()
        ng_schedule_list = [
            [("state", 0), ("gate", 0), ("povm", 0)],  # OK
            [("state"), ("gate", 1), ("povm", 1)],  # NG
            [("state", 0), ("gate", 0), ("povm", 0)],  # OK
        ]

        # Act & Assert
        with pytest.raises(QuaraScheduleItemError):
            # The item in the schedules[1] is invalid.
            # Detail: A schedule item must be a tuple of str and int.
            _ = Experiment(
                states=ok_states,
                povms=ok_povms,
                gates=ok_gates,
                schedules=ng_schedule_list,
            )

    def test_expeption_item_too_long(self):
        # Array
        ok_states, ok_povms, ok_gates = self.array_states_povms_gates()
        ng_schedule_list = [
            [("state", 0), ("gate", 0), ("povm", 0)],  # OK
            [("state", 1, 1), ("gate", 1), ("povm", 1)],  # NG
            [("state", 0), ("gate", 0), ("povm", 0)],  # OK
        ]

        # Act & Assert
        with pytest.raises(QuaraScheduleItemError):
            # The item in the schedules[1] is invalid.
            # Detail: A schedule item must be a tuple of str and int.
            _ = Experiment(
                states=ok_states,
                povms=ok_povms,
                gates=ok_gates,
                schedules=ng_schedule_list,
            )

    def test_expeption_item_invalid_name_type(self):
        # Array
        ok_states, ok_povms, ok_gates = self.array_states_povms_gates()
        ng_schedule_list = [
            [("state", 0), ("gate", 0), ("povm", 0)],  # OK
            [(1, 1), ("gate", 1), ("povm", 1)],  # NG
            [("state", 0), ("gate", 0), ("povm", 0)],  # OK
        ]

        # Act & Assert
        with pytest.raises(QuaraScheduleItemError):
            # The item in the schedules[1] is invalid.
            # Detail: A schedule item must be a tuple of str and int.
            _ = Experiment(
                states=ok_states,
                povms=ok_povms,
                gates=ok_gates,
                schedules=ng_schedule_list,
            )

    def test_expeption_item_invalid_index_type(self):
        # Array
        ok_states, ok_povms, ok_gates = self.array_states_povms_gates()
        ng_schedule_list = [
            [("state", 0), ("gate", 0), ("povm", 0)],  # OK
            [("state", "1"), ("gate", 1), ("povm", 1)],  # NG
            [("state", 0), ("gate", 0), ("povm", 0)],  # OK
        ]

        # Act & Assert
        with pytest.raises(QuaraScheduleItemError):
            # The item in the schedules[1] is invalid.
            # Detail: A schedule item must be a tuple of str and int.
            _ = Experiment(
                states=ok_states,
                povms=ok_povms,
                gates=ok_gates,
                schedules=ng_schedule_list,
            )

    def test_expeption_item_unknown_name(self):
        # Array
        ok_states, ok_povms, ok_gates = self.array_states_povms_gates()
        ng_schedule_list = [
            [("state", 0), ("gate", 0), ("povm", 0)],  # OK
            [("state?", 1), ("gate", 1), ("povm", 1)],  # NG
            [("state", 0), ("gate", 0), ("povm", 0)],  # OK
        ]

        # Act & Assert
        with pytest.raises(QuaraScheduleItemError):
            # The item in the schedules[1] is invalid.
            # Detail: The item of schedule can be specified as either 'state', 'povm', 'gate', or 'mprocess'.
            _ = Experiment(
                states=ok_states,
                povms=ok_povms,
                gates=ok_gates,
                schedules=ng_schedule_list,
            )

    def test_expeption_item_out_of_range(self):
        # Array
        ok_states, ok_povms, ok_gates = self.array_states_povms_gates()
        ng_schedule_list = [
            [("state", 0), ("gate", 0), ("povm", 0)],  # OK
            [("state", 3), ("gate", 1), ("povm", 1)],  # NG
            [("state", 0), ("gate", 0), ("povm", 0)],  # OK
        ]

        # Act & Assert
        with pytest.raises(QuaraScheduleItemError):
            # The item in the schedules[1] is invalid.
            # Detail: The index out of range.'states' is 3 in length, but an index out of range was referenced in the schedule.
            _ = Experiment(
                states=ok_states,
                povms=ok_povms,
                gates=ok_gates,
                schedules=ng_schedule_list,
            )
