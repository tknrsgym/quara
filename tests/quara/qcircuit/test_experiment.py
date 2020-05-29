from typing import Dict, List

import numpy.testing as npt
import pytest

from quara.objects import matrix_basis
from quara.objects.composite_system import CompositeSystem
from quara.objects.elemental_system import ElementalSystem
from quara.objects.gate import Gate, get_h, get_i, get_x
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
    def test_calc_probdist(self):
        # Array
        e_sys1 = ElementalSystem(1, matrix_basis.get_normalized_pauli_basis())
        c_sys1 = CompositeSystem([e_sys1])

        state_list = [get_x0_1q(c_sys1), get_y0_1q(c_sys1)]
        gate_list = [get_i(c_sys1), get_x(c_sys1)]
        povm_list = [get_x_measurement(c_sys1), get_y_measurement(c_sys1)]
        schedule_list = [
            [("state", 0), ("gate", 0), ("povm", 0)],
            [("state", 0), ("gate", 0), ("povm", 1)],
        ]
        trial_nums = [1, 1]
        exp = Experiment(
            states=state_list,
            povms=povm_list,
            gates=gate_list,
            schedules=schedule_list,
            trial_nums=trial_nums,
        )

        # Act
        actual = exp.calc_probdist(index=0)

        # Assert
        expected = [1, 0]
        npt.assert_almost_equal(actual, expected, decimal=15)

        # Act
        actual = exp.calc_probdist(index=1)

        # Assert
        expected = [0.5, 0.5]
        npt.assert_almost_equal(actual, expected, decimal=15)

    def test_calc_probdists(self):
        # Array
        e_sys1 = ElementalSystem(1, matrix_basis.get_normalized_pauli_basis())
        c_sys1 = CompositeSystem([e_sys1])

        state_list = [get_x0_1q(c_sys1), get_y0_1q(c_sys1)]
        gate_list = [get_i(c_sys1), get_x(c_sys1)]
        povm_list = [get_x_measurement(c_sys1), get_y_measurement(c_sys1)]
        schedule_list = [
            [("state", 0), ("gate", 0), ("povm", 0)],
            [("state", 0), ("gate", 0), ("povm", 1)],
        ]
        trial_nums = [1, 1]
        exp = Experiment(
            states=state_list,
            povms=povm_list,
            gates=gate_list,
            schedules=schedule_list,
            trial_nums=trial_nums,
        )

        # Act
        actual = exp.calc_probdists()

        # Assert
        expected = [[1, 0], [0.5, 0.5]]
        assert len(actual) == len(expected)
        for a, e in zip(actual, expected):
            npt.assert_almost_equal(a, e, decimal=15)

    def test_calc_probdist_exist_none(self):
        # Array
        e_sys1 = ElementalSystem(1, matrix_basis.get_normalized_pauli_basis())
        c_sys1 = CompositeSystem([e_sys1])

        state_list = [None, get_y0_1q(c_sys1)]
        gate_list = [get_i(c_sys1), get_x(c_sys1)]
        povm_list = [get_x_measurement(c_sys1), get_y_measurement(c_sys1)]
        schedule_list = [
            [("state", 0), ("gate", 0), ("povm", 0)],
            [("state", 0), ("gate", 0), ("povm", 1)],
        ]
        trial_nums = [1, 1]
        exp = Experiment(
            states=state_list,
            povms=povm_list,
            gates=gate_list,
            schedules=schedule_list,
            trial_nums=trial_nums,
        )

        # Act
        with pytest.raises(ValueError):
            # ValueError: states[0] is None.
            _ = exp.calc_probdist(index=0)

    def test_generate_data(self):
        # Array
        e_sys1 = ElementalSystem(1, matrix_basis.get_normalized_pauli_basis())
        c_sys1 = CompositeSystem([e_sys1])

        state_list = [get_x0_1q(c_sys1), get_y0_1q(c_sys1)]
        gate_list = [get_i(c_sys1), get_x(c_sys1)]
        povm_list = [get_x_measurement(c_sys1), get_y_measurement(c_sys1)]
        schedule_list = [
            [("state", 0), ("gate", 0), ("povm", 0)],
            [("state", 0), ("gate", 0), ("povm", 1)],
        ]
        trial_nums = [1, 1]
        exp = Experiment(
            states=state_list,
            povms=povm_list,
            gates=gate_list,
            schedules=schedule_list,
            trial_nums=trial_nums,
        )

        # Act
        # Case 1:
        actual = exp.generate_data(index=0, data_num=10, seed=7)

        # Assert
        expected = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        assert actual == expected

        # Case 2:
        actual = exp.generate_data(index=1, data_num=10, seed=7)

        # Assert
        expected = [0, 1, 0, 1, 1, 1, 1, 0, 0, 0]
        assert actual == expected

        # Case 3:
        actual = exp.generate_dataset(data_num_list=[10, 10], seed=7)
        expected = [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 1, 0, 1, 1, 1, 1, 0, 0, 0]]
        assert actual == expected

    def test_generate_empidist(self):
        # Array
        e_sys1 = ElementalSystem(1, matrix_basis.get_normalized_pauli_basis())
        c_sys1 = CompositeSystem([e_sys1])

        state_list = [get_x0_1q(c_sys1), get_y0_1q(c_sys1)]
        gate_list = [get_i(c_sys1), get_x(c_sys1)]
        povm_list = [get_x_measurement(c_sys1), get_y_measurement(c_sys1)]
        schedule_list = [
            [("state", 0), ("gate", 0), ("povm", 0)],
            [("state", 0), ("gate", 0), ("povm", 1)],
        ]
        trial_nums = [1, 1]
        exp = Experiment(
            states=state_list,
            povms=povm_list,
            gates=gate_list,
            schedules=schedule_list,
            trial_nums=trial_nums,
        )

        # Act
        # Case 1:
        list_num_sum = [1]
        actual = exp.generate_empidist(index=0, list_num_sum=list_num_sum)

        # Assert
        # TODO

    def test_generate_empidist(self):
        # Array
        e_sys1 = ElementalSystem(1, matrix_basis.get_normalized_pauli_basis())
        c_sys1 = CompositeSystem([e_sys1])

        state_list = [get_x0_1q(c_sys1), get_y0_1q(c_sys1)]
        gate_list = [get_i(c_sys1), get_x(c_sys1)]
        povm_list = [get_x_measurement(c_sys1), get_y_measurement(c_sys1)]
        schedule_list = [
            [("state", 0), ("gate", 0), ("povm", 0)],
            [("state", 0), ("gate", 0), ("povm", 1)],
        ]
        trial_nums = [1, 1]
        exp = Experiment(
            states=state_list,
            povms=povm_list,
            gates=gate_list,
            schedules=schedule_list,
            trial_nums=trial_nums,
        )

        # Act
        # Case 1:
        list_num_sums = [[1], [1]]
        actual = exp.generate_empidists(list_num_sums=list_num_sums)

        # Assert
        # TODO

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

    def test_getter(self):
        # Array
        states, povms, gates = self.array_states_povms_gates()
        schedule_list = [
            [("state", 0), ("gate", 0), ("povm", 0)],
            [("state", 1), ("gate", 1), ("povm", 1)],
        ]
        trial_nums = [1] * len(schedule_list)

        # Act & Assert
        _ = Experiment(
            states=states,
            povms=povms,
            gates=gates,
            schedules=schedule_list,
            trial_nums=trial_nums,
        )

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
            trial_nums=trial_nums,
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

        actual, expected = exp.trial_nums, trial_nums
        assert len(actual) == len(expected)
        for a, e in zip(actual, expected):
            assert a == e

    def test_setter_validation(self):
        # Array
        states, povms, gates = self.array_states_povms_gates()
        schedule_list = [
            [("state", 0), ("gate", 0), ("povm", 0)],
            [("state", 1), ("gate", 1), ("povm", 1)],
        ]
        trial_nums = [1] * len(schedule_list)

        exp = Experiment(
            states=states,
            povms=povms,
            gates=gates,
            schedules=schedule_list,
            trial_nums=trial_nums,
        )

        # Act & Assert
        ng_new_states = [states[0], povms[0]]
        with pytest.raises(TypeError):
            # TypeError: 'states' must be a list of State.
            exp.states = ng_new_states

        # Act & Assert
        ng_new_states = [states[0]]
        with pytest.raises(QuaraScheduleItemError):
            # New State does not match schedules.
            exp.states = ng_new_states
        # Assert
        actual, expected = exp.states, states
        assert len(actual) == len(expected)
        for a, e in zip(actual, expected):
            assert a == e

        # Act & Assert
        ng_new_povms = [states[0], povms[0]]

        with pytest.raises(TypeError):
            # TypeError: 'povms' must be a list of Povm.
            exp.povms = ng_new_povms

        # Act & Assert
        ng_new_povms = [povms[0]]
        with pytest.raises(QuaraScheduleItemError):
            # New 'povms' does not match schedules.
            exp.povms = ng_new_povms
        # Assert
        actual, expected = exp.povms, povms
        assert len(actual) == len(expected)
        for a, e in zip(actual, expected):
            assert a == e

        # Act & Assert
        ng_new_gates = [states[0], gates[0]]
        with pytest.raises(TypeError):
            # TypeError: 'gates' must be a list of Gate.
            exp.gates = ng_new_gates

        # Act & Assert
        ng_new_gates = [gates[0]]
        with pytest.raises(QuaraScheduleItemError):
            # New 'gates' does not match schedules.
            exp.gates = ng_new_gates
        # Assert
        actual, expected = exp.gates, gates
        assert len(actual) == len(expected)
        for a, e in zip(actual, expected):
            assert a == e

        # Act & Assert
        ng_new_schedules = [
            [("povm"), ("gate", 1), ("povm", 1)],
            [("state", 1), ("gate", 1), ("povm", 1)],
        ]
        with pytest.raises(QuaraScheduleItemError):
            # A schedule item must be a tuple of str and int.
            exp.schedules = ng_new_schedules

        # Act & Assert
        ng_new_schedules = [
            [("povm", 1), ("gate", 1), ("povm", 1)],
            [("state", 1), ("gate", 1), ("povm", 1)],
        ]
        with pytest.raises(QuaraScheduleOrderError):
            # he first element of the schedule must be a 'state'.
            exp.schedules = ng_new_schedules

        # Act & Assert
        ng_new_schedules = [[("state", 1), ("gate", 1), ("povm", 1)]]
        with pytest.raises(ValueError):
            # ValueError: 'trial_nums' and 'schedules' must be equal in length.
            exp.schedules = ng_new_schedules

        # Act & Assert
        ng_new_trial_nums = [1]
        with pytest.raises(ValueError):
            # ValueError: 'trial_nums' and 'schedules' must be equal in length.
            exp.trial_nums = ng_new_trial_nums

        # Act & Assert
        ng_new_trial_nums = 1
        with pytest.raises(TypeError):
            # TypeError: 'trial_nums' must be a list with int elements.
            exp.trial_nums = ng_new_trial_nums

    def test_init_unexpected_type(self):
        # Array
        ok_states, ok_povms, ok_gates = self.array_states_povms_gates()
        schedule_list = [
            [("state", 0), ("gate", 0), ("povm", 0)],
            [("state", 1), ("gate", 1), ("povm", 1)],
        ]
        trial_nums = [1] * len(schedule_list)

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
                trial_nums=trial_nums,
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
                trial_nums=trial_nums,
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
                trial_nums=trial_nums,
            )

        # Case4: Invalid trial_nums
        ng_trial_nums = [1, 1, 1]
        with pytest.raises(ValueError):
            _ = Experiment(
                states=ok_states,
                povms=ok_povms,
                gates=ok_gates,
                schedules=schedule_list,
                trial_nums=ng_trial_nums,
            )

    def test_expeption_order_too_short_schedule(self):
        # Array
        ok_states, ok_povms, ok_gates = self.array_states_povms_gates()
        ng_schedule_list = [
            [("state", 0)],
            [("state", 1), ("gate", 1), ("povm", 1)],
        ]
        trial_nums = [1] * len(ng_schedule_list)

        # Act & Assert
        with pytest.raises(QuaraScheduleOrderError):
            # There is a schedule with an invalid order.
            _ = Experiment(
                states=ok_states,
                povms=ok_povms,
                gates=ok_gates,
                schedules=ng_schedule_list,
                trial_nums=trial_nums,
            )

    def test_expeption_order_not_start_with_state(self):
        # Array
        ok_states, ok_povms, ok_gates = self.array_states_povms_gates()
        ng_schedule_list = [
            [("state", 0), ("gate", 0), ("povm", 0)],  # OK
            [("povm", 1), ("gate", 1), ("povm", 1)],  # NG
            [("state", 1), ("gate", 1), ("povm", 1)],  # OK
        ]
        trial_nums = [1] * len(ng_schedule_list)

        # Act & Assert
        with pytest.raises(QuaraScheduleOrderError):
            # There is a schedule with an invalid order.
            # Detail: The first element of the schedule must be a 'state'.
            _ = Experiment(
                states=ok_states,
                povms=ok_povms,
                gates=ok_gates,
                schedules=ng_schedule_list,
                trial_nums=trial_nums,
            )

    def test_expeption_order_not_end_with_povm_mprocess(self):
        # Array
        ok_states, ok_povms, ok_gates = self.array_states_povms_gates()
        ng_schedule_list = [
            [("state", 0), ("gate", 0), ("povm", 0)],  # OK
            [("state"), ("gate", 1), ("povm", 1)],  # NG
            [("state", 0), ("gate", 0), ("povm", 0)],  # OK
        ]
        trial_nums = [1] * len(ng_schedule_list)

        # Act & Assert
        with pytest.raises(QuaraScheduleItemError):
            # There is a schedule with an invalid order.
            # Detail: The last element of the schedule must be either 'povm' or 'mprocess'.
            _ = Experiment(
                states=ok_states,
                povms=ok_povms,
                gates=ok_gates,
                schedules=ng_schedule_list,
                trial_nums=trial_nums,
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
        trial_nums = [1] * len(ng_schedule_list)

        # Act & Assert
        with pytest.raises(QuaraScheduleOrderError):
            # There is a schedule with an invalid order.
            # Detail: There are too many States; one schedule can only contain one State.
            _ = Experiment(
                states=ok_states,
                povms=ok_povms,
                gates=ok_gates,
                schedules=ng_schedule_list,
                trial_nums=trial_nums,
            )

    def test_expeption_order_too_many_povm(self):
        # Array
        ok_states, ok_povms, ok_gates = self.array_states_povms_gates()
        ng_schedule_list = [
            [("state", 0), ("gate", 0), ("povm", 0)],  # OK
            [("state", 0), ("gate", 1), ("povm", 0), ("povm", 1)],  # NG
            [("state", 1), ("gate", 1), ("povm", 1)],  # OK
        ]
        trial_nums = [1] * len(ng_schedule_list)

        # Act & Assert
        with pytest.raises(QuaraScheduleOrderError):
            # There is a schedule with an invalid order.
            # Detail: There are too many POVMs; one schedule can only contain one POVM.
            _ = Experiment(
                states=ok_states,
                povms=ok_povms,
                gates=ok_gates,
                schedules=ng_schedule_list,
                trial_nums=trial_nums,
            )

    def test_expeption_item_no_mprocess(self):
        # Array
        ok_states, ok_povms, ok_gates = self.array_states_povms_gates()
        ng_schedule_list = [
            [("state", 0), ("gate", 0), ("povm", 0)],  # OK
            [("state", 0), ("gate", 0), ("mprocess", 0)],  # NG
            [("state", 1), ("gate", 1), ("povm", 1)],  # OK
        ]
        trial_nums = [1] * len(ng_schedule_list)

        # Act & Assert
        with pytest.raises(QuaraScheduleItemError):
            # There is a schedule with an invalid order.
            # Detail: The first element of the schedule must be a 'state'.
            _ = Experiment(
                states=ok_states,
                povms=ok_povms,
                gates=ok_gates,
                schedules=ng_schedule_list,
                trial_nums=trial_nums,
            )

    def test_expeption_item_not_tuple(self):
        # Array
        ok_states, ok_povms, ok_gates = self.array_states_povms_gates()
        ng_schedule_list = [
            [("state", 0), ("gate", 0), ("povm", 0)],  # OK
            [1, ("gate", 1), ("povm", 1)],  # NG
            [("state", 0), ("gate", 0), ("povm", 0)],  # OK
        ]
        trial_nums = [1] * len(ng_schedule_list)

        # Act & Assert
        with pytest.raises(QuaraScheduleItemError):
            # The item in the schedules[1] is invalid.
            # Detail: A schedule item must be a tuple of str and int.
            _ = Experiment(
                states=ok_states,
                povms=ok_povms,
                gates=ok_gates,
                schedules=ng_schedule_list,
                trial_nums=trial_nums,
            )

    def test_expeption_item_too_short(self):
        # Array
        ok_states, ok_povms, ok_gates = self.array_states_povms_gates()
        ng_schedule_list = [
            [("state", 0), ("gate", 0), ("povm", 0)],  # OK
            [("state"), ("gate", 1), ("povm", 1)],  # NG
            [("state", 0), ("gate", 0), ("povm", 0)],  # OK
        ]
        trial_nums = [1] * len(ng_schedule_list)

        # Act & Assert
        with pytest.raises(QuaraScheduleItemError):
            # The item in the schedules[1] is invalid.
            # Detail: A schedule item must be a tuple of str and int.
            _ = Experiment(
                states=ok_states,
                povms=ok_povms,
                gates=ok_gates,
                schedules=ng_schedule_list,
                trial_nums=trial_nums,
            )

    def test_expeption_item_too_long(self):
        # Array
        ok_states, ok_povms, ok_gates = self.array_states_povms_gates()
        ng_schedule_list = [
            [("state", 0), ("gate", 0), ("povm", 0)],  # OK
            [("state", 1, 1), ("gate", 1), ("povm", 1)],  # NG
            [("state", 0), ("gate", 0), ("povm", 0)],  # OK
        ]
        trial_nums = [1] * len(ng_schedule_list)

        # Act & Assert
        with pytest.raises(QuaraScheduleItemError):
            # The item in the schedules[1] is invalid.
            # Detail: A schedule item must be a tuple of str and int.
            _ = Experiment(
                states=ok_states,
                povms=ok_povms,
                gates=ok_gates,
                schedules=ng_schedule_list,
                trial_nums=trial_nums,
            )

    def test_expeption_item_invalid_name_type(self):
        # Array
        ok_states, ok_povms, ok_gates = self.array_states_povms_gates()
        ng_schedule_list = [
            [("state", 0), ("gate", 0), ("povm", 0)],  # OK
            [(1, 1), ("gate", 1), ("povm", 1)],  # NG
            [("state", 0), ("gate", 0), ("povm", 0)],  # OK
        ]
        trial_nums = [1] * len(ng_schedule_list)

        # Act & Assert
        with pytest.raises(QuaraScheduleItemError):
            # The item in the schedules[1] is invalid.
            # Detail: A schedule item must be a tuple of str and int.
            _ = Experiment(
                states=ok_states,
                povms=ok_povms,
                gates=ok_gates,
                schedules=ng_schedule_list,
                trial_nums=trial_nums,
            )

    def test_expeption_item_invalid_index_type(self):
        # Array
        ok_states, ok_povms, ok_gates = self.array_states_povms_gates()
        ng_schedule_list = [
            [("state", 0), ("gate", 0), ("povm", 0)],  # OK
            [("state", "1"), ("gate", 1), ("povm", 1)],  # NG
            [("state", 0), ("gate", 0), ("povm", 0)],  # OK
        ]
        trial_nums = [1] * len(ng_schedule_list)

        # Act & Assert
        with pytest.raises(QuaraScheduleItemError):
            # The item in the schedules[1] is invalid.
            # Detail: A schedule item must be a tuple of str and int.
            _ = Experiment(
                states=ok_states,
                povms=ok_povms,
                gates=ok_gates,
                schedules=ng_schedule_list,
                trial_nums=trial_nums,
            )

    def test_expeption_item_unknown_name(self):
        # Array
        ok_states, ok_povms, ok_gates = self.array_states_povms_gates()
        ng_schedule_list = [
            [("state", 0), ("gate", 0), ("povm", 0)],  # OK
            [("state?", 1), ("gate", 1), ("povm", 1)],  # NG
            [("state", 0), ("gate", 0), ("povm", 0)],  # OK
        ]
        trial_nums = [1] * len(ng_schedule_list)

        # Act & Assert
        with pytest.raises(QuaraScheduleItemError):
            # The item in the schedules[1] is invalid.
            # Detail: The item of schedule can be specified as either 'state', 'povm', 'gate', or 'mprocess'.
            _ = Experiment(
                states=ok_states,
                povms=ok_povms,
                gates=ok_gates,
                schedules=ng_schedule_list,
                trial_nums=trial_nums,
            )

    def test_expeption_item_out_of_range(self):
        # Array
        ok_states, ok_povms, ok_gates = self.array_states_povms_gates()
        ng_schedule_list = [
            [("state", 0), ("gate", 0), ("povm", 0)],  # OK
            [("state", 3), ("gate", 1), ("povm", 1)],  # NG
            [("state", 0), ("gate", 0), ("povm", 0)],  # OK
        ]
        trial_nums = [1] * len(ng_schedule_list)

        # Act & Assert
        with pytest.raises(QuaraScheduleItemError):
            # The item in the schedules[1] is invalid.
            # Detail: The index out of range.'states' is 3 in length, but an index out of range was referenced in the schedule.
            _ = Experiment(
                states=ok_states,
                povms=ok_povms,
                gates=ok_gates,
                schedules=ng_schedule_list,
                trial_nums=trial_nums,
            )

    def test_init_unexpected_trial_nums(self):
        # Array
        ok_states, ok_povms, ok_gates = self.array_states_povms_gates()
        ok_schedule_list = [
            [("state", 0), ("gate", 0), ("povm", 0)],
            [("state", 1), ("gate", 1), ("povm", 1)],
        ]

        # Act & Assert
        # Case1: Invalid Type
        ng_trial_nums = 1
        with pytest.raises(TypeError):
            # TypeError: 'trial_nums' must be a list with int elements
            _ = Experiment(
                states=ok_states,
                povms=ok_povms,
                gates=ok_gates,
                schedules=ok_schedule_list,
                trial_nums=ng_trial_nums,
            )

        # Act & Assert
        # Case2: Invalid Type
        ng_trial_nums = [1, "1"]
        with pytest.raises(TypeError):
            # TypeError: 'trial_nums' must be a list with int elements
            _ = Experiment(
                states=ok_states,
                povms=ok_povms,
                gates=ok_gates,
                schedules=ok_schedule_list,
                trial_nums=ng_trial_nums,
            )

        # Act & Assert
        # Case3: Invalid Length; The length of the schedule is 2.
        ng_trial_nums = [1, 1, 1]
        with pytest.raises(ValueError):
            # TypeError: 'trial_nums' must be a list with int elements
            _ = Experiment(
                states=ok_states,
                povms=ok_povms,
                gates=ok_gates,
                schedules=ok_schedule_list,
                trial_nums=ng_trial_nums,
            )
