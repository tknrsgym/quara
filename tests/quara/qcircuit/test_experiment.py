from typing import Dict, List

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
from quara.objects.state import State, get_x0_1q, get_y0_1q, get_z0_1q
from quara.qcircuit.experiment import (
    Experiment,
    QuaraScheduleItemError,
    QuaraScheduleOrderError,
)
from quara.objects.operators import composite, tensor_product


class TestExperiment:
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
        povms = [povm_0, povm_1]
        # Gate
        gate_0 = get_i(c_sys)
        gate_1 = get_h(c_sys)
        gates = [gate_0, gate_1]
        return states, povms, gates

    def array_experiment_data(self):
        # Array
        e_sys1 = ElementalSystem(1, matrix_basis.get_normalized_pauli_basis())
        c_sys1 = CompositeSystem([e_sys1])

        state_list = [get_x0_1q(c_sys1), get_y0_1q(c_sys1)]
        gate_list = [get_i(c_sys1), get_x(c_sys1)]
        povm_list = [get_x_measurement(c_sys1), get_y_measurement(c_sys1)]
        schedules = [
            [("state", 0), ("gate", 0), ("povm", 0)],
            [("state", 0), ("gate", 0), ("povm", 1)],
        ]
        exp = Experiment(
            states=state_list, povms=povm_list, gates=gate_list, schedules=schedules,
        )
        return exp

    def array_experiment_data_2qubit(self):
        # Array
        e_sys1 = ElementalSystem(1, matrix_basis.get_normalized_pauli_basis())
        c_sys1 = CompositeSystem([e_sys1])
        e_sys2 = ElementalSystem(2, matrix_basis.get_normalized_pauli_basis())
        c_sys2 = CompositeSystem([e_sys2])
        c_sys12 = CompositeSystem([e_sys1, e_sys2])

        # Gate
        cnot = get_cnot(c_sys12, e_sys1)
        swap = get_swap(c_sys12)
        cz = get_cz(c_sys12)

        # POVM
        povm_xx = get_xx_measurement(c_sys12)
        povm_xy = get_xy_measurement(c_sys12)
        povm_yy = get_yy_measurement(c_sys12)
        povm_zz = get_zz_measurement(c_sys12)

        # State
        state1 = get_z0_1q(c_sys1)
        state2 = get_z0_1q(c_sys2)
        h1 = get_h(c_sys1)
        state1 = composite(h1, state1)
        state12 = tensor_product(state1, state2)

        state_list = [state12]
        povm_list = [povm_xx, povm_xy, povm_yy, povm_zz]
        gate_list = [cnot, swap, cz]

        schedules = [
            [("state", 0), ("gate", 0), ("povm", 0)],
            [("state", 0), ("gate", 0), ("povm", 1)],
            [("state", 0), ("gate", 0), ("povm", 2)],
            [("state", 0), ("gate", 1), ("povm", 0)],
        ]
        exp = Experiment(
            states=state_list, povms=povm_list, gates=gate_list, schedules=schedules,
        )
        return exp

    def array_experiment_data_2qubit_2gate(self):
        # Array
        e_sys1 = ElementalSystem(1, matrix_basis.get_normalized_pauli_basis())
        c_sys1 = CompositeSystem([e_sys1])
        e_sys2 = ElementalSystem(2, matrix_basis.get_normalized_pauli_basis())
        c_sys2 = CompositeSystem([e_sys2])
        c_sys12 = CompositeSystem([e_sys1, e_sys2])

        # Gate
        cnot = get_cnot(c_sys12, e_sys2)
        swap = get_swap(c_sys12)

        # State
        state1 = get_z0_1q(c_sys1)
        state2 = get_z0_1q(c_sys2)
        h = get_h(c_sys2)
        state2 = composite(h, state2)
        state12 = tensor_product(state1, state2)

        # POVM
        povm_xx = get_xx_measurement(c_sys12)
        povm_xy = get_xy_measurement(c_sys12)
        povm_yy = get_yy_measurement(c_sys12)

        state_list = [state12]
        povm_list = [povm_xx, povm_xy, povm_yy]
        gate_list = [cnot, swap]

        schedules = [
            [("state", 0), ("gate", 0), ("gate", 1), ("povm", 0)],
            [("state", 0), ("gate", 0), ("gate", 1), ("povm", 1)],
            [("state", 0), ("gate", 0), ("gate", 1), ("povm", 2)],
        ]
        exp = Experiment(
            states=state_list, povms=povm_list, gates=gate_list, schedules=schedules,
        )
        return exp

    def test_copy(self):
        e_sys = ElementalSystem(0, matrix_basis.get_normalized_pauli_basis())
        c_sys = CompositeSystem([e_sys])

        states = [get_z0_1q(c_sys)]
        gates = [get_x(c_sys)]
        povm_x = get_x_measurement(c_sys)
        povm_y = get_y_measurement(c_sys)
        povm_z = get_z_measurement(c_sys)
        povms = [povm_x, povm_y, povm_z]
        schedules = [
            [("state", 0), ("povm", 0)],
            [("state", 0), ("povm", 1)],
            [("state", 0), ("povm", 2)],
        ]
        experiment = Experiment(
            states=states, gates=gates, povms=povms, schedules=schedules,
        )
        experiment_copy = experiment.copy()

        assert experiment_copy.states is not experiment.states
        for actual, expected in zip(experiment_copy.states, experiment.states):
            assert np.all(actual.vec == expected.vec)

        assert experiment_copy.gates is not experiment.gates
        for actual, expected in zip(experiment_copy.gates, experiment.gates):
            assert np.all(actual.hs == expected.hs)

        assert experiment_copy.povms is not experiment.povms
        for actual, expected in zip(experiment_copy.povms, experiment.povms):
            assert np.all(actual.vecs == expected.vecs)

    def test_calc_prob_dist(self):
        # Array
        exp = self.array_experiment_data()

        # Case 1:
        # Act
        actual = exp.calc_prob_dist(schedule_index=0)

        # Assert
        expected = [1, 0]
        npt.assert_almost_equal(actual, expected, decimal=15)

        # Case 2:
        # Act
        actual = exp.calc_prob_dist(schedule_index=1)

        # Assert
        expected = [0.5, 0.5]
        npt.assert_almost_equal(actual, expected, decimal=15)

        # Case 3: Exception
        ng_schedule_index = len(exp.schedules)
        with pytest.raises(IndexError):
            # IndexError: The value of 'schedule_index' must be an integer between 0 and 1.
            _ = exp.calc_prob_dist(schedule_index=ng_schedule_index)

        # Case 4:
        ng_schedule_index = 0.1
        with pytest.raises(TypeError):
            # TypeError: The type of 'schedule_index' must be int.
            _ = exp.calc_prob_dist(schedule_index=ng_schedule_index)

    def test_calc_prob_dists(self):
        # Array
        e_sys1 = ElementalSystem(1, matrix_basis.get_normalized_pauli_basis())
        c_sys1 = CompositeSystem([e_sys1])

        state_list = [get_x0_1q(c_sys1), get_y0_1q(c_sys1)]
        gate_list = [get_i(c_sys1), get_x(c_sys1)]
        povm_list = [get_x_measurement(c_sys1), get_y_measurement(c_sys1)]
        schedules = [
            [("state", 0), ("gate", 0), ("povm", 0)],
            [("state", 0), ("gate", 0), ("povm", 1)],
        ]
        exp = Experiment(
            states=state_list, povms=povm_list, gates=gate_list, schedules=schedules,
        )

        # Act
        actual = exp.calc_prob_dists()

        # Assert
        expected = [[1, 0], [0.5, 0.5]]
        assert len(actual) == len(expected)
        for a, e in zip(actual, expected):
            npt.assert_almost_equal(a, e, decimal=15)

    def test_calc_prob_dist_exist_none(self):
        # Array
        e_sys1 = ElementalSystem(1, matrix_basis.get_normalized_pauli_basis())
        c_sys1 = CompositeSystem([e_sys1])

        state_list = [None, get_y0_1q(c_sys1)]
        gate_list = [get_i(c_sys1), get_x(c_sys1)]
        povm_list = [get_x_measurement(c_sys1), get_y_measurement(c_sys1)]
        schedules = [
            [("state", 0), ("gate", 0), ("povm", 0)],
            [("state", 0), ("gate", 0), ("povm", 1)],
        ]
        exp = Experiment(
            states=state_list, povms=povm_list, gates=gate_list, schedules=schedules,
        )

        # Act
        with pytest.raises(ValueError):
            # ValueError: states[0] is None.
            _ = exp.calc_prob_dist(schedule_index=0)

    def test_calc_prob_dist_2qubit(self):
        # Array
        exp = self.array_experiment_data_2qubit()

        # Case 1:
        # Act
        actual = exp.calc_prob_dist(schedule_index=0)

        # Assert
        expected = [0.5, 0, 0, 0.5]
        npt.assert_almost_equal(actual, expected, decimal=15)

        # Case 2:
        # Act
        actual = exp.calc_prob_dist(schedule_index=1)

        # Assert
        expected = [0.25, 0.25, 0.25, 0.25]
        npt.assert_almost_equal(actual, expected, decimal=15)

        # Case 3:
        # Act
        actual = exp.calc_prob_dist(schedule_index=2)

        # Assert
        expected = [0, 0.5, 0.5, 0]
        npt.assert_almost_equal(actual, expected, decimal=15)

        # Case 4:
        # Act
        actual = exp.calc_prob_dist(schedule_index=3)

        # Assert
        expected = [0.5, 0, 0.5, 0]
        npt.assert_almost_equal(actual, expected, decimal=15)

    def test_calc_prob_dist_2qubit_2gate(self):
        # Array
        exp = self.array_experiment_data_2qubit_2gate()

        # Case 1:
        # Act
        actual = exp.calc_prob_dist(schedule_index=0)

        # Assert
        expected = [0.5, 0, 0, 0.5]
        npt.assert_almost_equal(actual, expected, decimal=15)

        # Case 2:
        # Act
        actual = exp.calc_prob_dist(schedule_index=1)

        # Assert
        expected = [0.25, 0.25, 0.25, 0.25]
        npt.assert_almost_equal(actual, expected, decimal=15)

        # Case 3:
        # Act
        actual = exp.calc_prob_dist(schedule_index=2)

        # Assert
        expected = [0, 0.5, 0.5, 0]
        npt.assert_almost_equal(actual, expected, decimal=15)

    def test_generate_data(self):
        # Array
        exp = self.array_experiment_data()

        # Act
        # Case 1:
        actual = exp.generate_data(schedule_index=0, data_num=10, seed=7)

        # Assert
        expected = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        assert actual == expected

        # Case 2:
        actual = exp.generate_data(schedule_index=1, data_num=20, seed=77)

        # Assert
        expected = [1, 1, 1, 0, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1, 0, 0, 1, 0, 0, 1]
        assert actual == expected

        # Case 3:
        actual = exp.generate_data(schedule_index=1, data_num=0, seed=7)

        # Assert
        expected = []
        assert actual == expected

    def test_generate_data_exception(self):
        # Array
        exp = self.array_experiment_data()

        # Act
        # Case 1:
        ng_data_num = -1
        with pytest.raises(ValueError):
            # ValueError: The value of 'data_num' must be a non-negative integer.
            _ = exp.generate_data(schedule_index=0, data_num=ng_data_num, seed=7)

        # Case 2:
        ng_data_num = 0.1
        with pytest.raises(TypeError):
            # TypeError: The type of 'data_num' must be int.
            _ = exp.generate_data(schedule_index=0, data_num=ng_data_num, seed=7)

        # Case 3:
        ng_schedule_index = len(exp.schedules)

        with pytest.raises(IndexError):
            # IndexError: The value of 'schedule_index' must be an integer between 0 and 1.
            _ = exp.generate_data(schedule_index=ng_schedule_index, data_num=10, seed=7)

        # Case 4:
        ng_schedule_index = 0.1
        with pytest.raises(TypeError):
            # TypeError: The type of 'schedule_index' must be int.
            _ = exp.generate_data(schedule_index=ng_schedule_index, data_num=10, seed=7)

    def test_generate_dataset(self):
        # Array
        exp = self.array_experiment_data()

        # Case 1:
        actual = exp.generate_dataset(data_nums=[10, 20], seeds=[7, 77])
        expected = [
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [1, 1, 1, 0, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1, 0, 0, 1, 0, 0, 1],
        ]
        assert actual == expected

        # Case 2:
        actual = exp.generate_dataset(data_nums=[0, 5], seeds=[7, 77])
        expected = [[], [1, 1, 1, 0, 0]]
        assert actual == expected

    def test_generate_dataset_exception(self):
        # Array
        exp = self.array_experiment_data()
        ok_data_nums = [10, 20]
        ok_seed_list = [7, 77]

        # Case 1:
        ng_data_nums = [1, 1, 1]

        with pytest.raises(ValueError):
            # ValueError: The number of elements in 'data_nums' must be the same as the number of 'schedules';
            _ = exp.generate_dataset(data_nums=ng_data_nums, seeds=ok_seed_list)

        # Case 2:
        ng_data_nums = 1
        with pytest.raises(TypeError):
            # TypeError: The type of 'data_nums' must be list.
            _ = exp.generate_dataset(data_nums=ng_data_nums, seeds=ok_seed_list)

        # Case 3:
        ng_seed_list = [1]
        with pytest.raises(ValueError):
            # ValueError: The number of elements in 'seeds' must be the same as the number of 'schedules';
            _ = exp.generate_dataset(data_nums=ok_data_nums, seeds=[ng_seed_list])

        # Case 4:
        ng_seed_list = 1
        with pytest.raises(TypeError):
            # TypeError: The type of 'seeds' must be list.
            _ = exp.generate_dataset(data_nums=ok_data_nums, seeds=ng_seed_list)

    def test_generate_empi_dist(self):
        # Array
        exp = self.array_experiment_data()

        # Act
        # Case 1:
        # probdist: [1, 0]
        # data: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        num_sums = [5, 10, 20]
        # seed_list = [7, 77]
        actual = exp.generate_empi_dist(schedule_index=0, num_sums=num_sums)

        # Assert
        expected_1 = [
            (5, np.array([1, 0])),
            (10, np.array([1, 0])),
            (20, np.array([1, 0])),
        ]
        assert len(actual) == len(expected_1)
        for a, e in zip(actual, expected_1):
            assert a[0] == e[0]
            assert np.all(a[1] == e[1])

        # Act
        # Case 2:
        # probdist: [0.5, 0.5]
        # data: [1, 1, 1, 0, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1, 0, 0, 1, 0, 0, 1]
        num_sums = [5, 10, 15, 20]
        actual = exp.generate_empi_dist(schedule_index=1, num_sums=num_sums, seed=77)

        # Assert
        expected_2 = [
            (5, np.array([0.4, 0.6])),
            (10, np.array([0.4, 0.6])),
            (15, np.array([6 / 15, 9 / 15])),
            (20, np.array([0.45, 0.55])),
        ]
        assert len(actual) == len(expected_2)
        for a, e in zip(actual, expected_2):
            assert a[0] == e[0]
            assert np.all(a[1] == e[1])

    def test_generate_empi_dist_exception(self):
        # Array
        exp = self.array_experiment_data()
        ok_num_sums = [7]

        # Act & Assert
        # Case 1:
        ng_schedule_index = 0.1
        with pytest.raises(TypeError):
            actual = exp.generate_empi_dist(
                schedule_index=ng_schedule_index, num_sums=ok_num_sums, seed=7
            )

        # Act & Assert
        # Case 2:
        ng_schedule_index = len(exp.schedules)
        with pytest.raises(IndexError):
            actual = exp.generate_empi_dist(
                schedule_index=ng_schedule_index, num_sums=ok_num_sums, seed=7
            )

    def test_generate_empi_dists_sequence(self):
        # Array
        exp = self.array_experiment_data()
        list_num_sums = [[5, 10, 20], [5, 10, 15, 20]]
        seeds = [7, 77]

        # Act
        actual = exp.generate_empi_dists_sequence(
            list_num_sums=list_num_sums, seeds=seeds
        )

        # Assert
        expected = [
            [(5, np.array([1, 0])), (10, np.array([1, 0])), (20, np.array([1, 0])),],
            [
                (5, np.array([0.4, 0.6])),
                (10, np.array([0.4, 0.6])),
                (15, np.array([6 / 15, 9 / 15])),
                (20, np.array([0.45, 0.55])),
            ],
        ]
        assert len(actual) == len(expected)
        for a, e in zip(actual, expected):
            assert len(a) == len(e)
            for a_item, e_item in zip(a, e):
                assert a_item[0] == e_item[0]
                assert np.all(a_item[1] == e_item[1])

    def test_generate_empi_dists_sequence_exception(self):
        # Array
        exp = self.array_experiment_data()
        ok_list_num_sums = [[5, 10, 20], [5, 10, 15, 20]]
        ok_seeds = [7, 77]

        # Act
        # Case 1:
        ng_list_num_sums = [[5, 10, 20]]
        with pytest.raises(ValueError):
            actual = exp.generate_empi_dists_sequence(
                list_num_sums=ng_list_num_sums, seeds=ok_seeds
            )

        # Case 2:
        ng_list_num_sums = [[5, 10, 20], [5, 10, 20], [5, 10, 20]]
        with pytest.raises(ValueError):
            actual = exp.generate_empi_dists_sequence(
                list_num_sums=ng_list_num_sums, seeds=ok_seeds
            )

        # Case 3:
        ng_seeds = [7]
        with pytest.raises(ValueError):
            actual = exp.generate_empi_dists_sequence(
                list_num_sums=ok_list_num_sums, seeds=ng_seeds
            )

        # Case 4:
        ng_seeds = [7, 77, 777]
        with pytest.raises(ValueError):
            actual = exp.generate_empi_dists_sequence(
                list_num_sums=ok_list_num_sums, seeds=ng_seeds
            )

    def test_getter(self):
        # Array
        states, povms, gates = self.array_states_povms_gates()
        schedules = [
            [("state", 0), ("gate", 0), ("povm", 0)],
            [("state", 1), ("gate", 1), ("povm", 1)],
        ]

        # Act & Assert
        _ = Experiment(states=states, povms=povms, gates=gates, schedules=schedules,)

        # Arrange
        source_states = [states[0], None]
        source_povms = [None, povms[0]]
        source_gates = [gates[0], None]

        # Act & Assert
        exp = Experiment(
            states=source_states,
            povms=source_povms,
            gates=source_gates,
            schedules=schedules,
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

        actual, expected = exp.schedules, schedules
        assert len(actual) == len(expected)
        for a, e in zip(actual, expected):
            assert a == e

    def test_setter_validation(self):
        # Array
        states, povms, gates = self.array_states_povms_gates()
        schedules = [
            [("state", 0), ("gate", 0), ("povm", 0)],
            [("state", 1), ("gate", 1), ("povm", 1)],
        ]
        ok_new_states = [states[1], states[0]]
        ok_new_povms = [povms[1], povms[0]]
        ok_new_gates = [gates[1], gates[0]]
        ok_new_schedules = [schedules[1], schedules[0]]

        exp = Experiment(states=states, povms=povms, gates=gates, schedules=schedules,)

        # State
        # Act & Assert
        ng_new_states = [states[0], povms[0]]
        with pytest.raises(TypeError):
            # TypeError: 'states' must be a list of State.
            exp.states = ng_new_states
        assert exp.states == states

        # Act & Assert
        ng_new_states = [states[0]]
        with pytest.raises(QuaraScheduleItemError):
            # New State does not match schedules.
            exp.states = ng_new_states
        # Assert
        assert exp.states == states

        # Act & Assert
        exp.states = ok_new_states
        assert exp.states == ok_new_states

        # POVM
        # Act & Assert
        ng_new_povms = [states[0], povms[0]]
        with pytest.raises(TypeError):
            # TypeError: 'povms' must be a list of Povm.
            exp.povms = ng_new_povms
        assert exp.povms == povms

        # Act & Assert
        ng_new_povms = [povms[0]]
        with pytest.raises(QuaraScheduleItemError):
            # New 'povms' does not match schedules.
            exp.povms = ng_new_povms
        assert exp.povms == povms

        # Act & Assert
        exp.povms = ok_new_povms
        assert exp.povms == ok_new_povms

        # Gate
        # Act & Assert
        ng_new_gates = [states[0], gates[0]]
        with pytest.raises(TypeError):
            # TypeError: 'gates' must be a list of Gate.
            exp.gates = ng_new_gates
        assert exp.gates == gates

        # Act & Assert
        ng_new_gates = [gates[0]]
        with pytest.raises(QuaraScheduleItemError):
            # New 'gates' does not match schedules.
            exp.gates = ng_new_gates
        assert exp.gates == gates

        # Act & Assert
        exp.gates = ok_new_gates
        assert exp.gates == ok_new_gates

        # Schedule
        # Act & Assert
        ng_new_schedules = [
            [("povm"), ("gate", 1), ("povm", 1)],
            [("state", 1), ("gate", 1), ("povm", 1)],
        ]
        with pytest.raises(QuaraScheduleItemError):
            # A schedule item must be a tuple of str and int.
            exp.schedules = ng_new_schedules
        assert exp.schedules == schedules

        # Act & Assert
        ng_new_schedules = [
            [("povm", 1), ("gate", 1), ("povm", 1)],
            [("state", 1), ("gate", 1), ("povm", 1)],
        ]
        with pytest.raises(QuaraScheduleOrderError):
            # he first element of the schedule must be a 'state'.
            exp.schedules = ng_new_schedules
        assert exp.schedules == schedules

        # Act & Assert
        exp.schedules = ok_new_schedules
        assert exp.schedules == ok_new_schedules

    def test_init_unexpected_type(self):
        # Array
        ok_states, ok_povms, ok_gates = self.array_states_povms_gates()
        schedules = [
            [("state", 0), ("gate", 0), ("povm", 0)],
            [("state", 1), ("gate", 1), ("povm", 1)],
        ]

        # Act & Assert
        # Case1: Invalid states
        ng_states = [ok_states[0], ok_povms[0]]
        with pytest.raises(TypeError):
            # TypeError: 'states' must be a list of State.
            _ = Experiment(
                states=ng_states, povms=ok_povms, gates=ok_gates, schedules=schedules,
            )

        # Case2: Invalid povms
        ng_povms = [ok_states[0], ok_povms[0]]
        with pytest.raises(TypeError):
            # TypeError: 'povms' must be a list of Povm.
            _ = Experiment(
                states=ok_states, povms=ng_povms, gates=ok_gates, schedules=schedules,
            )

        # Case3: Invalid gates
        ng_gates = [ok_gates[0], ok_povms[0]]
        with pytest.raises(TypeError):
            # TypeError: 'povms' must be a list of Povm.
            _ = Experiment(
                states=ok_states, povms=ok_povms, gates=ng_gates, schedules=schedules,
            )

    def test_expeption_order_too_short_schedule(self):
        # Array
        ok_states, ok_povms, ok_gates = self.array_states_povms_gates()
        ng_schedules = [
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
                schedules=ng_schedules,
            )

    def test_expeption_order_not_start_with_state(self):
        # Array
        ok_states, ok_povms, ok_gates = self.array_states_povms_gates()
        ng_schedules = [
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
                schedules=ng_schedules,
            )

    def test_expeption_order_not_end_with_povm_mprocess(self):
        # Array
        ok_states, ok_povms, ok_gates = self.array_states_povms_gates()
        ng_schedules = [
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
                schedules=ng_schedules,
            )

        # TODO: mprocessを実装後、mprocessで終わるスケジュールを含めたテストを追加する

    def test_expeption_order_too_many_state(self):
        # Array
        ok_states, ok_povms, ok_gates = self.array_states_povms_gates()
        ng_schedules = [
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
                schedules=ng_schedules,
            )

    def test_expeption_order_too_many_povm(self):
        # Array
        ok_states, ok_povms, ok_gates = self.array_states_povms_gates()
        ng_schedules = [
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
                schedules=ng_schedules,
            )

    def test_exception_item_no_mprocess(self):
        # Array
        ok_states, ok_povms, ok_gates = self.array_states_povms_gates()
        ng_schedules = [
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
                schedules=ng_schedules,
            )

    def test_exception_item_not_tuple(self):
        # Array
        ok_states, ok_povms, ok_gates = self.array_states_povms_gates()
        ng_schedules = [
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
                schedules=ng_schedules,
            )

    def test_exception_item_too_short(self):
        # Array
        ok_states, ok_povms, ok_gates = self.array_states_povms_gates()
        ng_schedules = [
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
                schedules=ng_schedules,
            )

    def test_exception_item_too_long(self):
        # Array
        ok_states, ok_povms, ok_gates = self.array_states_povms_gates()
        ng_schedules = [
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
                schedules=ng_schedules,
            )

    def test_exception_item_invalid_name_type(self):
        # Array
        ok_states, ok_povms, ok_gates = self.array_states_povms_gates()
        ng_schedules = [
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
                schedules=ng_schedules,
            )

    def test_exception_item_invalid_index_type(self):
        # Array
        ok_states, ok_povms, ok_gates = self.array_states_povms_gates()
        ng_schedules = [
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
                schedules=ng_schedules,
            )

    def test_exception_item_unknown_name(self):
        # Array
        ok_states, ok_povms, ok_gates = self.array_states_povms_gates()
        ng_schedules = [
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
                schedules=ng_schedules,
            )

    def test_expeption_item_out_of_range(self):
        # Array
        ok_states, ok_povms, ok_gates = self.array_states_povms_gates()
        ng_schedules = [
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
                schedules=ng_schedules,
            )
