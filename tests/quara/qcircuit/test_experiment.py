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
from quara.qcircuit.experiment import Experiment


class TestExperiment:
    def test_init_ok(self):
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

        schedule_list = [
            [("state", 0), ("gate", 0), ("povm", 0)],
            [("state", 1), ("gate", 1), ("povm", 1)],
        ]

        # Act & Assert
        _ = Experiment(
            states=state_list, povms=povm_list, gates=gate_list, schedules=schedule_list
        )

        # Arrange
        state_list = [state_0, None]
        povm_list = [None, povm_1]
        gate_list = [gate_0, None]

        # Act & Assert
        # Noneが含まれていてもOK
        _ = Experiment(
            states=state_list, povms=povm_list, gates=gate_list, schedules=schedule_list
        )

    def test_init_unexpected_type(self):
        # Array
        e_sys = ElementalSystem(0, matrix_basis.get_comp_basis())
        c_sys = CompositeSystem([e_sys])
        # State
        state_0 = get_x0_1q(c_sys)
        state_1 = get_y0_1q(c_sys)
        ok_state_list = [state_0, state_1]

        # POVM
        povm_0 = get_x_measurement(c_sys)
        povm_1 = get_x_measurement(c_sys)
        ok_povm_list = [povm_0, povm_1]
        # Gate
        gate_0 = get_i(c_sys)
        gate_1 = get_h(c_sys)
        ok_gate_list = [gate_0, gate_1]

        schedule_list = [
            [("state", 0), ("gate", 0), ("povm", 0)],
            [("state", 1), ("gate", 1), ("povm", 1)],
        ]

        # Act & Assert
        # Case1: Invalid states
        ng_state_list = [state_0, povm_0]
        with pytest.raises(TypeError):
            # TypeError: 'states' must be a list of State.
            _ = Experiment(
                states=ng_state_list,
                povms=ok_povm_list,
                gates=ok_gate_list,
                schedules=schedule_list,
            )

        # Case2: Invalid povms
        ng_povm_list = [state_0, povm_0]
        with pytest.raises(TypeError):
            # TypeError: 'povms' must be a list of Povm.
            _ = Experiment(
                states=ok_state_list,
                povms=ng_povm_list,
                gates=ok_gate_list,
                schedules=schedule_list,
            )

        # Case3: Invalid gates
        ng_gate_list = [gate_0, povm_0]
        with pytest.raises(TypeError):
            # TypeError: 'povms' must be a list of Povm.
            _ = Experiment(
                states=ok_state_list,
                povms=ok_povm_list,
                gates=ng_gate_list,
                schedules=schedule_list,
            )
