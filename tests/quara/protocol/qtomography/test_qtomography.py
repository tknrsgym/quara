import pytest

from quara.objects.composite_system import CompositeSystem
from quara.objects.elemental_system import ElementalSystem
from quara.objects.matrix_basis import get_normalized_pauli_basis
from quara.objects.povm import (
    Povm,
    get_x_measurement,
    get_y_measurement,
    get_z_measurement,
)
from quara.objects.qoperations import SetQOperations
from quara.objects.state import get_z0_1q
from quara.protocol.qtomography.qtomography import QTomography
from quara.protocol.qtomography.standard.standard_qst import StandardQst
from quara.qcircuit.experiment import Experiment


def get_test_data():
    e_sys = ElementalSystem(0, get_normalized_pauli_basis())
    c_sys = CompositeSystem([e_sys])

    povm_x = get_x_measurement(c_sys)
    povm_y = get_y_measurement(c_sys)
    povm_z = get_z_measurement(c_sys)
    povms = [povm_x, povm_y, povm_z]

    schedules = []
    for index in range(len(povms)):
        schedule = [("state", 0), ("povm", index)]
        schedules.append(schedule)

    experiment = Experiment(states=[None], gates=[], povms=povms, schedules=schedules)
    set_qoperations = SetQOperations(states=[get_z0_1q(c_sys)], gates=[], povms=[])

    return experiment, set_qoperations


class TestQTomography:
    def test_num_schedules(self):
        experiment, set_qoperations = get_test_data()

        qt = QTomography(experiment, set_qoperations)
        assert qt.num_schedules == 3
