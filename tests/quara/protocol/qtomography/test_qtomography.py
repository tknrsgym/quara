import pytest

import numpy as np

from quara.objects.composite_system import CompositeSystem
from quara.objects.elemental_system import ElementalSystem
from quara.objects.gate import get_x
from quara.objects.matrix_basis import get_normalized_pauli_basis, get_comp_basis
from quara.objects.povm import (
    Povm,
    get_x_povm,
    get_y_povm,
    get_z_povm,
)
from quara.objects.qoperations import SetQOperations
from quara.objects.state import get_z0_1q
from quara.protocol.qtomography.qtomography import QTomography
from quara.protocol.qtomography.standard.standard_qst import StandardQst
from quara.qcircuit.experiment import Experiment


def get_test_data():
    e_sys = ElementalSystem(0, get_normalized_pauli_basis())
    c_sys = CompositeSystem([e_sys])

    povm_x = get_x_povm(c_sys)
    povm_y = get_y_povm(c_sys)
    povm_z = get_z_povm(c_sys)
    povms = [povm_x, povm_y, povm_z]

    schedules = []
    for index in range(len(povms)):
        schedule = [("state", 0), ("povm", index)]
        schedules.append(schedule)

    seed = 7

    experiment = Experiment(
        states=[None], gates=[], povms=povms, schedules=schedules, seed=seed
    )
    set_qoperations = SetQOperations(states=[get_z0_1q(c_sys)], gates=[], povms=[])

    return experiment, set_qoperations


class TestQTomography:
    def test_init_error(self):
        e_sys = ElementalSystem(0, get_comp_basis())
        c_sys = CompositeSystem([e_sys])

        # state is invalid
        state = get_z0_1q(c_sys)

        experiment = Experiment(states=[state], gates=[], povms=[], schedules=[])
        set_qoperations = SetQOperations(states=[], gates=[], povms=[])
        with pytest.raises(ValueError):
            QTomography(experiment, set_qoperations)

        experiment = Experiment(states=[], gates=[], povms=[], schedules=[])
        set_qoperations = SetQOperations(states=[state], gates=[], povms=[])
        with pytest.raises(ValueError):
            QTomography(experiment, set_qoperations)

        # gate is invalid
        gate = get_x(c_sys)

        experiment = Experiment(states=[], gates=[gate], povms=[], schedules=[])
        set_qoperations = SetQOperations(states=[], gates=[], povms=[])
        with pytest.raises(ValueError):
            QTomography(experiment, set_qoperations)

        experiment = Experiment(states=[], gates=[], povms=[], schedules=[])
        set_qoperations = SetQOperations(states=[], gates=[gate], povms=[])
        with pytest.raises(ValueError):
            QTomography(experiment, set_qoperations)

        # povm is invalid
        povm = get_z_povm(c_sys)

        experiment = Experiment(states=[], gates=[], povms=[povm], schedules=[])
        set_qoperations = SetQOperations(states=[], gates=[], povms=[])
        with pytest.raises(ValueError):
            QTomography(experiment, set_qoperations)

        experiment = Experiment(states=[], gates=[], povms=[], schedules=[])
        set_qoperations = SetQOperations(states=[], gates=[], povms=[povm])
        with pytest.raises(ValueError):
            QTomography(experiment, set_qoperations)

    def test_num_schedules(self):
        experiment, set_qoperations = get_test_data()

        qt = QTomography(experiment, set_qoperations)
        assert qt.num_schedules == 3

    def test_reset_seed(self):
        # Set up
        e_sys = ElementalSystem(0, get_normalized_pauli_basis())
        c_sys = CompositeSystem([e_sys])

        state_0 = get_z0_1q(c_sys)

        povm_x = get_x_povm(c_sys)
        povm_y = get_y_povm(c_sys)
        povm_z = get_z_povm(c_sys)
        povms = [povm_x, povm_y, povm_z]

        schedules = []
        for index in range(len(povms)):
            schedule = [("state", 0), ("povm", index)]
            schedules.append(schedule)

        seed = 7

        experiment = Experiment(
            states=[state_0], gates=[], povms=povms, schedules=schedules, seed=seed
        )
        set_qoperations = SetQOperations(states=[state_0], gates=[], povms=[])

        qt = QTomography(experiment, set_qoperations)

        # init
        actual = experiment.generate_data(1, 10)
        expected = [0, 1, 0, 1, 1, 1, 1, 0, 0, 0]
        assert np.all(actual == expected)

        # reset
        seed = 77
        experiment.reset_seed(seed)
        actual = experiment.generate_data(1, 10)
        expected = [1, 1, 1, 0, 0, 1, 0, 1, 0, 1]
        assert np.all(actual == expected)

        experiment.reset_seed(experiment.seed)
        actual = experiment.generate_data(1, 10)
        expected = [1, 1, 1, 0, 0, 1, 0, 1, 0, 1]
        assert np.all(actual == expected)
