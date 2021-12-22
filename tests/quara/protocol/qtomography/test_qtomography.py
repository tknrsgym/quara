import pytest

import numpy as np

from quara.objects.composite_system import CompositeSystem
from quara.objects.elemental_system import ElementalSystem
from quara.objects.gate import get_x
from quara.objects.gate_typical import generate_gate_from_gate_name
from quara.objects.matrix_basis import get_normalized_pauli_basis, get_comp_basis
from quara.objects.mprocess_typical import generate_mprocess_from_name
from quara.objects.povm import (
    Povm,
    get_x_povm,
    get_y_povm,
    get_z_povm,
)
from quara.objects.qoperations import SetQOperations
from quara.objects.state import get_z0_1q
from quara.protocol.qtomography.qtomography import QTomography
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
        states=[None], gates=[], povms=povms, schedules=schedules, seed_data=seed
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

    # TODO QOperation have non-real parameters
    """
    def test_init_error_not_real(self):
        e_sys = ElementalSystem(0, get_normalized_pauli_basis())
        c_sys = CompositeSystem([e_sys])

        # state is invalid
        vec = np.array([1j, 0, 0, 0], dtype=np.complex128)
        state = State(c_sys, vec, is_physicality_required=False)

        experiment = Experiment(states=[state], gates=[], povms=[], schedules=[])
        set_qoperations = SetQOperations(states=[], gates=[], povms=[])
        with pytest.raises(ValueError):
            QTomography(experiment, set_qoperations)

        # gate is invalid
        hs = np.array(
            [[1j, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
            dtype=np.complex128,
        )
        gate = Gate(c_sys, hs, is_physicality_required=False)

        experiment = Experiment(states=[], gates=[gate], povms=[], schedules=[])
        set_qoperations = SetQOperations(states=[], gates=[], povms=[])
        with pytest.raises(ValueError):
            QTomography(experiment, set_qoperations)

        # povm is invalid
        vecs = [
            np.array([1j, 0, 0, 0], dtype=np.complex128),
            np.array([1j, 0, 0, 0], dtype=np.complex128),
        ]
        povm = Povm(c_sys, vecs, is_physicality_required=False)

        experiment = Experiment(states=[], gates=[], povms=[povm], schedules=[])
        set_qoperations = SetQOperations(states=[], gates=[], povms=[])
        with pytest.raises(ValueError):
            QTomography(experiment, set_qoperations)
    """

    def test_experiment(self):
        experiment, set_qoperations = get_test_data()

        qt = QTomography(experiment, set_qoperations)
        assert len(qt.experiment.schedules) == 3

    def test_set_qoperations(self):
        experiment, set_qoperations = get_test_data()

        qt = QTomography(experiment, set_qoperations)
        assert qt.set_qoperations.num_states() == 1

    def test_num_schedules(self):
        experiment, set_qoperations = get_test_data()

        qt = QTomography(experiment, set_qoperations)
        assert qt.num_schedules == 3

    def test_access_states(self):
        experiment, set_qoperations = get_test_data()

        qt = QTomography(experiment, set_qoperations)
        assert len(qt.states) == 1

    def test_access_gates(self):
        e_sys = ElementalSystem(0, get_normalized_pauli_basis())
        c_sys = CompositeSystem([e_sys])

        gate_x = generate_gate_from_gate_name("x", c_sys)
        gate_y = generate_gate_from_gate_name("y", c_sys)
        gates = [gate_x, gate_y]

        povm_x = get_x_povm(c_sys)
        povm_y = get_y_povm(c_sys)
        povm_z = get_z_povm(c_sys)
        povms = [povm_x, povm_y, povm_z]

        schedules = []
        for index in range(len(gates)):
            schedule = [("state", 0), ("gate", index), ("povm", index)]
            schedules.append(schedule)

        seed = 7

        experiment = Experiment(
            states=[None], gates=gates, povms=povms, schedules=schedules, seed_data=seed
        )
        set_qoperations = SetQOperations(states=[get_z0_1q(c_sys)], gates=[], povms=[])

        qt = QTomography(experiment, set_qoperations)
        assert len(qt.gates) == 2

    def test_access_povms(self):
        experiment, set_qoperations = get_test_data()

        qt = QTomography(experiment, set_qoperations)
        assert len(qt.povms) == 3

    def test_access_mprocesss(self):
        e_sys = ElementalSystem(0, get_normalized_pauli_basis())
        c_sys = CompositeSystem([e_sys])

        mprocess_x = generate_mprocess_from_name(c_sys, "x-type1")
        mprocess_y = generate_mprocess_from_name(c_sys, "y-type1")
        mprocess_z = generate_mprocess_from_name(c_sys, "z-type1")
        mprocess_z2 = generate_mprocess_from_name(c_sys, "z-type1")
        mprocesses = [mprocess_x, mprocess_y, mprocess_z, mprocess_z2]

        povm_x = get_x_povm(c_sys)
        povm_y = get_y_povm(c_sys)
        povm_z = get_z_povm(c_sys)
        povm_z2 = get_z_povm(c_sys)
        povms = [povm_x, povm_y, povm_z, povm_z2]

        schedules = []
        for index in range(len(mprocesses)):
            schedule = [("state", 0), ("mprocess", index), ("povm", index)]
            schedules.append(schedule)

        seed = 7

        experiment = Experiment(
            states=[None],
            mprocesses=mprocesses,
            povms=povms,
            schedules=schedules,
            seed_data=seed,
        )
        set_qoperations = SetQOperations(
            states=[get_z0_1q(c_sys)], mprocesses=[], povms=[]
        )

        qt = QTomography(experiment, set_qoperations)
        assert len(qt.mprocesses) == 4

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
            states=[state_0], gates=[], povms=povms, schedules=schedules, seed_data=seed
        )
        set_qoperations = SetQOperations(states=[state_0], gates=[], povms=[])

        qt = QTomography(experiment, set_qoperations)

        # init
        actual = experiment.generate_data(1, 10)
        expected = [0, 1, 0, 1, 1, 1, 1, 0, 0, 0]
        assert np.all(actual == expected)

        # reset
        seed = 77
        experiment.reset_seed_data(seed)
        actual = experiment.generate_data(1, 10)
        expected = [1, 1, 1, 0, 0, 1, 0, 1, 0, 1]
        assert np.all(actual == expected)

        experiment.reset_seed_data(experiment.seed_data)
        actual = experiment.generate_data(1, 10)
        expected = [1, 1, 1, 0, 0, 1, 0, 1, 0, 1]
        assert np.all(actual == expected)
