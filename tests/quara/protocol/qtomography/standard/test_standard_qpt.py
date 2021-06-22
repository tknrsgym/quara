import numpy as np
import numpy.testing as npt
import pytest

from quara.objects.composite_system import CompositeSystem
from quara.objects.elemental_system import ElementalSystem
from quara.objects.matrix_basis import get_normalized_pauli_basis
from quara.objects.povm import (
    Povm,
    get_x_povm,
    get_y_povm,
    get_z_povm,
)
from quara.objects.state import (
    State,
    get_x0_1q,
    get_y0_1q,
    get_z0_1q,
    get_z1_1q,
)
from quara.protocol.qtomography.standard.standard_qpt import StandardQpt
from quara.objects.gate_typical import generate_gate_x


def get_test_data(on_para_eq_constraint=False):
    e_sys = ElementalSystem(0, get_normalized_pauli_basis())
    c_sys = CompositeSystem([e_sys])

    # Tester Objects
    tester_states = [
        get_x0_1q(c_sys),
        get_y0_1q(c_sys),
        get_z0_1q(c_sys),
        get_z1_1q(c_sys),
    ]
    tester_povms = [get_x_povm(c_sys), get_y_povm(c_sys), get_z_povm(c_sys)]

    qpt = StandardQpt(
        tester_states, tester_povms, on_para_eq_constraint=on_para_eq_constraint, seed=7
    )

    return qpt, c_sys


class TestStandardQpt:
    def test_num_variables(self):
        # on_para_eq_constraint=True
        qpt, c_sys = get_test_data(on_para_eq_constraint=True)
        assert qpt.num_variables == 12

        # on_para_eq_constraint=False
        qpt, c_sys = get_test_data(on_para_eq_constraint=False)
        assert qpt.num_variables == 16

    def test_validate_schedules(self):
        e_sys = ElementalSystem(0, get_normalized_pauli_basis())
        c_sys = CompositeSystem([e_sys])

        # Tester Objects
        tester_states = [
            get_x0_1q(c_sys),
            get_y0_1q(c_sys),
            get_z0_1q(c_sys),
            get_z1_1q(c_sys),
        ]
        tester_povms = [get_x_povm(c_sys), get_y_povm(c_sys), get_z_povm(c_sys)]

        # Act
        qpt = StandardQpt(
            tester_states, tester_povms, on_para_eq_constraint=True, seed=777
        )

        # Assert
        actual = qpt._experiment._schedules
        expected = [
            [("state", 0), ("gate", 0), ("povm", 0)],
            [("state", 0), ("gate", 0), ("povm", 1)],
            [("state", 0), ("gate", 0), ("povm", 2)],
            [("state", 1), ("gate", 0), ("povm", 0)],
            [("state", 1), ("gate", 0), ("povm", 1)],
            [("state", 1), ("gate", 0), ("povm", 2)],
            [("state", 2), ("gate", 0), ("povm", 0)],
            [("state", 2), ("gate", 0), ("povm", 1)],
            [("state", 2), ("gate", 0), ("povm", 2)],
            [("state", 3), ("gate", 0), ("povm", 0)],
            [("state", 3), ("gate", 0), ("povm", 1)],
            [("state", 3), ("gate", 0), ("povm", 2)],
        ]
        assert len(actual) == 12
        assert actual == expected

        # Case 2:
        # Act
        qpt = StandardQpt(
            tester_states,
            tester_povms,
            on_para_eq_constraint=True,
            seed=777,
            schedules="all",
        )

        # Assert
        actual = qpt._experiment._schedules
        assert len(actual) == 12
        assert actual == expected
        # Case 3:
        # Act
        schedules = [
            [("state", 3), ("gate", 0), ("povm", 2)],
            [("state", 1), ("gate", 0), ("povm", 0)],
        ]
        qpt = StandardQpt(
            tester_states,
            tester_povms,
            on_para_eq_constraint=True,
            seed=777,
            schedules=schedules,
        )

        # Assert
        actual = qpt._experiment._schedules
        assert actual == schedules

        # Case 4:
        # Act
        invalid_schedules = "invalid str"
        with pytest.raises(ValueError):
            _ = StandardQpt(
                tester_states,
                tester_povms,
                on_para_eq_constraint=True,
                seed=777,
                schedules=invalid_schedules,
            )

    def test_generate_empi_dist(self):
        qpt, c_sys = get_test_data()
        gate = generate_gate_x(c_sys)
        assert qpt.num_schedules == 12

        # schedule_index = 0
        actual = qpt.generate_empi_dist(0, gate, 10)
        expected = (10, np.array([1.0, 0.0], dtype=np.float64))
        assert actual[0] == expected[0]
        npt.assert_almost_equal(actual[1], expected[1], decimal=15)

        # schedule_index = 1
        actual = qpt.generate_empi_dist(1, gate, 10)
        expected = (10, np.array([0.6, 0.4], dtype=np.float64))
        assert actual[0] == expected[0]
        npt.assert_almost_equal(actual[1], expected[1], decimal=15)

        # schedule_index = 2
        actual = qpt.generate_empi_dist(2, gate, 10)
        expected = (10, np.array([0.3, 0.7], dtype=np.float64))
        assert actual[0] == expected[0]
        npt.assert_almost_equal(actual[1], expected[1], decimal=15)

    def test_generate_empi_dist__seed_or_stream(self):
        qpt, c_sys = get_test_data()
        gate = generate_gate_x(c_sys)

        # seed_or_stream : default
        np.random.seed(7)
        actual1 = qpt.generate_empi_dist(0, gate, 10)
        # seed_or_stream : int
        actual2 = qpt.generate_empi_dist(0, gate, 10, seed_or_stream=7)
        # seed_or_stream : np.random.RandomState
        actual3 = qpt.generate_empi_dist(
            0, gate, 10, seed_or_stream=np.random.RandomState(7)
        )

    def test_generate_empi_dists(self):
        qpt, c_sys = get_test_data()
        gate = generate_gate_x(c_sys)

        actual = qpt.generate_empi_dists(gate, 10)
        expected = [
            (10, np.array([1.0, 0.0], dtype=np.float64)),
            (10, np.array([0.6, 0.4], dtype=np.float64)),
            (10, np.array([0.3, 0.7], dtype=np.float64)),
            (10, np.array([0.7, 0.3], dtype=np.float64)),
            (10, np.array([0.0, 1.0], dtype=np.float64)),
            (10, np.array([0.5, 0.5], dtype=np.float64)),
            (10, np.array([0.4, 0.6], dtype=np.float64)),
            (10, np.array([0.8, 0.2], dtype=np.float64)),
            (10, np.array([0.0, 1.0], dtype=np.float64)),
            (10, np.array([0.4, 0.6], dtype=np.float64)),
            (10, np.array([0.5, 0.5], dtype=np.float64)),
            (10, np.array([1.0, 0.0], dtype=np.float64)),
        ]
        for a, e in zip(actual, expected):
            assert a[0] == e[0]
            npt.assert_almost_equal(a[1], e[1], decimal=15)

    def test_generate_empi_dists_sequence(self):
        qpt, c_sys = get_test_data()
        gate = generate_gate_x(c_sys)

        actual = qpt.generate_empi_dists_sequence(gate, [10, 20])
        expected = [
            [
                (10, np.array([1.0, 0.0], dtype=np.float64)),
                (10, np.array([0.3, 0.7], dtype=np.float64)),
                (10, np.array([0.6, 0.4], dtype=np.float64)),
                (10, np.array([0.4, 0.6], dtype=np.float64)),
                (10, np.array([0.0, 1.0], dtype=np.float64)),
                (10, np.array([0.5, 0.5], dtype=np.float64)),
                (10, np.array([0.4, 0.6], dtype=np.float64)),
                (10, np.array([0.8, 0.2], dtype=np.float64)),
                (10, np.array([0.0, 1.0], dtype=np.float64)),
                (10, np.array([0.6, 0.4], dtype=np.float64)),
                (10, np.array([0.3, 0.7], dtype=np.float64)),
                (10, np.array([1.0, 0.0], dtype=np.float64)),
            ],
            [
                (20, np.array([1.0, 0.0], dtype=np.float64)),
                (20, np.array([0.5, 0.5], dtype=np.float64)),
                (20, np.array([0.55, 0.45], dtype=np.float64)),
                (20, np.array([0.6, 0.4], dtype=np.float64)),
                (20, np.array([0.0, 1.0], dtype=np.float64)),
                (20, np.array([0.5, 0.5], dtype=np.float64)),
                (20, np.array([0.4, 0.6], dtype=np.float64)),
                (20, np.array([0.65, 0.35], dtype=np.float64)),
                (20, np.array([0.0, 1.0], dtype=np.float64)),
                (20, np.array([0.6, 0.4], dtype=np.float64)),
                (20, np.array([0.3, 0.7], dtype=np.float64)),
                (20, np.array([1.0, 0.0], dtype=np.float64)),
            ],
        ]
        for a_dists, e_dists in zip(actual, expected):
            count = 0
            for a, e in zip(a_dists, e_dists):
                assert a[0] == e[0]
                npt.assert_almost_equal(a[1], e[1], decimal=15)
                count += 1
