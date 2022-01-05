from itertools import product

import numpy as np
import numpy.testing as npt
import pytest

from quara.objects.composite_system import CompositeSystem
from quara.objects.elemental_system import ElementalSystem
from quara.objects.matrix_basis import get_normalized_pauli_basis
from quara.objects.povm import (
    get_x_povm,
    get_y_povm,
    get_z_povm,
)
from quara.objects.state import (
    get_x0_1q,
    get_y0_1q,
    get_z0_1q,
    get_z1_1q,
)
from quara.protocol.qtomography.standard.standard_qpt import StandardQpt, calc_c_qpt
from quara.objects.gate_typical import generate_gate_x
from quara.objects.composite_system_typical import generate_composite_system
from quara.objects.qoperation_typical import generate_qoperation


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
        tester_states,
        tester_povms,
        on_para_eq_constraint=on_para_eq_constraint,
        seed_data=7,
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
            tester_states, tester_povms, on_para_eq_constraint=True, seed_data=777
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
            seed_data=777,
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
            seed_data=777,
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
                seed_data=777,
                schedules=invalid_schedules,
            )

    def test_testers(self):
        qpt, _ = get_test_data()
        assert len(qpt.testers) == 7

    def test_is_valid_experiment(self):
        # is_valid_experiment == True
        qpt, _ = get_test_data()
        assert qpt.is_valid_experiment() == True

        # is_valid_experiment == False
        e_sys0 = ElementalSystem(0, get_normalized_pauli_basis())
        c_sys0 = CompositeSystem([e_sys0])
        e_sys1 = ElementalSystem(1, get_normalized_pauli_basis())
        c_sys1 = CompositeSystem([e_sys1])

        povm_x = get_x_povm(c_sys1)
        povm_y = get_y_povm(c_sys0)
        povm_z = get_z_povm(c_sys0)
        povms = [povm_x, povm_y, povm_z]

        qpt.experiment.povms = povms
        assert qpt.is_valid_experiment() == False

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
        expected = (10, np.array([0.5, 0.5], dtype=np.float64))
        assert actual[0] == expected[0]
        npt.assert_almost_equal(actual[1], expected[1], decimal=15)

    def test_generate_empi_dist__seed_or_generator(self):
        qpt, c_sys = get_test_data()
        gate = generate_gate_x(c_sys)

        # seed_or_generator : default
        np.random.seed(7)
        actual1 = qpt.generate_empi_dist(0, gate, 10)
        # seed_or_generator : int
        actual2 = qpt.generate_empi_dist(0, gate, 10, seed_or_generator=7)
        # seed_or_generator : np.random.Genrator
        random_gen = np.random.Generator(np.random.MT19937(7))
        actual3 = qpt.generate_empi_dist(0, gate, 10, seed_or_generator=random_gen)

    def test_generate_empi_dists(self):
        qpt, c_sys = get_test_data()
        gate = generate_gate_x(c_sys)

        actual = qpt.generate_empi_dists(gate, 10)
        print(actual)
        expected = [
            (10, np.array([1.0, 0.0], dtype=np.float64)),
            (10, np.array([0.6, 0.4], dtype=np.float64)),
            (10, np.array([0.5, 0.5], dtype=np.float64)),
            (10, np.array([0.6, 0.4], dtype=np.float64)),
            (10, np.array([0.0, 1.0], dtype=np.float64)),
            (10, np.array([0.8, 0.2], dtype=np.float64)),
            (10, np.array([0.5, 0.5], dtype=np.float64)),
            (10, np.array([0.5, 0.5], dtype=np.float64)),
            (10, np.array([0.0, 1.0], dtype=np.float64)),
            (10, np.array([0.3, 0.7], dtype=np.float64)),
            (10, np.array([0.4, 0.6], dtype=np.float64)),
            (10, np.array([1.0, 0.0], dtype=np.float64)),
        ]
        for a, e in zip(actual, expected):
            assert a[0] == e[0]
            npt.assert_almost_equal(a[1], e[1], decimal=15)

    def test_generate_empi_dists_sequence(self):
        qpt, c_sys = get_test_data()
        gate = generate_gate_x(c_sys)

        actual = qpt.generate_empi_dists_sequence(gate, [10, 20])
        print(actual)
        expected = [
            [
                (10, np.array([1.0, 0.0], dtype=np.float64)),
                (10, np.array([0.5, 0.5], dtype=np.float64)),
                (10, np.array([0.8, 0.2], dtype=np.float64)),
                (10, np.array([0.5, 0.5], dtype=np.float64)),
                (10, np.array([0.0, 1.0], dtype=np.float64)),
                (10, np.array([0.4, 0.6], dtype=np.float64)),
                (10, np.array([0.6, 0.4], dtype=np.float64)),
                (10, np.array([0.5, 0.5], dtype=np.float64)),
                (10, np.array([0.0, 1.0], dtype=np.float64)),
                (10, np.array([0.4, 0.6], dtype=np.float64)),
                (10, np.array([0.4, 0.6], dtype=np.float64)),
                (10, np.array([1.0, 0.0], dtype=np.float64)),
            ],
            [
                (20, np.array([1.0, 0.0], dtype=np.float64)),
                (20, np.array([0.55, 0.45], dtype=np.float64)),
                (20, np.array([0.5, 0.5], dtype=np.float64)),
                (20, np.array([0.35, 0.65], dtype=np.float64)),
                (20, np.array([0.0, 1.0], dtype=np.float64)),
                (20, np.array([0.5, 0.5], dtype=np.float64)),
                (20, np.array([0.6, 0.4], dtype=np.float64)),
                (20, np.array([0.35, 0.65], dtype=np.float64)),
                (20, np.array([0.0, 1.0], dtype=np.float64)),
                (20, np.array([0.65, 0.35], dtype=np.float64)),
                (20, np.array([0.5, 0.5], dtype=np.float64)),
                (20, np.array([1.0, 0.0], dtype=np.float64)),
            ],
        ]
        for a_dists, e_dists in zip(actual, expected):
            count = 0
            for a, e in zip(a_dists, e_dists):
                assert a[0] == e[0]
                npt.assert_almost_equal(a[1], e[1], decimal=15)
                count += 1


def test_calc_c_qpt():
    # Arrange
    c_sys = generate_composite_system(mode="qubit", num=1)
    states = [
        generate_qoperation("state", name, c_sys) for name in ["x0", "y0", "z0", "z1"]
    ]
    povms = [generate_qoperation("povm", name, c_sys) for name in ["x", "y", "z"]]

    schedules = []
    for i, j in product(range(len(states)), range(len(povms))):
        schedules.append([("state", i), ("gate", 0), ("povm", j)])

    # Act
    actual_coeffs_0th, actual_coeffs_1st, actual_c_dict = calc_c_qpt(
        states, povms, schedules, on_para_eq_constraint=True
    )

    # Assert
    expected_coeffs_0th = {
        (0, 0): 0.4999999999999998,
        (0, 1): 0.4999999999999998,
        (1, 0): 0.4999999999999998,
        (1, 1): 0.4999999999999998,
        (2, 0): 0.49999999999999983,
        (2, 1): 0.49999999999999983,
        (3, 0): 0.4999999999999998,
        (3, 1): 0.4999999999999998,
        (4, 0): 0.4999999999999998,
        (4, 1): 0.4999999999999998,
        (5, 0): 0.49999999999999983,
        (5, 1): 0.49999999999999983,
        (6, 0): 0.49999999999999983,
        (6, 1): 0.49999999999999983,
        (7, 0): 0.49999999999999983,
        (7, 1): 0.49999999999999983,
        (8, 0): 0.4999999999999999,
        (8, 1): 0.4999999999999999,
        (9, 0): 0.49999999999999983,
        (9, 1): 0.49999999999999983,
        (10, 0): 0.49999999999999983,
        (10, 1): 0.49999999999999983,
        (11, 0): 0.4999999999999999,
        (11, 1): 0.4999999999999999,
    }

    expected_coeffs_1st = {
        (0, 0): np.array([0.5, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
        (0, 1): np.array(
            [-0.5, -0.5, -0.0, -0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        ),
        (1, 0): np.array([0.0, 0.0, 0.0, 0.0, 0.5, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
        (1, 1): np.array(
            [0.0, 0.0, 0.0, 0.0, -0.5, -0.5, -0.0, -0.0, 0.0, 0.0, 0.0, 0.0]
        ),
        (2, 0): np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5, 0.5, 0.0, 0.0]),
        (2, 1): np.array(
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.5, -0.5, -0.0, -0.0]
        ),
        (3, 0): np.array([0.5, 0.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
        (3, 1): np.array(
            [-0.5, -0.0, -0.5, -0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        ),
        (4, 0): np.array([0.0, 0.0, 0.0, 0.0, 0.5, 0.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0]),
        (4, 1): np.array(
            [0.0, 0.0, 0.0, 0.0, -0.5, -0.0, -0.5, -0.0, 0.0, 0.0, 0.0, 0.0]
        ),
        (5, 0): np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5, 0.0, 0.5, 0.0]),
        (5, 1): np.array(
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.5, -0.0, -0.5, -0.0]
        ),
        (6, 0): np.array([0.5, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
        (6, 1): np.array(
            [-0.5, -0.0, -0.0, -0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        ),
        (7, 0): np.array([0.0, 0.0, 0.0, 0.0, 0.5, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 0.0]),
        (7, 1): np.array(
            [0.0, 0.0, 0.0, 0.0, -0.5, -0.0, -0.0, -0.5, 0.0, 0.0, 0.0, 0.0]
        ),
        (8, 0): np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5, 0.0, 0.0, 0.5]),
        (8, 1): np.array(
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.5, -0.0, -0.0, -0.5]
        ),
        (9, 0): np.array(
            [0.5, 0.0, 0.0, -0.5, 0.0, 0.0, 0.0, -0.0, 0.0, 0.0, 0.0, -0.0]
        ),
        (9, 1): np.array(
            [-0.5, -0.0, -0.0, 0.5, 0.0, 0.0, 0.0, -0.0, 0.0, 0.0, 0.0, -0.0]
        ),
        (10, 0): np.array(
            [0.0, 0.0, 0.0, -0.0, 0.5, 0.0, 0.0, -0.5, 0.0, 0.0, 0.0, -0.0]
        ),
        (10, 1): np.array(
            [0.0, 0.0, 0.0, -0.0, -0.5, -0.0, -0.0, 0.5, 0.0, 0.0, 0.0, -0.0]
        ),
        (11, 0): np.array(
            [0.0, 0.0, 0.0, -0.0, 0.0, 0.0, 0.0, -0.0, 0.5, 0.0, 0.0, -0.5]
        ),
        (11, 1): np.array(
            [0.0, 0.0, 0.0, -0.0, 0.0, 0.0, 0.0, -0.0, -0.5, -0.0, -0.0, 0.5]
        ),
    }

    expected_c_dict = {
        0: np.array(
            [
                [
                    0.5,
                    0.5,
                    0.0,
                    0.0,
                    0.5,
                    0.5,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                ],
                [
                    0.5,
                    0.5,
                    0.0,
                    0.0,
                    -0.5,
                    -0.5,
                    -0.0,
                    -0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                ],
            ]
        ),
        1: np.array(
            [
                [
                    0.5,
                    0.5,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.5,
                    0.5,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                ],
                [
                    0.5,
                    0.5,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    -0.5,
                    -0.5,
                    -0.0,
                    -0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                ],
            ]
        ),
        2: np.array(
            [
                [
                    0.5,
                    0.5,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.5,
                    0.5,
                    0.0,
                    0.0,
                ],
                [
                    0.5,
                    0.5,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    -0.5,
                    -0.5,
                    -0.0,
                    -0.0,
                ],
            ]
        ),
        3: np.array(
            [
                [
                    0.5,
                    0.0,
                    0.5,
                    0.0,
                    0.5,
                    0.0,
                    0.5,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                ],
                [
                    0.5,
                    0.0,
                    0.5,
                    0.0,
                    -0.5,
                    -0.0,
                    -0.5,
                    -0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                ],
            ]
        ),
        4: np.array(
            [
                [
                    0.5,
                    0.0,
                    0.5,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.5,
                    0.0,
                    0.5,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                ],
                [
                    0.5,
                    0.0,
                    0.5,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    -0.5,
                    -0.0,
                    -0.5,
                    -0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                ],
            ]
        ),
        5: np.array(
            [
                [
                    0.5,
                    0.0,
                    0.5,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.5,
                    0.0,
                    0.5,
                    0.0,
                ],
                [
                    0.5,
                    0.0,
                    0.5,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    -0.5,
                    -0.0,
                    -0.5,
                    -0.0,
                ],
            ]
        ),
        6: np.array(
            [
                [
                    0.5,
                    0.0,
                    0.0,
                    0.5,
                    0.5,
                    0.0,
                    0.0,
                    0.5,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                ],
                [
                    0.5,
                    0.0,
                    0.0,
                    0.5,
                    -0.5,
                    -0.0,
                    -0.0,
                    -0.5,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                ],
            ]
        ),
        7: np.array(
            [
                [
                    0.5,
                    0.0,
                    0.0,
                    0.5,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.5,
                    0.0,
                    0.0,
                    0.5,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                ],
                [
                    0.5,
                    0.0,
                    0.0,
                    0.5,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    -0.5,
                    -0.0,
                    -0.0,
                    -0.5,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                ],
            ]
        ),
        8: np.array(
            [
                [
                    0.5,
                    0.0,
                    0.0,
                    0.5,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.5,
                    0.0,
                    0.0,
                    0.5,
                ],
                [
                    0.5,
                    0.0,
                    0.0,
                    0.5,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    -0.5,
                    -0.0,
                    -0.0,
                    -0.5,
                ],
            ]
        ),
        9: np.array(
            [
                [
                    0.5,
                    0.0,
                    0.0,
                    -0.5,
                    0.5,
                    0.0,
                    0.0,
                    -0.5,
                    0.0,
                    0.0,
                    0.0,
                    -0.0,
                    0.0,
                    0.0,
                    0.0,
                    -0.0,
                ],
                [
                    0.5,
                    0.0,
                    0.0,
                    -0.5,
                    -0.5,
                    -0.0,
                    -0.0,
                    0.5,
                    0.0,
                    0.0,
                    0.0,
                    -0.0,
                    0.0,
                    0.0,
                    0.0,
                    -0.0,
                ],
            ]
        ),
        10: np.array(
            [
                [
                    0.5,
                    0.0,
                    0.0,
                    -0.5,
                    0.0,
                    0.0,
                    0.0,
                    -0.0,
                    0.5,
                    0.0,
                    0.0,
                    -0.5,
                    0.0,
                    0.0,
                    0.0,
                    -0.0,
                ],
                [
                    0.5,
                    0.0,
                    0.0,
                    -0.5,
                    0.0,
                    0.0,
                    0.0,
                    -0.0,
                    -0.5,
                    -0.0,
                    -0.0,
                    0.5,
                    0.0,
                    0.0,
                    0.0,
                    -0.0,
                ],
            ]
        ),
        11: np.array(
            [
                [
                    0.5,
                    0.0,
                    0.0,
                    -0.5,
                    0.0,
                    0.0,
                    0.0,
                    -0.0,
                    0.0,
                    0.0,
                    0.0,
                    -0.0,
                    0.5,
                    0.0,
                    0.0,
                    -0.5,
                ],
                [
                    0.5,
                    0.0,
                    0.0,
                    -0.5,
                    0.0,
                    0.0,
                    0.0,
                    -0.0,
                    0.0,
                    0.0,
                    0.0,
                    -0.0,
                    -0.5,
                    -0.0,
                    -0.0,
                    0.5,
                ],
            ]
        ),
    }

    assert actual_coeffs_0th == expected_coeffs_0th

    assert actual_coeffs_1st.keys() == expected_coeffs_1st.keys()
    for key in actual_coeffs_1st:
        npt.assert_allclose(
            actual_coeffs_1st[key], expected_coeffs_1st[key], atol=10 ** -15, rtol=0
        )

    assert actual_c_dict.keys() == expected_c_dict.keys()
    for key in actual_c_dict:
        npt.assert_allclose(
            actual_c_dict[key], expected_c_dict[key], atol=10 ** -15, rtol=0
        )
