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


class TestStandardQpt:
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
