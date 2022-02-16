import numpy as np
import numpy.testing as npt
import pytest

from quara.objects.composite_system_typical import generate_composite_system
from quara.objects.state_typical import generate_state_from_name
from quara.objects.povm_typical import generate_povm_from_name
from quara.protocol.qtomography.standard.preprocessing import (
    extract_nums_from_empi_dists,
)
from quara.protocol.qtomography.standard.standard_qst import StandardQst
from quara.protocol.qtomography.standard.standard_povmt import StandardPovmt
from quara.protocol.qtomography.standard.standard_qpt import StandardQpt
from quara.protocol.qtomography.standard.standard_qmpt import StandardQmpt


def get_test_data_qst(on_para_eq_constraint=True):
    c_sys = generate_composite_system("qubit", 1)

    povm_x = generate_povm_from_name("x", c_sys)
    povm_y = generate_povm_from_name("y", c_sys)
    povm_z = generate_povm_from_name("z", c_sys)
    povms = [povm_x, povm_y, povm_z]

    qst = StandardQst(povms, on_para_eq_constraint=on_para_eq_constraint, seed_data=7)

    return qst, c_sys


def get_test_data_povmt(on_para_eq_constraint=True):
    c_sys = generate_composite_system("qubit", 1)

    # |+><+|
    state_x0 = generate_state_from_name(c_sys, "x0")
    # |+i><+i|
    state_y0 = generate_state_from_name(c_sys, "y0")
    # |0><0|
    state_z0 = generate_state_from_name(c_sys, "z0")
    # |1><1|
    state_z1 = generate_state_from_name(c_sys, "z1")
    tester_objects = [state_x0, state_y0, state_z0, state_z1]

    povmt = StandardPovmt(
        tester_objects,
        on_para_eq_constraint=on_para_eq_constraint,
        num_outcomes=2,
        seed_data=7,
    )

    return povmt, c_sys


def test_extract_nums_from_empi_dists():
    # Arrange
    empi_dists = [
        (100, np.array([0.1, 0.9])),
        (200, np.array([0.2, 0.9])),
    ]

    # Act
    actual = extract_nums_from_empi_dists(empi_dists)

    # Assert
    expected = [100, 200]
    assert actual == expected
