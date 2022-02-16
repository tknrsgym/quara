from multiprocessing.sharedctypes import Value
import numpy as np
import numpy.testing as npt
import pytest

from quara.objects.composite_system_typical import generate_composite_system
from quara.objects.state_typical import generate_state_from_name
from quara.objects.povm_typical import generate_povm_from_name
from quara.protocol.qtomography.standard.preprocessing import (
    extract_nums_from_empi_dists,
)
import quara.protocol.qtomography.standard.preprocessing as pre
from quara.protocol.qtomography.standard.standard_qst import StandardQst
from quara.protocol.qtomography.standard.standard_povmt import StandardPovmt
from quara.protocol.qtomography.standard.standard_qpt import StandardQpt
from quara.protocol.qtomography.standard.standard_qmpt import StandardQmpt
from quara.objects.state import State
from quara.objects.povm import Povm
from quara.objects.gate import Gate
from quara.objects.qoperation_typical import generate_qoperation
from quara.protocol.qtomography.standard.standard_qtomography import StandardQTomography


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


def test_type_standard_qtomography():
    # Case 1:
    # Arrange
    c_sys = generate_composite_system(mode="qubit", num=1)
    povms = [generate_qoperation("povm", name, c_sys=c_sys) for name in "xyz"]
    source_sqt = StandardQst(povms)

    # Act
    actual = pre.type_standard_qtomography(source_sqt)

    # Assert
    expected = "state"
    assert actual == expected

    # Case 2:
    # Arrange
    c_sys = generate_composite_system(mode="qubit", num=1)
    states = [
        generate_qoperation("state", name, c_sys=c_sys) for name in ["x0", "x1", "y0"]
    ]
    source_sqt = StandardPovmt(states, num_outcomes=2)

    # Act
    actual = pre.type_standard_qtomography(source_sqt)

    # Assert
    expected = "povm"
    assert actual == expected

    # Case 3:
    # Arrange
    c_sys = generate_composite_system(mode="qubit", num=1)
    states = [
        generate_qoperation("state", name, c_sys=c_sys) for name in ["x0", "x1", "y0"]
    ]
    povms = [generate_qoperation("povm", name, c_sys=c_sys) for name in "xyz"]
    source_sqt = StandardQpt(states, povms)

    # Act
    actual = pre.type_standard_qtomography(source_sqt)

    # Assert
    expected = "gate"
    assert actual == expected

    # Case 4:
    # Arrange
    c_sys = generate_composite_system(mode="qubit", num=1)
    states = [
        generate_qoperation("state", name, c_sys=c_sys) for name in ["x0", "x1", "y0"]
    ]
    povms = [generate_qoperation("povm", name, c_sys=c_sys) for name in "xyz"]
    source_sqt = StandardQmpt(states, povms, num_outcomes=2)

    # Act
    actual = pre.type_standard_qtomography(source_sqt)

    # Assert
    expected = "mprocess"
    assert actual == expected

    # Case 5:
    invalid_source = ["dummy"]

    # Act & Assert
    with pytest.raises(TypeError):
        _ = pre.type_standard_qtomography(invalid_source)


def test_squared_distance_state():
    # 1qubit
    # Arrange
    c_sys = generate_composite_system(mode="qubit", num=1)
    state1 = generate_qoperation(mode="state", name="x0", c_sys=c_sys)
    state2 = generate_qoperation(mode="state", name="x1", c_sys=c_sys)
    # Act
    actual = pre.squared_distance_state(state1, state2)
    # Assert
    expected = 1.9999999999999991
    npt.assert_allclose(actual, expected)

    # 2qubit
    # Arrange
    c_sys = generate_composite_system(mode="qubit", num=2)
    state1 = generate_qoperation(mode="state", name="x0_y0", c_sys=c_sys)
    state2 = generate_qoperation(mode="state", name="bell_phi_plus", c_sys=c_sys)
    # Act
    actual = pre.squared_distance_state(state1, state2)
    # Assert
    expected = 1.4999999999999982
    npt.assert_allclose(actual, expected)


def test_squared_distance_povm():
    # 1qubit
    # Arrange
    c_sys = generate_composite_system(mode="qubit", num=1)
    povm1 = generate_qoperation(mode="povm", name="x", c_sys=c_sys)
    povm2 = generate_qoperation(mode="povm", name="y", c_sys=c_sys)
    # Act
    actual = pre.squared_distance_povm(povm1, povm2)
    # Assert
    expected = 0.9999999999999996
    npt.assert_allclose(actual, expected)

    # 2qubit
    # Arrange
    c_sys = generate_composite_system(mode="qubit", num=2)
    povm1 = generate_qoperation(mode="povm", name="x_x", c_sys=c_sys)
    povm2 = generate_qoperation(mode="povm", name="bell", c_sys=c_sys)
    # Act
    actual = pre.squared_distance_povm(povm1, povm2)
    # Assert
    expected = 1.4999999999999982
    npt.assert_allclose(actual, expected)

    # Exception
    c_sys_1 = generate_composite_system(mode="qubit", num=1)
    c_sys_2 = generate_composite_system(mode="qubit", num=2)
    povm1 = generate_qoperation(mode="povm", name="x", c_sys=c_sys_1)
    povm2 = generate_qoperation(mode="povm", name="bell", c_sys=c_sys_2)

    with pytest.raises(ValueError):
        _ = pre.squared_distance_povm(povm1, povm2)
