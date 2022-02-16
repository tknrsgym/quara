from multiprocessing.sharedctypes import Value
import numpy as np
import numpy.testing as npt
import pytest

from quara.objects.composite_system_typical import generate_composite_system
from quara.objects.povm_typical import generate_povm_from_name
from quara.protocol.qtomography.standard.preprocessing import (
    StandardQTomographyPreprocessing,
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


def get_test_data_qst(on_para_eq_constraint=True):
    c_sys = generate_composite_system("qubit", 1)

    povm_x = generate_povm_from_name("x", c_sys)
    povm_y = generate_povm_from_name("y", c_sys)
    povm_z = generate_povm_from_name("z", c_sys)
    povms = [povm_x, povm_y, povm_z]

    qst = StandardQst(povms, on_para_eq_constraint=on_para_eq_constraint, seed_data=7)

    return qst, c_sys


class TestStandardQTomographyPreprocessing:
    def test_init(self):
        # Arrange
        sqt, _ = get_test_data_qst()

        # Act
        preprocessing = StandardQTomographyPreprocessing(sqt)

        # Assert
        assert preprocessing.type_estimate == "state"
        assert type(preprocessing.sqt) == StandardQst
        assert preprocessing.prob_dists == None
        assert preprocessing.eps_prob_zero == 10 ** (-12)
        assert preprocessing.nums_data == None
        assert preprocessing.num_data_total == None
        assert preprocessing.num_data_ratios == None


def squared_distance_state(state1: State, state2: State) -> float:
    diff = state1.vec - state2.vec
    res = np.inner(diff, diff)
    return res


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
