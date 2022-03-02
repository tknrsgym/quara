import numpy as np
import numpy.testing as npt
import pytest

from quara.objects.composite_system_typical import generate_composite_system
import quara.protocol.qtomography.standard.preprocessing as pre
from quara.protocol.qtomography.standard.standard_qst import StandardQst
from quara.protocol.qtomography.standard.standard_povmt import StandardPovmt
from quara.protocol.qtomography.standard.standard_qpt import StandardQpt
from quara.protocol.qtomography.standard.standard_qmpt import StandardQmpt
from quara.objects.qoperation_typical import generate_qoperation


def test_extract_nums_from_empi_dists():
    # Arrange
    empi_dists = [
        (100, np.array([0.1, 0.9])),
        (200, np.array([0.2, 0.8])),
    ]

    # Act
    actual = pre.extract_nums_from_empi_dists(empi_dists)

    # Assert
    expected = [100, 200]
    assert actual == expected


def test_extract_prob_dists_from_empi_dists():
    # Arrange
    empi_dists = [
        (100, np.array([0.1, 0.9])),
        (200, np.array([0.2, 0.8])),
    ]

    # Act
    actual = pre.extract_prob_dists_from_empi_dists(empi_dists)

    # Assert
    expected = [np.array([0.1, 0.9]), np.array([0.2, 0.8])]
    for a, e in zip(actual, expected):
        npt.assert_almost_equal(a, e, decimal=15)


def test_calc_total_num():
    # Arrange
    source = [0.5, 0.3, 0.2]
    # Act
    actual = pre.calc_total_num(source)
    # Assert
    assert actual == 1


def test_calc_num_ratio():
    # Case 1: invalid input
    # Arrange
    source = [-1]
    # Act & Assert
    with pytest.raises(ValueError):
        _ = pre.calc_num_ratios(source)

    # Case 2:
    # Arrange
    source = [1, 1, 0]
    # Act
    actual = pre.calc_num_ratios(source)
    # Assert
    expected = [0.5, 0.5, 0]
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
