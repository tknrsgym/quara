import numpy as np
import numpy.testing as npt

from quara.objects.composite_system import CompositeSystem
from quara.objects.elemental_system import ElementalSystem
from quara.objects.matrix_basis import get_normalized_pauli_basis
from quara.objects.qoperation_typical import generate_qoperation_depolarized


def test_generate_qoperation_depolarized():
    ### case1: mode="state"
    # Arrange
    e_sys = ElementalSystem(0, get_normalized_pauli_basis())
    c_sys = CompositeSystem([e_sys])

    # Act
    actual = generate_qoperation_depolarized(
        mode="state", name="z0", c_sys=c_sys, error_rate=0.1
    )

    # Assert
    expected = np.array([1 / np.sqrt(2), 0, 0, 0.9 / np.sqrt(2)])
    npt.assert_almost_equal(actual.vec, expected, decimal=15)

    ### case2: mode="gate"
    # Arrange
    e_sys = ElementalSystem(0, get_normalized_pauli_basis())
    c_sys = CompositeSystem([e_sys])

    # Act
    actual = generate_qoperation_depolarized(
        mode="gate", name="x", c_sys=c_sys, error_rate=0.1
    )

    # Assert
    expected = np.array(
        [[1, 0, 0, 0], [0, 0.9, 0, 0], [0, 0, -0.9, 0], [0, 0, 0, -0.9]]
    )
    npt.assert_almost_equal(actual.hs, expected, decimal=15)

    ### case3: mode="mprocess"
    # Arrange
    e_sys = ElementalSystem(0, get_normalized_pauli_basis())
    c_sys = CompositeSystem([e_sys])

    # Act
    actual = generate_qoperation_depolarized(
        mode="mprocess", name="x-type1", c_sys=c_sys, error_rate=0.1
    )

    # Assert
    expected = [
        np.array([[1, 1, 0, 0], [0.9, 0.9, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]) / 2,
        np.array([[1, -1, 0, 0], [-0.9, 0.9, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]) / 2,
    ]
    for a, e in zip(actual.hss, expected):
        npt.assert_almost_equal(a, e, decimal=15)

    ### case4: mode="povm"
    # Arrange
    e_sys = ElementalSystem(0, get_normalized_pauli_basis())
    c_sys = CompositeSystem([e_sys])

    # Act
    actual = generate_qoperation_depolarized(
        mode="povm", name="z", c_sys=c_sys, error_rate=0.1
    )

    # Assert
    expected = [
        np.array([1 / np.sqrt(2), 0, 0, 0.9 / np.sqrt(2)]),
        np.array([1 / np.sqrt(2), 0, 0, -0.9 / np.sqrt(2)]),
    ]
    for a, e in zip(actual.vecs, expected):
        npt.assert_almost_equal(a, e, decimal=15)


def test_generate_qoperation_depolarized_is_physicality_required():
    # Arrange
    e_sys = ElementalSystem(0, get_normalized_pauli_basis())
    c_sys = CompositeSystem([e_sys])

    # Case 1:
    # Act
    actual = generate_qoperation_depolarized(
        mode="state",
        name="z0",
        c_sys=c_sys,
        error_rate=0.1,
        is_physicality_required=True,
    )

    # Assert
    assert actual.is_physicality_required is True

    # Case 2:
    # Act
    actual = generate_qoperation_depolarized(
        mode="state",
        name="z0",
        c_sys=c_sys,
        error_rate=0.1,
        is_physicality_required=False,
    )

    # Assert
    assert actual.is_physicality_required is False

    # Case 3:
    # Act
    actual = generate_qoperation_depolarized(
        mode="state", name="z0", c_sys=c_sys, error_rate=0.1
    )

    # Assert
    assert actual.is_physicality_required is True
