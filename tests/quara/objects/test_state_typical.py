import numpy as np
import numpy.testing as npt
import pytest

from quara.objects.matrix_basis import get_normalized_pauli_basis
from quara.objects.state_typical import (
    get_state_names_1qubit,
    get_state_names_2qubit,
    generate_state_from_state_name,
)
from quara.objects import matrix_basis
from quara.objects.composite_system import CompositeSystem
from quara.objects.elemental_system import ElementalSystem


def test_get_state_names_1qubit():
    actual = get_state_names_1qubit()
    expected = ["x0", "x1", "y0", "y1", "z0", "z1", "a"]
    assert actual == expected


def get_state_names_2qubit():
    actual = get_state_names_2qubit()
    expected = [
        "bell_phi_plus",
        "bell_phi_minus",
        "bell_psi_plus",
        "bell_psi_minus",
        "z0_z0",
    ]
    assert actual == expected


def test_generate_state_from_state_name_exception():
    e_sys = ElementalSystem(0, matrix_basis.get_normalized_pauli_basis())
    c_sys = CompositeSystem([e_sys])

    with pytest.raises(ValueError):
        _ = generate_state_from_state_name("x", c_sys)
    with pytest.raises(ValueError):
        _ = generate_state_from_state_name("1", c_sys)
    with pytest.raises(ValueError):
        _ = generate_state_from_state_name("x1_", c_sys)


def test_get_x0_1q():
    e_sys = ElementalSystem(0, matrix_basis.get_normalized_pauli_basis())
    c_sys = CompositeSystem([e_sys])
    state = generate_state_from_state_name("x0", c_sys)
    actual = state.to_density_matrix()
    expected = np.array([[0.5, 0.5], [0.5, 0.5]], dtype=np.complex128)
    npt.assert_almost_equal(actual, expected, decimal=15)


def test_get_x1_1q():
    e_sys = ElementalSystem(0, matrix_basis.get_normalized_pauli_basis())
    c_sys = CompositeSystem([e_sys])
    state = generate_state_from_state_name("x1", c_sys)
    actual = state.to_density_matrix()
    expected = np.array([[0.5, -0.5], [-0.5, 0.5]], dtype=np.complex128)
    npt.assert_almost_equal(actual, expected, decimal=15)


def test_get_y0_1q():
    e_sys = ElementalSystem(0, matrix_basis.get_normalized_pauli_basis())
    c_sys = CompositeSystem([e_sys])
    state = generate_state_from_state_name("y0", c_sys)
    actual = state.to_density_matrix()
    expected = np.array([[0.5, -0.5j], [0.5j, 0.5]], dtype=np.complex128)
    npt.assert_almost_equal(actual, expected, decimal=15)


def test_get_y1_1q():
    e_sys = ElementalSystem(0, matrix_basis.get_normalized_pauli_basis())
    c_sys = CompositeSystem([e_sys])
    state = generate_state_from_state_name("y1", c_sys)
    actual = state.to_density_matrix()
    expected = np.array([[0.5, 0.5j], [-0.5j, 0.5]], dtype=np.complex128)
    npt.assert_almost_equal(actual, expected, decimal=15)


def test_get_z0_1q():
    e_sys = ElementalSystem(0, matrix_basis.get_normalized_pauli_basis())
    c_sys = CompositeSystem([e_sys])
    state = generate_state_from_state_name("z0", c_sys)
    actual = state.to_density_matrix()
    expected = np.array([[1, 0], [0, 0]], dtype=np.complex128)
    npt.assert_almost_equal(actual, expected, decimal=15)


def test_get_z1_1q():
    e_sys = ElementalSystem(0, matrix_basis.get_normalized_pauli_basis())
    c_sys = CompositeSystem([e_sys])
    state = generate_state_from_state_name("z1", c_sys)
    actual = state.to_density_matrix()
    expected = np.array([[0, 0], [0, 1]], dtype=np.complex128)
    npt.assert_almost_equal(actual, expected, decimal=15)

