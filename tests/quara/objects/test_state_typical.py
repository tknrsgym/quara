import numpy as np
import numpy.testing as npt
import pytest

from quara.objects.matrix_basis import get_normalized_pauli_basis
from quara.objects import state_typical as st
from quara.objects import matrix_basis
from quara.objects.composite_system import CompositeSystem
from quara.objects.elemental_system import ElementalSystem


def test_get_state_names_1qubit():
    actual = st.get_state_names_1qubit()
    expected = ["x0", "x1", "y0", "y1", "z0", "z1", "a"]
    assert actual == expected


def get_state_names_2qubit():
    actual = st.get_state_names_2qubit()
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
        _ = st.generate_state_pure_state_vector_from_name("x")
    with pytest.raises(ValueError):
        _ = st.generate_state_pure_state_vector_from_name("1")
    with pytest.raises(ValueError):
        _ = st.generate_state_pure_state_vector_from_name("x1_")


# TODO: parametrizeを使って汎用的にする
def test_get_x0():
    # Arrange
    basis = get_normalized_pauli_basis()
    e_sys = ElementalSystem(0, basis)
    c_sys = CompositeSystem([e_sys])

    # density matrix
    actual = st.generate_state_density_mat_from_name("z0")
    expected = st.get_z0_1q(c_sys).to_density_matrix()
    npt.assert_almost_equal(actual, expected)

    actual = st.generate_state_object_from_state_name_object_name(
        state_name="z0", object_name="density_mat", c_sys=c_sys
    )
    npt.assert_almost_equal(actual, expected)

    # density matrix vec
    actual = st.generate_state_density_matrix_vector_from_name(basis, "z0")
    expected = st.get_z0_1q(c_sys).vec
    npt.assert_almost_equal(actual, expected)

    actual = st.generate_state_object_from_state_name_object_name(
        state_name="z0", object_name="density_matrix_vector", c_sys=c_sys
    )
    npt.assert_almost_equal(actual, expected)

    # State
    actual = st.generate_state_from_name(c_sys, "z0")
    expected = st.get_z0_1q(c_sys)
    npt.assert_almost_equal(actual.vec, expected.vec)

    actual = st.generate_state_object_from_state_name_object_name(
        state_name="z0", object_name="state", c_sys=c_sys
    )
    npt.assert_almost_equal(actual.vec, expected.vec)

