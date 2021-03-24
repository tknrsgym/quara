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
    with pytest.raises(ValueError):
        _ = st.generate_state_pure_state_vector_from_name("x")
    with pytest.raises(ValueError):
        _ = st.generate_state_pure_state_vector_from_name("1")
    with pytest.raises(ValueError):
        _ = st.generate_state_pure_state_vector_from_name("x1_")


@pytest.mark.parametrize(
    ("state_name"), [(state_name) for state_name in st.get_state_names_1qubit()],
)
def test_get_object_from_name_1qubit(state_name):
    # Arrange
    basis = get_normalized_pauli_basis()
    e_sys = ElementalSystem(0, basis)
    c_sys = CompositeSystem([e_sys])
    method_name = f"st.get_{state_name}_1q"
    method = eval(method_name)
    expected_state = method(c_sys)

    # density matrix
    # Act
    actual = st.generate_state_density_mat_from_name(state_name)
    # Assert
    expected = expected_state.to_density_matrix()
    npt.assert_almost_equal(actual, expected)
    # Act
    actual = st.generate_state_object_from_state_name_object_name(
        state_name=state_name, object_name="density_mat", c_sys=c_sys
    )
    # Assert
    npt.assert_almost_equal(actual, expected)

    # density matrix vec
    # Act
    actual = st.generate_state_density_matrix_vector_from_name(basis, state_name)
    # Assert
    expected = expected_state.vec
    npt.assert_almost_equal(actual, expected)

    # Act
    actual = st.generate_state_object_from_state_name_object_name(
        state_name=state_name, object_name="density_matrix_vector", c_sys=c_sys
    )
    # Assert
    npt.assert_almost_equal(actual, expected)

    # State
    # Act
    actual = st.generate_state_from_name(c_sys, state_name)
    # Assert
    expected = expected_state
    npt.assert_almost_equal(actual.vec, expected.vec)

    # Act
    actual = st.generate_state_object_from_state_name_object_name(
        state_name=state_name, object_name="state", c_sys=c_sys
    )
    # Assert
    npt.assert_almost_equal(actual.vec, expected.vec)


def test_get_state_a_pure_state_vec():
    # Act
    actual = st.get_state_a_pure_state_vec()
    # Assert
    expected = np.array([1 / np.sqrt(2), (1 / 2) * (1 + 1j)])
    npt.assert_almost_equal(actual, expected)
