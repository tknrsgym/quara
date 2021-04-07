import numpy as np
import numpy.testing as npt
import pytest

from quara.objects.matrix_basis import (
    get_normalized_pauli_basis,
    get_normalized_gell_mann_basis,
)
from quara.objects.state import State
from quara.objects import state_typical as st
from quara.objects import matrix_basis
from quara.objects.composite_system import CompositeSystem
from quara.objects.composite_system_typical import generate_composite_system
from quara.objects.elemental_system import ElementalSystem


def test_get_state_names_1qubit():
    actual = st.get_state_names_1qubit()
    expected = ["x0", "x1", "y0", "y1", "z0", "z1", "a"]
    assert actual == expected


def test_generate_state_from_state_name_exception():
    with pytest.raises(ValueError):
        _ = st.generate_state_pure_state_vector_from_name("x")
    with pytest.raises(ValueError):
        _ = st.generate_state_pure_state_vector_from_name("1")
    with pytest.raises(ValueError):
        _ = st.generate_state_pure_state_vector_from_name("x1_")


@pytest.mark.onequbit
@pytest.mark.parametrize(
    ("state_name"),
    [(state_name) for state_name in st.get_state_names_1qubit()],
)
def test_generate_state_from_name_1qubit(state_name):
    # Arrange
    basis = get_normalized_pauli_basis()
    e_sys = ElementalSystem(0, basis)
    c_sys = CompositeSystem([e_sys])
    method_name = f"st.get_state_{state_name}_1q"
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


@pytest.mark.twoqubit
@pytest.mark.parametrize(
    ("state_name"),
    [(state_name) for state_name in st.get_state_names_2qubit()],
)
def test_generate_state_from_name_2qubit(state_name):
    # Arrange
    c_sys = generate_composite_system("qubit", 2)

    # Act
    actual = st.generate_state_from_name(c_sys, state_name)

    # Assert
    # TODO implement various test cases
    assert type(actual) == State


@pytest.mark.threequbit
@pytest.mark.parametrize(
    ("state_name"),
    [(state_name) for state_name in st.get_state_names_3qubit()],
)
def test_generate_state_from_name_3qubit(state_name):
    # Arrange
    c_sys = generate_composite_system("qubit", 3)

    # Act
    actual = st.generate_state_from_name(c_sys, state_name)

    # Assert
    # TODO implement various test cases
    assert type(actual) == State


@pytest.mark.onequtrit
@pytest.mark.parametrize(
    ("state_name"),
    [(state_name) for state_name in st.get_state_names_1qutrit()],
)
def test_generate_state_from_name_1qutrit(state_name):
    # Arrange
    c_sys = generate_composite_system("qutrit", 1)

    # Act
    actual = st.generate_state_from_name(c_sys, state_name)

    # Assert
    # TODO implement various test cases
    assert type(actual) == State


@pytest.mark.twoqutrit
@pytest.mark.parametrize(
    ("state_name"),
    [(state_name) for state_name in st.get_state_names_2qutrit()],
)
def test_generate_state_from_name_2qutrit(state_name):
    # Arrange
    c_sys = generate_composite_system("qutrit", 2)

    # Act
    actual = st.generate_state_from_name(c_sys, state_name)

    # Assert
    # TODO implement various test cases
    assert type(actual) == State


def test_get_state_a_pure_state_vector():
    # Act
    actual = st.get_state_a_pure_state_vector()
    # Assert
    expected = np.array([1 / np.sqrt(2), (1 / 2) * (1 + 1j)])
    npt.assert_almost_equal(actual, expected)


def test_get_state_bell():
    # Arrange
    basis = get_normalized_pauli_basis()
    e_sys_0 = ElementalSystem(0, basis)
    e_sys_1 = ElementalSystem(1, basis)
    c_sys = CompositeSystem([e_sys_0, e_sys_1])
    expected_state = st.get_state_bell_2q(c_sys)
    state_name = "bell_phi_plus"

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
    actual = st.generate_state_density_matrix_vector_from_name(
        c_sys.basis(), state_name
    )
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


def test_generate_state_pure_state_vector_from_name_2q():
    # |0>|1>
    actual = st.generate_state_pure_state_vector_from_name("z0_z1")
    expected = np.array([0, 1, 0, 0])
    npt.assert_almost_equal(actual, expected)

    # |1>|0>
    actual = st.generate_state_pure_state_vector_from_name("z1_z0")
    expected = np.array([0, 0, 1, 0])
    npt.assert_almost_equal(actual, expected)


def test_generate_state_pure_state_vector_from_name_3q():
    # |0>|1>|+>
    actual = st.generate_state_pure_state_vector_from_name("z0_z1_x0")
    expected = np.array([0, 0, 1 / np.sqrt(2), 1 / np.sqrt(2), 0, 0, 0, 0])
    npt.assert_almost_equal(actual, expected)

    # |0>|+>|i>
    actual = st.generate_state_pure_state_vector_from_name("z0_x0_y0")
    expected = np.array(
        [
            1 / 2,
            1j / 2,
            1 / 2,
            1j / 2,
            0,
            0,
            0,
            0,
        ]
    )
    npt.assert_almost_equal(actual, expected)

    actual = st.generate_state_pure_state_vector_from_name("ghz")
    expected = (1 / np.sqrt(2)) * np.array([1, 0, 0, 0, 0, 0, 0, 1])
    npt.assert_almost_equal(actual, expected)

    actual = st.generate_state_pure_state_vector_from_name("werner")
    expected = (1 / np.sqrt(3)) * np.array([0, 1, 1, 0, 1, 0, 0, 0])
    npt.assert_almost_equal(actual, expected)


def test_get_state_bell_pure_state_vector():
    # |Φ+>
    actual = st.get_state_bell_pure_state_vector("bell_phi_plus")
    expected = (1 / np.sqrt(2)) * np.array([1, 0, 0, 1])
    npt.assert_almost_equal(actual, expected)

    # |Φ->
    actual = st.get_state_bell_pure_state_vector("bell_phi_minus")
    expected = (1 / np.sqrt(2)) * np.array([1, 0, 0, -1])
    npt.assert_almost_equal(actual, expected)

    # |Ψ+>
    actual = st.get_state_bell_pure_state_vector("bell_psi_plus")
    expected = (1 / np.sqrt(2)) * np.array([0, 1, 1, 0])
    npt.assert_almost_equal(actual, expected)

    # |Ψ->
    actual = st.get_state_bell_pure_state_vector("bell_psi_minus")
    expected = (1 / np.sqrt(2)) * np.array([0, 1, -1, 0])
    npt.assert_almost_equal(actual, expected)


def test_get_state_ghz_pure_state_vector():
    actual = st.get_state_ghz_pure_state_vector()
    expected = (1 / np.sqrt(2)) * np.array([1, 0, 0, 0, 0, 0, 0, 1])
    npt.assert_almost_equal(actual, expected)


def test_et_state_werner_pure_state_vector():
    actual = st.get_state_werner_pure_state_vector()
    expected = (1 / np.sqrt(3)) * np.array([0, 1, 1, 0, 1, 0, 0, 0])
    npt.assert_almost_equal(actual, expected)


def test_get_state_x0_1q():
    e_sys = ElementalSystem(0, matrix_basis.get_normalized_pauli_basis())
    c_sys = CompositeSystem([e_sys])
    state = st.get_state_x0_1q(c_sys)
    actual = state.to_density_matrix()
    expected = np.array([[0.5, 0.5], [0.5, 0.5]], dtype=np.complex128)
    npt.assert_almost_equal(actual, expected, decimal=15)

    # Test that not 1qubit CompositeSystem
    e_sys0 = ElementalSystem(0, matrix_basis.get_normalized_pauli_basis())
    e_sys1 = ElementalSystem(1, matrix_basis.get_normalized_pauli_basis())
    c_sys = CompositeSystem([e_sys0, e_sys1])
    with pytest.raises(ValueError):
        st.get_state_x0_1q(c_sys)


def test_get_state_x1_1q():
    e_sys = ElementalSystem(0, matrix_basis.get_normalized_pauli_basis())
    c_sys = CompositeSystem([e_sys])
    state = st.get_state_x1_1q(c_sys)
    actual = state.to_density_matrix()
    expected = np.array([[0.5, -0.5], [-0.5, 0.5]], dtype=np.complex128)
    npt.assert_almost_equal(actual, expected, decimal=15)

    # Test that not 1qubit CompositeSystem
    e_sys0 = ElementalSystem(0, matrix_basis.get_normalized_pauli_basis())
    e_sys1 = ElementalSystem(1, matrix_basis.get_normalized_pauli_basis())
    c_sys = CompositeSystem([e_sys0, e_sys1])
    with pytest.raises(ValueError):
        st.get_state_x1_1q(c_sys)


def test_get_state_y0_1q():
    e_sys = ElementalSystem(0, matrix_basis.get_normalized_pauli_basis())
    c_sys = CompositeSystem([e_sys])
    state = st.get_state_y0_1q(c_sys)
    actual = state.to_density_matrix()
    expected = np.array([[0.5, -0.5j], [0.5j, 0.5]], dtype=np.complex128)
    npt.assert_almost_equal(actual, expected, decimal=15)

    # Test that not 1qubit CompositeSystem
    e_sys0 = ElementalSystem(0, matrix_basis.get_normalized_pauli_basis())
    e_sys1 = ElementalSystem(1, matrix_basis.get_normalized_pauli_basis())
    c_sys = CompositeSystem([e_sys0, e_sys1])
    with pytest.raises(ValueError):
        st.get_state_y0_1q(c_sys)


def test_get_state_y1_1q():
    e_sys = ElementalSystem(0, matrix_basis.get_normalized_pauli_basis())
    c_sys = CompositeSystem([e_sys])
    state = st.get_state_y1_1q(c_sys)
    actual = state.to_density_matrix()
    expected = np.array([[0.5, 0.5j], [-0.5j, 0.5]], dtype=np.complex128)
    npt.assert_almost_equal(actual, expected, decimal=15)

    # Test that not 1qubit CompositeSystem
    e_sys0 = ElementalSystem(0, matrix_basis.get_normalized_pauli_basis())
    e_sys1 = ElementalSystem(1, matrix_basis.get_normalized_pauli_basis())
    c_sys = CompositeSystem([e_sys0, e_sys1])
    with pytest.raises(ValueError):
        st.get_state_y1_1q(c_sys)


def test_get_state_z0_1q():
    e_sys = ElementalSystem(0, matrix_basis.get_normalized_pauli_basis())
    c_sys = CompositeSystem([e_sys])
    state = st.get_state_z0_1q(c_sys)
    actual = state.to_density_matrix()
    expected = np.array([[1, 0], [0, 0]], dtype=np.complex128)
    npt.assert_almost_equal(actual, expected, decimal=15)

    # Test that not 1qubit CompositeSystem
    e_sys0 = ElementalSystem(0, matrix_basis.get_normalized_pauli_basis())
    e_sys1 = ElementalSystem(1, matrix_basis.get_normalized_pauli_basis())
    c_sys = CompositeSystem([e_sys0, e_sys1])
    with pytest.raises(ValueError):
        st.get_state_z0_1q(c_sys)


def test_get_state_z1_1q():
    e_sys = ElementalSystem(0, matrix_basis.get_normalized_pauli_basis())
    c_sys = CompositeSystem([e_sys])
    state = st.get_state_z1_1q(c_sys)
    actual = state.to_density_matrix()
    expected = np.array([[0, 0], [0, 1]], dtype=np.complex128)
    npt.assert_almost_equal(actual, expected, decimal=15)

    # Test that not 1qubit CompositeSystem
    e_sys0 = ElementalSystem(0, matrix_basis.get_normalized_pauli_basis())
    e_sys1 = ElementalSystem(1, matrix_basis.get_normalized_pauli_basis())
    c_sys = CompositeSystem([e_sys0, e_sys1])
    with pytest.raises(ValueError):
        st.get_state_z1_1q(c_sys)


def test_get_state_bell_2q():
    expected = (
        np.array(
            [[1, 0, 0, 1], [0, 0, 0, 0], [0, 0, 0, 0], [1, 0, 0, 1]],
            dtype=np.complex128,
        )
        / 2
    )

    # test for Pauli basis
    e_sys0 = ElementalSystem(0, matrix_basis.get_normalized_pauli_basis())
    e_sys1 = ElementalSystem(1, matrix_basis.get_normalized_pauli_basis())
    c_sys = CompositeSystem([e_sys0, e_sys1])
    state = st.get_state_bell_2q(c_sys)
    actual = state.to_density_matrix()
    npt.assert_almost_equal(actual, expected, decimal=15)

    # test for comp basis
    e_sys2 = ElementalSystem(2, matrix_basis.get_comp_basis())
    e_sys3 = ElementalSystem(3, matrix_basis.get_comp_basis())
    c_sys = CompositeSystem([e_sys2, e_sys3])
    state = st.get_state_bell_2q(c_sys)
    actual = state.to_density_matrix()
    npt.assert_almost_equal(actual, expected, decimal=15)

    # Test that not 2qubit CompositeSystem
    e_sys2 = ElementalSystem(2, matrix_basis.get_normalized_pauli_basis())
    c_sys = CompositeSystem([e_sys2])
    with pytest.raises(ValueError):
        st.get_state_bell_2q(c_sys)
