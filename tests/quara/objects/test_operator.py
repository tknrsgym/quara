import numpy as np
import numpy.testing as npt
import pytest

from quara.objects import matrix_basis
from quara.objects.composite_system import CompositeSystem
from quara.objects.elemental_system import ElementalSystem
from quara.objects.gate import Gate, get_Z
from quara.objects.operator import composite, tensor_product
from quara.objects.state import (
    State,
    get_X0_1q_with_normalized_pauli_basis,
    get_X1_1q_with_normalized_pauli_basis,
    get_Z0_1q_with_normalized_pauli_basis,
    get_Z1_1q_with_normalized_pauli_basis,
)


def test_tensor_product_MatrixBasis_MatrixBasis():
    # tensor_product of computational basis (multi arguments)
    comp1 = matrix_basis.get_comp_basis()
    comp2 = matrix_basis.get_comp_basis()
    actual = tensor_product(comp1, comp2)
    expected = [
        np.array(
            [
                [1.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j],
                [0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j],
                [0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j],
                [0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j],
            ]
        ),
        np.array(
            [
                [0.0 + 0.0j, 1.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j],
                [0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j],
                [0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j],
                [0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j],
            ]
        ),
        np.array(
            [
                [0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j],
                [1.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j],
                [0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j],
                [0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j],
            ]
        ),
        np.array(
            [
                [0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j],
                [0.0 + 0.0j, 1.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j],
                [0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j],
                [0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j],
            ]
        ),
        np.array(
            [
                [0.0 + 0.0j, 0.0 + 0.0j, 1.0 + 0.0j, 0.0 + 0.0j],
                [0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j],
                [0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j],
                [0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j],
            ]
        ),
        np.array(
            [
                [0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 1.0 + 0.0j],
                [0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j],
                [0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j],
                [0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j],
            ]
        ),
        np.array(
            [
                [0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j],
                [0.0 + 0.0j, 0.0 + 0.0j, 1.0 + 0.0j, 0.0 + 0.0j],
                [0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j],
                [0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j],
            ]
        ),
        np.array(
            [
                [0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j],
                [0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 1.0 + 0.0j],
                [0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j],
                [0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j],
            ]
        ),
        np.array(
            [
                [0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j],
                [0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j],
                [1.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j],
                [0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j],
            ]
        ),
        np.array(
            [
                [0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j],
                [0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j],
                [0.0 + 0.0j, 1.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j],
                [0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j],
            ]
        ),
        np.array(
            [
                [0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j],
                [0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j],
                [0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j],
                [1.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j],
            ]
        ),
        np.array(
            [
                [0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j],
                [0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j],
                [0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j],
                [0.0 + 0.0j, 1.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j],
            ]
        ),
        np.array(
            [
                [0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j],
                [0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j],
                [0.0 + 0.0j, 0.0 + 0.0j, 1.0 + 0.0j, 0.0 + 0.0j],
                [0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j],
            ]
        ),
        np.array(
            [
                [0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j],
                [0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j],
                [0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 1.0 + 0.0j],
                [0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j],
            ]
        ),
        np.array(
            [
                [0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j],
                [0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j],
                [0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j],
                [0.0 + 0.0j, 0.0 + 0.0j, 1.0 + 0.0j, 0.0 + 0.0j],
            ]
        ),
        np.array(
            [
                [0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j],
                [0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j],
                [0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j],
                [0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 1.0 + 0.0j],
            ]
        ),
    ]
    assert np.array_equal(actual, expected)

    # tensor_product of computational basis (list arguments)
    actual = tensor_product([comp1, comp2])
    assert np.array_equal(actual, expected)

    # error case: no arguments
    with pytest.raises(ValueError):
        actual = tensor_product()

    # error case: single arguments
    with pytest.raises(ValueError):
        actual = tensor_product(comp1)


def test_tensor_product_State_State():
    # tensor_product of computational basis and Pauli basis(multi arguments)
    basis1 = matrix_basis.get_comp_basis()
    e_sys1 = ElementalSystem("q1", basis1)
    c_sys1 = CompositeSystem([e_sys1])
    state1 = State(c_sys1, np.array([1, 0, 0, 0], dtype=np.float64))

    basis2 = matrix_basis.get_pauli_basis()
    e_sys2 = ElementalSystem("q2", basis2)
    c_sys2 = CompositeSystem([e_sys2])
    state2 = State(c_sys2, np.array([0, 1, 0, 0], dtype=np.float64))

    # actual
    actual = tensor_product(state1, state2)

    # expected
    expected_vec = np.kron(np.array([1, 0, 0, 0]), np.array([0, 1, 0, 0]),)
    expected_density_matrix = np.kron(
        np.array([[1, 0], [0, 0]], dtype=np.complex128),
        np.array([[0, 1], [1, 0]], dtype=np.complex128),
    )

    assert np.all(actual._vec == expected_vec)
    assert np.all(actual.get_density_matrix() == expected_density_matrix)

    assert e_sys1 is actual._composite_system._elemental_systems[0]
    assert e_sys2 is actual._composite_system._elemental_systems[1]


def test_composite_product_Gate_State():
    e_sys = ElementalSystem("q0", matrix_basis.get_normalized_pauli_basis())
    c_sys = CompositeSystem([e_sys])
    gate = get_Z(c_sys)

    # case: Z gate \circ X_0 state
    state = get_X0_1q_with_normalized_pauli_basis(c_sys)
    actual = composite(gate, state)
    expected = 1 / np.sqrt(2) * np.array([1, -1, 0, 0], dtype=np.float64)
    npt.assert_almost_equal(actual, expected, decimal=15)

    # case: Z gate \circ X_1 state
    state = get_X1_1q_with_normalized_pauli_basis(c_sys)
    actual = composite(gate, state)
    expected = 1 / np.sqrt(2) * np.array([1, 1, 0, 0], dtype=np.float64)
    npt.assert_almost_equal(actual, expected, decimal=15)

    # case: Z gate \circ Z_0 state
    state = get_Z0_1q_with_normalized_pauli_basis(c_sys)
    actual = composite(gate, state)
    expected = 1 / np.sqrt(2) * np.array([1, 0, 0, 1], dtype=np.float64)
    npt.assert_almost_equal(actual, expected, decimal=15)

    # case: Z gate \circ Z_1 state
    state = get_Z1_1q_with_normalized_pauli_basis(c_sys)
    actual = composite(gate, state)
    expected = 1 / np.sqrt(2) * np.array([1, 0, 0, -1], dtype=np.float64)
    npt.assert_almost_equal(actual, expected, decimal=15)
