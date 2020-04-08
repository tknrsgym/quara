import numpy as np
import numpy.testing as npt
import pytest

from quara.objects import matrix_basis
from quara.objects.composite_system import CompositeSystem
from quara.objects.elemental_system import ElementalSystem
from quara.objects.gate import Gate, get_h, get_i, get_s, get_x, get_y, get_z
from quara.objects.operators import (
    _composite,
    _tensor_product,
    _to_list,
    composite,
    tensor_product,
)
from quara.objects.povm import Povm
from quara.objects.state import State, get_x0_1q, get_x1_1q, get_z0_1q, get_z1_1q


def test_tensor_product_Gate_Gate():
    # case: HS(Z \otimes Z) on Pauli basis
    # coincidentally HS(Z \otimes Z) = HS(Z) \otimes HS(Z)
    e_sys1 = ElementalSystem(1, matrix_basis.get_normalized_pauli_basis())
    c_sys1 = CompositeSystem([e_sys1])
    e_sys2 = ElementalSystem(2, matrix_basis.get_normalized_pauli_basis())
    c_sys2 = CompositeSystem([e_sys2])

    z1 = get_z(c_sys1)
    z2 = get_z(c_sys2)

    actual = tensor_product(z1, z2)
    expected = np.kron(z1.hs, z2.hs)
    npt.assert_almost_equal(actual.hs, expected, decimal=15)

    # case: HS(g1 \otimes g2) != HS(g1) \otimes HS(g2)
    e_sys1 = ElementalSystem(1, matrix_basis.get_normalized_pauli_basis())
    c_sys1 = CompositeSystem([e_sys1])
    e_sys2 = ElementalSystem(2, matrix_basis.get_normalized_pauli_basis())
    c_sys2 = CompositeSystem([e_sys2])

    hs1 = np.array(
        [[0, 1, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0],], dtype=np.float64
    )
    gate1 = Gate(c_sys1, hs1)

    hs2 = np.array(
        [[1, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0],], dtype=np.float64
    )
    gate2 = Gate(c_sys2, hs2)

    actual = tensor_product(gate1, gate2)

    expected = np.array(
        [
            [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        ]
    )
    npt.assert_almost_equal(actual.hs, expected, decimal=15)


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
    e_sys1 = ElementalSystem(1, basis1)
    c_sys1 = CompositeSystem([e_sys1])
    state1 = State(c_sys1, np.array([1, 0, 0, 0], dtype=np.float64), is_physical=False)

    basis2 = matrix_basis.get_pauli_basis()
    e_sys2 = ElementalSystem(2, basis2)
    c_sys2 = CompositeSystem([e_sys2])
    state2 = State(c_sys2, np.array([0, 1, 0, 0], dtype=np.float64), is_physical=False)

    # actual
    actual = tensor_product(state1, state2)

    # expected
    expected_vec = np.kron(np.array([1, 0, 0, 0]), np.array([0, 1, 0, 0]),)
    expected_density_matrix = np.kron(
        np.array([[1, 0], [0, 0]], dtype=np.complex128),
        np.array([[0, 1], [1, 0]], dtype=np.complex128),
    )

    assert np.all(actual.vec == expected_vec)
    assert np.all(actual.get_density_matrix() == expected_density_matrix)

    assert e_sys1 is actual._composite_system._elemental_systems[0]
    assert e_sys2 is actual._composite_system._elemental_systems[1]


def test_tensor_product_povm_povm():
    # TODO
    pass


def test_tensor_product_unexpected_type():
    # Arrange
    basis1 = matrix_basis.get_comp_basis()
    e_sys1 = ElementalSystem(1, basis1)
    c_sys1 = CompositeSystem([e_sys1])
    state1 = State(c_sys1, np.array([1, 0, 0, 0], dtype=np.float64))

    # Act & Assert
    with pytest.raises(TypeError):
        # TypeError: Unsupported type combination!
        _ = _tensor_product(basis1, state1)

    # Act & Assert
    with pytest.raises(TypeError):
        # TypeError: Unsupported type combination!
        _ = _tensor_product(e_sys1, e_sys1)


def test_composite_error():
    e_sys0 = ElementalSystem(0, matrix_basis.get_normalized_pauli_basis())
    c_sys0 = CompositeSystem([e_sys0])
    i_gate0 = get_i(c_sys0)
    e_sys1 = ElementalSystem(1, matrix_basis.get_normalized_pauli_basis())
    c_sys1 = CompositeSystem([e_sys1])
    i_gate1 = get_i(c_sys1)

    # error case: composite different composite systems
    with pytest.raises(ValueError):
        composite(i_gate0, i_gate1)


def test_composite_unexpected_type():
    # Arrange
    e_sys = ElementalSystem(0, matrix_basis.get_normalized_pauli_basis())
    c_sys = CompositeSystem([e_sys])
    z_gate = get_z(c_sys)
    state = get_x0_1q(c_sys)

    # Act & Assert
    # Gate (x) State is OK, but State (x) Gate is NG
    with pytest.raises(TypeError):
        # TypeError: Unsupported type combination!
        _ = _composite(state, z_gate)


def test_composite_Gate_Gate():
    e_sys = ElementalSystem(0, matrix_basis.get_normalized_pauli_basis())
    c_sys = CompositeSystem([e_sys])
    i_gate = get_i(c_sys)
    x_gate = get_x(c_sys)
    y_gate = get_y(c_sys)
    z_gate = get_z(c_sys)

    # X \circ X = I
    actual = composite(x_gate, x_gate)
    expected = i_gate.hs
    npt.assert_almost_equal(actual.hs, expected, decimal=15)

    # X \circ Y = Z
    actual = composite(x_gate, y_gate)
    expected = z_gate.hs
    npt.assert_almost_equal(actual.hs, expected, decimal=15)

    # X \circ X \circ X = X
    actual = composite(x_gate, x_gate, x_gate)
    expected = x_gate.hs
    npt.assert_almost_equal(actual.hs, expected, decimal=15)


def test_composite_Gate_State():
    e_sys = ElementalSystem(0, matrix_basis.get_normalized_pauli_basis())
    c_sys = CompositeSystem([e_sys])
    z_gate = get_z(c_sys)
    h_gate = get_h(c_sys)
    s_gate = get_s(c_sys)

    # case: Z gate \circ x0 state. Z|+><+|Z^{\dagger} = |-><-|
    state = get_x0_1q(c_sys)
    actual = composite(z_gate, state)
    expected = 1 / np.sqrt(2) * np.array([1, -1, 0, 0], dtype=np.complex128)
    npt.assert_almost_equal(actual.vec, expected, decimal=15)

    # case: Z gate \circ x1 state. Z|-><-|Z^{\dagger} = |+><+|
    state = get_x1_1q(c_sys)
    actual = composite(z_gate, state)
    expected = 1 / np.sqrt(2) * np.array([1, 1, 0, 0], dtype=np.float64)
    npt.assert_almost_equal(actual.vec, expected, decimal=15)

    # case: Z gate \circ z0 state. Z|0><0|Z^{\dagger} = |0><0|
    state = get_z0_1q(c_sys)
    actual = composite(z_gate, state)
    expected = 1 / np.sqrt(2) * np.array([1, 0, 0, 1], dtype=np.float64)
    npt.assert_almost_equal(actual.vec, expected, decimal=15)

    # case: Z gate \circ z1 state. Z|1><1|Z^{\dagger} = |1><1|
    state = get_z1_1q(c_sys)
    actual = composite(z_gate, state)
    expected = 1 / np.sqrt(2) * np.array([1, 0, 0, -1], dtype=np.float64)
    npt.assert_almost_equal(actual.vec, expected, decimal=15)

    # case: S gate \circ S gate \circ z0 state. SS|0><0|S^{\dagger}S^{\dagger} = |0><0|
    # S = root(Z), hense SS = Z
    state = get_z0_1q(c_sys)
    actual = composite(s_gate, s_gate, state)
    expected = 1 / np.sqrt(2) * np.array([1, 0, 0, 1], dtype=np.float64)
    npt.assert_almost_equal(actual.vec, expected, decimal=15)

    # case: H gate \circ z0 state. H|0><0|H^{\dagger} = 1/2(I+X)
    state = get_z0_1q(c_sys)
    actual = composite(h_gate, state)
    expected = 1 / np.sqrt(2) * np.array([1, 1, 0, 0], dtype=np.float64)
    npt.assert_almost_equal(actual.vec, expected, decimal=15)

    # case: H gate \circ z1 state. H|1><1|H^{\dagger} = 1/2(I-X)
    state = get_z1_1q(c_sys)
    actual = composite(h_gate, state)
    expected = 1 / np.sqrt(2) * np.array([1, -1, 0, 0], dtype=np.float64)
    npt.assert_almost_equal(actual.vec, expected, decimal=15)


# @pytest.mark.skip(reasons="It only fails at CircleCI.")
def test_composite_Povm_Gate():
    e_sys = ElementalSystem(0, matrix_basis.get_comp_basis())
    c_sys = CompositeSystem([e_sys])
    vecs = [
        np.array([1, 0, 0, 0], dtype=np.complex128),
        np.array([0, 0, 0, 1], dtype=np.complex128),
    ]
    povm = Povm(c_sys, vecs)

    # composite Z-measurement and X gate
    gate = get_x(c_sys)
    actual = composite(povm, gate)
    expected = [
        np.array([0, 0, 0, 1], dtype=np.complex128),
        np.array([1, 0, 0, 0], dtype=np.complex128),
    ]
    npt.assert_almost_equal(actual.vecs, expected, decimal=15)


def test_composite_Povm_State():
    e_sys = ElementalSystem(0, matrix_basis.get_comp_basis())
    c_sys = CompositeSystem([e_sys])
    vecs = [
        np.array([1, 0, 0, 0], dtype=np.complex128),
        np.array([0, 0, 0, 1], dtype=np.complex128),
    ]
    povm = Povm(c_sys, vecs)

    # measurement z0 by Z-measurement
    state = get_z0_1q(c_sys)
    actual = composite(povm, state)
    expected = [1, 0]
    npt.assert_almost_equal(actual, expected, decimal=15)

    # measurement z1 by Z-measurement
    state = get_z1_1q(c_sys)
    actual = composite(povm, state)
    expected = [0, 1]
    npt.assert_almost_equal(actual, expected, decimal=15)

    # measurement x0 by Z-measurement
    state = get_x0_1q(c_sys)
    actual = composite(povm, state)
    expected = [0.5, 0.5]
    npt.assert_almost_equal(actual, expected, decimal=15)


def test_to_list():
    # Arrange & Act
    actual = _to_list(1, 2, 3)

    # Assert
    expected = [1, 2, 3]
    assert actual == expected

    # Arrange & Act
    actual = _to_list([4, 5, 6])

    # Assert
    expected = [4, 5, 6]
    assert actual == expected


def test_to_list_unexpected_value():
    # Arrange & Act & Assert
    with pytest.raises(ValueError):
        # ValueError: arguments must be at least two! arguments=0)
        _ = _to_list([], [])
