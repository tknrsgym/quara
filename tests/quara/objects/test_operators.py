import itertools

import numpy as np
import numpy.testing as npt
import pytest

from quara.objects import matrix_basis
from quara.objects.composite_system import CompositeSystem
from quara.objects.elemental_system import ElementalSystem
from quara.objects.gate import (
    Gate,
    get_cnot,
    get_cz,
    get_h,
    get_i,
    get_root_x,
    get_root_y,
    get_s,
    get_sdg,
    get_swap,
    get_t,
    get_x,
    get_y,
    get_z,
)
from quara.objects.gate_typical import generate_gate_from_gate_name
from quara.objects.mprocess_typical import generate_mprocess_from_name
from quara.objects.operators import (
    _compose_qoperations,
    _tensor_product,
    _to_list,
    compose_qoperations,
    tensor_product,
)
from quara.objects.povm import (
    Povm,
    get_x_povm,
    get_xx_povm,
    get_xy_povm,
    get_xz_povm,
    get_y_povm,
    get_yx_povm,
    get_yy_povm,
    get_yz_povm,
    get_z_povm,
    get_zx_povm,
    get_zy_povm,
    get_zz_povm,
)
from quara.objects.povm_typical import generate_povm_from_name
from quara.objects.state import (
    State,
    get_x0_1q,
    get_x1_1q,
    get_y0_1q,
    get_z0_1q,
    get_z1_1q,
)
from quara.objects.state_ensemble import StateEnsemble
from quara.objects.state_ensemble_typical import generate_state_ensemble_from_name
from quara.objects.state_typical import generate_state_from_name
from quara.objects.qoperation_typical import generate_qoperation_object
from quara.objects.composite_system_typical import generate_composite_system
from quara.objects.multinomial_distribution import MultinomialDistribution
from quara.utils import matrix_util as mutil


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
        [
            [0, 1, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
        ],
        dtype=np.float64,
    )
    gate1 = Gate(c_sys1, hs1, is_physicality_required=False)

    hs2 = np.array(
        [
            [1, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
        ],
        dtype=np.float64,
    )
    gate2 = Gate(c_sys2, hs2, is_physicality_required=False)

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

    # assert associativity
    # (A \otimes B) \otimes C = A \otimes (B \otimes C)
    basis3 = matrix_basis.get_normalized_pauli_basis()
    e_sys3 = ElementalSystem(3, basis3)
    c_sys3 = CompositeSystem([e_sys3])

    basis4 = matrix_basis.get_normalized_pauli_basis()
    e_sys4 = ElementalSystem(4, basis4)
    c_sys4 = CompositeSystem([e_sys4])

    basis5 = matrix_basis.get_normalized_pauli_basis()
    e_sys5 = ElementalSystem(5, basis5)
    c_sys5 = CompositeSystem([e_sys5])

    x = get_x(c_sys3)
    z = get_z(c_sys4)
    h = get_h(c_sys5)

    gate_XZ_H = tensor_product(tensor_product(x, z), h)
    gate_X_ZH = tensor_product(x, tensor_product(z, h))

    npt.assert_almost_equal(gate_XZ_H.hs, gate_X_ZH.hs, decimal=15)


def test_tensor_product_Gate_Gate_sort_ElementalSystem():
    e_sys1 = ElementalSystem(1, matrix_basis.get_normalized_pauli_basis())
    c_sys1 = CompositeSystem([e_sys1])
    e_sys2 = ElementalSystem(2, matrix_basis.get_normalized_pauli_basis())
    c_sys2 = CompositeSystem([e_sys2])
    e_sys3 = ElementalSystem(3, matrix_basis.get_normalized_pauli_basis())
    c_sys3 = CompositeSystem([e_sys3])

    x = get_x(c_sys1)
    y = get_y(c_sys2)
    z = get_z(c_sys3)

    xy = tensor_product(x, y)
    xy_z = tensor_product(xy, z)

    xz = tensor_product(x, z)
    xz_y = tensor_product(xz, y)
    npt.assert_almost_equal(xy_z.hs, xz_y.hs, decimal=15)

    zx = tensor_product(z, x)
    zx_y = tensor_product(zx, y)
    npt.assert_almost_equal(xy_z.hs, zx_y.hs, decimal=15)


def test_tensor_product_Gate_MProcess():
    # case: HS(Z \otimes Z) on Pauli basis
    # coincidentally HS(Z \otimes Z) = HS(Z) \otimes HS(Z)
    e_sys1 = ElementalSystem(1, matrix_basis.get_normalized_pauli_basis())
    c_sys1 = CompositeSystem([e_sys1])
    e_sys2 = ElementalSystem(2, matrix_basis.get_normalized_pauli_basis())
    c_sys2 = CompositeSystem([e_sys2])

    z1 = generate_gate_from_gate_name("z", c_sys1)
    z2 = generate_mprocess_from_name(c_sys2, "z-type1")

    actual = tensor_product(z1, z2)

    expected = [
        np.kron(z1.hs, z2.hss[0]),
        np.kron(z1.hs, z2.hss[1]),
    ]
    for a, e in zip(actual.hss, expected):
        npt.assert_almost_equal(a, e, decimal=15)
    assert actual.shape == (2,)


def test_tensor_product_MProcess_Gate():
    # case: HS(Z \otimes Z) on Pauli basis
    # coincidentally HS(Z \otimes Z) = HS(Z) \otimes HS(Z)
    e_sys1 = ElementalSystem(1, matrix_basis.get_normalized_pauli_basis())
    c_sys1 = CompositeSystem([e_sys1])
    e_sys2 = ElementalSystem(2, matrix_basis.get_normalized_pauli_basis())
    c_sys2 = CompositeSystem([e_sys2])

    z1 = generate_mprocess_from_name(c_sys1, "z-type1")
    z2 = generate_gate_from_gate_name("z", c_sys2)

    actual = tensor_product(z1, z2)

    expected = [
        np.kron(z1.hss[0], z2.hs),
        np.kron(z1.hss[1], z2.hs),
    ]
    for a, e in zip(actual.hss, expected):
        npt.assert_almost_equal(a, e, decimal=15)
    assert actual.shape == (2,)


def test_tensor_product_MProcess_MProcess():
    # case: HS(Z \otimes Z) on Pauli basis
    # coincidentally HS(Z \otimes Z) = HS(Z) \otimes HS(Z)
    e_sys1 = ElementalSystem(1, matrix_basis.get_normalized_pauli_basis())
    c_sys1 = CompositeSystem([e_sys1])
    e_sys2 = ElementalSystem(2, matrix_basis.get_normalized_pauli_basis())
    c_sys2 = CompositeSystem([e_sys2])

    z1 = generate_mprocess_from_name(c_sys1, "z-type1")
    z2 = generate_mprocess_from_name(c_sys2, "z-type1")

    actual = tensor_product(z1, z2)

    expected = [
        np.kron(z1.hss[0], z2.hss[0]),
        np.kron(z1.hss[1], z2.hss[0]),
        np.kron(z1.hss[0], z2.hss[1]),
        np.kron(z1.hss[1], z2.hss[1]),
    ]
    for a, e in zip(actual.hss, expected):
        npt.assert_almost_equal(a, e, decimal=15)
    assert actual.shape == (2, 2)


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
    for a, e in zip(actual, expected):
        assert mutil.allclose(a, e)

    # tensor_product of computational basis (list arguments)
    actual = tensor_product([comp1, comp2])
    for a, e in zip(actual, expected):
        assert mutil.allclose(a, e)

    # error case: no arguments
    with pytest.raises(ValueError):
        actual = tensor_product()

    # error case: single arguments
    with pytest.raises(ValueError):
        actual = tensor_product(comp1)

    # assert associativity
    # (A \otimes B) \otimes C = A \otimes (B \otimes C)
    comp3 = matrix_basis.get_comp_basis()
    comp4 = matrix_basis.get_normalized_pauli_basis()
    comp5 = matrix_basis.get_pauli_basis()

    basis34_5 = tensor_product(tensor_product(comp3, comp4), comp5)
    basis3_45 = tensor_product(comp3, tensor_product(comp4, comp5))
    for mat34_5, mat3_45 in zip(basis34_5.basis, basis3_45.basis):
        assert mutil.allclose(mat34_5, mat3_45)


def test_tensor_product_State_State():
    # tensor_product of computational basis and Pauli basis(multi arguments)
    basis1 = matrix_basis.get_comp_basis()
    e_sys1 = ElementalSystem(1, basis1)
    c_sys1 = CompositeSystem([e_sys1])
    state1 = State(
        c_sys1, np.array([0, 1, 0, 0], dtype=np.float64), is_physicality_required=False
    )

    basis2 = matrix_basis.get_comp_basis()
    e_sys2 = ElementalSystem(2, basis2)
    c_sys2 = CompositeSystem([e_sys2])
    state2 = State(
        c_sys2, np.array([1, 0, 0, 0], dtype=np.float64), is_physicality_required=False
    )

    # actual
    # actual = tensor_product(state1, state2)
    actual = tensor_product(state2, state1)

    # expected
    expected_vec = np.kron(
        np.array([0, 1, 0, 0]),
        np.array([1, 0, 0, 0]),
    )
    expected_density_matrix = np.kron(
        np.array([[0, 1], [0, 0]], dtype=np.complex128),
        np.array([[1, 0], [0, 0]], dtype=np.complex128),
    )
    assert np.all(actual.vec == expected_vec)
    assert np.all(actual.to_density_matrix() == expected_density_matrix)

    assert e_sys1 is actual.composite_system._elemental_systems[0]
    assert e_sys2 is actual.composite_system._elemental_systems[1]

    # assert associativity
    # (A \otimes B) \otimes C = A \otimes (B \otimes C)
    basis3 = matrix_basis.get_normalized_pauli_basis()
    e_sys3 = ElementalSystem(3, basis3)
    c_sys3 = CompositeSystem([e_sys3])

    basis4 = matrix_basis.get_normalized_pauli_basis()
    e_sys4 = ElementalSystem(4, basis4)
    c_sys4 = CompositeSystem([e_sys4])

    basis5 = matrix_basis.get_normalized_pauli_basis()
    e_sys5 = ElementalSystem(5, basis5)
    c_sys5 = CompositeSystem([e_sys5])

    state0 = get_z0_1q(c_sys3)
    state1 = get_z1_1q(c_sys4)
    stateP = get_x0_1q(c_sys5)

    state01_P = tensor_product(tensor_product(state0, state1), stateP)
    state0_1P = tensor_product(state0, tensor_product(state1, stateP))

    assert np.all(state01_P.to_density_matrix() == state0_1P.to_density_matrix())


def test_tensor_product_State_State_sort_ElementalSystem():
    basis1 = matrix_basis.get_comp_basis()
    e_sys1 = ElementalSystem(1, basis1)
    c_sys1 = CompositeSystem([e_sys1])
    state1 = State(
        c_sys1, np.array([2, 3, 5, 7], dtype=np.float64), is_physicality_required=False
    )

    basis2 = matrix_basis.get_comp_basis()
    e_sys2 = ElementalSystem(2, basis2)
    c_sys2 = CompositeSystem([e_sys2])
    state2 = State(
        c_sys2,
        np.array([11, 13, 17, 19], dtype=np.float64),
        is_physicality_required=False,
    )

    basis3 = matrix_basis.get_comp_basis()
    e_sys3 = ElementalSystem(3, basis3)
    c_sys3 = CompositeSystem([e_sys3])
    state3 = State(
        c_sys3,
        np.array([23, 29, 31, 37], dtype=np.float64),
        is_physicality_required=False,
    )

    state12 = tensor_product(state1, state2)
    expected12 = np.kron(state1.vec, state2.vec)
    assert np.all(state12.vec == expected12)
    assert state12.composite_system.elemental_systems[0] == e_sys1
    assert state12.composite_system.elemental_systems[1] == e_sys2

    state12_3 = tensor_product(state12, state3)
    expected12_3 = np.kron(np.kron(state1.vec, state2.vec), state3.vec)
    assert np.all(state12_3.vec == expected12_3)
    assert state12_3.composite_system.elemental_systems[0] == e_sys1
    assert state12_3.composite_system.elemental_systems[1] == e_sys2
    assert state12_3.composite_system.elemental_systems[2] == e_sys3

    state13 = tensor_product(state1, state3)
    expected13 = np.kron(state1.vec, state3.vec)
    assert np.all(state13.vec == expected13)
    assert state13.composite_system.elemental_systems[0] == e_sys1
    assert state13.composite_system.elemental_systems[1] == e_sys3

    state13_2 = tensor_product(state13, state2)
    assert np.all(state13_2.vec == expected12_3)
    assert state13_2.composite_system.elemental_systems[0] == e_sys1
    assert state13_2.composite_system.elemental_systems[1] == e_sys2
    assert state13_2.composite_system.elemental_systems[2] == e_sys3

    state3_12 = tensor_product(state3, state12)
    assert np.all(state3_12.vec == expected12_3)
    assert state3_12.composite_system.elemental_systems[0] == e_sys1
    assert state3_12.composite_system.elemental_systems[1] == e_sys2
    assert state3_12.composite_system.elemental_systems[2] == e_sys3


def test_tensor_product_povm_povm_is_physicality_required_true():
    # Arrange
    # Physical POVM
    physical_povm_list = []
    for i in range(2):
        e_sys = ElementalSystem(i, matrix_basis.get_normalized_pauli_basis())
        c_sys = CompositeSystem([e_sys])
        ps_1 = 1 / np.sqrt(2) * np.array([1, 0, 0, 1], dtype=np.float64)
        ps_2 = 1 / np.sqrt(2) * np.array([1, 0, 0, -1], dtype=np.float64)
        vecs = [ps_1, ps_2]

        povm = Povm(c_sys=c_sys, vecs=vecs)
        physical_povm_list.append(povm)

    # Not Physical POVM
    not_physical_povm_list = []
    for i in range(2, 4):
        ps = 1 / np.sqrt(2) * np.array([1, 0, 0, 1], dtype=np.float64)
        not_ps = 1 / np.sqrt(2) * np.array([1, 0, 1, 0], dtype=np.float64)
        vecs = [ps, not_ps]

        e_sys = ElementalSystem(i, matrix_basis.get_pauli_basis())
        c_sys = CompositeSystem([e_sys])
        povm = Povm(c_sys=c_sys, vecs=vecs, is_physicality_required=False)
        not_physical_povm_list.append(povm)

    povm_1, povm_2 = physical_povm_list
    not_povm_1, not_povm_2 = not_physical_povm_list

    # Act
    actual_povm = tensor_product(povm_1, not_povm_1)

    # Assert
    expected = False
    assert actual_povm.is_physicality_required is expected

    # Act
    actual_povm = tensor_product(not_povm_1, povm_1)

    # Assert
    expected = False
    assert actual_povm.is_physicality_required is expected

    # Act
    actual_povm = tensor_product(not_povm_1, not_povm_2)

    # Assert
    expected = False
    assert actual_povm.is_physicality_required is expected

    # Act
    actual_povm = tensor_product(povm_1, povm_2)

    # Assert
    expected = True
    assert actual_povm.is_physicality_required is expected


def test_tensor_product_Povm_Povm_sort_ElementalSystem():
    # Arange
    basis1 = matrix_basis.get_comp_basis()
    e_sys1 = ElementalSystem(1, basis1)
    c_sys1 = CompositeSystem([e_sys1])
    vecs1 = [
        np.array([2, 3, 5, 7], dtype=np.float64),
        np.array([11, 13, 17, 19], dtype=np.float64),
    ]
    povm1 = Povm(c_sys1, vecs1, is_physicality_required=False)

    basis2 = matrix_basis.get_comp_basis()
    e_sys2 = ElementalSystem(2, basis2)
    c_sys2 = CompositeSystem([e_sys2])
    vecs2 = [
        np.array([23, 29, 31, 37], dtype=np.float64),
        np.array([41, 43, 47, 53], dtype=np.float64),
    ]
    povm2 = Povm(c_sys2, vecs2, is_physicality_required=False)

    basis3 = matrix_basis.get_comp_basis()
    e_sys3 = ElementalSystem(3, basis3)
    c_sys3 = CompositeSystem([e_sys3])
    vecs3 = [
        np.array([59, 61, 67, 71], dtype=np.float64),
        np.array([73, 79, 83, 89], dtype=np.float64),
    ]
    povm3 = Povm(c_sys3, vecs3, is_physicality_required=False)

    # Case 1
    # Act
    povm12 = tensor_product(povm1, povm2)

    # Assert
    expected12 = [np.kron(vec1, vec2) for vec1, vec2 in itertools.product(vecs1, vecs2)]
    assert len(povm12.vecs) == len(expected12)
    for actual, expected in zip(povm12, expected12):
        assert np.all(actual == expected)
    assert povm12.composite_system.elemental_systems[0] == e_sys1
    assert povm12.composite_system.elemental_systems[1] == e_sys2

    # Case 2
    # Act
    povm12_3 = tensor_product(povm12, povm3)

    # Assert
    expected12_3 = [
        np.kron(vec12, vec3) for vec12, vec3 in itertools.product(expected12, vecs3)
    ]

    assert len(povm12_3.vecs) == len(expected12_3)
    for actual, expected in zip(povm12_3, expected12_3):
        assert np.all(actual == expected)

    assert povm12_3.composite_system.elemental_systems[0] == e_sys1
    assert povm12_3.composite_system.elemental_systems[1] == e_sys2
    assert povm12_3.composite_system.elemental_systems[2] == e_sys3

    # Case 3
    # Act
    povm13 = tensor_product(povm1, povm3)
    # Assert
    expected13 = [np.kron(vec1, vec3) for vec1, vec3 in itertools.product(vecs1, vecs3)]
    assert len(povm13.vecs) == len(expected13)
    for actual, expected in zip(povm13, expected13):
        assert np.all(actual == expected)
    assert povm13.composite_system.elemental_systems[0] == e_sys1
    assert povm13.composite_system.elemental_systems[1] == e_sys3

    # Case4 povm12_3 == povm1_23
    povm12_3 = tensor_product(povm12, povm3)
    povm23 = tensor_product(povm2, povm3)
    povm1_23 = tensor_product(povm1, povm23)

    assert len(povm12_3.vecs) == len(povm1_23.vecs)
    for a, b in zip(povm12_3.vecs, povm1_23.vecs):
        assert np.all(a == b)

    assert (
        povm12_3.composite_system.elemental_systems[0]
        == povm1_23.composite_system.elemental_systems[0]
    )
    assert (
        povm12_3.composite_system.elemental_systems[1]
        == povm1_23.composite_system.elemental_systems[1]
    )
    assert (
        povm12_3.composite_system.elemental_systems[2]
        == povm1_23.composite_system.elemental_systems[2]
    )

    # Case 5 povm13_2 == povm12_3
    # Act
    povm13_2 = tensor_product(povm13, povm2)

    # Assert
    assert len(povm13_2.vecs) == len(expected12_3)
    for actual, expected in zip(povm13_2, expected12_3):
        # assert np.all(actual == expected)
        npt.assert_almost_equal(actual, expected, decimal=15)
    assert povm13_2.composite_system.elemental_systems[0] == e_sys1
    assert povm13_2.composite_system.elemental_systems[1] == e_sys2
    assert povm13_2.composite_system.elemental_systems[2] == e_sys3


def test_tensor_product_Povm_Povm_3vecs():
    # Arange
    basis1 = matrix_basis.get_comp_basis()
    e_sys1 = ElementalSystem(1, basis1)
    c_sys1 = CompositeSystem([e_sys1])
    vecs1 = [
        np.array([2, 3, 5, 7], dtype=np.float64),
        np.array([11, 13, 17, 19], dtype=np.float64),
    ]
    povm1 = Povm(c_sys1, vecs1, is_physicality_required=False)

    basis2 = matrix_basis.get_comp_basis()
    e_sys2 = ElementalSystem(2, basis2)
    c_sys2 = CompositeSystem([e_sys2])
    vecs2 = [
        np.array([23, 29, 31, 37], dtype=np.float64),
        np.array([41, 43, 47, 53], dtype=np.float64),
        np.array([59, 61, 67, 71], dtype=np.float64),
    ]
    povm2 = Povm(c_sys2, vecs2, is_physicality_required=False)

    # Act
    povm21 = tensor_product(povm2, povm1)
    actual = povm21.nums_local_outcomes

    # Assert
    expected = [len(vecs1), len(vecs2)]
    assert actual == expected


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

    # error case: compose different composite systems
    with pytest.raises(ValueError):
        compose_qoperations(i_gate0, i_gate1)


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
        _ = _compose_qoperations(state, z_gate)


def test_compose_qoperations_Gate_Gate():
    e_sys = ElementalSystem(0, matrix_basis.get_normalized_pauli_basis())
    c_sys = CompositeSystem([e_sys])
    i_gate = get_i(c_sys)
    x_gate = get_x(c_sys)
    y_gate = get_y(c_sys)
    z_gate = get_z(c_sys)

    # X \circ X = I
    actual = compose_qoperations(x_gate, x_gate)
    expected = i_gate.hs
    npt.assert_almost_equal(actual.hs, expected, decimal=15)

    # X \circ Y = Z
    actual = compose_qoperations(x_gate, y_gate)
    expected = z_gate.hs
    npt.assert_almost_equal(actual.hs, expected, decimal=15)

    # X \circ X \circ X = X
    actual = compose_qoperations(x_gate, x_gate, x_gate)
    expected = x_gate.hs
    npt.assert_almost_equal(actual.hs, expected, decimal=15)

    # assert associativity
    # (X \circ Y) \circ Z = X \circ (Y \circ Z)
    xy_z = compose_qoperations(compose_qoperations(x_gate, y_gate), z_gate)
    x_yz = compose_qoperations(x_gate, compose_qoperations(y_gate, z_gate))
    npt.assert_almost_equal(xy_z.hs, x_yz.hs, decimal=15)


def test_compose_qoperations_Gate_MProcess():
    # Arrange
    e_sys = ElementalSystem(0, matrix_basis.get_normalized_pauli_basis())
    c_sys = CompositeSystem([e_sys])
    gate_x = generate_gate_from_gate_name("x", c_sys)
    mprocess_z = generate_mprocess_from_name(c_sys, "z-type1")

    # Act
    actual = compose_qoperations(gate_x, mprocess_z)

    # Assert
    assert actual.shape == (2,)
    assert len(actual.hss) == 2
    expected_hs_0 = (
        np.array(
            [[1, 0, 0, 1], [0, 0, 0, 0], [0, 0, 0, 0], [-1, 0, 0, -1]], dtype=np.float64
        )
        / 2
    )
    expected_hs_1 = (
        np.array(
            [[1, 0, 0, -1], [0, 0, 0, 0], [0, 0, 0, 0], [1, 0, 0, -1]], dtype=np.float64
        )
        / 2
    )
    npt.assert_almost_equal(actual.hss[0], expected_hs_0, decimal=15)
    npt.assert_almost_equal(actual.hss[1], expected_hs_1, decimal=15)


def test_compose_qoperations_MProcess_Gate():
    # Arrange
    e_sys = ElementalSystem(0, matrix_basis.get_normalized_pauli_basis())
    c_sys = CompositeSystem([e_sys])
    mprocess_z = generate_mprocess_from_name(c_sys, "z-type1")
    gate_x = generate_gate_from_gate_name("x", c_sys)

    # Act
    actual = compose_qoperations(mprocess_z, gate_x)

    # Assert
    assert actual.shape == (2,)
    assert len(actual.hss) == 2
    expected_hs_0 = (
        np.array(
            [[1, 0, 0, -1], [0, 0, 0, 0], [0, 0, 0, 0], [1, 0, 0, -1]], dtype=np.float64
        )
        / 2
    )
    expected_hs_1 = (
        np.array(
            [[1, 0, 0, 1], [0, 0, 0, 0], [0, 0, 0, 0], [-1, 0, 0, -1]], dtype=np.float64
        )
        / 2
    )
    npt.assert_almost_equal(actual.hss[0], expected_hs_0, decimal=15)
    npt.assert_almost_equal(actual.hss[1], expected_hs_1, decimal=15)


def test_compose_qoperations_Gate_State():
    e_sys = ElementalSystem(0, matrix_basis.get_normalized_pauli_basis())
    c_sys = CompositeSystem([e_sys])
    z_gate = get_z(c_sys)
    h_gate = get_h(c_sys)
    s_gate = get_s(c_sys)

    # case: Z gate \circ x0 state. Z|+><+|Z^{\dagger} = |-><-|
    state = get_x0_1q(c_sys)
    actual = compose_qoperations(z_gate, state)
    expected = 1 / np.sqrt(2) * np.array([1, -1, 0, 0], dtype=np.complex128)
    npt.assert_almost_equal(actual.vec, expected, decimal=15)

    # case: Z gate \circ x1 state. Z|-><-|Z^{\dagger} = |+><+|
    state = get_x1_1q(c_sys)
    actual = compose_qoperations(z_gate, state)
    expected = 1 / np.sqrt(2) * np.array([1, 1, 0, 0], dtype=np.float64)
    npt.assert_almost_equal(actual.vec, expected, decimal=15)

    # case: Z gate \circ z0 state. Z|0><0|Z^{\dagger} = |0><0|
    state = get_z0_1q(c_sys)
    actual = compose_qoperations(z_gate, state)
    expected = 1 / np.sqrt(2) * np.array([1, 0, 0, 1], dtype=np.float64)
    npt.assert_almost_equal(actual.vec, expected, decimal=15)

    # case: Z gate \circ z1 state. Z|1><1|Z^{\dagger} = |1><1|
    state = get_z1_1q(c_sys)
    actual = compose_qoperations(z_gate, state)
    expected = 1 / np.sqrt(2) * np.array([1, 0, 0, -1], dtype=np.float64)
    npt.assert_almost_equal(actual.vec, expected, decimal=15)

    # case: S gate \circ S gate \circ z0 state. SS|0><0|S^{\dagger}S^{\dagger} = |0><0|
    # S = root(Z), hense SS = Z
    state = get_z0_1q(c_sys)
    actual = compose_qoperations(s_gate, s_gate, state)
    expected = 1 / np.sqrt(2) * np.array([1, 0, 0, 1], dtype=np.float64)
    npt.assert_almost_equal(actual.vec, expected, decimal=15)

    # case: H gate \circ z0 state. H|0><0|H^{\dagger} = 1/2(I+X)
    state = get_z0_1q(c_sys)
    actual = compose_qoperations(h_gate, state)
    expected = 1 / np.sqrt(2) * np.array([1, 1, 0, 0], dtype=np.float64)
    npt.assert_almost_equal(actual.vec, expected, decimal=15)

    # case: H gate \circ z1 state. H|1><1|H^{\dagger} = 1/2(I-X)
    state = get_z1_1q(c_sys)
    actual = compose_qoperations(h_gate, state)
    expected = 1 / np.sqrt(2) * np.array([1, -1, 0, 0], dtype=np.float64)
    npt.assert_almost_equal(actual.vec, expected, decimal=15)

    # assert associativity
    # (H \circ Z) \circ |1> = H \circ (Z \circ |1>)
    state = get_z1_1q(c_sys)
    hz_1 = compose_qoperations(compose_qoperations(h_gate, z_gate), state)
    h_z1 = compose_qoperations(h_gate, compose_qoperations(z_gate, state))
    npt.assert_almost_equal(hz_1.vec, h_z1.vec, decimal=15)


def test_compose_qoperations_MProcess_MProcess():
    # Arrange
    e_sys = ElementalSystem(0, matrix_basis.get_normalized_pauli_basis())
    c_sys = CompositeSystem([e_sys])
    mprocess_z1 = generate_mprocess_from_name(c_sys, "z-type1")
    mprocess_z2 = generate_mprocess_from_name(c_sys, "z-type1")

    # Act
    actual = compose_qoperations(mprocess_z1, mprocess_z2)

    # Assert
    assert actual.shape == (2, 2)
    assert len(actual.hss) == 4
    expected_hs_0 = (
        np.array(
            [[1, 0, 0, 1], [0, 0, 0, 0], [0, 0, 0, 0], [1, 0, 0, 1]], dtype=np.float64
        )
        / 2
    )
    expected_hs_3 = (
        np.array(
            [[1, 0, 0, 1], [0, 0, 0, 0], [0, 0, 0, 0], [1, 0, 0, 1]], dtype=np.float64
        )
        / 2
    )
    npt.assert_almost_equal(actual.hss[0], expected_hs_0, decimal=15)
    npt.assert_almost_equal(actual.hss[1], np.zeros((4, 4)), decimal=15)
    npt.assert_almost_equal(actual.hss[2], np.zeros((4, 4)), decimal=15)
    npt.assert_almost_equal(actual.hss[0], expected_hs_3, decimal=15)


def test_compose_qoperations_MProcess_State():
    ## case 1: is_orthonormal_hermitian_0thprop_identity = True
    # Arrange
    e_sys = ElementalSystem(0, matrix_basis.get_normalized_pauli_basis())
    c_sys = CompositeSystem([e_sys])
    state_z0 = generate_state_from_name(c_sys, "z0")
    mprocess_z = generate_mprocess_from_name(c_sys, "z-type1")
    mprocess_z._eps_zero = 10 ** -5

    # Act
    actual = compose_qoperations(mprocess_z, state_z0)

    # Assert
    assert actual.prob_dist.shape == (2,)
    assert len(actual.states) == 2
    npt.assert_almost_equal(actual.states[0].vec, state_z0.vec, decimal=15)
    npt.assert_almost_equal(
        actual.states[1].vec, state_z0._generate_zero_obj(), decimal=15
    )
    expected_prob_dist = np.array([1, 0], dtype=np.float64)
    npt.assert_almost_equal(actual.prob_dist.ps, expected_prob_dist, decimal=15)
    assert actual.eps_zero == 10 ** -5

    # TODO allow complex numbers in variables
    """
    ## case 2: is_orthonormal_hermitian_0thprop_identity = False
    e_sys = ElementalSystem(0, matrix_basis.get_comp_basis())
    c_sys = CompositeSystem([e_sys])
    state_z0 = generate_state_from_name(c_sys, "z0")
    mprocess_z = generate_mprocess_from_name(c_sys, "z-type1")

    # Act
    actual = compose_qoperations(mprocess_z, state_z0)

    # Assert
    assert actual.prob_dist.shape == (2, 2)
    assert len(actual.states) == 2
    npt.assert_almost_equal(actual.states[0].vec, state_z0.vec, decimal=15)
    npt.assert_almost_equal(
        actual.states[1].vec, state_z0._generate_zero_obj(), decimal=15
    )
    expected_prob_dist = np.array([1, 0], dtype=np.float64)
    npt.assert_almost_equal(actual.prob_dist.ps, expected_prob_dist, decimal=15)
    """

    # case 3: mode_sampling = True (return State)
    # Arrange
    e_sys = ElementalSystem(0, matrix_basis.get_normalized_pauli_basis())
    c_sys = CompositeSystem([e_sys])
    state_z0 = generate_state_from_name(c_sys, "z0")
    mprocess_z = generate_mprocess_from_name(c_sys, "z-type1")
    mprocess_z._mode_sampling = True

    # Act
    actual = compose_qoperations(mprocess_z, state_z0)

    # Assert
    assert type(actual) == State
    expected = np.array([1, 0, 0, 1], dtype=np.float64) / np.sqrt(2)
    npt.assert_almost_equal(actual.vec, expected, decimal=15)


def test_compose_qoperations_MProcess_StateEnsemble():
    ## case 1: is_orthonormal_hermitian_0thprop_identity = True
    # Arrange
    e_sys = ElementalSystem(0, matrix_basis.get_normalized_pauli_basis())
    c_sys = CompositeSystem([e_sys])
    state_ens_z0 = generate_state_ensemble_from_name(c_sys, "x0")
    mprocess_z = generate_mprocess_from_name(c_sys, "z-type1")
    mprocess_z._eps_zero = 10 ** -5

    # Act
    actual = compose_qoperations(mprocess_z, state_ens_z0)

    # Assert
    assert actual.prob_dist.shape == (2, 2)
    expected_z0 = np.array([1, 0, 0, 1], dtype=np.float64) / np.sqrt(2)
    expected_z1 = np.array([1, 0, 0, -1], dtype=np.float64) / np.sqrt(2)
    assert len(actual.states) == 4
    npt.assert_almost_equal(actual.states[0].vec, expected_z0, decimal=15)
    npt.assert_almost_equal(actual.states[1].vec, expected_z1, decimal=15)
    npt.assert_almost_equal(actual.states[2].vec, expected_z0, decimal=15)
    npt.assert_almost_equal(actual.states[3].vec, expected_z1, decimal=15)
    expected_prob_dist = np.array([1 / 4, 1 / 4, 1 / 4, 1 / 4], dtype=np.float64)
    npt.assert_almost_equal(actual.prob_dist.ps, expected_prob_dist, decimal=15)
    assert actual.eps_zero == 10 ** -5

    # TODO allow complex numbers in variables
    """
    ## case 2: is_orthonormal_hermitian_0thprop_identity = False
    # Arrange
    e_sys = ElementalSystem(0, matrix_basis.get_comp_basis())
    c_sys = CompositeSystem([e_sys])
    state_ens_z0 = generate_state_ensemble_from_name(c_sys, "x0")
    mprocess_z = generate_mprocess_from_name(c_sys, "z-type1")

    # Act
    actual = compose_qoperations(mprocess_z, state_ens_z0)

    # Assert
    assert actual.prob_dist.shape == (2, 2)
    expected_z0 = np.array([1, 0, 0, 0], dtype=np.float64)
    expected_z1 = np.array([0, 0, 0, 1], dtype=np.float64)
    assert len(actual.states) == 4
    npt.assert_almost_equal(actual.states[0].vec, expected_z0, decimal=15)
    npt.assert_almost_equal(actual.states[1].vec, expected_z1, decimal=15)
    npt.assert_almost_equal(actual.states[2].vec, expected_z0, decimal=15)
    npt.assert_almost_equal(actual.states[3].vec, expected_z1, decimal=15)
    expected_prob_dist = np.array([1 / 4, 1 / 4, 1 / 4, 1 / 4], dtype=np.float64)
    npt.assert_almost_equal(actual.prob_dist.ps, expected_prob_dist, decimal=15)
    """

    # case 3: mode_sampling = True (return State)
    # Arrange
    e_sys = ElementalSystem(0, matrix_basis.get_normalized_pauli_basis())
    c_sys = CompositeSystem([e_sys])
    state_ens_z0 = generate_state_ensemble_from_name(c_sys, "z0")
    state_ens_z0._eps_zero = 10 ** -4
    mprocess_z = generate_mprocess_from_name(c_sys, "z-type1")
    mprocess_z._mode_sampling = True

    # Act
    actual = compose_qoperations(mprocess_z, state_ens_z0)

    # Assert
    assert type(actual) == StateEnsemble
    assert len(actual.states) == 2
    assert len(actual.prob_dist.ps) == 2
    assert actual.prob_dist.shape == (2,)
    expected = [
        np.array([1, 0, 0, 1], dtype=np.float64) / np.sqrt(2),
        np.array([0, 0, 0, 0], dtype=np.float64),
    ]
    for a, e in zip(actual.states, expected):
        npt.assert_almost_equal(a.vec, e, decimal=15)
    npt.assert_almost_equal(actual.prob_dist.ps, [1, 0], decimal=15)
    assert actual.eps_zero == 10 ** -4


def test_compose_qoperations_MProcess_StateEnsemble_is_zero_dist():
    ## case 1: is_orthonormal_hermitian_0thprop_identity = True
    # Arrange
    e_sys = ElementalSystem(0, matrix_basis.get_normalized_pauli_basis())
    c_sys = CompositeSystem([e_sys])
    vec = np.array([0, 0, 0, 0], dtype=np.float64)
    state_zero = State(c_sys, vec, is_physicality_required=False)
    states = [state_zero, state_zero]
    mult_dist = MultinomialDistribution(np.array([0.0, 0.0]))
    state_ens = StateEnsemble(states, mult_dist)

    mprocess_z = generate_mprocess_from_name(c_sys, "z-type1")
    mprocess_z._eps_zero = 10 ** -5

    # Act
    actual = compose_qoperations(mprocess_z, state_ens)

    # Assert
    assert actual.prob_dist.shape == (2, 2)
    expected_vec = np.array([0, 0, 0, 0], dtype=np.float64)
    assert len(actual.states) == 4
    npt.assert_almost_equal(actual.states[0].vec, expected_vec, decimal=15)
    npt.assert_almost_equal(actual.states[1].vec, expected_vec, decimal=15)
    npt.assert_almost_equal(actual.states[2].vec, expected_vec, decimal=15)
    npt.assert_almost_equal(actual.states[3].vec, expected_vec, decimal=15)
    expected_prob_dist = np.array([0, 0, 0, 0], dtype=np.float64)
    npt.assert_almost_equal(actual.prob_dist.ps, expected_prob_dist, decimal=15)
    assert actual.eps_zero == 10 ** -5


def test_compose_qoperations_Povm_Gate():
    e_sys = ElementalSystem(0, matrix_basis.get_normalized_pauli_basis())
    c_sys = CompositeSystem([e_sys])
    vecs = [
        1 / np.sqrt(2) * np.array([1, 0, 0, 1], dtype=np.float64),
        1 / np.sqrt(2) * np.array([1, 0, 0, -1], dtype=np.float64),
    ]
    povm = Povm(c_sys, vecs)

    # compose Z-measurement and X gate
    x_gate = get_x(c_sys)
    actual = compose_qoperations(povm, x_gate)
    expected = [
        1 / np.sqrt(2) * np.array([1, 0, 0, -1], dtype=np.float64),
        1 / np.sqrt(2) * np.array([1, 0, 0, 1], dtype=np.float64),
    ]
    npt.assert_almost_equal(actual.vecs, expected, decimal=15)

    # assert associativity
    # (POVM \circ X) \circ Z = POVM \circ (X \circ Z)
    z_gate = get_z(c_sys)
    px_z = compose_qoperations(compose_qoperations(povm, x_gate), z_gate)
    p_xz = compose_qoperations(povm, compose_qoperations(x_gate, z_gate))
    npt.assert_almost_equal(px_z.vecs[0], p_xz.vecs[0], decimal=15)
    npt.assert_almost_equal(px_z.vecs[1], p_xz.vecs[1], decimal=15)


def test_compose_qoperations_Povm_State():
    e_sys = ElementalSystem(0, matrix_basis.get_comp_basis())
    c_sys = CompositeSystem([e_sys])
    vecs = [
        np.array([1, 0, 0, 0], dtype=np.float64),
        np.array([0, 0, 0, 1], dtype=np.float64),
    ]
    povm = Povm(c_sys, vecs)

    # measurement z0 by Z-measurement
    state = get_z0_1q(c_sys)
    actual = compose_qoperations(povm, state)
    expected = [1, 0]
    npt.assert_almost_equal(actual.ps, expected, decimal=15)

    # measurement z1 by Z-measurement
    state = get_z1_1q(c_sys)
    actual = compose_qoperations(povm, state)
    expected = [0, 1]
    npt.assert_almost_equal(actual.ps, expected, decimal=15)

    # measurement x0 by Z-measurement
    state = get_x0_1q(c_sys)
    actual = compose_qoperations(povm, state)
    expected = [0.5, 0.5]
    npt.assert_almost_equal(actual.ps, expected, decimal=15)

    # assert associativity
    # (POVM \circ X) \circ |1> = POVM \circ (X \circ |1>)
    state = get_z1_1q(c_sys)
    x_gate = get_x(c_sys)
    px_z = compose_qoperations(compose_qoperations(povm, x_gate), state)
    p_xz = compose_qoperations(povm, compose_qoperations(x_gate, state))
    npt.assert_almost_equal(px_z.ps, p_xz.ps, decimal=15)


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


@pytest.mark.parametrize(
    ("state", "gate", "povm", "expected"),
    [
        # I gate
        (get_x0_1q, get_i, get_x_povm, [1, 0]),
        (get_x0_1q, get_i, get_y_povm, [0.5, 0.5]),
        (get_x0_1q, get_i, get_z_povm, [0.5, 0.5]),
        (get_y0_1q, get_i, get_x_povm, [0.5, 0.5]),
        (get_y0_1q, get_i, get_y_povm, [1, 0]),
        (get_y0_1q, get_i, get_z_povm, [0.5, 0.5]),
        (get_z0_1q, get_i, get_x_povm, [0.5, 0.5]),
        (get_z0_1q, get_i, get_y_povm, [0.5, 0.5]),
        (get_z0_1q, get_i, get_z_povm, [1, 0]),
        (get_z1_1q, get_i, get_x_povm, [0.5, 0.5]),
        (get_z1_1q, get_i, get_y_povm, [0.5, 0.5]),
        (get_z1_1q, get_i, get_z_povm, [0, 1]),
        # X gate
        (get_x0_1q, get_x, get_x_povm, [1, 0]),
        (get_x0_1q, get_x, get_y_povm, [0.5, 0.5]),
        (get_x0_1q, get_x, get_z_povm, [0.5, 0.5]),
        (get_y0_1q, get_x, get_x_povm, [0.5, 0.5]),
        (get_y0_1q, get_x, get_y_povm, [0, 1]),
        (get_y0_1q, get_x, get_z_povm, [0.5, 0.5]),
        (get_z0_1q, get_x, get_x_povm, [0.5, 0.5]),
        (get_z0_1q, get_x, get_y_povm, [0.5, 0.5]),
        (get_z0_1q, get_x, get_z_povm, [0, 1]),
        (get_z1_1q, get_x, get_x_povm, [0.5, 0.5]),
        (get_z1_1q, get_x, get_y_povm, [0.5, 0.5]),
        (get_z1_1q, get_x, get_z_povm, [1, 0]),
        # Y gate
        (get_x0_1q, get_y, get_x_povm, [0, 1]),
        (get_x0_1q, get_y, get_y_povm, [0.5, 0.5]),
        (get_x0_1q, get_y, get_z_povm, [0.5, 0.5]),
        (get_y0_1q, get_y, get_x_povm, [0.5, 0.5]),
        (get_y0_1q, get_y, get_y_povm, [1, 0]),
        (get_y0_1q, get_y, get_z_povm, [0.5, 0.5]),
        (get_z0_1q, get_y, get_x_povm, [0.5, 0.5]),
        (get_z0_1q, get_y, get_y_povm, [0.5, 0.5]),
        (get_z0_1q, get_y, get_z_povm, [0, 1]),
        (get_z1_1q, get_y, get_x_povm, [0.5, 0.5]),
        (get_z1_1q, get_y, get_y_povm, [0.5, 0.5]),
        (get_z1_1q, get_y, get_z_povm, [1, 0]),
        # Z gate
        (get_x0_1q, get_z, get_x_povm, [0, 1]),
        (get_x0_1q, get_z, get_y_povm, [0.5, 0.5]),
        (get_x0_1q, get_z, get_z_povm, [0.5, 0.5]),
        (get_y0_1q, get_z, get_x_povm, [0.5, 0.5]),
        (get_y0_1q, get_z, get_y_povm, [0, 1]),
        (get_y0_1q, get_z, get_z_povm, [0.5, 0.5]),
        (get_z0_1q, get_z, get_x_povm, [0.5, 0.5]),
        (get_z0_1q, get_z, get_y_povm, [0.5, 0.5]),
        (get_z0_1q, get_z, get_z_povm, [1, 0]),
        (get_z1_1q, get_z, get_x_povm, [0.5, 0.5]),
        (get_z1_1q, get_z, get_y_povm, [0.5, 0.5]),
        (get_z1_1q, get_z, get_z_povm, [0, 1]),
        # H gate
        (get_x0_1q, get_h, get_x_povm, [0.5, 0.5]),
        (get_x0_1q, get_h, get_y_povm, [0.5, 0.5]),
        (get_x0_1q, get_h, get_z_povm, [1, 0]),
        (get_y0_1q, get_h, get_x_povm, [0.5, 0.5]),
        (get_y0_1q, get_h, get_y_povm, [0, 1]),
        (get_y0_1q, get_h, get_z_povm, [0.5, 0.5]),
        (get_z0_1q, get_h, get_x_povm, [1, 0]),
        (get_z0_1q, get_h, get_y_povm, [0.5, 0.5]),
        (get_z0_1q, get_h, get_z_povm, [0.5, 0.5]),
        (get_z1_1q, get_h, get_x_povm, [0, 1]),
        (get_z1_1q, get_h, get_y_povm, [0.5, 0.5]),
        (get_z1_1q, get_h, get_z_povm, [0.5, 0.5]),
        # root X gate
        (get_x0_1q, get_root_x, get_x_povm, [1, 0]),
        (get_x0_1q, get_root_x, get_y_povm, [0.5, 0.5]),
        (get_x0_1q, get_root_x, get_z_povm, [0.5, 0.5]),
        (get_y0_1q, get_root_x, get_x_povm, [0.5, 0.5]),
        (get_y0_1q, get_root_x, get_y_povm, [0.5, 0.5]),
        (get_y0_1q, get_root_x, get_z_povm, [1, 0]),
        (get_z0_1q, get_root_x, get_x_povm, [0.5, 0.5]),
        (get_z0_1q, get_root_x, get_y_povm, [0, 1]),
        (get_z0_1q, get_root_x, get_z_povm, [0.5, 0.5]),
        (get_z1_1q, get_root_x, get_x_povm, [0.5, 0.5]),
        (get_z1_1q, get_root_x, get_y_povm, [1, 0]),
        (get_z1_1q, get_root_x, get_z_povm, [0.5, 0.5]),
        # root Y gate
        (get_x0_1q, get_root_y, get_x_povm, [0.5, 0.5]),
        (get_x0_1q, get_root_y, get_y_povm, [0.5, 0.5]),
        (get_x0_1q, get_root_y, get_z_povm, [0, 1]),
        (get_y0_1q, get_root_y, get_x_povm, [0.5, 0.5]),
        (get_y0_1q, get_root_y, get_y_povm, [1, 0]),
        (get_y0_1q, get_root_y, get_z_povm, [0.5, 0.5]),
        (get_z0_1q, get_root_y, get_x_povm, [1, 0]),
        (get_z0_1q, get_root_y, get_y_povm, [0.5, 0.5]),
        (get_z0_1q, get_root_y, get_z_povm, [0.5, 0.5]),
        (get_z1_1q, get_root_y, get_x_povm, [0, 1]),
        (get_z1_1q, get_root_y, get_y_povm, [0.5, 0.5]),
        (get_z1_1q, get_root_y, get_z_povm, [0.5, 0.5]),
        # S gate
        (get_x0_1q, get_s, get_x_povm, [0.5, 0.5]),
        (get_x0_1q, get_s, get_y_povm, [1, 0]),
        (get_x0_1q, get_s, get_z_povm, [0.5, 0.5]),
        (get_y0_1q, get_s, get_x_povm, [0, 1]),
        (get_y0_1q, get_s, get_y_povm, [0.5, 0.5]),
        (get_y0_1q, get_s, get_z_povm, [0.5, 0.5]),
        (get_z0_1q, get_s, get_x_povm, [0.5, 0.5]),
        (get_z0_1q, get_s, get_y_povm, [0.5, 0.5]),
        (get_z0_1q, get_s, get_z_povm, [1, 0]),
        (get_z1_1q, get_s, get_x_povm, [0.5, 0.5]),
        (get_z1_1q, get_s, get_y_povm, [0.5, 0.5]),
        (get_z1_1q, get_s, get_z_povm, [0, 1]),
        # SDG gate
        (get_x0_1q, get_sdg, get_x_povm, [0.5, 0.5]),
        (get_x0_1q, get_sdg, get_y_povm, [0, 1]),
        (get_x0_1q, get_sdg, get_z_povm, [0.5, 0.5]),
        (get_y0_1q, get_sdg, get_x_povm, [1, 0]),
        (get_y0_1q, get_sdg, get_y_povm, [0.5, 0.5]),
        (get_y0_1q, get_sdg, get_z_povm, [0.5, 0.5]),
        (get_z0_1q, get_sdg, get_x_povm, [0.5, 0.5]),
        (get_z0_1q, get_sdg, get_y_povm, [0.5, 0.5]),
        (get_z0_1q, get_sdg, get_z_povm, [1, 0]),
        (get_z1_1q, get_sdg, get_x_povm, [0.5, 0.5]),
        (get_z1_1q, get_sdg, get_y_povm, [0.5, 0.5]),
        (get_z1_1q, get_sdg, get_z_povm, [0, 1]),
        # T gate
        (
            get_x0_1q,
            get_t,
            get_x_povm,
            [(2 + np.sqrt(2)) / 4, (2 - np.sqrt(2)) / 4],
        ),
        (
            get_x0_1q,
            get_t,
            get_y_povm,
            [(2 + np.sqrt(2)) / 4, (2 - np.sqrt(2)) / 4],
        ),
        (get_x0_1q, get_t, get_z_povm, [0.5, 0.5]),
        (
            get_y0_1q,
            get_t,
            get_x_povm,
            [(2 - np.sqrt(2)) / 4, (2 + np.sqrt(2)) / 4],
        ),
        (
            get_y0_1q,
            get_t,
            get_y_povm,
            [(2 + np.sqrt(2)) / 4, (2 - np.sqrt(2)) / 4],
        ),
        (get_y0_1q, get_t, get_z_povm, [0.5, 0.5]),
        (get_z0_1q, get_t, get_x_povm, [0.5, 0.5]),
        (get_z0_1q, get_t, get_y_povm, [0.5, 0.5]),
        (get_z0_1q, get_t, get_z_povm, [1, 0]),
        (get_z1_1q, get_t, get_x_povm, [0.5, 0.5]),
        (get_z1_1q, get_t, get_y_povm, [0.5, 0.5]),
        (get_z1_1q, get_t, get_z_povm, [0, 1]),
    ],
)
def test_scenario_1qubit(state, gate, povm, expected):
    # Prepare
    e_sys1 = ElementalSystem(1, matrix_basis.get_normalized_pauli_basis())
    c_sys1 = CompositeSystem([e_sys1])

    state_obj = state(c_sys1)
    gate_obj = gate(c_sys1)
    povm_obj = povm(c_sys1)

    # Act
    actual = compose_qoperations(povm_obj, gate_obj, state_obj)

    # Assert
    npt.assert_almost_equal(actual.ps, expected, decimal=15)


@pytest.mark.parametrize(
    ("povm", "expected"),
    [
        (get_xx_povm, [0.5, 0, 0, 0.5]),
        (get_xy_povm, [0.25, 0.25, 0.25, 0.25]),
        (get_xz_povm, [0.25, 0.25, 0.25, 0.25]),
        (get_yx_povm, [0.25, 0.25, 0.25, 0.25]),
        (get_yy_povm, [0, 0.5, 0.5, 0]),
        (get_yz_povm, [0.25, 0.25, 0.25, 0.25]),
        (get_zx_povm, [0.25, 0.25, 0.25, 0.25]),
        (get_zy_povm, [0.25, 0.25, 0.25, 0.25]),
        (get_zz_povm, [0.5, 0, 0, 0.5]),
    ],
)
def test_scenario_2qubits_cnot(povm, expected):
    ### case 1
    # Prepare
    e_sys1 = ElementalSystem(1, matrix_basis.get_normalized_pauli_basis())
    c_sys1 = CompositeSystem([e_sys1])
    e_sys2 = ElementalSystem(2, matrix_basis.get_normalized_pauli_basis())
    c_sys2 = CompositeSystem([e_sys2])
    c_sys12 = CompositeSystem([e_sys1, e_sys2])

    state1 = get_z0_1q(c_sys1)
    state2 = get_z0_1q(c_sys2)
    h = get_h(c_sys1)
    cnot = get_cnot(c_sys12, e_sys1)
    povm_obj = povm(c_sys12)

    # Act
    state1 = compose_qoperations(h, state1)
    state12 = tensor_product(state1, state2)
    actual = compose_qoperations(povm_obj, cnot, state12)

    # Assert
    npt.assert_almost_equal(actual.ps, expected, decimal=15)

    ### case 2 (reverse cnot of case 1)
    # Prepare
    e_sys3 = ElementalSystem(3, matrix_basis.get_normalized_pauli_basis())
    c_sys3 = CompositeSystem([e_sys3])
    e_sys4 = ElementalSystem(4, matrix_basis.get_normalized_pauli_basis())
    c_sys4 = CompositeSystem([e_sys4])
    c_sys34 = CompositeSystem([e_sys3, e_sys4])

    state3 = get_z0_1q(c_sys3)
    state4 = get_z0_1q(c_sys4)
    h = get_h(c_sys4)
    cnot = get_cnot(c_sys34, e_sys4)
    swap = get_swap(c_sys34)
    povm_obj = povm(c_sys34)

    # Act
    state4 = compose_qoperations(h, state4)
    state34 = tensor_product(state3, state4)
    actual = compose_qoperations(povm_obj, swap, cnot, state34)

    # Assert
    npt.assert_almost_equal(actual.ps, expected, decimal=14)


@pytest.mark.parametrize(
    ("povm", "expected"),
    [
        (get_xx_povm, [0, 0.5, 0, 0.5]),
        (get_xy_povm, [0.25, 0.25, 0.25, 0.25]),
        (get_xz_povm, [0.25, 0.25, 0.25, 0.25]),
        (get_yx_povm, [0, 0.5, 0, 0.5]),
        (get_yy_povm, [0.25, 0.25, 0.25, 0.25]),
        (get_yz_povm, [0.25, 0.25, 0.25, 0.25]),
        (get_zx_povm, [0, 0, 0, 1]),
        (get_zy_povm, [0, 0, 0.5, 0.5]),
        (get_zz_povm, [0, 0, 0.5, 0.5]),
    ],
)
def test_scenario_2qubits_cz(povm, expected):
    ### case 1
    # Prepare
    e_sys1 = ElementalSystem(1, matrix_basis.get_normalized_pauli_basis())
    c_sys1 = CompositeSystem([e_sys1])
    e_sys2 = ElementalSystem(2, matrix_basis.get_normalized_pauli_basis())
    c_sys2 = CompositeSystem([e_sys2])
    c_sys12 = CompositeSystem([e_sys1, e_sys2])

    state1 = get_z1_1q(c_sys1)
    state2 = get_z0_1q(c_sys2)
    h = get_h(c_sys2)
    swap = get_cz(c_sys12)
    povm_obj = povm(c_sys12)

    # Act
    state2 = compose_qoperations(h, state2)
    state12 = tensor_product(state1, state2)
    actual = compose_qoperations(povm_obj, swap, state12)

    # Assert
    npt.assert_almost_equal(actual.ps, expected, decimal=14)

    ### case 2 (reverse cz of case 1)
    # Prepare
    e_sys3 = ElementalSystem(3, matrix_basis.get_normalized_pauli_basis())
    c_sys3 = CompositeSystem([e_sys3])
    e_sys4 = ElementalSystem(4, matrix_basis.get_normalized_pauli_basis())
    c_sys4 = CompositeSystem([e_sys4])
    c_sys34 = CompositeSystem([e_sys3, e_sys4])

    state3 = get_z0_1q(c_sys3)
    state4 = get_z1_1q(c_sys4)
    h = get_h(c_sys3)
    cz = get_cz(c_sys34)
    swap = get_swap(c_sys34)
    povm_obj = povm(c_sys34)

    # Act
    state3 = compose_qoperations(h, state3)
    state34 = tensor_product(state3, state4)
    actual = compose_qoperations(povm_obj, swap, cz, state34)

    # Assert
    npt.assert_almost_equal(actual.ps, expected, decimal=14)


@pytest.mark.parametrize(
    ("povm", "expected"),
    [
        (get_xx_povm, [0.5, 0, 0.5, 0]),
        (get_xy_povm, [0.25, 0.25, 0.25, 0.25]),
        (get_xz_povm, [0.25, 0.25, 0.25, 0.25]),
        (get_yx_povm, [0.5, 0, 0.5, 0]),
        (get_yy_povm, [0.25, 0.25, 0.25, 0.25]),
        (get_yz_povm, [0.25, 0.25, 0.25, 0.25]),
        (get_zx_povm, [1, 0, 0, 0]),
        (get_zy_povm, [0.5, 0.5, 0, 0]),
        (get_zz_povm, [0.5, 0.5, 0, 0]),
    ],
)
def test_scenario_2qubits_swap(povm, expected):
    ### case 1
    # Prepare
    e_sys1 = ElementalSystem(1, matrix_basis.get_normalized_pauli_basis())
    c_sys1 = CompositeSystem([e_sys1])
    e_sys2 = ElementalSystem(2, matrix_basis.get_normalized_pauli_basis())
    c_sys2 = CompositeSystem([e_sys2])
    c_sys12 = CompositeSystem([e_sys1, e_sys2])

    state1 = get_z0_1q(c_sys1)
    state2 = get_z0_1q(c_sys2)
    h = get_h(c_sys1)
    swap = get_swap(c_sys12)
    cnot12 = get_cnot(c_sys12, e_sys1)
    cnot21 = get_cnot(c_sys12, e_sys2)
    povm_obj = povm(c_sys12)

    # Act
    state1 = compose_qoperations(h, state1)
    state12 = tensor_product(state1, state2)
    actual = compose_qoperations(povm_obj, swap, state12)

    # Assert
    npt.assert_almost_equal(actual.ps, expected, decimal=14)

    ### case 2 (use cnot instead of swap)
    # Prepare
    e_sys3 = ElementalSystem(3, matrix_basis.get_normalized_pauli_basis())
    c_sys3 = CompositeSystem([e_sys3])
    e_sys4 = ElementalSystem(4, matrix_basis.get_normalized_pauli_basis())
    c_sys4 = CompositeSystem([e_sys4])
    c_sys34 = CompositeSystem([e_sys3, e_sys4])

    state3 = get_z0_1q(c_sys3)
    state4 = get_z0_1q(c_sys4)
    h = get_h(c_sys3)
    swap = get_swap(c_sys34)
    cnot12 = get_cnot(c_sys34, e_sys3)
    cnot21 = get_cnot(c_sys34, e_sys4)
    povm_obj = povm(c_sys34)

    # Act
    state3 = compose_qoperations(h, state3)
    state34 = tensor_product(state3, state4)
    actual = compose_qoperations(povm_obj, cnot12, cnot21, cnot12, state34)

    # Assert
    npt.assert_almost_equal(actual.ps, expected, decimal=14)


def _calculate_vecs(dim: int, density_matrix: np.array):
    # calculate vec of State and vecs of POVM
    comp_basis = matrix_basis.get_comp_basis(dim)
    herm_basis = matrix_basis.get_normalized_hermitian_basis(dim)

    state_vec = matrix_basis.convert_vec(
        density_matrix.flatten(), comp_basis, herm_basis
    ).real.astype(np.float64)

    state_vec_complement = matrix_basis.convert_vec(
        (np.eye(dim) - density_matrix).flatten(), comp_basis, herm_basis
    ).real.astype(np.float64)

    povm_vecs = [
        state_vec,
        state_vec_complement,
    ]

    return state_vec, povm_vecs


def _calculate_probability_of_povm_k(dim: int, beta: int):
    # probability distribution
    prob_per_povm = []
    for alpha in range(dim):
        if beta == alpha:
            prob_per_povm.append([1, 0])
        else:
            prob_per_povm.append([0, 1])

    for alpha_ket_k in range(dim):
        for alpha_ket_l in range(alpha_ket_k + 1, dim):
            if beta == alpha_ket_k or beta == alpha_ket_l:
                prob_per_povm.append([0.5, 0.5])
            else:
                prob_per_povm.append([0, 1])

    for alpha_ket_k in range(dim):
        for alpha_ket_l in range(alpha_ket_k + 1, dim):
            if beta == alpha_ket_k or beta == alpha_ket_l:
                prob_per_povm.append([0.5, 0.5])
            else:
                prob_per_povm.append([0, 1])

    return prob_per_povm


def _calculate_probability_of_povm_k_plus_l(dim: int, beta_ket_k: int, beta_ket_l: int):
    prob_per_povm = []
    for alpha in range(dim):
        if beta_ket_k == alpha or beta_ket_l == alpha:
            prob_per_povm.append([0.5, 0.5])
        else:
            prob_per_povm.append([0, 1])

    for alpha_ket_k in range(dim):
        for alpha_ket_l in range(alpha_ket_k + 1, dim):
            if beta_ket_k == alpha_ket_k and beta_ket_l == alpha_ket_l:
                prob_per_povm.append([1, 0])
            elif (
                beta_ket_k == alpha_ket_k
                or beta_ket_k == alpha_ket_l
                or beta_ket_l == alpha_ket_k
                or beta_ket_l == alpha_ket_l
            ):
                prob_per_povm.append([0.25, 0.75])
            else:
                prob_per_povm.append([0, 1])

    for alpha_ket_k in range(dim):
        for alpha_ket_l in range(alpha_ket_k + 1, dim):
            if beta_ket_k == alpha_ket_k and beta_ket_l == alpha_ket_l:
                prob_per_povm.append([0.5, 0.5])
            elif (
                beta_ket_k == alpha_ket_k
                or beta_ket_k == alpha_ket_l
                or beta_ket_l == alpha_ket_k
                or beta_ket_l == alpha_ket_l
            ):
                prob_per_povm.append([0.25, 0.75])
            else:
                prob_per_povm.append([0, 1])

    return prob_per_povm


def _calculate_probability_of_povm_k_plus_il(
    dim: int, beta_ket_k: int, beta_ket_l: int
):
    prob_per_povm = []
    for alpha in range(dim):
        if beta_ket_k == alpha or beta_ket_l == alpha:
            prob_per_povm.append([0.5, 0.5])
        else:
            prob_per_povm.append([0, 1])

    for alpha_ket_k in range(dim):
        for alpha_ket_l in range(alpha_ket_k + 1, dim):
            if beta_ket_k == alpha_ket_k and beta_ket_l == alpha_ket_l:
                prob_per_povm.append([0.5, 0.5])
            elif (
                beta_ket_k == alpha_ket_k
                or beta_ket_k == alpha_ket_l
                or beta_ket_l == alpha_ket_k
                or beta_ket_l == alpha_ket_l
            ):
                prob_per_povm.append([0.25, 0.75])
            else:
                prob_per_povm.append([0, 1])

    for alpha_ket_k in range(dim):
        for alpha_ket_l in range(alpha_ket_k + 1, dim):
            if beta_ket_k == alpha_ket_k and beta_ket_l == alpha_ket_l:
                prob_per_povm.append([1, 0])
            elif (
                beta_ket_k == alpha_ket_k
                or beta_ket_k == alpha_ket_l
                or beta_ket_l == alpha_ket_k
                or beta_ket_l == alpha_ket_l
            ):
                prob_per_povm.append([0.25, 0.75])
            else:
                prob_per_povm.append([0, 1])

    return prob_per_povm


@pytest.mark.parametrize(
    ("d"),
    [
        (2),
        (3),
        (4),
    ],
)
def test_scenario_tomographically_complete_sets(d):
    # see G. C. Knee et al., \Quantum process tomography via completely positive and trace-preserving projection", Phys Rev. A 98, 062336 (2018).

    # Prepare
    e_sys = ElementalSystem(1, matrix_basis.get_normalized_hermitian_basis(d))
    c_sys = CompositeSystem([e_sys])
    gate = get_i(c_sys)

    list_of_state_vec = []
    list_of_povm_vecs = []
    list_of_prob = []

    for ket_k in range(d):
        # calculate state_vec and povm_vecs
        density_matrix = np.zeros((d, d), dtype=np.complex128)
        density_matrix[ket_k, ket_k] = 1
        state_vec, povm_vecs = _calculate_vecs(d, density_matrix)
        list_of_state_vec.append(state_vec)
        list_of_povm_vecs.append(povm_vecs)

        # probability distribution (=expected)
        prob_per_povm = _calculate_probability_of_povm_k(d, ket_k)
        list_of_prob.append(prob_per_povm)

    for ket_k in range(d):
        for ket_l in range(ket_k + 1, d):
            # calculate state_vec and povm_vecs
            density_matrix = np.zeros((d, d), dtype=np.complex128)
            density_matrix[ket_k, ket_k] = 1 / 2
            density_matrix[ket_k, ket_l] = 1 / 2
            density_matrix[ket_l, ket_k] = 1 / 2
            density_matrix[ket_l, ket_l] = 1 / 2

            state_vec, povm_vecs = _calculate_vecs(d, density_matrix)
            list_of_state_vec.append(state_vec)
            list_of_povm_vecs.append(povm_vecs)

            # probability distribution (=expected)
            prob_per_povm = _calculate_probability_of_povm_k_plus_l(d, ket_k, ket_l)
            list_of_prob.append(prob_per_povm)

    for ket_k in range(d):
        for ket_l in range(ket_k + 1, d):
            # calculate state_vec and povm_vecs
            density_matrix = np.zeros((d, d), dtype=np.complex128)
            density_matrix[ket_k, ket_k] = 1 / 2
            density_matrix[ket_k, ket_l] = -1j / 2
            density_matrix[ket_l, ket_k] = 1j / 2
            density_matrix[ket_l, ket_l] = 1 / 2

            state_vec, povm_vecs = _calculate_vecs(d, density_matrix)
            list_of_state_vec.append(state_vec)
            list_of_povm_vecs.append(povm_vecs)

            # probability distribution (=expected)
            prob_per_povm = _calculate_probability_of_povm_k_plus_il(d, ket_k, ket_l)
            list_of_prob.append(prob_per_povm)

    # Act
    for beta, povm_vecs in enumerate(list_of_povm_vecs):
        prob_per_povm = list_of_prob[beta]
        for alpha, state_vec in enumerate(list_of_state_vec):
            state = State(c_sys, state_vec)
            povm = Povm(c_sys, povm_vecs)
            actual = compose_qoperations(povm, gate, state)

            # Assert
            expected = prob_per_povm[alpha]
            npt.assert_almost_equal(actual.ps, expected, decimal=14)


def test_tensor_product_StateEnsemble_StateEnsemble_shape_2_3():
    # Arrange
    c_sys_1q_0 = generate_composite_system(mode="qubit", num=1, ids_esys=[0])
    c_sys_1q_1 = generate_composite_system(mode="qubit", num=1, ids_esys=[1])
    c_sys_1q_2 = generate_composite_system(mode="qubit", num=1, ids_esys=[2])
    c_sys_1q_3 = generate_composite_system(mode="qubit", num=1, ids_esys=[3])
    c_sys_1q_4 = generate_composite_system(mode="qubit", num=1, ids_esys=[4])

    state_z0 = generate_qoperation_object(
        mode="state", object_name="state", name="z0", c_sys=c_sys_1q_0
    )
    state_z1 = generate_qoperation_object(
        mode="state", object_name="state", name="z1", c_sys=c_sys_1q_1
    )
    state_y0 = generate_qoperation_object(
        mode="state", object_name="state", name="y0", c_sys=c_sys_1q_2
    )
    state_y1 = generate_qoperation_object(
        mode="state", object_name="state", name="y1", c_sys=c_sys_1q_3
    )
    state_x0 = generate_qoperation_object(
        mode="state", object_name="state", name="x0", c_sys=c_sys_1q_4
    )
    prob_dist = MultinomialDistribution(ps=np.array([0.1, 0.9]), shape=(2,))
    se_0 = StateEnsemble([state_z0, state_z1], prob_dist)
    prob_dist = MultinomialDistribution(ps=np.array([0.05, 0.25, 0.7]), shape=(3,))
    se_1 = StateEnsemble([state_y0, state_y1, state_x0], prob_dist)

    # Act
    actual = tensor_product(se_0, se_1)

    # Assert
    expected = np.array([0.005, 0.025, 0.07, 0.045, 0.225, 0.63])
    npt.assert_almost_equal(actual.prob_dist.ps, expected, decimal=16)

    expected = [
        tensor_product(state_z0, state_y0),
        tensor_product(state_z0, state_y1),
        tensor_product(state_z0, state_x0),
        tensor_product(state_z1, state_y0),
        tensor_product(state_z1, state_y1),
        tensor_product(state_z1, state_x0),
    ]
    assert len(actual.states) == len(expected)
    for a, e in zip(actual.states, expected):
        npt.assert_almost_equal(a.vec, e.vec, decimal=15)
    assert actual.prob_dist.shape == (2, 3)


def test_tensor_product_State_StateEnsemble():
    # State (x) StateEnsemble
    c_sys_list = []
    for i in range(3):
        c_sys_list.append(generate_composite_system(mode="qubit", num=1, ids_esys=[i]))

    state_z0 = generate_qoperation_object(
        mode="state", object_name="state", name="z0", c_sys=c_sys_list[0]
    )
    state_z1 = generate_qoperation_object(
        mode="state", object_name="state", name="z1", c_sys=c_sys_list[1]
    )
    state_y0 = generate_qoperation_object(
        mode="state", object_name="state", name="y0", c_sys=c_sys_list[2]
    )

    states = [state_z0, state_z1]
    ps = np.array([0.1, 0.9])
    prob_dist = MultinomialDistribution(ps=ps, shape=(2,))
    state_ensemble = StateEnsemble(states=states, prob_dist=prob_dist)

    # Act
    actual = tensor_product(state_y0, state_ensemble)

    # Assert
    # check prob_dist
    expected_prob_dist = state_ensemble.prob_dist
    npt.assert_almost_equal(actual.prob_dist.ps, expected_prob_dist.ps, decimal=15)
    assert expected_prob_dist.shape == (2,)

    # check state
    expected_state = tensor_product(state_y0, state_z0)
    npt.assert_almost_equal(actual.states[0].vec, expected_state.vec)

    expected_state = tensor_product(state_y0, state_z1)
    npt.assert_almost_equal(actual.states[1].vec, expected_state.vec)


def test_tensor_product_StateEnsemble_State():
    # StateEnsemble (x)
    c_sys_list = []
    for i in range(3):
        c_sys_list.append(generate_composite_system(mode="qubit", num=1, ids_esys=[i]))

    state_z0 = generate_qoperation_object(
        mode="state", object_name="state", name="z0", c_sys=c_sys_list[0]
    )
    state_z1 = generate_qoperation_object(
        mode="state", object_name="state", name="z1", c_sys=c_sys_list[1]
    )
    state_y0 = generate_qoperation_object(
        mode="state", object_name="state", name="y0", c_sys=c_sys_list[2]
    )

    states = [state_z0, state_z1]
    ps = np.array([0.1, 0.9])
    prob_dist = MultinomialDistribution(ps=ps, shape=(2,))
    state_ensemble = StateEnsemble(states=states, prob_dist=prob_dist)

    # Act
    actual = tensor_product(state_ensemble, state_y0)

    # Assert
    # check prob_dist
    expected_prob_dist = state_ensemble.prob_dist
    npt.assert_almost_equal(actual.prob_dist.ps, expected_prob_dist.ps, decimal=15)
    assert expected_prob_dist.shape == (2,)

    # check state
    expected_state = tensor_product(state_z0, state_y0)
    npt.assert_almost_equal(actual.states[0].vec, expected_state.vec)

    expected_state = tensor_product(state_z1, state_y0)
    npt.assert_almost_equal(actual.states[1].vec, expected_state.vec)


def test_composite_qoperations_Gate_StateEnsemble():
    # (Gate, StateEnsemble)
    # Arrange
    c_sys_1q = generate_composite_system(mode="qubit", num=1, ids_esys=[101])

    # StateEnsemble
    state_z0 = generate_qoperation_object(
        mode="state", object_name="state", name="z0", c_sys=c_sys_1q
    )
    state_z1 = generate_qoperation_object(
        mode="state", object_name="state", name="z1", c_sys=c_sys_1q
    )
    states = [state_z0, state_z1]
    ps = np.array([0.1, 0.9])
    prob_dist = MultinomialDistribution(ps=ps, shape=(2,))
    state_ensemble = StateEnsemble(states=states, prob_dist=prob_dist)

    # Gate
    gate_z = generate_qoperation_object(
        mode="gate", object_name="gate", name="x", c_sys=c_sys_1q
    )

    # Act
    actual = compose_qoperations(gate_z, state_ensemble)

    # Assert
    # check prob_dist
    expected_prob_dist = state_ensemble.prob_dist
    npt.assert_almost_equal(actual.prob_dist.ps, expected_prob_dist.ps, decimal=15)
    assert expected_prob_dist.shape == (2,)

    expected_state = compose_qoperations(gate_z, state_z0)
    npt.assert_almost_equal(actual.states[0].vec, expected_state.vec)

    expected_state = compose_qoperations(gate_z, state_z1)
    npt.assert_almost_equal(actual.states[1].vec, expected_state.vec)


def test_composite_qoperations_Povm_StateEnsemble():
    # (Povm, StateEnsemble)
    c_sys_1q = generate_composite_system(mode="qubit", num=1, ids_esys=[101])

    # StateEnsemble
    state_z0 = generate_qoperation_object(
        mode="state", object_name="state", name="z0", c_sys=c_sys_1q
    )
    state_z1 = generate_qoperation_object(
        mode="state", object_name="state", name="z1", c_sys=c_sys_1q
    )
    state_y0 = generate_qoperation_object(
        mode="state", object_name="state", name="y0", c_sys=c_sys_1q
    )
    states = [state_z0, state_z1, state_y0]
    ps = np.array([0.15, 0.35, 0.5])
    prob_dist = MultinomialDistribution(ps=ps, shape=(3,))
    state_ensemble = StateEnsemble(states=states, prob_dist=prob_dist)

    povm_z = generate_qoperation_object(
        mode="povm", object_name="povm", name="z", c_sys=c_sys_1q
    )
    actual = compose_qoperations(povm_z, state_ensemble)

    # Assert
    # 0.15 * [1, 0], 0.35 * [0, 1], 0.5 * [0.5, 0.5]
    expected_ps = np.array([0.15, 0] + [0, 0.35] + [0.25, 0.25])
    expected_shape = (3, 2)

    npt.assert_almost_equal(actual.ps, expected_ps, decimal=15)
    assert actual.shape == expected_shape


def make_state_ensemble_2_3_different_c_sys():
    c_sys_list = []
    for i in range(6):
        c_sys_list.append(generate_composite_system(mode="qubit", num=1, ids_esys=[i]))

    state_z0 = generate_qoperation_object(
        mode="state", object_name="state", name="z0", c_sys=c_sys_list[0]
    )
    state_z1 = generate_qoperation_object(
        mode="state", object_name="state", name="z1", c_sys=c_sys_list[1]
    )
    state_y0 = generate_qoperation_object(
        mode="state", object_name="state", name="y0", c_sys=c_sys_list[2]
    )
    state_y1 = generate_qoperation_object(
        mode="state", object_name="state", name="y1", c_sys=c_sys_list[3]
    )
    state_x0 = generate_qoperation_object(
        mode="state", object_name="state", name="x0", c_sys=c_sys_list[4]
    )
    state_x1 = generate_qoperation_object(
        mode="state", object_name="state", name="x0", c_sys=c_sys_list[5]
    )
    states = [state_z0, state_z1, state_y0, state_y1, state_x0, state_x1]

    ps = np.array([0.005, 0.025, 0.07, 0.045, 0.225, 0.63])
    prob_dist = MultinomialDistribution(ps=ps, shape=(2, 3))

    state_ensemble = StateEnsemble(states=states, prob_dist=prob_dist)
    return state_ensemble


def make_state_ensemble_2_3_same_c_sys():
    c_sys_1q = generate_composite_system(mode="qubit", num=1, ids_esys=[101])
    names = ["z0", "z1", "y0", "y1", "x0", "x1"]
    states = [
        generate_qoperation_object(
            mode="state", object_name="state", name=name, c_sys=c_sys_1q
        )
        for name in names
    ]

    ps = np.array([0.005, 0.025, 0.07, 0.045, 0.225, 0.63])
    prob_dist = MultinomialDistribution(ps=ps, shape=(2, 3))

    state_ensemble = StateEnsemble(states=states, prob_dist=prob_dist)
    return state_ensemble


def test_tensor_prodcut_State_StateEnsemble_multi_dimension():
    # Arrange
    c_sys = generate_composite_system(mode="qubit", num=1, ids_esys=[100])
    state_ensemble = make_state_ensemble_2_3_different_c_sys()
    state_a = generate_qoperation_object(
        mode="state", object_name="state", name="a", c_sys=c_sys
    )
    # Act
    actual = tensor_product(state_a, state_ensemble)

    # Assert
    assert actual.prob_dist.shape == (2, 3)


def test_tensor_prodcut_StateEnsemble_State_multi_dimension():
    # Arrange
    c_sys = generate_composite_system(mode="qubit", num=1, ids_esys=[100])
    state_ensemble = make_state_ensemble_2_3_different_c_sys()
    state_a = generate_qoperation_object(
        mode="state", object_name="state", name="a", c_sys=c_sys
    )
    actual = tensor_product(state_ensemble, state_a)

    assert actual.prob_dist.shape == (2, 3)


def test_compose_qoperations_Gate_StateEnsemble_multi_dimension():
    # Arrange
    state_ensemble = make_state_ensemble_2_3_same_c_sys()
    c_sys = state_ensemble.states[0].composite_system
    gate_z = generate_qoperation_object(
        mode="gate", object_name="gate", name="z", c_sys=c_sys
    )

    actual = compose_qoperations(gate_z, state_ensemble)

    assert actual.prob_dist.shape == (2, 3)


def test_compose_qoperations_Povm_MProcess():
    # Arrange
    e_sys = ElementalSystem(0, matrix_basis.get_normalized_pauli_basis())
    c_sys = CompositeSystem([e_sys])
    mprocess_z = generate_mprocess_from_name(c_sys, "z-type1")
    povm_z = generate_povm_from_name("z", c_sys)

    # Act
    actual = compose_qoperations(povm_z, mprocess_z)

    # Assert
    assert len(actual.vecs) == 4
    expected_z0 = np.array([1, 0, 0, 1], dtype=np.float64) / np.sqrt(2)
    expected_z1 = np.array([1, 0, 0, -1], dtype=np.float64) / np.sqrt(2)
    npt.assert_almost_equal(actual.vecs[0], expected_z0, decimal=15)
    npt.assert_almost_equal(actual.vecs[1], np.zeros(4), decimal=15)
    npt.assert_almost_equal(actual.vecs[2], np.zeros(4), decimal=15)
    npt.assert_almost_equal(actual.vecs[3], expected_z1, decimal=15)


def test_compose_qoperations_Povm_StateEnsemble():
    # Arrange
    c_sys_1q = generate_composite_system(mode="qubit", num=1, ids_esys=[0])
    # Case 1
    povm = generate_qoperation_object(
        mode="povm", object_name="povm", name="z", c_sys=c_sys_1q
    )
    state_ens = generate_qoperation_object(
        mode="state_ensemble", object_name="state_ensemble", name="z0", c_sys=c_sys_1q
    )

    # Act
    actual = compose_qoperations(povm, state_ens)
    # Assert
    expected = np.array([1, 0, 0, 0])
    npt.assert_almost_equal(actual.ps, expected, decimal=15)
    assert actual.shape == (2, 2)

    # Case 2:
    povm = generate_qoperation_object(
        mode="povm", object_name="povm", name="x", c_sys=c_sys_1q
    )
    state_0 = State(
        vec=np.array(
            [1 / np.sqrt(2), 1 / np.sqrt(2), 0.0, 0.0],
            dtype=np.float64,
        ),
        c_sys=c_sys_1q,
    )
    state_1 = State(
        vec=np.array(
            [
                0,
                0,
                0,
                0,
            ],
            dtype=np.float64,
        ),
        is_physicality_required=False,
        c_sys=c_sys_1q,
    )
    ps = MultinomialDistribution(ps=[1, 0])
    state_ens = StateEnsemble(states=[state_0, state_1], prob_dist=ps)

    # Act
    actual = compose_qoperations(povm, state_ens)
    # Assert
    expected = np.array([1, 0, 0, 0])
    npt.assert_almost_equal(actual.ps, expected, decimal=15)
    assert actual.shape == (2, 2)


def test_compose_qoperations_Povm_StateEnsemble_multidimension():
    # Arrange
    c_sys_1q = generate_composite_system(mode="qubit", num=1, ids_esys=[101])
    # StateEnsemble
    names = ["z0", "z1", "y0", "y1", "x0", "x1"]
    states = [
        generate_qoperation_object(
            mode="state", object_name="state", name=name, c_sys=c_sys_1q
        )
        for name in names
    ]

    ps = np.array([0.005, 0.025, 0.07, 0.045, 0.225, 0.63])
    prob_dist = MultinomialDistribution(ps=ps, shape=(2, 3))
    state_ensemble = StateEnsemble(states=states, prob_dist=prob_dist)
    # Povm
    povm_z = generate_qoperation_object(
        mode="povm", object_name="povm", name="z", c_sys=c_sys_1q
    )
    # Act
    actual = compose_qoperations(povm_z, state_ensemble)
    # Assert
    expected = np.array(
        [
            [[0.005, 0], [0, 0.025], [0.035, 0.035]],
            [[0.0225, 0.0225], [0.1125, 0.1125], [0.315, 0.315]],
        ]
    )
    npt.assert_almost_equal(actual.ps, expected.flatten(), decimal=15)
    assert actual.shape == (2, 3, 2)
