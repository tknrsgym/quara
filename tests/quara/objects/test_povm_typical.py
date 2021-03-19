import itertools
import os
from pathlib import Path
import copy
from typing import List

import numpy as np
import numpy.testing as npt
import pytest

from quara.objects.composite_system import CompositeSystem
from quara.objects.elemental_system import ElementalSystem
from quara.objects.matrix_basis import (
    get_comp_basis,
    get_gell_mann_basis,
    get_normalized_pauli_basis,
    get_pauli_basis,
    convert_vec,
)
from quara.objects.operators import tensor_product
from quara.objects.povm import (
    Povm,
    convert_var_index_to_povm_index,
    convert_povm_index_to_var_index,
    convert_var_to_povm,
    convert_povm_to_var,
    calc_gradient_from_povm,
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
from quara.objects import povm_typical

from quara.objects.state import get_x0_1q
from quara.settings import Settings


def _test_povm(
    gate_name: str,
    dims: List[int] = [],
    ids: List[int] = [],
    c_sys: CompositeSystem = None,
    decimal: int = 15,
):
    pass
    """
    # Arrange
    object_name = "effective_lindbladian_class"
    el = generate_gate_object_from_gate_name_object_name(
        gate_name, object_name, dims, ids, c_sys
    )
    g_from_el = el.to_gate()

    object_name = "gate_class"
    g = generate_gate_object_from_gate_name_object_name(
        gate_name, object_name, dims, ids, c_sys
    )

    # Act
    actual = g.hs

    # Assert
    expected = g_from_el.hs
    npt.assert_almost_equal(actual, expected, decimal=decimal)
    """


@pytest.mark.onequbit
@pytest.mark.parametrize(
    ("povm_name"), [(povm_name) for povm_name in povm_typical.get_povm_names_1qubit()],
)
def test_calc_povm_vecs_from_matrices_with_hermitian_basis_1qubit(povm_name: str,):
    # Arrange
    basis = get_normalized_pauli_basis()
    matrices = povm_typical.generate_povm_matrices_from_povm_name(povm_name)
    vecs_complex = povm_typical.calc_povm_vecs_from_matrices(matrices, basis)
    vecs_truncate = povm_typical.calc_povm_vecs_from_matrices_with_hermitian_basis(
        matrices, basis
    )

    # Act
    actual = vecs_truncate

    # Assert
    expected = povm_typical.generate_povm_vecs_from_povm_name(povm_name)
    npt.assert_almost_equal(actual, expected, decimal=15)
    npt.assert_almost_equal(vecs_complex, expected, decimal=15)
    print(f"matrices={matrices}")
    print(f"vecs_complex={vecs_complex}")
    print(f"vecs_truncate={vecs_truncate}")
    print(f"expected={expected}")
    assert False


@pytest.mark.onequbit
@pytest.mark.parametrize(
    ("povm_name"), [(povm_name) for povm_name in povm_typical.get_povm_names_1qubit()],
)
def test_povm_1qubit(povm_name: str):
    # Arrange
    e_sys = ElementalSystem(0, get_normalized_pauli_basis())
    c_sys = CompositeSystem([e_sys])
    dims = [2]
    ids = []

    _test_povm(povm_name, dims, ids, c_sys)
