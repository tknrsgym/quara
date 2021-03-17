import numpy as np
import numpy.testing as npt
import pytest

from typing import List

from quara.objects import matrix_basis
from quara.objects.matrix_basis import (
    get_normalized_pauli_basis,
)
from quara.objects.elemental_system import ElementalSystem
from quara.objects.composite_system import CompositeSystem
from quara.objects.qoperation_typical import (
    generate_gate_object_from_gate_name_object_name,
)
from quara.objects.gate_typical import (
    get_gate_names_1qubit,
    get_gate_names_2qubit,
    get_gate_names_2qubit_asymmetric,
    generate_unitary_mat_from_gate_name,
    calc_gate_mat_from_unitary_mat,
    calc_gate_mat_from_unitary_mat_with_hermitian_basis,
    generate_gate_mat_from_gate_name,
)


def _test_gate(
    gate_name: str,
    dims: List[int] = [],
    ids: List[int] = [],
    c_sys: CompositeSystem = None,
    decimal: int = 15,
):
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


@pytest.mark.onequbit
@pytest.mark.parametrize(
    ("gate_name"),
    [(gate_name) for gate_name in get_gate_names_1qubit()],
)
def test_calc_gate_mat_from_unitary_mat_with_hermitian_basis_1qubit_case01(
    gate_name: str,
):
    # Arrange
    b = get_normalized_pauli_basis()
    u = generate_unitary_mat_from_gate_name(gate_name)
    G_complex = calc_gate_mat_from_unitary_mat(u, b)
    G_truncate = calc_gate_mat_from_unitary_mat_with_hermitian_basis(u, b)

    # Act
    actual = G_truncate

    # Assert
    expected = generate_gate_mat_from_gate_name(gate_name)
    npt.assert_almost_equal(actual, expected, decimal=15)


@pytest.mark.onequbit
@pytest.mark.parametrize(
    ("gate_name"),
    [(gate_name) for gate_name in get_gate_names_1qubit()],
)
def test_gate_1qubit_case01(gate_name: str):
    # Arrange
    e_sys = ElementalSystem(0, matrix_basis.get_normalized_pauli_basis())
    c_sys = CompositeSystem([e_sys])
    dims = [2]
    ids = []

    _test_gate(gate_name, dims, ids, c_sys)


@pytest.mark.twoqubit
@pytest.mark.parametrize(
    ("gate_name", "decimal"),
    [(gate_name, 15) for gate_name in get_gate_names_2qubit()],
)
def test_gate_2qubit_case01(gate_name: str, decimal: int):
    # Arrange
    e_sys0 = ElementalSystem(0, matrix_basis.get_normalized_pauli_basis())
    e_sys1 = ElementalSystem(1, matrix_basis.get_normalized_pauli_basis())
    c_sys = CompositeSystem([e_sys0, e_sys1])
    dims = [2, 2]

    ids = [0, 1]
    _test_gate(gate_name=gate_name, dims=dims, ids=ids, c_sys=c_sys, decimal=decimal)

    if gate_name in get_gate_names_2qubit_asymmetric():
        ids = [1, 0]
        _test_gate(
            gate_name=gate_name, dims=dims, ids=ids, c_sys=c_sys, decimal=decimal
        )
