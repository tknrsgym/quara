import numpy as np
import numpy.testing as npt
import pytest

from quara.objects.matrix_basis import (
    get_normalized_pauli_basis,
)
from quara.objects.gate_typical import (
    get_gate_names_1qubit,
    generate_unitary_mat_from_gate_name,
    calc_gate_mat_from_unitary_mat,
    calc_gate_mat_from_unitary_mat_with_hermitian_basis,
    generate_gate_mat_from_gate_name,
)


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
