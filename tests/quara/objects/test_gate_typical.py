import numpy as np
import numpy.testing as npt
import pytest

from typing import List
from itertools import product, permutations

from quara.objects import matrix_basis
from quara.objects.matrix_basis import (
    get_normalized_pauli_basis,
)
from quara.objects.elemental_system import ElementalSystem
from quara.objects.composite_system import CompositeSystem
from quara.objects.qoperation_typical import (
    generate_gate_object,
    generate_effective_lindbladian_object,
)
from quara.objects.gate_typical import (
    get_gate_names_1qubit,
    get_gate_names_2qubit,
    get_gate_names_2qubit_asymmetric,
    get_gate_names_3qubit,
    get_gate_names_1qutrit_single_gellmann,
    generate_unitary_mat_from_gate_name,
    calc_gate_mat_from_unitary_mat,
    calc_gate_mat_from_unitary_mat_with_hermitian_basis,
    generate_gate_mat_from_gate_name,
    calc_quadrant_from_pauli_symbol,
    calc_decimal_number_from_pauli_symbol,
    calc_pauli_symbol_from_decimal_number,
    calc_hamiltonian_mat_from_gate_name_2qutrit_base_matrices,
)
from quara.objects.effective_lindbladian_typical import (
    generate_gate_1qutrit_single_gellmann_effective_linabladian,
)


def _test_gate(
    gate_name: str,
    dims: List[int] = None,
    ids: List[int] = None,
    c_sys: CompositeSystem = None,
    decimal: int = 15,
):
    # Arrange
    object_name = "effective_lindbladian"
    el = generate_effective_lindbladian_object(gate_name, object_name, dims, ids, c_sys)
    g_from_el = el.to_gate()

    object_name = "gate"
    g = generate_gate_object(gate_name, object_name, dims, ids, c_sys)

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


@pytest.mark.threequbit
@pytest.mark.parametrize(
    ("gate_name", "decimal"),
    [(gate_name, 15) for gate_name in get_gate_names_3qubit()],
)
def test_gate_3qubit_case01(gate_name: str, decimal: int):
    # Arrange
    e_sys0 = ElementalSystem(0, matrix_basis.get_normalized_pauli_basis())
    e_sys1 = ElementalSystem(1, matrix_basis.get_normalized_pauli_basis())
    e_sys2 = ElementalSystem(2, matrix_basis.get_normalized_pauli_basis())
    c_sys = CompositeSystem([e_sys0, e_sys1, e_sys2])
    dims = [2, 2, 2]

    ids_base = [0, 1, 2]
    for ids in permutations(ids_base):
        _test_gate(
            gate_name=gate_name, dims=dims, ids=ids, c_sys=c_sys, decimal=decimal
        )


@pytest.mark.onequtrit
@pytest.mark.parametrize(
    ("gate_name"),
    [(gate_name) for gate_name in get_gate_names_1qutrit_single_gellmann()],
)
def test_gate_1qutrit_case01(gate_name: str):
    # Arrange
    e_sys = ElementalSystem(0, matrix_basis.get_normalized_gell_mann_basis())
    c_sys = CompositeSystem([e_sys])
    dims = [3]
    ids = []
    _test_gate(gate_name, dims, ids, c_sys)


@pytest.mark.parametrize(
    ("num_qubit"),
    [(1), (2), (3)],
)
def test_calc_decimal_number_from_pauli_symbol(num_qubit: int):
    # Arrange
    pauli_symbols = ["i", "x", "y", "z"]
    p = product(pauli_symbols, repeat=num_qubit)
    for i, pi in enumerate(p):
        symbol = ""
        for s in pi:
            symbol = symbol + s
        q = calc_quadrant_from_pauli_symbol(symbol)

        # Act
        n = calc_decimal_number_from_pauli_symbol(symbol)
        actual = n

        # Assert
        expected = i
        npt.assert_equal(actual, expected)


@pytest.mark.parametrize(
    ("num_qubit"),
    [(1), (2), (3)],
)
def test_calc_pauli_symbol_from_decimal_number(num_qubit: int):
    # Arrange
    pauli_symbols = ["i", "x", "y", "z"]
    p = product(pauli_symbols, repeat=num_qubit)
    for i, pi in enumerate(p):
        symbol = ""
        for s in pi:
            symbol = symbol + s
        q = calc_quadrant_from_pauli_symbol(symbol)
        n = calc_decimal_number_from_pauli_symbol(symbol)

        # Act
        actual = symbol

        # Assert
        expected = calc_pauli_symbol_from_decimal_number(n, num_qubit)
        npt.assert_equal(actual, expected)


@pytest.mark.threequbit
def test_generate_unitary_mat_from_gate_name_toffoli():
    # Arrange
    gate_name = "toffoli"
    dims = [2, 2, 2]
    ids = [0, 1, 2]

    # Act
    actual = generate_unitary_mat_from_gate_name(
        gate_name=gate_name, dims=dims, ids=ids
    )

    # Assert
    expected = np.array(
        [
            [1, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 1],
            [0, 0, 0, 0, 0, 0, 1, 0],
        ],
        dtype=np.complex128,
    )
    npt.assert_almost_equal(actual, expected)


@pytest.mark.threequbit
def test_generate_unitary_mat_from_gate_name_fredkin():
    # Arrange
    gate_name = "fredkin"
    dims = [2, 2, 2]
    ids = [0, 1, 2]

    # Act
    actual = generate_unitary_mat_from_gate_name(
        gate_name=gate_name, dims=dims, ids=ids
    )

    # Assert
    expected = np.array(
        [
            [1, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 1],
        ],
        dtype=np.complex128,
    )
    npt.assert_almost_equal(actual, expected)


@pytest.mark.twoqutrit
def test_calc_hamiltonian_mat_from_gate_name_2qutrit_base_matrices_case01():
    # Arrange
    gate_name = "i01x90"

    # Act
    actual = calc_hamiltonian_mat_from_gate_name_2qutrit_base_matrices(gate_name)

    # Expected
    mat0 = np.eye(3, dtype=np.complex128)
    mat1 = np.array([[0, 1, 0], [1, 0, 0], [0, 0, 0]], dtype=np.complex128)
    coeff = 0.25 * np.pi
    expected = coeff * np.kron(mat0, mat1)

    # Assert
    npt.assert_almost_equal(actual, expected)


@pytest.mark.twoqutrit
def test_calc_hamiltonian_mat_from_gate_name_2qutrit_base_matrices_case02():
    # Arrange
    gate_name = "02yi180"

    # Act
    actual = calc_hamiltonian_mat_from_gate_name_2qutrit_base_matrices(gate_name)

    # Expected
    mat0 = np.array([[0, 0, -1j], [0, 0, 0], [1j, 0, 0]], dtype=np.complex128)
    mat1 = np.eye(3, dtype=np.complex128)
    coeff = 0.5 * np.pi
    expected = coeff * np.kron(mat0, mat1)

    # Assert
    npt.assert_almost_equal(actual, expected)


@pytest.mark.twoqutrit
def test_calc_hamiltonian_mat_from_gate_name_2qutrit_base_matrices_case03():
    # Arrange
    gate_name = "12z02y90"

    # Act
    actual = calc_hamiltonian_mat_from_gate_name_2qutrit_base_matrices(gate_name)

    # Expected
    mat0 = np.array([[0, 0, 0], [0, 1, 0], [0, 0, -1]], dtype=np.complex128)
    mat1 = np.array([[0, 0, -1j], [0, 0, 0], [1j, 0, 0]], dtype=np.complex128)
    coeff = 0.25 * np.pi
    expected = coeff * np.kron(mat0, mat1)

    # Assert
    npt.assert_almost_equal(actual, expected)


@pytest.mark.twoqutrit
def test_calc_hamiltonian_mat_from_gate_name_2qutrit_base_matrices_case04():
    # Arrange
    gate_name = "i01x180_12z02y90"

    # Act
    actual = calc_hamiltonian_mat_from_gate_name_2qutrit_base_matrices(gate_name)

    # Expected
    mat0 = np.eye(3, dtype=np.complex128)
    mat1 = np.array([[0, 1, 0], [1, 0, 0], [0, 0, 0]], dtype=np.complex128)
    coeff = 0.5 * np.pi
    h_first = coeff * np.kron(mat0, mat1)

    mat0 = np.array([[0, 0, 0], [0, 1, 0], [0, 0, -1]], dtype=np.complex128)
    mat1 = np.array([[0, 0, -1j], [0, 0, 0], [1j, 0, 0]], dtype=np.complex128)
    coeff = 0.25 * np.pi
    h_second = coeff * np.kron(mat0, mat1)

    expected = h_first + h_second

    # Assert
    npt.assert_almost_equal(actual, expected)


@pytest.mark.twoqutrit
@pytest.mark.parametrize(
    ("gate_name"),
    [
        (gate_name)
        for gate_name in ["i01x90", "02yi180", "12z01y90", "i01y90_02x01z180"]
    ],
)
def test_gate_2qutrit_case01(gate_name: str):
    # Arrange
    e_sys0 = ElementalSystem(0, matrix_basis.get_normalized_gell_mann_basis())
    e_sys1 = ElementalSystem(1, matrix_basis.get_normalized_gell_mann_basis())
    c_sys = CompositeSystem([e_sys0, e_sys1])
    dims = [3, 3]
    ids = []

    # Act & Assert
    _test_gate(gate_name, dims, ids, c_sys)