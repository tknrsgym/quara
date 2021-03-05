import numpy as np
import numpy.testing as npt
import pytest

from typing import List
from scipy.linalg import expm

from quara.math.matrix import (
    project_to_traceless_matrix,
)
from quara.objects import matrix_basis
from quara.objects.matrix_basis import MatrixBasis
from quara.objects.composite_system import CompositeSystem
from quara.objects.elemental_system import ElementalSystem
from quara.objects.gate import Gate
from quara.objects.effective_lindbladian import EffectiveLindbladian
from quara.objects.effective_lindbladian import (
    generate_effective_lindbladian_from_h,
)
from quara.objects.gate_typical import (
    get_gate_names_1qubit,
)
from quara.objects.qoperation_typical import (
    get_object_names,
    generate_gate_object_from_gate_name_object_name,
)


def _test_validity_hamiltonian_vec_hamiltonian_mat(
    gate_name: str,
    dims: List[int] = [],
    ids: List[int] = [],
    c_sys: CompositeSystem = None,
):
    # Arrange
    object_name = "hamiltonian_vec"
    h_vec = generate_gate_object_from_gate_name_object_name(
        gate_name, object_name, dims, ids, c_sys
    )
    object_name = "hamiltonian_mat"
    h_mat = generate_gate_object_from_gate_name_object_name(
        gate_name, object_name, dims, ids, c_sys
    )

    dim_sys = c_sys.dim
    b = c_sys.basis()
    if dim_sys * dim_sys != h_vec.size:
        raise ValueError(f"dimensions of c_sys and h_vec are inconsistent.")
    h_mat_from_vec = np.zeros((dim_sys, dim_sys), dtype=np.complex128)
    for i, bi in enumerate(b):
        h_mat_from_vec += h_vec[i] * bi

    # Act
    actual = h_mat_from_vec

    # Assert
    expected = h_mat
    # The case of decimal=16 below returns AssertionError.
    npt.assert_almost_equal(actual, expected, decimal=15)


def _test_validity_hamiltonian_mat_unitary_mat(
    gate_name: str,
    dims: List[int] = [],
    ids: List[int] = [],
    c_sys: CompositeSystem = None,
):
    # Arrange
    object_name = "hamiltonian_mat"
    h_mat = generate_gate_object_from_gate_name_object_name(
        gate_name, object_name, dims, ids, c_sys
    )
    object_name = "unitary_mat"
    u_mat = generate_gate_object_from_gate_name_object_name(
        gate_name, object_name, dims, ids, c_sys
    )

    # Act
    actual = expm(-1j * h_mat)

    # Assert
    expected = u_mat
    npt.assert_almost_equal(actual, expected, decimal=15)


def _test_validity_effective_lindladian_mat_gate_mat(
    gate_name: str,
    dims: List[int] = [],
    ids: List[int] = [],
    c_sys: CompositeSystem = None,
):
    # Arrange
    object_name = "effective_lindbladian_mat"
    el_mat = generate_gate_object_from_gate_name_object_name(
        gate_name, object_name, dims, ids, c_sys
    )
    object_name = "gate_mat"
    g_mat = generate_gate_object_from_gate_name_object_name(
        gate_name, object_name, dims, ids, c_sys
    )

    # Act
    actual = expm(el_mat)
    # Assert
    expected = g_mat
    npt.assert_almost_equal(actual, expected, decimal=15)


def _test_generate_effective_lindbladian_from_h(
    gate_name: str,
    dims: List[int] = [],
    ids: List[int] = [],
    c_sys: CompositeSystem = None,
):
    # Arrange
    object_name = "hamiltonian_mat"
    h_mat = generate_gate_object_from_gate_name_object_name(
        gate_name, object_name, dims, ids, c_sys
    )
    object_name = "effective_lindbladian_mat"
    el_mat = generate_gate_object_from_gate_name_object_name(
        gate_name, object_name, dims, ids, c_sys
    )

    el_from_h = generate_effective_lindbladian_from_h(c_sys, h_mat)
    # Act
    actual = el_from_h.hs
    # Assert
    expected = el_mat
    npt.assert_almost_equal(actual, expected, decimal=15)


def _test_calc_h(
    gate_name: str,
    dims: List[int] = [],
    ids: List[int] = [],
    c_sys: CompositeSystem = None,
):
    # Arrange
    object_name = "hamiltonian_mat"
    h_mat = generate_gate_object_from_gate_name_object_name(
        gate_name, object_name, dims, ids, c_sys
    )

    el_from_h = generate_effective_lindbladian_from_h(c_sys, h_mat)
    # Act
    actual = project_to_traceless_matrix(el_from_h.calc_h_mat())
    # Assert
    expected = project_to_traceless_matrix(h_mat)
    npt.assert_almost_equal(actual, expected, decimal=15)


# Tests for 1-qubit system


@pytest.mark.parametrize(
    ("gate_name"),
    [(gate_name) for gate_name in get_gate_names_1qubit()],
)
def test_validity_hamiltonian_vec_hamiltonian_mat_1qubit(gate_name):
    # Arrange
    e_sys = ElementalSystem(0, matrix_basis.get_normalized_pauli_basis())
    c_sys = CompositeSystem([e_sys])
    dims = [2]
    ids = []

    _test_validity_hamiltonian_vec_hamiltonian_mat(
        gate_name=gate_name, dims=dims, ids=ids, c_sys=c_sys
    )


@pytest.mark.parametrize(
    ("gate_name"),
    [(gate_name) for gate_name in get_gate_names_1qubit()],
)
def test_validity_hamiltonian_mat_unitary_mat_1qubit(gate_name):
    # Arrange
    e_sys = ElementalSystem(0, matrix_basis.get_normalized_pauli_basis())
    c_sys = CompositeSystem([e_sys])
    dims = [2]
    ids = []

    _test_validity_hamiltonian_mat_unitary_mat(
        gate_name=gate_name, dims=dims, ids=ids, c_sys=c_sys
    )


@pytest.mark.parametrize(
    ("gate_name"),
    [(gate_name) for gate_name in get_gate_names_1qubit()],
)
def test_validity_effective_lindladian_mat_gate_mat_1qubit(gate_name):
    # Arrange
    e_sys = ElementalSystem(0, matrix_basis.get_normalized_pauli_basis())
    c_sys = CompositeSystem([e_sys])
    dims = [2]
    ids = []

    _test_validity_effective_lindladian_mat_gate_mat(
        gate_name=gate_name, dims=dims, ids=ids, c_sys=c_sys
    )


@pytest.mark.parametrize(
    ("gate_name"),
    [(gate_name) for gate_name in get_gate_names_1qubit()],
)
def test_generate_effective_lindbladian_from_h_1qubit(gate_name):
    # Arrange
    e_sys = ElementalSystem(0, matrix_basis.get_normalized_pauli_basis())
    c_sys = CompositeSystem([e_sys])
    dims = [2]
    ids = []

    _test_generate_effective_lindbladian_from_h(
        gate_name=gate_name, dims=dims, ids=ids, c_sys=c_sys
    )


@pytest.mark.parametrize(
    ("gate_name"),
    [(gate_name) for gate_name in get_gate_names_1qubit()],
)
def test_calc_h_1qubit(gate_name):
    # Arrange
    e_sys = ElementalSystem(0, matrix_basis.get_normalized_pauli_basis())
    c_sys = CompositeSystem([e_sys])
    dims = [2]
    ids = []

    _test_calc_h(gate_name=gate_name, dims=dims, ids=ids, c_sys=c_sys)
