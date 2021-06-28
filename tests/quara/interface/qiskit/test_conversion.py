import numpy as np
import numpy.testing as npt
import pytest

from qiskit.ignis.verification.tomography.basis.paulibasis import (
    pauli_preparation_matrix,
)

from quara.interface.qiskit.conversion import (
    convert_state_qiskit_to_quara,
    convert_state_quara_to_qiskit,
    convert_povm_qiskit_to_quara,
    convert_povm_quara_to_qiskit,
    convert_empi_dists_qiskit_to_quara,
    convert_empi_dists_quara_to_qiskit,
    convert_gate_qiskit_to_quara,
    convert_gate_quara_to_qiskit,
)
from quara.interface.qiskit.qiskit_state_typical import (
    get_qiskit_state_names_1qubit,
    get_qiskit_state_names_2qubit,
    get_qiskit_state_names_3qubit,
)
from quara.objects.state import State
from quara.objects.povm import Povm
from quara.objects.gate import Gate
from quara.objects.composite_system import CompositeSystem
from quara.objects.composite_system_typical import generate_composite_system
from quara.objects.state_typical import generate_state_from_name


def _test_convert_state_qiskit_to_quara(mode, num, state_name):
    # Arrange
    c_sys = generate_composite_system(mode, num)
    expected = generate_state_from_name(c_sys, state_name)
    # density matrix
    source = pauli_preparation_matrix(state_name)
    actual = convert_state_qiskit_to_quara(source, c_sys)
    npt.assert_array_almost_equal(
        actual.to_density_matrix, expected.to_density_matrix, decimal=10
    )


@pytest.mark.qiskit
@pytest.mark.onequbit
@pytest.mark.parametrize(
    ("state_name"),
    [(state_name) for state_name in get_qiskit_state_names_1qubit()],
)
def test_convert_state_qiskit_to_quara_1qubit(state_name):
    _test_convert_state_qiskit_to_quara("qubit", 1, state_name)


def _test_convert_state_quara_to_qiskit(mode, num, state_name):
    # Arrange
    c_sys = generate_composite_system(mode, num)
    expected = pauli_preparation_matrix(state_name)

    source = generate_state_from_name(state_name)
    actual = convert_state_quara_to_qiskit(source)
    npt.assert_almost_equal(actual, expected, dicimal=10)


@pytest.mark.qiskit
@pytest.mark.onequbit
@pytest.mark.parametrize(
    ("state_name"),
    [(state_name) for state_name in get_qiskit_state_names_1qubit()],
)
def test_convert_state_quara_to_qiskit_1qubit(state_name):
    _test_convert_state_quara_to_qiskit("qubit", 1, state_name)
