from numpy.lib import source
from quara.objects.gate_typical import generate_gate_from_gate_name
from quara.objects import povm
from quara.objects.povm_typical import generate_povm_from_name
import numpy as np
import numpy.testing as npt
import pytest

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
    generate_qiskit_state_from_name,
)
from quara.interface.qiskit.qiskit_povm_typical import (
    get_qiskit_povm_names_1qubit,
    get_qiskit_povm_names_2qubit,
    get_qiskit_povm_names_3qubit,
    generate_qiskit_povm_from_povm_name,
)

from quara.interface.qiskit.qiskit_gate_typical import (
    get_qiskit_gate_names_1qubit,
    get_qiskit_gate_names_2qubit,
    get_qiskit_gate_names_3qubit,
    generate_qiskit_gate_from_gate_name,
)

from quara.interface.qiskit.qiskit_empi_dists_typical import (
    get_empi_dists_label,
    get_empi_dists_qiskit,
    get_empi_dists_quara,
    get_empi_dists_shots,
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
    source = generate_qiskit_state_from_name(state_name)
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


@pytest.mark.qiskit
@pytest.mark.twoqubit
@pytest.mark.parametrize(
    ("state_name"), [(state_name) for state_name in get_qiskit_state_names_2qubit]
)
def test_convert_state_qiskit_to_quara_2qubit(state_name):
    _test_convert_state_qiskit_to_quara("qubit", 2, state_name)


@pytest.mark.qiskit
@pytest.mark.threequbit
@pytest.mark.parametrize(
    ("state_name"), [(state_name) for state_name in get_qiskit_state_names_3qubit]
)
def test_convert_state_qiskit_to_quara_3qubit(state_name):
    _test_convert_state_qiskit_to_quara("qubit", 3, state_name)


def _test_convert_state_quara_to_qiskit(mode, num, state_name):
    # Arrange
    c_sys = generate_composite_system(mode, num)
    expected = generate_qiskit_state_from_name(state_name)

    source = generate_state_from_name(c_sys, state_name)
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


@pytest.mark.qiskit
@pytest.mark.twoqubit
@pytest.mark.parametrize(
    ("state_name"), [(state_name) for state_name in get_qiskit_state_names_2qubit]
)
def test_convert_state_quara_to_qiskit_2qubit(state_name):
    _test_convert_state_quara_to_qiskit("qubit", 2, state_name)


@pytest.mark.qiskit
@pytest.mark.threequbit
@pytest.mark.parametrize(
    ("state_name"), [(state_name) for state_name in get_qiskit_state_names_3qubit]
)
def test_convert_state_quara_to_qiksit_3qubit(state_name):
    _test_convert_state_quara_to_qiskit("qubit", 3, state_name)


def _test_convert_povm_qiskit_to_quara(mode, num, povm_name):
    # Arrange
    c_sys = generate_composite_system(mode, num)
    expected = generate_povm_from_name(povm_name, c_sys)

    source = generate_qiskit_povm_from_povm_name(povm_name)
    actual = convert_povm_qiskit_to_quara(source, c_sys)
    npt.assert_almost_equal(actual.matrices, expected.matrices, dicimal=10)


@pytest.mark.qiskit
@pytest.mark.onequbit
@pytest.mark.parametrize(
    ("povm_name"),
    [(povm_name) for povm_name in get_qiskit_povm_names_1qubit()],
)
def test_convert_povm_qiskit_to_quara(povm_name):
    _test_convert_povm_qiskit_to_quara("qubit", 1, povm_name)


@pytest.mark.qiskit
@pytest.mark.twoqubit
@pytest.mark.parametrize(
    ("povm_name"), [(povm_name) for povm_name in get_qiskit_povm_names_2qubit]
)
def test_convert_povm_qiskit_to_quara(povm_name):
    _test_convert_povm_qiskit_to_quara("qubit", 2, povm_name)


@pytest.mark.qiskit
@pytest.mark.threequbit
@pytest.mark.parametrize(
    ("povm_name"), [(povm_name) for povm_name in get_qiskit_povm_names_3qubit]
)
def test_convert_povm_qiskit_to_quara(povm_name):
    _test_convert_povm_qiskit_to_quara("qubit", 3, povm_name)


def _test_convert_povm_quara_to_qiskit(mode, num, povm_name):
    # Arrange
    c_sys = generate_composite_system(mode, num)
    expected = generate_qiskit_povm_from_povm_name(povm_name)

    source = generate_povm_from_name(povm_name, c_sys)
    actual = convert_povm_quara_to_qiskit(source)
    npt.assert_almost_equal(actual, expected, dicimal=10)


@pytest.mark.qiskit
@pytest.mark.onequbit
@pytest.mark.parametrize(
    ("povm_name"),
    [(povm_name) for povm_name in get_qiskit_povm_names_1qubit()],
)
def test_convert_povm_quara_to_qiskit(povm_name):
    _test_convert_povm_quara_to_qiskit("qubit", 1, povm_name)


@pytest.mark.qiskit
@pytest.mark.twoqubit
@pytest.mark.parametrize(
    ("povm_name"), [(povm_name) for povm_name in get_qiskit_povm_names_2qubit]
)
def test_convert_povm_quara_to_qiskit(povm_name):
    _test_convert_povm_quara_to_qiskit("qubit", 2, povm_name)


@pytest.mark.qiskit
@pytest.mark.threequbit
@pytest.mark.parametrize(
    ("povm_name"), [(povm_name) for povm_name in get_qiskit_povm_names_3qubit]
)
def test_convert_povm_quara_to_qiskit(povm_name):
    _test_convert_povm_quara_to_qiskit("qubit", 3, povm_name)


def _test_convert_gate_qiskit_to_quara(mode, num, gate_name):
    c_sys = generate_composite_system(mode, num)
    expected = generate_gate_from_gate_name(gate_name, c_sys)

    source = generate_qiskit_gate_from_gate_name(gate_name)
    actual = convert_gate_qiskit_to_quara(source, c_sys)
    npt.assert_almost_equal(
        actual.to_kraus_matrices, expected.to_kraus_matrices, dicimal=10
    )


@pytest.mark.qiskit
@pytest.mark.onequbit
@pytest.mark.parametrize(
    ("gate_name"),
    [(gate_name) for gate_name in get_qiskit_gate_names_1qubit()],
)
def test_convert_gate_qiskit_to_quara(gate_name):
    _test_convert_gate_qiskit_to_quara("qubit", 1, gate_name)


@pytest.mark.qiskit
@pytest.mark.twoqubit
@pytest.mark.parametrize(
    ("gate_name"), [(gate_name) for gate_name in get_qiskit_gate_names_2qubit]
)
def test_convert_gate_qiskit_to_quara(gate_name):
    _test_convert_gate_qiskit_to_quara("qubit", 2, gate_name)


@pytest.mark.qiskit
@pytest.mark.threequbit
@pytest.mark.parametrize(
    ("gate_name"), [(gate_name) for gate_name in get_qiskit_gate_names_3qubit]
)
def test_convert_gate_qiskit_to_quara(gate_name):
    _test_convert_gate_qiskit_to_quara("qubit", 3, gate_name)


def _test_convert_gate_quara_to_qiskit(mode, num, gate_name):
    c_sys = generate_composite_system(mode, num)
    expected = generate_qiskit_gate_from_gate_name(gate_name)

    source = generate_gate_from_gate_name(gate_name, c_sys)
    actual = convert_gate_quara_to_qiskit(source)
    npt.assert_almost_equal(actual, expected, dicimal=10)


@pytest.mark.qiskit
@pytest.mark.onequbit
@pytest.mark.parametrize(
    ("gate_name"),
    [(gate_name) for gate_name in get_qiskit_gate_names_1qubit()],
)
def test_convert_gate_quara_to_qiskit(gate_name):
    _test_convert_gate_quara_to_qiskit("qubit", 1, gate_name)


@pytest.mark.qiskit
@pytest.mark.twoqubit
@pytest.mark.parametrize(
    ("gate_name"), [(gate_name) for gate_name in get_qiskit_gate_names_2qubit]
)
def test_convert_gate_quara_to_qiskit(gate_name):
    _test_convert_gate_quara_to_qiskit("qubit", 2, gate_name)


@pytest.mark.qiskit
@pytest.mark.threequbit
@pytest.mark.parametrize(
    ("gate_name"), [(gate_name) for gate_name in get_qiskit_gate_names_3qubit]
)
def test_convert_gate_quara_to_qiskit(gate_name):
    _test_convert_gate_quara_to_qiskit("qubit", 3, gate_name)


def _test_convert_empi_dists_qiskit_to_quara(
    empi_dists_quara, empi_dists_qiskit, shots, label
):
    expected = empi_dists_quara
    source = empi_dists_qiskit
    actual = convert_empi_dists_qiskit_to_quara(source, shots, label)
    npt.assert_equal(expected, actual)


@pytest.mark.qiskit
@pytest.mark.parametrize(
    "empi_dists_quara, empi_dists_qiskit, shots, label",
    [
        (
            get_empi_dists_quara,
            get_empi_dists_qiskit,
            get_empi_dists_shots,
            get_empi_dists_label,
        )
    ],
)
def test_convert_empi_dists_qiskit_to_quara(
    empi_dists_quara, empi_dists_qiskit, shots, label
):
    _test_convert_empi_dists_qiskit_to_quara(
        empi_dists_quara, empi_dists_qiskit, shots, label
    )
