from quara.objects.gate_typical import generate_gate_from_gate_name
from quara.objects.povm_typical import generate_povm_from_name
from qiskit.circuit.library.standard_gates import rzx, x, y, z, swap
from qiskit.quantum_info.operators.channel import Choi
import numpy as np
import numpy.testing as npt
import pytest

from quara.interface.qiskit.conversion import (
    convert_empi_dists_quara_to_qiskit_shots,
    convert_state_qiskit_to_quara,
    convert_state_quara_to_qiskit,
    convert_povm_qiskit_to_quara,
    convert_povm_quara_to_qiskit,
    convert_empi_dists_qiskit_to_quara,
    convert_empi_dists_quara_to_qiskit,
    convert_gate_qiskit_to_quara,
    convert_gate_quara_to_qiskit,
    calc_swap_matrix,
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
    get_swap_matrix_2dim,
    get_swap_matrix_3dim,
    generate_quara_gate_from_ids,
)

from quara.interface.qiskit.qiskit_empi_dists_typical import (
    get_empi_dists_label,
    get_empi_dists_qiskit,
    get_empi_dists_quara_int,
    get_empi_dists_quara_list,
    get_empi_dists_shots_int,
    get_empi_dists_shots_list,
)

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
        actual.to_density_matrix(), expected.to_density_matrix(), decimal=10
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
    ("state_name"), [(state_name) for state_name in get_qiskit_state_names_2qubit()]
)
def test_convert_state_qiskit_to_quara_2qubit(state_name):
    _test_convert_state_qiskit_to_quara("qubit", 2, state_name)


@pytest.mark.qiskit
@pytest.mark.threequbit
@pytest.mark.parametrize(
    ("state_name"), [(state_name) for state_name in get_qiskit_state_names_3qubit()]
)
def test_convert_state_qiskit_to_quara_3qubit(state_name):
    _test_convert_state_qiskit_to_quara("qubit", 3, state_name)


def _test_convert_state_quara_to_qiskit(mode, num, state_name):
    # Arrange
    c_sys = generate_composite_system(mode, num)
    expected = generate_qiskit_state_from_name(state_name)

    source = generate_state_from_name(c_sys, state_name)
    actual = convert_state_quara_to_qiskit(source)
    npt.assert_almost_equal(actual, expected, decimal=10)


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
    ("state_name"), [(state_name) for state_name in get_qiskit_state_names_2qubit()]
)
def test_convert_state_quara_to_qiskit_2qubit(state_name):
    _test_convert_state_quara_to_qiskit("qubit", 2, state_name)


@pytest.mark.qiskit
@pytest.mark.threequbit
@pytest.mark.parametrize(
    ("state_name"), [(state_name) for state_name in get_qiskit_state_names_3qubit()]
)
def test_convert_state_quara_to_qiskit_3qubit(state_name):
    _test_convert_state_quara_to_qiskit("qubit", 3, state_name)


def _test_convert_povm_qiskit_to_quara(mode, num, povm_name):
    # Arrange
    c_sys = generate_composite_system(mode, num)
    expected = generate_povm_from_name(povm_name, c_sys)

    source = generate_qiskit_povm_from_povm_name(povm_name)
    actual = convert_povm_qiskit_to_quara(source, c_sys)
    npt.assert_almost_equal(actual.matrices(), expected.matrices(), decimal=10)


@pytest.mark.qiskit
@pytest.mark.onequbit
@pytest.mark.parametrize(
    ("povm_name"),
    [(povm_name) for povm_name in get_qiskit_povm_names_1qubit()],
)
def test_convert_povm_qiskit_to_quara_1qubit(povm_name):
    _test_convert_povm_qiskit_to_quara("qubit", 1, povm_name)


@pytest.mark.qiskit
@pytest.mark.twoqubit
@pytest.mark.parametrize(
    ("povm_name"), [(povm_name) for povm_name in get_qiskit_povm_names_2qubit()]
)
def test_convert_povm_qiskit_to_quara_2qubit(povm_name):
    _test_convert_povm_qiskit_to_quara("qubit", 2, povm_name)


@pytest.mark.qiskit
@pytest.mark.threequbit
@pytest.mark.parametrize(
    ("povm_name"), [(povm_name) for povm_name in get_qiskit_povm_names_3qubit()]
)
def test_convert_povm_qiskit_to_quara_3qubit(povm_name):
    _test_convert_povm_qiskit_to_quara("qubit", 3, povm_name)


def _test_convert_povm_quara_to_qiskit(mode, num, povm_name):
    # Arrange
    c_sys = generate_composite_system(mode, num)
    expected = generate_qiskit_povm_from_povm_name(povm_name)

    source = generate_povm_from_name(povm_name, c_sys)
    actual = convert_povm_quara_to_qiskit(source)
    npt.assert_almost_equal(actual, expected, decimal=10)


@pytest.mark.qiskit
@pytest.mark.onequbit
@pytest.mark.parametrize(
    ("povm_name"),
    [(povm_name) for povm_name in get_qiskit_povm_names_1qubit()],
)
def test_convert_povm_quara_to_qiskit_1qubit(povm_name):
    _test_convert_povm_quara_to_qiskit("qubit", 1, povm_name)


@pytest.mark.qiskit
@pytest.mark.twoqubit
@pytest.mark.parametrize(
    ("povm_name"), [(povm_name) for povm_name in get_qiskit_povm_names_2qubit()]
)
def test_convert_povm_quara_to_qiskit_2qubit(povm_name):
    _test_convert_povm_quara_to_qiskit("qubit", 2, povm_name)


@pytest.mark.qiskit
@pytest.mark.threequbit
@pytest.mark.parametrize(
    ("povm_name"), [(povm_name) for povm_name in get_qiskit_povm_names_3qubit()]
)
def test_convert_povm_quara_to_qiskit_3qubit(povm_name):
    _test_convert_povm_quara_to_qiskit("qubit", 3, povm_name)


def _test_convert_gate_qiskit_to_quara(mode, num, gate_name, ids):
    dim = 2 ** num
    c_sys = generate_composite_system(mode, num)
    expected = generate_quara_gate_from_ids(gate_name, c_sys, ids)
    source = generate_qiskit_gate_from_gate_name(gate_name, ids)
    actual = convert_gate_qiskit_to_quara(source, c_sys, dim)
    npt.assert_almost_equal(actual.hs, expected.hs, decimal=10)


@pytest.mark.qiskit
@pytest.mark.onequbit
@pytest.mark.parametrize(
    ("gate_name"),
    [(gate_name) for gate_name in get_qiskit_gate_names_1qubit()],
)
def test_convert_gate_qiskit_to_quara_1qubit(gate_name):
    _test_convert_gate_qiskit_to_quara("qubit", 1, gate_name, ids=None)


@pytest.mark.qiskit
@pytest.mark.twoqubit
@pytest.mark.parametrize(
    ("gate_name,ids"),
    [
        ("cx", [0, 1]),
        ("cx", [1, 0]),
    ],
)
def test_convert_gate_qiskit_to_quara_2qubit(gate_name, ids):
    _test_convert_gate_qiskit_to_quara("qubit", 2, gate_name, ids)


@pytest.mark.qiskit
@pytest.mark.threequbit
@pytest.mark.parametrize(
    ("gate_name,ids"),
    [("toffoli", [0, 1, 2]), ("toffoli", [0, 2, 1]), ("toffoli", [2, 0, 1])],
)
def test_convert_gate_qiskit_to_quara_3qubit(gate_name, ids):
    _test_convert_gate_qiskit_to_quara("qubit", 3, gate_name, ids)


def _test_convert_gate_quara_to_qiskit(mode, num, gate_name, ids):
    dim = 2 ** num
    c_sys = generate_composite_system(mode, num)
    expected = generate_qiskit_gate_from_gate_name(gate_name, ids)
    source = generate_quara_gate_from_ids(gate_name, c_sys, ids)
    actual = convert_gate_quara_to_qiskit(source, dim)

    if (mode == "qubit" and num > 3) or (mode == "qutrit" and num > 2):
        actual = actual.toarray()

    npt.assert_almost_equal(actual, expected, decimal=10)


@pytest.mark.qiskit
@pytest.mark.onequbit
@pytest.mark.parametrize(
    ("gate_name"),
    [(gate_name) for gate_name in get_qiskit_gate_names_1qubit()],
)
def test_convert_gate_quara_to_qiskit_1qubit(gate_name):
    _test_convert_gate_quara_to_qiskit("qubit", 1, gate_name, ids=None)


@pytest.mark.qiskit
@pytest.mark.twoqubit
@pytest.mark.parametrize(
    ("gate_name,ids"),
    [
        ("cx", [0, 1]),
        ("cx", [1, 0]),
    ],
)
def test_convert_gate_quara_to_qiskit_2qubit(gate_name, ids):
    _test_convert_gate_quara_to_qiskit("qubit", 2, gate_name, ids)


@pytest.mark.qiskit
@pytest.mark.threequbit
@pytest.mark.parametrize(
    ("gate_name,ids"),
    [("toffoli", [0, 1, 2]), ("toffoli", [0, 2, 1]), ("toffoli", [2, 0, 1])],
)
def test_convert_gate_quara_to_qiskit_3qubit(gate_name, ids):
    _test_convert_gate_quara_to_qiskit("qubit", 3, gate_name, ids)


def _test_convert_empi_dists_qiskit_to_quara(
    empi_dists_quara, empi_dists_qiskit, shots, label
):
    expected = empi_dists_quara
    source = empi_dists_qiskit
    actual = convert_empi_dists_qiskit_to_quara(source, shots, label)
    npt.assert_equal(expected, actual)


@pytest.mark.qiskit
@pytest.mark.parametrize(
    ("empi_dists_quara, empi_dists_qiskit, shots, label"),
    [
        (
            get_empi_dists_quara_list(),
            get_empi_dists_qiskit(),
            get_empi_dists_shots_list(),
            get_empi_dists_label(),
        )
    ],
)
def test_convert_empi_dists_qiskit_to_quara_list(
    empi_dists_quara, empi_dists_qiskit, shots, label
):
    _test_convert_empi_dists_qiskit_to_quara(
        empi_dists_quara, empi_dists_qiskit, shots, label
    )


@pytest.mark.qiskit
@pytest.mark.parametrize(
    ("empi_dists_quara, empi_dists_qiskit, shots, label"),
    [
        (
            get_empi_dists_quara_int(),
            get_empi_dists_qiskit(),
            get_empi_dists_shots_int(),
            get_empi_dists_label(),
        )
    ],
)
def test_convert_empi_dists_qiskit_to_quara_int(
    empi_dists_quara, empi_dists_qiskit, shots, label
):
    _test_convert_empi_dists_qiskit_to_quara(
        empi_dists_quara, empi_dists_qiskit, shots, label
    )


def _test_convert_empi_dists_quara_to_qiskit(empi_dists_quara, empi_dists_qiskit):
    expected = empi_dists_qiskit
    source = empi_dists_quara
    actual = convert_empi_dists_quara_to_qiskit(source)
    npt.assert_equal(expected, actual)


@pytest.mark.qiskit
@pytest.mark.parametrize(
    ("empi_dists_quara, empi_dists_qiskit"),
    [(get_empi_dists_quara_list(), get_empi_dists_qiskit())],
)
def test_convert_empi_dists_quara_to_qiskit_list(empi_dists_quara, empi_dists_qiskit):
    _test_convert_empi_dists_quara_to_qiskit(empi_dists_quara, empi_dists_qiskit)


@pytest.mark.qiskit
@pytest.mark.parametrize(
    ("empi_dists_quara, empi_dists_qiskit"),
    [(get_empi_dists_quara_int(), get_empi_dists_qiskit())],
)
def test_convert_empi_dists_quara_to_qiskit_int(empi_dists_quara, empi_dists_qiskit):
    _test_convert_empi_dists_quara_to_qiskit(empi_dists_quara, empi_dists_qiskit)


def _test_convert_empi_dists_quara_to_qiskit_shots(empi_dists_quara, shots):
    expected = shots
    source = empi_dists_quara
    actual = convert_empi_dists_quara_to_qiskit_shots(quara_dists=source)
    npt.assert_equal(expected, actual)


@pytest.mark.qiskit
@pytest.mark.parametrize(
    ("empi_dists_quara,shots"),
    [(get_empi_dists_quara_list(), get_empi_dists_shots_list())],
)
def test_convert_empi_dists_quara_to_qiskit_shots_list(empi_dists_quara, shots):
    _test_convert_empi_dists_quara_to_qiskit_shots(empi_dists_quara, shots)


@pytest.mark.qiskit
@pytest.mark.parametrize(
    ("empi_dists_quara,shots"),
    [(get_empi_dists_quara_int(), get_empi_dists_shots_int())],
)
def test_convert_empi_dists_quara_to_qiskit_shots_int(empi_dists_quara, shots):
    _test_convert_empi_dists_quara_to_qiskit_shots(empi_dists_quara, shots)


@pytest.mark.qiskit
def test_calc_swap_matrix_2dim():
    expected = get_swap_matrix_2dim()
    actual = calc_swap_matrix(2)
    npt.assert_equal(expected, actual)


@pytest.mark.qiskit
def test_calc_swap_matrix_3dim():
    expected = get_swap_matrix_3dim()
    actual = calc_swap_matrix(3)
    npt.assert_equal(expected, actual)
