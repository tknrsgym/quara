import numpy as np
import numpy.testing as npt
import pytest

from quara.interface.qutip.conversion import (
    convert_state_quara_to_qutip,
    convert_state_qutip_to_quara,
    convert_povm_quara_to_qutip,
    convert_povm_qutip_to_quara,
    convert_gate_quara_to_qutip,
    convert_gate_qutip_to_quara,
)
from quara.interface.qutip.qutip_state_typical import (
    get_qutip_state_names_1qubit,
    get_qutip_state_names_1qutrit,
    get_qutip_state_names_2qutrit,
    get_qutip_state_names_2qubit,
    get_qutip_state_names_3qubit,
    generate_qutip_state_from_state_name,
)
from quara.interface.qutip.qutip_povm_typical import (
    get_qutip_povm_names_1qubit,
    get_qutip_povm_names_2qubit,
    get_qutip_povm_names_3qubit,
    get_qutip_povm_names_1qutrit,
    get_qutip_povm_names_2qutrit,
    generate_qutip_povm_from_povm_name,
)
from quara.interface.qutip.qutip_gate_typical import (
    get_qutip_gate_names_1qubit,
    get_qutip_gate_names_2qubit,
    get_qutip_gate_names_3qubit,
    get_qutip_gate_names_1qutrit,
    get_qutip_gate_names_2qutrit,
    generate_qutip_gate_from_gate_name,
)
from quara.objects.composite_system import CompositeSystem
from quara.objects.composite_system_typical import generate_composite_system
from quara.objects.state_typical import generate_state_from_name
from quara.objects.povm_typical import generate_povm_from_name
from quara.objects.gate_typical import generate_gate_from_gate_name


def _test_convert_state_qutip_to_quara(mode, num, state_name):
    # Arrange
    c_sys = generate_composite_system(mode, num)
    expected = generate_state_from_name(c_sys, state_name)

    # Case 1: Qobj.type=="ket"
    source = generate_qutip_state_from_state_name(state_name, "ket")
    actual = convert_state_qutip_to_quara(source, c_sys)
    npt.assert_array_almost_equal(actual.vec, expected.vec, decimal=10)

    # Case 2: Qobj.type=="oper"
    source = generate_qutip_state_from_state_name(state_name, "oper")
    actual = convert_state_qutip_to_quara(source, c_sys)
    npt.assert_array_almost_equal(actual.vec, expected.vec, decimal=10)

    # Case 2: test Qobj type validation
    source = generate_qutip_state_from_state_name(state_name, "ket")
    with pytest.raises(ValueError):
        # type == "bra"
        convert_state_qutip_to_quara(source.dag(), c_sys)


@pytest.mark.qutip
@pytest.mark.onequbit
@pytest.mark.parametrize(
    ("state_name"),
    [(state_name) for state_name in get_qutip_state_names_1qubit()],
)
def test_convert_state_qutip_to_quara_1qubit(state_name):
    _test_convert_state_qutip_to_quara("qubit", 1, state_name)


@pytest.mark.qutip
@pytest.mark.twoqubit
@pytest.mark.parametrize(
    ("state_name"),
    [(state_name) for state_name in get_qutip_state_names_2qubit()],
)
def test_convert_state_qutip_to_quara_2qubit(state_name):
    _test_convert_state_qutip_to_quara("qubit", 2, state_name)


@pytest.mark.qutip
@pytest.mark.threequbit
@pytest.mark.parametrize(
    ("state_name"),
    [(state_name) for state_name in get_qutip_state_names_3qubit()],
)
def test_convert_state_qutip_to_quara_3qubit(state_name):
    _test_convert_state_qutip_to_quara("qubit", 3, state_name)


@pytest.mark.qutip
@pytest.mark.onequtrit
@pytest.mark.parametrize(
    ("state_name"),
    [(state_name) for state_name in get_qutip_state_names_1qutrit()],
)
def test_convert_state_qutip_to_quara_1qutrit(state_name):
    _test_convert_state_qutip_to_quara("qutrit", 1, state_name)


@pytest.mark.qutip
@pytest.mark.twoqutrit
@pytest.mark.parametrize(
    ("state_name"),
    [(state_name) for state_name in get_qutip_state_names_2qutrit()],
)
def test_convert_state_qutip_to_quara_2qutrit(state_name):
    _test_convert_state_qutip_to_quara("qutrit", 2, state_name)


def _test_convert_state_quara_to_qutip(mode, num, state_name):
    # Arrange
    c_sys = generate_composite_system(mode, num)
    expected = generate_qutip_state_from_state_name(state_name, "oper")

    # Case 1: target_type=="oper"
    source = generate_state_from_name(c_sys, state_name)
    actual = convert_state_quara_to_qutip(source)

    npt.assert_array_almost_equal(
        actual.data.toarray(), expected.data.toarray(), decimal=10
    )


@pytest.mark.qutip
@pytest.mark.onequbit
@pytest.mark.parametrize(
    ("state_name"),
    [(state_name) for state_name in get_qutip_state_names_1qubit()],
)
def test_convert_state_quara_to_qutip_1qubit(state_name):
    _test_convert_state_quara_to_qutip("qubit", 1, state_name)


@pytest.mark.qutip
@pytest.mark.twoqubit
@pytest.mark.parametrize(
    ("state_name"),
    [(state_name) for state_name in get_qutip_state_names_2qubit()],
)
def test_convert_state_quara_to_qutip_2qubit(state_name):
    _test_convert_state_quara_to_qutip("qubit", 2, state_name)


@pytest.mark.qutip
@pytest.mark.threequbit
@pytest.mark.parametrize(
    ("state_name"),
    [(state_name) for state_name in get_qutip_state_names_3qubit()],
)
def test_convert_state_quara_to_qutip_3qubit(state_name):
    _test_convert_state_quara_to_qutip("qubit", 3, state_name)


@pytest.mark.qutip
@pytest.mark.onequtrit
@pytest.mark.parametrize(
    ("state_name"),
    [(state_name) for state_name in get_qutip_state_names_1qutrit()],
)
def test_convert_state_quara_to_qutip_1qutrit(state_name):
    _test_convert_state_quara_to_qutip("qutrit", 1, state_name)


@pytest.mark.qutip
@pytest.mark.twoqutrit
@pytest.mark.parametrize(
    ("state_name"),
    [(state_name) for state_name in get_qutip_state_names_2qutrit()],
)
def test_convert_state_quara_to_qutip_2qutrit(state_name):
    _test_convert_state_quara_to_qutip("qutrit", 2, state_name)


def _test_convert_povm_qutip_to_quara(mode, num, povm_name):
    # Arrange
    c_sys = generate_composite_system(mode, num)
    expected = generate_povm_from_name(povm_name, c_sys)

    # Test
    source = generate_qutip_povm_from_povm_name(povm_name)
    actual = convert_povm_qutip_to_quara(source, c_sys)
    assert expected.dim == actual.dim
    for expected_vec, actual_vec in zip(expected.vecs, actual.vecs):
        npt.assert_array_almost_equal(expected_vec, actual_vec, decimal=10)


@pytest.mark.qutip
@pytest.mark.onequbit
@pytest.mark.parametrize(
    ("povm_name"),
    [(povm_name) for povm_name in get_qutip_povm_names_1qubit()],
)
def test_convert_povm_qutip_to_quara_1qubit(povm_name):
    _test_convert_povm_qutip_to_quara("qubit", 1, povm_name)


@pytest.mark.qutip
@pytest.mark.twoqubit
@pytest.mark.parametrize(
    ("povm_name"),
    [(povm_name) for povm_name in get_qutip_povm_names_2qubit()],
)
def test_convert_povm_qutip_to_quara_2qubit(povm_name):
    _test_convert_povm_qutip_to_quara("qubit", 2, povm_name)


@pytest.mark.qutip
@pytest.mark.threequbit
@pytest.mark.parametrize(
    ("povm_name"),
    [(povm_name) for povm_name in get_qutip_povm_names_3qubit()],
)
def test_convert_povm_qutip_to_quara_3qubit(povm_name):
    _test_convert_povm_qutip_to_quara("qubit", 3, povm_name)


@pytest.mark.qutip
@pytest.mark.onequtrit
@pytest.mark.parametrize(
    ("povm_name"),
    [(povm_name) for povm_name in get_qutip_povm_names_1qutrit()],
)
def test_convert_povm_qutip_to_quara_1qutrit(povm_name):
    _test_convert_povm_qutip_to_quara("qutrit", 1, povm_name)


@pytest.mark.qutip
@pytest.mark.twoqutrit
@pytest.mark.parametrize(
    ("povm_name"),
    [(povm_name) for povm_name in get_qutip_povm_names_2qutrit()],
)
def test_convert_povm_qutip_to_quara_2qutrit(povm_name):
    _test_convert_povm_qutip_to_quara("qutrit", 2, povm_name)


def _test_convert_povm_quara_to_qutip(mode, num, povm_name):
    # Arrange
    c_sys = generate_composite_system(mode, num)
    expected = generate_qutip_povm_from_povm_name(povm_name)

    # Test
    source = generate_povm_from_name(povm_name, c_sys)
    actual = convert_povm_quara_to_qutip(source)
    assert len(expected) == len(actual)
    for expected_qobj, actual_qobj in zip(expected, actual):
        npt.assert_array_almost_equal(
            expected_qobj.data.toarray(), actual_qobj.data.toarray(), decimal=10
        )


@pytest.mark.qutip
@pytest.mark.onequbit
@pytest.mark.parametrize(
    ("povm_name"),
    [(povm_name) for povm_name in get_qutip_povm_names_1qubit()],
)
def test_convert_povm_quara_to_qutip_1qubit(povm_name):
    _test_convert_povm_quara_to_qutip("qubit", 1, povm_name)


@pytest.mark.qutip
@pytest.mark.twoqubit
@pytest.mark.parametrize(
    ("povm_name"),
    [(povm_name) for povm_name in get_qutip_povm_names_2qubit()],
)
def test_convert_povm_quara_to_qutip_2qubit(povm_name):
    _test_convert_povm_quara_to_qutip("qubit", 2, povm_name)


@pytest.mark.qutip
@pytest.mark.threequbit
@pytest.mark.parametrize(
    ("povm_name"),
    [(povm_name) for povm_name in get_qutip_povm_names_3qubit()],
)
def test_convert_povm_quara_to_qutip_3qubit(povm_name):
    _test_convert_povm_quara_to_qutip("qubit", 3, povm_name)


@pytest.mark.qutip
@pytest.mark.onequtrit
@pytest.mark.parametrize(
    ("povm_name"),
    [(povm_name) for povm_name in get_qutip_povm_names_1qutrit()],
)
def test_convert_povm_quara_to_qutip_1qutrit(povm_name):
    _test_convert_povm_quara_to_qutip("qutrit", 1, povm_name)


@pytest.mark.qutip
@pytest.mark.twoqutrit
@pytest.mark.parametrize(
    ("povm_name"),
    [(povm_name) for povm_name in get_qutip_povm_names_2qutrit()],
)
def test_convert_povm_quara_to_qutip_2qutrit(povm_name):
    _test_convert_povm_quara_to_qutip("qutrit", 2, povm_name)


def _test_convert_gate_qutip_to_quara(mode, num, dim, gate_name, ids=None):
    # Arrange
    c_sys = generate_composite_system(mode, num)
    expected = generate_gate_from_gate_name(gate_name, c_sys, ids)

    # Test
    source = generate_qutip_gate_from_gate_name(gate_name, dim, ids)
    actual = convert_gate_qutip_to_quara(source, c_sys)
    npt.assert_array_almost_equal(expected.hs, actual.hs, decimal=10)


@pytest.mark.qutip
@pytest.mark.onequbit
@pytest.mark.parametrize(
    ("gate_name"),
    [(gate_name) for gate_name in get_qutip_gate_names_1qubit()],
)
def test_convert_gate_qutip_to_quara_1qubit(gate_name):
    _test_convert_gate_qutip_to_quara("qubit", 1, 2, gate_name)


@pytest.mark.qutip
@pytest.mark.twoqubit
@pytest.mark.parametrize(
    ("gate_name"),
    [(gate_name) for gate_name in get_qutip_gate_names_2qubit()],
)
def test_convert_gate_qutip_to_quara_2qubit(gate_name):
    _test_convert_gate_qutip_to_quara("qubit", 2, 4, gate_name, [0, 1])


@pytest.mark.qutip
@pytest.mark.threequbit
@pytest.mark.parametrize(
    ("gate_name"),
    [(gate_name) for gate_name in get_qutip_gate_names_3qubit()],
)
def test_convert_gate_qutip_to_quara_3qubit(gate_name):
    _test_convert_gate_qutip_to_quara("qubit", 3, 8, gate_name, [0, 1, 2])
    _test_convert_gate_qutip_to_quara("qubit", 3, 8, gate_name, [1, 0, 2])
    with pytest.raises(NotImplementedError):
        _test_convert_gate_qutip_to_quara("qubit", 3, 8, gate_name, [2, 0, 1])


@pytest.mark.qutip
@pytest.mark.onequtrit
@pytest.mark.parametrize(
    ("gate_name"),
    [(gate_name) for gate_name in get_qutip_gate_names_1qutrit()],
)
def test_convert_gate_qutip_to_quara_1qutrit(gate_name):
    _test_convert_gate_qutip_to_quara("qutrit", 1, 3, gate_name)


@pytest.mark.qutip
@pytest.mark.twoqutrit
@pytest.mark.parametrize(
    ("gate_name"),
    [(gate_name) for gate_name in get_qutip_gate_names_2qutrit()],
)
def test_convert_gate_qutip_to_quara_2qutrit(gate_name):
    _test_convert_gate_qutip_to_quara("qutrit", 2, 9, gate_name)


def _test_convert_gate_quara_to_qutip(mode, num, dim, gate_name, ids=None):
    # Arrange
    c_sys = generate_composite_system(mode, num)
    expected = generate_qutip_gate_from_gate_name(gate_name, dim, ids)

    # Test
    source = generate_gate_from_gate_name(gate_name, c_sys, ids)
    actual = convert_gate_quara_to_qutip(source)
    npt.assert_array_almost_equal(
        expected.data.toarray(), actual.data.toarray(), decimal=10
    )


@pytest.mark.qutip
@pytest.mark.onequbit
@pytest.mark.parametrize(
    ("gate_name"),
    [(gate_name) for gate_name in get_qutip_gate_names_1qubit()],
)
def test_convert_gate_quara_to_qutip_1qubit(gate_name):
    _test_convert_gate_quara_to_qutip("qubit", 1, 2, gate_name)


@pytest.mark.qutip
@pytest.mark.twoqubit
@pytest.mark.parametrize(
    ("gate_name"),
    [(gate_name) for gate_name in get_qutip_gate_names_2qubit()],
)
def test_convert_gate_quara_to_qutip_1qubit(gate_name):
    _test_convert_gate_quara_to_qutip("qubit", 2, 4, gate_name, [0, 1])
    _test_convert_gate_quara_to_qutip("qubit", 2, 4, gate_name, [1, 0])


@pytest.mark.qutip
@pytest.mark.threequbit
@pytest.mark.parametrize(
    ("gate_name"),
    [(gate_name) for gate_name in get_qutip_gate_names_3qubit()],
)
def test_convert_gate_quara_to_qutip_3qubit(gate_name):
    _test_convert_gate_quara_to_qutip("qubit", 3, 8, gate_name, [0, 1, 2])
    _test_convert_gate_quara_to_qutip("qubit", 3, 8, gate_name, [1, 0, 2])


@pytest.mark.qutip
@pytest.mark.onequtrit
@pytest.mark.parametrize(
    ("gate_name"),
    [(gate_name) for gate_name in get_qutip_gate_names_1qutrit()],
)
def test_convert_gate_quara_to_qutip_1qutrit(gate_name):
    _test_convert_gate_quara_to_qutip("qutrit", 1, 3, gate_name)


@pytest.mark.qutip
@pytest.mark.twoqutrit
@pytest.mark.parametrize(
    ("gate_name"),
    [(gate_name) for gate_name in get_qutip_gate_names_2qutrit()],
)
def test_convert_gate_quara_to_qutip_2qutrit(gate_name):
    _test_convert_gate_quara_to_qutip("qutrit", 2, 9, gate_name)