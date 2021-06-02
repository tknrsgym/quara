import numpy as np
import numpy.testing as npt
import pytest

from qutip import (
    Qobj,
    basis as qutip_basis,
    qutrit_basis as qutip_qutrit_basis,
    ket2dm as qutip_ket2dm,
)

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
from quara.objects.composite_system import CompositeSystem
from quara.objects.composite_system_typical import generate_composite_system
from quara.objects.state import State
from quara.objects.state_typical import (
    generate_state_from_name,
    get_state_bell_2q,
    get_state_bell_pure_state_vector,
)
from quara.objects.povm import Povm
from quara.objects.gate import Gate
from quara.objects.matrix_basis import (
    calc_hermitian_matrix_expansion_coefficient_hermitian_basis,
)
from quara.utils.matrix_util import calc_mat_from_vector_adjoint


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


@pytest.mark.onequbit
@pytest.mark.parametrize(
    ("state_name"),
    [(state_name) for state_name in get_qutip_state_names_1qubit()],
)
def test_convert_state_qutip_to_quara_1qubit(state_name):
    _test_convert_state_qutip_to_quara("qubit", 1, state_name)


@pytest.mark.twoqubit
@pytest.mark.parametrize(
    ("state_name"),
    [(state_name) for state_name in get_qutip_state_names_2qubit()],
)
def test_convert_state_qutip_to_quara_2qubit(state_name):
    _test_convert_state_qutip_to_quara("qubit", 2, state_name)


@pytest.mark.threequbit
@pytest.mark.parametrize(
    ("state_name"),
    [(state_name) for state_name in get_qutip_state_names_3qubit()],
)
def test_convert_state_qutip_to_quara_3qubit(state_name):
    _test_convert_state_qutip_to_quara("qubit", 3, state_name)


@pytest.mark.onequtrit
@pytest.mark.parametrize(
    ("state_name"),
    [(state_name) for state_name in get_qutip_state_names_1qutrit()],
)
def test_convert_state_qutip_to_quara_1qutrit(state_name):
    _test_convert_state_qutip_to_quara("qutrit", 1, state_name)


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


@pytest.mark.onequbit
@pytest.mark.parametrize(
    ("state_name"),
    [(state_name) for state_name in get_qutip_state_names_1qubit()],
)
def test_convert_state_quara_to_qutip_1qubit(state_name):
    _test_convert_state_quara_to_qutip("qubit", 1, state_name)


@pytest.mark.twoqubit
@pytest.mark.parametrize(
    ("state_name"),
    [(state_name) for state_name in get_qutip_state_names_2qubit()],
)
def test_convert_state_quara_to_qutip_2qubit(state_name):
    _test_convert_state_quara_to_qutip("qubit", 2, state_name)


@pytest.mark.threequbit
@pytest.mark.parametrize(
    ("state_name"),
    [(state_name) for state_name in get_qutip_state_names_3qubit()],
)
def test_convert_state_quara_to_qutip_3qubit(state_name):
    _test_convert_state_quara_to_qutip("qubit", 3, state_name)


@pytest.mark.onequtrit
@pytest.mark.parametrize(
    ("state_name"),
    [(state_name) for state_name in get_qutip_state_names_1qutrit()],
)
def test_convert_state_quara_to_qutip_1qutrit(state_name):
    _test_convert_state_quara_to_qutip("qutrit", 1, state_name)


@pytest.mark.twoqutrit
@pytest.mark.parametrize(
    ("state_name"),
    [(state_name) for state_name in get_qutip_state_names_2qutrit()],
)
def test_convert_state_quara_to_qutip_2qutrit(state_name):
    _test_convert_state_quara_to_qutip("qutrit", 2, state_name)


def test_convert_povm_qutip_to_quara():
    pass


def test_convert_povm_quara_to_qutip():
    pass


def test_convert_gate_qutip_to_quara():
    pass


def test_convert_gate_quara_to_qutip():
    pass
