import pytest
from quara.objects.gate_typical import (
    get_gate_names_1qubit,
    get_gate_names_2qubit,
    get_gate_names_3qubit,
    get_gate_names_1qutrit,
    get_gate_names_2qutrit,
)
from quara.interface.qutip.qutip_gate_typical import (
    get_qutip_gate_names_1qubit,
    get_qutip_gate_names_2qubit,
    get_qutip_gate_names_3qubit,
    get_qutip_gate_names_1qutrit,
    get_qutip_gate_names_2qutrit,
    generate_qutip_gate_from_gate_name,
)


@pytest.mark.qutip
@pytest.mark.parametrize(("type"), ["1qubit", "2qubit", "3qubit", "1qutrit", "2qutrit"])
def test_get_qutip_gate_names(type: str):
    quara_method_name = f"get_gate_names_{type}"
    qutip_method_name = f"get_qutip_gate_names_{type}"
    quara_method = eval(quara_method_name)
    qutip_method = eval(qutip_method_name)
    quara_gate_names = quara_method()
    qutip_gate_names = qutip_method()
    for qutip_gate_name in qutip_gate_names:
        assert qutip_gate_name in quara_gate_names


@pytest.mark.qutip
@pytest.mark.parametrize(
    ("type", "dim", "ids"),
    [
        ("1qubit", 2, None),
        ("2qubit", 4, [0, 1]),
        ("2qubit", 4, [1, 0]),
        ("3qubit", 8, [0, 1, 2]),
        ("1qutrit", 3, None),
        ("2qutrit", 9, None),
    ],
)
def test_generate_qutip_gate_from_gate_name(type, dim, ids):
    get_gate_name_method_name = f"get_qutip_gate_names_{type}"
    get_gate_name_method = eval(get_gate_name_method_name)
    gate_names = get_gate_name_method()
    for gate_name in gate_names:
        generate_qutip_gate_from_gate_name(gate_name, dim, ids)
    # TODO: check values


@pytest.mark.qutip
@pytest.mark.parametrize(
    ("type", "dim", "ids"),
    [
        ("3qubit", 8, [2, 1, 0]),
        ("3qubit", 8, [1, 2, 0]),
    ],
)
def test_generate_qutip_gate_from_gate_name_not_implemented(type, dim, ids):
    get_gate_name_method_name = f"get_qutip_gate_names_{type}"
    get_gate_name_method = eval(get_gate_name_method_name)
    gate_names = get_gate_name_method()
    for gate_name in gate_names:
        with pytest.raises(NotImplementedError):
            generate_qutip_gate_from_gate_name(gate_name, dim, ids)
    # TODO: check values


@pytest.mark.qutip
@pytest.mark.parametrize(("dim"), [2, 3, 4, 8, 9])
def test_generate_qutip_identity_gate_from_gate_name(dim):
    qutip_gate = generate_qutip_gate_from_gate_name("identity", dim)
    assert qutip_gate.shape[0] == dim ** 2
    # TODO: check values
