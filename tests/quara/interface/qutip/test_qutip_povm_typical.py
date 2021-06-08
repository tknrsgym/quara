import pytest
from quara.objects.povm_typical import (
    get_povm_names_1qubit,
    get_povm_names_2qubit,
    get_povm_names_3qubit,
    get_povm_names_1qutrit,
    get_povm_names_2qutrit,
)
from quara.interface.qutip.qutip_povm_typical import (
    get_qutip_povm_names_1qubit,
    get_qutip_povm_names_2qubit,
    get_qutip_povm_names_3qubit,
    get_qutip_povm_names_1qutrit,
    get_qutip_povm_names_2qutrit,
    generate_qutip_povm_from_povm_name,
)


@pytest.mark.qutip
@pytest.mark.parametrize(("type"), ["1qubit", "2qubit", "3qubit", "1qutrit", "2qutrit"])
def test_get_qutip_povm_names(type: str):
    quara_method_name = f"get_povm_names_{type}"
    qutip_method_name = f"get_qutip_povm_names_{type}"
    quara_method = eval(quara_method_name)
    qutip_method = eval(qutip_method_name)
    quara_povm_names = quara_method()
    qutip_povm_names = qutip_method()
    for qutip_povm_name in qutip_povm_names:
        assert qutip_povm_name in quara_povm_names


@pytest.mark.qutip
@pytest.mark.parametrize(("type"), ["1qubit", "2qubit", "3qubit", "1qutrit", "2qutrit"])
def test_generate_qutip_povm_from_povm_name(type):
    get_povm_name_method_name = f"get_qutip_povm_names_{type}"
    get_povm_name_method = eval(get_povm_name_method_name)
    povm_names = get_povm_name_method()
    for povm_name in povm_names:
        generate_qutip_povm_from_povm_name(povm_name)
    # TODO: check values
