import pytest
from quara.objects.state_typical import (
    get_state_names_1qubit,
    get_state_names_2qubit,
    get_state_names_3qubit,
    get_state_names_1qutrit,
    get_state_names_2qutrit,
)
from quara.interface.qutip.qutip_state_typical import (
    get_qutip_state_names_1qubit,
    get_qutip_state_names_2qubit,
    get_qutip_state_names_3qubit,
    get_qutip_state_names_1qutrit,
    get_qutip_state_names_2qutrit,
    generate_qutip_state_from_state_name,
)


@pytest.mark.qutip
@pytest.mark.parametrize(("type"), ["1qubit", "2qubit", "3qubit", "1qutrit", "2qutrit"])
def test_get_qutip_state_names(type: str):
    quara_method_name = f"get_state_names_{type}"
    qutip_method_name = f"get_qutip_state_names_{type}"
    quara_method = eval(quara_method_name)
    qutip_method = eval(qutip_method_name)
    quara_state_names = quara_method()
    qutip_state_names = qutip_method()
    for qutip_state_name in qutip_state_names:
        assert qutip_state_name in quara_state_names


@pytest.mark.qutip
@pytest.mark.parametrize(("type"), ["1qubit", "2qubit", "3qubit", "1qutrit", "2qutrit"])
def test_generate_qutip_state_from_state_name(type):
    get_state_name_method_name = f"get_qutip_state_names_{type}"
    get_state_name_method = eval(get_state_name_method_name)
    state_names = get_state_name_method()
    for state_name in state_names:
        generate_qutip_state_from_state_name(state_name, target_type="ket")
        generate_qutip_state_from_state_name(state_name, target_type="oper")
    # TODO: check values
