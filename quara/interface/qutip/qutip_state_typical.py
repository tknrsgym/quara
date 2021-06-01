from typing import List
from qutip import Qobj, basis, ket2dm


def get_qutip_state_names_1qubit() -> List[str]:
    return ["z0"]


def get_qutip_state_names_2qubit() -> List[str]:
    return ["z0_z0"]


def get_qutip_state_names_3qubit() -> List[str]:
    return ["z0_z0_z0"]


def get_qutip_state_names_1qutrit() -> List[str]:
    return ["01z0"]


def get_qutip_state_names_2qutrit() -> List[str]:
    return ["01z0_01z0"]


def generate_qutip_state_from_state_name(state_name: str, target_type: str) -> Qobj:
    if state_name == "z0":
        ket = basis(2, 0)
    elif state_name == "z0_z0":
        ket = basis([2, 2], [0, 0])
    elif state_name == "z0_z0_z0":
        ket = basis([2, 2, 2], [0, 0, 0])
    elif state_name == "01z0":
        ket = basis(3, 0)
    elif state_name == "01z0_01z0":
        ket = basis([3, 3], [0, 0])
    else:
        raise ValueError("state_name is out of range")
    if target_type == "oper":
        return ket2dm(ket)
    elif target_type == "ket":
        return ket
    else:
        raise ValueError("target_type is out of range")