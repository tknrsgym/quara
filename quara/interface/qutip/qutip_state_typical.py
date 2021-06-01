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


def generate_qutip_state_from_state_name(state_name: str) -> Qobj:
    if state_name == "z0":
        return ket2dm(basis(2, 0))
    if state_name == "z0_z0":
        return ket2dm(basis([2, 2], [0, 0]))
    if state_name == "z0_z0_z0":
        return ket2dm(basis([2, 2, 2], [0, 0, 0]))
    if state_name == "01z0":
        return ket2dm(basis(3, 0))
    if state_name == "01z0_01z0":
        return ket2dm(basis([3, 3], [0, 0]))
    else:
        raise ValueError("state_name is out of range")
