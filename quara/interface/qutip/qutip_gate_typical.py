from typing import List
from qutip import Qobj, to_super, basis, ket2dm, sigmax, sigmay, sigmaz


def get_qutip_gate_names_1qubit() -> List[str]:
    return ["x", "y", "z"]


# TODO: implement
def get_qutip_gate_names_2qubit() -> List[str]:
    return NotImplementedError


def get_qutip_gate_names_3qubit() -> List[str]:
    return NotImplementedError


def get_qutip_gate_names_1qutrit() -> List[str]:
    return NotImplementedError


def get_qutip_gate_names_2qutrit() -> List[str]:
    return NotImplementedError


def generate_qutip_gate_from_gate_name(gate_name: str) -> Qobj:
    if gate_name == "x":
        return to_super(sigmax())
    if gate_name == "y":
        return to_super(sigmay())
    if gate_name == "z":
        return to_super(sigmaz())
    else:
        raise ValueError("gate_name is out of range")