from typing import List, Union
from qutip import (
    Qobj,
    to_super,
    basis,
    ket2dm,
    identity,
    sigmax,
    sigmay,
    sigmaz,
)


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


def generate_qutip_gate_from_gate_name(
    gate_name: str, dim: Union[None, int] = None, ids: Union[None, List[int]] = None
) -> Qobj:
    if gate_name == "identity":
        if dim == None:
            raise ValueError('dim must be specified for gate_name=="identity"')
        return to_super(identity(dim))
    if gate_name == "x":
        return to_super(sigmax())
    if gate_name == "y":
        return to_super(sigmay())
    if gate_name == "z":
        return to_super(sigmaz())
    else:
        raise ValueError("gate_name is out of range")