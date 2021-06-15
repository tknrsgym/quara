from typing import List, Union
from math import sqrt
from qutip import (
    Qobj,
    to_super,
    basis,
    ket2dm,
    identity,
    sigmax,
    sigmay,
    sigmaz,
    tensor,
)


def get_qutip_gate_names_1qubit() -> List[str]:
    return ["x", "y", "z"]


def get_qutip_gate_names_2qubit() -> List[str]:
    return ["zx90"]


def get_qutip_gate_names_3qubit() -> List[str]:
    return ["toffoli"]


def get_qutip_gate_names_1qutrit() -> List[str]:
    return ["02y90"]


def get_qutip_gate_names_2qutrit() -> List[str]:
    raise NotImplementedError


def generate_qutip_gate_from_gate_name(
    gate_name: str, dim: Union[None, int] = None, ids: Union[None, List[int]] = None
) -> Qobj:
    if gate_name == "identity":
        if dim == None:
            raise ValueError('dim must be specified for gate_name=="identity"')
        return to_super(identity(dim))
    elif gate_name == "x":
        return to_super(sigmax())
    elif gate_name == "y":
        return to_super(sigmay())
    elif gate_name == "z":
        return to_super(sigmaz())
    elif gate_name == "zx90":
        if ids == None or len(ids) != 2:
            raise ValueError("ids is None or invalid value")
        if ids[0] < ids[1]:
            matrix = (
                tensor(identity(2), identity(2)) - 1j * tensor(sigmaz(), sigmax())
            ) / sqrt(2)
        else:
            matrix = (
                tensor(identity(2), identity(2)) - 1j * tensor(sigmax(), sigmaz())
            ) / sqrt(2)
        return to_super(matrix)
    elif gate_name == "toffoli":
        if ids == None or len(ids) != 3:
            raise ValueError("ids is None or invalid value")
        if ids[2] == 2:
            matrix = (
                tensor(
                    (
                        tensor(identity(2), identity(2))
                        - ket2dm(tensor(basis(2, 1), basis(2, 1)))
                    ),
                    identity(2),
                )
                + tensor(ket2dm(tensor(basis(2, 1), basis(2, 1))), sigmax())
            )
        else:
            raise NotImplementedError("only for case where ids==[0,1,2] is implemented")
        return to_super(matrix)
    elif gate_name == "02y90":
        return to_super(
            Qobj([[1 / sqrt(2), 0, 0], [0, 1, 0], [0, 0, 1 / sqrt(2)]])
            - 1j * Qobj([[0, 0, -1j], [0, 0, 0], [1j, 0, 0]]) / sqrt(2)
        )
    else:
        raise ValueError("gate_name is out of range")