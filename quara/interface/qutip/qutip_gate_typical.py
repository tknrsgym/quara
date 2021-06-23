from typing import List, Union
from math import pi

import numpy as np
from scipy.linalg import expm
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
    return ["12xi90_i02z90"]


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
        if ids == None or len(ids) != 2 or False in [i in ids for i in range(2)]:
            raise ValueError("ids is None or invalid value")
        if ids[0] < ids[1]:
            hamiltonian = (
                pi / 4 * np.kron(sigmaz().data.toarray(), sigmax().data.toarray())
            )
        else:
            hamiltonian = (
                pi / 4 * np.kron(sigmax().data.toarray(), sigmaz().data.toarray())
            )
        return to_super(Qobj(expm(-1j * hamiltonian)))
    elif gate_name == "toffoli":
        if ids == None or len(ids) != 3 or False in [i in ids for i in range(3)]:
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
            # TODO: implement
            raise NotImplementedError("only for case where ids==[0,1,2] is implemented")
        return to_super(matrix)
    elif gate_name == "02y90":
        hamiltonian = pi / 4 * np.array([[0, 0, -1j], [0, 0, 0], [1j, 0, 0]])
        return to_super(Qobj(expm(-1j * hamiltonian)))
    elif gate_name == "12xi90_i02z90":
        hamiltonian = (
            pi
            / 4
            * (
                np.kron([[0, 0, 0], [0, 0, 1], [0, 1, 0]], np.eye(3))
                + np.kron(np.eye(3), [[1, 0, 0], [0, 0, 0], [0, 0, -1]])
            )
        )
        return to_super(Qobj(expm(-1j * hamiltonian)))
    else:
        raise ValueError("gate_name is out of range")