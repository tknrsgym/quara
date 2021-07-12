from typing import List, Union
import numpy as np
from qiskit.circuit.library.standard_gates import (
    rzx,
    x,
    y,
    z,
)


def get_qiskit_gate_names_1qubit() -> List[str]:
    return ["x", "y", "z"]


def get_qiskit_gate_names_2qubit() -> List[str]:
    return ["cx"]


def get_qiskit_gate_names_3qubit() -> List[str]:
    return ["toffoli"]


def generate_qiskit_gate_from_gate_name(
    gate_name: str, ids: Union[None, List[int]] = None
) -> np.ndarray:
    if gate_name == "x":
        gate = x.XGate()
        mat = gate.__array__()

    elif gate_name == "y":
        gate = y.YGate()
        mat = gate.__array__()

    elif gate_name == "z":
        gate = z.ZGate()
        mat = gate.__array__()

    elif gate_name == "cx":
        if ids == None or len(ids) != 2 or False in [i in ids for i in range(2)]:
            raise ValueError("ids is None or invalid value")
        elif ids[0] < ids[1]:
            gate = x.CXGate()
        else:
            gate = x.CXGate(ctrl_state=0)
        mat = gate.__array__()

    else:
        raise ValueError("gate_name is out of range")

    return mat
