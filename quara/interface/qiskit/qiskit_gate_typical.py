from typing import List
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
    return ["zx90"]


def get_qiskit_gate_names_3qubit() -> List[str]:
    return ["toffoli"]


def generate_qiskit_gate_from_gate_name(
    gate_name: str,
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

    elif gate_name == "zx90":
        gate = rzx.RZXGate(np.pi/2)
        mat = gate.__array__()

    elif gate_name == "toffoli":
        gate = 

    else:
        raise ValueError("gate_name is out of range")

    return mat
