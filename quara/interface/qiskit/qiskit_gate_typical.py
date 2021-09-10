from typing import List, Union
import numpy as np
from qiskit.circuit.library.standard_gates import rzx, x, y, z, swap
from qiskit.quantum_info.operators.channel import Choi


def get_qiskit_gate_names_1qubit() -> List[str]:
    return ["x", "y", "z"]


def get_qiskit_gate_names_2qubit() -> List[str]:
    return ["swap"]


def get_qiskit_gate_names_3qubit() -> List[str]:
    return ["toffoli"]


def generate_qiskit_gate_from_gate_name(
    gate_name: str, ids: Union[None, List[int]] = None
) -> np.ndarray:
    if gate_name == "x":
        gate = x.XGate()
        mat = Choi(gate)

    elif gate_name == "y":
        gate = y.YGate()
        mat = Choi(gate)

    elif gate_name == "z":
        gate = z.ZGate()
        mat = Choi(gate)

    elif gate_name == "swap":
        gate = swap.SwapGate()
        mat = Choi(gate)

    return mat


def get_swap_matrix_2dim() -> np.ndarray:
    mat = np.zeros((4, 4), dtype=np.complex64)
    mat[0, 0] = 1
    mat[1, 2] = 1
    mat[2, 1] = 1
    mat[3, 3] = 1
    return mat


def get_swap_matrix_3dim() -> np.ndarray:
    mat = np.zeros((9, 9), dtype=np.complex64)
    mat[0, 0] = 1
    mat[1, 3] = 1
    mat[2, 6] = 1
    mat[3, 1] = 1
    mat[4, 4] = 1
    mat[5, 7] = 1
    mat[6, 2] = 1
    mat[7, 5] = 1
    mat[8, 8] = 1
    return mat
