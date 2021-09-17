from typing import List, Union
import numpy as np
from qiskit.circuit.library.standard_gates import rzx, x, y, z, swap
from qiskit.quantum_info.operators.channel import Choi


def get_qiskit_gate_names_1qubit() -> List[str]:
    return ["x", "y", "z"]


def get_qiskit_gate_names_2qubit() -> List[str]:
    return ["cx"]


def get_qiskit_gate_names_3qubit() -> List[str]:
    return ["toffoli"]


def generate_qiskit_gate_from_gate_name(
    gate_name: str, ids: Union[None, List[int]]
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

    elif gate_name == "cx":
        if ids[0] < ids[1]:
            xx = get_xx_matrix_2dim()
            gate_qiskit = x.CXGate(ctrl_state=0)
            qis = gate_qiskit.__array__()
            gate_quara = np.dot(xx, np.dot(qis, xx))
            mat = Choi(gate_quara)
        elif ids[1] < ids[0]:
            gate = x.CXGate(ctrl_state=1)
            mat = Choi(gate)

    elif gate_name == "toffoli":
        if ids == [2, 0, 1]:
            gate = x.CCXGate(ctrl_state=3)
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


def get_xx_matrix_2dim() -> np.ndarray:
    mat = np.zeros((4, 4))
    mat[0, 3] = 1
    mat[1, 2] = 1
    mat[2, 1] = 1
    mat[3, 0] = 1
    return mat
