from typing import List
import numpy as np
from qiskit.ignis.verification.tomography.basis.paulibasis import (
    pauli_preparation_matrix,
    pauli_measurement_matrix,
)


def get_qiskit_state_names_1qubit() -> List[str]:
    return ["z0"]


def get_qiskit_state_names_2qubit() -> List[str]:
    return ["z0_z0"]


def get_qiskit_state_names_3qubit() -> List[str]:
    return ["z0_z0_z0"]


def generate_qiskit_state_from_name(state_name: str) -> np.ndarray:
    Zp = pauli_preparation_matrix("Zp")

    if state_name == "z0":
        mat = pauli_preparation_matrix("Zp")
    elif state_name == "z0_z0":
        mat = np.kron(Zp.T, Zp)
    elif state_name == "z0_z0_z0":
        mat1 = np.kron(Zp.T, Zp)
        mat = np.kron(Zp.T, Zp)
    else:
        raise ValueError("state_name is out of range")

    return mat
