from typing import List
import numpy as np
from qiskit.ignis.verification.tomography.basis.paulibasis import (
    pauli_measurement_matrix,
)


def get_qiskit_povm_names_1qubit() -> List[str]:
    return ["z"]


def get_qiskit_povm_names_2qubit() -> List[str]:
    return ["z_z"]


def get_qiskit_povm_names_3qubit() -> List[str]:
    return ["z_z_z"]


def generate_qiskit_povm_from_povm_name(povm_name: str) -> List[np.ndarray]:
    Z = [pauli_measurement_matrix("Z", i) for i in range(2)]
    if povm_name == "z":
        mat = [pauli_measurement_matrix("Z", i) for i in range(2)]
    elif povm_name == "z_z":
        mat = []
        for i in range(2):
            for j in range(2):
                a = np.kron(Z[i].T, Z[j])
                mat.append(a)

    elif povm_name == "z_z_z":
        mat = []
        for i in range(2):
            for j in range(2):
                for k in range(2):
                    a = np.kron(Z[j].T, Z[k])
                    b = np.kron(Z[i].T, a)
                    mat.append(b)

    else:
        raise ValueError("povm_name is out of range")

    return mat
