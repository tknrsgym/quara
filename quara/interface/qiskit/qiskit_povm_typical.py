from typing import List


def get_qiskit_povm_names_1qubit() -> List[str]:
    return ["Z"]


def get_qiskit_povm_names_2qubit() -> List[str]:
    return ["Z", "Z"]


def get_qiskit_povm_names_3qubit() -> List[str]:
    return ["Z", "Z", "Z"]
