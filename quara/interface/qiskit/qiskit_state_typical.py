from typing import List


def get_qiskit_state_names_1qubit() -> List[str]:
    return ["Zp"]


def get_qiskit_state_names_2qubit() -> List[str]:
    return ["Zp", "Zp"]


def get_qiskit_state_names_3qubit() -> List[str]:
    return ["Zp", "Zp", "Zp"]
