from typing import List
from qutip import Qobj, basis, ket2dm, tensor


def get_qutip_povm_names_1qubit() -> List[str]:
    return ["z"]


def get_qutip_povm_names_2qubit() -> List[str]:
    return ["z_z"]


def get_qutip_povm_names_3qubit() -> List[str]:
    return ["z_z_z"]


def get_qutip_povm_names_1qutrit() -> List[str]:
    return ["z2", "z3"]


def get_qutip_povm_names_2qutrit() -> List[str]:
    return ["z2_z2", "z3_z3"]


def generate_qutip_povm_from_povm_name(povm_name: str) -> List[Qobj]:
    if povm_name == "z":
        return [ket2dm(basis(2, i)) for i in range(2)]
    elif povm_name == "z_z":
        return [ket2dm(basis(2 ** 2, i)) for i in range(2 ** 2)]
    elif povm_name == "z_z_z":
        return [ket2dm(basis(2 ** 3, i)) for i in range(2 ** 3)]
    elif povm_name == "z2":
        return [ket2dm(basis(3, 0)), ket2dm(basis(3, 1)) + ket2dm(basis(3, 2))]
    elif povm_name == "z3":
        return [ket2dm(basis(3, 0)), ket2dm(basis(3, 1)), ket2dm(basis(3, 2))]
    elif povm_name == "z2_z2":
        z2_matrices = [ket2dm(basis(3, 0)), ket2dm(basis(3, 1)) + ket2dm(basis(3, 2))]
        return [tensor(mat1, mat2) for mat1 in z2_matrices for mat2 in z2_matrices]
    elif povm_name == "z3_z3":
        z3_matrices = [ket2dm(basis(3, 0)), ket2dm(basis(3, 1)), ket2dm(basis(3, 2))]
        return [tensor(mat1, mat2) for mat1 in z3_matrices for mat2 in z3_matrices]
    else:
        raise ValueError("povm_name is out of range")