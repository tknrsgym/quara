from typing import List
from qutip import Qobj, basis, ket2dm


def get_qutip_povm_names_1qubit() -> List[str]:
    return ["z"]


def get_qutip_povm_names_2qubit() -> List[str]:
    return ["z_z"]


def get_qutip_povm_names_3qubit() -> List[str]:
    return ["z_z_z"]


def get_qutip_povm_names_1qutrit() -> List[str]:
    raise NotImplementedError


def get_qutip_povm_names_2qutrit() -> List[str]:
    raise NotImplementedError


def generate_qutip_povm_from_povm_name(povm_name: str) -> List[Qobj]:
    # TODO: implement for qutrits
    if povm_name == "z":
        return [ket2dm(basis(2, i)) for i in range(2)]
    elif povm_name == "z_z":
        return [ket2dm(basis(2 ** 2, i)) for i in range(2 ** 2)]
    elif povm_name == "z_z_z":
        return [ket2dm(basis(2 ** 3, i)) for i in range(2 ** 3)]
    else:
        raise ValueError("povm_name is out of range")