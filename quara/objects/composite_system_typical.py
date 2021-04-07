from typing import List

# Quara
from quara.objects.matrix_basis import (
    MatrixBasis,
    get_normalized_pauli_basis,
    get_normalized_gell_mann_basis,
)
from quara.objects.elemental_system import ElementalSystem
from quara.objects.composite_system import CompositeSystem


def generate_composite_system(
    mode: str, num: int, ids_esys: List[int] = None, basis: MatrixBasis = None
) -> CompositeSystem:
    """return a composite system consisting identical elemental systems.

    Parameters
    ----------
    mode: str
        "qubit" or "qutrit"

    num: int
        number of qubits or qutrits

    ids_esys: List[int] = None
        list of ids for each elemental systems

    basis: MatrixBasis = None
        orthonormal matrix basis. When it is None, the normalized Pauli or Normalized Gell-Mann basis is chosen automatically.

    Returns
    ----------
    CompositeSystem
    """
    assert mode == "qubit" or mode == "qutrit"
    assert num > 0

    # ids_esys
    if ids_esys == None:
        ids_esys = list(range(num))
    assert len(ids_esys) == num

    # basis
    if basis == None:
        if mode == "qubit":
            basis = get_normalized_pauli_basis()
        elif mode == "qutrit":
            basis = get_normalized_gell_mann_basis()

    # elemental systems
    l_esys = []
    for i in ids_esys:
        esys = ElementalSystem(name=i, basis=basis)
        l_esys.append(esys)

    # composite system
    c_sys = CompositeSystem(systems=l_esys)

    return c_sys
