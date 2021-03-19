from typing import List

import numpy as np

from quara.objects.composite_system import CompositeSystem
from quara.objects.matrix_basis import (
    MatrixBasis,
    calc_matrix_expansion_coefficient,
    calc_hermitian_matrix_expansion_coefficient_hermitian_basis,
)
from quara.objects.povm import Povm
from quara.utils.matrix_util import truncate_hs


def get_povm_names() -> List[str]:
    """Return the list of valid povm names."""
    names = []
    names.extend(get_povm_names_1qubit())
    names.extend(get_povm_names_2qubit())
    # names.extend(get_povm_names_1qutrit())
    # names.extend(get_povm_names_2qutrit())
    return names


def get_povm_names_1qubit() -> List[str]:
    """Return the list of valid povm names of 1-qubit povms."""
    names = [
        # "x",
        # "y",
        "z",
    ]
    return names


def get_povm_names_2qubit() -> List[str]:
    """Return the list of valid povm names of 2-qubits povms."""
    names = [
        "bell",
    ]
    return names


def generate_povm_matrices_from_povm_name(
    povm_name: str, dims: List[int] = [], ids: List[int] = []
) -> List[np.array]:
    """returns the list of the matrices of a povm.

    Parameters
    ----------
    povm_name : str
        name of povm

    dims : List[int]
        list of dimensions of elemental systems that the povm acts on.

    ids : List[int] (optional)
        list of ids for elemental systems

    Returns
    -------
    List[np.array]
        list of the matrices of the povms, to be complex.
    """
    # _is_valid_dims_ids(dims, ids)
    assert povm_name in get_povm_names()

    # 1-qubit
    if povm_name in get_povm_names_1qubit():
        method_name = "generate_povm_" + povm_name + "_matrices"
        method = eval(method_name)
        matrices = method()
    # 2-qubit
    elif povm_name in get_povm_names_2qubit():
        method_name = "generate_povm_" + povm_name + "_matrices"
        method = eval(method_name)
        matrices = method()
    # 3-qubit
    # 1-qutrit
    # 2-qutrit
    else:
        raise ValueError(f"povm_name is out of range. povm_name is {povm_name}")

    return matrices


def generate_povm_vecs_from_povm_name(
    povm_name: str, dims: List[int] = [], ids: List[int] = []
) -> List[np.array]:
    """returns the list of vectors of a povm.

    Parameters
    ----------
    povm_name : str
        name of povm

    dims : List[int]
        list of dimensions of elemental systems that the povm acts on.

    ids : List[int] (optional)
        list of ids for elemental systems

    Returns
    -------
    List[np.array]
        list of the vectors of the povm, to be complex.
    """
    # _is_valid_dims_ids(dims, ids)
    assert povm_name in get_povm_names()

    # 1-qubit
    if povm_name in get_povm_names_1qubit():
        method_name = "generate_povm_" + povm_name + "_vecs"
        method = eval(method_name)
        vecs = method()
    # 2-qubit
    elif povm_name in get_povm_names_2qubit():
        method_name = "generate_povm_" + povm_name + "_vecs"
        method = eval(method_name)
        vecs = method()
    # 3-qubit
    # 1-qutrit
    # 2-qutrit
    else:
        raise ValueError(f"povm_name is out of range. povm_name is {povm_name}")

    return vecs


def generate_povm_from_povm_name(
    povm_name: str, c_sys: CompositeSystem, dims: List[int] = [], ids: List[int] = []
) -> Povm:
    """returns povm class.

    Parameters
    ----------
    povm_name : str
        name of povm

    c_sys : CompositeSystem

    dims : List[int]
        list of dimensions of elemental systems that the povm acts on.

    ids : List[int] (optional)
        list of ids for elemental systems

    Returns
    -------
    Povm
        The povm class for the input
    """
    # _is_valid_dims_ids(dims, ids)
    assert povm_name in get_povm_names()

    # 1-qubit
    if povm_name in get_povm_names_1qubit():
        method_name = "generate_povm_" + povm_name
        method = eval(method_name)
        povm = method(c_sys)
    # 2-qubit
    elif povm_name in get_povm_names_2qubit():
        method_name = "generate_povm_" + povm_name
        method = eval(method_name)
        povm = method(c_sys)
    # 3-qubit
    # 1-qutrit
    # 2-qutrit
    else:
        raise ValueError(f"povm_name is out of range. povm_name is {povm_name}")

    return povm


def calc_povm_vecs_from_matrices(
    from_matrices: List[np.array], to_basis: MatrixBasis
) -> List[np.array]:
    """Return the HS vectors for a povm represented by povm matrices.

    Parameters
    ----------
    from_matrices : List[np.array]
        The list of matrices, to be square complex np.array.
        np.array((dim, dim), dtype=np.complex128)

    to_basis : MatrixBasis
        The matrix basis for representing the HS matrices, to be orthonormal.

    Returns
    -------
    List[np.array]
        The HS vectors of the povm corresponding to the povm matrices.
        np.array((dim^2, dim^2), dtype=np.complex128)
    """
    for from_matrix in from_matrices:
        shape = from_matrix.shape
        assert shape[0] == shape[1]
    dim = shape[0]

    assert to_basis.dim == dim
    assert to_basis.is_orthogonal() == True
    assert to_basis.is_normal() == True

    vecs = []
    for from_matrix in from_matrices:
        vecs.append(calc_matrix_expansion_coefficient(from_matrix, to_basis))
    return vecs


def calc_povm_vecs_from_matrices_with_hermitian_basis(
    from_matrices: np.array, to_basis: MatrixBasis
) -> List[np.array]:
    """Return the HS vectors w.r.t. a Hermitian (orthonormal) matrix basis for a povm represented by povm matrices.

    Parameters
    ----------
    from_matrices :
        The list of matrices, to be square complex np.array.
        np.array((dim, dim), dtype=np.complex128)

    to_basis : MatrixBasis
        The Hermitian matrix basis for representing the HS matrix

    Returns
    -------
    List[np.array]
        The HS vectors of the povm corresponding to the povm matrices, to be real.
        np.array((dim^2, dim^2), dtype=np.float64)
    """
    assert to_basis.is_hermitian() == True
    vecs = []
    for from_matrix in from_matrices:
        vecs.append(
            calc_hermitian_matrix_expansion_coefficient_hermitian_basis(
                from_matrix, to_basis
            )
        )
    return vecs


# z povm on 1-qubit


def generate_povm_z_matrices() -> List[np.array]:
    """Return the matrices for an z povm.

    The result is the list of the 2 times 2 complex matrices.

    Returns
    -------
    List[np.array]
        The list of unitary matrices, which is a complex matrix.
    """
    matrices = [
        np.array([[1 + 0j, 0 + 0j], [0 + 0j, 0 + 0j]], dtype=np.complex128),
        np.array([[0 + 0j, 0 + 0j], [0 + 0j, 1 + 0j]], dtype=np.complex128),
    ]
    return matrices


def generate_povm_z_vecs() -> List[np.array]:
    """Return the Hilbert-Schmidt representation vectors for an z povm with respect to the orthonormal Hermitian matrix basis with the normalized identity matrix as the 0th element.

    The result is the list of 4 dimension real vectors.

    Parameters
    ----------

    Returns
    -------
    List[np.array]
        The real Hilbert-Schmidt representation vectors for the povm.
    """
    vecs = [
        1 / np.sqrt(2) * np.array([1, 0, 0, 1], dtype=np.float64),
        1 / np.sqrt(2) * np.array([1, 0, 0, -1], dtype=np.float64),
    ]
    return vecs


def generate_povm_z(c_sys: CompositeSystem) -> Povm:
    """Return the Povm class for the z povm on the composite system.

    Parameters
    ----------
    c_sys: CompositeSystem

    Returns
    -------
    Povm
        The Povm class for the z povm on the composite system.
    """
    assert len(c_sys.elemental_systems) == 1
    vecs = generate_povm_z_vecs()
    povm = Povm(c_sys, vecs)
    return povm
