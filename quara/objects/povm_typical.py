import itertools
from typing import List, Union
import re

import numpy as np

from quara.objects.composite_system import CompositeSystem
from quara.objects.matrix_basis import (
    MatrixBasis,
    calc_matrix_expansion_coefficient,
    calc_hermitian_matrix_expansion_coefficient_hermitian_basis,
)
from quara.objects.povm import Povm
from quara.objects.state_typical import (
    generate_state_density_mat_from_name,
    generate_state_pure_state_vector_from_name,
)
from quara.utils.matrix_util import truncate_hs, calc_mat_from_vector_adjoint


def get_povm_object_names() -> List[str]:
    """Return the list of valid povm-related object names.

    Returns
    -------
    List[str]
        the list of valid povm-related object names.
    """
    names = ["pure_state_vectors", "matrices", "vectors", "povm"]
    return names


def get_povm_names() -> List[str]:
    """Return the list of valid povm names.

    Returns
    -------
    List[str]
        the list of valid povm names.
    """
    names = []
    names += get_povm_names_rank1()
    names += get_povm_names_not_rank1()
    return names


def get_povm_names_rank1() -> List[str]:
    """Return the list of valid povm names of rank 1.

    Returns
    -------
    List[str]
        the list of valid povm names of rank 1.
    """
    names = [
        "x",
        "y",
        "z",
        "bell",
        "z3",
        "01x3",
        "02x3",
        "01z3",
        "21y3",
    ]
    return names


def get_povm_names_not_rank1() -> List[str]:
    """Return the list of valid povm names of not rank 1.

    Returns
    -------
    List[str]
        the list of valid povm names of not rank 1.
    """
    names = ["z2"]
    return names


def generate_povm_object_from_povm_name_object_name(
    povm_name: str,
    object_name: str,
    c_sys: CompositeSystem = None,
    basis: MatrixBasis = None,
) -> Union[List[np.array], Povm]:
    """Return a povm-related object.

    Parameters
    ----------
    povm_name : str
        Valid gate_name. It is given by :func:`~quara.objects.povm_typical.get_povm_names()`
    object_name : str
        Valid object_name. It is given by :func:`~quara.objects.povm_typical.get_povm_object_names`
    c_sys : CompositeSystem, optional
        To be given for object_name = 'povm', by default None.
    basis : MatrixBasis, optional
        To be given for object_name = 'vectors', by default None.

    Returns
    -------
    Union[List[np.array], Povm]
        np.array
            pure state vectors related elements of POVM for object_name = 'pure_state_vectors'
                Complex vectors
            list of elements of POVM(matrices) for object_name = 'matrices'
                Complex matrices
            vectors on Hermitian basis for object_name = 'vectors'
                Real vectors
        Povm
            Povm class for object_name = 'povm'

    Raises
    ------
    ValueError
        [description]
    """
    if object_name == "pure_state_vectors":
        obj = generate_povm_pure_state_vectors_from_name(povm_name)
    elif object_name == "matrices":
        obj = generate_povm_matrices_from_name(povm_name)
    elif object_name == "vectors":
        obj = generate_povm_vectors_from_name(povm_name, basis)
    elif object_name == "povm":
        obj = generate_povm_from_name(povm_name, c_sys)
    else:
        raise ValueError(f"object_name is out of range. object_name={object_name}")
    return obj


def _generate_povm_pure_state_vectors_from_single_name(
    povm_name: str,
) -> List[np.array]:
    if povm_name not in get_povm_names_rank1():
        raise ValueError(f"povm_name is not rank 1. povm_name={povm_name}")

    if povm_name == "x":
        pure_state_vector_names = ["x0", "x1"]
    elif povm_name == "y":
        pure_state_vector_names = ["y0", "y1"]
    elif povm_name == "z":
        pure_state_vector_names = ["z0", "z1"]
    elif povm_name == "bell":
        pure_state_vector_names = [
            "bell_phi_plus",
            "bell_phi_minus",
            "bell_psi_plus",
            "bell_psi_minus",
        ]
    elif povm_name == "z3":
        pure_state_vector_names = ["01z0", "01z1", "02z1"]
    elif povm_name == "01x3":
        pure_state_vector_names = ["01x0", "01x1", "02z1"]
    elif povm_name == "02x3":
        pure_state_vector_names = ["02x0", "02x1", "01z1"]
    elif povm_name == "01z3":
        pure_state_vector_names = ["01z0", "01z1", "02z1"]
    elif povm_name == "21y3":
        pure_state_vector_names = ["12y0", "12y1", "01z0"]

    vectors = [
        generate_state_pure_state_vector_from_name(pure_state_vector_name)
        for pure_state_vector_name in pure_state_vector_names
    ]
    return vectors


def generate_povm_pure_state_vectors_from_name(povm_name: str) -> List[np.array]:
    """returns pure state vectors.

    Parameters
    ----------
    povm_name : str
        name of povm.

    Returns
    -------
    List[np.array]
        pure state vectors.

    Raises
    ------
    ValueError
        povm_name is invalid.
    """
    # split and get each pure state vectors
    single_povm_names = povm_name.split("_")
    pure_state_vectors_list = [
        _generate_povm_pure_state_vectors_from_single_name(single_povm_name)
        for single_povm_name in single_povm_names
    ]

    # tensor product
    temp = pure_state_vectors_list[0]
    for pure_state_vectors in pure_state_vectors_list[1:]:
        temp = [
            np.kron(vec1, vec2)
            for vec1, vec2 in itertools.product(temp, pure_state_vectors)
        ]

    return temp


def _generate_povm_matrices_from_single_name(povm_name: str) -> List[np.array]:
    if povm_name in get_povm_names_rank1():
        pure_state_vectors = generate_povm_pure_state_vectors_from_name(povm_name)
        matrices = [
            calc_mat_from_vector_adjoint(pure_state_vector)
            for pure_state_vector in pure_state_vectors
        ]
    else:
        if povm_name == "z2":
            matrices = [
                generate_state_density_mat_from_name("01z0"),
                generate_state_density_mat_from_name("01z1")
                + generate_state_density_mat_from_name("02z1"),
            ]
    return matrices


def generate_povm_matrices_from_name(povm_name: str) -> List[np.array]:
    """returns list of elements of POVM(matrices).

    Parameters
    ----------
    povm_name : str
        name of povm.

    Returns
    -------
    List[np.array]
        list of elements of POVM(matrices).

    Raises
    ------
    ValueError
        povm_name is invalid.
    """
    # split and get each pure state vectors
    single_povm_names = povm_name.split("_")
    matrices_list = [
        _generate_povm_matrices_from_single_name(single_povm_name)
        for single_povm_name in single_povm_names
    ]

    # tensor product
    temp = matrices_list[0]
    for matrices in matrices_list[1:]:
        temp = [np.kron(vec1, vec2) for vec1, vec2 in itertools.product(temp, matrices)]

    return temp


def generate_povm_vectors_from_name(
    povm_name: str, basis: MatrixBasis
) -> List[np.array]:
    """returns vectors on Hermitian basis.

    Parameters
    ----------
    povm_name : str
        name of povm.
    basis : MatrixBasis
        Hermitian basis of povm.

    Returns
    -------
    List[np.array]
        vectors on Hermitian basis.

    Raises
    ------
    ValueError
        povm_name is invalid.
    ValueError
        basis is not Hermitian.
    """
    matrices = generate_povm_matrices_from_name(povm_name)
    vecs = [
        calc_hermitian_matrix_expansion_coefficient_hermitian_basis(matrix, basis)
        for matrix in matrices
    ]
    return vecs


def generate_povm_from_name(povm_name: str, c_sys: CompositeSystem) -> Povm:
    """returns Povm class.

    Parameters
    ----------
    povm_name : str
        name of povm.
    c_sys : CompositeSystem
        CompositeSystem of povm.

    Returns
    -------
    Povm
        Povm class.

    Raises
    ------
    ValueError
        povm_name is invalid.
    """
    vecs = generate_povm_vectors_from_name(povm_name, c_sys.basis())
    return Povm(c_sys, vecs)
