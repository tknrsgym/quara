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
from quara.objects import state_typical
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
    """Return the list of valid povm names of rank 1."""
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
    """Return the list of valid povm names of not rank 1."""
    names = ["z2"]
    return names


def generate_povm_object_from_povm_name_object_name(
    povm_name: str,
    object_name: str,
    c_sys: CompositeSystem = None,
    basis: MatrixBasis = None,
) -> Union[List[np.array], Povm]:
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


def generate_povm_pure_state_vectors_from_name(povm_name: str) -> List[np.array]:
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

    vectors = []
    for pure_state_vector_name in pure_state_vector_names:
        method_name = (
            "state_typical.get_state_" + pure_state_vector_name + "_pure_state_vector"
        )
        method = eval(method_name)
        vectors.append(method())
    return vectors


def generate_povm_matrices_from_name(povm_name: str) -> List[np.array]:
    if povm_name in get_povm_names_rank1():
        pure_state_vectors = generate_povm_pure_state_vectors_from_name(povm_name)
        matrices = [
            calc_mat_from_vector_adjoint(pure_state_vector)
            for pure_state_vector in pure_state_vectors
        ]
    else:
        if povm_name == "z2":
            matrices = [
                state_typical.generate_state_density_mat_from_name("01z0"),
                state_typical.generate_state_density_mat_from_name("01z1")
                + state_typical.generate_state_density_mat_from_name("02z1"),
            ]
    return matrices


def generate_povm_vectors_from_name(
    povm_name: str, basis: MatrixBasis
) -> List[np.array]:
    matrices = generate_povm_matrices_from_name(povm_name)
    vecs = [
        calc_hermitian_matrix_expansion_coefficient_hermitian_basis(matrix, basis)
        for matrix in matrices
    ]
    return vecs


def generate_povm_from_name(povm_name: str, c_sys: CompositeSystem) -> Povm:
    vecs = generate_povm_vectors_from_name(povm_name, c_sys.basis)
    return Povm(c_sys, vecs)
