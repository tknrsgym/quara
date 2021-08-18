from typing import List, Union
import numpy as np

from quara.objects.composite_system import CompositeSystem
from quara.objects.gate import convert_hs
from quara.objects.mprocess import MProcess
from quara.objects.state_typical import (
    generate_state_density_mat_from_name,
    generate_state_pure_state_vector_from_name,
)
from quara.utils.matrix_util import calc_mat_from_vector_adjoint, truncate_hs


def get_mprocess_names_type1():
    names = [
        "x-type1",
        "y-type1",
        "z-type1",
        # "bell-type1",
        # "z3-type1",
        # "z2-type1",
    ]
    return names


def generate_mprocess_set_pure_state_vectors_from_name(
    mprocess_name: str,
) -> List[List[np.ndarray]]:
    # split and get each pure state vectors
    single_mprocess_names = mprocess_name.split("_")
    set_pure_state_vectors_list = [
        _generate_mprocess_set_pure_state_vectors_from_single_name(single_mprocess_name)
        for single_mprocess_name in single_mprocess_names
    ]

    # tensor product
    temp = set_pure_state_vectors_list[0]
    for pure_state_vectors in set_pure_state_vectors_list[1:]:
        temp = [np.kron(vec1, vec2) for vec1, vec2 in product(temp, pure_state_vectors)]

    return temp


def _generate_mprocess_set_pure_state_vectors_from_single_name(
    mprocess_name: str,
) -> List[np.ndarray]:
    if mprocess_name not in get_mprocess_names_type1():
        raise ValueError(f"mprocess_name is not type 1. mprocess_name={mprocess_name}")

    method_name = (
        "get_mprocess_" + mprocess_name.replace("-", "") + "_set_pure_state_vectors"
    )
    method = eval(method_name)
    set_pure_state_vectors = method()
    return set_pure_state_vectors


def generate_mprocess_set_kraus_matrices_from_name(
    mprocess_name: str,
) -> List[List[np.ndarray]]:
    set_kraus_matrices = []
    if mprocess_name in get_mprocess_names_type1():
        set_pure_state_vectors = generate_mprocess_set_pure_state_vectors_from_name(
            mprocess_name
        )

        for pure_state_vectors in set_pure_state_vectors:
            matrices = [
                calc_mat_from_vector_adjoint(pure_state_vector)
                for pure_state_vector in pure_state_vectors
            ]
            set_kraus_matrices.append(matrices)
    else:
        pass

    return set_kraus_matrices


def generate_mprocess_hss_from_name(
    mprocess_name: str, c_sys: CompositeSystem
) -> List[np.ndarray]:
    size = c_sys.dim ** 2
    hss = []
    if mprocess_name in get_mprocess_names_type1():
        hss_cb = []
        set_kraus_matrices = generate_mprocess_set_kraus_matrices_from_name(
            mprocess_name
        )
        for kraus_matrices in set_kraus_matrices:
            tmp_hs = np.zeros((size, size), dtype=np.complex128)
            for kraus_matrix in kraus_matrices:
                tmp_hs += np.kron(kraus_matrix, kraus_matrix.conjugate())
            hss_cb.append(tmp_hs)
        # TODO remove truncate_hs
        hss = [
            truncate_hs(convert_hs(hs_cb, c_sys.comp_basis(), c_sys.basis()))
            for hs_cb in hss_cb
        ]
    else:
        pass

    return hss


def generate_mprocess_from_name(c_sys: CompositeSystem, mprocess_name: str) -> MProcess:
    # TODO:
    hss = generate_mprocess_hss_from_name(mprocess_name, c_sys)
    mprocess = MProcess(hss=hss, c_sys=c_sys)
    return mprocess


def generate_mprocess_object_from_mprocess_name_object_name(
    mprocess_name: str, object_name: str, c_sys: CompositeSystem = None
) -> Union[MProcess, np.ndarray]:
    expected_object_names = [
        "set_pure_state_vectors",
        "set_kraus_matrices",
        "hss",
        "mprocess",
    ]

    if object_name not in expected_object_names:
        raise ValueError("object_name is out of range.")
    elif object_name == "set_pure_state_vectors":
        return generate_mprocess_set_pure_state_vectors_from_name(mprocess_name)
    elif object_name == "set_kraus_matrices":
        return generate_mprocess_set_kraus_matrices_from_name(mprocess_name)
    elif object_name == "hss":
        return generate_mprocess_hss_from_name(mprocess_name, c_sys)
    elif object_name == "mprocess":
        return generate_mprocess_from_name(c_sys, mprocess_name)


def get_mprocess_xtype1_set_pure_state_vectors() -> List[List[np.ndarray]]:
    set_pure_state_vectors = [
        [
            (1 / np.sqrt(2)) * np.array([1, 1], dtype=np.complex128),
        ],
        [
            (1 / np.sqrt(2)) * np.array([1, -1], dtype=np.complex128),
        ],
    ]
    return set_pure_state_vectors


def get_mprocess_ytype1_set_pure_state_vectors() -> List[List[np.ndarray]]:
    set_pure_state_vectors = [
        [
            (1 / np.sqrt(2)) * np.array([1, 1j], dtype=np.complex128),
        ],
        [
            (1 / np.sqrt(2)) * np.array([1, -1j], dtype=np.complex128),
        ],
    ]
    return set_pure_state_vectors


def get_mprocess_ztype1_set_pure_state_vectors() -> List[List[np.ndarray]]:
    set_pure_state_vectors = [
        [
            np.array([1, 0], dtype=np.complex128),
        ],
        [
            np.array([0, 1], dtype=np.complex128),
        ],
    ]
    return set_pure_state_vectors