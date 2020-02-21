import itertools
from typing import List

import numpy as np

import quara.objects.matrix_basis as m_basis
from quara.objects.composite_system import CompositeSystem
from quara.objects.matrix_basis import MatrixBasis
from quara.objects.state import State
import quara.utils.matrix_util as mutil


def tensor_product(*elements):
    # convert argument to list
    element_list = _to_list(*elements)

    # recursively calculate tensor products(calculate from head to tail of list)
    temp = element_list[0]
    for elem in element_list[1:]:
        temp = _tensor_product(temp, elem)
    return temp


def _tensor_product(elem1, elem2):
    # implement tensor product calculation for each type
    if type(elem1) == MatrixBasis and type(elem2) == MatrixBasis:
        new_basis = [
            np.kron(val1, val2) for val1, val2 in itertools.product(elem1, elem2)
        ]
        m_basis = MatrixBasis(new_basis)
        return m_basis
    elif type(elem1) == State and type(elem2) == State:
        # Stateを含むCompositeSystemを作成
        c_sys = CompositeSystem([elem1.composite_system, elem2.composite_system])
        # TODO Stateそのもののテンソル積を計算
        # |elem1>> \otimes |elem2>>の行列表現を計算し、|elem1 \otimes elem2>>の表現と解釈する

        return None
    else:
        raise ValueError(
            f"Unknown type combination! type=({type(elem1)}, {type(elem2)})"
        )


def composite(*elements):
    # convert argument to list
    element_list = _to_list(*elements)

    # recursively calculate composite(calculate from tail to head of list)
    temp = element_list[-1]
    for elem in reversed(element_list[:-1]):
        temp = _composite(elem, temp)
    return temp


def _composite(elem1, elem2):
    # implement composite calculation for each type
    pass


def _to_list(*elements):
    # convert argument to list
    element_list = []
    for element in elements:
        if type(element) == list:
            element_list.extend(element)
        else:
            element_list.append(element)

    # length of list must be at least two
    if len(element_list) < 2:
        raise ValueError(
            f"arguments must be at least two! arguments={len(element_list)})"
        )

    assert len(element_list) >= 2

    return element_list
