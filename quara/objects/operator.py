import copy
import itertools
from typing import List, Union

import numpy as np

from quara.objects.composite_system import CompositeSystem
from quara.objects.gate import Gate
from quara.objects.matrix_basis import MatrixBasis
from quara.objects.state import State


def tensor_product(*elements) -> Union[MatrixBasis, State]:
    """calculates tensor product of ``elements``.
    
    this function can calculate tensor product of the following combinations of types:

    - (MatrixBasis, MatrixBasis)
    - (State, State)
    - list conststs of these combinations

    Returns
    -------
    Union[MatrixBasis, State]
        tensor product of ``elements``

    Raises
    ------
    ValueError
        Unsupported type combination.
    """

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
        # create CompositeSystem
        e_sys_list = copy.copy(elem1._composite_system._elemental_systems)
        e_sys_list.extend(elem2._composite_system._elemental_systems)
        c_sys = CompositeSystem(e_sys_list)
        # calculate vecs of stetes
        tensor_vec = np.kron(elem1._vec, elem2._vec)

        # create State
        tensor_state = State(c_sys, tensor_vec)
        return tensor_state
    else:
        raise ValueError(
            f"Unsupported type combination! type=({type(elem1)}, {type(elem2)})"
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
    if type(elem1) == Gate and type(elem2) == Gate:
        # TODO check same CompositeSystem

        # create Gate
        matrix = elem1.hs @ elem2.hs
        gate = Gate(elem1._composite_system, matrix)
        return gate
    elif type(elem1) == Gate and type(elem2) == State:
        # TODO check same CompositeSystem

        # create State
        vec = elem1.hs @ elem2.vec
        state = State(elem1._composite_system, vec.real.astype(np.float64))
        return state
    else:
        raise ValueError(
            f"Unsupported type combination! type=({type(elem1)}, {type(elem2)})"
        )


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
