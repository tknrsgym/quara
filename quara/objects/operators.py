import copy
from functools import reduce
from operator import add, mul
import itertools
from typing import List, Tuple, Union

import numpy as np

from quara.objects.composite_system import CompositeSystem
from quara.objects.elemental_system import ElementalSystem
from quara.objects.gate import Gate
from quara.objects.matrix_basis import MatrixBasis
from quara.objects.povm import Povm
from quara.objects.state import State


def tensor_product(*elements) -> Union[MatrixBasis, State, Povm, Gate]:
    """calculates tensor product of ``elements``.

    this function can calculate tensor product of the following combinations of types:

    - (Gate, Gate) -> Gate
    - (MatrixBasis, MatrixBasis) -> MatrixBasis
    - (State, State) -> State
    - (Povm, Povm) -> Povm
    - list conststs of these combinations

    Returns
    -------
    Union[MatrixBasis, State, Povm, Gate]
        tensor product of ``elements``

    Raises
    ------
    TypeError
        Unsupported type combination.
    """

    # convert argument to list
    element_list = _to_list(*elements)

    # recursively calculate tensor products(calculate from head to tail of list)
    temp = element_list[0]
    for elem in element_list[1:]:
        temp = _tensor_product(temp, elem)
    return temp


def _U(dim1, dim2, i, j):
    matrix = np.zeros((dim1, dim2))
    matrix[i, j] = 1
    return matrix


def _K(dim1: int, dim2: int) -> np.array:
    matrix = np.zeros((dim1 * dim2, dim1 * dim2))
    for row in range(dim1):
        for col in range(dim2):
            matrix += np.kron(_U(dim1, dim2, row, col), _U(dim2, dim1, col, row))

    return matrix


def _check_cross_elemental_system_position(e_sys_list: List[ElementalSystem]) -> int:
    # check cross ElementalSystem position
    # let [0, 10, 5] be a list of names of ElementalSystem, this functions returns 2(position of value 1)
    former_name = None
    for current_position, e_sys in enumerate(e_sys_list):
        current_name = e_sys.name
        if not former_name is None and former_name > current_name:
            return current_position
        else:
            former_name = current_name

    # if cross ElementalSystem position does not exist, returns None
    return None


def _permutation_matrix_for_vec(
    position: int, e_sys_list: List[ElementalSystem], dim_list: List[int]
) -> np.array:
    # identity matrix for head of permutation matrix
    if position < 2:
        I_head = np.eye(1)
    else:
        size = reduce(add, dim_list[: position - 1])
        I_head = np.eye(size)

    # create matrix K
    K_matrix = _K(dim_list[position], dim_list[position - 1])

    # identity matrix for tail of permutation matrix
    if position < len(dim_list) - 1:
        size = reduce(add, dim_list[position + 1 :])
        I_tail = np.eye(size)
    else:
        I_tail = np.eye(1)

    # calculate permutation matrix
    perm_matrix = np.kron(np.kron(I_head, K_matrix), I_tail)
    return perm_matrix


def _permutation_matrix_for_matrix(
    position: int, e_sys_list: List[ElementalSystem], dim_list: List[int]
) -> Tuple[np.array, np.array]:
    # identity matrix for head of permutation matrix
    if position < 2:
        I_head = np.eye(1)
    else:
        size = reduce(add, dim_list[: position - 1])
        I_head = np.eye(size)

    # create matrix K
    left_K_matrix = _K(dim_list[position], dim_list[position - 1])
    right_K_matrix = _K(dim_list[position - 1], dim_list[position])

    # identity matrix for tail of permutation matrix
    if position < len(dim_list) - 1:
        size = reduce(add, dim_list[position + 1 :])
        I_tail = np.eye(size)
    else:
        I_tail = np.eye(1)

    # calculate permutation matrix
    left_perm_matrix = np.kron(np.kron(I_head, left_K_matrix), I_tail)
    right_perm_matrix = np.kron(np.kron(I_head, right_K_matrix), I_tail)
    return left_perm_matrix, right_perm_matrix


def _tensor_product_Gate_Gate(gate1: Gate, gate2: Gate) -> Gate:
    # create CompositeSystem
    e_sys_list = list(gate1._composite_system._elemental_systems)
    e_sys_list.extend(gate2._composite_system._elemental_systems)
    c_sys = CompositeSystem(e_sys_list)

    # How to calculate HS(g1 \otimes g2)
    #
    # notice:
    #   HS(g1 \otimes g2) != HS(g1) \otimes HS(g2).
    #   so, we convert "|HS(g1)>> \otimes |HS(g2)>>" to "|HS(g1 \otimes g2)>>".
    #
    # method:
    #   use vec-permutation matrix.
    #   see "Matrix Algebra From a Statistician's Perspective" section 16.3.

    # calculate |HS(g1)>> \otimes |HS(g2)>>
    from_vec = np.kron(gate1.hs.flatten(), gate2.hs.flatten())

    # convert |HS(g1)>> \otimes |HS(g2)>> to |HS(g1 \otimes g2)>>
    d1 = gate1.dim ** 2
    d2 = gate2.dim ** 2
    permutation = np.kron(np.kron(np.eye(d1), _K(d2, d1)), np.eye(d2))
    to_vec = permutation @ from_vec
    to_hs = to_vec.reshape((d1 * d2, d1 * d2))

    # permutation the tensor product matrix according to the position of the sorted ElementalSystem
    # see "Matrix Algebra From a Statistician's Perspective" section 16.3.
    position = _check_cross_elemental_system_position(e_sys_list)
    while not position is None:
        dim_list = [e_sys.dim ** 2 for e_sys in e_sys_list]
        left_perm, right_perm = _permutation_matrix_for_matrix(
            position, e_sys_list, dim_list
        )
        # B \otimes A = perm_matrix @ (A \otimes B)
        to_hs = left_perm @ to_hs @ right_perm
        # swap e_sys_list
        e_sys_list[position - 1], e_sys_list[position] = (
            e_sys_list[position],
            e_sys_list[position - 1],
        )
        position = _check_cross_elemental_system_position(e_sys_list)

    # create Gate
    is_physical = gate1.is_physical and gate2.is_physical
    gate = Gate(c_sys, to_hs, is_physical=is_physical)
    return gate


def _tensor_product_State_State(state1: State, state2: State) -> State:
    e_sys_list = list(state1._composite_system.elemental_systems)
    e_sys_list.extend(state2._composite_system.elemental_systems)

    tensor_vec = np.kron(state1.vec, state2.vec)

    # permutation the tensor product matrix according to the position of the sorted ElementalSystem
    # see "Matrix Algebra From a Statistician's Perspective" section 16.3.
    position = _check_cross_elemental_system_position(e_sys_list)
    while not position is None:
        dim_list = [e_sys.dim ** 2 for e_sys in e_sys_list]
        perm_matrix = _permutation_matrix_for_vec(position, e_sys_list, dim_list)
        # B \otimes A = perm_matrix @ (A \otimes B)
        tensor_vec = perm_matrix @ tensor_vec
        # swap e_sys_list
        e_sys_list[position - 1], e_sys_list[position] = (
            e_sys_list[position],
            e_sys_list[position - 1],
        )
        position = _check_cross_elemental_system_position(e_sys_list)

    # create State
    c_sys = CompositeSystem(e_sys_list)
    is_physical = state1.is_physical and state2.is_physical
    return State(c_sys, tensor_vec, is_physical=is_physical)


def _tensor_product_Povm_Povm(povm1: Povm, povm2: Povm) -> Povm:
    # Povm (x) Povm -> Povm
    e_sys_list = list(povm1.composite_system.elemental_systems)
    e_sys_list.extend(povm2.composite_system.elemental_systems)
    c_sys = CompositeSystem(e_sys_list)

    tensor_vecs = [
        np.kron(vec1, vec2) for vec1, vec2 in itertools.product(povm1.vecs, povm2.vecs)
    ]

    is_physical = povm1.is_physical and povm2.is_physical

    tensor_povm = Povm(c_sys, tensor_vecs, is_physical=is_physical)
    return tensor_povm


def _tensor_product(elem1, elem2) -> Union[MatrixBasis, State, Povm, Gate]:
    # implement tensor product calculation for each type
    if type(elem1) == Gate and type(elem2) == Gate:
        return _tensor_product_Gate_Gate(elem1, elem2)

    elif type(elem1) == MatrixBasis and type(elem2) == MatrixBasis:
        new_basis = [
            np.kron(val1, val2) for val1, val2 in itertools.product(elem1, elem2)
        ]
        m_basis = MatrixBasis(new_basis)
        return m_basis
    elif type(elem1) == State and type(elem2) == State:
        return _tensor_product_State_State(elem1, elem2)

    elif type(elem1) == Povm and type(elem2) == Povm:
        # Povm (x) Povm -> Povm
        return _tensor_product_Povm_Povm(elem1, elem2)
    else:
        raise TypeError(
            f"Unsupported type combination! type=({type(elem1)}, {type(elem2)})"
        )


def composite(*elements) -> Union[Gate, Povm, State, List[float]]:
    """calculates composite of ``elements``.

    this function can calculate composite of the following combinations of types:

    - (Gate, Gate) -> Gate
    - (Gate, State) -> State
    - (Povm, Gate) -> Povm
    - (Povm, State) -> List[float] (probability distribution)
    - list conststs of these combinations

    Returns
    -------
    Union[Gate, State]
        composite of ``elements``

    Raises
    ------
    TypeError
        Unsupported type combination.
    """
    # convert argument to list
    element_list = _to_list(*elements)

    # recursively calculate composite(calculate from tail to head of list)
    temp = element_list[-1]
    for elem in reversed(element_list[:-1]):
        temp = _composite(elem, temp)
    return temp


def _composite(elem1, elem2):
    # check CompositeSystem
    if elem1._composite_system != elem2._composite_system:
        raise ValueError(f"Cannot composite different composite systems.")

    # implement composite calculation for each type
    if type(elem1) == Gate and type(elem2) == Gate:
        # create Gate
        matrix = elem1.hs @ elem2.hs
        gate = Gate(elem1._composite_system, matrix)
        return gate
    elif type(elem1) == Gate and type(elem2) == State:
        # create State
        vec = elem1.hs @ elem2.vec
        state = State(elem1._composite_system, vec.real.astype(np.float64))
        return state
    elif type(elem1) == Povm and type(elem2) == Gate:
        # calculate Povm
        vecs = [povm_element.conjugate() @ elem2.hs for povm_element in elem1.vecs]
        povm = Povm(elem1._composite_system, vecs)
        return povm
    elif type(elem1) == Povm and type(elem2) == State:
        # calculate probability distribution
        prob = [np.vdot(povm_element, elem2.vec) for povm_element in elem1.vecs]
        return prob
    else:
        raise TypeError(
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
