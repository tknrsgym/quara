import copy
import itertools
from typing import List, Union

import numpy as np

from quara.objects.composite_system import CompositeSystem
from quara.objects.gate import Gate
from quara.objects.matrix_basis import MatrixBasis
from quara.objects.povm import Povm
from quara.objects.state import State


def tensor_product(*elements) -> Union[Gate, MatrixBasis, State]:
    """calculates tensor product of ``elements``.

    this function can calculate tensor product of the following combinations of types:

    - (Gate, Gate)
    - (MatrixBasis, MatrixBasis)
    - (State, State)
    - (Povm, Povm)
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


def _tensor_product_Gate_Gate(gate1: Gate, gate2: Gate) -> Gate:
    # create CompositeSystem
    e_sys_list = list(gate1._composite_system._elemental_systems)
    e_sys_list.extend(gate2._composite_system._elemental_systems)
    c_sys = CompositeSystem(e_sys_list)

    # How to calculate HS(g1 \otimes g2)
    #
    # notice:
    #   HS(g1 \otimes g2) != HS(g1) \otimes HS(g2).
    #   so, we convert "entries of |HS(g1)>> \otimes |HS(g2)>>" to "entries of |HS(g1 \otimes g2)>>".
    #
    # notations:
    #   let E1 be d1-dim square matrix and e_{m1, n1} be the matrix its (m1, n1) entry is 1, otherwise 0.
    #   let E2 be d2-dim square matrix and e_{m2, n2} be the matrix its (m2, n2) entry is 1, otherwise 0.
    #
    # method:
    # - (|e_{m1, n1}>> \otimes |e_{m2, n2}>>)_{d1*d2*(d1*m1+n1) + d2*m2+n2} = 1. other entry is 0.
    # - on the other hand,
    #   e_{m1, n1} \otimes e_{m2, n2} = e_{d1*m1+m2, d1*n1+n2}.
    #   so, |e_{m1, n1} \otimes e_{m2, n2}>>_{d1+d2(d1*m1+m2) + d1*n1+n2} = 1. other entry is 0.
    # =>
    #   convert d1*d2*(d1*m1+n1) + d2*m2+n2 entry of |e_{m1, n1}>> \otimes  |e_{m2, n2}>>.
    #   to d1*d2*(d1*m1+m2) + d1*n1+n2 entry of |e_{m1, n1} \otimes e_{m2, n2}>>

    d1 = gate1.dim ** 2
    d2 = gate2.dim ** 2

    # calculate |HS(g1)>> \otimes |HS(g2)>>
    from_vec = np.kron(gate1.hs.flatten(), gate2.hs.flatten())

    # convert |HS(g1)>> \otimes |HS(g2)>> to |HS(g1 \otimes g2)>>
    vec_entries = []
    for m1 in range(d1):
        for m2 in range(d2):
            for n1 in range(d1):
                for n2 in range(d2):
                    vec_entries.append(
                        from_vec[d1 * d2 * (d1 * m1 + n1) + d2 * m2 + n2]
                    )
    to_hs = np.array(vec_entries).reshape((d1 * d2, d1 * d2))
    gate = Gate(c_sys, to_hs)
    return gate


def _tensor_product(elem1, elem2):
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
        # create CompositeSystem
        e_sys_list = list(elem1._composite_system._elemental_systems)
        e_sys_list.extend(elem2._composite_system._elemental_systems)
        c_sys = CompositeSystem(e_sys_list)
        # calculate vecs of stetes
        tensor_vec = np.kron(elem1._vec, elem2._vec)

        # create State
        tensor_state = State(c_sys, tensor_vec)
        return tensor_state
    elif type(elem1) == Povm and type(elem2) == Povm:
        # Povm (x) Povm -> Povm
        e_sys_list = list(elem1.composite_system.elemental_systems)
        e_sys_list.extend(elem2.composite_system.elemental_systems)
        c_sys = CompositeSystem(e_sys_list)

        tensor_vecs = []
        # TODO: It is working in progress
        for vec1, vec2 in zip(elem1.vecs, elem2.vecs):
            tensor_vec = np.kron(vec1, vec2)
            tensor_vecs.append(tensor_vec)

        # or
        # tensor_vecs = [
        #     np.kron(vec1, vec2) for vec1, vec2 in itertools.product(elem1.vecs, elem2.vecs)
        # ]

        tensor_povm = Povm(c_sys, tensor_vecs)
        return tensor_povm
    else:
        raise ValueError(
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
    ValueError
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
        vecs = [povm_element.conjugate() @ elem2.hs for povm_element in elem1._vecs]
        povm = Povm(elem1._composite_system, vecs)
        return povm
    elif type(elem1) == Povm and type(elem2) == State:
        # calculate probability distribution
        prob = [np.vdot(povm_element, elem2._vec) for povm_element in elem1._vecs]
        return prob
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
