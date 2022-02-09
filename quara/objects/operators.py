import copy
import itertools
from functools import reduce
from operator import add, itemgetter
from typing import List, Tuple, Union

import numpy as np
from scipy.stats import multinomial

from quara.objects.composite_system import CompositeSystem
from quara.objects.elemental_system import ElementalSystem
from quara.objects.gate import Gate
from quara.objects.matrix_basis import SparseMatrixBasis, MatrixBasis, convert_vec
from quara.objects.povm import Povm
from quara.objects.state import State
from quara.objects.mprocess import MProcess
from quara.objects.state_ensemble import StateEnsemble
from quara.objects.multinomial_distribution import MultinomialDistribution
from quara.utils import matrix_util


def tensor_product(*elements) -> Union[SparseMatrixBasis, State, Povm, Gate]:
    """calculates tensor product of ``elements``.

    this function can calculate tensor product of the following combinations of types:

    - (Gate, Gate) -> Gate
    - (Gate, MProcess) -> MProcess
    - (MProcess, Gate) -> MProcess
    - (MProcess, MProcess) -> MProcess
    - (MatrixBasis, MatrixBasis) -> MatrixBasis
    - (State, State) -> State
    - (State, StateEnsemble) -> StateEnsemble
    - (StateEnsemble, State) -> StateEnsemble
    - (StateEnsemble, StateEnsemble) -> StateEnsemble
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


def _K(dim1: int, dim2: int) -> np.ndarray:
    matrix = np.zeros((dim1 * dim2, dim1 * dim2))
    for row in range(dim1):
        for col in range(dim2):
            matrix += np.kron(_U(dim1, dim2, row, col), _U(dim2, dim1, col, row))

    return matrix


def _permutation_matrix(
    position: int, dim_list: List[int]
) -> Tuple[np.ndarray, np.ndarray]:
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


def _check_cross_elemental_system_position(
    e_sys_list: List[ElementalSystem],
) -> Union[int, None]:
    # check cross ElementalSystem position
    # for example, if [0, 10, 5] is a list of names of ElementalSystem, then this functions returns 2(position of value 5)
    former_name = None
    for current_position, e_sys in enumerate(e_sys_list):
        current_name = e_sys.name
        if not former_name is None and former_name > current_name:
            return current_position
        else:
            former_name = current_name

    # if cross ElementalSystem position does not exist, returns None
    return None


def _tensor_product_hs_hs(
    hs1: np.ndarray, hs2: np.ndarray, e_sys_list: List[ElementalSystem]
) -> np.ndarray:
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
    from_vec = np.kron(hs1.flatten(), hs2.flatten())

    # convert |HS(g1)>> \otimes |HS(g2)>> to |HS(g1 \otimes g2)>>
    d1 = hs1.shape[0]
    d2 = hs2.shape[0]
    permutation = np.kron(np.kron(np.eye(d1), _K(d2, d1)), np.eye(d2))
    to_vec = permutation @ from_vec
    to_hs = to_vec.reshape((d1 * d2, d1 * d2))

    # permutate the tensor product matrix according to the position of the sorted ElementalSystem
    # see "Matrix Algebra From a Statistician's Perspective" section 16.3.
    system_order = [e_sys.name for e_sys in e_sys_list]
    size_list = [e_sys.dim ** 2 for e_sys in e_sys_list]
    perm_matrix = matrix_util.calc_permutation_matrix(system_order, size_list)
    to_hs = perm_matrix @ to_hs @ perm_matrix.T

    return to_hs


def _tensor_product_Gate_Gate(gate1: Gate, gate2: Gate) -> Gate:
    # create CompositeSystem
    e_sys_list = list(gate1.composite_system._elemental_systems)
    e_sys_list.extend(gate2.composite_system._elemental_systems)
    c_sys = CompositeSystem(e_sys_list)

    # calc HS(g1 \otimes g2)
    to_hs = _tensor_product_hs_hs(gate1.hs, gate2.hs, e_sys_list)

    # create Gate
    is_physicality_required = (
        gate1.is_physicality_required and gate2.is_physicality_required
    )
    gate = Gate(c_sys, to_hs, is_physicality_required=is_physicality_required)
    return gate


def _tensor_product_Gate_MProcess(elem1: MProcess, elem2: Gate) -> MProcess:
    # create CompositeSystem
    e_sys_list = list(elem1.composite_system._elemental_systems)
    e_sys_list.extend(elem2.composite_system._elemental_systems)
    c_sys = CompositeSystem(e_sys_list)

    # calc list of HS(g1 \otimes g2)
    hss = []
    for hs2 in elem2.hss:
        hs = _tensor_product_hs_hs(elem1.hs, hs2, e_sys_list)
        hss.append(hs)

    # create MProcess
    is_physicality_required = (
        elem1.is_physicality_required and elem2.is_physicality_required
    )
    mprocess = MProcess(
        c_sys,
        hss,
        shape=elem2.shape,
        is_physicality_required=is_physicality_required,
    )
    return mprocess


def _tensor_product_MProcess_Gate(elem1: MProcess, elem2: Gate) -> MProcess:
    # create CompositeSystem
    e_sys_list = list(elem1.composite_system._elemental_systems)
    e_sys_list.extend(elem2.composite_system._elemental_systems)
    c_sys = CompositeSystem(e_sys_list)

    # calc list of HS(g1 \otimes g2)
    hss = []
    for hs1 in elem1.hss:
        hs = _tensor_product_hs_hs(hs1, elem2.hs, e_sys_list)
        hss.append(hs)

    # create MProcess
    is_physicality_required = (
        elem1.is_physicality_required and elem2.is_physicality_required
    )
    mprocess = MProcess(
        c_sys,
        hss,
        shape=elem1.shape,
        is_physicality_required=is_physicality_required,
    )
    return mprocess


def _tensor_product_MProcess_MProcess(elem1: MProcess, elem2: MProcess) -> MProcess:
    # create CompositeSystem
    e_sys_list = list(elem1.composite_system._elemental_systems)
    e_sys_list.extend(elem2.composite_system._elemental_systems)
    c_sys = CompositeSystem(e_sys_list)

    # calc list of HS(g1 \otimes g2)
    hss = []
    for hs2 in elem2.hss:
        for hs1 in elem1.hss:
            hs = _tensor_product_hs_hs(hs1, hs2, e_sys_list)
            hss.append(hs)
    shape = elem1.shape + elem2.shape

    # create MProcess
    is_physicality_required = (
        elem1.is_physicality_required and elem2.is_physicality_required
    )
    mprocess = MProcess(
        c_sys,
        hss,
        shape=shape,
        is_physicality_required=is_physicality_required,
    )
    return mprocess


def _tensor_product_State_State(state1: State, state2: State) -> State:
    # create CompositeSystem
    e_sys_list = list(state1.composite_system.elemental_systems)
    e_sys_list.extend(state2.composite_system.elemental_systems)
    c_sys = CompositeSystem(e_sys_list)

    tensor_vec = np.kron(state1.vec, state2.vec)

    # permutate the tensor product matrix according to the position of the sorted ElementalSystem
    # see "Matrix Algebra From a Statistician's Perspective" section 16.3.
    system_order = [e_sys.name for e_sys in e_sys_list]
    size_list = [e_sys.dim ** 2 for e_sys in e_sys_list]
    perm_matrix = matrix_util.calc_permutation_matrix(system_order, size_list)
    tensor_vec = perm_matrix @ tensor_vec

    # create State
    is_physicality_required = (
        state1.is_physicality_required and state2.is_physicality_required
    )
    state = State(c_sys, tensor_vec, is_physicality_required=is_physicality_required)
    return state


def _tensor_product_StateEnsemble_StateEnsemble(
    elem1: StateEnsemble, elem2: StateEnsemble
) -> StateEnsemble:
    new_states = []
    new_prob_dist = []
    for i, state1 in enumerate(elem1.states):
        for j, state2 in enumerate(elem2.states):
            new_state = tensor_product(state1, state2)
            new_p = elem1.prob_dist[i] * elem2.prob_dist[j]
            new_states.append(new_state)
            new_prob_dist.append(new_p)
    shape = tuple(list(elem1.prob_dist.shape) + list(elem2.prob_dist.shape))
    new_md = MultinomialDistribution(new_prob_dist, shape=shape)
    return StateEnsemble(new_states, new_md)


def _tensor_product_Povm_Povm(povm1: Povm, povm2: Povm) -> Povm:
    # Povm (x) Povm -> Povm
    e_sys_list = list(povm1.composite_system.elemental_systems)
    e_sys_list.extend(povm2.composite_system.elemental_systems)
    c_sys = CompositeSystem(e_sys_list)

    tensor_vecs = [
        np.kron(vec1, vec2) for vec1, vec2 in itertools.product(povm1.vecs, povm2.vecs)
    ]

    # permutate the tensor product matrix according to the position of the sorted ElementalSystem
    # see "Matrix Algebra From a Statistician's Perspective" section 16.3.

    # permutate each tensor vecs
    system_order = [e_sys.name for e_sys in e_sys_list]
    size_list = [e_sys.dim ** 2 for e_sys in e_sys_list]
    perm_matrix = matrix_util.calc_permutation_matrix(system_order, size_list)
    tensor_vecs = [perm_matrix @ tensor_vec for tensor_vec in tensor_vecs]

    # permutate list of tensor vecs
    nums_local_outcomes = copy.copy(povm1.nums_local_outcomes)
    nums_local_outcomes.extend(povm2.nums_local_outcomes)
    perm_matrix = matrix_util.calc_permutation_matrix(system_order, nums_local_outcomes)
    tensor_vecs = matrix_util.convert_list_by_permutation_matrix(
        tensor_vecs, perm_matrix
    )

    # permutate nums_local_outcomes
    system_outcomes = [
        (system_name, num_outcome)
        for system_name, num_outcome in zip(system_order, nums_local_outcomes)
    ]
    system_outcomes = sorted(system_outcomes, key=itemgetter(0))
    new_nums_local_outcomes = [system_outcome[1] for system_outcome in system_outcomes]

    # create Povm
    is_physicality_required = (
        povm1.is_physicality_required and povm2.is_physicality_required
    )
    tensor_povm = Povm(
        c_sys, tensor_vecs, is_physicality_required=is_physicality_required
    )
    tensor_povm._nums_local_outcomes = new_nums_local_outcomes
    return tensor_povm


def _tensor_product(elem1, elem2) -> Union[SparseMatrixBasis, State, Povm, Gate]:
    # implement tensor product calculation for each type
    if type(elem1) == Gate and type(elem2) == Gate:
        # Gate (x) Gate -> Gate
        return _tensor_product_Gate_Gate(elem1, elem2)
    elif type(elem1) == Gate and type(elem2) == MProcess:
        # Gate (x) MProcess -> MProcess
        return _tensor_product_Gate_MProcess(elem1, elem2)
    elif type(elem1) == MProcess and type(elem2) == Gate:
        # MProcess (x) Gate -> MProcess
        return _tensor_product_MProcess_Gate(elem1, elem2)
    elif type(elem1) == MProcess and type(elem2) == MProcess:
        # MProcess (x) MProcess -> MProcess
        return _tensor_product_MProcess_MProcess(elem1, elem2)
    elif type(elem1) == SparseMatrixBasis and type(elem2) == SparseMatrixBasis:
        # MatrixBasis (x) MatrixBasis -> MatrixBasis
        new_basis = [
            matrix_util.kron(val1, val2)
            for val1, val2 in itertools.product(elem1, elem2)
        ]
        m_basis = SparseMatrixBasis(new_basis)
        return m_basis
    elif type(elem1) == MatrixBasis and type(elem2) == MatrixBasis:
        # MatrixBasis (x) MatrixBasis -> MatrixBasis
        new_basis = [
            matrix_util.kron(val1, val2)
            for val1, val2 in itertools.product(elem1, elem2)
        ]
        m_basis = MatrixBasis(new_basis)
        return m_basis
    elif type(elem1) == State and type(elem2) == State:
        # State (x) State -> State
        return _tensor_product_State_State(elem1, elem2)
    elif type(elem1) == State and type(elem2) == StateEnsemble:
        # State (x) StateEnsemble -> StateEnsemble
        new_states = [tensor_product(elem1, state) for state in elem2.states]
        return StateEnsemble(new_states, elem2.prob_dist)
    elif type(elem1) == StateEnsemble and type(elem2) == State:
        # StateEnsemble (x) State -> StateEnsemble
        new_states = [tensor_product(state, elem2) for state in elem1.states]
        return StateEnsemble(new_states, elem1.prob_dist)
    elif type(elem1) == StateEnsemble and type(elem2) == StateEnsemble:
        # StateEnsemble (x) StateEnsemble -> StateEnsemble
        return _tensor_product_StateEnsemble_StateEnsemble(elem1, elem2)
    elif type(elem1) == Povm and type(elem2) == Povm:
        # Povm (x) Povm -> Povm
        return _tensor_product_Povm_Povm(elem1, elem2)
    else:
        raise TypeError(
            f"Unsupported type combination! type=({type(elem1)}, {type(elem2)})"
        )


def compose_qoperations(
    *elements,
) -> Union[
    Gate, Povm, State, List[float], MProcess, StateEnsemble, MultinomialDistribution
]:
    """calculates composition of qoperations.

    this function can calculate composition of the following combinations of types:

    - (Gate, Gate) -> Gate
    - (Gate, MProcess) -> MProcess
    - (MProcess, Gate) -> MProcess
    - (MProcess, MProcess) -> MProcess
    - (Gate, State) -> State
    - (Gate, StateEnsemble) -> StateEnsemble
    - (MProcess, State) -> State or StateEnsemble
    - (Mprocess, StateEnsemble) -> StateEnsemble
    - (Povm, Gate) -> Povm
    - (Povm, MProcess) -> Povm
    - (Povm, State) -> MultinomialDistribution
    - (Povm, StateEnsemble) -> List[MultinomialDistribution]
    - list conststs of these combinations

    Returns
    -------
    Union[Gate, Povm, State, List[float], MProcess, StateEnsemble, MultinomialDistribution]
        composition of qoperations.

    Raises
    ------
    TypeError
        Unsupported type combination.
    """
    # convert argument to list
    element_list = _to_list(*elements)
    # recursively calculate composition(calculate from tail to head of list)
    temp = element_list[-1]
    for elem in reversed(element_list[:-1]):
        temp = _compose_qoperations(elem, temp)
    return temp


def _compose_qoperations(elem1, elem2):
    # check CompositeSystem
    # Skip if elem1 or elem2 is StateEnsemble.
    if StateEnsemble not in {type(elem1), type(elem2)}:
        if elem1.composite_system != elem2.composite_system:
            raise ValueError(f"Cannot compose different composite systems.")

        # is_physicality_required
        is_physicality_required = (
            elem1.is_physicality_required and elem2.is_physicality_required
        )

    # implement compose calculation for each type
    if type(elem1) == Gate and type(elem2) == Gate:
        # create Gate
        matrix = elem1.hs @ elem2.hs
        gate = Gate(
            elem1.composite_system,
            matrix,
            is_physicality_required=is_physicality_required,
        )
        return gate
    elif type(elem1) == Gate and type(elem2) == MProcess:
        # -> MProcess
        hss = [elem1.hs @ hs for hs in elem2.hss]
        mprocess = MProcess(
            elem1.composite_system,
            hss,
            shape=elem2.shape,
            is_physicality_required=is_physicality_required,
        )
        return mprocess
    elif type(elem1) == MProcess and type(elem2) == Gate:
        # -> MProcess
        hss = [hs @ elem2.hs for hs in elem1.hss]
        mprocess = MProcess(
            elem1.composite_system,
            hss,
            shape=elem1.shape,
            is_physicality_required=is_physicality_required,
        )
        return mprocess
    elif type(elem1) == MProcess and type(elem2) == MProcess:
        # -> MProcess
        return _compose_qoperations_MProcess_MProcess(elem1, elem2)
    elif type(elem1) == Gate and type(elem2) == State:
        # create State
        vec = elem1.hs @ elem2.vec
        state = State(
            elem1.composite_system,
            vec.real.astype(np.float64),
            is_physicality_required=is_physicality_required,
        )
        return state
    elif type(elem1) == Gate and type(elem2) == StateEnsemble:
        # -> StateEnsemble
        new_states = []
        for state in elem2.states:
            new_state = compose_qoperations(elem1, state)
            new_states.append(new_state)
        return StateEnsemble(new_states, elem2.prob_dist)
    elif type(elem1) == MProcess and type(elem2) == State:
        # -> State or StateEnsemble
        state_ensemble = _compose_qoperations_MProcess_State(elem1, elem2)
        return state_ensemble
    elif type(elem1) == MProcess and type(elem2) == StateEnsemble:
        # -> StateEnsemble
        return _compose_qoperations_MProcess_StateEnsemble(elem1, elem2)
    elif type(elem1) == Povm and type(elem2) == Gate:
        # calculate Povm
        vecs = [povm_element.conjugate() @ elem2.hs for povm_element in elem1.vecs]
        povm = Povm(
            elem1.composite_system,
            vecs,
            is_physicality_required=is_physicality_required,
        )
        return povm
    elif type(elem1) == Povm and type(elem2) == MProcess:
        # -> Povm
        return _compose_qoperations_Povm_MProcess(elem1, elem2)
    elif type(elem1) == Povm and type(elem2) == State:
        # calculate probability distribution
        prob_list = [np.vdot(povm_element, elem2.vec) for povm_element in elem1.vecs]
        prob = np.array(prob_list, dtype=np.float64)
        prob = matrix_util.truncate_and_normalize(prob)
        dist = MultinomialDistribution(prob, prob.shape)
        return dist
    elif type(elem1) == Povm and type(elem2) == StateEnsemble:
        # -> MultinomialDistribution
        return _compose_qoperations_Povm_StateEnsemble(elem1, elem2)
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


def _compose_qoperations_MProcess_MProcess(
    elem1: MProcess, elem2: MProcess
) -> MProcess:
    # is_physicality_required
    is_physicality_required = (
        elem1.is_physicality_required and elem2.is_physicality_required
    )

    hss = []
    for hs2 in elem2.hss:
        for hs1 in elem1.hss:
            hss.append(hs2 @ hs1)
    shape = elem1.shape + elem2.shape

    mprocess = MProcess(
        elem1.composite_system,
        hss,
        shape,
        is_physicality_required=is_physicality_required,
    )
    return mprocess


def _compose_qoperations_MProcess_State_for_States(
    elem1: MProcess, elem2: State, weight: float = 1.0
) -> Tuple[State, List[float]]:
    # is_physicality_required
    is_physicality_required = (
        elem1.is_physicality_required and elem2.is_physicality_required
    )

    states = []
    ps = []
    Mx_rhos = []
    truncate = False
    if elem1.composite_system.is_orthonormal_hermitian_0thprop_identity:
        # calc Mx_rho and p_x before normalization
        for hs in elem1.hss:
            Mx_rho = hs @ elem2.vec
            p_x = np.sqrt(elem2.composite_system.dim) * Mx_rho[0]
            if weight * p_x <= elem1.eps_zero:
                p_x = 0
                truncate = True

            Mx_rhos.append(Mx_rho)
            ps.append(p_x)
    else:
        I_vec_cb = np.eye(elem1.composite_system.dim, dtype=np.float64).flatten()
        I_vec_gb = convert_vec(
            I_vec_cb,
            elem2.composite_system.comp_basis(),
            elem2.composite_system.basis(),
        )

        # calc Mx_rho and p_x before normalization
        for hs in elem1.hss:
            Mx_rho = hs @ elem2.vec
            p_x = np.vdot(I_vec_gb, Mx_rho)
            if weight * p_x <= elem1.eps_zero:
                p_x = 0
                truncate = True

            Mx_rhos.append(Mx_rho)
            ps.append(p_x)

    # normalize prob dist
    if truncate and np.sum(ps) != 0:
        ps = ps / np.sum(ps)

    # calc rho_x(vec of State) after normalization
    for Mx_rho, p_x in zip(Mx_rhos, ps):
        if p_x == 0:
            rho_x = np.zeros(elem2.vec.shape, dtype=elem2.vec.dtype)
            state = State(
                elem2.composite_system,
                rho_x,
                is_physicality_required=False,
            )
        else:
            rho_x = Mx_rho / p_x
            state = State(
                elem2.composite_system,
                rho_x,
                is_physicality_required=is_physicality_required,
            )
        states.append(state)

    ps = [weight * prob for prob in ps]

    return states, ps


def _compose_qoperations_MProcess_State(
    elem1: MProcess, elem2: State
) -> Union[State, StateEnsemble]:
    states, ps = _compose_qoperations_MProcess_State_for_States(elem1, elem2)

    if elem1.mode_sampling:
        # return State
        sample = multinomial.rvs(1, ps)
        sample_index = np.argmax(sample)
        return states[sample_index]
    else:
        # return StateEnsemble
        mult_dist = MultinomialDistribution(
            np.array(ps, dtype=np.float64), shape=elem1.shape
        )
        state_ens = StateEnsemble(states, mult_dist, eps_zero=elem1.eps_zero)
        return state_ens


def _compose_qoperations_MProcess_StateEnsemble(
    elem1: MProcess, elem2: StateEnsemble
) -> StateEnsemble:
    if elem2.prob_dist.is_zero_dist == True:
        # calc new ps
        shape = elem2.prob_dist.shape + elem1.shape
        length = 1
        for value in shape:
            length = length * value
        ps = [0.0] * length

        # calc new states
        states = [elem2.states[0].generate_zero_obj() for _ in range(len(ps))]
    else:
        states = []
        ps = []
        for state_old, prob in zip(elem2.states, elem2.prob_dist):
            states_local, ps_local = _compose_qoperations_MProcess_State_for_States(
                elem1, state_old, prob
            )
            states.extend(states_local)
            ps.extend(ps_local)

    eps_zero = max(elem1.eps_zero, elem2.eps_zero)
    if elem1.mode_sampling:
        # return StateEnsemble
        num_hss = len(elem1.hss)
        new_states = []
        for x_index, state in enumerate(elem2.states):
            local_ps = ps[x_index * num_hss : (x_index + 1) * num_hss]
            sample = multinomial.rvs(1, local_ps)
            sample_index = np.argmax(sample)
            new_states.append(states[x_index * num_hss + sample_index])
        mult_dist = MultinomialDistribution(
            np.array(elem2.prob_dist.ps, dtype=np.float64), shape=elem2.prob_dist.shape
        )
        state_ens = StateEnsemble(new_states, mult_dist, eps_zero=eps_zero)
        return state_ens
    else:
        # return StateEnsemble
        shape = elem2.prob_dist.shape + elem1.shape
        mult_dist = MultinomialDistribution(np.array(ps, dtype=np.float64), shape=shape)
        state_ens = StateEnsemble(states, mult_dist, eps_zero=eps_zero)
        return state_ens


def _compose_qoperations_Povm_MProcess(elem1: Povm, elem2: MProcess) -> Povm:
    # is_physicality_required
    is_physicality_required = (
        elem1.is_physicality_required and elem2.is_physicality_required
    )

    # calc vecs
    vecs = []
    for hs in elem2.hss:
        for vec in elem1.vecs:
            vecs.append(hs.T @ vec)

    povm = Povm(
        elem1.composite_system,
        vecs,
        is_physicality_required=is_physicality_required,
    )
    return povm


def _compose_qoperations_Povm_StateEnsemble(
    elem1: Povm, elem2: StateEnsemble
) -> MultinomialDistribution:
    for i, state in enumerate(elem2.states):
        # (Povm, State)

        if elem2.prob_dist[i] < elem2.eps_zero:
            prob_dist_size = elem1.num_outcomes
            ps = [0] * prob_dist_size
            ps = np.array(ps)
        else:
            prob_dist = compose_qoperations(elem1, state)
            ps = elem2.prob_dist[i] * prob_dist.ps

        if i == 0:
            new_prob_dist = ps
        else:
            new_prob_dist = np.hstack([new_prob_dist, ps])
    shape = tuple(list(elem2.prob_dist.shape) + elem1.nums_local_outcomes)
    new_md = MultinomialDistribution(ps=new_prob_dist, shape=shape)
    return new_md
