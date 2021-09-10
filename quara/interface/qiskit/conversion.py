import numpy as np
from typing import List, Tuple, Union

from quara.objects.state import State
from quara.objects.povm import Povm
from quara.objects.gate import Gate, to_hs_from_choi
from quara.objects.composite_system import CompositeSystem
from quara.objects.gate_typical import (
    calc_gate_mat_from_unitary_mat_with_hermitian_basis,
)
from quara.objects.matrix_basis import (
    calc_hermitian_matrix_expansion_coefficient_hermitian_basis,
)


def convert_state_qiskit_to_quara(
    qiskit_state: np.ndarray,
    c_sys: CompositeSystem,
) -> State:

    """converts densitymatrix in Qiskit to Quara State.

    Parameters
    ----------
    qiskit_state: np.ndarray
        this represents density matrix of quantum state.

    c_sys: CompositeSystem
        CompositeSystem contains state.

    Returns
    -------
    State
        Quara State.
    """

    qiskit_state_vec = calc_hermitian_matrix_expansion_coefficient_hermitian_basis(
        qiskit_state, c_sys.basis()
    )
    quara_state = State(c_sys=c_sys, vec=qiskit_state_vec)
    return quara_state


def convert_state_quara_to_qiskit(
    quara_state: State,
) -> np.ndarray:

    """converts Quara State to densitymatrix in Qiskit.

    Parameters
    ----------
    quara_state: State
        Quara State.

    Returns
    -------
    np.ndarray
       Qiskit density matrix of quantum state.
    """

    qiskit_state = quara_state.to_density_matrix()
    return qiskit_state


def convert_povm_qiskit_to_quara(
    qiskit_povm: List[np.ndarray],
    c_sys: CompositeSystem,
) -> Povm:

    """converts Qiskit representation matrix to Quara Povm.

    Parameters
    ----------
    qiskit_povm: np.ndarray
        this represents representation matrix of quantum state.

    c_sys: CompositeSystem
        CompositeSystem contains state.

    Returns
    -------
    Povm
        Quara Povm.
    """

    qiskit_povm_vec = []
    for mat in qiskit_povm:
        vec = calc_hermitian_matrix_expansion_coefficient_hermitian_basis(
            mat, c_sys.basis()
        )
        qiskit_povm_vec.append(vec)
    quara_povm = Povm(c_sys=c_sys, vecs=qiskit_povm_vec)
    return quara_povm


def convert_povm_quara_to_qiskit(
    quara_povm: Povm,
) -> List[np.ndarray]:

    """converts Quara Povm to Qiskit representation matrix .

    Parameters
    ----------
    quara_povm:Povm
        Quara Povm.

    Returns
    -------
    List[np.ndarray]
       list of Qiskit representation matrix of quantum povm.
    """

    qiskit_povm = quara_povm.matrices()
    return qiskit_povm


def convert_empi_dists_qiskit_to_quara(
    qiskit_dists: np.ndarray,
    shots: Union[List[int], int],
    label: List[int],
) -> List[Tuple[int, np.ndarray]]:

    """converts Qiskit empirical distribution to Quara empirical distribution.

    Parameters
    ----------
    qiskit_dists: np.ndarray
        this represents empirical distribution.

    shots: Union[List[int], int]
        shots represents the number of times.

    label: List[int]
        label provides the number of unit for one measurement.

    Returns
    -------
    List[Tuple[int, np.ndarray]]
        Quara empirical distribution.
    """

    quara_dists = []
    cts = 0
    if type(shots) == int:
        for i in label:
            tup = (shots, qiskit_dists[cts : cts + i])
            quara_dists.append(tup)
            cts = cts + i
    else:
        for i in label:
            tup = (shots[cts], qiskit_dists[cts * i : cts * i + i])
            quara_dists.append(tup)
            cts = cts + 1
    return quara_dists


def convert_empi_dists_quara_to_qiskit(
    quara_dists: List[Tuple[int, np.ndarray]],
) -> np.ndarray:

    """converts Quara empirical distribution to Qiskit empirical distribution.

    Parameters
    ----------
    quara_dists: List[Tuple[int, np.ndarray]]
        Quara empirical distribution.

    Returns
    -------
    np.ndarray
         Qiskit empirical distribution
    """

    qiskit_dists = []
    for i in quara_dists:
        qiskit_dists = np.hstack((qiskit_dists, i[1]))
    return qiskit_dists


def convert_empi_dists_quara_to_qiskit_shots(
    quara_dists: List[Tuple[int, np.ndarray]],
) -> List[int]:

    """returns the list of shots from Quara empirical distribution.

    Parameters
    ----------
    quara_dists: List[Tuple[int, np.ndarray]]
        Quara empirical distribution.

    Returns
    -------
    List[int]
        each number of shots.
    """

    qiskit_shots = []
    for i in quara_dists:
        qiskit_shots.append(i[0])
    return qiskit_shots


def convert_gate_qiskit_to_quara(
    qiskit_gate: np.ndarray,
    c_sys: CompositeSystem,
    dim: int,
) -> Gate:

    """converts qiskit gate choi matrix to Quara Gate.

    Parameters
    ----------
    qiskit_gate: np.ndarray
         Qiskit choi matrix.

    c_sys: Compositesystem
         CompositeSystem contains state.

    dim: int
         Dimension of system.

    Returns
    -------
    Gate
         Quara gate.
    """

    swap = calc_swap_matrix(dim)
    qiskit_gate_for_quara = np.dot(swap, (np.dot(qiskit_gate, swap)))
    qiskit_gate_hs = to_hs_from_choi(qiskit_gate_for_quara, c_sys)
    quara_gate = Gate(c_sys, qiskit_gate_hs)
    return quara_gate


def convert_gate_quara_to_qiskit(
    quara_gate: Gate,
    dim: int,
) -> np.ndarray:

    """converts Quara Gate to qiskit gate choi matrix.

    Parameters
    ----------
    quara_gate: Gate
         Quara gate.

    dim: int
         Dimension of system.

    Returns
    -------
    np.ndarray
        Qiskit choi matrix.

    """

    swap = calc_swap_matrix(dim)
    qiskit_gate_for_quara = quara_gate.to_choi_matrix()
    qiskit_gate = np.dot(swap, (np.dot(qiskit_gate_for_quara, swap)))
    return qiskit_gate


def calc_swap_matrix(d: int) -> np.ndarray:

    mat = np.zeros((d ** 2, d ** 2), dtype=np.complex64)
    for k in range(d ** 2):
        i1 = k // d
        j1 = k % d
        for l in range(d ** 2):
            i2 = l // d
            j2 = l % d
            if i1 == j2 and j1 == i2:
                mat[k][l] = 1
    return mat
