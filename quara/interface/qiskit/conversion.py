import numpy as np
from typing import List, Tuple, Union

from quara.objects.state import State
from quara.objects.povm import Povm
from quara.objects.gate import Gate
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
            tup = (shots[cts], qiskit_dists[cts : cts + i])
            quara_dists.append(tup)
            cts = cts + i
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
        qiskit_dists = qiskit_dists + i[1]
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
) -> Gate:

    """converts qiskit gate unitary matrix to Quara Gate.

    Parameters
    ----------
    qiskit_gate: np.ndarray
         Qiskit unitary matrix.

    c_sys: Compositesystem
         CompositeSystem contains state.

    Returns
    -------
    Gate
         Quara gate.
    """

    qiskit_gate_hs = calc_gate_mat_from_unitary_mat_with_hermitian_basis(
        qiskit_gate, c_sys.basis()
    )
    quara_gate = Gate(c_sys, qiskit_gate_hs)
    return quara_gate


def convert_gate_quara_to_qiskit(
    quara_gate: Gate,
) -> np.ndarray:

    """converts Quara Gate to qiskit gate unitary matrix.

    Parameters
    ----------
    quara_gate: Gate
         Quara gate.

    Returns
    -------
    np.ndarray
        Qiskit unitary matrix.

    """

    qiskit_gate_l = quara_gate.to_kraus_matrices()
    qiskit_gate = qiskit_gate_l[0]
    return qiskit_gate
