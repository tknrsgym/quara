from typing import List, Tuple

from quara.objects.composite_system import CompositeSystem
from quara.objects.matrix_basis import (
    calc_hermitian_matrix_expansion_coefficient_hermitian_basis,
)
from quara.objects.state import State
from quara.objects.gate import Gate
from quara.objects.mprocess import MProcess

from qulacs import DensityMatrix
from qulacs.gate import DenseMatrix, CPTP, Instrument

import numpy as np


def convert_state_quara_to_qulacs(quara_state: State) -> DensityMatrix:
    """converts Quara State into Qulacs DensityMatrix.

    Parameters
    ----------
    qulacs_density_mat: DensityMatrix
        DensityMatrix object in Qulacs.

    c_sys: CompositeSystem
        CompositeSystem containing state.

    Returns
    -------
    State
        Quara State.
    """
    density_mat = quara_state.to_density_matrix()
    qulacs_density_mat = DensityMatrix(int(np.log2(quara_state.dim)))
    qulacs_density_mat.load(density_mat)
    return qulacs_density_mat


def convert_state_qulacs_to_quara(
    qulacs_density_mat: DensityMatrix, c_sys: CompositeSystem
) -> State:
    """converts Qulacs DensityMatrix into Quara State.

    Parameters
    ----------
    qulacs_density_mat: DensityMatrix
        DensityMatrix object in Qulacs.

    c_sys: CompositeSystem
        CompositeSystem containing state.

    Returns
    -------
    State
        Quara State.
    """
    density_mat = qulacs_density_mat.get_matrix()
    vec = calc_hermitian_matrix_expansion_coefficient_hermitian_basis(
        density_mat, c_sys.basis()
    )
    return State(vec=vec, c_sys=c_sys)


def convert_gate_quara_to_qulacs(quara_gate: Gate, qubits: List[int]) -> CPTP:
    """converts Quara Gate object into Qulacs CPTP object.

    Parameters
    ----------
    quara_gate: Gate

    qubits: List[int]

    Returns
    -------
    CPTP
        Qulacs CPTP object which Kraus-represented matrices are stored.
    """
    kraus_matrices = quara_gate.to_kraus_matrices()
    assert 2**len(qubits) == quara_gate.dim or 2**(2*len(qubits)) == quara_gate.dim
    qulacs_gate_list = []
    for kraus_matrix in kraus_matrices:
        qulacs_gate = DenseMatrix(qubits, kraus_matrix)
        qulacs_gate_list.append(qulacs_gate)
    return CPTP(qulacs_gate_list)


def convert_instrument_quara_to_qulacs(
    quara_mprocess: MProcess, qubits: List[int]
) -> Tuple[Instrument, List[int]]:
    """converts Quara State into Qulacs Instrument in form of Lists of Kraus operators

    Parameters
    ----------
    quara_mprocess: MProcess
        Quara MProcess object.

    qubits: List[int]
        List of indices of qubits which intruments are assosiated to.

    Returns
    -------
    Tuple[Instrument, List[int]]
        Qulacs Instrument object and list of number of matrices for every Kraus-represented operators.
    """
    num_indices = len(quara_mprocess.hss)
    kraus_matrices_indices = []
    assert 2**len(qubits) == quara_mprocess.dim or 2**(2*len(qubits)) == quara_mprocess.dim
    qulacs_gate_list = []
    for index in range(num_indices):
        kraus_matrices = quara_mprocess.to_kraus_matrices(index)
        kraus_matrices_indices.append(len(kraus_matrices))
        for kraus_matrix in kraus_matrices:
            qulacs_gate = DenseMatrix(qubits, kraus_matrix)
            qulacs_gate_list.append(qulacs_gate)
    # posistion of the applied matrix will be stored at position=0
    return Instrument(qulacs_gate_list, 0), kraus_matrices_indices


def get_index_tuple_from_kraus_matrices_indices(
    index: int, indices: List[int]
) -> Tuple[int, int]:
    """Get index i, j of Instrument object from a list of numbers corresponding to matrices of Kraus operators.

    Parameters
    ----------
    index: int
        Index of the matrix in a list of matrices of Kraus operators.

    indices: List[int]
        List of numbers of matrices per Kraus operator.

    Returns
    -------
    Tuple[int, int]
        returns (i, j) meaning that j-th matrix of i-th Kraus operator.
    """
    sum = 0
    for j, num in enumerate(indices):
        sum += num
        if index < sum:
            return (j, num + index - sum)
    assert index < sum
    return (len(indices) - 1, sum - index)
