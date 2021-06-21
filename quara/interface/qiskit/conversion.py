import numpy as np
from typing import List, Tuple, Union

from qiskit.ignis.verification.tomography.basis import TomographyBasis, default_basis
from qiskit.quantum_info.operators import Operator
from qiskit import QuantumRegister, QuantumCircuit, ClassicalRegister

from quara.objects.composite_system_typical import generate_composite_system
from quara.objects.povm_typical import generate_povm_from_name
from quara.objects.qoperation_typical import generate_qoperation
from quara.protocol.qtomography.standard.standard_qst import generate_empi_dists
from quara.objects.gate_typical import generate_gate_hadamard
from quara.objects.state import State
from quara.objects.povm import Povm
from quara.objects.composite_system import CompositeSystem
from quara.objects.matrix_basis import (
    calc_hermitian_matrix_expansion_coefficient_hermitian_basis,
)
from quara.utils.matrix_util import calc_mat_from_vector_adjoint


def convert_state_qiskit_to_quara(
    qiskit_state: np.ndarray,
    c_sys: CompositeSystem,
) -> State:  ##physicaly_correctどうしよう
    qiskit_state_vec = calc_hermitian_matrix_expansion_coefficient_hermitian_basis(
        qiskit_state, c_sys.basis()
    )
    quara_state = State(
        c_sys=c_sys, vec=qiskit_state_vec, is_physicality_required=False
    )
    return quara_state


def convert_state_quara_to_qiskit(
    quara_state: State,
) -> np.ndarray:
    quara_state_vec = quara_state.vec
    qiskit_state = calc_mat_from_vector_adjoint(quara_state_vec)
    return qiskit_state


def convert_povm_qiskit_to_quara(
    qiskit_povm: List[np.ndarray], c_sys: CompositeSystem
) -> Povm:
    qiskit_povm_vec = []
    for mat in qiskit_povm:
        vec = calc_hermitian_matrix_expansion_coefficient_hermitian_basis(
            mat, c_sys.basis()
        )
        qiskit_povm_vec.append(vec)
    quara_povm = Povm(c_sys=c_sys, vecs=qiskit_povm_vec, is_physicality_required=False)
    return quara_povm


def convert_povm_quara_to_qiskit(quara_povm: Povm) -> List[np.ndarray]:
    qiskit_povm = []
    quara_povm_vecs = quara_povm.vecs
    for vec in quara_povm_vecs:
        mat = calc_mat_from_vector_adjoint(vec)
        qiskit_povm.append(mat)
    return qiskit_povm


def convert_empi_dists_qiskit_to_quara(
    qiskit_dists: np.ndarray, shots: Union[List, int], label: List[int]
) -> List[Tuple[int, np.ndarray]]:
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
