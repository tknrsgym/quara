import numpy as np
from typing import List, Tuple, Union

from qiskit.ignis.verification.tomography.basis import TomographyBasis, default_basis
from qiskit.quantum_info.operators import Operator
from qiskit import QuantumRegister, QuantumCircuit, ClassicalRegister

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
    qiskit_state_vec = calc_hermitian_matrix_expansion_coefficient_hermitian_basis(
        qiskit_state, c_sys.basis()
    )
    quara_state = State(c_sys=c_sys, vec=qiskit_state_vec)
    return quara_state


def convert_state_quara_to_qiskit(
    quara_state: State,
) -> np.ndarray:
    qiskit_state = quara_state.to_density_matrix()
    return qiskit_state


def convert_povm_qiskit_to_quara(
    qiskit_povm: List[np.ndarray],
    c_sys: CompositeSystem,
) -> Povm:
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
    qiskit_povm = quara_povm.matrices()
    return qiskit_povm


def convert_empi_dists_qiskit_to_quara(
    qiskit_dists: np.ndarray,
    shots: Union[List, int],
    label: List[int],
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


def convert_empi_dists_quara_to_qiskit(
    quara_dists: List[Tuple[int, np.ndarray]],
) -> np.ndarray:
    qiskit_dists = []
    for i in quara_dists:
        qiskit_dists = qiskit_dists + i[1]
    return qiskit_dists


def convert_gate_qiskit_to_quara(
    qiskit_gate: Operator,
    c_sys: CompositeSystem,
) -> Gate:
    qiskit_gate_hs = calc_gate_mat_from_unitary_mat_with_hermitian_basis(
        qiskit_gate.data, c_sys.basis()
    )
    quara_gate = Gate(c_sys, qiskit_gate_hs)
    return quara_gate
