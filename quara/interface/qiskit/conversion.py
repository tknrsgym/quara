import numpy as np

from qiskit.ignis.verification.tomography.basis import TomographyBasis, default_basis
from qiskit.quantum_info.operators import Operator
from qiskit import QuantumRegister, QuantumCircuit, ClassicalRegister

from quara.objects.composite_system_typical import generate_composite_system
from quara.objects.povm_typical import generate_povm_from_name
from quara.objects.qoperation_typical import generate_qoperation
from quara.protocol.qtomography.standard.standard_qst import generate_empi_dists
from quara.objects.gate_typical import generate_gate_hadamard

def convert_state_qiskit_to_quara(
    qiskit_state: np.ndarray,
    c_sys: CompositSystem,
) -> State:                        ##physicaly_correctどうしよう

    quara_state = state.State(c_sys = c_sys, vec = qiskit_state, is_physicality_required = False)
    return quara_state

def convert_state_quara_to_qiskit(
    quara_state: State,
) -> np.ndarray:

    qiskit_state = quara_state.vec
    return qiskit_state

def convert_povm_qiskit_to_quara(
    qiskit_povm: List[np.ndarray],
    c_sys: CompositSystem
    ) -> Povm:
    quara_povm = povm.Povm(c_sys = c_sys, vecs = qiskit_povm, is_physicality_required = False)
    return quara_povm
