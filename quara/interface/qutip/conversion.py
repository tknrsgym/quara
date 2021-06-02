from qutip import Qobj

from quara.objects.composite_system import CompositeSystem
from quara.objects.state import State
from quara.objects.povm import Povm
from quara.objects.gate import Gate
from quara.objects.matrix_basis import (
    calc_hermitian_matrix_expansion_coefficient_hermitian_basis,
)
from quara.utils.matrix_util import calc_mat_from_vector_adjoint


def convert_state_qutip_to_quara(qutip_qobj: Qobj, c_sys: CompositeSystem) -> State:
    data_array = qutip_qobj.data.toarray()
    if qutip_qobj.isket:
        density_mat = calc_mat_from_vector_adjoint(data_array.flatten())
        print(density_mat)
        vec = calc_hermitian_matrix_expansion_coefficient_hermitian_basis(
            density_mat, c_sys.basis()
        )
        quara_state = State(vec=vec, c_sys=c_sys)
    elif qutip_qobj.isoper:
        vec = calc_hermitian_matrix_expansion_coefficient_hermitian_basis(
            data_array, c_sys.basis()
        )
        quara_state = State(vec=vec, c_sys=c_sys)
    else:
        raise ValueError("invalid Qobj type")
    return quara_state


def convert_state_quara_to_qutip(quara_state: State) -> Qobj:
    density_mat = quara_state.to_density_matrix()
    return Qobj(density_mat)


# TODO: implement
def convert_povm_qutip_to_quara(qutip_qobj: Qobj, c_sys: CompositeSystem) -> Povm:
    pass


# TODO: implement
def convert_povm_quara_to_qutip(quara_povm: Povm) -> Qobj:
    pass


# TODO: implement
def convert_gate_qutip_to_quara(qutip_qobj: Qobj, c_sys: CompositeSystem) -> Gate:
    pass


# TODO: implement
def convert_gate_quara_to_qutip(quara_gate: Gate) -> Qobj:
    pass
