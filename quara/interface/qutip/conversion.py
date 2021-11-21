from typing import List

import numpy as np
from qutip import Qobj

from quara.objects.composite_system import CompositeSystem
from quara.objects.state import State
from quara.objects.povm import Povm
from quara.objects.gate import Gate, convert_hs
from quara.objects.matrix_basis import (
    get_comp_basis,
    calc_hermitian_matrix_expansion_coefficient_hermitian_basis,
)
from quara.utils.matrix_util import calc_mat_from_vector_adjoint, truncate_hs


def convert_state_qutip_to_quara(qutip_qobj: Qobj, c_sys: CompositeSystem) -> State:
    """converts QuTiP Qobj to Quara State.

    Parameters
    ----------
    qutip_qobj: Qobj
        Qobj representing quantum state.
    c_sys : CompositeSystem
        CompositeSystem containing state.

    Returns
    -------
    State
        Quara State.

    Raises
    ------
    ValueError
        Invalid Qobj type.
    ValueError
        Invalid argument for State constructor.
    """
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
    """converts Quara State to QuTiP Qobj.

    Parameters
    ----------
    quara_state: State
        Quara State.

    Returns
    -------
    Qobj
        Qobj representing quantum state.

    """
    density_mat = quara_state.to_density_matrix()
    return Qobj(density_mat)


def convert_povm_qutip_to_quara(
    qutip_qobjs: List[Qobj], c_sys: CompositeSystem
) -> Povm:
    """converts QuTiP Qobj to Quara Povm.

    Parameters
    ----------
    qutip_qobjs: List[Qobj]
        List of Qobj representing POVM. The type of Qobj has to be 'oper'.
    c_sys : CompositeSystem
        CompositeSystem containing POVM.

    Returns
    -------
    Povm
        Quara Povm.

    Raises
    ------
    ValueError
        Invalid argument for Povm constructor.
    """
    vecs = []
    for item in qutip_qobjs:
        matrix = item.data.toarray()
        vecs.append(
            calc_hermitian_matrix_expansion_coefficient_hermitian_basis(
                matrix, c_sys.basis()
            )
        )
    return Povm(c_sys, vecs)


def convert_povm_quara_to_qutip(quara_povm: Povm) -> List[Qobj]:
    """converts Quara Povm to QuTiP Qobj.

    Parameters
    ----------
    quara_povm: Povm
        Quara Povm.

    Returns
    -------
    List[Qobj]
        List of Qobj representing POVM.
    """
    matrices = quara_povm.matrices()
    qutip_povm = [Qobj(matrix) for matrix in matrices]
    return qutip_povm


def convert_gate_qutip_to_quara(qutip_qobj: Qobj, c_sys: CompositeSystem) -> Gate:
    """converts QuTiP Qobj to Quara Gate.

    Parameters
    ----------
    qutip_qobj: Qobj
        Qobj representing quantum gate in super representation.
    c_sys : CompositeSystem
        CompositeSystem containing gate.

    Returns
    -------
    Gate
        Quara Gate.

    Raises
    ------
    ValueError
        Invalid argument for Gate constructor.
    """
    hs_matrix_column_major = qutip_qobj.data.toarray()
    dim = int(np.sqrt(qutip_qobj.shape[0]))
    comp_basis_column_major = get_comp_basis(dim, "column_major")
    hs_matrix_converted = convert_hs(
        hs_matrix_column_major, comp_basis_column_major, c_sys.basis()
    )
    hs_matrix_truncated = truncate_hs(hs_matrix_converted)
    return Gate(c_sys=c_sys, hs=hs_matrix_truncated)


def convert_gate_quara_to_qutip(quara_gate: Gate) -> Qobj:
    """converts Quara Gate to QuTiP Qobj.

    Parameters
    ----------
    quara_gate: Gate
        Quara Gate.

    Returns
    -------
    Qobj
        Qobj representing quantum gate in super representation.
    """
    comp_basis = get_comp_basis(quara_gate.dim, "column_major")
    hs_matrix = quara_gate.convert_basis(comp_basis)
    return Qobj(hs_matrix, type="super")