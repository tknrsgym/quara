import numpy as np
from typing import List, Union

from quara.objects.composite_system import CompositeSystem
from quara.objects.qoperation import QOperation
from quara.objects.operators import compose_qoperations
from quara.objects.state import State
from quara.objects.povm import Povm
from quara.objects.gate import (
    Gate,
    get_depolarizing_channel,
)
from quara.objects.state_typical import (
    generate_state_from_name,
    generate_state_object_from_state_name_object_name,
)
from quara.objects.povm_typical import generate_povm_object_from_povm_name_object_name
from quara.objects.gate_typical import (
    generate_gate_object_from_gate_name_object_name,
    generate_gate_piover8_mat,
)
from quara.objects.effective_lindbladian_typical import (
    generate_effective_lindbladian_object_from_gate_name_object_name,
)


def generate_qoperation(
    mode: str, name: str, c_sys: CompositeSystem, ids: List[int] = None
) -> QOperation:
    return generate_qoperation_object(
        mode=mode, name=name, object_name=mode, ids=ids, c_sys=c_sys
    )


def generate_qoperation_depolarized(
    mode: str,
    name: str,
    c_sys: CompositeSystem,
    error_rate: np.float64,
    ids: List[int] = None,
) -> QOperation:
    dp = get_depolarizing_channel(p=error_rate, c_sys=c_sys)
    qoperation = generate_qoperation(mode=mode, name=name, c_sys=c_sys, ids=ids)

    if mode == "state":
        qoperation_depolarized = compose_qoperations(dp, qoperation)
    elif mode == "povm":
        qoperation_depolarized = compose_qoperations(qoperation, dp)
    elif mode == "gate":
        qoperation_depolarized = compose_qoperations(dp, qoperation)
    else:
        raise ValueError(f"mode is invalid.")

    return qoperation_depolarized


def generate_state_object(
    state_name: str, object_name: str, c_sys: CompositeSystem = None
):
    return generate_state_object_from_state_name_object_name(
        state_name, object_name, c_sys
    )


def generate_povm_object(
    povm_name: str, object_name: str, c_sys: CompositeSystem = None
):
    return generate_povm_object_from_povm_name_object_name(
        povm_name, object_name, c_sys
    )


def generate_qoperation_object(
    mode: str,
    name: str,
    object_name: str,
    dims: List[int] = None,
    ids: List[int] = None,
    c_sys: CompositeSystem = None,
):
    if mode == "state":
        return generate_state_object(
            state_name=name, object_name=object_name, c_sys=c_sys
        )
    elif mode == "povm":
        return generate_povm_object(
            povm_name=name, object_name=object_name, c_sys=c_sys
        )
    elif mode == "gate":
        return generate_gate_object(
            gate_name=name, object_name=object_name, c_sys=c_sys, dims=dims, ids=ids
        )
    else:
        error_message = "mode is out of range."
        raise ValueError(error_message)


def get_gate_object_names() -> List[str]:
    """Return the list of valid gate-related object names."""
    names = []
    names.append("unitary_mat")
    names.append("gate_mat")
    names.append("gate")

    return names


def generate_gate_object(
    gate_name: str,
    object_name: str,
    dims: List[int] = None,
    ids: List[int] = None,
    c_sys: CompositeSystem = None,
) -> Union[np.ndarray, "Gate"]:
    """Return a gate-related object.

    Parameters
    ----------
    gate_name: str
        The list of valid gate_name is given by quara.objects.gate_typical.get_gate_names().

    object_name: str
        The list of valid object_name is given by get_gate_object_names().

    dims: List[int] = None, Optional
        To be given for gate_name = 'identity'

    ids: List[int] = None, Optional
        This is a list of elmental system's ids.
        To be given for specific asymmetric multi-partite gates
        For example, in the case of gate_name = 'cx', id[0] is for the id of the control qubit and id[1] is for the id of the target qubit.

    c_sys: CompositeSystem = None, Optional
        To be given for object_name = 'gate'

    Returns
    ----------
    Union[np.ndarray, "Gate"]
        np.ndarray
            Unitary matrix for object_name = 'unitary_mat'
                Complex matrix
            HS matrix for object_name = 'gate_mat'
                Real matrix
        "Gate"
            Gate class for object_name = 'gate'
    """
    res = generate_gate_object_from_gate_name_object_name(
        gate_name, object_name, dims, ids, c_sys
    )
    return res


def get_effective_lindbladian_object_names() -> List[str]:
    """Return the list of valid effective-lindbladian-related object names."""
    names = []
    names.append("hamiltonian_vec")
    names.append("hamiltonian_mat")
    names.append("effective_lindbladian_mat")
    names.append("effective_lindbladian")

    return names


def generate_effective_lindbladian_object(
    gate_name: str,
    object_name: str,
    dims: List[int] = None,
    ids: List[int] = None,
    c_sys: CompositeSystem = None,
) -> Union[np.ndarray, "EffectiveLindbladian"]:
    """Return an effective-llindbladian-related object.

    Parameters
    ----------
    gate_name: str
        The list of valid gate_name is given by quara.objects.gate_typical.get_gate_names().

    object_name: str
        The list of valid object_name is given by get_effective_lindbladian_object_names().

    dims: List[int] = None, Optional
        To be given for gate_name = 'identity'

    ids: List[int] = None, Optional
        This is a list of elmental system's ids.
        To be given for specific asymmetric multi-partite gates
        For example, in the case of gate_name = 'cx', id[0] is for the id of the control qubit and id[1] is for the id of the target qubit.

    c_sys: CompositeSystem = None, Optional
        To be given for object_name = 'effective_lindbladian'

    Returns
    ----------
    Union[np.ndarray, "EffectiveLindbladian"]
        np.ndarray
            Hamiltonian vector for object_name = 'hamiltonian_vec'
                Real vector (The representaiton matrix basis is chosen to be Hermitian)
            Hamiltonian matrix for object_name = 'hamiltonian_mat'
                Complex matrix
            HS matrix for object_name = 'effective_lindbladian_mat'
                Reak matrix
        "EffectiveLindbladian"
            EffectiveLindbladian class for object_name = 'effective_lindbladian'
    """
    dims = [] if dims is None else dims
    ids = [] if dims is None else ids
    res = generate_effective_lindbladian_object_from_gate_name_object_name(
        gate_name, object_name, dims, ids, c_sys
    )
    return res
