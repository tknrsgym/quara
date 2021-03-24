import numpy as np
from typing import List, Union

from quara.objects.composite_system import CompositeSystem
from quara.objects.qoperation import QOperation
from quara.objects.state import State
from quara.objects.povm import Povm
from quara.objects.gate import Gate
import quara.objects.state_typical
import quara.objects.povm_typical
from quara.objects.gate_typical import (
    generate_gate_object_from_gate_name_object_name,
)
from quara.objects.effective_lindbladian_typical import (
    generate_effective_lindbladian_object_from_gate_name_object_name,
)


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
    dims: List[int] = [],
    ids: List[int] = [],
    c_sys: CompositeSystem = None,
) -> Union[np.array, "Gate"]:
    """Return a gate-related object.

    Parameters
    ----------
    gate_name: str
        The list of valid gate_name is given by quara.objects.gate_typical.get_gate_names().

    object_name: str
        The list of valid object_name is given by get_gate_object_names().

    dims: List[int] = [], Optional
        To be given for gate_name = 'identity'

    ids: List[int] = [], Optional
        This is a list of elmental system's ids.
        To be given for specific asymmetric multi-partite gates
        For example, in the case of gate_name = 'cx', id[0] is for the id of the control qubit and id[1] is for the id of the target qubit.

    c_sys: CompositeSystem = None, Optional
        To be given for object_name = 'gate'

    Returns
    ----------
    Union[np.array, "Gate"]
        np.array
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
    dims: List[int] = [],
    ids: List[int] = [],
    c_sys: CompositeSystem = None,
) -> Union[np.array, "EffectiveLindbladian"]:
    """Return an effective-llindbladian-related object.

    Parameters
    ----------
    gate_name: str
        The list of valid gate_name is given by quara.objects.gate_typical.get_gate_names().

    object_name: str
        The list of valid object_name is given by get_effective_lindbladian_object_names().

    dims: List[int] = [], Optional
        To be given for gate_name = 'identity'

    ids: List[int] = [], Optional
        This is a list of elmental system's ids.
        To be given for specific asymmetric multi-partite gates
        For example, in the case of gate_name = 'cx', id[0] is for the id of the control qubit and id[1] is for the id of the target qubit.

    c_sys: CompositeSystem = None, Optional
        To be given for object_name = 'effective_lindbladian'

    Returns
    ----------
    Union[np.array, "EffectiveLindbladian"]
        np.array
            Hamiltonian vector for object_name = 'hamiltonian_vec'
                Real vector (The representaiton matrix basis is chosen to be Hermitian)
            Hamiltonian matrix for object_name = 'hamiltonian_mat'
                Complex matrix
            HS matrix for object_name = 'effective_lindbladian_mat'
                Reak matrix
        "EffectiveLindbladian"
            EffectiveLindbladian class for object_name = 'effective_lindbladian'
    """
    res = generate_effective_lindbladian_object_from_gate_name_object_name(
        gate_name, object_name, dims, ids, c_sys
    )
    return res
