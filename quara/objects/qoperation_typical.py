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
    generate_unitary_mat_from_gate_name,
    generate_gate_mat_from_gate_name,
    generate_gate_from_gate_name,
)
from quara.objects.effective_lindbladian_typical import (
    generate_hamiltonian_vec_from_gate_name,
    generate_hamiltonian_mat_from_gate_name,
    generate_effective_lindbladian_mat_from_gate_name,
    generate_effective_lindbladian_from_gate_name,
)


def get_gate_object_names() -> List[str]:
    """Return the list of valid object names for generate_object_of_gate_from_gate_name_object_name()."""
    names = []
    names.append("hamiltonian_vec")
    names.append("hamiltonian_mat")
    names.append("unitary_mat")
    names.append("effective_lindbladian_mat")
    names.append("effective_lindbladian_class")
    names.append("gate_mat")
    names.append("gate_class")

    return names


def generate_gate_object_from_gate_name_object_name(
    gate_name: str,
    object_name: str,
    dims: List[int] = [],
    ids: List[int] = [],
    c_sys: CompositeSystem = None,
) -> Union[np.array, "EffectiveLindbladian", "Gate"]:
    if object_name == "hamiltonian_vec":
        obj = generate_hamiltonian_vec_from_gate_name(gate_name, dims, ids)
    elif object_name == "hamiltonian_mat":
        obj = generate_hamiltonian_mat_from_gate_name(gate_name, dims, ids)
    elif object_name == "unitary_mat":
        obj = generate_unitary_mat_from_gate_name(gate_name, dims, ids)
    elif object_name == "effective_lindbladian_mat":
        obj = generate_effective_lindbladian_mat_from_gate_name(gate_name, dims, ids)
    elif object_name == "gate_mat":
        obj = generate_gate_mat_from_gate_name(gate_name, dims, ids)
    elif object_name == "effective_lindbladian_class":
        obj = generate_effective_lindbladian_from_gate_name(gate_name, c_sys, ids)
    elif object_name == "gate_class":
        obj = generate_gate_from_gate_name(gate_name, c_sys, ids)
    else:
        raise ValueError(f"object_name is out of range.")
    return obj
