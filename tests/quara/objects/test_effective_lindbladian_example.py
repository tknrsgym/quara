import numpy as np
import numpy.testing as npt
import pytest

from typing import List
from scipy.linalg import expm

from quara.objects import matrix_basis
from quara.objects.composite_system import CompositeSystem
from quara.objects.elemental_system import ElementalSystem
from quara.objects.gate import Gate
from quara.objects.effective_lindbladian import EffectiveLindbladian

from quara.objects.effective_lindbladian_example import (
    generate_vec_identity_gate_hamiltonian,
    generate_matrix_identity_gate_hamiltonian,
    generate_matrix_identity_gate_unitary,
    generate_matrix_identity_gate_lindbladian,
    generate_matrix_identity_gate,
    generate_effective_lindbladian_identity_gate,
)


def _test_generate_gate_objects(
    c_sys: CompositeSystem, gate_name: str, ids: List[int] = []
):
    assert c_sys.is_orthonormal_hermitian_0thpropI == True
    b = c_sys.basis

    _test_validity_hamiltonian_vec_hamiltonian_mat(gate_name, b)
    _test_validity_hamiltonian_mat_unitary_mat(gate_name)
    _test_validity_effective_lindladian_mat_gate_mat(gate_name)

    _test_generate_effective_lindbladian_from_h(c_sys, gate_name, ids)
    _test_calc_h(c_sys, gate_name, ids)


def _test_generate_effective_lindbladian_from_h(
    c_sys: CompositeSystem, gate_name: str
) -> bool:
    pass