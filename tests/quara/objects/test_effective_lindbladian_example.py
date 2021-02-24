import numpy as np
import numpy.testing as npt
import pytest

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


def _test_generate_effective_lindbladian_from_h(
    c_sys: CompositeSystem, gate_name: str
) -> bool:
    pass