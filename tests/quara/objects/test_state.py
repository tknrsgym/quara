import numpy as np

from quara.objects import elemental_system
from quara.objects.composite_system import CompositeSystem
from quara.objects.state import State


def test_with_normalized_pauli_basis():
    e_sys = elemental_system.get_with_normalized_pauli_basis("q1")
    c_sys = CompositeSystem([e_sys])

    state = State(c_sys, np.array([1, 0, 0, 0], dtype=np.float64))
    mat = state.get_density_matrix()
    print(mat)
    assert state.dim == 2
    assert np.all(
        state.get_density_matrix()
        == 1 / np.sqrt(2) * np.array([[1, 0], [0, 1]], dtype=np.float64)
    )
    assert state.is_trace_one() == False

    # TODO 他の関数も一通りテストすること
