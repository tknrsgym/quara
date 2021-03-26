import numpy as np
import numpy.testing as npt
import pytest

from quara.objects.composite_system import CompositeSystem
from quara.objects.elemental_system import ElementalSystem
from quara.objects.matrix_basis import get_normalized_pauli_basis
from quara.objects.operators import tensor_product
from quara.objects.povm import Povm
from quara.objects import povm_typical


@pytest.mark.onequbit
@pytest.mark.parametrize(
    ("povm_name", "expected_vecs"),
    [
        (
            "x",
            [
                np.array([1, 1, 0, 0], dtype=np.float64) / np.sqrt(2),
                np.array([1, -1, 0, 0], dtype=np.float64) / np.sqrt(2),
            ],
        ),
        (
            "y",
            [
                np.array([1, 0, 1, 0], dtype=np.float64) / np.sqrt(2),
                np.array([1, 0, -1, 0], dtype=np.float64) / np.sqrt(2),
            ],
        ),
        (
            "z",
            [
                np.array([1, 0, 0, 1], dtype=np.float64) / np.sqrt(2),
                np.array([1, 0, 0, -1], dtype=np.float64) / np.sqrt(2),
            ],
        ),
    ],
)
def test_generate_povm_from_name_1qubit(povm_name, expected_vecs):
    e_sys = ElementalSystem(0, get_normalized_pauli_basis())
    c_sys = CompositeSystem([e_sys])

    actual = povm_typical.generate_povm_from_name(povm_name, c_sys=c_sys)
    for actual_vec, expected_vec in zip(actual.vecs, expected_vecs):
        npt.assert_almost_equal(actual_vec, expected_vec, decimal=15)

    """
    actual = state_typical.generate_state_pure_state_vector_from_name("bell_phi_plus")
    print(f"actual={actual}")
    actual = state_typical.generate_state_pure_state_vector_from_name("bell_phi_minus")
    print(f"actual={actual}")
    actual = state_typical.generate_state_pure_state_vector_from_name("bell_psi_plus")
    print(f"actual={actual}")
    actual = state_typical.generate_state_pure_state_vector_from_name("bell_psi_minus")
    print(f"actual={actual}")
    """
