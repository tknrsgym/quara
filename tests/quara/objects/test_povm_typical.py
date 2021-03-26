import numpy as np
import numpy.testing as npt
import pytest

from quara.objects.composite_system import CompositeSystem
from quara.objects.elemental_system import ElementalSystem
from quara.objects.matrix_basis import (
    get_normalized_pauli_basis,
    get_normalized_gell_mann_basis,
)
from quara.objects.operators import tensor_product
from quara.objects.povm import Povm
from quara.objects import povm_typical


def test_generate_povm_object_from_povm_name_object_name():
    e_sys = ElementalSystem(0, get_normalized_pauli_basis())
    c_sys = CompositeSystem([e_sys])

    # generate_povm_pure_state_vectors_from_name
    actual = povm_typical.generate_povm_pure_state_vectors_from_name("z")
    expected = [
        np.array([1, 0]),
        np.array([0, 1]),
    ]
    for a, e in zip(actual, expected):
        npt.assert_almost_equal(a, e, decimal=15)

    # generate_povm_matrices_from_name
    actual = povm_typical.generate_povm_matrices_from_name("z")
    expected = [
        np.array([[1, 0], [0, 0]]),
        np.array([[0, 0], [0, 1]]),
    ]
    for a, e in zip(actual, expected):
        npt.assert_almost_equal(a, e, decimal=15)

    # generate_povm_vectors_from_name
    actual = povm_typical.generate_povm_vectors_from_name(
        "z", get_normalized_pauli_basis()
    )
    expected = [
        np.array([1, 0, 0, 1], dtype=np.float64) / np.sqrt(2),
        np.array([1, 0, 0, -1], dtype=np.float64) / np.sqrt(2),
    ]
    for a, e in zip(actual, expected):
        npt.assert_almost_equal(a, e, decimal=15)

    # generate_povm_from_name
    actual = povm_typical.generate_povm_from_name("z", c_sys)
    expected = [
        np.array([1, 0, 0, 1], dtype=np.float64) / np.sqrt(2),
        np.array([1, 0, 0, -1], dtype=np.float64) / np.sqrt(2),
    ]
    for a, e in zip(actual.vecs, expected):
        npt.assert_almost_equal(a, e, decimal=15)


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


@pytest.mark.twoqubits
@pytest.mark.parametrize(
    ("povm_name", "expected_vecs"),
    [
        (
            "z_z",
            [
                np.array(
                    [1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1], dtype=np.float64
                )
                / 2,
                np.array(
                    [1, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, -1], dtype=np.float64
                )
                / 2,
                np.array(
                    [1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, -1, 0, 0, -1], dtype=np.float64
                )
                / 2,
                np.array(
                    [1, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0, -1, 0, 0, 1], dtype=np.float64
                )
                / 2,
            ],
        ),
    ],
)
def test_generate_povm_from_name_2qubit(povm_name, expected_vecs):
    e_sys0 = ElementalSystem(0, get_normalized_pauli_basis())
    e_sys1 = ElementalSystem(1, get_normalized_pauli_basis())
    c_sys = CompositeSystem([e_sys0, e_sys1])

    actual = povm_typical.generate_povm_from_name(povm_name, c_sys=c_sys)
    for actual_vec, expected_vec in zip(actual.vecs, expected_vecs):
        npt.assert_almost_equal(actual_vec, expected_vec, decimal=15)
    # TODO bell


"""
    from quara.objects import state_typical

    actual = state_typical.generate_state_pure_state_vector_from_name("bell_phi_plus")
    print(f"actual={actual}")
    actual = state_typical.generate_state_pure_state_vector_from_name("bell_phi_minus")
    print(f"actual={actual}")
    actual = state_typical.generate_state_pure_state_vector_from_name("bell_psi_plus")
    print(f"actual={actual}")
    actual = state_typical.generate_state_pure_state_vector_from_name("bell_psi_minus")
    print(f"actual={actual}")
    assert False
"""


@pytest.mark.onequbit
@pytest.mark.parametrize(
    ("povm_name", "expected_vecs"),
    [
        (
            "z3",
            [
                np.array(
                    [np.sqrt(1 / 3), 0, 0, np.sqrt(1 / 2), 0, 0, 0, 0, np.sqrt(1 / 6)],
                    dtype=np.float64,
                ),
                np.array(
                    [np.sqrt(1 / 3), 0, 0, -np.sqrt(1 / 2), 0, 0, 0, 0, np.sqrt(1 / 6)],
                    dtype=np.float64,
                ),
                np.array(
                    [np.sqrt(1 / 3), 0, 0, 0, 0, 0, 0, 0, -2 * np.sqrt(1 / 6)],
                    dtype=np.float64,
                ),
            ],
        ),
        (
            "z2",
            [
                np.array(
                    [np.sqrt(1 / 3), 0, 0, np.sqrt(1 / 2), 0, 0, 0, 0, np.sqrt(1 / 6)],
                    dtype=np.float64,
                ),
                np.array(
                    [np.sqrt(1 / 3), 0, 0, -np.sqrt(1 / 2), 0, 0, 0, 0, np.sqrt(1 / 6)],
                    dtype=np.float64,
                )
                + np.array(
                    [np.sqrt(1 / 3), 0, 0, 0, 0, 0, 0, 0, -2 * np.sqrt(1 / 6)],
                    dtype=np.float64,
                ),
            ],
        ),
        (
            "01z3",
            [
                np.array(
                    [np.sqrt(1 / 3), 0, 0, np.sqrt(1 / 2), 0, 0, 0, 0, np.sqrt(1 / 6)],
                    dtype=np.float64,
                ),
                np.array(
                    [np.sqrt(1 / 3), 0, 0, -np.sqrt(1 / 2), 0, 0, 0, 0, np.sqrt(1 / 6)],
                    dtype=np.float64,
                ),
                np.array(
                    [np.sqrt(1 / 3), 0, 0, 0, 0, 0, 0, 0, -2 * np.sqrt(1 / 6)],
                    dtype=np.float64,
                ),
            ],
        ),
    ],
)
def test_generate_povm_from_name_1qutrit(povm_name, expected_vecs):
    # TODO 01x3, 02x3, 21y3
    e_sys = ElementalSystem(0, get_normalized_gell_mann_basis())
    c_sys = CompositeSystem([e_sys])

    actual = povm_typical.generate_povm_from_name(povm_name, c_sys=c_sys)
    for actual_vec, expected_vec in zip(actual.vecs, expected_vecs):
        npt.assert_almost_equal(actual_vec, expected_vec, decimal=15)
